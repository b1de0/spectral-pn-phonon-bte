"""Spectral-PN GPU validation runs (Project A).

Trains the PN-coefficient net on the full-Si TTG system with NO waypoint
supervision — the question is whether gamma EMERGES from the exact-in-(x,mu)
residual where the sampled-residual MLP collapses (gamma_ratio 3.7-8.7 at
L=1.0 wp0). Metrics use the pilot-table conventions (analyze_shape_pilots).

Usage:
    uv run python experiments/transient/run_spectral_pn.py --L 1.0 \
        --epochs 20000 --device cuda --seed 42
"""
import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pinn_bte.training.transient.spectral_pn_trainer import (  # noqa: E402
    SETTLE_TOL_DECLARED,
    SETTLE_WINDOW_DECLARED,
    WINDOW_CROSSING_THRESH,
    WINDOW_MODES,
    SpectralPNConfig,
    build_si_case,
    train_spectral_pn,
)

#: Which phonon mode comb the case (and therefore the DOM arbiter it is scored
#: against) is built from.  'joint' is the PRODUCTION comb and the DEFAULT
#: restored AND silicon's optical heat capacity supplied as a slow reservoir.
#: The two defects were adjudicated and independently re-confirmed in
#: _2026_08_07.md, and they must be fixed TOGETHER because they bias D in
#: opposite directions — the measure alone puts kappa at 255.6 W/(m K).
#: 'shipped' is the pre-adoption weight, kept ONLY to reproduce numbers
#: published under it; its sum C is not a heat capacity.
MODE_SOURCES = ("shipped", "joint", "joint-sumrule")

#: Both members now PATCH the mode source — neither is "no context".  That is
#: deliberate: after the adoption, the bare module default is the acoustic
#: measure-corrected comb, which is an intermediate state no run should use
#: (it has the Jacobian but not the optical capacity).  Making both arms
#: explicit means a run's comb is never whatever the module happens to default
#: to on the day.
MODE_SOURCE_DEFAULT = "joint"

@contextlib.contextmanager
def mode_source_ctx(mode_source: str, Nk: int, T_ref: float = 300.0):
    """Route BOTH `phonon_modes` namespaces for the duration of a run.

    The trainer takes (v, tau, C) by value, but `build_si_case` also builds the
    ARBITER reference (`gamma_dom_hz`, `A_ref`) from the module-level
    `phonon_modes` — a re-weight must move the reference and the solved
    operator TOGETHER or gamma_ratio is comparing two different physics.
    """
    if mode_source not in MODE_SOURCES:
        raise ValueError(f"mode_source={mode_source!r} not in {MODE_SOURCES}")
    if mode_source in ("joint", "joint-sumrule"):
        from pinn_bte.physics.optical_reservoir import joint_mode_source
        from pinn_bte.physics.ttg_dispersion import sum_rule_grid

        # 3-modes-per-primitive-cell sum rule.  A DIFFERENT COMB, therefore a
        # different pin: never mix its runs with "joint" ones in one ladder.
        grid = sum_rule_grid(Nk) if mode_source == "joint-sumrule" else {}
        with joint_mode_source(Nk=Nk, T_ref=T_ref, **grid):
            yield
        return
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    with shipped_mode_source(Nk=Nk, T_ref=T_ref):
        yield

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=float, required=True, help="grating period, um")
    p.add_argument("--nk", type=int, default=20)
    p.add_argument("--lmax", type=int, default=8,
                   help="PN truncation (8 suffices at L >= 1 um; see design)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20000,
                   help="training budget; becomes the MINIMUM budget (a floor "
                        "the stopping rule may not fire below) when "
                        "--settle-window > 0")
    p.add_argument("--lr", type=float, default=2e-3)
    # through a negative-amplitude transient during training and flips positive
    # late (measured: 51 %, 75 %, 87.5 % of budget at L = 10, 0.5, 1.0 um); a
    # rung is "broken" iff a FIXED epoch count expires before its transient
    # completes. The cure is to train each rung until its RATE has settled.
    p.add_argument("--lr-decay-epochs", type=int, default=None,
                   help="cosine horizon; default = --epochs, i.e. the shipped "
                        "schedule bit-for-bit. Set BELOW --epochs-cap to anneal "
                        "over a fixed horizon and then HOLD at eta_min. Needed "
                        "because CosineAnnealingLR(T_max=epochs) reaches the "
                        "low-LR tail -- where the transient completes -- only "
                        "in the last fifth of whatever total you declare, so a "
                        "large cap alone parks every arm at high LR")
    p.add_argument("--settle-window", type=int, default=0,
                   help="trailing window (epochs) for the stopping rule; "
                        "0 = OFF = --epochs is the exact budget. Declared "
                        f"value: {SETTLE_WINDOW_DECLARED}")
    p.add_argument("--settle-tol", type=float, default=0.0,
                   help="stop when gamma_ratio's spread over the trailing "
                        "window falls below this. Declared value: "
                        f"{SETTLE_TOL_DECLARED}. The project's older bar "
                        "(0.010) is TOO LOOSE: it fires before the amplitude "
                        "transient completes on 3 of 9 calibration arms")
    p.add_argument("--gauge", default="additive", choices=["additive", "positive", "corridor"],
                   help="FORM of the slow-head gauge A_hat. 'additive' "
                        "(default) is the shipped 1+(1-e)h that every pinned "
                        "arm uses. 'positive' is exp(-ts*softplus(h)): A_hat(0)"
                        "=1 exactly, A_hat>0 everywhere, and three decades cost "
                        "an O(1) network output instead of a cancellation. The "
                        "additive gauge is inflated 221-2056x over the physical "
                        "amplitude at the window end, and the measured absolute "
                        "error floor in A matches the accuracy that cancellation "
                        "demands."
                        "p127_gauge_form_gate_2026_08_11.md.  'corridor' "
                        " instead bounds the RATE from both sides by "
                        "physics with no tuned constant: gamma_hat in "
                        "[min_m suppressed rate, max_m kinetic rate], measured "
                        "to contain the true rate at 18/18 ladder arms with a "
                        "9.7x minimum margin")
    p.add_argument("--residual-weight-floor", type=float, default=None,
                   help="weight the residual by 1/max(|A(t)|, FLOOR) so "
                        "a given RELATIVE error costs the same at every time. "
                        "Default None = OFF = the shipped objective bit-for-bit. "
                        "Without it the last 25%% of the window carries 2.7e-06 "
                        "of the gradient weight (the system is linear, so the "
                        "residual falls with the amplitude and its square falls "
                        "twice as fast), and the optimiser buys head accuracy "
                        "with tail accuracy. FLOOR is the maximum amplification "
                        "and keeps the weight finite through the ballistic zero "
                        "crossing."
                        "p126_residual_weighting_gate_2026_08_11.md")
    p.add_argument("--epochs-cap", type=int, default=0,
                   help="hard ceiling on the extended budget; 0 = --epochs. An "
                        "arm that reaches it is reported settled=False")
    p.add_argument("--nt", type=int, default=1200)
    p.add_argument("--t-grid", default="log-mix", choices=["uniform", "log-mix"])
    p.add_argument("--n-decay-times", type=float, default=7.0)
    p.add_argument("--window-mode", default="n_decay", choices=list(WINDOW_MODES),
                   help="evaluation-window rule . 'n_decay' (default) = "
                        "t_end = n_decay_times/gamma_dom, the PRE-UNION rule "
                        "every pinned run was measured on — kept default so "
                        "the pins reproduce. 'crossing-union' = LOBE's r11 "
                        "rule max(t_1%%, n_dec/gamma_dom); measured a bit-exact "
                        "no-op at L=0.3..100 um and longer only at L=0.01 "
                        "(x3.347) and L=0.1 (x1.476). The winning rule and BOTH "
                        "candidates are recorded in the npz.")
    p.add_argument("--window-crossing-thresh", type=float,
                   default=WINDOW_CROSSING_THRESH,
                   help="crossing threshold as a fraction of A0 (LOBE r11: "
                        "0.01); inert unless --window-mode uses the crossing")
    p.add_argument("--geometry", default="1d", choices=["1d", "2d"],
                   help="TTG geometry for the DOM reference (kinetically "
                        "identical: uniform full-sphere Omega_x marginal)")
    p.add_argument("--ap-scaling", action="store_true",
                   help="asymptotic-preserving l>=1 row weighting "
                        "(deep-diffusive freeze fix)")
    p.add_argument("--slow-head", action="store_true",
                   help="slow-manifold factorized A_hat head "
                        "(conditioning fix; pair with --ap-scaling)")
    p.add_argument("--duhamel", action="store_true",
                   help="exact collisionless part in the ansatz + "
                        "truncation-edge residual correction "
                        "(deep-ballistic fix, L <= 0.01 um)")
    p.add_argument("--formulation", default="plain",
                   choices=["plain", "ce", "xihybrid"],
                   help="'ce' = Chapman-Enskog asymptotic-preserving net "
                        "+ macro row (diffusive fix; subsumes ap/slow); "
                        "'xihybrid' = per-mode CE/Duhamel split by xi_m "
                        "vs --xi-split (one formulation for every L)")
    p.add_argument("--blend", default="banded", choices=["banded", "ugks"],
                   help="round 1 (xihybrid 1q only): 'ugks' replaces "
                        "the xi-band routing with the UGKS convex form — "
                        "global exact free part + resolvent-kernel slaved "
                        "l=0/1 flux at the self-consistent gamma_hat; no "
                        "xi-split/xi-duh/carrier machinery in that path. "
                        "Default 'banded' = shipped routing, byte-identical")
    p.add_argument("--macro-row", default="on", choices=["on", "off"],
                   help="(--blend ugks only): 'off' EXCLUDES the "
                        "global macro energy-balance row from the loss "
                        "(no value, no gradient path) — the single-flag "
                        "blend-path ablation of the collective closure "
                        "(sec:failure; C21 is banded-only). Default 'on' "
                        "= shipped ugks loss res^2 + macro^2, "
                        "byte-identical")
    p.add_argument("--closure-pin", default="off", choices=["off", "on"],
                   help="repair class 1 (--formulation xihybrid --blend "
                        "ugks only): 'on' adds the guarded "
                        "closure-normalised shim — a rank-1, PARAMETER-FREE "
                        "relabelling of the l=0 slaved slot that makes the "
                        "assembled closure moment reproduce the ansatz's own "
                        "gauge A_hat identically, so 'the gauge tracks the "
                        "physical rate' and 'the closure is reproduced' become "
                        "the SAME requirement. Default 'off' = the shipped "
                        "path BIT-FOR-BIT (gate axis A0 — this is what "
                        "licenses reusing the shipped 20k ladder as the "
                        "control at 0 GPU-h). Spelled on|off, not store_true, "
                        "so a control arm can state 'off' EXPLICITLY on its "
                        "own command line (the --mode-source idiom: a run's "
                        "setting is never whatever the default happens to be "
                        "on the day). Recorded in the npz as `closure_pin`; ")
    p.add_argument("--xi-split", type=float, default=1.0,
                   help="xihybrid CE threshold on xi_m = q v tau (default 1)")
    p.add_argument("--xi-duh", type=float, default=10.0,
                   help="xihybrid Duhamel threshold; [xi-split, xi-duh) "
                        "is the plain band (default 10)")
    p.add_argument("--shared-ahat", action="store_true",
                   help="xihybrid v3: CE+plain bands ride one collective "
                        "A_hat carrier (anti-freeze on mixed spectra)")
    p.add_argument("--carrier-energy", action="store_true",
                   help="xihybrid v5 (T1 closure fix): energy-only "
                        "projection — A_hat = carrier energy mean, the "
                        "closure T_hat floats; pair with --carrier-row")
    p.add_argument("--carrier-row", action="store_true",
                   help="add the carrier-consistency row (the carrier "
                        "subset's own energy balance, collective-rate "
                        "normalized) — the O(1) dA_hat/dt equation the "
                        "AP-scaled rows lack")
    p.add_argument("--mode-source", default=MODE_SOURCE_DEFAULT,
                   choices=list(MODE_SOURCES),
                   help="phonon comb for BOTH the solved operator and the DOM "
                        "reference. 'joint' (default) = k-space measure "
                        "restored + optical heat-capacity reservoir (adopted "
                        "2026-08-09; kappa_bulk = 144.264 W/(m K) at Nk=20). "
                        "'shipped' = the pre-adoption weight, for reproducing "
                        "numbers published under it only.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--outdir", default=None,
                   help="default: data/runs/spectral_pn_{L}um/{timestamp}")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--gamma-log-every", type=int, default=0,
                   help="also sample gamma_ratio every N epochs "
                        "(the PUBLISHED estimator, evaluated under no_grad) to "
                        "record the budget curve")
    p.add_argument("--snapshot-selection", default="final",
                   choices=["final", "min-loss"],
                   help="(terminal Adam-bounce): 'min-loss' ships "
                        "the lowest-logged-loss snapshot as the model "
                        "artifacts (tracked on the existing loss/gamma "
                        "logging cadence; pure observation — the training "
                        "trajectory is byte-identical); the npz records "
                        "best_epoch/best_loss/final_loss. Default 'final' = "
                        "ship the final epoch, shipped behaviour. Re-pin "
                        "adoption is a queue-script decision and must be "
                        "UNIFORM across all rungs of a sweep")
    p.add_argument("--component-log-every", type=int, default=0,
                   help="also sample the loss DECOMPOSITION "
                        "(per-term MSE + the weights that reconstruct the "
                        "aggregate), the per-head grad norms and the LR every "
                        "N epochs — the aggregate hides a head sitting at its "
                        "minimum while another still learns; default 0 = OFF "
                        "= byte-identical trajectory")
    args = p.parse_args(argv)
    if args.gamma_log_every < 0:
        p.error("--gamma-log-every must be >= 0 (0 = off)")
    if args.component_log_every < 0:
        p.error("--component-log-every must be >= 0 (0 = off)")
    if args.blend != "banded":
        if args.formulation != "xihybrid":
            p.error("--blend ugks requires --formulation xihybrid")
        if args.shared_ahat or args.carrier_energy or args.carrier_row:
            p.error("--blend ugks replaces the carrier machinery; drop "
                    "--shared-ahat/--carrier-energy/--carrier-row")
    if args.macro_row == "off" and args.blend != "ugks":
        p.error("--macro-row off is the ugks-path ablation; it "
                "requires --blend ugks (banded macro licensing is a "
                "derived physical condition, not a dial)")
    if args.closure_pin == "on" and not (args.formulation == "xihybrid"
                                         and args.blend == "ugks"):
        # Fail in 0 s at the CLI, not after the case build. On every other
        # path the net never reads the flag, so the arm would train the
        # CONTROL and record closure_pin=True — a mislabelled control, which
        # is strictly worse than a crash.
        p.error("--closure-pin on is the repair-class-1 instrument on "
                "the ugks single-harmonic l=0 slaved slot; it requires "
                "--formulation xihybrid --blend ugks")
    return args

def main(argv=None) -> Path:
    args = parse_args(argv)
    # The comb must be routed for the WHOLE run: `build_si_case` reads it for
    # the solved operator AND for the arbiter reference it is scored against.
    with mode_source_ctx(args.mode_source, Nk=args.nk):
        return _run(args)

def _run(args) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = (Path(args.outdir) if args.outdir
              else REPO / "data" / "runs" / f"spectral_pn_{args.L}um" / run_id)
    label = f"spn-L{args.L}-s{args.seed}"

    print(f"[spectral-pn] case: L={args.L} um, Nk={args.nk}, "
          f"L_max={args.lmax}, run={run_id}", flush=True)
    case = build_si_case(args.L, Nk=args.nk,
                         n_decay_times=args.n_decay_times,
                         geometry=args.geometry,
                         window_mode=args.window_mode,
                         window_crossing_thresh=args.window_crossing_thresh)
    # NB: no square brackets around the rule — a rich console handler eats them
    # as markup and the winning rule vanishes from the log (LOBE learned this).
    print(f"[spectral-pn] window t_end={case.t_end:.3e} s "
          f"({case.t_end / case.tau_ref:.1f} tau_ref), "
          f"gamma_dom={case.gamma_dom_hz:.4e} Hz", flush=True)
    print(f"[spectral-pn] window mode={case.window_mode} rule={case.window_rule} "
          f"(t_1% at {case.window_crossing_thresh:g}*A0: "
          f"{case.t_candidate_crossing_s:.4e} s; "
          f"n_dec={args.n_decay_times:g}/gamma_dom: "
          f"{case.t_candidate_ndecay_s:.4e} s)", flush=True)

    t_ref, A_ref, gamma_ref = case.t_ref, case.A_ref, case.gamma_dom_hz

    cfg = SpectralPNConfig(
        v=case.v, tau=case.tau, C=case.C, q=case.q, t_end=case.t_end,
        L_max=args.lmax, width=args.width, depth=args.depth,
        epochs=args.epochs, lr=args.lr, nt=args.nt, t_grid=args.t_grid,
        # the cosine horizon cfg.epochs, and settle_window=0 makes `epochs` the
        # exact budget -- i.e. the shipped schedule and loop, unchanged.
        gauge=args.gauge,
        residual_weight_floor=args.residual_weight_floor,
        lr_decay_epochs=args.lr_decay_epochs,
        settle_window=args.settle_window, settle_tol=args.settle_tol,
        epochs_cap=args.epochs_cap,
        seed=args.seed, device=args.device, dtype=args.dtype,
        ap_scaling=args.ap_scaling, slow_head=args.slow_head,
        duhamel=args.duhamel, formulation=args.formulation,
        blend=args.blend, macro_row=args.macro_row,
        # writes THIS value into the npz, so the artifact records what the model did.
        closure_pin=(args.closure_pin == "on"),
        xi_split=args.xi_split, xi_duh=args.xi_duh,
        shared_ahat=args.shared_ahat,
        carrier_energy=args.carrier_energy, carrier_row=args.carrier_row,
        Nk=args.nk,
        t_ref=t_ref, A_ref=A_ref,
        gamma_ref=gamma_ref,
        label=label, outdir=outdir, run_id=run_id,
        log_every=args.log_every, gamma_log_every=args.gamma_log_every,
        component_log_every=args.component_log_every,
        snapshot_selection=args.snapshot_selection,
        extra_meta=dict(mode_source=args.mode_source,
                        tau_ref=case.tau_ref, Nk=args.nk, L_um=args.L,
                        n_decay_times=args.n_decay_times,
                        gamma_dom_hz=case.gamma_dom_hz,
                        geometry=args.geometry,
                        gauge=args.gauge,
                        # (Lt_rule / Lt_candidate_*_tau). Storing WHICH rule
                        # won and what BOTH candidates were is the only reason
                        # an intended one; PN had none of it.
                        **case.window_provenance()),
    )
    res = train_spectral_pn(cfg)

    for it, lv in res["history"]:
        print(f"  epoch {it:6d}  residual {lv:.3e}", flush=True)
    for it, gr, lv in res.get("gamma_history", []):
        print(f"  [gamma] epoch {it:6d}  gamma_ratio {gr:.4f} "
              f"residual {lv:.3e}", flush=True)
    # SHAPE rides the gamma cadence (one amplitude evaluation feeds both). A
    # gamma_ratio near 1 means only that the AREA under A matches — the ansatz
    # pins A(0)=1, so gamma_eff is a pure functional of that area. These are the
    # readings that separate a correct curve from an accidentally-right one.
    snames = list(res.get("shape_names", []))
    for row in res.get("shape_history", []):
        vals = " ".join(f"{n}={x:.4g}" for n, x in zip(snames, row[1:]))
        print(f"  [shape] epoch {int(row[0]):6d}  {vals}", flush=True)
    cnames, gnames = res.get("component_names", []), res.get("gradnorm_names", [])
    for row in res.get("component_history", []):
        it, lr_now, lv = row[0], row[1], row[2]
        mses = row[3:3 + len(cnames)]
        gns = row[3 + len(cnames):]
        terms = " ".join(f"{n}={m:.3e}" for n, m in zip(cnames, mses))
        grads = " ".join(f"|g|{n}={x:.2e}" for n, x in zip(gnames, gns))
        print(f"  [comp] epoch {it:6d}  loss {lv:.3e}  lr {lr_now:.2e} "
              f"{terms}  {grads}", flush=True)
    print(f"[spectral-pn] {label}: loss {res['loss_initial']:.3e} -> "
          f"{res['loss_final']:.3e} "
          f"(x{res['loss_initial'] / max(res['loss_final'], 1e-300):.0f} drop)",
          flush=True)
    print(f"[spectral-pn] RMSE={res['rmse']:.4f} max|dA|={res['max_abs_dA']:.4f} "
          f"gamma_eff={res['gamma_eff']:.4e} gamma_ratio={res['gamma_ratio']:.4f}",
          flush=True)
    # epoch, best_epoch == epochs-1 and best_loss == final loss by convention).
    print(f"[spectral-pn] snapshot={res['snapshot_selection']} "
          f"best_epoch={res['best_epoch']} best_loss={res['best_loss']:.3e} "
          f"final_loss={res['loss_final']:.3e}", flush=True)
    # The shape verdict on the SAME endpoint the gamma_ratio above reports.
    # cancel = |int(A-A_ref)dt| / int|A-A_ref|dt in [0,1]: ->1 a uniform bias,
    # ->0 two errors cancelling into a convincing rate. A_min<0 means
    # _integral_gamma's clip was live and the rate came off a truncated curve;
    # max_dA_fwd>0 means A(t) is not monotone decaying.
    if "int_abs_dA" in res:
        cancel = (abs(res["int_signed_dA"]) / res["int_abs_dA"]
                  if res["int_abs_dA"] > 0 else float("nan"))
        print(f"[spectral-pn-shape] cancel={cancel:.3f} "
              f"int_signed_dA={res['int_signed_dA']:.4e} "
              f"int_abs_dA={res['int_abs_dA']:.4e} "
              f"A_min={res['A_min']:.4e} max_dA_fwd={res['max_dA_fwd']:.4e}",
              flush=True)
    print(f"[spectral-pn] saved: {outdir}", flush=True)
    return outdir

if __name__ == "__main__":
    main()
