"""Score the sum-rule re-pin ladder, and decide whether 2D may be launched."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn_bte.physics.decay_metrics import GAMMA_ESTIMATOR_RULED, integral_gamma
from pinn_bte.physics.optical_reservoir import (PRIMITIVE_CELLS_PER_M3,
                                                joint_mode_source, joint_modes)
from pinn_bte.physics.ttg_dispersion import phonon_modes, sum_rule_grid

REPO = Path(__file__).resolve().parents[1]
LADDERS = {
    "1d": REPO / "data" / "runs" / "pn_sumrule_ladder_1d",
    "2d": REPO / "data" / "runs" / "pn_sumrule_ladder_2d",
}
LADDERS_FILE = REPO / "data" / "ladders.yaml"
PERIODS = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)
NK = 20
KB = 1.380649e-23
RHO_CP = 1.63e6
KAPPA_BAND = (142.0, 156.0)
ALPHA_MEASURED = 8.8e-5

SHIPPED_WORST_DEV = 0.0055   # worst |1 - gamma_ratio| of the shipped 1D sweep
#: after the sum-rule pin is adopted `ladder_source` names the ladder under
BASELINE_LADDER = "corridor60k"

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#: G2: the re-pin must not degrade the fit.  The shipped sweep's worst rung is
#: 0.55 %; the first four re-pinned arms drifted by <= 0.06 pp.  A 0.25 pp
#: allowance is generous for grid-induced drift and still catches a real
#: degradation (the ablations' near misses live at 8-40 %).
GATE_WORST_DEV = SHIPPED_WORST_DEV + 0.0025          # 0.80 %
#: G3: per-rung shape must not blow up relative to its own shipped value.
GATE_RMSE_FACTOR = 1.5
#: G6: Fourier recovery at the diffusive end, the elementary limit.
GATE_DIFFUSIVE_RATIO = 0.99
#: The manuscript's existing two-geometry agreement claim, in percentage
#: points of gamma_PN / gamma_DOM.  Re-use it for the replacement ladders.
GATE_GEOMETRY_GAP_PP = 0.07

def comb_facts() -> dict:
    """The comb's own dimensional axes -- measured from the shipped code path."""
    g = sum_rule_grid(NK)
    v, tau, C = joint_modes(Nk=NK, **g)
    C_tot = float(C.sum())
    kappa = float((C * (v / 1e10) ** 2 * tau).sum() / 3.0)
    n_modes = float(np.sum(phonon_modes(Nk=NK, T_ref=4000.0, measure=True, **g)[2])) / KB
    return dict(grid=g, C_tot=C_tot, kappa_bulk=kappa, D_bulk=kappa / C_tot,
                capacity_ratio=C_tot / RHO_CP,
                mode_count_ratio=n_modes / (3.0 * PRIMITIVE_CELLS_PER_M3),
                mfp_max_um=float(((v[:-1] / 1e10) * tau[:-1]).max() * 1e6))

def load_shipped_baseline(geometry: str = "1d") -> dict[float, dict]:
    """Re-score one geometry of the BASELINE ladder (pinned by name, not by
    `ladder_source` -- see BASELINE_LADDER)."""
    ladders = yaml.safe_load(LADDERS_FILE.read_text())
    source = BASELINE_LADDER
    rung_paths = ladders["ladders"][source][geometry]
    v_old, tau_old, C_old = joint_modes(Nk=NK)
    D_old = float((C_old * (v_old / 1e10) ** 2 * tau_old).sum()
                  / 3.0 / C_old.sum())
    baseline = {}
    for L in PERIODS:
        run_dir = Path(rung_paths[L])
        hits = sorted(run_dir.glob("*_results.npz"))
        if len(hits) != 1:
            raise RuntimeError(f"shipped L={L}: expected one npz in {run_dir}, got {hits}")
        z = np.load(hits[0], allow_pickle=True)
        A = np.asarray(z["A"]).ravel()
        t = np.asarray(z["t"]).ravel()
        ref = np.interp(t, np.asarray(z["t_ref"]).ravel(),
                        np.asarray(z["A_ref"]).ravel())
        A_norm = A / A[0]
        ref_norm = ref / ref[0]
        gamma_pn = integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
        q = 2.0 * np.pi / (L * 1e-6)
        baseline[L] = dict(
            npz=str(hits[0]),
            gamma_ratio=(gamma_pn
                         / integral_gamma(t, ref, estimator=GAMMA_ESTIMATOR_RULED)),
            rmse=float(np.sqrt(np.mean((A_norm - ref_norm) ** 2))),
            S=gamma_pn / q**2 / D_old,
        )
    return baseline

def score_arm(L: float, D_bulk: float, kappa_bulk: float,
              geometry: str = "1d") -> dict | None:
    d = LADDERS[geometry] / f"ladder_L{L}_{geometry}"
    hits = sorted(glob.glob(str(d / "*_results.npz")))
    if not hits:
        return None
    z = np.load(hits[0], allow_pickle=True)
    A = np.asarray(z["A"]).ravel()
    t = np.asarray(z["t"]).ravel()
    ref = np.interp(t, np.asarray(z["t_ref"]).ravel(),
                    np.asarray(z["A_ref"]).ravel())
    g_pn = integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
    g_dom = integral_gamma(t, ref, estimator=GAMMA_ESTIMATOR_RULED)
    loss = np.asarray(z["loss_history"])
    loss = loss[:, -1] if loss.ndim > 1 else loss
    q = 2.0 * np.pi / (L * 1e-6)
    S = g_pn / q**2 / D_bulk
    return dict(
        L_um=L, npz=hits[0],
        gamma_pn=g_pn, gamma_dom=g_dom, gamma_ratio=g_pn / g_dom,
        dev=abs(1.0 - g_pn / g_dom),
        rmse=float(np.sqrt(np.mean((A / A[0] - ref / ref[0]) ** 2))),
        S=S, kappa_eff=S * kappa_bulk,
        A_min=float((A / A[0]).min()),
        A_ref_min=float((ref / ref[0]).min()),
        sign_changes=int(np.sum(np.diff(np.sign(A)) != 0)),
        sign_changes_ref=int(np.sum(np.diff(np.sign(ref)) != 0)),
        loss_final=float(loss[-1]), loss_drop=float(loss[0] / loss[-1]),
    )

def max_geometry_gap_pp(rows_1d: list[dict | None],
                        rows_2d: list[dict | None]) -> tuple[float, float]:
    """Return the largest 1D--2D rate-ratio gap and its period."""
    by_1d = {r["L_um"]: r for r in rows_1d if r is not None}
    by_2d = {r["L_um"]: r for r in rows_2d if r is not None}
    common = sorted(set(by_1d) & set(by_2d))
    if not common:
        return float("inf"), float("nan")
    return max((abs(by_1d[L]["gamma_ratio"] - by_2d[L]["gamma_ratio"])
                * 100.0, L) for L in common)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=("1d", "2d"), default="1d")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    geometry = args.geometry

    facts = comb_facts()
    shipped = load_shipped_baseline(geometry)
    plan = "" if geometry == "1d" else ""
    print("=" * 96)
    print(f"{plan}  SUM-RULE LADDER, {geometry.upper()} -- multi-axis score")
    print("=" * 96)
    print(f"comb: {facts['grid']}")
    print(f"  mode count / exact {facts['mode_count_ratio']:.4f} "
          f"sum C / rho c_p {facts['capacity_ratio']:.4f} "
          f"kappa_bulk {facts['kappa_bulk']:.2f} "
          f"D/alpha {facts['D_bulk'] / ALPHA_MEASURED:.4f} "
          f"longest MFP {facts['mfp_max_um']:.2f} um")

    with joint_mode_source(Nk=NK, **facts["grid"]):
        rows = [r for r in (score_arm(L, facts["D_bulk"], facts["kappa_bulk"])
                            if geometry == "1d" else
                            score_arm(L, facts["D_bulk"], facts["kappa_bulk"], "2d")
                            for L in PERIODS) if r]
        rows_1d = ([score_arm(L, facts["D_bulk"], facts["kappa_bulk"], "1d")
                    for L in PERIODS] if geometry == "2d" else [])

    print(f"\n{'L [um]':>8} {'gamma_ratio':>12} {'|1-g| %':>8} {'RMSE':>9} "
          f"{'x ship':>7} {'S':>8} {'S ship':>8} {'kappa_eff':>10} "
          f"{'sgn/ref':>8} {'loss':>10} {'drop':>9}")
    for r in rows:
        old = shipped[r["L_um"]]
        print(f"{r['L_um']:>8g} {r['gamma_ratio']:>12.6f} {r['dev'] * 100:>8.3f} "
              f"{r['rmse']:>9.6f} {r['rmse'] / old['rmse']:>7.2f} "
              f"{r['S']:>8.4f} {old['S']:>8.4f} "
              f"{r['kappa_eff']:>10.2f} {r['sign_changes']:>4d}/{r['sign_changes_ref']:<3d} "
              f"{r['loss_final']:>10.2e} {r['loss_drop']:>9.0f}")
    missing = [L for L in PERIODS if L not in {r["L_um"] for r in rows}]
    if missing:
        print(f"\n  NOT YET TRAINED: {missing}")

    # ---- the gates -------------------------------------------------------
    gates: list[tuple[str, bool, str]] = []
    gates.append(("G1 artefact: all nine arms wrote an npz",
                  not missing, f"{len(rows)}/9 present"))
    worst = max((r["dev"] for r in rows), default=1.0)
    gates.append((f"G2 fit preserved: worst |1-gamma| <= {GATE_WORST_DEV:.2%}",
                  worst <= GATE_WORST_DEV,
                  f"worst {worst:.2%} (shipped sweep {SHIPPED_WORST_DEV:.2%})"))
    bad = [(r["L_um"], r["rmse"] / shipped[r["L_um"]]["rmse"]) for r in rows
           if r["rmse"] / shipped[r["L_um"]]["rmse"] > GATE_RMSE_FACTOR]
    gates.append((f"G3 shape: every rung within {GATE_RMSE_FACTOR}x its shipped RMSE",
                  not bad, "ok" if not bad else f"over: {bad}"))
    S = [r["S"] for r in rows]
    mono = all(b > a for a, b in zip(S, S[1:])) if len(rows) > 1 else False
    gates.append(("G4 physics, arbiter-free: S strictly increasing in L and S <= 1",
                  mono and max(S, default=2) <= 1.0,
                  f"monotone={mono} max S={max(S, default=float('nan')):.4f}"))
    dim_ok = (abs(facts["capacity_ratio"] - 1) <= 0.01
              and KAPPA_BAND[0] <= facts["kappa_bulk"] <= KAPPA_BAND[1]
              and abs(facts["D_bulk"] / ALPHA_MEASURED - 1) <= 0.03)
    gates.append(("G5 dimensional: capacity, kappa band and diffusivity together",
                  dim_ok,
                  f"C {facts['capacity_ratio']:.4f}, kappa {facts['kappa_bulk']:.2f}, "
                  f"D/alpha {facts['D_bulk'] / ALPHA_MEASURED:.4f}"))
    diff = [r for r in rows if r["L_um"] == 100.0]
    diff_ok = bool(diff) and diff[0]["kappa_eff"] / facts["kappa_bulk"] >= GATE_DIFFUSIVE_RATIO
    gates.append((f"G6 Fourier limit: kappa_eff(100 um)/kappa_bulk >= {GATE_DIFFUSIVE_RATIO}",
                  diff_ok,
                  f"{diff[0]['kappa_eff'] / facts['kappa_bulk']:.4f}" if diff else "arm missing"))
    if geometry == "2d":
        gap_pp, gap_L = max_geometry_gap_pp(rows_1d, rows)
        gates.append((f"G7 two-geometry agreement: max gap <= {GATE_GEOMETRY_GAP_PP:.2f} pp",
                      not missing and gap_pp <= GATE_GEOMETRY_GAP_PP,
                      f"{gap_pp:.5f} pp at L={gap_L:g} um"))

    print("\nGATES INHERITED FROM THE 1D SCREEN AND THE MANUSCRIPT")
    for name, ok, note in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {note}")

    print("\nAXES NOT RUN -- an unrun check must not read as a passed one:")
    print("  * seed spread: single seed (42) at every rung, as in production.")
    if geometry == "1d":
        print("  * 2D: score separately; the marginal identity is a claim about the")
        print("    kinetic system and needs its own independently trained arm.")
    print("  * visual: no figure regenerated; run scripts/make_figures.py")
    print("    against a re-pinned ladders.yaml only if the re-pin is adopted.")
    print("  * the ablation cells (Table 4's PN columns) are still joint-comb runs.")
    print("  * no figure and no stored number was touched by this script.")

    ok = all(g[1] for g in gates)
    if geometry == "1d":
        success = "ALL 1D GATES PASSED."
    else:
        success = "ALL 2D GATES PASSED -- complete-ladder review may proceed."
    print("\n" + (success if ok else "AT LEAST ONE GATE FAILED -- report before adoption."))
    if args.json:
        args.json.write_text(json.dumps(
            dict(geometry=geometry, comb=facts, rows=rows,
                 gates=[dict(name=n, passed=p, note=t) for n, p, t in gates],
                 all_passed=ok), indent=1, default=float) + "\n")
        print(f"wrote {args.json}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
