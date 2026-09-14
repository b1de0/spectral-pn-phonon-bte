"""C22 FULL RE-PIN: the expm-miss Nk sweep at CONVERGED dt (/)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pn_expm_miss_nk_sweep import CFG, NK_FIT, NK_REPORT, fit_limit  # noqa: E402

N_DECAY = 4.0
H = 2
# Solver defaults CFG does not carry.  Passed EXPLICITLY (identical values =>
# tuple -- T_ref sets tau(T_ref), i.e. the whole frozen arm.
SOLVER_DEFAULTS = {"T_ref": 300.0, "ic_2s": 0.0}
# |inc_last|/|gamma| above which a ladder the two-tail bracket REFUSES is
# treated as un-converged (RAISE) rather than as a zero-crossing (see
# arm_limit).  A floor guard on the residual, NOT a convergence proof.
CROSSING_REL_TOL = 1e-5
# Largest increment ratio the two-tail bracket's own model admits (see
# two_tail_bracket): the model asserts r stays in [r_last, 2], so r_last > 2
# inverts it.  2.1 keeps a margin over the largest benign ratio measured in
# the campaign (2.0097, L=10 localsym -- a first-order tail wobbling about 2)
# and still refuses the smallest crossing-contaminated one (3.93).
RATIO_MAX = 2.1
NK_ALL = sorted(set(NK_REPORT + NK_FIT))            # [20, 30, 40, 60, 80, 120, 160]

# across Nk.  Every rung ds = 1/2^k for k = 0 .. log2(depth).
# L=100 localsym was extended 256 -> 512 by the C1 correction (the campaign
# ladder stopped exactly ON the increment's zero-crossing, see arm_limit).
ARM_DEPTHS = {
    100.0: {"local": 1024, "frozen": 512, "localsym": 512},
    10.0: {"local": 512, "frozen": 256, "localsym": 256},
}

# extend the L=100 localsym ladder past its zero-crossing (C1); the third is
# the deeper L=10 headline rung that PROVES the two-tail bracket honest (it
# lands strictly inside the campaign bracket).
AUDIT_GAMMAS = {
    (100.0, 20, "localsym", 512): 297822.08355874615,
    (100.0, 30, "localsym", 512): 323873.45365197206,
    (10.0, 20, "local", 1024): 28476508.75633468,
}

# inside/at these published numbers before the sweep is trusted.
H135_GATES = {
    (100.0, "miss_asym"): (2.7755, 0.0030),   # value, tolerance (pp)
    (10.0, "miss_asym"): (1.7190, 0.0010),
    # the C1 correction (+0.4301) -- that value came from a two-tail bracket
    # applied ON the increment's zero-crossing.  L=10 sym is unaffected.
    (100.0, "miss_sym"): (0.4301, 0.0010),
    (10.0, "miss_sym"): (0.3007, 0.0010),
    (100.0, "frozen_hz"): (2.965464e5, 1.0),  # value, tolerance (Hz)
    (10.0, "frozen_hz"): (2.799496e7, 100.0),
}
# Pinned old-dt (dt_scale=1) values the regression guard re-checks.
PINNED_OLD = {
    (100.0, 20): 3.5923, (10.0, 20): 2.1571,           # miss %, Nk=20
    (100.0, "limit"): 3.6245, (10.0, "limit"): 2.2018,  # a - c/Nk fit limits
    (100.0, "sym20"): 0.4339, (10.0, "sym20"): 0.3033,  # symmetric control
}

# --------------------------------------------------------------- extrapolation
def monotone_tail_start(incs: list[float]) -> int:
    """Index into `incs` where the final same-sign, strictly |shrinking|
    suffix begins (coarse->fine increments)."""
    i = len(incs) - 1
    while (i > 0 and np.sign(incs[i]) == np.sign(incs[i - 1])
           and abs(incs[i]) < abs(incs[i - 1])):
        i -= 1
    return i

def two_tail_bracket(gammas: list[float]) -> tuple[float, float, float, float]:
    """dt->0 limit bracket from a coarse->fine gamma sequence (successive dt halvings). Returns (lo, hi, mid, half_width)."""
    if len(gammas) < 3:
        raise ValueError("need >= 3 ladder rungs for a two-tail bracket")
    incs = list(np.diff(gammas))
    start = monotone_tail_start(incs)
    tail = incs[start:]
    if len(tail) < 2:
        raise ValueError(
            f"no monotone-shrinking increment suffix (tail={tail}); "
            f"the ladder is not in its asymptotic regime — deepen it")
    inc_last, inc_prev = tail[-1], tail[-2]
    if inc_prev / inc_last > RATIO_MAX:
        raise ValueError(
            f"increment ratio {inc_prev / inc_last:.4g} > RATIO_MAX="
            f"{RATIO_MAX}: outside the two-tail model's domain (it asserts "
            f"r in [r_last, 2]); the ladder is at an O(dt)/O(dt^2) "
            f"cancellation, not converging")
    rho = inc_last / inc_prev
    lim_r2 = gammas[-1] + inc_last
    lim_geo = gammas[-1] + inc_last * rho / (1.0 - rho)
    lo, hi = min(lim_r2, lim_geo), max(lim_r2, lim_geo)
    return lo, hi, 0.5 * (lo + hi), 0.5 * (hi - lo)

def two_term_limit(gammas: list[float]) -> float:
    """Exact elimination of gamma(x) = gamma_inf + A*x + B*x^2 off the finest
    THREE rungs (x = 4h, 2h, h).  This is the estimator that models the
    O(dt)/O(dt^2) cancellation, so it is the second model of the crossing
    band; on a clean first-order ladder (B=0) it reduces to p=1 Richardson."""
    if len(gammas) < 3:
        raise ValueError("need >= 3 ladder rungs for a 2-term elimination")
    g1, g2, g3 = gammas[-3], gammas[-2], gammas[-1]
    b = ((g1 - g2) - 2.0 * (g2 - g3)) / 6.0          # B*h^2
    a = (g2 - g3) - 3.0 * b                          # A*h
    return float(g3 - a - b)

def arm_limit(gammas: list[float], crossing_rel_tol: float = CROSSING_REL_TOL
              ) -> tuple[float, float, float, float, str]:
    """dt->0 limit of one arm: (lo, hi, mid, half_width, method)."""
    try:
        lo, hi, mid, half = two_tail_bracket(gammas)
        return lo, hi, mid, half, "two-tail"
    except ValueError:
        incs = np.diff(gammas)
        if len(incs) < 2 or abs(incs[-1]) / abs(gammas[-1]) >= crossing_rel_tol:
            raise
        lim2 = two_term_limit(gammas)
        cands = [gammas[-1], gammas[-1] + incs[-1], lim2,
                 gammas[-1] + 2.0 * (lim2 - gammas[-1])]
        lo, hi = float(min(cands)), float(max(cands))
        return lo, hi, 0.5 * (lo + hi), 0.5 * (hi - lo), "crossing"

def frozen_richardson(gammas: list[float]) -> float:
    """p=1 Richardson from the last pair (the frozen arm is textbook
    first-order, ratio -> 2, so this equals the ratio-2 tail limit)."""
    return 2.0 * gammas[-1] - gammas[-2]

def ladder_diag(gammas: list[float]) -> dict:
    """Shrinking-increment diagnostics for ONE arm's dt ladder (gate d)."""
    incs = list(np.diff(gammas))
    ratios = [incs[i] / incs[i + 1] if incs[i + 1] else float("inf")
              for i in range(len(incs) - 1)]
    start = monotone_tail_start(incs)
    tail_len = len(incs) - start
    method = arm_limit(gammas)[4]
    return {
        "n_rungs": len(gammas), "inc_first": incs[0], "inc_last": incs[-1],
        "rel_inc_last": abs(incs[-1]) / abs(gammas[-1]),
        "tail_start_idx": start, "tail_len": tail_len,
        "shrinking": bool(tail_len >= 2), "ratio_tail_first":
            ratios[start] if start < len(ratios) else float("nan"),
        "ratio_tail_last": ratios[-1] if ratios else float("nan"),
        "method": method,
    }

def miss_bracket(local_gammas: list[float], frozen_gammas: list[float]
                 ) -> tuple[float, float, float, float]:
    """Converged miss %% bracket: the local dt->0 band against the frozen
    dt->0 band, propagated OUTWARD (conservative)."""
    l_lo, l_hi, *_ = arm_limit(local_gammas)
    f_lo, f_hi, *_ = arm_limit(frozen_gammas)
    m_lo = (l_lo - f_hi) / f_hi * 100.0
    m_hi = (l_hi - f_lo) / f_lo * 100.0
    return m_lo, m_hi, 0.5 * (m_lo + m_hi), 0.5 * (m_hi - m_lo)

# ---------------------------------------------------------------------- jobs
def build_job_list() -> list[dict]:
    """All (L, Nk, arm, ds) rungs, cost-sorted descending (longest first —
    minimizes makespan under a worker pool)."""
    jobs = []
    for L, arms in ARM_DEPTHS.items():
        for arm, depth in arms.items():
            den = 1
            while den <= depth:
                for nk in NK_ALL:
                    # empirical wall model: steps ~ den * (L=100 ? 10 : 1),
                    # per-step cost ~ (4 + Nk/14)
                    cost = den * (10.0 if L == 100.0 else 1.0) * (4 + nk / 14)
                    jobs.append({"L_um": L, "Nk": nk, "arm": arm,
                                 "ds_den": den, "cost": cost})
                den *= 2
    return sorted(jobs, key=lambda j: -j["cost"])

def _rung_path(out_dir, L_um, nk, arm, ds_den) -> Path:
    return Path(out_dir) / f"rung_L{L_um:g}_Nk{nk}_{arm}_ds{ds_den}.json"

def comb_context_for(mode_source: str, nk: int, T_ref: float):
    """The arbiter comb for one rung. ``"default"`` is the module's BARE default comb -- since the measure-only acoustic comb, NOT the one C22 was pinned on (its L=10 ds=1 gamma reads 58591763.37 vs the pinned 28413919.94, x2.06); ``"shipped"`` reproduces every C22 rung bit-for-bit; ``"shipped"`` is the pre- legacy weight (`shipped_mode_source` forwards (Nk, T), which the tau(T) tables sweep on purpose); ``"joint"`` the production comb of ..09-12; ``"joint-sumrule"`` the comb adopted  -- the SAME joint comb on `sum_rule_grid(nk)` (]]: sum-rule radius, midpoint rule, first node anchored at the production floor). Appendix B must be measured on the comb the paper uses (]]), which is why this is a parameter and not a default."""
    import contextlib
    if mode_source == "joint":
        from pinn_bte.physics.optical_reservoir import joint_mode_source
        return joint_mode_source(Nk=nk, T_ref=T_ref)
    if mode_source == "joint-sumrule":
        from pinn_bte.physics.optical_reservoir import joint_mode_source
        from pinn_bte.physics.ttg_dispersion import sum_rule_grid
        return joint_mode_source(Nk=nk, T_ref=T_ref, **sum_rule_grid(nk))
    if mode_source == "shipped":
        from pinn_bte.physics.ttg_dispersion import shipped_mode_source
        return shipped_mode_source(Nk=nk, T_ref=T_ref)
    if mode_source == "default":
        return contextlib.nullcontext()
    raise ValueError(f"mode_source={mode_source!r} not in "
                     "(default, shipped, joint, joint-sumrule)")

def run_rung(L_um: float, nk: int, arm: str, ds_den: int, out_dir,
             mode_source: str = "default") -> Path:
    """One deterministic solve; writes a JSON with FULL provenance."""
    from pinn_bte.physics.ttg_pnhq import solve_pnHq_decay, TauGridEdgeError
    _comb = comb_context_for(mode_source, nk, CFG.get("T_ref", 300.0))

    base = {k: v for k, v in CFG.items() if k != "ic_2c"} | SOLVER_DEFAULTS
    ic_2c = CFG["ic_2c"] if arm in ("local", "frozen") else 0.0
    tau_model = "frozen" if arm == "frozen" else "local"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = _rung_path(out, L_um, nk, arm, ds_den)
    rec = {"L_um": L_um, "Nk": nk, "arm": arm, "ds_den": ds_den,
           "ds": 1.0 / ds_den, "ic_2c": ic_2c, "tau_model": tau_model,
           "n_decay": N_DECAY, "H": H, "mode_source": mode_source, **base}
    t0 = time.time()
    try:
        with _comb:
            gamma, tr = solve_pnHq_decay(
                L_um, H=H, Nk=nk, tau_model=tau_model, n_decay=N_DECAY,
                dt_scale=1.0 / ds_den, ic_2c=ic_2c, return_trace=True, **base)
        rec.update(gamma=gamma, dt=tr["dt"], nsteps=tr["nsteps"],
                   status="ok")
    except TauGridEdgeError as e:      # LOUD landmine: record, don't crash
        rec.update(gamma=None, status=f"TauGridEdgeError: {e}")
    rec["wall_s"] = time.time() - t0
    json.dump(rec, open(f, "w"), indent=1)
    print(f"[rung] {f.name}: gamma={rec.get('gamma')} "
          f"wall={rec['wall_s']:.1f}s", flush=True)
    return f

# -------------------------------------------------------------------- report
def _load_ladders(out_dir) -> dict:
    """{(L, Nk, arm): {ds_den: gamma}} from the rung JSONs."""
    ladders: dict = {}
    for f in sorted(Path(out_dir).glob("rung_*.json")):
        d = json.load(open(f))
        if d.get("status") != "ok":
            print(f"!! {f.name}: {d.get('status')}")
            continue
        ladders.setdefault((d["L_um"], d["Nk"], d["arm"]), {})[d["ds_den"]] = d["gamma"]
    return ladders

def _seq(ladder: dict) -> list[float]:
    return [ladder[d] for d in sorted(ladder)]

def ladder_depth_gate(ladders: dict) -> dict:
    """Gate (f): every (L, Nk, arm) ladder must be the COMPLETE dyadic set
    ds = 1 .. 1/ARM_DEPTHS[L][arm].

    Shallower = a limit quoted off an under-resolved ladder — the C1 failure
    (the L=100 symmetric ladder stopped exactly ON its increment's
    zero-crossing).  Deeper/ragged = the Nk sweep stops being a pure Nk sweep
    (uniform-params rule).  Violations are [tag, got_depth, want_depth].
    """
    viol = []
    for L, arms in ARM_DEPTHS.items():
        for arm, depth in arms.items():
            want = {2 ** i for i in range(int(round(np.log2(depth))) + 1)}
            for nk in NK_ALL:
                got = set(ladders.get((L, nk, arm), {}))
                if got != want:
                    viol.append([f"{L:g}/Nk{nk}/{arm}",
                                 max(got) if got else 0, depth])
    return {"got": len(viol), "want": 0, "pass": not viol, "violations": viol}

def _audit_gate(ladders: dict) -> dict:
    """Gate (e): every AUDIT_GAMMAS rung present in the campaign must match
    the physicist's fresh solve BIT-EXACTLY (the solver is deterministic), so
    the sign-off evidence stays welded to the artifacts."""
    checks = []
    for (L, nk, arm, den), want in AUDIT_GAMMAS.items():
        got = ladders.get((L, nk, arm), {}).get(den)
        if got is not None:
            checks.append({"rung": f"L{L:g}/Nk{nk}/{arm}/ds{den}", "got": got,
                           "want": want, "pass": bool(got == want)})
    g = {"n_checked": len(checks), "checks": checks,
         "pass": bool(checks) and all(c["pass"] for c in checks)}
    print(f"AUDIT gamma gate: {sum(c['pass'] for c in checks)}/{len(checks)} "
          f"bit-exact -> {'PASS' if g['pass'] else 'FAIL'}")
    return g

def make_report(out_dir) -> dict:
    ladders = _load_ladders(out_dir)
    depth_gate = ladder_depth_gate(ladders)
    print(f"LADDER-DEPTH gate: {'PASS' if depth_gate['pass'] else 'FAIL'}"
          + ("" if depth_gate["pass"] else f"  {depth_gate['violations']}"))
    rep: dict = {"per_nk": {}, "nk_limits": {}, "gates": {}, "commutation": {},
                 "audit_gate": _audit_gate(ladders), "depth_gate": depth_gate}

    for L in sorted(ARM_DEPTHS, reverse=True):
        rows = []
        for nk in NK_ALL:
            loc = _seq(ladders[(L, nk, "local")])
            frz = _seq(ladders[(L, nk, "frozen")])
            sym = _seq(ladders[(L, nk, "localsym")])
            m_lo, m_hi, m_mid, m_half = miss_bracket(loc, frz)
            s_lo, s_hi, s_mid, s_half = miss_bracket(sym, frz)
            f_lim = frozen_richardson(frz)
            old = (loc[0] - frz[0]) / frz[0] * 100.0
            old_sym = (sym[0] - frz[0]) / frz[0] * 100.0
            diag = {arm: ladder_diag(g) for arm, g in
                    (("local", loc), ("frozen", frz), ("localsym", sym))}
            rows.append({
                "Nk": nk, "miss_old_ds1": old, "miss_conv": m_mid,
                "miss_conv_lo": m_lo, "miss_conv_hi": m_hi,
                "miss_conv_half": m_half, "sym_old_ds1": old_sym,
                "sym_conv": s_mid, "sym_conv_half": s_half,
                "frozen_limit_hz": f_lim,
                "local_bracket_hz": list(arm_limit(loc)[:2]),
                "local_last_hz": loc[-1], "frozen_last_hz": frz[-1],
                "sym_last_hz": sym[-1], "diag": diag,
            })
            print(f"L={L:5g} Nk={nk:4d} old={old:+.4f}% "
                  f"conv={m_mid:+.4f} +/- {m_half:.4f}% "
                  f"sym {old_sym:+.4f} -> {s_mid:+.4f}% "
                  f"frozen_lim={f_lim:.6e} "
                  + " ".join(f"{a}:{d['method'][:3]}/tail{d['tail_len']}"
                             for a, d in diag.items()))
        rep["per_nk"][L] = rows

        # ---- Nk->inf on the CONVERGED misses (fit a - c/Nk, Nk >= 30) ----
        sel = [r for r in rows if r["Nk"] >= 30]
        nks = [r["Nk"] for r in sel]
        a_mid, c_mid, resid = fit_limit(nks, [r["miss_conv"] for r in sel])
        a_lo, _, _ = fit_limit(nks, [r["miss_conv_lo"] for r in sel])
        a_hi, _, _ = fit_limit(nks, [r["miss_conv_hi"] for r in sel])
        n1, n2 = nks[-2], nks[-1]
        m1 = next(r["miss_conv"] for r in sel if r["Nk"] == n1)
        m2 = next(r["miss_conv"] for r in sel if r["Nk"] == n2)
        rich = (m2 * n2 - m1 * n1) / (n2 - n1)
        a_old, c_old, resid_old = fit_limit(
            nks, [r["miss_old_ds1"] for r in sel])
        a_sym, _, resid_sym = fit_limit(nks, [r["sym_conv"] for r in sel])
        u_total = 0.5 * abs(a_hi - a_lo) + abs(a_mid - rich) + resid
        rep["nk_limits"][L] = {
            "fit_a": a_mid, "fit_c": c_mid, "fit_max_resid_pp": resid,
            "endpoint_band": [min(a_lo, a_hi), max(a_lo, a_hi)],
            "richardson": rich, "u_total_pp": u_total,
            "fit_a_old_ds1": a_old, "fit_c_old_ds1": c_old,
            "fit_max_resid_old_pp": resid_old, "fit_a_sym": a_sym,
            "fit_max_resid_sym_pp": resid_sym,
            "monotone_conv": all(
                sel[i]["miss_conv"] < sel[i + 1]["miss_conv"]
                for i in range(len(sel) - 1)),
        }
        print(f"L={L:5g} Nk->inf: conv {a_mid:+.4f} +/- {u_total:.4f}% "
              f"(endpoint band [{min(a_lo, a_hi):+.4f}, {max(a_lo, a_hi):+.4f}],"
              f" Richardson {rich:+.4f}, resid {resid:.4f} pp); "
              f"old-ds1 fit {a_old:+.4f} (pinned "
              f"{PINNED_OLD[(L, 'limit')]:+.4f}); sym limit {a_sym:+.4f}")

        r20 = next(r for r in rows if r["Nk"] == 20)
        g = {}
        for key, got in (("miss_asym", r20["miss_conv"]),
                         ("miss_sym", r20["sym_conv"]),
                         ("frozen_hz", r20["frozen_limit_hz"])):
            want, tol = H135_GATES[(L, key)]
            g[key] = {"got": got, "want": want, "tol": tol,
                      "pass": bool(abs(got - want) <= tol)}
        g["old_nk20"] = {"got": r20["miss_old_ds1"],
                         "want": PINNED_OLD[(L, 20)], "tol": 5e-4,
                         "pass": bool(abs(r20["miss_old_ds1"]
                                          - PINNED_OLD[(L, 20)]) <= 5e-4)}
        g["old_limit"] = {"got": a_old, "want": PINNED_OLD[(L, "limit")],
                          "tol": 5e-4,
                          "pass": bool(abs(a_old - PINNED_OLD[(L, "limit")])
                                       <= 5e-4)}
        g["old_sym_nk20"] = {
            "got": r20["sym_old_ds1"], "want": PINNED_OLD[(L, "sym20")],
            "tol": 5e-4,
            "pass": bool(abs(r20["sym_old_ds1"] - PINNED_OLD[(L, "sym20")])
                         <= 5e-4)}

        # gate (d): NO geometric-tail limit may be quoted off a ladder whose
        # a >=2-long monotone-shrinking suffix; a `crossing` arm is exempt
        # BECAUSE it is not tail-extrapolated (band = hull over models).
        bad = [(r["Nk"], arm) for r in rows for arm, d in r["diag"].items()
               if d["method"] == "two-tail" and not d["shrinking"]]
        unshrunk = [(r["Nk"], arm, d["rel_inc_last"])
                    for r in rows for arm, d in r["diag"].items()
                    if d["method"] == "crossing"]
        g["shrinking_tail"] = {
            "got": len(bad), "want": 0, "tol": 0, "pass": not bad,
            "violations": bad, "crossing_arms": unshrunk,
            "min_tail_len": min(d["tail_len"] for r in rows
                                for d in r["diag"].values()
                                if d["method"] == "two-tail")}
        rep["gates"][L] = g
        print(f"L={L:5g} GATES: " + " ".join(
            f"{k}:{'PASS' if v['pass'] else 'FAIL'}" for k, v in g.items()))

        # ---- commutation: Nk->inf at each ds, then dt->0 of a(ds) ----
        dens = sorted(ladders[(L, 30, "local")])
        a_of_ds = []
        for den in dens:
            miss_ds = []
            for nk in nks:
                gl = ladders[(L, nk, "local")][den]
                gf = ladders[(L, nk, "frozen")].get(den)
                if gf is None:
                    break
                miss_ds.append((gl - gf) / gf * 100.0)
            else:
                a_ds, _, _ = fit_limit(nks, miss_ds)
                a_of_ds.append((den, a_ds))
        seq = [a for _, a in a_of_ds]
        try:
            lo, hi, mid, half = two_tail_bracket(seq)
            rep["commutation"][L] = {
                "a_of_ds": a_of_ds, "swapped_limit": mid,
                "swapped_half": half,
                "headline_limit": a_mid,
                "commute_gap_pp": abs(mid - a_mid)}
            print(f"L={L:5g} commutation: swapped-order limit {mid:+.4f} "
                  f"+/- {half:.4f} vs headline {a_mid:+.4f} "
                  f"(gap {abs(mid - a_mid):.4f} pp)")
        except ValueError as e:
            rep["commutation"][L] = {"a_of_ds": a_of_ds, "error": str(e)}
            print(f"L={L:5g} commutation: NOT extrapolable ({e})")

    json.dump(rep, open(Path(out_dir) / "c22_repin_report.json", "w"),
              indent=1)
    print(f"wrote {Path(out_dir) / 'c22_repin_report.json'}")
    return rep

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, choices=["list", "rung", "report"])
    ap.add_argument("--L", type=float)
    ap.add_argument("--Nk", type=int)
    ap.add_argument("--arm", choices=["local", "frozen", "localsym"])
    ap.add_argument("--ds-den", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode-source", default="default",
                    choices=["default", "shipped", "joint", "joint-sumrule"])
    a = ap.parse_args()
    if a.job == "list":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "jobs.txt", "w") as fh:
            for j in build_job_list():
                fh.write(f"{j['L_um']:g} {j['Nk']} {j['arm']} {j['ds_den']}\n")
        print(f"wrote {out / 'jobs.txt'} ({len(build_job_list())} jobs)")
    elif a.job == "rung":
        run_rung(a.L, a.Nk, a.arm, a.ds_den, a.out, mode_source=a.mode_source)
    elif a.job == "report":
        make_report(a.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
