"""Nk-convergence of the deterministic exp(At) miss (Appendix A control)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from pinn_bte.physics.ttg_pn2q import solve_pn2q_decay

CFG = dict(A0_K=75.0, L_max=8, n_phi=32, ic_2c=-1.0, T_lo=140.0, T_hi=460.0, nT=160)

# Nk>=30 is the asymptotic branch used for the fit; 20 is the production value.
NK_FIT = [30, 40, 60, 80, 120, 160]
NK_REPORT = [20]


def miss_at(L_um: float, nk: int) -> tuple[float, float, float]:
    """Return (miss_percent, gamma_local, gamma_frozen) at this Nk."""
    loc = solve_pn2q_decay(L_um, Nk=nk, tau_model="local", **CFG)
    frz = solve_pn2q_decay(L_um, Nk=nk, tau_model="frozen", **CFG)
    return (loc - frz) / frz * 100.0, loc, frz


def fit_limit(nks: list[int], misses: list[float]) -> tuple[float, float, float]:
    """Least-squares fit miss = a - c/Nk. Returns (a, c, max_abs_resid)."""
    x = 1.0 / np.asarray(nks, dtype=float)
    y = np.asarray(misses, dtype=float)
    # y = a - c*x  =>  design [1, -x]
    A = np.stack([np.ones_like(x), -x], axis=1)
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(y - (a - c * x))
    return float(a), float(c), float(resid.max())


def richardson(nk1: int, m1: float, nk2: int, m2: float) -> float:
    """2-point Richardson for a 1/Nk error model: eliminate the 1/Nk term."""
    return (m2 * nk2 - m1 * nk1) / (nk2 - nk1)


def main() -> int:
    out: dict = {"config": {k: v for k, v in CFG.items()}, "lengths": {}}
    for L in (100.0, 10.0):
        rows = []
        for nk in sorted(set(NK_REPORT + NK_FIT)):
            t0 = time.time()
            m, loc, frz = miss_at(L, nk)
            rows.append({"Nk": nk, "miss_pct": m, "gamma_local": loc,
                         "gamma_frozen": frz, "secs": time.time() - t0})
            print(f"L={L:6.1f} Nk={nk:4d}  miss={m:+.4f} %  "
                  f"local={loc:.6e}  frozen={frz:.6e}  ({rows[-1]['secs']:.0f}s)",
                  flush=True)

        fit_rows = [r for r in rows if r["Nk"] >= 30]
        nks = [r["Nk"] for r in fit_rows]
        ms = [r["miss_pct"] for r in fit_rows]
        a, c, resid = fit_limit(nks, ms)
        rich = richardson(nks[-2], ms[-2], nks[-1], ms[-1])
        prod = next(r["miss_pct"] for r in rows if r["Nk"] == 20)

        # The absolute rate is NOT converged -- quantify it so nobody quotes it.
        g20 = next(r["gamma_local"] for r in rows if r["Nk"] == 20)
        gmax = fit_rows[-1]["gamma_local"]
        abs_growth = (gmax - g20) / g20 * 100.0

        print(f"  --> fit  miss(Nk) = {a:.4f} - {c:.4f}/Nk   "
              f"(max resid {resid:.4f} pp over Nk>={nks[0]})")
        print(f"  --> limit {a:+.4f} %   Richardson {rich:+.4f} %   "
              f"(agree {abs(a - rich):.4f} pp)")
        print(f"  --> production Nk=20 {prod:+.4f} % -> UNDER-reports by "
              f"{a - prod:+.4f} pp ({(a - prod) / prod * 100:.2f}% rel)")
        print(f"  --> ABSOLUTE gamma_local NOT converged: "
              f"{abs_growth:+.1f}% from Nk=20 to Nk={nks[-1]} -- ratio only.\n",
              flush=True)

        out["lengths"][str(L)] = {
            "rows": rows, "fit_a": a, "fit_c": c, "fit_max_resid_pp": resid,
            "richardson": rich, "production_Nk20": prod,
            "under_report_pp": a - prod, "abs_gamma_growth_pct": abs_growth,
        }

    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
