"""Multi-axis solution battery for the archived spectral-PN runs — CPU only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pinn_bte.physics.decay_metrics import (  # noqa: E402
    GAMMA_ESTIMATOR_LEGACY, integral_gamma,
)
from pinn_bte.training.transient.spectral_pn_trainer import (  # noqa: E402
    SHAPE_NAMES, _shape_metrics,
)


def npz_estimator(d: dict) -> str:
    """Which functional wrote THIS artifact's `gamma_eff`."""
    return str(d["gamma_estimator"]) if "gamma_estimator" in d\
        else GAMMA_ESTIMATOR_LEGACY


#: The 18 corridor-ladder arms (nine periods x two geometries), addressed by
#: their arm directory under data/runs/pn_corridor_ladder_60k/.
_LADDER_LS = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)
RUNS = [(f"L={L:g}um {geo}", f"pn_corridor_ladder_60k/ladder_L{L}_{geo}")
        for geo in ("1d", "2d") for L in _LADDER_LS]


def load(arm_dir: str) -> dict:
    """Load the single *_results.npz of an arm directory under data/runs/."""
    hits = sorted((REPO / "data" / "runs" / arm_dir).glob("*_results.npz"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{arm_dir}: expected exactly one *_results.npz, got {len(hits)}")
    d = np.load(hits[0], allow_pickle=True)
    return {k: d[k] for k in d.files}


def gate(d: dict) -> dict:
    """Reproduce the STORED scalars from the STORED arrays."""
    A, A_ref, t, t_ref = d["A"], d["A_ref"], d["t"], d["t_ref"]
    A_ref_i = np.interp(t, t_ref, A_ref)
    m = dict(zip(SHAPE_NAMES, _shape_metrics(t, A, A_ref_i)))
    est = npz_estimator(d)
    return {
        "rmse_ok": m["rmse"] == float(d["rmse"]),
        "max_abs_dA_ok": m["max_abs_dA"] == float(d["max_abs_dA"]),
        "gamma_eff_ok": integral_gamma(t, A, estimator=est) == float(
            d["gamma_eff"]),
        "gamma_estimator": est,
    }


def battery(d: dict) -> dict:
    A, A_ref, t, t_ref = d["A"], d["A_ref"], d["t"], d["t_ref"]
    A_ref_i = np.interp(t, t_ref, A_ref)
    m = dict(zip(SHAPE_NAMES, _shape_metrics(t, A, A_ref_i)))
    dA = A - A_ref_i
    gamma_eff = float(d["gamma_eff"])
    gamma_ratio = float(d["gamma_ratio"])
    gamma_ref = gamma_eff / gamma_ratio            # the published denominator
    area_ref = 1.0 / gamma_ref                     # == int A_ref dt

    out = dict(m)
    out["gamma_ratio"] = gamma_ratio
    out["A0"] = float(A[0])
    out["A0_exact"] = bool(A[0] == 1.0)

    out["cancel"] = (abs(m["int_signed_dA"]) / m["int_abs_dA"]
                     if m["int_abs_dA"] > 0 else np.nan)
    # area errors as a fraction of the reference area (dimensionless)
    out["int_signed_frac"] = m["int_signed_dA"] / area_ref
    out["int_abs_frac"] = m["int_abs_dA"] / area_ref

    out["clip_live"] = bool(A.min() < 0.0)
    out["gamma_eff_noclip"] = float(A[0] / np.trapezoid(A, t))
    out["clip_shift_pct"] = 100.0 * (gamma_eff - out["gamma_eff_noclip"]) / gamma_eff
    out["gamma_eff_absA"] = integral_gamma(t, A, estimator="absA")
    out["gamma_eff_clip"] = integral_gamma(t, A, estimator="clip")
    out["absA_shift_pct"] = 100.0 * (gamma_eff - out["gamma_eff_absA"]) / gamma_eff

    live = A_ref_i >= 0.01          # 1% of the initial amplitude
    out["live_frac"] = float(live.mean())
    out["rmse_live"] = float(np.sqrt(np.mean(dA[live] ** 2)))
    out["rmse_dead"] = float(np.sqrt(np.mean(dA[~live] ** 2)))
    # rmse == 0 only for the exact identity (nothing to inflate).
    out["rmse_inflation"] = (out["rmse_live"] / m["rmse"] if m["rmse"] > 0
                             else np.nan)

    # --- LOCALIZATION: where in the decay does the error live? -------------
    i = int(np.abs(dA).argmax())
    out["t_at_maxerr_decays"] = float(t[i] * gamma_ref)   # in 1/gamma_ref units
    out["A_ref_at_maxerr"] = float(A_ref_i[i])
    out["sign_at_maxerr"] = float(np.sign(dA[i]))

    pos = (A > 0) & (A_ref_i > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_ours = -np.gradient(np.log(np.where(pos, A, np.nan)), t)
        r_ref = -np.gradient(np.log(np.where(pos, A_ref_i, np.nan)), t)
    for lbl, lo, hi in [("early", 0.5, 0.9), ("mid", 0.1, 0.5), ("late", 0.01, 0.1)]:
        w = pos & (A_ref_i >= lo) & (A_ref_i < hi)
        if w.sum() < 5:
            out[f"rate_{lbl}"] = np.nan
            continue
        # denominator = the REFERENCE's mean local rate in the same A_ref band
        out[f"rate_{lbl}"] = float(np.nanmean(r_ours[w]) / np.nanmean(r_ref[w]))
    # non-exponentiality of the REFERENCE itself: late/early local-rate ratio.
    # ==1 means a single exponential (the elementary baseline).
    we = pos & (A_ref_i >= 0.5) & (A_ref_i < 0.9)
    wl = pos & (A_ref_i >= 0.01) & (A_ref_i < 0.1)
    out["ref_nonexp"] = (float(np.nanmean(r_ref[wl]) / np.nanmean(r_ref[we]))
                         if we.sum() >= 5 and wl.sum() >= 5 else np.nan)
    out["ours_nonexp"] = (float(np.nanmean(r_ours[wl]) / np.nanmean(r_ours[we]))
                          if we.sum() >= 5 and wl.sum() >= 5 else np.nan)

    late_bands = [(0.01, 0.1), (0.02, 0.2), (0.05, 0.3), (0.005, 0.05)]
    vals = []
    for lo, hi in late_bands:
        w = pos & (A_ref_i >= lo) & (A_ref_i < hi)
        if w.sum() >= 5:
            vals.append(float(np.nanmean(r_ours[w]) / np.nanmean(r_ref[w])))
    out["rate_late_spread"] = (max(vals) - min(vals)) if len(vals) > 1 else np.nan
    out["rate_late_robust"] = bool(out["rate_late_spread"] < 0.10)

    out["A_end"] = float(A[-1])
    out["A_ref_end"] = float(A_ref_i[-1])
    out["term_ratio"] = float(A[-1] / A_ref_i[-1])
    out["term_gap"] = float(A[-1] - A_ref_i[-1])
    # is the gap GROWING through the late window (a stalling curve) or shrinking?
    i60 = int(0.6 * len(A))
    out["term_gap_growing"] = bool(abs(A[-1] - A_ref_i[-1]) > abs(A[i60] - A_ref_i[i60]))
    # sign disagreement at t_end: ours undershoots where the reference does not
    out["term_sign_flip"] = bool(np.sign(A[-1]) != np.sign(A_ref_i[-1]))

    # --- LOSS PLATEAU: is this run trained-out, or still descending? --------
    lh = d["loss_history"]
    n = len(lh)
    q = lh[3 * n // 4:, 1]
    out["loss_final"] = float(lh[-1, 1])
    out["loss_lastq_drop"] = float(q[0] / q[-1])   # >1 == still falling
    return out


def fmt(rows: list) -> str:
    lines = []
    lines.append("=" * 118)
    lines.append("VALIDATION GATE (recompute stored scalars from stored arrays)")
    lines.append("=" * 118)
    lines.append(f"{'run':22s} {'rmse':>8s} {'max_abs_dA':>11s} "
                 f"{'gamma_eff':>10s} {'estimator':>10s}")
    for lbl, _, g in rows:
        lines.append(f"{lbl:22s} {str(g['rmse_ok']):>8s} "
                     f"{str(g['max_abs_dA_ok']):>11s} "
                     f"{str(g['gamma_eff_ok']):>10s} {g['gamma_estimator']:>10s}")
    lines.append("")
    lines.append("=" * 118)
    lines.append("BATTERY  (A-unit metrics: denominator = A(0) = 1, i.e. the INITIAL AMPLITUDE)")
    lines.append("=" * 118)
    hdr = (f"{'run':22s} {'gamma_r':>8s} {'rmse':>8s} {'max|dA|':>8s} "
           f"{'cancel':>7s} {'A_min':>9s} {'clip':>5s} {'live%':>6s} "
           f"{'rmse_liv':>8s} {'infl':>5s}")
    lines.append(hdr)
    for lbl, b, _ in rows:
        lines.append(f"{lbl:22s} {b['gamma_ratio']:8.4f} {b['rmse']:8.5f} "
                     f"{b['max_abs_dA']:8.5f} {b['cancel']:7.3f} "
                     f"{b['A_min']:9.5f} {str(b['clip_live']):>5s} "
                     f"{100*b['live_frac']:6.1f} {b['rmse_live']:8.5f} "
                     f"{b['rmse_inflation']:5.2f}")
    lines.append("")
    lines.append("=" * 118)
    lines.append("SHAPE-RESOLVED RATE — local -dlnA/dt OURS / ARBITER at matched A_ref levels")
    lines.append("(denominator = the arbiter's mean local rate in the SAME A_ref band; 1.000 == right shape)")
    lines.append("=" * 118)
    lines.append(f"{'run':22s} {'early':>8s} {'mid':>8s} {'late':>8s} {'lateSprd':>8s} "
                 f"{'robust':>6s} {'ref_nonx':>8s} {'our_nonx':>8s} {'t@maxerr':>9s}")
    for lbl, b, _ in rows:
        lines.append(f"{lbl:22s} {b['rate_early']:8.4f} {b['rate_mid']:8.4f} "
                     f"{b['rate_late']:8.4f} {b['rate_late_spread']:8.4f} "
                     f"{str(b['rate_late_robust']):>6s} {b['ref_nonexp']:8.4f} "
                     f"{b['ours_nonexp']:8.4f} {b['t_at_maxerr_decays']:9.3f}")
    lines.append("")
    lines.append("TERMINAL FLOOR — band-free (valid where the late rate is not)")
    lines.append(f"{'run':22s} {'A(t_end)':>10s} {'A_ref(end)':>11s} {'ratio':>8s} "
                 f"{'gap/A(0)':>9s} {'growing':>8s} {'signflip':>9s}")
    for lbl, b, _ in rows:
        lines.append(f"{lbl:22s} {b['A_end']:10.6f} {b['A_ref_end']:11.6f} "
                     f"{b['term_ratio']:8.2f} {b['term_gap']:+9.5f} "
                     f"{str(b['term_gap_growing']):>8s} {str(b['term_sign_flip']):>9s}")
    lines.append("")
    lines.append("=" * 118)
    lines.append("AREA (denominator = int A_ref dt = 1/gamma_ref) and LOSS PLATEAU")
    lines.append("=" * 118)
    lines.append(f"{'run':22s} {'int_sgn%':>9s} {'int_abs%':>9s} "
                 f"{'loss_fin':>10s} {'lastQ_drop':>11s} {'max_dA_fwd':>11s}")
    for lbl, b, _ in rows:
        lines.append(f"{lbl:22s} {100*b['int_signed_frac']:9.3f} "
                     f"{100*b['int_abs_frac']:9.3f} {b['loss_final']:10.3e} "
                     f"{b['loss_lastq_drop']:11.3f} {b['max_dA_fwd']:11.3e}")
    return "\n".join(lines)


def main() -> None:
    rows = []
    for lbl, arm in RUNS:
        d = load(arm)
        g = gate(d)
        if not all(g.values()):
            raise SystemExit(f"GATE FAILED on {lbl}: {g} — STOP, do not trust downstream")
        rows.append((lbl, battery(d), g))
    print(fmt(rows))


if __name__ == "__main__":
    main()
