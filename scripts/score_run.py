"""Score an archived run npz to the printed decay-rate ratio."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pinn_bte.physics.decay_metrics import (  # noqa: E402
    GAMMA_ESTIMATOR_RULED,
    integral_gamma,
)


def amp_1d(T: np.ndarray) -> np.ndarray:
    """Fundamental cosine amplitude of a 1D field T (Nt, Nx): 2*<T cos(2 pi x)>."""
    Nx = T.shape[1]
    x = np.linspace(0, 1, Nx, endpoint=False)
    return 2.0 * np.mean(T * np.cos(2 * np.pi * x)[None, :], axis=1)


def _load(run_dir: Path):
    hits = sorted(run_dir.glob("*_results.npz"))
    if len(hits) != 1:
        raise SystemExit(f"{run_dir}: expected exactly one *_results.npz, "
                         f"got {len(hits)}")
    return hits[0], np.load(hits[0], allow_pickle=True)


def own_mode_source(d) -> str:
    """The run's OWN reference kind, read from the artifact."""
    return str(d["mode_source"]) if "mode_source" in d.files else "shipped"


def resolve_mode_source(d, requested: str | None) -> str:
    """Dispatch by the artifact, never by a name default."""
    own = own_mode_source(d)
    if requested is None or requested == own:
        return own
    if "mode_source" in d.files:
        raise SystemExit(
            f"--mode-source {requested!r} contradicts the run's own declared "
            f"mode_source={own!r}.  A gamma_ratio formed across two mode sets "
            f"reproduces no published number; drop the flag to score on the "
            f"run's own reference kind.")
    print(f"WARNING: DIAGNOSTIC cross-comb ratio.  This run predates the "
          f"joint adoption (no mode_source field; its own reference kind is "
          f"{own!r}); the number below divides by the {requested!r} set and "
          f"reproduces no published value.")
    return requested


def score_trace_run(d) -> dict:
    """A run that stores its own (t, A, t_ref, A_ref)."""
    t, A = np.asarray(d["t"], float), np.asarray(d["A"], float)
    t_ref, A_ref = np.asarray(d["t_ref"], float), np.asarray(d["A_ref"], float)
    if t_ref.size != 12001:
        raise SystemExit(
            f"stored reference has {t_ref.size} samples, not the reference "
            f"solver's native 12001 -- the same-window denominator does not "
            f"apply to it")
    g = integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
    g_ref = integral_gamma(t_ref, A_ref, estimator=GAMMA_ESTIMATOR_RULED)
    out = dict(gamma_eff=g, gamma_ref=g_ref, gamma_ratio=g / g_ref)
    if "gamma_ratio" in d.files and np.isfinite(float(d["gamma_ratio"])):
        out["gamma_ratio_stored"] = float(d["gamma_ratio"])
    if "rmse" in d.files:
        out["rmse_stored"] = float(d["rmse"])
    return out


def score_field_run(d, mode_source: str) -> dict:
    """A sampled-residual control run that stores the field T(t, x)."""
    from pinn_bte.physics.ttg_dom import ttg_gamma_same_window
    from pinn_bte.physics.optical_reservoir import joint_mode_source
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    A = amp_1d(np.asarray(d["T"], float))
    t = np.asarray(d["t"], float)
    Lt, tau_ref = float(d["Lt"]), float(d["tau_ref"])
    L_um = float(d["L_um"])
    Nk = int(d.get("Nk", 20)) if hasattr(d, "get") else 20
    t_tau = t / t.max() * Lt
    g = integral_gamma(t_tau, A, estimator=GAMMA_ESTIMATOR_RULED)
    ctx = (joint_mode_source if mode_source == "joint" else shipped_mode_source)
    with ctx(Nk=Nk, T_ref=300.0):
        g_ref = ttg_gamma_same_window(L_um, "1d", Nk=Nk,
                                      t_end_s=Lt * tau_ref)
    return dict(gamma_eff_per_tau=g, gamma_ref_hz=g_ref,
                gamma_ratio=(g / tau_ref) / g_ref, mode_source=mode_source,
                L_um=L_um, n_samples=int(A.size), A_min=float(A.min()))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path,
                    help="an arm directory under data/runs/ containing one "
                         "*_results.npz")
    ap.add_argument("--mode-source", default=None,
                    choices=["joint", "shipped"],
                    help="mode set for the DOM denominator of FIELD runs "
                         "(trace runs carry their own stored reference). "
                         "DEFAULT: the run's OWN reference kind, read from "
                         "the npz `mode_source` field (absent = the shipped, "
                         "pre-adoption set).  An explicit value that "
                         "contradicts a declared field is refused; on a "
                         "pre-adoption run it produces a clearly labelled "
                         "DIAGNOSTIC cross-comb number.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path, d = _load(args.run_dir)
    if "A" in d.files and "t_ref" in d.files and np.asarray(d["t_ref"]).size:
        res = score_trace_run(d)
    elif "T" in d.files:
        res = score_field_run(d, resolve_mode_source(d, args.mode_source))
    else:
        raise SystemExit(f"{path}: neither a trace run (t/A/t_ref/A_ref) nor "
                         f"a field run (T)")
    print(f"run: {path}")
    for k, v in res.items():
        print(f"  {k:>20} = {v:.10g}" if isinstance(v, float)
              else f"  {k:>20} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
