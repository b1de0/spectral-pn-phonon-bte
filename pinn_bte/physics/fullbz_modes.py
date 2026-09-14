"""Isotropized full-BZ silicon phonon mode set (first-principles kappa spectrum)."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

SI_KAPPA_BULK_300K: float = 148.0      # W/m/K  (Esfarjani 2011; expt 142-156)
SI_C_VOLUMETRIC_300K: float = 1.63e6   # J/m^3/K (rho c_p = 2329 * 700)

SI_MFP_ACCUM_LAMBDA_UM: np.ndarray = np.array(
    [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
)
SI_MFP_ACCUM_CUMFRAC: np.ndarray = np.array(
    [0.005, 0.02, 0.06, 0.17, 0.32, 0.52, 0.74, 0.93, 0.99, 1.00]
)

_V_LO_MS: float = 1800.0
_V_HI_MS: float = 6000.0

_ANG_PER_M: float = 1e10        # Angstrom per metre
_UM_PER_M: float = 1e6


def _accum_fraction(lambda_um: np.ndarray) -> np.ndarray:
    """Cumulative kappa fraction G(Lambda) at the given MFPs (um), clamped [0,1]."""
    g = np.interp(lambda_um, SI_MFP_ACCUM_LAMBDA_UM, SI_MFP_ACCUM_CUMFRAC)
    return np.clip(g, 0.0, 1.0)


def build_fullbz_modes(
    n_bins: int = 200,
    T_ref: float = 300.0,
    kappa_bulk: float = SI_KAPPA_BULK_300K,
    v_lo: float = _V_LO_MS,
    v_hi: float = _V_HI_MS,
):
    """Isotropized full-BZ Si mode comb; drop-in for ``phonon_modes``."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if abs(T_ref - 300.0) > 1.0:
        # 300-K-only set for stage (i); do not silently fabricate a T-scaling.
        import warnings

        warnings.warn(
            f"build_fullbz_modes is a 300 K set; T_ref={T_ref} returned unscaled",
            stacklevel=2,
        )

    lo_um = float(SI_MFP_ACCUM_LAMBDA_UM[0])
    hi_um = float(SI_MFP_ACCUM_LAMBDA_UM[-1])
    edges = np.geomspace(lo_um, hi_um, n_bins + 1)          # um
    centre = np.sqrt(edges[:-1] * edges[1:])                # geometric bin centre
    d_kappa = kappa_bulk * np.diff(_accum_fraction(edges))  # W/m/K per bin

    # Physical monotone v(Lambda): log-linear from v_lo (0.01 um) to v_hi (10 um).
    span = np.log10(10.0) - np.log10(0.01)
    t = np.clip((np.log10(centre) - np.log10(0.01)) / span, 0.0, 1.0)
    v_ms = v_lo + (v_hi - v_lo) * t

    lam_m = centre / _UM_PER_M
    tau = lam_m / v_ms                                      # s
    C = 3.0 * d_kappa / (v_ms * lam_m)                      # J/m^3/K

    keep = d_kappa > 0.0
    v_ang = v_ms[keep] * _ANG_PER_M                         # Angstrom/s
    return v_ang, tau[keep], C[keep]


def _mode_kappa(v_ang: np.ndarray, tau: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-mode kappa contribution (1/3) C v^2 tau, W/m/K (v given in Angstrom/s)."""
    v_ms = np.asarray(v_ang) / _ANG_PER_M
    return C * v_ms ** 2 * tau / 3.0


def reconstruct_kappa(v_ang: np.ndarray, tau: np.ndarray, C: np.ndarray) -> float:
    """Bulk kappa = sum (1/3) C v^2 tau (W/m/K); acceptance gate (a)."""
    return float(np.sum(_mode_kappa(v_ang, tau, C)))


def total_heat_capacity(C: np.ndarray) -> float:
    """Total volumetric heat capacity sum_b C_b (J/m^3/K); acceptance gate (c)."""
    return float(np.sum(np.asarray(C)))


def mfp_of_modes_um(v_ang: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Per-mode MFP Lambda = v tau in micrometres (v in Angstrom/s)."""
    return np.asarray(v_ang) * np.asarray(tau) * 1e-4  # Angstrom -> um


def mfp_fraction_above(
    v_ang: np.ndarray, tau: np.ndarray, C: np.ndarray, lam_um: float = 1.0
) -> float:
    """Fraction of kappa carried by modes with MFP > lam_um; acceptance gate (b)."""
    km = _mode_kappa(v_ang, tau, C)
    lam = mfp_of_modes_um(v_ang, tau)
    tot = float(np.sum(km))
    if tot <= 0.0:
        return float("nan")
    return float(np.sum(km[lam > lam_um]) / tot)


def mfp_accumulation(
    v_ang: np.ndarray, tau: np.ndarray, C: np.ndarray, lam_um_grid: np.ndarray
) -> np.ndarray:
    """Cumulative kappa fraction G(Lambda) = kappa(<Lambda)/kappa on a MFP grid."""
    km = _mode_kappa(v_ang, tau, C)
    lam = mfp_of_modes_um(v_ang, tau)
    order = np.argsort(lam)
    lam_s, km_s = lam[order], km[order]
    cum = np.cumsum(km_s) / np.sum(km_s)
    return np.interp(np.asarray(lam_um_grid), lam_s, cum, left=0.0, right=1.0)


@contextmanager
def fullbz_mode_source(n_bins: int = 200, T_ref: float = 300.0, **build_kwargs):
    """Temporarily route the TTG solvers to the full-BZ Si comb."""
    import pinn_bte.physics.ttg_dispersion as _disp
    import pinn_bte.physics.ttg_dom as _dom

    modes = build_fullbz_modes(n_bins=n_bins, T_ref=T_ref, **build_kwargs)

    def _inject(Nk=None, T=None, *args, **kwargs):
        return modes

    saved = (_disp.phonon_modes, _dom.phonon_modes)
    _disp.phonon_modes = _inject
    _dom.phonon_modes = _inject
    try:
        yield modes
    finally:
        _disp.phonon_modes, _dom.phonon_modes = saved
