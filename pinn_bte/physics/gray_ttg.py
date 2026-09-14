"""Gray-BTE transient thermal grating: exact collective rate, windowed
amplitude reference and the same-window decay-rate denominator used by the
gray control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinn_bte.physics.decay_metrics import (
    DENOMINATOR_RULE_SAME_WINDOW,
    GAMMA_ESTIMATOR_RULED,
    denominator_provenance,
    integral_gamma,
)

#: xi above which the discrete collective pole vanishes (arctan sup = pi/2).
GRAY_XI_CRITICAL = np.pi / 2.0

ZHOU_DT_TRAIN_MAX_TAU = 0.05


def gray_dominant_gamma(xi: float) -> float | None:
    """Dominant collective decay rate (1/tau), or None beyond the transition."""
    xi = float(xi)
    if xi <= 0.0:
        raise ValueError(f"xi must be positive, got {xi}")
    if xi >= GRAY_XI_CRITICAL:
        return None
    return 1.0 - xi / np.tan(xi)


def gray_window_gamma(xi: float) -> float:
    """The rate the n_decay window is keyed to: pole if it exists, else the
    ballistic sweep rate q<v> = xi — ttg_dom's own fallback for one mode.
    """
    gamma_dominant = gray_dominant_gamma(xi)
    return float(xi) if gamma_dominant is None else gamma_dominant


def gray_amplitude_curve(xi: float, t_end_tau: float,
                         nsteps: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    xi = float(xi)
    nsteps = int(nsteps)
    t = np.linspace(0.0, float(t_end_tau), nsteps + 1)
    dt = t[1] - t[0]
    x = xi * t
    sinc = np.ones_like(x)
    nz = x != 0
    sinc[nz] = np.sin(x[nz]) / x[nz]
    F = sinc * np.exp(-t)          # free-streaming term; K(u) = F(u) for gray
    K = F
    A = np.empty(nsteps + 1)
    A[0] = 1.0
    denom = 1.0 - 0.5 * dt * K[0]  # K(0) = 1: the implicit diagonal
    for n in range(1, nsteps + 1):
        s = 0.5 * K[n] * A[0]
        if n > 1:
            s += K[1:n][::-1] @ A[1:n]
        A[n] = (F[n] + dt * s) / denom
    return t, A


def gray_gamma_same_window(
        xi: float, t_end_tau: float, nsteps: int = 12000,
        estimator: str = GAMMA_ESTIMATOR_RULED,
        return_diag: bool = False) -> float | tuple[float, dict]:
    """The RULED denominator: arbiter rate on the CALLER's window (per tau)."""
    t, A = gray_amplitude_curve(xi, t_end_tau, nsteps=nsteps)
    gamma = integral_gamma(t, A, estimator=estimator)
    if return_diag:
        prov = denominator_provenance(
            estimator=estimator, rule=DENOMINATOR_RULE_SAME_WINDOW,
            t_end_s=float(t_end_tau), n_samples=int(t.size),
            gamma_ref_hz=gamma, xi=xi, time_unit="tau")
        return gamma, dict(prov, t=t, A=A)
    return gamma


@dataclass(frozen=True)
class GrayTTGCase:
    """Everything the runner needs to train and score one gray-TTG arm."""

    xi: float
    Kn: float
    n_decay: float
    t_end_tau: float
    nsteps: int
    #: pole rate (1/tau) or None beyond GRAY_XI_CRITICAL
    gamma_dominant_per_tau: float | None
    gamma_window_per_tau: float
    #: same-window ruled denominator on the native grid
    gamma_ref_per_tau: float
    #: native reference trace (t in tau; A signed, A[0] = 1)
    t_ref: np.ndarray
    A_ref: np.ndarray
    #: decay_metrics.denominator_provenance block (+ xi, time_unit)
    provenance: dict


def build_gray_case(xi: float, n_decay: float = 7.0,
                    t_end_tau: float | None = None,
                    nsteps: int = 12000) -> GrayTTGCase:
    """Arbiter case for one arm: window, native trace, ruled denominator."""
    xi = float(xi)
    gamma_dominant = gray_dominant_gamma(xi)
    gamma_window = gray_window_gamma(xi)
    if t_end_tau is None:
        t_end_tau = float(n_decay) / gamma_window
    gamma_ref, diag = gray_gamma_same_window(
        xi, t_end_tau, nsteps=nsteps, return_diag=True)
    prov = {k: v for k, v in diag.items() if k not in ("t", "A")}
    return GrayTTGCase(
        xi=xi, Kn=xi / (2.0 * np.pi), n_decay=float(n_decay),
        t_end_tau=float(t_end_tau), nsteps=int(nsteps),
        gamma_dominant_per_tau=gamma_dominant,
        gamma_window_per_tau=gamma_window,
        gamma_ref_per_tau=gamma_ref,
        t_ref=diag["t"], A_ref=diag["A"], provenance=prov)
