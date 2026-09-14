"""H-parameterized reference integrator of the (DC, q, .., Hq) nonlinear PN system
— the `solve_pn2q_decay` algorithm with the harmonic truncation H as a
parameter (2H+1 rows) instead of hard-coded 5 rows.
"""
from __future__ import annotations

import numpy as np

from pinn_bte.physics.decay_metrics import GAMMA_ESTIMATOR_RULED, integral_gamma
from pinn_bte.physics.ttg_dispersion import phonon_modes, dominant_gamma

from pinn_bte.physics.ttg_dom import _tables
from pinn_bte.physics.ttg_pn2q import ANGSTROM_PER_UM, _streaming_matrix

_EDGE_TOL_K = 1e-9   # clamp beyond this (in K) is a real table-edge touch


class TauGridEdgeError(ValueError):
    """The driving temperature left the tau(T) table — widen T_lo/T_hi."""


def _hw(H: int) -> np.ndarray:
    """Real-Fourier projection weights (1 for DC, 2 for each AC row)."""
    w = np.full(2 * H + 1, 2.0)
    w[0] = 1.0
    return w


def _phase_basis_h(n_phi: int, H: int) -> np.ndarray:
    """(2H+1, n_phi) rows [1, cos(phi), sin(phi), .., cos(H phi), sin(H phi)] on
    the midpoint grid — bit-identical to ttg_pn2q._phase_basis at H=2.
    """
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    rows = [np.ones_like(phi)]
    for h in range(1, H + 1):
        rows += [np.cos(h * phi), np.sin(h * phi)]
    return np.stack(rows)


def _rate_field_h(dT_h, tau0, tau_table, Tg, T_ref, tau_model, B):
    """Per-mode rate field g_m(x)=1/tau_m(T_ref+dT(x)) on the phase grid."""
    n_phi = B.shape[1]
    if tau_model == "frozen":
        return np.tile(1.0 / tau0, (n_phi, 1))
    dTx = B.T @ np.asarray(dT_h)                                     # (n_phi,)
    Tx_raw = T_ref + dTx
    lo, hi = Tg[0], Tg[-1]
    if Tx_raw.min() < lo - _EDGE_TOL_K or Tx_raw.max() > hi + _EDGE_TOL_K:
        raise TauGridEdgeError(
            f"driving T in [{Tx_raw.min():.3f}, {Tx_raw.max():.3f}] K leaves "
            f"the tau(T) table [{lo:g}, {hi:g}] K — widen T_lo/T_hi (the "
            f"Appendix-A asymmetric drive needs 140/460, not the 180/420 "
            f"defaults; silent clamping biases the nonlinear miss)")
    Tx = np.clip(Tx_raw, lo, hi)                       # inert after the check
    inv = 1.0 / tau_table                                           # (nT, M)
    idx = np.clip(np.searchsorted(Tg, Tx), 1, len(Tg) - 1)
    frac = ((Tx - Tg[idx - 1]) / (Tg[idx] - Tg[idx - 1]))[:, None]
    return inv[idx - 1] + frac * (inv[idx] - inv[idx - 1])          # (n_phi, M)


def solve_pnHq_decay(L_um, H=2, A0_K=1.0, Nk=8, L_max=6, T_ref=300.0,
                     tau_model="local", n_decay=4.0, n_phi=16,
                     return_trace=False, ic_2c=0.0, ic_2s=0.0,
                     T_lo=180.0, T_hi=420.0, nT=160, dt_scale=1.0,
                     snap_idx=None, gamma_estimator=GAMMA_ESTIMATOR_RULED):
    if H < 2:
        raise ValueError(f"H={H}: the H-solver carries the {{DC,q,2q}} core; "
                         f"use spectral_pn for the single-harmonic system")
    NH = 2 * H + 1
    v, tau0, C = phonon_modes(Nk, T_ref)
    M = len(v)
    C = C / C.mean()                     # O(1) normalization (see ttg_pn2q)
    Tg, tauT, _, _, _ = _tables(Nk, T_ref, T_lo=T_lo, T_hi=T_hi, nT=nT)
    if tauT.shape[1] != M:
        raise ValueError(
            f"mode-source mismatch: phonon_modes gave {M} modes but the "
            f"temperature tables gave {tauT.shape[1]}. The solver's mode "
            f"arrays and its tau(T) tables are on different combs -- route "
            f"BOTH through the same mode source.")
    Lang = L_um * ANGSTROM_PER_UM
    q = 2 * np.pi / Lang
    Tmat = _streaming_matrix(L_max)
    L1 = L_max + 1
    Dfroz = np.sum(C / tau0)

    Tt = Tmat.T
    vqr = v * q
    B = _phase_basis_h(n_phi, H)                                     # (NH, n_phi)
    hw = _hw(H)
    projmat = (hw[:, None] * B) / n_phi                              # (NH, n_phi)

    def clos0(E):                                                   # frozen closure
        return np.sum(E / tau0) / Dfroz

    gd_c = dominant_gamma(L_um, "1d", Nk=Nk)[0]
    gd = gd_c.real if gd_c is not None else float(q * np.sum(C * v) / np.sum(C))
    t_end = n_decay / gd
    # CFL from the fastest harmonic H*q*v (== the 2q solver's dt at H=2)
    dt = 0.4 / (float(H) * q * v.max()) * dt_scale
    nsteps = max(int(np.ceil(t_end / dt)), 300)
    dt = t_end / nsteps

    coeffs = np.zeros((NH, M, L1))
    coeffs[1, :, 0] = C                                            # cos(qx) IC
    coeffs[3, :, 0] = ic_2c * C                                    # cos(2qx): asym drive
    coeffs[4, :, 0] = ic_2s * C                                    # sin(2qx): asym drive
    A_hist = np.empty(nsteps + 1); A_hist[0] = 1.0
    dTh_hist = np.zeros((NH, nsteps + 1))
    dTh_hist[:, 0] = [clos0(coeffs[h][:, 0]) for h in range(NH)]
    snap = set(snap_idx) if snap_idx is not None else set()
    out_pc = {}

    for n in range(nsteps):
        dT_drive = A0_K * np.array([clos0(coeffs[h][:, 0]) for h in range(NH)])
        g_ph = _rate_field_h(dT_drive, tau0, tauT, Tg, T_ref, tau_model, B)
        gT = g_ph.T                                                # (M, n_phi)

        # --- IMPLICIT collision on the phase grid (verbatim ttg_pn2q) ---
        fph = np.einsum("hml,hx->mlx", coeffs, B)                 # (M,L1,n_phi)
        denom = 1.0 + dt * gT                                      # (M,n_phi)
        p = fph[:, 0, :] / denom                                  # (M,n_phi)
        r = dt * gT * C[:, None] / denom
        Gden = np.sum(gT * C[:, None], axis=0)                    # (n_phi,)
        dTeq = (np.sum(gT * p, axis=0) / Gden) / (1.0 - np.sum(gT * r, axis=0) / Gden)
        fph[:, 0, :] = p + r * dTeq[None, :]
        fph[:, 1:, :] = fph[:, 1:, :] / denom[:, None, :]         # l>=1: pure decay
        coeffs = np.einsum("mlx,hx->hml", fph, projmat)           # project back

        if n in snap:
            out_pc[n] = coeffs.copy()      # MUST copy: streaming is in-place

        for h in range(1, H + 1):
            ch, sh = coeffs[2 * h - 1], coeffs[2 * h]
            ch -= dt * h * vqr[:, None] * (sh @ Tt)
            sh += dt * h * vqr[:, None] * (ch @ Tt)

        A_hist[n + 1] = clos0(coeffs[1][:, 0])
        dTh_hist[:, n + 1] = [clos0(coeffs[h][:, 0]) for h in range(NH)]

    A = A_hist / A_hist[0]
    t = np.arange(nsteps + 1) * dt
    gamma = integral_gamma(t, A, estimator=gamma_estimator)
    if return_trace:
        return gamma, dict(t=t, A=A, dTh=dTh_hist, tau_model=tau_model,
                           dt=dt, nsteps=nsteps, H=H, coeff_pc=out_pc,
                           gamma_estimator=gamma_estimator)
    return gamma
