"""Reference (non-neural) integrator of the (DC, q, 2q) nonlinear PN system."""
from __future__ import annotations

import numpy as np

from pinn_bte.physics.decay_metrics import GAMMA_ESTIMATOR_RULED, integral_gamma
from pinn_bte.physics.ttg_dispersion import phonon_modes, dominant_gamma
from pinn_bte.physics.ttg_dom import _tables

ANGSTROM_PER_UM = 1e4


def _streaming_matrix(L_max):
    T = np.zeros((L_max + 1, L_max + 1))
    for l in range(L_max + 1):
        if l - 1 >= 0:
            T[l, l - 1] = l / (2 * l - 1)
        if l + 1 <= L_max:
            T[l, l + 1] = (l + 1) / (2 * l + 3)
    return T


_HW = np.array([1.0, 2.0, 2.0, 2.0, 2.0])   # real-Fourier projection weights


def _phase_basis(n_phi):
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    return np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                     np.cos(2 * phi), np.sin(2 * phi)])              # (5, n_phi)


def _rate_field(dT_h, tau0, tau_table, Tg, T_ref, tau_model, B):
    """Per-mode rate field g_m(x)=1/tau_m(T_ref+dT(x)) on the phase grid."""
    n_phi = B.shape[1]
    if tau_model == "frozen":
        return np.tile(1.0 / tau0, (n_phi, 1))
    dTx = B.T @ np.asarray(dT_h)                                     # (n_phi,)
    Tx = np.clip(T_ref + dTx, Tg[0], Tg[-1])
    inv = 1.0 / tau_table                                           # (nT, M)
    idx = np.clip(np.searchsorted(Tg, Tx), 1, len(Tg) - 1)
    frac = ((Tx - Tg[idx - 1]) / (Tg[idx] - Tg[idx - 1]))[:, None]
    return inv[idx - 1] + frac * (inv[idx] - inv[idx - 1])          # (n_phi, M)


def _project(field, B):
    """Project a phase-grid field onto the {DC,q,2q} harmonics."""
    return (_HW[:, None] * B) @ field / B.shape[1]


def solve_pn2q_decay(L_um, A0_K=1.0, Nk=8, L_max=6, T_ref=300.0,
                     tau_model="local", n_decay=4.0, n_phi=16,
                     single_harmonic=False, return_trace=False,
                     ic_2c=0.0, ic_2s=0.0, T_lo=180.0, T_hi=420.0, nT=160,
                     gamma_estimator=GAMMA_ESTIMATOR_RULED):
    v, tau0, C = phonon_modes(Nk, T_ref)
    M = len(v)
    C = C / C.mean()
    Tg, tauT, _, _, _ = _tables(Nk, T_ref, T_lo=T_lo, T_hi=T_hi, nT=nT)
    Lang = L_um * ANGSTROM_PER_UM
    q = 2 * np.pi / Lang
    Tmat = _streaming_matrix(L_max)
    L1 = L_max + 1
    Dfroz = np.sum(C / tau0)

    Tt = Tmat.T                                                     # [T a]_l = a @ Tt
    vqr = (v * q)                                                    # (M,)
    B = _phase_basis(n_phi)                                          # (5, n_phi)
    projmat = (_HW[:, None] * B) / n_phi                             # (5, n_phi)

    def clos0(E):                                                   # frozen closure
        return np.sum(E / tau0) / Dfroz

    gd_c = dominant_gamma(L_um, "1d", Nk=Nk)[0]
    gd = gd_c.real if gd_c is not None else float(q * np.sum(C * v) / np.sum(C))
    t_end = n_decay / gd
    dt = 0.4 / (2.0 * q * v.max())
    nsteps = max(int(np.ceil(t_end / dt)), 300)
    dt = t_end / nsteps

    coeffs = np.zeros((5, M, L1))
    coeffs[1, :, 0] = C                                            # cos(qx) IC
    coeffs[3, :, 0] = ic_2c * C                                    # cos(2qx): asym drive
    coeffs[4, :, 0] = ic_2s * C                                    # sin(2qx): asym drive
    A_hist = np.empty(nsteps + 1); A_hist[0] = 1.0
    dT2c_hist = np.empty(nsteps + 1); dT2c_hist[0] = 0.0
    for n in range(nsteps):
        a0, c1, s1, c2, s2 = coeffs
        dT_drive = A0_K * np.array([clos0(a0[:, 0]), clos0(c1[:, 0]),
                                    clos0(s1[:, 0]), clos0(c2[:, 0]),
                                    clos0(s2[:, 0])])
        g_ph = _rate_field(dT_drive, tau0, tauT, Tg, T_ref, tau_model, B)
        if single_harmonic:            # mean-field control: spatially-uniform rate
            g_ph = np.broadcast_to(g_ph.mean(0), g_ph.shape)
        gT = g_ph.T                                                # (M, n_phi)

        # --- IMPLICIT collision on the phase grid ---
        fph = np.einsum("hml,hx->mlx", coeffs, B)                 # (M,L1,n_phi)
        denom = 1.0 + dt * gT                                      # (M,n_phi)
        # l=0: implicit-moment closure per phase point (energy conserving)
        p = fph[:, 0, :] / denom                                  # (M,n_phi)
        r = dt * gT * C[:, None] / denom
        Gden = np.sum(gT * C[:, None], axis=0)                    # (n_phi,)
        dTeq = (np.sum(gT * p, axis=0) / Gden) / (1.0 - np.sum(gT * r, axis=0) / Gden)
        fph[:, 0, :] = p + r * dTeq[None, :]
        fph[:, 1:, :] = fph[:, 1:, :] / denom[:, None, :]         # l>=1: pure decay
        coeffs = np.einsum("mlx,hx->hml", fph, projmat)           # project back
        if single_harmonic:                                       # drop DC/2q blocks
            coeffs[0] = 0.0; coeffs[3] = 0.0; coeffs[4] = 0.0

        a0, c1, s1, c2, s2 = coeffs
        c1 -= dt * vqr[:, None] * (s1 @ Tt)
        s1 += dt * vqr[:, None] * (c1 @ Tt)
        c2 -= dt * 2.0 * vqr[:, None] * (s2 @ Tt)
        s2 += dt * 2.0 * vqr[:, None] * (c2 @ Tt)
        coeffs = np.stack([a0, c1, s1, c2, s2])

        A_hist[n + 1] = clos0(c1[:, 0])
        dT2c_hist[n + 1] = clos0(c2[:, 0])

    A = A_hist / A_hist[0]
    t = np.arange(nsteps + 1) * dt
    gamma = integral_gamma(t, A, estimator=gamma_estimator)
    if return_trace:
        return gamma, dict(t=t, A=A, tau_model=tau_model, dT2c=dT2c_hist,
                           gamma_estimator=gamma_estimator)
    return gamma


def taut_shift(L_um, A0_K=100.0, Nk=8, L_max=6, single_harmonic=False, **kw):
    """(gamma_local - gamma_frozen)/gamma_frozen in percent — the pure tau(T)
    effect, directly comparable to the DOM's ttg_decay local-vs-frozen.
    """
    gl = solve_pn2q_decay(L_um, A0_K=A0_K, Nk=Nk, L_max=L_max,
                          tau_model="local", single_harmonic=single_harmonic, **kw)
    gf = solve_pn2q_decay(L_um, A0_K=A0_K, Nk=Nk, L_max=L_max,
                          tau_model="frozen", single_harmonic=single_harmonic, **kw)
    return gl, gf, (gl - gf) / gf * 100.0
