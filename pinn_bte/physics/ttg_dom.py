"""Deterministic discrete-ordinates (DOM) solver for transient-thermal-grating
decay.
"""
from __future__ import annotations

import numpy as np

from pinn_bte.physics.decay_metrics import (
    DENOMINATOR_RULE_RULED,
    DENOMINATOR_RULE_SAME_WINDOW,
    DENOMINATOR_RULES,
    GAMMA_ESTIMATOR_RULED,
    denominator_provenance,
    integral_gamma,
)
from pinn_bte.physics.ttg_dispersion import phonon_modes, dominant_gamma

ANGSTROM_PER_UM = 1e4


WINDOW_RULE_FULL = "full"
WINDOW_RULE_LEGACY = "legacy-earlystop"
_WINDOW_RULES = (WINDOW_RULE_FULL, WINDOW_RULE_LEGACY)

_LEGACY_STOP_REL = 1e-4

STEPS_PER_DECAY_DEFAULT = 1000


def _tables(Nk, T_ref, T_lo=180.0, T_hi=420.0, nT=160):
    """Per-mode tau(T), C(T), and cumulative dE^eq(T)=int_{T_ref}^T C dT', on a
    grid.
    """
    Tg = np.linspace(T_lo, T_hi, nT)
    v0, _, _ = phonon_modes(Nk, T_ref)
    M = len(v0)
    tauT = np.empty((nT, M)); CT = np.empty((nT, M))
    for a, T in enumerate(Tg):
        _, tau, C = phonon_modes(Nk, float(T)); tauT[a] = tau; CT[a] = C
    dE = np.zeros((nT, M))
    iref = int(np.argmin(np.abs(Tg - T_ref)))
    cum = np.zeros(M)
    for a in range(1, nT):
        cum = cum + 0.5 * (CT[a] + CT[a - 1]) * (Tg[a] - Tg[a - 1])
        dE[a] = cum
    dE -= dE[iref]
    v, _, _ = phonon_modes(Nk, T_ref)
    return Tg, tauT, CT, dE, v


def ttg_decay(L_um, A0_K=1.0, Nk=20, Nmu=16, Nx=80, n_decay=6, T_ref=300.0,
              tau_model="local", collision="explicit", return_diag=False,
              n_harmonics=0, gamma_estimator=GAMMA_ESTIMATOR_RULED):
    """Decay rate gamma_eff (Hz) of a TTG of amplitude A0_K at length L_um."""
    Tg, tauT, CT, dE, v = _tables(Nk, T_ref)
    M = len(v)

    def interp(grid, T):  # grid (nT,M), T (Nx,) -> (M,Nx)
        return np.stack([np.interp(T, Tg, grid[:, m]) for m in range(M)])

    Lang = L_um * ANGSTROM_PER_UM
    q = 2 * np.pi / Lang
    mu, wmu = np.polynomial.legendre.leggauss(Nmu); wmu = wmu / wmu.sum()
    dx = Lang / Nx; xj = (np.arange(Nx) + 0.5) * dx; cosx = np.cos(q * xj)
    dT0 = A0_K * cosx

    g = interp(dE, T_ref + dT0)[:, None, :] * np.ones((M, Nmu, Nx))
    Tcell = T_ref + dT0.copy()
    tau_ref_modes = interp(tauT, np.full(Nx, T_ref))  # (M,Nx), used if frozen
    cfl = dx / (v.max() * max(abs(mu).max(), 1e-3))
    dt = 0.4 * cfl if collision == "implicit" else 0.4 * min(cfl, tauT.min())
    gseed_c = dominant_gamma(L_um, "1d", Nk=Nk)[0]
    if gseed_c is None:  # ballistic: no eigenvalue -> use the gray rate to size the window
        _v, _t, _C = phonon_modes(Nk, T_ref); _Kn = _v * _t * q
        gseed = float(np.sum(_C * (_v * q) ** 2 * _t / (1 + _Kn ** 2)) / np.sum(_C)) / 3.0
    else:
        gseed = gseed_c.real
    nsteps = int((n_decay / gseed) / dt)
    mpos = mu > 0

    n_harm = int(n_harmonics)
    cos_kx = (np.cos(np.arange(1, n_harm + 1)[:, None] * q * xj[None, :])
              if n_harm else None)
    a_harm = []
    A = []; tt = []; tau_ratio = 1.0; T_lo = T_ref; T_hi = T_ref
    e_mean0 = float(np.sum(np.tensordot(wmu, g, axes=([0], [1]))))  # total energy (~0 for cos)
    e_drift = 0.0
    for n in range(nsteps):
        gbar = np.tensordot(wmu, g, axes=([0], [1]))   # (M,Nx) <g_m>
        a_m = 1.0 / interp(tauT, Tcell)                # tau-weight at prev T
        tgt = np.sum(a_m * gbar, axis=0)
        LHS = np.sum(a_m * interp(dE, Tcell), axis=0)
        dLHS = np.sum(a_m * interp(CT, Tcell), axis=0)
        Tcell = Tcell + (tgt - LHS) / dLHS             # Newton step (tau-weighted moment)
        geq = interp(dE, Tcell)                         # (M,Nx) isotropic equilibrium
        tau_now = tau_ref_modes if tau_model == "frozen" else interp(tauT, Tcell)

        dgdx = np.empty_like(g)
        bwd = (g - np.roll(g, 1, 2)) / dx; fwd = (np.roll(g, -1, 2) - g) / dx
        dgdx[:, mpos, :] = bwd[:, mpos, :]; dgdx[:, ~mpos, :] = fwd[:, ~mpos, :]
        stream = v[:, None, None] * mu[None, :, None] * dgdx
        tau3 = tau_now[:, None, :]
        if collision == "implicit":  # unconditionally stable in tau (diffusive L feasible)
            g = (g - dt * stream + dt * geq[:, None, :] / tau3) / (1.0 + dt / tau3)
        else:
            g = g - dt * (stream + (g - geq[:, None, :]) / tau3)

        At = 2 * np.mean((Tcell - T_ref) * cosx); A.append(At); tt.append(n * dt)
        if n_harm:
            a_harm.append(2 * np.mean((Tcell - T_ref)[None, :] * cos_kx, axis=1))
        T_lo = min(T_lo, Tcell.min()); T_hi = max(T_hi, Tcell.max())
        tau_ratio = max(tau_ratio, float(np.max(interp(tauT, Tcell)) / np.min(interp(tauT, Tcell))))
        e_drift = max(e_drift, abs(float(np.sum(np.tensordot(wmu, g, axes=([0], [1]))) - e_mean0)))
        if abs(At) < 1e-3 * abs(A[0]):
            break
    A = np.array(A); tt = np.array(tt)
    gamma = integral_gamma(tt, A, estimator=gamma_estimator)
    if return_diag:
        # normalise energy drift by a per-mode energy scale
        escale = abs(float(np.sum(interp(dE, T_ref + A0_K * np.ones(Nx)))))
        diag = dict(A=A, t=tt, gamma=gamma, energy_drift=e_drift / (escale + 1e-30),
                    tau_ratio=tau_ratio, T_range=(T_lo, T_hi),
                    gamma_estimator=gamma_estimator)
        if n_harm:
            diag["a_harm"] = np.array(a_harm)
        return gamma, diag
    return gamma


def _angular(geometry, Nmu=32, ntheta=32, nphi=64):
    """(Omega_x, weights) for the streaming direction-cosine along the grating
    (x). '1d' thin film: axisymmetric about x -> Omega_x = mu, uniform on
    [-1,1] (exact, closed quadrature).
    """
    if geometry == "1d":
        mu, w = np.polynomial.legendre.leggauss(Nmu)
        return mu, w / w.sum()
    th, wth = np.polynomial.legendre.leggauss(ntheta)
    ph, wph = np.polynomial.legendre.leggauss(nphi)
    TH, PH = np.meshgrid((th + 1) * np.pi / 2, (ph + 1) * np.pi, indexing="ij")
    W = (np.outer(wth * np.pi / 2, wph * np.pi) * np.sin(TH)).ravel(); W /= W.sum()
    return (np.sin(TH) * np.cos(PH)).ravel(), W


def _floor_nmu(geometry, angular_kw, nmu_floor):
    """Applied AFTER any adaptive guard so it can only raise the order, never
    bypass the guard. nmu_floor=None/0 is a no-op (byte-identical default).
    """
    if not nmu_floor:
        return angular_kw
    if geometry == "1d":
        angular_kw["Nmu"] = max(int(nmu_floor), angular_kw.get("Nmu", 32))
    else:
        angular_kw["ntheta"] = max(int(nmu_floor), angular_kw.get("ntheta", 32))
        angular_kw["nphi"] = max(2 * int(nmu_floor), angular_kw.get("nphi", 64))
    return angular_kw


def ttg_decay_spectral(L_um, geometry="1d", A0_K=1.0, Nk=20, T_ref=300.0,
                       n_decay=12, nsteps=12000, nmu_floor=None,
                       window_rule=WINDOW_RULE_FULL, steps_per_decay=None,
                       return_diag=False, **angular_kw):
    if window_rule not in _WINDOW_RULES:
        raise ValueError(f"window_rule must be one of {_WINDOW_RULES}, "
                         f"got {window_rule!r}")
    v, tau, C = phonon_modes(Nk, T_ref)
    q = 2 * np.pi / (L_um * ANGSTROM_PER_UM)
    Ox, w = _angular(geometry, **_floor_nmu(geometry, angular_kw, nmu_floor))
    invtau = 1.0 / tau; Cinvtau = np.sum(C * invtau)

    def mom(x):  # tau-weighted temperature moment ΔT̂
        return np.sum(invtau * (x @ w)) / Cinvtau

    ghat = (C[:, None] * A0_K).astype(complex) * np.ones((len(v), len(Ox)), dtype=complex)
    gd_c = dominant_gamma(L_um, "1d", Nk=Nk)[0]
    gd = gd_c.real if gd_c is not None else float(q * np.sum(C * v) / np.sum(C))
    if steps_per_decay:
        nsteps = int(round(n_decay * steps_per_decay))
    dt = (n_decay / gd) / nsteps
    denom = 1.0 + dt * 1j * q * v[:, None] * Ox[None, :] + dt * invtau[:, None]
    r = dt * invtau[:, None] * C[:, None] / denom
    mom_r = mom(r)
    A = [A0_K]; tt = [0.0]; A_signed = [A0_K]
    for n in range(nsteps):
        p = ghat / denom
        dT_new = mom(p) / (1.0 - mom_r)
        ghat = p + r * dT_new
        At = abs(dT_new); A.append(At); tt.append((n + 1) * dt)
        A_signed.append(dT_new.real)
        if window_rule == WINDOW_RULE_LEGACY and At < _LEGACY_STOP_REL * A[0]:
            break
    A = np.array(A); tt = np.array(tt)
    gamma = A[0] / np.trapezoid(A, tt)
    if return_diag:
        return gamma, dict(A=A, A_signed=np.array(A_signed), t=tt, dt_s=dt,
                           nsteps=nsteps, n_steps_used=len(A) - 1,
                           t_declared_s=nsteps * dt, t_realized_s=tt[-1],
                           n_decay=n_decay, gamma_dominant_hz=gd,
                           window_rule=window_rule)
    return gamma


def ttg_amplitude_curve(L_um, geometry="1d", Nk=20, T_ref=300.0, t_end_s=None,
                        n_decay=7, nsteps=12000, nmu_floor=None,
                        steps_per_decay=None, **angular_kw):
    v, tau, C = phonon_modes(Nk, T_ref)
    q = 2 * np.pi / (L_um * ANGSTROM_PER_UM)
    if t_end_s is None or steps_per_decay:  # lazy: dominant_gamma is an eig solve
        gd_c = dominant_gamma(L_um, "1d", Nk=Nk)[0]
        gd = gd_c.real if gd_c is not None else float(q * np.sum(C * v) / np.sum(C))
        if t_end_s is None:
            t_end_s = n_decay / gd
        if steps_per_decay:
            nsteps = int(round(t_end_s * gd * steps_per_decay))
    if geometry == "1d" and "Nmu" not in angular_kw:
        angular_kw["Nmu"] = max(32, int(np.ceil(q * v.max() * t_end_s / 2.0)) + 8)
    Ox, w = _angular(geometry, **_floor_nmu(geometry, angular_kw, nmu_floor))
    invtau = 1.0 / tau; Cinvtau = np.sum(C * invtau)

    def mom(x):
        return np.sum(invtau * (x @ w)) / Cinvtau

    ghat = (C[:, None] * 1.0).astype(complex) * np.ones((len(v), len(Ox)), dtype=complex)
    dt = t_end_s / nsteps
    denom = 1.0 + dt * 1j * q * v[:, None] * Ox[None, :] + dt * invtau[:, None]
    r = dt * invtau[:, None] * C[:, None] / denom
    mom_r = mom(r)
    A = [1.0]; tt = [0.0]
    for n in range(nsteps):
        p = ghat / denom
        dT_new = mom(p) / (1.0 - mom_r)
        ghat = p + r * dT_new
        A.append(dT_new.real)
        tt.append((n + 1) * dt)
    return np.array(tt), np.array(A)


def ttg_gamma_same_window(L_um, geometry="1d", Nk=20, T_ref=300.0, *, t_end_s,
                          estimator=GAMMA_ESTIMATOR_RULED,
                          denominator_rule=DENOMINATOR_RULE_RULED,
                          nsteps=12000, nmu_floor=None, return_diag=False,
                          **angular_kw):
    """The RULED denominator: the arbiter re-scored on the CALLER's window."""
    if denominator_rule not in DENOMINATOR_RULES:
        raise ValueError(
            f"unknown denominator rule {denominator_rule!r}; expected one of "
            f"{DENOMINATOR_RULES}")
    if denominator_rule != DENOMINATOR_RULE_SAME_WINDOW:
        raise ValueError(
            f"ttg_gamma_same_window implements {DENOMINATOR_RULE_SAME_WINDOW!r} "
            f"only; {denominator_rule!r} is a different denominator and must "
            f"come from the function that actually computes it (the legacy "
            f"arbiter-window rate is `ttg_decay_spectral`).")
    t, A = ttg_amplitude_curve(L_um, geometry, Nk=Nk, T_ref=T_ref,
                               t_end_s=t_end_s, nsteps=nsteps,
                               nmu_floor=nmu_floor, **angular_kw)
    gamma = integral_gamma(t, A, estimator=estimator)
    if return_diag:
        prov = denominator_provenance(
            estimator=estimator, rule=denominator_rule, t_end_s=t_end_s,
            n_samples=int(t.size), gamma_ref_hz=gamma,
            L_um=float(L_um), geometry=str(geometry), Nk=int(Nk))
        return gamma, dict(prov, t=t, A=A, n_samples=int(t.size))
    return gamma


def ttg_window_crossing(L_um, geometry="1d", Nk=20, thresh=0.01, T_ref=300.0,
                        seed_n_decay=None, return_diag=False, nmu_floor=None,
                        window_rule=WINDOW_RULE_FULL):
    if seed_n_decay is None:
        seed_n_decay = 7.0 if geometry == "1d" else 5.0
    gamma = ttg_decay_spectral(L_um, geometry, Nk=Nk, T_ref=T_ref,
                               nmu_floor=nmu_floor, window_rule=window_rule)
    t_seed = seed_n_decay / gamma
    factor = 3.0
    while True:
        nsteps = int(12000 * min(factor, 6))
        tt, A = ttg_amplitude_curve(L_um, geometry, Nk=Nk, T_ref=T_ref,
                                    t_end_s=factor * t_seed, nsteps=nsteps,
                                    nmu_floor=nmu_floor)
        absA = np.abs(A)
        idx = np.nonzero(absA >= thresh)[0][-1]
        tail_ok = idx < 0.8 * len(tt)
        if tail_ok or factor >= 48:
            if not tail_ok:
                import warnings
                warnings.warn(
                    f"ttg_window_crossing(L={L_um}um, {geometry}, Nk={Nk}): "
                    f"tail still >= {thresh:g}*A0 in the last 20% of the x48 "
                    f"window — returned crossing is a LOWER bound.")
            t_cross = float(tt[idx])
            if return_diag:
                return t_cross, dict(t=tt, A=A, tail_ok=tail_ok, factor=factor,
                                     gamma_spectral_hz=gamma, t_seed_s=t_seed)
            return t_cross
        factor *= 2.0


def suppression(L_um, geometry="1d", A0_K=1.0, Nk=20, method="spectral", **kw):
    """S = gamma_eff / gamma_Fourier_physical (physical, with the 1/3)."""
    v, tau, C = phonon_modes(Nk, 300.0)
    q = 2 * np.pi / (L_um * ANGSTROM_PER_UM)
    gF = (np.sum(C * v ** 2 * tau) / np.sum(C)) * q ** 2 / 3.0
    g = ttg_decay_spectral(L_um, geometry=geometry, A0_K=A0_K, Nk=Nk, **kw) \
        if method == "spectral" else ttg_decay(L_um, A0_K=A0_K, Nk=Nk, **kw)
    return g / gF


def self_test():
    """(1) diffusive S->1; (2) all S<=1; (3) 1d == 2d for a grating in x."""
    assert abs(suppression(100.0) - 1.0) < 0.02, "diffusive limit must give S~1"
    for L in (0.1, 1.0, 10.0, 100.0):
        assert suppression(L) <= 1.02, f"S(L={L}) must be physical (<=1)"
        s1, s2 = suppression(L, "1d"), suppression(L, "2d")
        assert abs(s1 - s2) / s1 < 0.02, f"1d vs 2d mismatch at L={L}: {s1:.4f} {s2:.4f}"
    return True


if __name__ == "__main__":
    print("self_test:", "PASS" if self_test() else "FAIL")
