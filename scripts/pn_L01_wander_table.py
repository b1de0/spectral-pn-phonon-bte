"""Per-period diagnosis table for the quasi-ballistic wander."""
from __future__ import annotations
import contextlib, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pinn_bte.physics.ttg_dispersion as _disp
import pinn_bte.physics.ttg_dom as _dom
from pinn_bte.physics.ttg_dispersion import phonon_modes
from pinn_bte.physics.optical_reservoir import joint_modes
from pinn_bte.physics.ttg_dom import _angular, _floor_nmu
from pinn_bte.models.spectral_pn import ugks_resolvent_kernels

ANG_PER_UM = 1e4
LADDER = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)
YMAX = 50.0                      # spectral_pn.XiHybridPNCoefficientNet._BLEND_YMAX
N_DECAY = 7.0                    # the PN window rule
MEASURED = {
    'SHIPPED': dict(zip(LADDER, (0.0421, 0.0050, 0.0042, 0.0052, 0.0036,
                                 0.0070, 0.0287, 0.0019, 0.0000))),
    'JOINT':   dict(zip(LADDER, (0.0024, 0.0295, 0.0048, 0.0060, 0.0007,
                                 0.0022, 0.0147, 0.0002, 0.0000))),
}


@contextlib.contextmanager
def mode_source(modes):
    """Route BOTH phonon_modes namespaces to a fixed mode set (fullbz_modes
    contract).
    """
    saved = (_disp.phonon_modes, _dom.phonon_modes)
    _disp.phonon_modes = _dom.phonon_modes = lambda *a, **k: modes
    try:
        yield
    finally:
        _disp.phonon_modes, _dom.phonon_modes = saved


def signed_trace(L_um, v, tau, C, t_end, nsteps, geometry='1d'):
    """ttg_dom.ttg_decay_spectral's stepper, returning the SIGNED amplitude."""
    q = 2 * np.pi / (L_um * ANG_PER_UM)
    Ox, w = _angular(geometry, **_floor_nmu(geometry, {}, None))
    invtau = 1.0 / tau
    Cinvtau = np.sum(C * invtau)
    mom = lambda x: np.sum(invtau * (x @ w)) / Cinvtau
    ghat = (C[:, None] * 1.0).astype(complex) * np.ones((len(v), len(Ox)), complex)
    dt = t_end / nsteps
    den = 1.0 + dt * 1j * q * v[:, None] * Ox[None, :] + dt * invtau[:, None]
    r = dt * invtau[:, None] * C[:, None] / den
    mom_r = mom(r)
    A, t = [1.0], [0.0]
    for n in range(nsteps):
        p = ghat / den
        dT = mom(p) / (1.0 - mom_r)
        ghat = p + r * dT
        A.append(dT.real); t.append((n + 1) * dt)
    return np.array(t), np.array(A)


def rung(modes, L_um):
    v, tau, C = modes
    with mode_source(modes):
        root, ok = _disp.dominant_gamma(L_um, '1d', Nk=20)[:2]
    q = 2 * np.pi / (L_um * ANG_PER_UM)
    gd = root.real if root is not None else float(q * np.sum(C * v) / np.sum(C))
    t_end = N_DECAY / gd
    for _ in range(3):                                   # self-consistent window
        t, A = signed_trace(L_um, v, tau, C, t_end, 12000)
        g = A[0] / np.trapezoid(np.abs(A), t)
        t_end = N_DECAY / g
    t, A = signed_trace(L_um, v, tau, C, t_end, 12000)
    g = A[0] / np.trapezoid(np.abs(A), t)
    tl, Al = signed_trace(L_um, v, tau, C, 6 * t_end, 48000)
    sgn = np.sign(Al); iz = np.where(sgn[1:] != sgn[0])[0]
    t_zero = tl[iz[0] + 1] / t_end if len(iz) else np.inf
    g_loc = -np.gradient(np.log(np.abs(A) + 1e-300), t)  # EXACT local decay rate
    xi = q * v * tau
    y_bounded = YMAX * np.tanh(np.outer(g_loc, tau) / YMAX)
    K0 = ugks_resolvent_kernels(torch.as_tensor(y_bounded),
                                torch.as_tensor(xi))[0].numpy()
    wC = C / C.sum()
    tail = slice(len(t) // 10, None)                     # skip the initial transient
    return dict(g=g, t_end=t_end, t_zero=t_zero,
                y_bulk=g * tau.max(),                    # kernel lever arm
                y_edge=abs(g_loc[-2]) * tau.max(),
                K0bar=float(np.mean(K0[tail] @ wC)),     # slaved share of the closure
                gloc_edge=g_loc[-2] / g,
                root_over_continuum=gd * tau.max(), ok=bool(ok),
                xi_med=float(xi[np.argsort(xi)][np.searchsorted(
                    np.cumsum(wC[np.argsort(xi)]), 0.5)]))


def main():
    print(f"{'comb':>8} {'L[um]':>7} {'gamma_eff':>11} {'root/cont':>9} {'ok':>5} "
          f"{'tz/te':>7} {'gloc(e)/g':>9} {'xi_med':>8} {'y_bulk':>8} {'y_edge':>8} "
          f"{'K0bar':>8} {'PRODUCT':>8} {'|1-gr|':>7}")
    for name, modes in (('SHIPPED', phonon_modes(20, measure=False)),
                        ('JOINT', joint_modes(20))):
        for L in LADDER:
            d = rung(modes, L)
            print(f"{name:>8} {L:7.2f} {d['g']:11.4e} {d['root_over_continuum']:9.2f} "
                  f"{str(d['ok']):>5} {d['t_zero']:7.3f} {d['gloc_edge']:9.3f} "
                  f"{d['xi_med']:8.3f} {d['y_bulk']:8.2f} {d['y_edge']:8.1f} "
                  f"{d['K0bar']:8.4f} {d['y_bulk']*max(d['K0bar'],0.0):8.2f} "
                  f"{MEASURED[name][L]:7.4f}")


if __name__ == '__main__':
    main()
