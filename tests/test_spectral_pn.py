"""Spectral-PN core: exact-in-(x,mu) representation of the TTG BTE."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import (
    PNCoefficientNet,
    legendre_streaming_matrix,
    pn_amplitude,
    pn_residual,
    solve_pn_ode,
)

REPO = Path(__file__).parent.parent

_spec = importlib.util.spec_from_file_location(
    "vdsv", REPO / "scripts" / "validate_dom_spectral_volterra.py")
_vdsv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vdsv)


def test_streaming_matrix_matches_quadrature():
    L_max = 12
    T = legendre_streaming_matrix(L_max)
    mu, w = np.polynomial.legendre.leggauss(64)
    P = np.stack([np.polynomial.legendre.Legendre.basis(l)(mu)
                  for l in range(L_max + 1)])
    # <P_l' mu P_l> / <P_l' P_l'> via GL quadrature (angular MEAN convention)
    for lp in range(L_max + 1):
        norm = np.sum(w * P[lp] * P[lp])
        for l in range(L_max + 1):
            num = np.sum(w * P[lp] * mu * P[l])
            expected = num / norm
            assert abs(T[lp, l] - expected) < 1e-12, (lp, l)


@pytest.mark.parametrize("xi,L_max,tol", [(0.6, 12, 2e-3), (3.0, 24, 5e-3)])
def test_gray_pn_ode_matches_volterra(xi, L_max, tol):
    from pinn_bte.utils.plotting_transient import compute_analytical_amplitude
    v = np.array([1.0]); tau = np.array([1.0]); C = np.array([1.0])
    t = np.linspace(0.0, 8.0, 801)
    A = solve_pn_ode(v, tau, C, q=xi, t=t, L_max=L_max)
    A_ref = compute_analytical_amplitude(t, xi)
    assert np.abs(A - A_ref).max() < tol


def test_spectral_si_pn_ode_matches_volterra_arbiter():
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    v, tau, C = phonon_modes(20, 300.0)
    L_um = 1.0
    q = 2 * np.pi / (L_um * 1e4)
    # window ~ the trainer's: use the reference's own grid
    t_end = 97.0 * (np.sum(C) / np.sum(C / tau))   # ~Lt * tau_ref seconds
    tv, Av = _vdsv.volterra_trace(v, tau, C, q, t_end,
                                  _vdsv.auto_N(v, tau, q, t_end))
    t = np.linspace(0.0, t_end, 2001)
    A = solve_pn_ode(v, tau, C, q=q, t=t, L_max=8)
    assert np.abs(A - np.interp(t, tv, Av)).max() < 5e-3


def test_parity_structure():
    v = np.array([1.0]); tau = np.array([1.0]); C = np.array([1.0])
    t = np.linspace(0.0, 4.0, 401)
    _, y = solve_pn_ode(v, tau, C, q=1.5, t=t, L_max=8, return_state=True)
    c, s = y  # (Nt, M, L+1) each
    even = np.arange(0, 9, 2); odd = np.arange(1, 9, 2)
    assert np.abs(c[..., odd]).max() < 1e-10 * max(np.abs(c).max(), 1)
    assert np.abs(s[..., even]).max() < 1e-10 * max(np.abs(s).max(), 1)


def test_training_smoke_gray_no_waypoints():
    """A tiny CPU fit of the gray system: residual falls and A(t) moves toward the
    exact curve with NO supervision beyond the equations.
    """
    from pinn_bte.utils.plotting_transient import compute_analytical_amplitude
    torch.manual_seed(0)
    v = np.array([1.0]); tau = np.array([1.0]); C = np.array([1.0])
    xi, t_end, L_max = 0.6, 6.0, 8
    net = PNCoefficientNet(n_modes=1, L_max=L_max, width=48, depth=3)
    v_t = torch.tensor(v, dtype=torch.float64)
    tau_t = torch.tensor(tau, dtype=torch.float64)
    C_t = torch.tensor(C, dtype=torch.float64)
    net = net.double()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t_coll = torch.linspace(0.0, t_end, 241, dtype=torch.float64)
    loss0 = None
    for it in range(600):
        opt.zero_grad()
        res = pn_residual(net, t_coll, v_t, tau_t, C_t, q=xi)
        loss = (res ** 2).mean()
        loss.backward()
        opt.step()
        if loss0 is None:
            loss0 = float(loss)
    assert float(loss) < 0.02 * loss0, "residual did not drop"
    with torch.no_grad():
        A = pn_amplitude(net, t_coll, tau_t, C_t, v=v_t, q=xi).numpy()
    A_ref = compute_analytical_amplitude(t_coll.numpy(), xi)
    err = np.abs(A - A_ref).max()
    assert err < 0.08, f"amplitude off by {err:.3f} after smoke training"
