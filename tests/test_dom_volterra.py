"""The spectral DOM reference against the angle-exact Volterra solve and the
exact gray limit; 1d == 2d for a grating in x.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pinn_bte.physics import ttg_dom
from pinn_bte.physics.ttg_dispersion import phonon_modes

_spec = importlib.util.spec_from_file_location(
    "validate_dom_spectral_volterra",
    Path(__file__).parent.parent / "scripts"
    / "validate_dom_spectral_volterra.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
volterra_trace, auto_N = _mod.volterra_trace, _mod.auto_N


def test_gray_limit_matches_exact_reference():
    """Single gray mode: Volterra arbiter == compute_analytical_amplitude."""
    from pinn_bte.utils.plotting_transient import compute_analytical_amplitude
    one = np.array([1.0])
    for xi in (0.6, 3.0):
        t, A = volterra_trace(one, one, one, xi, 8.0, 1500)
        assert np.abs(A - compute_analytical_amplitude(t, xi)).max() < 1e-9


def test_dom_matches_volterra_at_ballistic_L():
    """DOM trace tracks the angle-exact arbiter in the ballistic regime."""
    Nk, L = 6, 0.01
    v, tau, C = phonon_modes(Nk, 300.0)
    q = 2 * np.pi / (L * 1e4)
    td, Ad = ttg_dom.ttg_amplitude_curve(L, "1d", Nk=Nk)
    tv, Av = volterra_trace(v, tau, C, q, td[-1], auto_N(v, tau, q, td[-1]))
    assert np.abs(np.interp(tv, td, Ad) - Av).max() < 2e-3


def test_1d_equals_2d_for_grating_in_x():
    """Full-sphere Omega_x marginal is uniform on [-1,1] => same trace as 1d."""
    Nk, L = 6, 0.1
    t1, A1 = ttg_dom.ttg_amplitude_curve(L, "1d", Nk=Nk)
    t2, A2 = ttg_dom.ttg_amplitude_curve(L, "2d", Nk=Nk)
    assert np.abs(np.interp(t1, t2, A2) - A1).max() < 2e-3
