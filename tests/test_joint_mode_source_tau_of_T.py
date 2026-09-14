"""joint_mode_source must not silently freeze tau(T)."""

import numpy as np
import pytest

from pinn_bte.physics import ttg_dom
from pinn_bte.physics.optical_reservoir import joint_mode_source, joint_modes
from pinn_bte.physics import ttg_dispersion as _disp

NK = 10


def test_same_T_call_is_bit_identical_to_the_prebuilt_comb():
    """The no-regression guarantee: consumers that re-call with the context's own
    T_ref must get exactly what they got before the fix.
    """
    pre = joint_modes(Nk=NK, T_ref=300.0)
    with joint_mode_source(Nk=NK, T_ref=300.0):
        got = _disp.phonon_modes(NK, 300.0)
    for a, b in zip(pre, got):
        assert np.array_equal(a, b)


def test_a_different_T_actually_changes_tau():
    with joint_mode_source(Nk=NK, T_ref=300.0):
        _, t200, _ = _disp.phonon_modes(NK, 200.0)
        _, t400, _ = _disp.phonon_modes(NK, 400.0)
    r = t200 / t400
    assert np.max(r) > 1.2, r
    assert int(np.sum(np.abs(r - 1.0) < 1e-12)) == 0, "a T-flat mode is back"


def test_the_tau_of_T_table_is_no_longer_flat():
    with joint_mode_source(Nk=NK, T_ref=300.0):
        _, tauT, _, _, _ = ttg_dom._tables(NK, 300.0, T_lo=140.0, T_hi=460.0, nT=8)
    r = tauT[0] / tauT[-1]
    assert np.max(r) > 1.2, r


def test_the_nonlinear_arbiter_is_nonlinear_again():
    kw = dict(Nk=NK, Nmu=8, Nx=24, n_decay=3)
    with joint_mode_source(Nk=NK, T_ref=300.0):
        small = float(ttg_dom.ttg_decay(1.0, A0_K=1.0, **kw))
        large = float(ttg_dom.ttg_decay(1.0, A0_K=80.0, **kw))
    assert abs(large / small - 1.0) > 1e-5, (small, large)


def test_restores_on_exit():
    before = _disp.phonon_modes
    with joint_mode_source(Nk=NK, T_ref=300.0):
        pass
    assert _disp.phonon_modes is before
