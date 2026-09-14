"""The lumped optical reservoir must carry tau(T), anchored at 300 K."""

import numpy as np
import pytest

from pinn_bte.physics.optical_reservoir import (
    TAU_OPT_S_DEFAULT, joint_modes, optical_reservoir_modes, tau_opt_of_T,
)


def test_anchored_exactly_at_300K():
    """No existing 300 K number may move."""
    assert tau_opt_of_T(300.0) == pytest.approx(TAU_OPT_S_DEFAULT, rel=1e-12)


def test_the_whole_comb_is_bit_identical_at_300K():
    v, t, c = joint_modes(Nk=20, T_ref=300.0)
    assert t[-1] == pytest.approx(TAU_OPT_S_DEFAULT, rel=1e-12)


def test_tau_falls_with_temperature_and_by_a_physical_amount():
    """Anharmonic decay: more phonons to decay into, shorter lifetime."""
    hot, cold = tau_opt_of_T(400.0), tau_opt_of_T(200.0)
    assert hot < TAU_OPT_S_DEFAULT < cold
    # Klemens over 200->400 K is a factor of roughly 1.5-2.5; anything outside
    # that is a parameterisation error, not physics.
    assert 1.5 < cold / hot < 2.5, cold / hot


def test_the_reservoir_mode_now_varies_with_T():
    _, t300, _ = optical_reservoir_modes(T_ref=300.0)
    _, t400, _ = optical_reservoir_modes(T_ref=400.0)
    assert t400[0] < t300[0]


def test_no_flat_mode_remains_in_the_production_comb():
    _, t3, _ = joint_modes(Nk=20, T_ref=300.0)
    _, t4, _ = joint_modes(Nk=20, T_ref=400.0)
    r = t3 / t4
    assert int(np.sum(np.abs(r - 1.0) < 1e-12)) == 0, "a T-flat mode survives"
