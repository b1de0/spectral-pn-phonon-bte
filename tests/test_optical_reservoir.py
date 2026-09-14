"""The optical capacity the acoustic Holland mode set does not carry."""
from __future__ import annotations

import numpy as np
import pytest

from pinn_bte.physics.optical_reservoir import (
    SI_RHO_CP_300K,
    einstein_capacity,
    joint_mode_source,
    joint_modes,
    optical_reservoir_modes,
)
from pinn_bte.physics.ttg_dispersion import phonon_modes
from tests.test_mode_source_capacity_contract import (
    SI_KAPPA_BAND,
    comb_diffusivity_m2s,
    comb_kappa_si,
)

# Full-BZ acoustic reference for this mode set (measure-corrected, converged Nk).
FULLBZ_GRID = dict(k_lo=1e-3, k_hi=1.0)         # joint_modes: measure is implied
FULLBZ_KW = dict(measure=True, **FULLBZ_GRID)   # phonon_modes: measure is opt-in


def test_einstein_capacity_per_mode_is_the_textbook_expression():
    """c(w,T) = kB x^2 e^x/(e^x-1)^2 -> kB in the classical limit x -> 0."""
    kB = 1.380649e-23
    assert einstein_capacity(1e9, 300.0) == pytest.approx(kB, rel=1e-6)
    # Si optical at 300 K is partially frozen out: x ~ 2.2-2.5 -> 0.60-0.69 kB
    c = einstein_capacity(2 * np.pi * 14.0e12, 300.0)
    assert 0.60 < c / kB < 0.70


def test_optical_capacity_is_derived_not_fitted():
    """3 branches x 4/a^3 primitive cells x Einstein occupancy at w_opt."""
    _v, _tau, C = optical_reservoir_modes(T_ref=300.0)
    Copt = float(np.sum(C))
    assert 6.0e5 < Copt < 7.5e5, f"C_opt = {Copt:.3e} J/(m^3 K)"


def test_total_capacity_lands_near_rho_cp_as_an_EMERGENT_check():
    """sum C_acoustic + sum C_optical ~ rho c_p, without being fit to it."""
    _v, _t, C_ac = phonon_modes(Nk=800, T_ref=300.0, **FULLBZ_KW)
    _v, _t, C_op = optical_reservoir_modes(T_ref=300.0)
    total = float(np.sum(C_ac) + np.sum(C_op))
    assert abs(total - SI_RHO_CP_300K) / SI_RHO_CP_300K < 0.05, (
        f"sum C = {total:.4e} vs rho c_p = {SI_RHO_CP_300K:.4e}")


def test_joint_comb_kappa_stays_in_the_experimental_band():
    v, tau, C = joint_modes(Nk=800, T_ref=300.0, **FULLBZ_GRID)
    kap = comb_kappa_si(v, tau, C)
    assert SI_KAPPA_BAND[0] <= kap <= SI_KAPPA_BAND[1]


def test_optical_branches_carry_a_percent_or_two_of_kappa():
    """Slow + short-lived: the reservoir is a CAPACITY correction, not a kappa
    one.
    """
    va, ta, Ca = phonon_modes(Nk=800, T_ref=300.0, **FULLBZ_KW)
    vo, to, Co = optical_reservoir_modes(T_ref=300.0)
    frac = comb_kappa_si(vo, to, Co) / comb_kappa_si(
        np.r_[va, vo], np.r_[ta, to], np.r_[Ca, Co])
    assert 0.0 < frac < 0.03, f"optical kappa fraction {frac:.4f}"


def test_joint_comb_diffusivity_matches_silicon():
    """D_Si = kappa/(rho c_p) ~ 9.1e-5 m^2/s at 300 K."""
    v, tau, C = joint_modes(Nk=800, T_ref=300.0, **FULLBZ_GRID)
    D = comb_diffusivity_m2s(v, tau, C)
    D_si = 148.0 / SI_RHO_CP_300K
    assert abs(D - D_si) / D_si < 0.08, f"D = {D:.4e} vs {D_si:.4e}"


def test_measure_alone_overshoots_and_is_rescued_by_the_reservoir():
    """The load-bearing reason the two fixes ship together, as a number."""
    D_si = 148.0 / SI_RHO_CP_300K
    D_measure_only = comb_diffusivity_m2s(*phonon_modes(Nk=800, T_ref=300.0,
                                                        **FULLBZ_KW))
    D_joint = comb_diffusivity_m2s(*joint_modes(Nk=800, T_ref=300.0, **FULLBZ_GRID))
    assert D_measure_only / D_si > 1.5          # measure alone: ~1.7x too fast
    assert abs(D_joint / D_si - 1.0) < 0.08     # joint: on silicon


@pytest.mark.parametrize("v_ms", [200.0, 600.0, 1200.0, 2000.0])
@pytest.mark.parametrize("tau_s", [2.0e-12, 3.7e-12, 5.0e-12])
def test_D_is_insensitive_to_the_optical_transport_parameters(v_ms, tau_s):
    base = comb_diffusivity_m2s(*joint_modes(Nk=800, T_ref=300.0, **FULLBZ_GRID))
    alt = comb_diffusivity_m2s(*joint_modes(Nk=800, T_ref=300.0,
                                            v_opt_ms=v_ms, tau_opt_s=tau_s,
                                            **FULLBZ_GRID))
    assert abs(alt - base) / base < 0.03


@pytest.mark.parametrize("f_THz", [13.0, 14.0, 15.6])
def test_D_sensitivity_to_the_optical_frequency_is_bounded(f_THz):
    """w_opt sets C_opt (first order in D) -- the one parameter that matters."""
    base = comb_diffusivity_m2s(*joint_modes(Nk=800, T_ref=300.0, **FULLBZ_GRID))
    alt = comb_diffusivity_m2s(*joint_modes(Nk=800, T_ref=300.0,
                                            omega_opt_THz=f_THz, **FULLBZ_GRID))
    assert abs(alt - base) / base < 0.06


def test_injection_redirects_both_namespaces_and_restores_byte_identical():
    import pinn_bte.physics.ttg_dispersion as disp
    import pinn_bte.physics.ttg_dom as dom

    before = disp.phonon_modes(20, 300.0)
    saved = (disp.phonon_modes, dom.phonon_modes)
    with joint_mode_source(Nk=20) as (v, tau, C):
        assert np.array_equal(disp.phonon_modes(20, 300.0)[0], v)
        assert np.array_equal(dom.phonon_modes(20, 300.0)[0], v)
        assert len(v) == 3 * 20 + 1  # 60 acoustic + 1 lumped optical
    assert disp.phonon_modes is saved[0] and dom.phonon_modes is saved[1]
    for x, y in zip(before, disp.phonon_modes(20, 300.0)):
        assert np.array_equal(x, y)


def test_injected_source_ignores_caller_Nk_drift():
    """Consumers re-call phonon_modes(Nk, T) internally; the mode set must be
    fixed.
    """
    import pinn_bte.physics.ttg_dom as dom

    with joint_mode_source(Nk=20) as (v, _t, _C):
        assert np.array_equal(dom.phonon_modes(7, 300.0)[0], v)
        assert np.array_equal(dom.phonon_modes(20, 300.0)[0], v)
