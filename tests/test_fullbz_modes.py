"""Acceptance tests for the phono3py-derived isotropized full-BZ Si mode set."""
from __future__ import annotations

import numpy as np
import pytest

from pinn_bte.physics.fullbz_modes import (
    SI_C_VOLUMETRIC_300K,
    SI_KAPPA_BULK_300K,
    build_fullbz_modes,
    fullbz_mode_source,
    mfp_fraction_above,
    reconstruct_kappa,
    total_heat_capacity,
)

ANG_PER_S_TO_M_PER_S = 1e-10  # v is returned in Angstrom/s


def test_returns_three_equal_length_positive_arrays():
    v, tau, C = build_fullbz_modes(n_bins=200)
    assert v.shape == tau.shape == C.shape
    assert v.ndim == 1 and len(v) > 0
    for a in (v, tau, C):
        assert np.all(np.isfinite(a))
        assert np.all(a > 0)


def test_velocity_units_are_angstrom_per_second():
    # Si acoustic group velocities are ~1.5-8.5 km/s; returned in Ang/s => 1e13-1e14.
    v, tau, C = build_fullbz_modes(n_bins=200)
    v_ms = v * ANG_PER_S_TO_M_PER_S
    assert v_ms.min() > 1000.0
    assert v_ms.max() < 9000.0


def test_mfp_range_spans_the_tail():
    # Lambda = v * tau must span from < 30 nm to > 10 um (the long tail is the point).
    v, tau, C = build_fullbz_modes(n_bins=200)
    lam_um = v * tau * ANG_PER_S_TO_M_PER_S * 1e6  # (Ang/s * s) -> Ang... see helper
    # v[Ang/s]*tau[s] = Angstrom; Angstrom * 1e-4 = um
    lam_um = v * tau * 1e-4
    assert lam_um.min() < 0.03
    assert lam_um.max() > 10.0


def test_reconstructed_kappa_matches_bulk():
    v, tau, C = build_fullbz_modes(n_bins=200)
    kappa = reconstruct_kappa(v, tau, C)
    assert 140.0 <= kappa <= 155.0, f"kappa={kappa:.1f} outside 140-155 W/mK"
    assert abs(kappa - SI_KAPPA_BULK_300K) / SI_KAPPA_BULK_300K < 0.10


def test_half_of_kappa_from_mfp_above_1um():
    v, tau, C = build_fullbz_modes(n_bins=200)
    frac = mfp_fraction_above(v, tau, C, lam_um=1.0)
    assert 0.40 <= frac <= 0.60, f"frac(MFP>1um)={frac:.3f} outside 0.40-0.60"


def test_holland_comb_has_almost_no_tail_by_contrast():
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    v, tau, C = phonon_modes(Nk=20, T_ref=300.0, measure=False)
    frac = mfp_fraction_above(v, tau, C, lam_um=1.0)
    assert frac < 0.05


def test_total_heat_capacity_matches_si_300K():
    v, tau, C = build_fullbz_modes(n_bins=200)
    Ctot = total_heat_capacity(C)
    # Emergent from the physical v(Lambda) split (NOT fit to this target).
    assert abs(Ctot - SI_C_VOLUMETRIC_300K) / SI_C_VOLUMETRIC_300K < 0.15


def test_bin_count_convergence():
    kap, Ctot, frac = {}, {}, {}
    for nb in (100, 200, 400):
        v, tau, C = build_fullbz_modes(n_bins=nb)
        kap[nb] = reconstruct_kappa(v, tau, C)
        Ctot[nb] = total_heat_capacity(C)
        frac[nb] = mfp_fraction_above(v, tau, C, lam_um=1.0)
    assert abs(kap[400] - kap[100]) / kap[100] < 0.02
    assert abs(Ctot[400] - Ctot[100]) / Ctot[100] < 0.02
    assert abs(frac[400] - frac[100]) < 0.02


def test_injection_redirects_and_restores_byte_identical():
    import pinn_bte.physics.ttg_dispersion as disp
    import pinn_bte.physics.ttg_dom as dom

    v_hol0, t_hol0, c_hol0 = disp.phonon_modes(20, 300.0)
    saved_disp, saved_dom = disp.phonon_modes, dom.phonon_modes
    with fullbz_mode_source(n_bins=120) as (v, tau, C):
        # inside: both namespaces yield the full-BZ mode set
        assert np.array_equal(disp.phonon_modes(20, 300.0)[0], v)
        assert np.array_equal(dom.phonon_modes(20, 300.0)[0], v)
    # after: originals restored (same object) -> byte-identical behaviour
    assert disp.phonon_modes is saved_disp
    assert dom.phonon_modes is saved_dom
    v_hol1, t_hol1, c_hol1 = disp.phonon_modes(20, 300.0)
    assert np.array_equal(v_hol0, v_hol1)
    assert np.array_equal(t_hol0, t_hol1)
    assert np.array_equal(c_hol0, c_hol1)


def test_injection_makes_solver_more_suppressed_at_110nm():
    # The load-bearing stage-(iii) result: the long-MFP tail suppresses S(110 nm)
    # below the Holland value (toward the measured 0.053), solver-only.
    from pinn_bte.physics.ttg_dom import suppression

    s_holland = suppression(0.11, "1d", Nk=20, method="spectral", nsteps=4000)
    with fullbz_mode_source(n_bins=120):
        s_fullbz = suppression(0.11, "1d", Nk=120, method="spectral", nsteps=4000)
    assert s_fullbz < s_holland
    assert 0.04 < s_fullbz < 0.11  # first-principles band-ish, below Holland ~0.114
