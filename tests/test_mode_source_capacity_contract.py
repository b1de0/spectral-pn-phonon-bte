"""Capacity contract of every mode set: sum C is a heat capacity and
kappa = (1/3) sum C v^2 tau lands in silicon's measured band; the
mode-source managers route every namespace.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinn_bte.physics.mesh_1d import (
    compute_phonon_properties,
    phonon_capacity_si,
    phonon_kappa_si,
)
from pinn_bte.physics.ttg_dispersion import phonon_modes
from pinn_bte.config.physics import LATTICE_CONSTANT

ANG_PER_M = 1e10
SI_KAPPA_BAND = (142.0, 156.0)  # experimental bulk Si kappa at 300 K, W/(m K)


def comb_diffusivity_m2s(v_ang, tau, C) -> float:
    """D = (1/3) <v^2 tau>_C in m^2/s -- the q->0 limit of the DOM operator."""
    v_ms = np.asarray(v_ang) / ANG_PER_M
    return float(np.sum(C * v_ms ** 2 * tau) / np.sum(C) / 3.0)


def comb_kappa_si(v_ang, tau, C) -> float:
    """kappa = (1/3) sum C_bin v^2 tau in W/(m K) -- absolute, needs C per-bin."""
    v_ms = np.asarray(v_ang) / ANG_PER_M
    return float(np.sum(C * v_ms ** 2 * tau) / 3.0)


def independent_si_diffusivity(Nk: int, T_ref: float = 300.0) -> tuple[float, float, float]:
    """(kappa, C, kappa/C) on the SAME k-grid via mesh_1d's SI path."""
    k_norm = np.linspace(0.05, 0.95, Nk).reshape(-1, 1)
    k_phys = k_norm * (2 * np.pi / LATTICE_CONSTANT)
    branch = np.vstack([np.zeros_like(k_norm), np.zeros_like(k_norm),
                        np.ones_like(k_norm)])
    omega, v, D, tau, dfdT = compute_phonon_properties(
        np.tile(k_phys, (3, 1)), branch, T_ref)
    wk = float(k_phys[1, 0] - k_phys[0, 0])
    kappa = phonon_kappa_si(omega, v, D, dfdT, tau, wk)
    cap = phonon_capacity_si(omega, v, D, dfdT, wk)
    return kappa, cap, kappa / cap


def assert_capacity_contract(v_ang, tau, C, Nk: int, T_ref: float = 300.0,
                             rtol: float = 1e-12) -> float:
    """The full contract for one mode set; returns the measured relative error."""
    d_comb = comb_diffusivity_m2s(v_ang, tau, C)
    _, _, d_si = independent_si_diffusivity(Nk, T_ref)
    rel = abs(d_comb - d_si) / d_si
    assert rel < rtol, (
        f"mode-source capacity contract VIOLATED at Nk={Nk}: "
        f"(1/3)<v^2 tau>_C = {d_comb:.6e} m^2/s but the same k-grid's "
        f"kappa_SI/C_SI = {d_si:.6e} m^2/s -- ratio {d_si / d_comb:.4f}. "
        f"A ratio of ~2 with no Nk convergence is the missing k-space "
        f"measure dw = v dk (C is per-unit-omega, the grid is uniform in k)."
    )
    return rel


@pytest.mark.parametrize("Nk", [10, 20, 40, 200])
def test_measure_corrected_holland_satisfies_capacity_contract(Nk):
    v, tau, C = phonon_modes(Nk=Nk, T_ref=300.0, measure=True)
    assert_capacity_contract(v, tau, C, Nk)


def test_fullbz_comb_satisfies_the_same_contract_shape():
    """Sanity that the contract is satisfiable: the repo's other mode set obeys
    it.
    """
    from pinn_bte.physics.fullbz_modes import build_fullbz_modes

    v, tau, C = build_fullbz_modes(n_bins=200)
    d = comb_diffusivity_m2s(v, tau, C)
    kap = comb_kappa_si(v, tau, C)
    assert abs(d - kap / np.sum(C)) / d < 1e-12
    assert SI_KAPPA_BAND[0] - 2.0 <= kap <= SI_KAPPA_BAND[1]


@pytest.mark.parametrize("Nk", [20, 200, 800])
def test_measure_corrected_kappa_lands_in_experimental_band(Nk):
    """kappa = (1/3) sum C_bin v^2 tau over the FULL BZ, at converged Nk."""
    v, tau, C = phonon_modes(Nk=Nk, T_ref=300.0, measure=True,
                             k_lo=1e-3, k_hi=1.0)
    kap = comb_kappa_si(v, tau, C)
    assert SI_KAPPA_BAND[0] <= kap <= SI_KAPPA_BAND[1], (
        f"Nk={Nk}: kappa={kap:.2f} W/(m K) outside {SI_KAPPA_BAND}")


def test_production_grid_truncation_cost_is_disclosed():
    """The k in [0.05, 0.95] production grid sits just under the band at high Nk."""
    kap20 = comb_kappa_si(*phonon_modes(Nk=20, T_ref=300.0, measure=True))
    kap800 = comb_kappa_si(*phonon_modes(Nk=800, T_ref=300.0, measure=True))
    assert 142.0 <= kap20 <= 145.0
    assert 139.5 <= kap800 <= 141.5
    full = comb_kappa_si(*phonon_modes(Nk=800, T_ref=300.0, measure=True,
                                       k_lo=1e-3, k_hi=1.0))
    assert full - kap800 > 4.0  # the truncated slivers are worth >4 W/(m K)


def test_shipped_default_violates_the_contract_by_a_factor_that_does_not_converge():
    """PINNED so it stays legible."""
    ratios = {}
    for Nk in (20, 200, 800):
        v, tau, C = phonon_modes(Nk=Nk, T_ref=300.0, measure=False)
        _, _, d_si = independent_si_diffusivity(Nk)
        ratios[Nk] = d_si / comb_diffusivity_m2s(v, tau, C)
    assert ratios[20] == pytest.approx(2.1223, abs=1e-3)
    assert ratios[800] == pytest.approx(1.7650, abs=1e-3)
    # not a discretisation artefact: it converges to ~1.77, it does not vanish
    assert ratios[800] > 1.7


def test_production_default_mode_source_satisfies_capacity_contract():
    """A plain assertion, not a ratchet."""
    # Bare ON PURPOSE -- the DEFAULT is what is being contracted.
    v, tau, C = phonon_modes(Nk=20, T_ref=300.0)
    assert_capacity_contract(v, tau, C, 20)


@pytest.mark.parametrize("Nk", [8, 20, 120])
def test_default_call_is_byte_identical_to_measure_on(Nk):
    a = phonon_modes(Nk=Nk, T_ref=300.0)
    b = phonon_modes(Nk=Nk, T_ref=300.0, measure=True)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("Nk", [8, 20, 120])
def test_legacy_weight_is_still_reachable_and_still_differs(Nk):
    a = phonon_modes(Nk=Nk, T_ref=300.0)
    b = phonon_modes(Nk=Nk, T_ref=300.0, measure=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[2], b[2])


def test_the_namespace_list_covers_every_module_level_binding():
    """The override list is only as good as its completeness."""
    import ast
    import pathlib

    from pinn_bte.physics.ttg_dispersion import _MODE_SOURCE_NAMESPACES

    root = pathlib.Path(__file__).resolve().parents[1] / "pinn_bte"
    found = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.col_offset == 0
                    and any(a.name == "phonon_modes" for a in node.names)):
                found.add(
                    str(path.relative_to(root.parent).with_suffix("")
                        ).replace("/", "."))
    # `optical_reservoir` builds the joint mode set by asking for measure=True
    # explicitly, so it must NOT be routed; ttg_dispersion defines the function.
    expected = found - {"pinn_bte.physics.optical_reservoir",
                        "pinn_bte.physics.ttg_dispersion"}
    assert set(_MODE_SOURCE_NAMESPACES) == expected, (
        f"mode-source override list is stale: missing {expected - set(_MODE_SOURCE_NAMESPACES)}, "
        f"extra {set(_MODE_SOURCE_NAMESPACES) - expected}")


def test_shipped_mode_source_routes_and_restores_every_namespace():
    """`shipped_mode_source` is the mirror of `joint_mode_source`."""
    import importlib

    import pinn_bte.physics.ttg_dispersion as disp
    from pinn_bte.physics.ttg_dispersion import (_MODE_SOURCE_NAMESPACES,
                                                 shipped_mode_source)

    mods = [importlib.import_module(n) for n in _MODE_SOURCE_NAMESPACES]
    saved = [disp.phonon_modes] + [m.phonon_modes for m in mods]
    legacy = phonon_modes(Nk=20, T_ref=300.0, measure=False)
    with shipped_mode_source():
        for m in [disp] + mods:
            for x, y in zip(m.phonon_modes(20, 300.0), legacy):
                assert np.array_equal(x, y), m.__name__
    assert [disp.phonon_modes] + [m.phonon_modes for m in mods] == saved


def test_shipped_mode_source_forwards_temperature():
    """It must change the WEIGHT only, never freeze (Nk, T)."""
    import pinn_bte.physics.ttg_dom as dom
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    with shipped_mode_source():
        c_lo = dom.phonon_modes(12, 200.0)[2]
        c_hi = dom.phonon_modes(12, 400.0)[2]
    assert not np.array_equal(c_lo, c_hi)
    assert np.array_equal(c_lo, phonon_modes(12, 200.0, measure=False)[2])


def test_a_leaked_patch_would_be_caught(monkeypatch):
    import importlib
    import sys

    import pinn_bte.physics.ttg_dom as dom
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    before = dom.phonon_modes
    probe = "pinn_bte.physics.ttg_pn2q"
    original = sys.modules.get(probe)
    try:
        with shipped_mode_source():
            sys.modules.pop(probe, None)
            importlib.import_module(probe)
        assert dom.phonon_modes is before
    finally:
        # the re-imported copy bound the patched name; put the original back
        if original is not None:
            sys.modules[probe] = original
    corrected = phonon_modes(Nk=20, T_ref=300.0)
    assert not np.array_equal(corrected[2],
                              phonon_modes(Nk=20, T_ref=300.0, measure=False)[2])


def test_default_grid_is_unchanged_by_the_new_knobs():
    """k_lo/k_hi default to the pinned production grid (0.05, 0.95)."""
    a = phonon_modes(Nk=37, T_ref=317.0)
    b = phonon_modes(Nk=37, T_ref=317.0, k_lo=0.05, k_hi=0.95)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_measure_only_rescales_C_and_leaves_v_tau_untouched():
    v0, t0, c0 = phonon_modes(Nk=20, T_ref=300.0)
    v1, t1, c1 = phonon_modes(Nk=20, T_ref=300.0, measure=True)
    assert np.array_equal(v0, v1)
    assert np.array_equal(t0, t1)
    # C -> C * v * wk * (code->SI); v is the only per-mode factor
    ratio = c1 / c0
    assert np.allclose(ratio / v0, ratio[0] / v0[0], rtol=1e-13)
