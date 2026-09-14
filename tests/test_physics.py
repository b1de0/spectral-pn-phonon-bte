"""Adversarial tests for physics calculations."""

import numpy as np
import pytest
import torch

from pinn_bte.physics.mesh_1d import (
    generate_1d_mesh,
    compute_phonon_properties,
    compute_relaxation_time_torch,
    compute_group_velocity_torch,
    compute_omega_torch,
)
from pinn_bte.config.physics import (
    LATTICE_CONSTANT,
    C10_TA,
    C20_TA,
    C11_LA,
    C21_LA,
    HKB,
)
from pinn_bte.config.materials import (
    Material,
    SILICON,
    GERMANIUM,
    SILICON_CARBIDE,
    GALLIUM_NITRIDE,
    get_material,
    list_materials,
)


class TestMeshGenerationEdgeCases:
    """Try to break mesh generation with edge cases."""

    def test_single_point_mesh(self):
        """Single point mesh - minimum viable case."""
        x, mu, w, k = generate_1d_mesh(1, 2, 1)
        assert x.shape == (1, 1)
        assert mu.shape == (2, 1)
        assert k.shape == (1, 1)

    def test_zero_points_should_fail(self):
        """Zero points should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            generate_1d_mesh(0, 16, 10)

    def test_negative_points_should_fail(self):
        """Negative points should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            generate_1d_mesh(-5, 16, 10)

    def test_odd_quadrature_points(self):
        _, mu, w, _ = generate_1d_mesh(10, 15, 5)  # 15 is odd
        assert np.isclose(np.sum(w), 4 * np.pi, rtol=1e-10)

    def test_very_large_mesh(self):
        # 1000 points in each dimension - should work but slow
        x, mu, w, k = generate_1d_mesh(100, 64, 50)
        assert x.shape == (100, 1)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(k))


class TestDispersionPhysicsLimits:
    """Test dispersion against known physics limits."""

    def test_zero_k_gives_zero_omega(self):
        """At k=0, acoustic phonon frequency must be zero (definition of
        acoustic).
        """
        k = np.array([[0.0]])
        branch_ta = np.zeros_like(k)
        branch_la = np.ones_like(k)

        omega_ta, v_ta, _, _, _ = compute_phonon_properties(k, branch_ta, 300.0)
        omega_la, v_la, _, _, _ = compute_phonon_properties(k, branch_la, 300.0)

        # ω(k=0) = c1*0 + c2*0² = 0
        assert np.isclose(omega_ta[0], 0, atol=1e-15), f"ω_TA(0)={omega_ta[0]} should be 0"
        assert np.isclose(omega_la[0], 0, atol=1e-15), f"ω_LA(0)={omega_la[0]} should be 0"

    def test_zone_center_velocity_equals_sound_speed(self):
        """At k→0, group velocity = sound speed = c1 (dispersion coefficient)."""
        k = np.array([[1e-10]])  # Very small k, approaching zone center
        branch_ta = np.zeros_like(k)
        branch_la = np.ones_like(k)

        _, v_ta, _, _, _ = compute_phonon_properties(k, branch_ta, 300.0)
        _, v_la, _, _, _ = compute_phonon_properties(k, branch_la, 300.0)

        # v = dω/dk = c1 + 2*c2*k ≈ c1 at k→0
        assert np.isclose(v_ta[0], C10_TA, rtol=1e-5), f"v_TA(0)={v_ta[0]} should be {C10_TA}"
        assert np.isclose(v_la[0], C11_LA, rtol=1e-5), f"v_LA(0)={v_la[0]} should be {C11_LA}"

    def test_negative_k_behavior(self):
        """Negative k - physically meaningless but should not crash."""
        k = np.array([[-0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.zeros_like(k)

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        # Should produce values (possibly negative omega) but not NaN
        assert np.all(np.isfinite(omega)), "Negative k should not produce NaN omega"

    def test_group_velocity_can_become_negative(self):
        """At large k, group velocity can become negative (v = c1 + 2*c2*k, c2 <
        0).
        """
        # For TA: c2 = C20_TA < 0, so v becomes negative at large k
        # v = 0 when k = -c1/(2*c2)
        k_zero_v_ta = -C10_TA / (2 * C20_TA)

        # Test just past this point
        k = np.array([[k_zero_v_ta * 1.1]])
        branch = np.zeros_like(k)

        _, v, _, _, _ = compute_phonon_properties(k, branch, 300.0)

        # v should be negative here (physical: phonon slowing down at zone boundary)
        assert v[0] < 0, f"Group velocity should be negative at k={k[0,0]}, got v={v[0]}"

    def test_invalid_branch_value(self):
        """Branch values outside [0,1] - what happens?"""
        k = np.array([[0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch_invalid = np.array([[2.0]])  # Not 0 or 1

        # This should still compute (linear interpolation beyond bounds)
        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch_invalid, 300.0)

        # Result is extrapolated - may be unphysical but should not crash
        assert np.all(np.isfinite(omega)), "Invalid branch should not produce NaN"


class TestRelaxationTimeFailureModes:
    """Try to break relaxation time calculations."""

    def test_zero_temperature_blows_up(self):
        """T=0 causes division by zero in scattering formulas."""
        k = np.array([[0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.zeros_like(k)

        # T=0 should cause problems: T^3, T^4 terms go to zero
        # But also sinh(ℏω/kT) → sinh(∞) → ∞, so 1/sinh → 0
        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 0.0)

        # This might produce inf or nan - that's a BUG if it does
        # After our fixes, it should be finite
        assert np.all(np.isfinite(tau)), f"τ at T=0 is {tau}, should be finite"

    def test_very_low_temperature_overflow(self):
        """Very low T causes exp(ℏω/kT) overflow."""
        k = np.array([[0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.zeros_like(k)

        # At T=1K, exp(ℏω/kT) is huge
        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 1.0)

        assert np.all(np.isfinite(tau)), f"τ at T=1K should be finite, got {tau}"
        assert np.all(np.isfinite(dfeq)), f"dfeq at T=1K should be finite, got {dfeq}"

    def test_zero_omega_sinh_singularity(self):
        """At ω=0, sinh(ℏω/kT)=0 causes division by zero."""
        # ω=0 happens at k=0
        k = np.array([[0.0]])
        branch = np.zeros_like(k)

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        # sinh(0) = 0, so tau_NU_TA_high = B_U * 0² / 0 = 0/0
        # Our fix should handle this
        assert np.all(np.isfinite(tau)), f"τ at k=0 should be finite, got {tau}"

    def test_ta_branch_transition_at_pi_over_a(self):
        """TA scattering formula changes at k = π/a - known discontinuity."""
        k_boundary = np.pi / LATTICE_CONSTANT

        # Test just below and above the transition
        k_below = np.array([[k_boundary * 0.99]])
        k_above = np.array([[k_boundary * 1.01]])
        branch = np.zeros_like(k_below)

        _, _, _, tau_below, _ = compute_phonon_properties(k_below, branch, 300.0)
        _, _, _, tau_above, _ = compute_phonon_properties(k_above, branch, 300.0)

        # Both should be finite (no NaN or inf)
        assert np.isfinite(tau_below[0]), f"τ below transition: {tau_below}"
        assert np.isfinite(tau_above[0]), f"τ above transition: {tau_above}"

        # Document the known discontinuity (Umklapp region has ~50x longer τ)
        ratio = tau_above[0] / tau_below[0]
        assert ratio > 10, f"Expected large discontinuity at π/a, got ratio={ratio}"

        # The discontinuity should be bounded (not infinite)
        assert ratio < 200, f"Discontinuity too large: ratio={ratio}"

    def test_torch_numpy_mismatch_at_extremes(self):
        """Torch and numpy might diverge at numerical extremes."""
        # Test at very small k where sinh singularity lurks
        k_np = np.array([[1e-12]]) * np.pi * 2 / LATTICE_CONSTANT
        k_torch = torch.DoubleTensor(k_np)  # Use double for fair comparison
        branch_np = np.zeros_like(k_np)
        branch_torch = torch.zeros_like(k_torch)

        _, _, _, tau_np, _ = compute_phonon_properties(k_np, branch_np, 300.0)
        tau_torch = compute_relaxation_time_torch(
            k_torch, branch_torch, torch.tensor(300.0)
        ).numpy()

        # They should match within reasonable tolerance
        if np.isfinite(tau_np[0]) and np.isfinite(tau_torch[0]):
            np.testing.assert_allclose(tau_torch, tau_np, rtol=0.1)


class TestBoseEinsteinOverflow:
    """Test Bose-Einstein distribution for overflow conditions."""

    def test_classical_limit_high_temperature(self):
        """At high T, f_eq ≈ kT/ℏω (classical limit)."""
        k = np.array([[0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.zeros_like(k)

        # At T=10000K, ℏω << kT, so exp(ℏω/kT) ≈ 1 + ℏω/kT
        # f_eq = 1/(exp(x)-1) ≈ 1/x = kT/ℏω
        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 10000.0)

        assert np.all(np.isfinite(dfeq)), f"dfeq at high T should be finite: {dfeq}"

    def test_quantum_limit_low_temperature(self):
        """At very low T, f_eq → 0 exponentially."""
        k = np.array([[0.5]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.zeros_like(k)

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 5.0)

        # dfeq should be very small but finite, not NaN from overflow
        assert np.all(np.isfinite(dfeq)), f"dfeq at T=5K: {dfeq}"
        assert np.all(dfeq >= 0), f"dfeq should be non-negative: {dfeq}"


class TestDensityOfStatesEdgeCases:
    """Test DOS calculation edge cases."""

    def test_dos_at_zero_k(self):
        """D(k) = k²/(2π²v) at k=0 should be 0 or handled."""
        k = np.array([[0.0]])
        branch = np.zeros_like(k)

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        # D = 0²/(2π²*c1) = 0
        assert np.isclose(D[0], 0, atol=1e-20), f"D(0) should be 0, got {D[0]}"

    def test_dos_diverges_when_velocity_zero(self):
        """D = k²/v diverges when v→0 (van Hove singularity)."""
        # Find k where v=0: k = -c1/(2*c2)
        k_zero_v = -C10_TA / (2 * C20_TA)

        # Just before the singularity
        k = np.array([[k_zero_v * 0.999]])
        branch = np.zeros_like(k)

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        # D should be large but finite (v is small but not zero)
        assert np.isfinite(D[0]), f"D near v=0 should be finite, got {D}"

        # At exactly v=0, D = k²/0 = inf - this is physical (van Hove singularity)
        k_exact = np.array([[k_zero_v]])
        _, v_exact, D_exact, _, _ = compute_phonon_properties(k_exact, branch, 300.0)

        # v should be ~0
        assert np.abs(v_exact[0]) < 1e-10, f"v at k_zero_v should be ~0, got {v_exact}"
        # D will be inf or very large
        assert D_exact[0] > 1e10 or np.isinf(D_exact[0]), \
            f"D should diverge at v=0, got {D_exact}"


class TestBranchInterpolation:
    """Test that branch parameter interpolates correctly."""

    def test_branch_zero_gives_ta(self):
        """branch=0 should give pure TA properties."""
        k = np.array([[0.3]]) * np.pi * 2 / LATTICE_CONSTANT

        omega_ta, v_ta, _, _, _ = compute_phonon_properties(k, np.array([[0.0]]), 300.0)

        # Manual calculation for TA
        expected_omega = C10_TA * k[0, 0] + C20_TA * k[0, 0]**2
        expected_v = C10_TA + 2 * C20_TA * k[0, 0]

        assert np.isclose(omega_ta[0], expected_omega, rtol=1e-10)
        assert np.isclose(v_ta[0], expected_v, rtol=1e-10)

    def test_branch_one_gives_la(self):
        """branch=1 should give pure LA properties."""
        k = np.array([[0.3]]) * np.pi * 2 / LATTICE_CONSTANT

        omega_la, v_la, _, _, _ = compute_phonon_properties(k, np.array([[1.0]]), 300.0)

        # Manual calculation for LA
        expected_omega = C11_LA * k[0, 0] + C21_LA * k[0, 0]**2
        expected_v = C11_LA + 2 * C21_LA * k[0, 0]

        assert np.isclose(omega_la[0], expected_omega, rtol=1e-10)
        assert np.isclose(v_la[0], expected_v, rtol=1e-10)

    def test_branch_half_interpolates(self):
        """branch=0.5 should give average of TA and LA."""
        k = np.array([[0.3]]) * np.pi * 2 / LATTICE_CONSTANT

        omega_ta, v_ta, _, _, _ = compute_phonon_properties(k, np.array([[0.0]]), 300.0)
        omega_la, v_la, _, _, _ = compute_phonon_properties(k, np.array([[1.0]]), 300.0)
        omega_mid, v_mid, _, _, _ = compute_phonon_properties(k, np.array([[0.5]]), 300.0)

        expected_omega = (omega_ta[0] + omega_la[0]) / 2
        expected_v = (v_ta[0] + v_la[0]) / 2

        assert np.isclose(omega_mid[0], expected_omega, rtol=1e-10)
        assert np.isclose(v_mid[0], expected_v, rtol=1e-10)


class TestArrayBroadcasting:
    """Test that array operations broadcast correctly."""

    def test_mismatched_shapes_should_fail_or_broadcast(self):
        """k and branch with different shapes - what happens?"""
        k = np.array([[0.1], [0.2], [0.3]]) * np.pi * 2 / LATTICE_CONSTANT
        branch = np.array([[0.0]])  # Shape mismatch: (3,1) vs (1,1)

        # NumPy will broadcast this - result shape should be (3,1)
        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        assert omega.shape == (3, 1), f"Should broadcast to (3,1), got {omega.shape}"

    def test_1d_vs_2d_arrays(self):
        """1D array vs 2D column vector - should work."""
        k_1d = np.array([0.1, 0.2, 0.3]) * np.pi * 2 / LATTICE_CONSTANT
        k_2d = k_1d.reshape(-1, 1)
        branch = np.zeros_like(k_2d)

        # 1D should fail or be handled
        try:
            omega_1d, _, _, _, _ = compute_phonon_properties(k_1d, branch.flatten(), 300.0)
            omega_2d, _, _, _, _ = compute_phonon_properties(k_2d, branch, 300.0)
            # If both work, results should match
            np.testing.assert_allclose(omega_1d.flatten(), omega_2d.flatten())
        except (ValueError, IndexError):
            # If 1D fails, that's acceptable behavior
            pass


class TestMaterialParametrization:
    """Test material-specific physics calculations."""

    def test_default_is_silicon(self):
        """No material specified should use Silicon."""
        k = np.array([[0.3]]) * np.pi * 2 / SILICON.lattice_constant
        branch = np.zeros_like(k)

        omega_default, v_default, _, _, _ = compute_phonon_properties(k, branch, 300.0)
        omega_silicon, v_silicon, _, _, _ = compute_phonon_properties(
            k, branch, 300.0, material=SILICON
        )

        np.testing.assert_allclose(omega_default, omega_silicon, rtol=1e-10)
        np.testing.assert_allclose(v_default, v_silicon, rtol=1e-10)

    def test_material_by_name_string(self):
        """Material can be specified by string name."""
        k = np.array([[0.3]]) * np.pi * 2 / SILICON.lattice_constant
        branch = np.zeros_like(k)

        omega_obj, v_obj, _, _, _ = compute_phonon_properties(
            k, branch, 300.0, material=SILICON
        )
        omega_str, v_str, _, _, _ = compute_phonon_properties(
            k, branch, 300.0, material="silicon"
        )

        np.testing.assert_allclose(omega_obj, omega_str, rtol=1e-10)
        np.testing.assert_allclose(v_obj, v_str, rtol=1e-10)

    def test_unknown_material_raises(self):
        """Unknown material name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown material"):
            get_material("unobtanium")

    def test_different_materials_give_different_results(self):
        """Different materials should have different phonon properties."""
        k_val = 0.5  # Relative to BZ edge
        k_si = np.array([[k_val]]) * np.pi * 2 / SILICON.lattice_constant
        k_ge = np.array([[k_val]]) * np.pi * 2 / GERMANIUM.lattice_constant
        branch = np.zeros((1, 1))

        omega_si, v_si, _, tau_si, _ = compute_phonon_properties(
            k_si, branch, 300.0, material=SILICON
        )
        omega_ge, v_ge, _, tau_ge, _ = compute_phonon_properties(
            k_ge, branch, 300.0, material=GERMANIUM
        )

        # Different materials should give different results
        assert not np.isclose(omega_si[0], omega_ge[0], rtol=0.01), \
            f"Silicon and Germanium should have different ω: {omega_si[0]} vs {omega_ge[0]}"
        assert not np.isclose(v_si[0], v_ge[0], rtol=0.01), \
            f"Silicon and Germanium should have different v: {v_si[0]} vs {v_ge[0]}"

    def test_mesh_uses_material_lattice_constant(self):
        """Mesh generation should use material-specific lattice constant."""
        _, _, _, k_si = generate_1d_mesh(10, 16, 5, material=SILICON)
        _, _, _, k_ge = generate_1d_mesh(10, 16, 5, material=GERMANIUM)

        # k_max = π*2/a, so different lattice constants → different k ranges
        k_max_si = k_si[-1, 0]
        k_max_ge = k_ge[-1, 0]

        # Ge has larger lattice constant → smaller k_max
        ratio = k_max_si / k_max_ge
        expected_ratio = GERMANIUM.lattice_constant / SILICON.lattice_constant
        assert np.isclose(ratio, expected_ratio, rtol=0.01), \
            f"k_max ratio should match lattice constant ratio: {ratio} vs {expected_ratio}"

    def test_material_temperature_validation(self):
        """Material should validate temperature range."""
        # Silicon valid range: 10-1500K
        SILICON.validate_temperature(300.0)  # Should pass

        with pytest.raises(ValueError, match="outside valid range"):
            SILICON.validate_temperature(5.0)  # Too low

        with pytest.raises(ValueError, match="outside valid range"):
            SILICON.validate_temperature(2000.0)  # Too high

    def test_torch_functions_accept_material(self):
        """PyTorch versions should accept material parameter."""
        k = torch.tensor([[0.3]]) * np.pi * 2 / SILICON.lattice_constant
        branch = torch.zeros_like(k)
        temp = torch.tensor(300.0)

        # Should work without error
        tau_si = compute_relaxation_time_torch(k, branch, temp, material=SILICON)
        tau_ge = compute_relaxation_time_torch(k, branch, temp, material="ge")

        assert torch.isfinite(tau_si).all()
        assert torch.isfinite(tau_ge).all()
        # Different materials should give different tau values
        # (they won't be exactly equal)
        assert tau_si.item() != tau_ge.item(), \
            f"Si and Ge should have different tau: {tau_si.item()} vs {tau_ge.item()}"

        v_si = compute_group_velocity_torch(k, branch, material=SILICON)
        omega_si = compute_omega_torch(k, branch, material=SILICON)

        assert torch.isfinite(v_si).all()
        assert torch.isfinite(omega_si).all()

    def test_list_materials_returns_all(self):
        """list_materials should return all defined materials."""
        materials = list_materials()

        # Should have core materials for nanoscale thermal transport
        names = [m["name"] for m in materials]
        assert "silicon" in names
        assert "germanium" in names
        assert "silicon_carbide" in names
        assert "gallium_nitride" in names

        # All materials now have validated parameters (no placeholders)
        for mat_info in materials:
            assert not mat_info["placeholder"], f"{mat_info['name']} is still a placeholder"

    def test_case_insensitive_material_lookup(self):
        """Material lookup should be case-insensitive."""
        si_lower = get_material("silicon")
        si_mixed = get_material("SiLiCoN")

        assert si_lower == si_mixed

    def test_symbol_lookup(self):
        """Materials should be accessible by chemical symbol."""
        si_name = get_material("silicon")
        si_symbol = get_material("si")
        ge_symbol = get_material("ge")

        assert si_name == si_symbol
        assert ge_symbol.name == "germanium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
