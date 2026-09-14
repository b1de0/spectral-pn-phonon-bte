"""Tests for 1D mesh generation and phonon properties."""

import numpy as np
import pytest

from pinn_bte.physics.mesh_1d import (
    generate_1d_mesh,
    compute_phonon_properties,
)
from pinn_bte.config.physics import LATTICE_CONSTANT


class TestMesh1DShapes:
    """Test 1D mesh output shapes and ranges."""

    def test_mesh_shapes(self):
        """Output arrays should have correct shapes."""
        Nx, Ns, Nk = 50, 16, 10
        x, mu, w, k = generate_1d_mesh(Nx, Ns, Nk)

        assert x.shape == (Nx, 1), f"x shape: {x.shape}"
        assert mu.shape == (Ns, 1), f"mu shape: {mu.shape}"
        assert w.shape == (Ns, 1), f"w shape: {w.shape}"
        assert k.shape == (Nk, 1), f"k shape: {k.shape}"

    def test_spatial_interior_only(self):
        """Spatial points should be in (0, 1), excluding boundaries."""
        x, _, _, _ = generate_1d_mesh(100, 8, 5)

        assert x.min() > 0, "x should not include left boundary (x=0)"
        assert x.max() < 1, "x should not include right boundary (x=1)"

    def test_mu_range(self):
        """Angular quadrature points should be in [-1, 1]."""
        _, mu, _, _ = generate_1d_mesh(10, 20, 5)

        assert mu.min() >= -1.0
        assert mu.max() <= 1.0

    def test_k_range(self):
        """k values should be in (0, 2π/a) - first Brillouin zone."""
        _, _, _, k = generate_1d_mesh(10, 8, 20)

        k_max = 2 * np.pi / LATTICE_CONSTANT
        assert k.min() > 0, "k should not include zone center"
        assert k.max() < k_max, f"k should be < 2π/a = {k_max}"


class TestQuadratureWeights:
    """Test angular quadrature integration properties."""

    def test_weights_sum_to_4pi(self):
        """Quadrature weights should sum to 4π (full solid angle)."""
        for Ns in [8, 16, 32, 64]:
            _, _, w, _ = generate_1d_mesh(10, Ns, 5)

            weight_sum = np.sum(w)
            expected = 4 * np.pi
            assert np.isclose(weight_sum, expected, rtol=1e-10), \
                f"Weight sum {weight_sum} != {expected} for Ns={Ns}"

    def test_integral_of_mu_squared(self):
        """Test: ∫μ² dΩ = 4π/3 (isotropic average of cos²θ)."""
        _, mu, w, _ = generate_1d_mesh(10, 32, 5)

        integral = np.sum(mu**2 * w)
        expected = 4 * np.pi / 3
        assert np.isclose(integral, expected, rtol=1e-6), \
            f"∫μ² dΩ = {integral}, expected {expected}"


class TestPhononProperties:
    """Test phonon property calculations."""

    def test_properties_are_finite(self):
        """All computed phonon properties should be finite (no NaN/inf)."""
        _, _, _, k = generate_1d_mesh(10, 8, 20)
        T_ref = 300.0
        Nk = k.shape[0]

        # Test both TA (branch=0) and LA (branch=1)
        for branch_val in [0.0, 1.0]:
            branch = np.full_like(k, branch_val)
            omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, T_ref)

            # Check shapes: (Nk, 1)
            assert omega.shape == (Nk, 1), f"omega shape: {omega.shape}"
            assert v.shape == (Nk, 1), f"v shape: {v.shape}"
            assert tau.shape == (Nk, 1), f"tau shape: {tau.shape}"
            assert D.shape == (Nk, 1), f"D shape: {D.shape}"
            assert dfeq.shape == (Nk, 1), f"dfeq shape: {dfeq.shape}"

            # All values should be finite
            assert np.all(np.isfinite(omega)), f"omega has NaN/inf for branch={branch_val}"
            assert np.all(np.isfinite(v)), f"v has NaN/inf for branch={branch_val}"
            assert np.all(np.isfinite(tau)), f"tau has NaN/inf for branch={branch_val}"
            assert np.all(np.isfinite(D)), f"D has NaN/inf for branch={branch_val}"
            assert np.all(np.isfinite(dfeq)), f"dfeq has NaN/inf for branch={branch_val}"

    def test_properties_positive(self):
        """Physical properties should be positive."""
        _, _, _, k = generate_1d_mesh(10, 8, 20)
        branch = np.zeros_like(k)  # TA branch

        omega, v, D, tau, dfeq = compute_phonon_properties(k, branch, 300.0)

        assert np.all(omega >= 0), "omega should be >= 0"
        assert np.all(tau > 0), "tau should be > 0"
        assert np.all(D > 0), "D (density of states) should be > 0"
        # Note: v can be negative (group velocity at zone edge)
        # Note: dfeq is negative (∂f/∂T for Bose-Einstein)

    def test_la_higher_frequency_than_ta(self):
        """LA branch should have higher frequency than TA at same k."""
        _, _, _, k = generate_1d_mesh(10, 8, 10)

        branch_ta = np.zeros_like(k)
        branch_la = np.ones_like(k)

        omega_ta, _, _, _, _ = compute_phonon_properties(k, branch_ta, 300.0)
        omega_la, _, _, _, _ = compute_phonon_properties(k, branch_la, 300.0)

        assert np.all(omega_la >= omega_ta), \
            "LA frequency should be >= TA frequency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
