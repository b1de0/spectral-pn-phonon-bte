"""Physics: silicon phonon properties, mode sets, and deterministic reference
solvers (spectral DOM, gray Volterra, matrix-exponential P_N).
"""

from pinn_bte.physics.mesh_1d import (
    compute_phonon_properties,
    compute_relaxation_time_torch,
)

__all__ = ["compute_phonon_properties", "compute_relaxation_time_torch"]
