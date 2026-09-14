"""Mesh generation and phonon physics calculations for BTE simulation."""

from typing import Tuple, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from pinn_bte.config.physics import (
    LATTICE_CONSTANT,
    HBAR,
    HKB,
    C10_TA,
    C20_TA,
    C11_LA,
    C21_LA,
    IMPURITY_SCATTERING_COEFF,
    NORMAL_SCATTERING_B_L,
    NORMAL_SCATTERING_B_T,
    UMKLAPP_SCATTERING_B_U,
)
from pinn_bte.config.materials import Material, SILICON, get_material

# Get device for torch operations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_1d_mesh(
    num_spatial_points: int,
    num_quadrature_points: int,
    num_frequency_bands: int,
    material: Optional[Union[Material, str]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate 1D mesh for spatial, angular, and frequency domains."""
    # Resolve material
    if material is None:
        mat = SILICON
    elif isinstance(material, str):
        mat = get_material(material)
    else:
        mat = material

    # Input validation
    if num_spatial_points < 1:
        raise ValueError(f"num_spatial_points must be >= 1, got {num_spatial_points}")
    if num_quadrature_points < 1:
        raise ValueError(f"num_quadrature_points must be >= 1, got {num_quadrature_points}")
    if num_frequency_bands < 1:
        raise ValueError(f"num_frequency_bands must be >= 1, got {num_frequency_bands}")

    # Generate spatial mesh (interior points only, exclude boundaries)
    x = np.linspace(0, 1, num_spatial_points + 2)[1:num_spatial_points + 1]
    x = x.reshape(-1, 1)

    mu, w = np.polynomial.legendre.leggauss(num_quadrature_points)
    mu = mu.reshape(-1, 1)
    w = w.reshape(-1, 1) * 2 * np.pi  # Scale weights by 2π for full solid angle

    # Generate frequency mesh (k-space) using material's lattice constant
    # Use midpoint sampling: take odd-indexed points from finer grid
    k = np.linspace(0, 1, num_frequency_bands * 2 + 1)[1:num_frequency_bands * 2 + 1]
    k = k.reshape(-1, 1)
    k = k[np.arange(0, num_frequency_bands * 2 - 1, 2)] * np.pi * 2 / mat.lattice_constant

    return x, mu, w, k


def augment_wall_mesh_j3(
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    ladder = np.array([1e-1, 1e-2, 1e-3])
    pts = np.concatenate([x.ravel(), ladder, 1.0 - ladder])
    if len(np.unique(pts)) != len(pts):
        raise ValueError("wall ladder collides with an existing mesh node")
    return np.sort(pts).reshape(-1, 1)


def compute_phonon_properties(
    k: npt.NDArray[np.float64],
    branch: npt.NDArray[np.float64],
    temperature: float,
    material: Optional[Union[Material, str]] = None,
    dopant_concentration: float = 0.0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64]]:
    """Compute phonon properties for given wave vectors and temperature."""
    # Resolve material
    if material is None:
        mat = SILICON
    elif isinstance(material, str):
        mat = get_material(material)
    else:
        mat = material

    # Get material-specific dispersion coefficients
    c1_ta, c2_ta = mat.c1_ta, mat.c2_ta
    c1_la, c2_la = mat.c1_la, mat.c2_la

    # Compute phonon dispersion coefficients (interpolate between TA and LA)
    c1 = (c1_la - c1_ta) * branch + c1_ta
    c2 = (c2_la - c2_ta) * branch + c2_ta

    # Phonon dispersion relation: ω(k) = c1*k + c2*k²
    omega = c1 * k + c2 * (k ** 2)  # Unit: 10^13 rad/s

    # Group velocity: v = dω/dk = c1 + 2*c2*k
    v = c1 + 2 * c2 * k  # Unit: 10^13 Angstrom/s

    # Compute relaxation times with material-specific scattering
    tau = _compute_relaxation_time(k, omega, branch, temperature, v, mat,
                                  dopant_concentration)

    # Derivative of Bose-Einstein distribution with respect to temperature
    # Clamp exponent to prevent overflow and ensure denominator > 0
    exp_arg = np.clip(HKB * omega / temperature, None, 50)
    exp_val = np.exp(exp_arg)
    dfeq = (HKB * omega / (temperature ** 2) * exp_val /
            (exp_val - 1 + 1e-30) ** 2)

    # Density of states: D(k) = k² / (2π² * v)
    D = (k ** 2) / (2 * np.pi ** 2 * v)  # Unit: 10^-13 s/Angstrom^3

    return omega, v, D, tau, dfeq


_HBAR_SI = 1.054572e-34  # J·s


def phonon_kappa_si(
    omega: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    dfeq: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    wk: float,
) -> float:
    """Bulk phonon thermal conductivity in SI [W/(m·K)]."""
    code_to_si = _HBAR_SI * 1e13 * 1e17 * 1e6 * (1e3 * 1e10)
    return float(
        (1.0 / 3.0) * code_to_si
        * np.sum(omega * D * dfeq * v ** 3 * tau * wk)
    )


def phonon_capacity_si(
    omega: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    D: npt.NDArray[np.float64],
    dfeq: npt.NDArray[np.float64],
    wk: float,
) -> float:
    """Volumetric phonon heat capacity in SI [J/(m³·K)]."""
    code_to_si = _HBAR_SI * 1e13 * 1e17 * (1e3 * 1e10)
    return float(code_to_si * np.sum(omega * D * dfeq * v * wk))


def _compute_relaxation_time(
    k: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    branch: npt.NDArray[np.float64],
    temperature: float,
    v: npt.NDArray[np.float64],
    material: Optional[Material] = None,
    dopant_concentration: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Compute phonon relaxation time from scattering processes."""
    # Use material-specific scattering coefficients, or defaults (Silicon)
    mat = material if material is not None else SILICON

    # Impurity scattering rate: τ_I^-1 = A_i·ω^4 (both branches)
    tau_impurity_inv = mat.impurity_coeff * (omega ** 4)

    # LA branch: τ_NU^-1 = B_L·ω^2·T^3
    tau_NU_LA_inv = mat.normal_b_l * (omega ** 2) * (temperature ** 3)

    # TA branch: depends on k value
    # Step function: 1 if k >= π/a, 0 otherwise
    step = np.heaviside(k - np.pi / mat.lattice_constant, 1)

    # TA low-k region (0 ≤ k < π/a): τ_NU^-1 = B_T·ω·T^4
    tau_NU_TA_low_inv = mat.normal_b_t * omega * (temperature ** 4)

    # TA high-k region (π/a ≤ k ≤ 2π/a): τ_NU^-1 = B_U·ω^2/sinh(ℏω/k_B·T)
    # Clamp sinh argument to avoid division by zero when omega→0
    sinh_arg = np.clip(HKB * omega / temperature, 1e-10, None)
    tau_NU_TA_high_inv = mat.umklapp_b_u * (omega ** 2) / np.sinh(sinh_arg)

    # Combine TA scattering based on k-region
    tau_NU_TA_inv = tau_NU_TA_low_inv * (1 - step) + tau_NU_TA_high_inv * step

    # Select between TA and LA based on branch
    # branch = 0 → TA, branch = 1 → LA
    tau_NU_inv = tau_NU_TA_inv * (1 - branch) + tau_NU_LA_inv * branch

    # Dopant impurity scattering: τ_dopant^-1 = A_d·n_d·ω² (Tamura 1983)
    tau_dopant_inv = 0.0
    if dopant_concentration > 0:
        from pinn_bte.config.electron import DOPANT_PHONON_SCATTERING_COEFF
        tau_dopant_inv = DOPANT_PHONON_SCATTERING_COEFF * dopant_concentration * (omega ** 2)

    # Total relaxation time (Matthiessen's rule): τ^-1 = τ_I^-1 + τ_NU^-1 + τ_dopant^-1
    # Add small epsilon to prevent division by zero
    tau = 1 / (tau_impurity_inv + tau_NU_inv + tau_dopant_inv + 1e-30)

    return tau


def compute_relaxation_time_torch(
    k: torch.Tensor,
    branch: torch.Tensor,
    temperature: torch.Tensor,
    material: Optional[Union[Material, str]] = None,
    dopant_concentration: float = 0.0,
) -> torch.Tensor:
    """Compute phonon relaxation time using PyTorch tensors."""
    # Resolve material
    if material is None:
        mat = SILICON
    elif isinstance(material, str):
        mat = get_material(material)
    else:
        mat = material

    # Ensure inputs are tensors on the same device
    if not isinstance(k, torch.Tensor):
        k = torch.FloatTensor(k)
    if not isinstance(branch, torch.Tensor):
        branch = torch.FloatTensor(branch)
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.FloatTensor(temperature)

    # Get device from branch (most likely the primary tensor)
    device = branch.device if isinstance(branch, torch.Tensor) else DEVICE
    k = k.to(device)
    branch = branch.to(device)
    temperature = temperature.to(device)

    # Get material-specific dispersion coefficients
    c1_ta, c2_ta = mat.c1_ta, mat.c2_ta
    c1_la, c2_la = mat.c1_la, mat.c2_la

    # Compute phonon dispersion
    c1 = (c1_la - c1_ta) * branch + c1_ta
    c2 = (c2_la - c2_ta) * branch + c2_ta
    omega = c1 * k + c2 * (k ** 2)

    # Impurity scattering rate: τ_I^-1 = A_i·ω^4 (both branches)
    tau_impurity_inv = mat.impurity_coeff * (omega ** 4)

    # LA branch: τ_NU^-1 = B_L·ω^2·T^3
    tau_NU_LA_inv = mat.normal_b_l * (omega ** 2) * (temperature ** 3)

    step = (k >= np.pi / mat.lattice_constant).to(k.dtype)

    # TA low-k region (0 ≤ k < π/a): τ_NU^-1 = B_T·ω·T^4
    tau_NU_TA_low_inv = mat.normal_b_t * omega * (temperature ** 4)

    # TA high-k region (π/a ≤ k ≤ 2π/a): τ_NU^-1 = B_U·ω^2/sinh(ℏω/k_B·T)
    # Clamp sinh argument to avoid division by zero when omega→0
    sinh_arg = torch.clamp(HKB * omega / temperature, min=1e-10)
    tau_NU_TA_high_inv = mat.umklapp_b_u * (omega ** 2) / torch.sinh(sinh_arg)

    # Combine TA scattering based on k-region
    tau_NU_TA_inv = tau_NU_TA_low_inv * (1 - step) + tau_NU_TA_high_inv * step

    # Select between TA and LA based on branch
    # branch = 0 → TA, branch = 1 → LA
    tau_NU_inv = tau_NU_TA_inv * (1 - branch) + tau_NU_LA_inv * branch

    # Dopant impurity scattering: τ_dopant^-1 = A_d·n_d·ω² (Tamura 1983)
    tau_dopant_inv = 0.0
    if isinstance(dopant_concentration, torch.Tensor) or dopant_concentration > 0:
        from pinn_bte.config.electron import DOPANT_PHONON_SCATTERING_COEFF
        tau_dopant_inv = DOPANT_PHONON_SCATTERING_COEFF * dopant_concentration * (omega ** 2)

    # Total relaxation time (Matthiessen's rule): τ^-1 = τ_I^-1 + τ_NU^-1 + τ_dopant^-1
    # Add small epsilon to prevent division by zero
    tau = 1 / (tau_impurity_inv + tau_NU_inv + tau_dopant_inv + 1e-30)

    return tau


def compute_group_velocity_torch(
    k: torch.Tensor,
    branch: torch.Tensor,
    material: Optional[Union[Material, str]] = None,
) -> torch.Tensor:
    """Compute phonon group velocity using PyTorch tensors."""
    # Resolve material
    if material is None:
        mat = SILICON
    elif isinstance(material, str):
        mat = get_material(material)
    else:
        mat = material

    c1 = (mat.c1_la - mat.c1_ta) * branch + mat.c1_ta
    c2 = (mat.c2_la - mat.c2_ta) * branch + mat.c2_ta
    v = c1 + 2 * c2 * k  # Unit: 10^13 Angstrom/s
    return v


def compute_omega_torch(
    k: torch.Tensor,
    branch: torch.Tensor,
    material: Optional[Union[Material, str]] = None,
) -> torch.Tensor:
    """Compute phonon frequency using PyTorch tensors."""
    # Resolve material
    if material is None:
        mat = SILICON
    elif isinstance(material, str):
        mat = get_material(material)
    else:
        mat = material

    c1 = (mat.c1_la - mat.c1_ta) * branch + mat.c1_ta
    c2 = (mat.c2_la - mat.c2_ta) * branch + mat.c2_ta
    omega = c1 * k + c2 * (k ** 2)  # Unit: 10^13 rad/s
    return omega


def compute_group_velocities(
    k: torch.Tensor,
    material: Optional[Union[Material, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute group velocities for both TA and LA branches with unit scaling."""
    v_TA = compute_group_velocity_torch(k, torch.zeros_like(k), material)
    v_LA = compute_group_velocity_torch(k, torch.ones_like(k), material)

    # Apply 1e11 scaling factor for BTE residual unit consistency
    # (matches thermal conductivity calculation scaling)
    v_TA = v_TA * 1e11
    v_LA = v_LA * 1e11

    return v_TA, v_LA

