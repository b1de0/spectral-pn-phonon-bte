"""Electron transport properties for doped silicon."""

from dataclasses import dataclass
from typing import Final

import numpy as np


ELECTRON_CHARGE: Final[float] = 1.602176634e-19  # C
ELECTRON_MASS: Final[float] = 9.1093837015e-31  # kg
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23  # J/K
HBAR_SI: Final[float] = 1.054571817e-34  # J·s
LORENZ_NUMBER: Final[float] = 2.44e-8  # W·Ω/K² (Wiedemann-Franz)


SI_ELECTRON_DOS_MASS_RATIO: Final[float] = 1.08

# Conductivity effective mass (harmonic mean of longitudinal and transverse)
# 1/m_cond = (1/3)(1/m_l + 2/m_t) → m_cond ≈ 0.26 m₀
SI_ELECTRON_COND_MASS_RATIO: Final[float] = 0.26

# Hole effective masses (for p-type)
SI_HOLE_HH_MASS_RATIO: Final[float] = 0.537  # heavy hole
SI_HOLE_LH_MASS_RATIO: Final[float] = 0.154  # light hole


@dataclass(frozen=True)
class MobilityParams:
    """Caughey-Thomas mobility model parameters."""

    mu_min: float  # cm²/V·s — minimum mobility at high doping
    mu_max_300: float  # cm²/V·s — max mobility at 300K (undoped limit)
    gamma_T: float  # temperature exponent for μ_max
    n_ref: float  # cm⁻³ — reference concentration
    alpha: float  # doping exponent


# Electron mobility in n-type Si -- see the source chain above
# (form Caughey-Thomas; μ_min/n_ref/α Masetti 300 K; γ_T, μ_max Klaassen).
MOBILITY_ELECTRON_SI = MobilityParams(
    mu_min=68.5,  # cm²/V·s
    mu_max_300=1414.0,  # cm²/V·s
    gamma_T=2.285,  # T exponent
    n_ref=9.20e16,  # cm⁻³
    alpha=0.711,
)

# Hole mobility in p-type Si -- same three-source chain as the n-type row.
MOBILITY_HOLE_SI = MobilityParams(
    mu_min=44.9,  # cm²/V·s
    mu_max_300=470.5,  # cm²/V·s
    gamma_T=2.247,  # T exponent
    n_ref=2.23e17,  # cm⁻³
    alpha=0.719,
)


DOPANT_PHONON_SCATTERING_COEFF: Final[float] = 7.5e-12


@dataclass(frozen=True)
class DopingConfig:
    """Doping-dependent electron transport configuration."""

    carrier_concentration: float  # cm⁻³
    carrier_type: str = "n"

    def __post_init__(self) -> None:
        if self.carrier_concentration < 0:
            raise ValueError(
                f"carrier_concentration must be positive, got {self.carrier_concentration}"
            )
        if self.carrier_type not in ("n", "p"):
            raise ValueError(
                f"carrier_type must be 'n' or 'p', got '{self.carrier_type}'"
            )

    @property
    def mobility_params(self) -> MobilityParams:
        """Get mobility model parameters for this carrier type."""
        if self.carrier_type == "n":
            return MOBILITY_ELECTRON_SI
        return MOBILITY_HOLE_SI

    @property
    def effective_mass_ratio(self) -> float:
        """Conductivity effective mass ratio m*/m₀."""
        if self.carrier_type == "n":
            return SI_ELECTRON_COND_MASS_RATIO
        # p-type: use DOS average of heavy and light holes
        return (SI_HOLE_HH_MASS_RATIO ** 1.5 + SI_HOLE_LH_MASS_RATIO ** 1.5) ** (
            2 / 3
        )

    @property
    def dos_mass_ratio(self) -> float:
        """Density-of-states effective mass ratio for Fermi energy."""
        if self.carrier_type == "n":
            return SI_ELECTRON_DOS_MASS_RATIO
        return (SI_HOLE_HH_MASS_RATIO ** 1.5 + SI_HOLE_LH_MASS_RATIO ** 1.5) ** (
            2 / 3
        )


def fermi_energy(n_d: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO) -> float:
    """Compute Fermi energy from carrier concentration (degenerate limit)."""
    n_si = n_d * 1e6  # cm⁻³ → m⁻³
    m_star = m_dos_ratio * ELECTRON_MASS
    return (HBAR_SI ** 2 / (2 * m_star)) * (3 * np.pi ** 2 * n_si) ** (2 / 3)


def fermi_energy_eV(
    n_d: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO
) -> float:
    """Compute Fermi energy in eV."""
    return fermi_energy(n_d, m_dos_ratio) / ELECTRON_CHARGE


def sommerfeld_coefficient(
    n_d: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO
) -> float:
    """Compute Sommerfeld coefficient γ for electronic specific heat."""
    n_si = n_d * 1e6  # cm⁻³ → m⁻³
    E_F = fermi_energy(n_d, m_dos_ratio)
    return np.pi ** 2 * BOLTZMANN_CONSTANT ** 2 * n_si / (2 * E_F)


_FD_U_MAX: Final[float] = np.sqrt(200.0)
_FD_N_POINTS: Final[int] = 4000
_FD_NEWTON_ITERS: Final[int] = 60

_FD_DEGENERATE_ETA: Final[float] = 100.0


def fermi_dirac_integral(j: float, eta: float) -> float:
    """Fermi-Dirac integral F_j(η) = ∫₀^∞ x^j / (1 + e^(x-η)) dx."""
    u = np.linspace(0.0, _FD_U_MAX, _FD_N_POINTS)
    exponent = np.clip(u ** 2 - eta, -700.0, 700.0)
    integrand = 2.0 * u ** (2 * j + 1) / (1.0 + np.exp(exponent))
    return float(np.trapezoid(integrand, u))


def effective_dos_conduction(
    T: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO
) -> float:
    """Effective conduction-band DOS N_c = 2·(m*·k_B·T / 2πℏ²)^(3/2) in m⁻³."""
    m_star = m_dos_ratio * ELECTRON_MASS
    return 2.0 * (
        m_star * BOLTZMANN_CONSTANT * T / (2.0 * np.pi * HBAR_SI ** 2)
    ) ** 1.5


def reduced_fermi_level(
    n_d: float, T: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO
) -> float:
    """Reduced Fermi level η = μ/(k_B·T) from carrier concentration."""
    n_si = n_d * 1e6  # cm⁻³ → m⁻³
    N_c = effective_dos_conduction(T, m_dos_ratio)
    r = n_si / N_c
    target = r * np.sqrt(np.pi) / 2.0  # = F_{1/2}(η)
    # Asymptotic initial guess: Boltzmann (r << 1) or degenerate (r >> 1)
    eta = np.log(r) if r < 1.0 else (3.0 * np.sqrt(np.pi) * r / 4.0) ** (2.0 / 3.0)
    for _ in range(_FD_NEWTON_ITERS):
        f = fermi_dirac_integral(0.5, eta) - target
        fp = 0.5 * fermi_dirac_integral(-0.5, eta)
        eta -= float(np.clip(f / fp, -5.0, 5.0))
    return float(eta)


def electron_specific_heat(
    n_d: float, T: float, m_dos_ratio: float = SI_ELECTRON_DOS_MASS_RATIO
) -> float:
    """Compute electronic specific heat from exact Fermi-Dirac statistics."""
    eta = reduced_fermi_level(n_d, T, m_dos_ratio)
    if eta > _FD_DEGENERATE_ETA:
        # Strongly degenerate (cryo): exact Sommerfeld gamma*T, well outside
        # the exact-FD quadrature's valid window. See _FD_DEGENERATE_ETA.
        return sommerfeld_coefficient(n_d, m_dos_ratio) * T
    N_c = effective_dos_conduction(T, m_dos_ratio)
    F12 = fermi_dirac_integral(0.5, eta)
    F32 = fermi_dirac_integral(1.5, eta)
    Fm12 = fermi_dirac_integral(-0.5, eta)
    return (
        N_c * BOLTZMANN_CONSTANT * (2.0 / np.sqrt(np.pi))
        * (2.5 * F32 - 4.5 * F12 ** 2 / Fm12)
    )


def mobility(n_d: float, T: float, params: MobilityParams) -> float:
    """Compute carrier mobility using Caughey-Thomas model."""
    mu_max_T = params.mu_max_300 * (T / 300.0) ** (-params.gamma_T)
    return params.mu_min + (mu_max_T - params.mu_min) / (
        1 + (n_d / params.n_ref) ** params.alpha
    )


def electrical_conductivity(n_d: float, T: float, carrier_type: str = "n") -> float:
    """Compute electrical conductivity σ = n × e × μ."""
    params = MOBILITY_ELECTRON_SI if carrier_type == "n" else MOBILITY_HOLE_SI
    mu = mobility(n_d, T, params)  # cm²/(V·s)
    mu_si = mu * 1e-4  # → m²/(V·s)
    n_si = n_d * 1e6  # cm⁻³ → m⁻³
    return n_si * ELECTRON_CHARGE * mu_si


def electron_thermal_conductivity(sigma: float, T: float) -> float:
    """Compute electron thermal conductivity via Wiedemann-Franz law."""
    return LORENZ_NUMBER * sigma * T


def electron_kappa(n_d: float, T: float, carrier_type: str = "n") -> float:
    """Compute electron thermal conductivity from doping and temperature."""
    sigma = electrical_conductivity(n_d, T, carrier_type)
    return electron_thermal_conductivity(sigma, T)


SI_ELECTRON_ENERGY_RELAXATION_TIME = 2.931e-13  # s

# G at the reference doping, DERIVED rather than pinned, so it can never drift
# from the specific heat the solver actually uses. 6.8497e14 W/(m^3 K).
_G_REF = electron_specific_heat(1e19, 300.0) / SI_ELECTRON_ENERGY_RELAXATION_TIME


def electron_phonon_coupling(n_d: float, T: float) -> float:
    """Compute electron-phonon energy-transfer coefficient G for doped Si."""
    n_ref = 1e19  # cm⁻³ reference concentration
    return _G_REF * (n_d / n_ref) * (T / 300.0)


def electron_mfp(n_d: float, T: float, carrier_type: str = "n") -> float:
    """Compute electron mean free path."""
    m_cond = SI_ELECTRON_COND_MASS_RATIO * ELECTRON_MASS
    params = MOBILITY_ELECTRON_SI if carrier_type == "n" else MOBILITY_HOLE_SI
    mu = mobility(n_d, T, params) * 1e-4  # → m²/(V·s)
    tau_e = m_cond * mu / ELECTRON_CHARGE  # relaxation time (s)
    # Fermi velocity
    n_si = n_d * 1e6  # → m⁻³
    k_F = (3 * np.pi ** 2 * n_si) ** (1 / 3)
    v_F = HBAR_SI * k_F / m_cond
    return v_F * tau_e


def electron_knudsen(n_d: float, T: float, L_m: float, carrier_type: str = "n") -> float:
    """Compute electron Knudsen number Kn_e = λ_e / L."""
    return electron_mfp(n_d, T, carrier_type) / L_m


def dopant_phonon_scattering_rate(
    n_d: float, omega: np.ndarray
) -> np.ndarray:
    """Compute additional phonon scattering rate from dopant impurities."""
    return DOPANT_PHONON_SCATTERING_COEFF * n_d * omega ** 2
