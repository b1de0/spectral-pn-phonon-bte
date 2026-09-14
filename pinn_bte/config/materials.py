"""Material properties for phonon transport simulations."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Material:
    """Material properties for phonon transport."""
    name: str
    symbol: str

    # Crystal structure
    lattice_constant: float  # Angstrom

    # Dispersion coefficients (10^13 rad/s units)
    c1_ta: float
    c2_ta: float
    c1_la: float
    c2_la: float

    # Scattering coefficients (adjusted for 10^13 rad/s)
    impurity_coeff: float
    normal_b_l: float
    normal_b_t: float
    umklapp_b_u: float

    # Valid temperature range
    temp_min: float = 10.0
    temp_max: float = 1000.0

    # Reference
    reference: str = ""

    def validate_temperature(self, T: float) -> None:
        """Check if temperature is in valid range for this material."""
        if T < self.temp_min or T > self.temp_max:
            raise ValueError(
                f"Temperature {T}K outside valid range [{self.temp_min}, {self.temp_max}]K "
                f"for {self.name}"
            )


SILICON = Material(
    name="silicon",
    symbol="Si",
    lattice_constant=5.431,
    # Dispersion from DFT fitting (Li et al. 2022)
    c1_ta=5.23,
    c2_ta=-2.26,
    c1_la=9.01,
    c2_la=-2.0,
    # Scattering (Paper Table 3, adjusted for ω in 10^13 rad/s)
    # Original values in SI, converted: coeff_adjusted = coeff_SI * (10^13)^n
    impurity_coeff=1.498e7,   # A_i: 1.498e-45 s³ → * (1e13)^4
    normal_b_l=1.18e2,        # B_L: 1.18e-24 K⁻³ → * (1e13)^2
    normal_b_t=8.708,         # B_T: 8.708e-13 K⁻⁴ → * (1e13)^1
    umklapp_b_u=2.89e8,       # B_U: 2.89e-18 s → * (1e13)^2
    temp_min=10.0,
    temp_max=1500.0,  # Below melting point (1687K)
    reference="Li et al. (2022), DOI: 10.1038/s41524-022-00712-y"
)


GERMANIUM = Material(
    name="germanium",
    symbol="Ge",
    lattice_constant=5.658,  # Å, Ioffe Institute NSM Archive
    c1_ta=3.57,
    c2_ta=-1.5,  # Scaled from Si ratio (c2/c1 ≈ -0.43)
    c1_la=4.87,
    c2_la=-1.1,  # Scaled from Si ratio
    # Scattering: κ_Ge ≈ 60 W/m·K vs κ_Si ≈ 150 W/m·K
    # Scattering rates ~2.5× higher than Si (heavier atoms, softer bonds)
    impurity_coeff=3.75e7,   # A_i: Si×2.5
    normal_b_l=2.95e2,       # B_L: Si×2.5
    normal_b_t=21.8,         # B_T: Si×2.5
    umklapp_b_u=7.23e8,      # B_U: Si×2.5
    temp_min=10.0,
    temp_max=1200.0,  # Below melting point (1211K)
    reference="Ioffe NSM Archive, scaled from Si (Li et al. 2022)"
)


SILICON_CARBIDE = Material(
    name="silicon_carbide",
    symbol="SiC",
    lattice_constant=3.073,  # Å (4H-SiC in-plane), Ioffe NSM Archive
    # Dispersion from acoustic velocities (Ioffe NSM, 4H-SiC):
    # v_TA[001] = 7.1×10⁵ cm/s, v_LA[001] = 13.1×10⁵ cm/s
    c1_ta=7.1,
    c2_ta=-2.8,  # Scaled from Si ratio
    c1_la=13.1,
    c2_la=-2.9,  # Scaled from Si ratio
    # Scattering: κ_SiC ≈ 490 W/m·K at 300K (4H-SiC)
    # Scattering rates ~0.3× of Si (stiff bonds, low anharmonicity)
    impurity_coeff=0.45e7,   # A_i: Si×0.3
    normal_b_l=0.35e2,       # B_L: Si×0.3
    normal_b_t=2.6,          # B_T: Si×0.3
    umklapp_b_u=0.87e8,      # B_U: Si×0.3
    temp_min=10.0,
    temp_max=2700.0,  # Sublimes at ~2700°C
    reference="Ioffe NSM Archive (4H-SiC), thermal conductivity from Slack (1964)"
)


GALLIUM_ARSENIDE = Material(
    name="gallium_arsenide",
    symbol="GaAs",
    lattice_constant=5.653,  # Å, Ioffe NSM Archive
    # Dispersion from acoustic velocities (Ioffe NSM):
    # v_TA[100] = 3.35×10⁵ cm/s, v_LA[100] = 4.73×10⁵ cm/s
    c1_ta=3.35,
    c2_ta=-1.4,  # Scaled from Si ratio
    c1_la=4.73,
    c2_la=-1.0,  # Scaled from Si ratio
    # Scattering: κ_GaAs ≈ 55 W/m·K at 300K
    # Scattering rates ~2.7× of Si (polar compound, more optical phonon channels)
    impurity_coeff=4.0e7,    # A_i: Si×2.7
    normal_b_l=3.2e2,        # B_L: Si×2.7
    normal_b_t=23.5,         # B_T: Si×2.7
    umklapp_b_u=7.8e8,       # B_U: Si×2.7
    temp_min=10.0,
    temp_max=1500.0,  # Below melting point (1511K)
    reference="Ioffe NSM Archive, Luo et al. (2013) arXiv:1209.6350"
)


ALUMINUM_NITRIDE = Material(
    name="aluminum_nitride",
    symbol="AlN",
    lattice_constant=3.112,  # Å (a-axis, wurtzite), Ioffe NSM Archive
    # Dispersion from acoustic velocities (Ioffe NSM):
    # v_TA = 6.22×10⁵ cm/s (shear), v_LA = 11.27×10⁵ cm/s (longitudinal)
    c1_ta=6.22,
    c2_ta=-2.5,  # Scaled from Si ratio
    c1_la=11.27,
    c2_la=-2.5,  # Scaled from Si ratio
    # Scattering: κ_AlN ≈ 320 W/m·K at 300K
    # Scattering rates ~0.47× of Si (stiff bonds, wide bandgap)
    impurity_coeff=0.70e7,   # A_i: Si×0.47
    normal_b_l=0.55e2,       # B_L: Si×0.47
    normal_b_t=4.1,          # B_T: Si×0.47
    umklapp_b_u=1.36e8,      # B_U: Si×0.47
    temp_min=10.0,
    temp_max=2400.0,  # Sublimes at ~2400°C
    reference="Ioffe NSM Archive, thermal conductivity from Slack et al. (1987)"
)

GALLIUM_NITRIDE = Material(
    name="gallium_nitride",
    symbol="GaN",
    lattice_constant=3.189,  # Å (a-axis, wurtzite), Ioffe NSM Archive
    # Dispersion from acoustic velocities (Ioffe NSM, wurtzite):
    # v_TA[001] = 4.13×10⁵ cm/s, v_LA[001] = 8.04×10⁵ cm/s
    c1_ta=4.13,
    c2_ta=-1.7,  # Scaled from Si ratio
    c1_la=8.04,
    c2_la=-1.8,  # Scaled from Si ratio
    # Scattering: κ_GaN ≈ 230 W/m·K at 300K
    # Scattering rates ~0.65× of Si (wide bandgap, less anharmonic)
    impurity_coeff=0.97e7,   # A_i: Si×0.65
    normal_b_l=0.77e2,       # B_L: Si×0.65
    normal_b_t=5.7,          # B_T: Si×0.65
    umklapp_b_u=1.88e8,      # B_U: Si×0.65
    temp_min=10.0,
    temp_max=2500.0,  # Decomposes at ~1100°C but sublimes above 2500°C in N₂
    reference="Ioffe NSM Archive, thermal conductivity from Mion et al. (2006)"
)


MATERIALS: Dict[str, Material] = {
    # Group IV
    "silicon": SILICON,
    "si": SILICON,
    "germanium": GERMANIUM,
    "ge": GERMANIUM,
    "silicon_carbide": SILICON_CARBIDE,
    "sic": SILICON_CARBIDE,
    # III-V
    "gallium_arsenide": GALLIUM_ARSENIDE,
    "gaas": GALLIUM_ARSENIDE,
    "gallium_nitride": GALLIUM_NITRIDE,
    "gan": GALLIUM_NITRIDE,
    "aluminum_nitride": ALUMINUM_NITRIDE,
    "aln": ALUMINUM_NITRIDE,
}


def get_material(name: str) -> Material:
    """Get material by name or symbol."""
    key = name.lower()
    if key not in MATERIALS:
        available = sorted(set(m.name for m in MATERIALS.values()))
        raise KeyError(
            f"Unknown material '{name}'. Available: {available}"
        )
    return MATERIALS[key]


def list_materials() -> list:
    """List all available materials."""
    seen = set()
    result = []
    for mat in MATERIALS.values():
        if mat.name not in seen:
            seen.add(mat.name)
            is_placeholder = "PLACEHOLDER" in mat.reference
            result.append({
                "name": mat.name,
                "symbol": mat.symbol,
                "placeholder": is_placeholder,
            })
    return result
