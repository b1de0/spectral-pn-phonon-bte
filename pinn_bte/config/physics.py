"""Physics constants and training hyperparameters for PINN-pBTE simulation."""

from typing import Final


# Phonon velocity scale: v_physical [Å/s] = v_code × VELOCITY_SCALE
VELOCITY_SCALE: Final[float] = 1e13

# Angular frequency scale: ω_physical [rad/s] = ω_code × FREQUENCY_SCALE
FREQUENCY_SCALE: Final[float] = 1e13

# Length: micrometers ↔ Angstroms (1 μm = 10,000 Å)
UM_TO_ANGSTROM: Final[float] = 1e4
ANGSTROM_TO_UM: Final[float] = 1e-4

# Time: nanoseconds ↔ seconds
NS_TO_S: Final[float] = 1e-9
S_TO_NS: Final[float] = 1e9

# Paper's time unit: 10^-11 seconds (10 picoseconds)
# Used in 2D small-ΔT trainer for time normalization
PAPER_TIME_UNIT: Final[float] = 1e-11

# Base length unit for log_L encoding: 100 Å = 10 nm
# L_physical [m] = 10^{log_L} × BASE_LENGTH_M
BASE_LENGTH_M: Final[float] = 1e-8


# Reduced Planck constant (dimensionless in this simulation)
HBAR: Final[float] = 1.0

# Ratio of Planck constant to Boltzmann constant (K⋅ps)
# (1.054572 / 1.380649) × 10^2
HKB: Final[float] = (1.054572 / 1.380649) * 1e2


# Lattice constant (Angstrom)
LATTICE_CONSTANT: Final[float] = 5.431

# Phonon dispersion coefficients for TA (Transverse Acoustic) branch
# Unit: 10^13 rad/s
C10_TA: Final[float] = 5.23
C20_TA: Final[float] = -2.26

# Phonon dispersion coefficients for LA (Longitudinal Acoustic) branch  
# Unit: 10^13 rad/s
C11_LA: Final[float] = 9.01
C21_LA: Final[float] = -2.0

IMPURITY_SCATTERING_COEFF: Final[float] = 1.498e7  # A_i (adjusted for ω in 10^13 rad/s)
NORMAL_SCATTERING_B_L: Final[float] = 1.18e2  # B_L (adjusted for ω in 10^13 rad/s)
NORMAL_SCATTERING_B_T: Final[float] = 8.708  # B_T (adjusted for ω in 10^13 rad/s)
UMKLAPP_SCATTERING_B_U: Final[float] = 2.89e8  # B_U (adjusted for ω in 10^13 rad/s)


HOT_BC_X_MIN: Final[float] = 0.1
HOT_BC_X_MAX: Final[float] = 0.9

# Cold side walls extend from y=0 to y=COLD_SIDE_Y_FRACTION * y_max
# Above this fraction is the "corner" region near the hot boundary
COLD_SIDE_Y_FRACTION: Final[float] = 0.9


SI_PHONON_MFP_BASE_UNITS: Final[float] = 5.0

VT_UNIT_CONVERSION: Final[float] = 1e11


# Adam optimizer parameters
ADAM_BETA1: Final[float] = 0.9
ADAM_BETA2: Final[float] = 0.99
ADAM_EPSILON: Final[float] = 1e-10

# Loss normalization floor: prevents division by zero when normalizing
# per-mode BTE/BC/IC losses by output_scale² or energy term magnitudes
LOSS_NORM_EPSILON: Final[float] = 1e-8

# Logging frequencies (epochs)
PRINT_FREQUENCY: Final[int] = 200
CHECKPOINT_FREQUENCY: Final[int] = 200

# Plot settings
PLOT_DPI: Final[int] = 300


def energy_weight(Kn: float, dim: int = 1) -> float:
    """Energy conservation loss weight from kinetic theory + empirical floor."""
    ew_kinetic = Kn ** 2 / (1.0 + Kn ** 2)
    return max(ew_kinetic, min(1.0, 10.0 * Kn ** 2))

