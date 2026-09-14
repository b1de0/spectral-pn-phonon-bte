"""Transient trainers: spectral-PN solver, gray control, mode-resolved
sampled-residual control.
"""

from pinn_bte.training.transient.components.types import (
    BCType,
    ICType,
    HeatSourceType,
    get_transient_wandb_tags,
    get_transient_wandb_group,
    build_transient_run_name,
)
from pinn_bte.training.transient.components.boundary_conditions import (
    TransientBC,
    PeriodicBC,
    IsothermalBC,
    AdiabaticBC,
    GaussianBC,
)
from pinn_bte.training.transient.components.initial_conditions import (
    InitialCondition,
    UniformIC,
    CosineIC,
    SteadyStateIC,
)
from pinn_bte.training.transient.components.heat_sources import (
    HeatSource,
    NoHeatSource,
    StepHeatSource,
    PulseHeatSource,
    PeriodicHeatSource,
)
from pinn_bte.training.transient.components.factories import (
    create_bc,
    create_bc_pair,
    create_ic,
    create_heat_source,
)
from pinn_bte.training.transient.gray.gray_1d import Transient1DGrayTrainer
from pinn_bte.training.transient.nongray.nongray_1d_large_dt import (
    Transient1DNongrayLargeDTTrainer,
)

__all__ = [
    "BCType", "ICType", "HeatSourceType",
    "get_transient_wandb_tags", "get_transient_wandb_group",
    "build_transient_run_name",
    "TransientBC", "PeriodicBC", "IsothermalBC", "AdiabaticBC", "GaussianBC",
    "InitialCondition", "UniformIC", "CosineIC", "SteadyStateIC",
    "HeatSource", "NoHeatSource", "StepHeatSource", "PulseHeatSource",
    "PeriodicHeatSource",
    "create_bc", "create_bc_pair", "create_ic", "create_heat_source",
    "Transient1DGrayTrainer", "Transient1DNongrayLargeDTTrainer",
]
