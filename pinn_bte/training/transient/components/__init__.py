"""Composable components for transient BTE trainers."""

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
    CompositeBCLoss,
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
    create_ic,
    create_heat_source,
    create_bc_pair,
)

__all__ = [
    # Types and enums
    "BCType",
    "ICType",
    "HeatSourceType",
    "get_transient_wandb_tags",
    "get_transient_wandb_group",
    "build_transient_run_name",
    # Boundary conditions
    "TransientBC",
    "PeriodicBC",
    "IsothermalBC",
    "AdiabaticBC",
    "GaussianBC",
    "CompositeBCLoss",
    # Initial conditions
    "InitialCondition",
    "UniformIC",
    "CosineIC",
    "SteadyStateIC",
    # Heat sources
    "HeatSource",
    "NoHeatSource",
    "StepHeatSource",
    "PulseHeatSource",
    "PeriodicHeatSource",
    # Factories
    "create_bc",
    "create_ic",
    "create_heat_source",
    "create_bc_pair",
]
