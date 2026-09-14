"""Factory functions for creating BC, IC, and heat source components."""

from typing import Any, Optional, Callable

from pinn_bte.training.transient.components.types import BCType, ICType, HeatSourceType
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


def create_bc(bc_type: BCType, **kwargs: Any) -> TransientBC:
    """Factory for creating boundary condition components."""
    if bc_type == BCType.PERIODIC:
        return PeriodicBC(
            include_distribution=kwargs.get("include_distribution", True)
        )

    elif bc_type == BCType.HOT:
        T_hot = kwargs.get("T_hot", 1.0)
        return IsothermalBC(T_target=T_hot, bc_type_value=BCType.HOT)

    elif bc_type == BCType.COLD:
        T_cold = kwargs.get("T_cold", 0.0)
        return IsothermalBC(T_target=T_cold, bc_type_value=BCType.COLD)

    elif bc_type == BCType.INSULATED:
        return AdiabaticBC()

    elif bc_type == BCType.GAUSSIAN:
        return GaussianBC(
            T_c=kwargs.get("T_c", 0.0),
            delta_T=kwargs.get("delta_T", 1.0),
            x0=kwargs.get("x0", 0.5),
            sigma=kwargs.get("sigma", 1 / 6),
        )

    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def create_ic(ic_type: ICType, **kwargs: Any) -> InitialCondition:
    """Factory for creating initial condition components."""
    if ic_type == ICType.UNIFORM:
        return UniformIC(T_initial=kwargs.get("T_initial", 0.0))

    elif ic_type == ICType.COSINE:
        import numpy as np

        return CosineIC(
            amplitude=kwargs.get("amplitude", 1.0),
            k=kwargs.get("k", 2 * np.pi),
        )

    elif ic_type == ICType.STEADY_STATE:
        T_field = kwargs.get("T_field")
        if T_field is None:
            raise ValueError("STEADY_STATE IC requires T_field callable")
        return SteadyStateIC(
            T_field=T_field,
            n_field=kwargs.get("n_field"),
        )

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")


def create_heat_source(hs_type: HeatSourceType, **kwargs: Any) -> HeatSource:
    """Factory for creating heat source components."""
    if hs_type == HeatSourceType.NONE:
        return NoHeatSource()

    elif hs_type == HeatSourceType.STEP:
        return StepHeatSource()

    elif hs_type == HeatSourceType.PULSE:
        return PulseHeatSource(
            Q0=kwargs.get("Q0", 1.0),
            x0=kwargs.get("x0", 0.5),
            y0=kwargs.get("y0", 0.5),
            t0=kwargs.get("t0", 0.1),
            sigma_x=kwargs.get("sigma_x", 0.1),
            sigma_y=kwargs.get("sigma_y"),
            sigma_t=kwargs.get("sigma_t", 0.05),
        )

    elif hs_type == HeatSourceType.PERIODIC_TIME:
        import numpy as np

        return PeriodicHeatSource(
            Q0=kwargs.get("Q0", 1.0),
            omega=kwargs.get("omega", 2 * np.pi),
            phase=kwargs.get("phase", 0.0),
            x0=kwargs.get("x0", 0.5),
            y0=kwargs.get("y0", 0.5),
            sigma_x=kwargs.get("sigma_x", 0.1),
            sigma_y=kwargs.get("sigma_y"),
            uniform_spatial=kwargs.get("uniform_spatial", False),
        )

    else:
        raise ValueError(f"Unknown heat source type: {hs_type}")


def create_bc_pair(
    bc_left_type: BCType,
    bc_right_type: BCType,
    **kwargs: Any,
) -> tuple[TransientBC, TransientBC]:
    """Create a pair of boundary conditions for left and right boundaries."""
    left_kwargs = kwargs.get("bc_left_kwargs", kwargs)
    right_kwargs = kwargs.get("bc_right_kwargs", kwargs)

    return create_bc(bc_left_type, **left_kwargs), create_bc(bc_right_type, **right_kwargs)
