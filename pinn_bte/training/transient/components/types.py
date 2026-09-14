"""Type definitions and enums for transient BTE trainers."""

from enum import Enum
from typing import Any, List, Optional


class BCType(Enum):
    """Boundary condition types for transient BTE."""

    PERIODIC = "periodic"  # T(0,t) = T(L,t), n(0,t) = n(L,t)
    HOT = "hot"  # T = T_hot (Dirichlet, fixed high temperature)
    COLD = "cold"  # T = T_cold (Dirichlet, fixed low temperature)
    INSULATED = "insulated"  # dT/dn = 0 (Neumann, adiabatic)
    GAUSSIAN = "gaussian"  # T = Gaussian profile (used for hotspots)


class ICType(Enum):
    """Initial condition types for transient BTE."""

    UNIFORM = "uniform"  # T(x,0) = T_ref (constant)
    COSINE = "cosine"  # T(x,0) = T_ref + A*cos(2πx/L) (TTG setup)
    STEADY_STATE = "steady_state"  # Load from steady-state solution


class HeatSourceType(Enum):
    """Volumetric heat source types for transient BTE."""

    NONE = "none"  # No volumetric heat source
    STEP = "step"  # Sudden heating at t=0 (implemented via BC)
    PULSE = "pulse"  # Gaussian pulse in time and space
    PERIODIC_TIME = "periodic_time"  # Periodic heating in time


def get_transient_wandb_tags(
    dimension: str,
    model: str,
    bc_left: BCType,
    bc_right: BCType,
    ic_type: ICType,
    heat_source: HeatSourceType,
    delta_T_regime: str = "small_dt",
    bc_top: Optional[BCType] = None,
    bc_bottom: Optional[BCType] = None,
    geometry: Optional[str] = None,
    **extra: Any,
) -> List[str]:
    """Create flat W&B tags for transient experiments."""
    tags = ["transient", dimension, model, delta_T_regime]

    if geometry:
        tags.append(geometry)

    tags.append(f"ic={ic_type.value}")

    # Summarize BC: if symmetric, single tag; otherwise enumerate
    bc_set = {bc_left.value, bc_right.value}
    if bc_top is not None:
        bc_set.add(bc_top.value)
    if bc_bottom is not None:
        bc_set.add(bc_bottom.value)
    for bc in sorted(bc_set):
        tags.append(f"bc={bc}")

    if heat_source != HeatSourceType.NONE:
        tags.append(f"source={heat_source.value}")

    # Physics params from **extra as key=value tags
    _TAG_FORMATTERS = {
        "L_um": "L={}um",
        "delta_T_K": "dT={}K",
        "Kn": "Kn={}",
        "Kn_eff": "Kn={}",
        "eta": "eta={}",
    }
    for key, value in extra.items():
        if key == "has_beta_network":
            if str(value).lower() == "true":
                tags.append("beta_net")
            continue
        fmt = _TAG_FORMATTERS.get(key)
        if fmt:
            tags.append(fmt.format(value))
        else:
            tags.append(f"{key}={value}")

    return tags


def get_transient_wandb_group(
    dimension: str,
    model: str,
    delta_T_regime: str = "small_dt",
    geometry: Optional[str] = None,
) -> str:
    """Build W&B group name for transient experiments."""
    parts = ["transient", dimension, model]
    if geometry:
        parts.append(geometry)
    parts.append(delta_T_regime)
    return "_".join(parts)


def build_transient_run_name(
    dimension: str,
    model: str,
    delta_T_regime: str = "small_dt",
    geometry: Optional[str] = None,
    L_um: Optional[str] = None,
    delta_T_K: Optional[str] = None,
    Kn: Optional[str] = None,
    eta: Optional[str] = None,
) -> str:
    """Build human-readable W&B run name for transient experiments."""
    model_short = "ng" if model == "nongray" else model

    parts = ["tr", dimension, model_short]
    if geometry:
        parts.append(geometry)

    # Physics params (most informative first)
    if L_um:
        parts.append(f"L{L_um}um")
    if delta_T_K:
        parts.append(f"dT{delta_T_K}K")
    if Kn:
        parts.append(f"Kn{Kn}")
    if eta:
        parts.append(f"eta{eta}")

    # Add regime suffix only when no physics params make it obvious
    if not L_um and not Kn and not eta:
        dt_short = "lgDT" if delta_T_regime == "large_dt" else "smDT"
        parts.append(dt_short)

    return "_".join(parts)
