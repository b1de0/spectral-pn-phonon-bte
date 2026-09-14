"""Heat source abstractions for transient BTE trainers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from pinn_bte.training.transient.components.types import HeatSourceType


class HeatSource(ABC):
    """Abstract base class for volumetric heat sources."""

    @property
    @abstractmethod
    def source_type(self) -> HeatSourceType:
        """Return the heat source type enum."""
        pass

    @abstractmethod
    def get_power(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute volumetric power density Q(x,t)."""
        pass

    def compute_source_term(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute source term for BTE residual."""
        return self.get_power(x, t, y)


class NoHeatSource(HeatSource):
    """No volumetric heat source (Q = 0)."""

    @property
    def source_type(self) -> HeatSourceType:
        return HeatSourceType.NONE

    def get_power(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return zero power density."""
        return torch.zeros_like(x)


class StepHeatSource(HeatSource):
    """Step function heat source at t=0."""

    @property
    def source_type(self) -> HeatSourceType:
        return HeatSourceType.STEP

    def get_power(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return zero (step heating typically via BC)."""
        return torch.zeros_like(x)


class PulseHeatSource(HeatSource):
    """Gaussian pulse heat source in time and space."""

    def __init__(
        self,
        Q0: float = 1.0,
        x0: float = 0.5,
        t0: float = 0.1,
        sigma_x: float = 0.1,
        sigma_t: float = 0.05,
        y0: float = 0.5,
        sigma_y: Optional[float] = None,
    ):
        """Initialize Gaussian pulse heat source."""
        self.Q0 = Q0
        self.x0 = x0
        self.y0 = y0
        self.t0 = t0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y if sigma_y is not None else sigma_x
        self.sigma_t = sigma_t

    @property
    def source_type(self) -> HeatSourceType:
        return HeatSourceType.PULSE

    def get_power(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Gaussian pulse power density."""
        # Spatial Gaussian
        spatial = torch.exp(-((x - self.x0) ** 2) / (2 * self.sigma_x**2))

        if y is not None:
            spatial = spatial * torch.exp(
                -((y - self.y0) ** 2) / (2 * self.sigma_y**2)
            )

        # Temporal Gaussian
        temporal = torch.exp(-((t - self.t0) ** 2) / (2 * self.sigma_t**2))

        return self.Q0 * spatial * temporal


class PeriodicHeatSource(HeatSource):
    """Periodic (sinusoidal) heat source in time."""

    def __init__(
        self,
        Q0: float = 1.0,
        omega: float = 2 * np.pi,
        phase: float = 0.0,
        x0: float = 0.5,
        sigma_x: float = 0.1,
        uniform_spatial: bool = False,
        y0: float = 0.5,
        sigma_y: Optional[float] = None,
    ):
        """Initialize periodic heat source."""
        self.Q0 = Q0
        self.omega = omega
        self.phase = phase
        self.x0 = x0
        self.y0 = y0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y if sigma_y is not None else sigma_x
        self.uniform_spatial = uniform_spatial

    @property
    def source_type(self) -> HeatSourceType:
        return HeatSourceType.PERIODIC_TIME

    def get_power(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute periodic power density."""
        # Temporal sinusoid
        temporal = torch.sin(self.omega * t + self.phase)

        # Spatial profile
        if self.uniform_spatial:
            spatial = torch.ones_like(x)
        else:
            spatial = torch.exp(-((x - self.x0) ** 2) / (2 * self.sigma_x**2))
            if y is not None:
                spatial = spatial * torch.exp(
                    -((y - self.y0) ** 2) / (2 * self.sigma_y**2)
                )

        return self.Q0 * spatial * temporal
