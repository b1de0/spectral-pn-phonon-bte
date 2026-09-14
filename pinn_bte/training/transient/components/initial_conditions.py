"""Initial condition abstractions for transient BTE trainers."""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from pinn_bte.training.transient.components.types import ICType


class InitialCondition(ABC):
    """Abstract base class for initial conditions."""

    @property
    @abstractmethod
    def ic_type(self) -> ICType:
        """Return the IC type enum."""
        pass

    @abstractmethod
    def get_T_target(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute target temperature at t=0."""
        pass

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        x: torch.Tensor = None,
        y: Optional[torch.Tensor] = None,
        device: torch.device = None,
        include_distribution: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IC loss for temperature and distribution."""
        if x is None:
            raise ValueError("x tensor required for IC")
        if device is None:
            device = x.device

        n_ic = x.shape[0]
        t_zero = torch.zeros(n_ic, 1, device=device)

        # Temperature IC
        if y is not None:
            T_pred = net_T(torch.cat([x, y, t_zero], dim=1))
        else:
            T_pred = net_T(torch.cat([x, t_zero], dim=1))

        T_target = self.get_T_target(x, y)
        loss_T = torch.mean((T_pred - T_target) ** 2)

        # Distribution IC (default: n=0 at t=0)
        loss_n = torch.tensor(0.0, device=device)
        if include_distribution and net_n is not None:
            loss_n = self._compute_distribution_ic_loss(
                net_n, x, y, t_zero, device, **kwargs
            )

        return loss_T, loss_n

    def _compute_distribution_ic_loss(
        self,
        net_n: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        t_zero: torch.Tensor,
        device: torch.device,
        mu: torch.Tensor = None,
        k_samples: torch.Tensor = None,
        p_samples: torch.Tensor = None,
        n_angular_sample: int = 4,
        n_modes_sample: int = 4,
        Ns: int = None,
        Nk: int = None,
        Np: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute IC loss for distribution function (n=0 at t=0)."""
        if mu is None:
            return torch.tensor(0.0, device=device)

        n_ic = x.shape[0]
        loss_n = torch.tensor(0.0, device=device)

        # Sample a subset of modes and angles
        n_modes = min(n_modes_sample, Nk * Np if Nk and Np else 4)
        n_angular = min(n_angular_sample, Ns if Ns else 4)

        for i_mode in range(n_modes):
            if k_samples is not None and p_samples is not None:
                k_val = k_samples[i_mode % len(k_samples)].item()
                p_val = p_samples[i_mode % len(p_samples)].item()
            else:
                k_val = 0.5
                p_val = 0.0

            for i_s in range(n_angular):
                mu_val = mu[i_s % len(mu)].item()

                # Build input to n network
                if y is not None:
                    # 2D: (x, y, t, μ, k, p) or similar
                    n_input = torch.cat(
                        [
                            x,
                            y,
                            t_zero,
                            torch.full((n_ic, 1), mu_val, device=device),
                            torch.full((n_ic, 1), k_val, device=device),
                            torch.full((n_ic, 1), p_val, device=device),
                        ],
                        dim=1,
                    )
                else:
                    # 1D: (x, t, μ) for gray or (x, t, μ, k, p) for nongray
                    if Nk is not None and Nk > 1:
                        n_input = torch.cat(
                            [
                                x,
                                t_zero,
                                torch.full((n_ic, 1), mu_val, device=device),
                                torch.full((n_ic, 1), k_val, device=device),
                                torch.full((n_ic, 1), p_val, device=device),
                            ],
                            dim=1,
                        )
                    else:
                        # Gray model: (x, t, μ)
                        n_input = torch.cat(
                            [
                                x,
                                t_zero,
                                torch.full((n_ic, 1), mu_val, device=device),
                            ],
                            dim=1,
                        )

                n_pred = net_n(n_input)
                loss_n = loss_n + torch.mean(n_pred**2)

        return loss_n / (n_modes * n_angular)


class UniformIC(InitialCondition):
    """Uniform initial temperature: T(x,0) = T_initial."""

    def __init__(self, T_initial: float = 0.0):
        """Initialize uniform IC."""
        self.T_initial = T_initial

    @property
    def ic_type(self) -> ICType:
        return ICType.UNIFORM

    def get_T_target(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return uniform temperature."""
        return torch.full_like(x, self.T_initial)


class CosineIC(InitialCondition):
    """Cosine initial temperature: T(x,0) = A * cos(k*x)."""

    def __init__(self, amplitude: float = 1.0, k: float = 2 * np.pi):
        """Initialize cosine IC."""
        self.amplitude = amplitude
        self.k = k

    @property
    def ic_type(self) -> ICType:
        return ICType.COSINE

    def get_T_target(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return cosine temperature profile."""
        return self.amplitude * torch.cos(self.k * x)


class SteadyStateIC(InitialCondition):
    """Initial condition loaded from a steady-state solution."""

    def __init__(
        self,
        T_field: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        n_field: Optional[
            Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]
        ] = None,
    ):
        """Initialize steady-state IC."""
        self.T_field = T_field
        self.n_field = n_field

    @property
    def ic_type(self) -> ICType:
        return ICType.STEADY_STATE

    def get_T_target(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return steady-state temperature."""
        return self.T_field(x, y)

    def _compute_distribution_ic_loss(
        self,
        net_n: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        t_zero: torch.Tensor,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        """Compute IC loss for distribution using steady-state target."""
        if self.n_field is None:
            # Fall back to default (n=0)
            return super()._compute_distribution_ic_loss(
                net_n, x, y, t_zero, device, **kwargs
            )

        return super()._compute_distribution_ic_loss(
            net_n, x, y, t_zero, device, **kwargs
        )
