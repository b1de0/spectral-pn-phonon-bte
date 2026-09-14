"""Boundary condition abstractions for transient BTE trainers."""

from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from pinn_bte.training.transient.components.types import BCType


class TransientBC(ABC):
    """Abstract base class for transient boundary conditions."""

    @property
    @abstractmethod
    def bc_type(self) -> BCType:
        """Return the BC type enum."""
        pass

    @abstractmethod
    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the BC loss contribution."""
        pass


class PeriodicBC(TransientBC):
    """Periodic boundary condition: T(0,t) = T(L,t)."""

    @property
    def bc_type(self) -> BCType:
        return BCType.PERIODIC

    def __init__(self, include_distribution: bool = True):
        """Initialize periodic BC."""
        self.include_distribution = include_distribution

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        t: torch.Tensor = None,
        mu: torch.Tensor = None,
        w: torch.Tensor = None,
        device: torch.device = None,
        Ns: int = None,
        Nt: int = None,
        v_tau_L: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute periodic BC loss."""
        if t is None:
            raise ValueError("t tensor required for periodic BC")
        if device is None:
            device = t.device

        # Flatten t for temperature network (Nt,1)
        if t.shape[0] > Nt:
            # t is already expanded for (Nt*Ns,)
            t_T = t[::Ns]  # Take every Ns-th sample
        else:
            t_T = t

        n_t = t_T.shape[0]

        # Temperature at left (x=0) and right (x=1)
        T_left = net_T(torch.cat([torch.zeros(n_t, 1, device=device), t_T], dim=1))
        T_right = net_T(torch.cat([torch.ones(n_t, 1, device=device), t_T], dim=1))

        loss = torch.mean((T_left - T_right) ** 2)

        # Distribution periodicity (if requested and network provided)
        if self.include_distribution and net_n is not None and mu is not None:
            if Ns is None or Nt is None:
                raise ValueError("Ns and Nt required for distribution BC")

            # Expand t for angular dimensions
            tb_exp = t_T.repeat(1, Ns).reshape(-1, 1)
            mub = mu.repeat(Nt, 1)

            # Left boundary
            l_in = torch.cat([torch.zeros_like(tb_exp), tb_exp, mub], dim=1)
            nl_raw = net_n(l_in) * v_tau_L

            # Right boundary
            r_in = torch.cat([torch.ones_like(tb_exp), tb_exp, mub], dim=1)
            nr_raw = net_n(r_in) * v_tau_L

            # Apply zero-mean constraint
            nl_reshaped = nl_raw.reshape(-1, Ns)
            sum_nl = torch.matmul(nl_reshaped, w).reshape(-1, 1) / (4 * np.pi)
            nl = nl_raw - sum_nl.repeat(1, Ns).reshape(-1, 1)

            nr_reshaped = nr_raw.reshape(-1, Ns)
            sum_nr = torch.matmul(nr_reshaped, w).reshape(-1, 1) / (4 * np.pi)
            nr = nr_raw - sum_nr.repeat(1, Ns).reshape(-1, 1)

            # Periodic BC on distribution: f(0) = f(1) → n + T = n + T
            loss = loss + torch.mean((nr + T_right - nl - T_left) ** 2)

        return loss


class IsothermalBC(TransientBC):
    """Isothermal (Dirichlet) boundary condition: T = T_target."""

    def __init__(self, T_target: float, bc_type_value: BCType = BCType.COLD):
        """Initialize isothermal BC."""
        self._bc_type = bc_type_value
        self.T_target = T_target

    @property
    def bc_type(self) -> BCType:
        return self._bc_type

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        t: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute isothermal BC loss."""
        if x is None or t is None:
            raise ValueError("x and t tensors required for isothermal BC")

        # Build input based on dimensionality
        if y is not None:
            # 2D case
            T_pred = net_T(torch.cat([x, y, t], dim=1))
        else:
            # 1D case
            T_pred = net_T(torch.cat([x, t], dim=1))

        return torch.mean((T_pred - self.T_target) ** 2)


class AdiabaticBC(TransientBC):
    """Adiabatic (Neumann) boundary condition: dT/dn = 0."""

    @property
    def bc_type(self) -> BCType:
        return BCType.INSULATED

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        t: torch.Tensor = None,
        normal_dir: str = "x",
        **kwargs,
    ) -> torch.Tensor:
        """Compute adiabatic BC loss."""
        if x is None or t is None:
            raise ValueError("x and t tensors required for adiabatic BC")

        x = x.requires_grad_(True)

        if y is not None:
            y = y.requires_grad_(True)
            T = net_T(torch.cat([x, y, t], dim=1))

            if normal_dir in ("x", "-x"):
                dT_dn = torch.autograd.grad(
                    T, x, torch.ones_like(T), create_graph=True
                )[0]
            else:  # y or -y
                dT_dn = torch.autograd.grad(
                    T, y, torch.ones_like(T), create_graph=True
                )[0]
        else:
            T = net_T(torch.cat([x, t], dim=1))
            dT_dn = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]

        return torch.mean(dT_dn**2)


class GaussianBC(TransientBC):
    """Gaussian temperature profile boundary condition."""

    def __init__(
        self,
        T_c: float = 0.0,
        delta_T: float = 1.0,
        x0: float = 0.5,
        sigma: float = 1 / 6,
    ):
        """Initialize Gaussian BC."""
        self.T_c = T_c
        self.delta_T = delta_T
        self.x0 = x0
        self.sigma = sigma

    @property
    def bc_type(self) -> BCType:
        return BCType.GAUSSIAN

    def get_target(self, x: torch.Tensor) -> torch.Tensor:
        """Compute target temperature profile."""
        return self.T_c + self.delta_T * torch.exp(
            -((x - self.x0) ** 2) / (2 * self.sigma**2)
        )

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        t: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Gaussian BC loss."""
        if x is None or t is None:
            raise ValueError("x and t tensors required for Gaussian BC")

        # Build input based on dimensionality
        if y is not None:
            T_pred = net_T(torch.cat([x, y, t], dim=1))
        else:
            T_pred = net_T(torch.cat([x, t], dim=1))

        T_target = self.get_target(x)
        return torch.mean((T_pred - T_target) ** 2)


class CompositeBCLoss:
    """Combines multiple boundary conditions into a single loss."""

    def __init__(self, bcs: list[TransientBC], weights: Optional[list[float]] = None):
        """Initialize composite BC."""
        self.bcs = bcs
        self.weights = weights or [1.0] * len(bcs)

    def compute_loss(
        self,
        net_T: nn.Module,
        net_n: Optional[nn.Module] = None,
        bc_kwargs: list[dict] = None,
    ) -> torch.Tensor:
        """Compute total BC loss."""
        if bc_kwargs is None:
            bc_kwargs = [{}] * len(self.bcs)

        total_loss = torch.tensor(0.0, device=next(net_T.parameters()).device)
        for bc, weight, kwargs in zip(self.bcs, self.weights, bc_kwargs):
            total_loss = total_loss + weight * bc.compute_loss(net_T, net_n, **kwargs)

        return total_loss
