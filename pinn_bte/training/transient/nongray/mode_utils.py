"""Pre-computed per-mode constants for nongray trainers."""

from dataclasses import dataclass
from typing import List

import torch

from pinn_bte.config.physics import LOSS_NORM_EPSILON


@dataclass
class ModeDescriptor:
    """Pre-computed constants for a single phonon mode (branch, frequency)."""

    p: int
    ik: int
    v_star: float
    tau_star_ref: float
    k_val: float
    p_val: float
    output_scale: float
    scale_sq: float


def precompute_modes(
    vk_norm: torch.Tensor,
    tau_norm: torch.Tensor,
    k_norm: torch.Tensor,
    Nk: int,
    Np: int,
    Kn_eff: float,
) -> List[ModeDescriptor]:
    """Pre-compute per-mode constants to avoid repeated computation in loops."""
    modes = []
    for mode_idx in range(Np * Nk):
        p = mode_idx // Nk
        ik = mode_idx % Nk
        v_star = vk_norm[p, ik].item()
        tau_star_ref = tau_norm[p, ik].item()
        k_val = k_norm[ik].item() if k_norm.dim() == 1 else k_norm[ik, 0].item()
        p_val = float(p) / (Np - 1) if Np > 1 else 0.0
        output_scale = v_star * tau_star_ref * Kn_eff
        scale_sq = max(output_scale ** 2, LOSS_NORM_EPSILON)
        modes.append(ModeDescriptor(
            p=p, ik=ik, v_star=v_star, tau_star_ref=tau_star_ref,
            k_val=k_val, p_val=p_val, output_scale=output_scale, scale_sq=scale_sq,
        ))
    return modes
