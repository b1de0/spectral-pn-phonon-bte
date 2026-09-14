"""Neural network model for PINN-pBTE simulation."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Net(nn.Module):
    """Fully connected neural network with SiLU (Swish) activation."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int,
        use_checkpointing: bool = False
    ) -> None:
        """Initialize the neural network."""
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()
        self.use_checkpointing = use_checkpointing

    def _hidden_block(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """Single hidden layer forward pass (for checkpointing)."""
        return self.activation(layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        output = self.activation(self.input_layer(x))

        if self.use_checkpointing and self.training:
            # Use gradient checkpointing for hidden layers
            for layer in self.hidden_layers:
                output = checkpoint(self._hidden_block, output, layer, use_reentrant=False)
        else:
            for layer in self.hidden_layers:
                output = self.activation(layer(output))

        output = self.output_layer(output)
        return output


class PirateNet(nn.Module):
    """Adaptive residual network with learnable skip gates (Wang et al., 2024)."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.SiLU()
        self.use_checkpointing = use_checkpointing

    def _hidden_block(self, x: torch.Tensor, layer: nn.Linear, alpha: nn.Parameter) -> torch.Tensor:
        """Single gated residual block (for checkpointing)."""
        a = torch.sigmoid(alpha)
        return a * self.activation(layer(x)) + (1 - a) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_layer(x))

        if self.use_checkpointing and self.training:
            for layer, alpha in zip(self.hidden_layers, self.alphas):
                h = checkpoint(self._hidden_block, h, layer, alpha, use_reentrant=False)
        else:
            for layer, alpha in zip(self.hidden_layers, self.alphas):
                a = torch.sigmoid(alpha)
                h = a * self.activation(layer(h)) + (1 - a) * h

        return self.output_layer(h)


class SeparableNet(nn.Module):

    def __init__(
        self,
        rank: int = 4,
        hidden_dim: int = 30,
        num_layers: int = 4,
        n_inputs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.n_inputs = n_inputs
        self.branches = nn.ModuleList()
        for _r in range(rank):
            branch_nets = nn.ModuleList()
            for _d in range(n_inputs):
                layers = [nn.Linear(1, hidden_dim), nn.SiLU()]
                for _ in range(num_layers - 1):
                    layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
                layers.append(nn.Linear(hidden_dim, 1))
                branch_nets.append(nn.Sequential(*layers))
            self.branches.append(branch_nets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sum of rank-1 products."""
        result = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for branch_nets in self.branches:
            product = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            for d, net in enumerate(branch_nets):
                product = product * net(x[:, d:d + 1])
            result = result + product
        return result
