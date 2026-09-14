"""Gray 1D transient BTE trainer."""

import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pinn_bte.models import Net
from pinn_bte.config.physics import PRINT_FREQUENCY
from pinn_bte.training.core.wandb_mixin import WandbMixin
from pinn_bte.training.transient.components.types import (
    BCType,
    ICType,
    HeatSourceType,
    get_transient_wandb_tags,
    get_transient_wandb_group,
    build_transient_run_name,
)
from pinn_bte.training.transient.components.factories import create_bc, create_ic, create_heat_source
from pinn_bte.utils.logging import (
    get_logger,
    log_training_start,
    log_training_complete,
    console,
    setup_file_logging,
    close_file_logging,
)

logger = get_logger(__name__)


class Transient1DGrayTrainer(WandbMixin):
    """Trainer for time-dependent gray phonon BTE."""

    def __init__(
        self,
        eta: float = 0.6,
        window_tau: float = 3.0,
        bc_type: BCType = BCType.PERIODIC,
        ic_type: ICType = ICType.COSINE,
        heat_source: HeatSourceType = HeatSourceType.NONE,
        epochs: int = 8000,
        learning_rate: float = 4e-3,
        Nx: int = 60,
        Nt: int = 60,
        Ns: int = 16,
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
        residual_chunks: int = 1,
    ):
        """Initialize gray 1D transient trainer."""
        if residual_chunks < 1:
            raise ValueError(
                f"residual_chunks must be >= 1, got {residual_chunks}")
        if residual_chunks > Nx * Nt:
            raise ValueError(
                f"residual_chunks={residual_chunks} exceeds the number of "
                f"(x,t) collocation points Nx*Nt={Nx * Nt}")
        self.residual_chunks = int(residual_chunks)
        # Physical parameters (dimensionless)
        self.v = 1.0
        self.tau = 1.0
        self.eta = eta
        self.L = 2 * np.pi * self.v * self.tau / eta  # ~10.47 for η=0.6
        self.Lt = window_tau * self.tau  # Time range: window_tau in units of tau (Zhou benchmark: 3)

        # Mesh parameters
        self.Nx = Nx
        self.Nt = Nt
        self.Ns = Ns

        # Training parameters
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output
        self.output_dir = output_dir or Path("runs/transient_1d_gray")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"gray1d_{time.strftime('%Y%m%d_%H%M%S')}"

        # Store BC/IC/heat source types for reference
        self.bc_type = bc_type
        self.ic_type = ic_type
        self.heat_source_type = heat_source

        # Create composable components via factories
        self.bc = create_bc(bc_type, include_distribution=True)
        self.ic = create_ic(ic_type, amplitude=1.0, k=2 * np.pi)
        self.heat_source = create_heat_source(heat_source)

        # Initialize networks (for gray BTE)
        self._init_networks()

        # Prepare mesh and data
        self._prepare_data()

        # Loss names for tracking
        self.loss_names = ['bte', 'heat_flux', f'{bc_type.value}_bc', 'ic_fneq', 'ic_T']

        # Initialize W&B with standardized tags
        _Kn = f'{eta/(2*np.pi):.3f}'
        self.__init_wandb__(
            experiment_name=get_transient_wandb_group('1d', 'gray', 'small_dt'),
            tags=get_transient_wandb_tags(
                dimension='1d',
                model='gray',
                bc_left=bc_type,
                bc_right=bc_type,
                ic_type=ic_type,
                heat_source=heat_source,
                delta_T_regime='small_dt',
                eta=str(eta),
                Kn=_Kn,
            ),
            run_name=build_transient_run_name(
                '1d', 'gray', 'small_dt', Kn=_Kn, eta=str(eta),
            ),
        )

    def _init_networks(self):
        """Initialize neural networks for gray BTE."""
        # net0: (x, t, μ) → n (non-equilibrium deviation)
        # 5 hidden layers, 30 neurons each
        self.net0 = Net(
            input_dim=3,  # (x, t, μ)
            hidden_dim=30,
            num_layers=5,
        ).to(self.device)

        # net1: (x, t) → T (pseudo-temperature)
        # 5 hidden layers, 30 neurons each
        self.net1 = Net(
            input_dim=2,  # (x, t)
            hidden_dim=30,
            num_layers=5,
        ).to(self.device)

        # Optimizers
        self.optimizer0 = optim.Adam(
            self.net0.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )
        self.optimizer1 = optim.Adam(
            self.net1.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )

        # Weight initialization (Kaiming)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.net0.apply(init_weights)
        self.net1.apply(init_weights)

    def _prepare_data(self):
        """Prepare mesh and quadrature data."""
        # Spatial grid (normalized to [0, 1])
        xm = np.linspace(0, 1, self.Nx).reshape(-1, 1)
        tm = np.linspace(0, 1, self.Nt).reshape(-1, 1)

        # Create mesh grid
        x, t = np.meshgrid(xm.flatten(), tm.flatten())
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)

        # Boundary and IC samples
        xi = np.linspace(0, 1, self.Nx).reshape(-1, 1)  # IC spatial
        tb = np.linspace(0, 1, self.Nt + 2)[1:self.Nt + 1].reshape(-1, 1)  # BC time

        mu, w = np.polynomial.legendre.leggauss(self.Ns)
        mu = mu.reshape(-1, 1)
        w = w.reshape(-1, 1) * 2 * np.pi  # Weight includes 2π factor

        # Store as tensors
        self.data = {
            'x': torch.FloatTensor(x).to(self.device),
            't': torch.FloatTensor(t).to(self.device),
            'mu': torch.FloatTensor(mu).to(self.device),
            'w': torch.FloatTensor(w).to(self.device),
            'xi': torch.FloatTensor(xi).to(self.device),
            'tb': torch.FloatTensor(tb).to(self.device),
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
        w: torch.Tensor,
        mub: torch.Tensor,
        mui: torch.Tensor,
        tb: torch.Tensor,
        xi: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute all loss terms following BTE formulation."""
        # Enable gradients
        x.requires_grad = True
        t.requires_grad = True

        Nx, Nt, Ns = self.Nx, self.Nt, self.Ns
        L, Lt = self.L, self.Lt
        v, tau = self.v, self.tau

        # ========== INTERIOR POINTS ==========
        # Raw network outputs
        n_raw = self.net0(torch.cat([x, t, mu], dim=1)) * (v * tau / L)
        T = self.net1(torch.cat([x, t], dim=1))

        # Gradients
        n_x = torch.autograd.grad(n_raw, x, torch.ones_like(x), create_graph=True)[0]
        n_t = torch.autograd.grad(n_raw, t, torch.ones_like(t), create_graph=True)[0]
        T_x = torch.autograd.grad(T, x, torch.ones_like(x), create_graph=True)[0]
        T_t = torch.autograd.grad(T, t, torch.ones_like(t), create_graph=True)[0]

        n_reshaped = n_raw.reshape(-1, Ns)
        sum_n = torch.matmul(n_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        n = n_raw - sum_n.repeat(1, Ns).reshape(-1, 1)

        # Same for gradients
        n_x_reshaped = n_x.reshape(-1, Ns)
        sum_n_x = torch.matmul(n_x_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        n_x = n_x - sum_n_x.repeat(1, Ns).reshape(-1, 1)

        n_t_reshaped = n_t.reshape(-1, Ns)
        sum_n_t = torch.matmul(n_t_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        n_t = n_t - sum_n_t.repeat(1, Ns).reshape(-1, 1)

        # Heat flux divergence: dq/dx
        # q = (1/4π) ∫ μ·n dΩ
        dq_x = torch.matmul((mu * n_x).reshape(-1, Ns), w).reshape(-1, 1)
        dq_t = torch.matmul(T_t.reshape(-1, Ns), w).reshape(-1, 1)

        # ========== PERIODIC BOUNDARY ==========
        # Right boundary (x=1)
        r_in = torch.cat([torch.ones_like(tb), tb, mub], dim=1)
        nr_raw = self.net0(r_in) * (v * tau / L)
        nr_reshaped = nr_raw.reshape(-1, Ns)
        sum_nr = torch.matmul(nr_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        nr = nr_raw - sum_nr.repeat(1, Ns).reshape(-1, 1)
        Tr = self.net1(torch.cat([torch.ones_like(tb), tb], dim=1))

        # Left boundary (x=0)
        l_in = torch.cat([torch.zeros_like(tb), tb, mub], dim=1)
        nl_raw = self.net0(l_in) * (v * tau / L)
        nl_reshaped = nl_raw.reshape(-1, Ns)
        sum_nl = torch.matmul(nl_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        nl = nl_raw - sum_nl.repeat(1, Ns).reshape(-1, 1)
        Tl = self.net1(torch.cat([torch.zeros_like(tb), tb], dim=1))

        # ========== INITIAL CONDITION ==========
        # T(x, 0) = cos(2πx)
        t0_in = torch.cat([xi, torch.zeros_like(xi)], dim=1)
        Ti = self.net1(t0_in)

        # n(x, μ, 0) = 0
        n0_in = torch.cat([xi, torch.zeros_like(xi), mui], dim=1)
        ni_raw = self.net0(n0_in) * (v * tau / L)
        ni_reshaped = ni_raw.reshape(-1, Ns)
        sum_ni = torch.matmul(ni_reshaped, w).reshape(-1, 1) / (4 * np.pi)
        ni = ni_raw - sum_ni.repeat(1, Ns).reshape(-1, 1)

        bte_residual = (n_t + T_t) / Lt + v * mu * (n_x + T_x) / L + n / tau
        loss1 = torch.mean(bte_residual ** 2)

        # Loss 2: Heat flux divergence
        # (∂q/∂t)/Lt + v·(∂q/∂x)/L = 0
        flux_residual = (dq_t / Lt + dq_x * v / L) / 4
        loss2 = torch.mean(flux_residual ** 2)

        # Loss 3: Periodic boundary condition
        # n(0,t) + T(0,t) = n(1,t) + T(1,t)
        bc_residual = (nr + Tr - nl - Tl)
        loss3 = torch.mean(bc_residual ** 2)

        # Loss 4: IC for f_neq
        # n(x, μ, 0) = 0
        loss4 = torch.mean(ni ** 2)

        # Loss 5: IC for temperature
        # T(x, 0) = cos(2πx)
        T_target = torch.cos(xi * 2 * np.pi)
        loss5 = torch.mean((Ti - T_target) ** 2)

        return loss1, loss2, loss3, loss4, loss5

    def _build_residual_chunks(
        self, x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        """Split the expanded interior tensors into `residual_chunks` contiguous
        slices at (x,t)-POINT granularity.
        """
        Ns = self.Ns
        assert x.shape == t.shape == mu.shape, (
            f"interior tensor misalignment: x={x.shape}, t={t.shape}, "
            f"mu={mu.shape}")
        n_rows = x.shape[0]
        assert n_rows % Ns == 0, f"rows {n_rows} not a multiple of Ns={Ns}"
        n_points = n_rows // Ns
        K = self.residual_chunks
        bounds = [round(i * n_points / K) for i in range(K + 1)]
        chunks = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            sl = slice(a * Ns, b * Ns)
            xc, tc, mc = x[sl].clone(), t[sl].clone(), mu[sl].clone()
            assert xc.shape == tc.shape == mc.shape == ((b - a) * Ns, 1), (
                f"chunk misalignment at [{a}:{b}]: {xc.shape}, {tc.shape}, "
                f"{mc.shape}")
            chunks.append((xc, tc, mc, (b - a) / n_points))
        return chunks

    def _chunked_backward(
        self,
        chunks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
        w: torch.Tensor,
        mub: torch.Tensor,
        mui: torch.Tensor,
        tb: torch.Tensor,
        xi: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """One gradient-accumulated epoch over the interior chunks."""
        totals = None
        for c, (xc, tc, mc, share) in enumerate(chunks):
            lc = self.compute_loss(xc, tc, mc, w, mub, mui, tb, xi)
            contrib = (lc[0] + lc[1]) * share
            if c == 0:
                contrib = contrib + lc[2] + lc[3] + lc[4]
            contrib.backward()
            if c == 0:
                totals = [lc[0].detach() * share, lc[1].detach() * share,
                          lc[2].detach(), lc[3].detach(), lc[4].detach()]
            else:
                totals[0] = totals[0] + lc[0].detach() * share
                totals[1] = totals[1] + lc[1].detach() * share
        return tuple(totals)

    def train(self) -> List[List[float]]:
        """Run training loop."""
        # Setup logging
        log_file = setup_file_logging(self.output_dir, self.run_id)
        logger.info(f"Logging to: {log_file}")

        with self.wandb_run():
            return self._training_loop()

    def _training_loop(self) -> List[List[float]]:
        """Core training loop implementation."""
        log_training_start(
            mode="Transient1D_Gray",
            num_epochs=self.epochs,
            device=str(self.device),
            run_id=self.run_id,
        )

        console.print(f"\n[bold cyan]Transient 1D Gray BTE Training[/bold cyan]")
        console.print(f"  η = {self.eta}, L = {self.L:.3f}, Lt = {self.Lt}")
        console.print(f"  Kn = {self.eta / (2 * np.pi):.3f}")
        console.print(f"  Mesh: Nx={self.Nx}, Nt={self.Nt}, Ns={self.Ns}")
        console.print("")

        # Prepare expanded data tensors
        x = self.data['x'].repeat(1, self.Ns).reshape(-1, 1)
        t = self.data['t'].repeat(1, self.Ns).reshape(-1, 1)
        mu = self.data['mu'].repeat(self.Nx * self.Nt, 1)
        w = self.data['w']
        mub = self.data['mu'].repeat(self.Nt, 1)
        mui = self.data['mu'].repeat(self.Nx, 1)
        tb = self.data['tb'].repeat(1, self.Ns).reshape(-1, 1)
        xi = self.data['xi'].repeat(1, self.Ns).reshape(-1, 1)

        use_chunks = self.residual_chunks > 1
        if use_chunks:
            interior_chunks = self._build_residual_chunks(x, t, mu)

        loss_history = []
        min_loss = float('inf')
        start_time = time.time()

        self.net0.train()
        self.net1.train()

        for epoch in range(self.epochs):
            self.optimizer0.zero_grad()
            self.optimizer1.zero_grad()

            if use_chunks:
                # backward happens inside; `losses` are detached scalars
                losses = self._chunked_backward(
                    interior_chunks, w, mub, mui, tb, xi)
                total_loss = sum(losses)
            else:
                losses = self.compute_loss(x, t, mu, w, mub, mui, tb, xi)
                total_loss = sum(losses)

                total_loss.backward()
            self.optimizer0.step()
            self.optimizer1.step()

            # Record losses
            loss_values = [l.item() for l in losses] + [total_loss.item()]
            loss_history.append(loss_values)

            # Log to W&B
            self.log_epoch_metrics(epoch, loss_values[:-1])

            # Console logging
            if epoch % PRINT_FREQUENCY == 0:
                elapsed = time.time() - start_time
                console.print(
                    f"  {elapsed:7.1f}s | Epoch {epoch:5d} | "
                    f"BTE={losses[0].item():.2e} | "
                    f"Flux={losses[1].item():.2e} | "
                    f"BC={losses[2].item():.2e} | "
                    f"IC_n={losses[3].item():.2e} | "
                    f"IC_T={losses[4].item():.2e} | "
                    f"Total={total_loss.item():.2e}"
                )

            # Save best model
            if total_loss.item() < min_loss:
                min_loss = total_loss.item()
                self._save_models()

        # Final save
        elapsed = time.time() - start_time
        self._save_models()
        self._save_loss_history(loss_history)

        # Log artifacts
        self.log_training_artifacts()

        log_training_complete(
            elapsed_time=elapsed,
            output_path=str(self.output_dir),
            model_files=[f"model0_{self.run_id}.pt", f"model1_{self.run_id}.pt"]
        )

        close_file_logging()
        return loss_history

    def _save_models(self):
        """Save model checkpoints."""
        torch.save(self.net0.state_dict(), self.output_dir / f"model0_{self.run_id}.pt")
        torch.save(self.net1.state_dict(), self.output_dir / f"model1_{self.run_id}.pt")

    def _save_loss_history(self, loss_history: List[List[float]]):
        """Save loss history to file."""
        np.savetxt(
            self.output_dir / f"Loss_{self.run_id}.txt",
            np.array(loss_history),
            fmt='%.6f',
            header="BTE HeatFlux PeriodicBC IC_fneq IC_T Total"
        )

    def predict(self, x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict temperature and heat flux fields."""
        self.net0.eval()
        self.net1.eval()

        Nx, Nt = len(x), len(t)
        Ns = self.Ns
        L, Lt = self.L, self.Lt
        v = self.v

        # Create test mesh
        x_grid, t_grid = np.meshgrid(x, t)
        x_flat = x_grid.reshape(-1, 1)
        t_flat = t_grid.reshape(-1, 1)

        # Convert to tensors
        x_t = torch.FloatTensor(x_flat).repeat(1, Ns).reshape(-1, 1).to(self.device)
        t_t = torch.FloatTensor(t_flat).repeat(1, Ns).reshape(-1, 1).to(self.device)
        mu_t = self.data['mu'].repeat(Nx * Nt, 1)
        w = self.data['w']

        x_t.requires_grad = True
        t_t.requires_grad = True

        with torch.no_grad():
            # Get T field (doesn't need angular dim)
            T_input = torch.cat([
                torch.FloatTensor(x_flat).to(self.device),
                torch.FloatTensor(t_flat).to(self.device)
            ], dim=1)
            T = self.net1(T_input)
            T_out = T.reshape(Nt, Nx).cpu().numpy()

        # For heat flux, we need gradients
        with torch.enable_grad():
            n_raw = self.net0(torch.cat([x_t, t_t, mu_t], dim=1)) * (v * self.tau / L)

            n_x = torch.autograd.grad(n_raw, x_t, torch.ones_like(x_t), create_graph=True)[0]

            T_in = torch.cat([x_t[:, 0:1], t_t[:, 0:1]], dim=1)
            T_for_grad = self.net1(T_in)
            T_t_grad = torch.autograd.grad(T_for_grad, t_t, torch.ones_like(T_for_grad), create_graph=True)[0]

            # Zero-mean constraint
            n_reshaped = n_raw.reshape(-1, Ns)
            sum_n = torch.matmul(n_reshaped, w).reshape(-1, 1) / (4 * np.pi)
            n_x_reshaped = n_x.reshape(-1, Ns)
            sum_n_x = torch.matmul(n_x_reshaped, w).reshape(-1, 1) / (4 * np.pi)
            n_x_zm = n_x - sum_n_x.repeat(1, Ns).reshape(-1, 1)

            # Heat flux divergence
            q_x = torch.matmul((mu_t * n_x_zm).reshape(-1, Ns), w).reshape(-1, 1)
            q_t = torch.matmul(T_t_grad.reshape(-1, Ns), w).reshape(-1, 1)

            q_residual = (q_t / Lt + q_x * v / L) / (4 * np.pi)
            q_out = q_residual.reshape(Nt, Nx).cpu().detach().numpy()

        return T_out, q_out

    def test(self, Nx_test: int = 81, Nt_test: int = 81) -> Dict[str, np.ndarray]:
        """Run test evaluation matching standard test procedure."""
        x = np.linspace(0, 1, Nx_test)
        t = np.linspace(0, 1, Nt_test)

        T, q = self.predict(x, t)

        output_file = self.output_dir / f"{self.run_id}_results.npz"
        np.savez(output_file, x=x, t=t, T=T, q=q, Lt=self.Lt)
        console.print(f"[green]Results saved to:[/green] {output_file}")

        return {'x': x, 't': t, 'T': T, 'q': q}


def compare_with_reference(
    trainer_results: Dict[str, np.ndarray],
    reference_path: Path,
    output_dir: Path,
) -> Dict[str, float]:
    """Compare trainer results with reference."""
    import matplotlib.pyplot as plt

    # Load reference
    ref = np.load(reference_path)
    x_ref = ref['x'].flatten()
    t_ref = ref['t'].flatten()
    T_ref = ref['T']

    # Get our results
    x_ours = trainer_results['x']
    t_ours = trainer_results['t']
    T_ours = trainer_results['T']

    # Reshape reference T if needed
    Nx_ref = len(np.unique(x_ref))
    Nt_ref = len(np.unique(t_ref))
    T_ref_grid = T_ref.reshape(Nt_ref, Nx_ref)

    # Interpolate if grids don't match
    if T_ours.shape != T_ref_grid.shape:
        from scipy.interpolate import RectBivariateSpline
        # Interpolate reference to our grid
        x_unique_ref = np.unique(x_ref)
        t_unique_ref = np.unique(t_ref)
        interp = RectBivariateSpline(t_unique_ref, x_unique_ref, T_ref_grid)
        T_ref_interp = interp(t_ours, x_ours)
    else:
        T_ref_interp = T_ref_grid

    # Compute metrics
    rmse = np.sqrt(np.mean((T_ours - T_ref_interp) ** 2))
    max_error = np.max(np.abs(T_ours - T_ref_interp))
    mean_abs_error = np.mean(np.abs(T_ours - T_ref_interp))

    metrics = {
        'rmse': rmse,
        'max_error': max_error,
        'mean_abs_error': mean_abs_error,
    }

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Temperature fields
    # Reference
    ax = axes[0, 0]
    im = ax.contourf(x_ours, t_ours, T_ref_interp, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='T')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Reference')

    # Ours
    ax = axes[0, 1]
    im = ax.contourf(x_ours, t_ours, T_ours, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='T')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Our Implementation')

    # Error
    ax = axes[0, 2]
    im = ax.contourf(x_ours, t_ours, T_ours - T_ref_interp, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Error')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'Error (RMSE={rmse:.4f})')

    # Row 2: Line plots at different times
    ax = axes[1, 0]
    time_indices = [0, len(t_ours) // 4, len(t_ours) // 2, 3 * len(t_ours) // 4, -1]
    for idx in time_indices:
        t_val = t_ours[idx]
        ax.plot(x_ours, T_ref_interp[idx, :], '--', label=f'Ref t={t_val:.2f}')
        ax.plot(x_ours, T_ours[idx, :], '-', label=f'Ours t={t_val:.2f}', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title('T(x) at different times')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # IC comparison
    ax = axes[1, 1]
    T_ic_ref = T_ref_interp[0, :]
    T_ic_ours = T_ours[0, :]
    T_ic_exact = np.cos(2 * np.pi * x_ours)
    ax.plot(x_ours, T_ic_exact, 'k-', label='Exact: cos(2πx)', linewidth=2)
    ax.plot(x_ours, T_ic_ref, 'b--', label='Reference', linewidth=1.5)
    ax.plot(x_ours, T_ic_ours, 'r-', label='Ours', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title('Initial Condition T(x, 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics summary
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = (
        f"Comparison Metrics:\n\n"
        f"RMSE: {rmse:.6f}\n"
        f"Max Error: {max_error:.6f}\n"
        f"Mean Abs Error: {mean_abs_error:.6f}\n\n"
        f"Reference: the reference paper\n"
        f"Grid: {len(x_ours)} x {len(t_ours)}"
    )
    ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / 'reference_comparison.png', dpi=150)
    plt.close()

    console.print(f"\n[bold cyan]Comparison with Reference:[/bold cyan]")
    console.print(f"  RMSE: {rmse:.6f}")
    console.print(f"  Max Error: {max_error:.6f}")
    console.print(f"  Mean Abs Error: {mean_abs_error:.6f}")
    console.print(f"  [green]Saved comparison:[/green] {output_dir / 'reference_comparison.png'}")

    return metrics


