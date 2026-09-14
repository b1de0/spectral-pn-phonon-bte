"""Plotting utilities for 1D transient BTE experiments."""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc

from pinn_bte.utils.logging import console


def analytical_ttg_nongray(
    t: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    L: float,
    tau_ref: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Analytical TTG amplitude decay for mode-resolved (nongray) BTE."""
    # TTG wavevector: q = 2π/L
    q = 2 * np.pi / L

    # Flatten arrays
    v_flat = v.flatten()
    tau_flat = tau.flatten()
    C_flat = C.flatten()

    # Mode-specific Knudsen numbers
    Kn_mode = v_flat * tau_flat * q  # (N_modes,)

    gamma = (v_flat * q) ** 2 * tau_flat / (3.0 * (1 + Kn_mode ** 2))  # (N_modes,)

    # Convert time to physical units (t_physical = t * tau_ref)
    # Then γ·t_physical = γ·t·τ_ref
    t_phys = t * tau_ref  # (Nt,)

    # Compute amplitude decay: A(t) = Σ C·exp(-γt) / Σ C
    # Shape: (Nt, N_modes)
    exp_terms = np.exp(-np.outer(t_phys, gamma))  # (Nt, N_modes)
    A = np.sum(C_flat * exp_terms, axis=1) / np.sum(C_flat)

    return A


def analytical_ttg_gray(
    t: npt.NDArray[np.float64],
    eta: float,
) -> npt.NDArray[np.float64]:
    """Analytical TTG amplitude decay for gray BTE."""
    # Decay rate from kinetic theory
    gamma = eta ** 2 / (1 + eta ** 2)

    return np.exp(-gamma * t)


def analytical_heat_equation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_hot: float,
    T_cold: float,
    T_initial: float,
    L: float,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Analytical solution for 1D heat equation with step heating."""
    Nx, Nt = len(x), len(t)
    T = np.zeros((Nx, Nt))

    # Steady-state solution
    T_steady = T_hot + (T_cold - T_hot) * x / L

    # Number of Fourier terms
    N_terms = 50

    for i, ti in enumerate(t):
        if ti == 0:
            T[:, i] = T_initial
        else:
            # Fourier series for transient part
            T_transient = np.zeros_like(x)
            for n in range(1, N_terms + 1):
                # Coefficient from initial condition
                bn = 2 / L * np.trapezoid(
                    (T_initial - T_steady) * np.sin(n * np.pi * x / L),
                    x
                )
                # Add transient term
                T_transient += bn * np.sin(n * np.pi * x / L) * np.exp(
                    -alpha * (n * np.pi / L) ** 2 * ti
                )
            T[:, i] = T_steady + T_transient

    return T


def plot_temperature_heatmap(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    output_path: Path,
    title: str = "Temperature Evolution T(x,t)",
) -> None:
    """Plot temperature field as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create meshgrid for pcolormesh
    X, T_grid = np.meshgrid(x, t)

    # Plot heatmap
    im = ax.pcolormesh(X, T_grid, T.T, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='Temperature (K)')

    ax.set_xlabel('Position x (normalized)')
    ax.set_ylabel('Time t (τ_ref)')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    console.print(f"[green]Saved heatmap:[/green] {output_path}")


def plot_temperature_snapshots(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    T_analytical: Optional[npt.NDArray[np.float64]],
    output_path: Path,
    n_snapshots: int = 6,
) -> None:
    """Plot temperature profiles at selected time snapshots."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select time indices for snapshots
    t_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    for i, (idx, color) in enumerate(zip(t_indices, colors)):
        label = f't = {t[idx]:.2f} τ'
        ax.plot(x, T_pinn[:, idx], '-', color=color, label=label, linewidth=2)

        if T_analytical is not None:
            ax.plot(x, T_analytical[:, idx], '--', color=color, alpha=0.5)

    ax.set_xlabel('Position x (normalized)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Profiles at Different Times')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if T_analytical is not None:
        # Add legend entry for analytical
        ax.plot([], [], 'k--', alpha=0.5, label='Analytical (heat eq.)')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    console.print(f"[green]Saved snapshots:[/green] {output_path}")


def create_animation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    T_analytical: Optional[npt.NDArray[np.float64]],
    output_path: Path,
    T_range: Optional[tuple] = None,
) -> None:
    """Create animated GIF of temperature evolution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if T_range is None:
        T_min = min(T_pinn.min(), T_analytical.min() if T_analytical is not None else T_pinn.min())
        T_max = max(T_pinn.max(), T_analytical.max() if T_analytical is not None else T_pinn.max())
        margin = (T_max - T_min) * 0.1
        T_range = (T_min - margin, T_max + margin)

    # Initialize lines
    line_pinn, = ax.plot([], [], 'b-', linewidth=2, label='PINN')
    if T_analytical is not None:
        line_analytical, = ax.plot([], [], 'r--', linewidth=2, label='Heat Eq.')

    ax.set_xlim(0, 1)
    ax.set_ylim(T_range)
    ax.set_xlabel('Position x (normalized)')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    def init():
        line_pinn.set_data([], [])
        if T_analytical is not None:
            line_analytical.set_data([], [])
            return line_pinn, line_analytical, time_text
        return line_pinn, time_text

    def animate(i):
        line_pinn.set_data(x, T_pinn[:, i])
        time_text.set_text(f't = {t[i]:.2f} τ')
        if T_analytical is not None:
            line_analytical.set_data(x, T_analytical[:, i])
            return line_pinn, line_analytical, time_text
        return line_pinn, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=100, blit=True
    )

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=10)
    plt.close()

    console.print(f"[green]Saved animation:[/green] {output_path}")


def plot_comparison_error(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    T_analytical: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot error between PINN and analytical solution."""
    error = np.abs(T_pinn - T_analytical)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of error
    ax1 = axes[0]
    X, T_grid = np.meshgrid(x, t)
    im = ax1.pcolormesh(X, T_grid, error.T, shading='auto', cmap='Reds')
    plt.colorbar(im, ax=ax1, label='|T_PINN - T_analytical| (K)')
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Time t (τ)')
    ax1.set_title('Absolute Error')

    # Error over time
    ax2 = axes[1]
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    max_error = np.max(error, axis=0)
    ax2.plot(t, rmse, 'b-', linewidth=2, label='RMSE')
    ax2.plot(t, max_error, 'r--', linewidth=2, label='Max Error')
    ax2.set_xlabel('Time t (τ)')
    ax2.set_ylabel('Error (K)')
    ax2.set_title('Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Print summary statistics
    console.print(f"\n[bold]Comparison with Heat Equation:[/bold]")
    console.print(f"  Overall RMSE: {np.sqrt(np.mean(error**2)):.3f} K")
    console.print(f"  Max Error: {np.max(error):.3f} K")
    console.print(f"  Final RMSE (t=t_max): {rmse[-1]:.3f} K")
    console.print(f"[green]Saved error analysis:[/green] {output_path}")


def plot_loss_history(
    loss_history: list,
    output_path: Path,
) -> None:
    """Plot training loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_array = np.array(loss_history)
    labels = ['BTE_TA', 'BTE_LA', 'T_cons', 'LocalEnergy', 'GlobalEnergy', 'BC_left', 'BC_right', 'IC']

    for i, label in enumerate(labels):
        if i < loss_array.shape[1] and np.any(loss_array[:, i] > 0):
            ax.semilogy(loss_array[:, i], label=label, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    console.print(f"[green]Saved loss history:[/green] {output_path}")


def plot_step_validation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    T_analytical: npt.NDArray[np.float64],
    config: "Experiment1DTransientConfig",
    output_dir: Path,
    run_id: str,
) -> None:
    """Visualizations for step heating mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Heat flux at boundaries over time
    ax1 = axes[0, 0]
    dx = x[1] - x[0]
    q_left = -(T_pinn[1, :] - T_pinn[0, :]) / dx  # -dT/dx at x=0
    q_right = -(T_pinn[-1, :] - T_pinn[-2, :]) / dx  # -dT/dx at x=1
    ax1.plot(t, q_left, 'r-', linewidth=2, label='q(x=0) - hot')
    ax1.plot(t, q_right, 'b-', linewidth=2, label='q(x=1) - cold')
    ax1.axhline(y=q_left[-1], color='r', linestyle='--', alpha=0.5, label=f'Steady q={q_left[-1]:.1f}')
    ax1.set_xlabel('Time t (τ)')
    ax1.set_ylabel('Heat flux (a.u.)')
    ax1.set_title('Boundary Heat Flux Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Temperature at center vs time
    ax2 = axes[0, 1]
    center_idx = len(x) // 2
    T_center_pinn = T_pinn[center_idx, :]
    T_center_analytical = T_analytical[center_idx, :]
    ax2.plot(t, T_center_pinn, 'b-', linewidth=2, label='PINN')
    ax2.plot(t, T_center_analytical, 'r--', linewidth=2, label='Heat Eq.')
    ax2.axhline(y=(config.T_hot + config.T_cold)/2, color='g', linestyle=':',
                label=f'Steady T={(config.T_hot + config.T_cold)/2:.1f}K')
    ax2.set_xlabel('Time t (τ)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature at Center (x=0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Check monotonicity - dT/dt should be >= 0 for heating
    ax3 = axes[1, 0]
    dT_dt = np.diff(T_pinn, axis=1) / np.diff(t)
    dT_dt_mean = np.mean(dT_dt, axis=0)
    dT_dt_min = np.min(dT_dt, axis=0)
    ax3.plot(t[:-1], dT_dt_mean, 'b-', linewidth=2, label='Mean dT/dt')
    ax3.plot(t[:-1], dT_dt_min, 'r--', linewidth=1, label='Min dT/dt')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.fill_between(t[:-1], 0, dT_dt_min, where=dT_dt_min < 0,
                     color='red', alpha=0.3, label='Violations')
    ax3.set_xlabel('Time t (τ)')
    ax3.set_ylabel('dT/dt (K/τ)')
    ax3.set_title('Monotonicity Check (should be ≥0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Approach to steady state
    ax4 = axes[1, 1]
    T_steady = config.T_hot + (config.T_cold - config.T_hot) * x
    deviation = np.sqrt(np.mean((T_pinn - T_steady[:, np.newaxis])**2, axis=0))
    ax4.semilogy(t, deviation, 'b-', linewidth=2)
    ax4.set_xlabel('Time t (τ)')
    ax4.set_ylabel('RMSE from steady state (K)')
    ax4.set_title('Approach to Steady State')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"step_validation_{run_id}.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved step validation:[/green] step_validation_{run_id}.png")


def plot_pulse_validation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    config: "Experiment1DTransientConfig",
    output_dir: Path,
    run_id: str,
) -> None:
    """Visualizations for pulse heating mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    T_ref = config.reference_temperature
    T_excess = T_pinn - T_ref  # Temperature above reference

    # 1. Peak temperature location over time
    ax1 = axes[0, 0]
    peak_idx = np.argmax(T_excess, axis=0)
    peak_x = x[peak_idx]
    peak_T = np.max(T_excess, axis=0)

    ax1.plot(t, peak_x, 'b-', linewidth=2)
    ax1.set_xlabel('Time t (τ)')
    ax1.set_ylabel('Peak position x')
    ax1.set_title('Heat Wave Propagation')
    ax1.grid(True, alpha=0.3)

    # Ballistic prediction: x = v*t (normalized)
    v_ballistic = 0.3  # rough estimate
    t_plot = np.linspace(0, t[-1], 100)
    ax1.plot(t_plot, v_ballistic * t_plot, 'r--', alpha=0.5, label='Ballistic ~v·t')
    ax1.plot(t_plot, np.sqrt(t_plot) * 0.3, 'g--', alpha=0.5, label='Diffusive ~√t')
    ax1.legend()

    # 2. Peak temperature decay
    ax2 = axes[0, 1]
    ax2.semilogy(t[1:], peak_T[1:] + 1e-10, 'b-', linewidth=2)
    ax2.set_xlabel('Time t (τ)')
    ax2.set_ylabel('Peak ΔT (K)')
    ax2.set_title('Peak Temperature Decay')
    ax2.grid(True, alpha=0.3)

    # 3. Spatial profiles at different times
    ax3 = axes[1, 0]
    n_profiles = 6
    t_indices = np.linspace(1, len(t) - 1, n_profiles, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for idx, color in zip(t_indices, colors):
        ax3.plot(x, T_excess[:, idx], '-', color=color,
                 linewidth=2, label=f't={t[idx]:.2f}τ')
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('ΔT (K)')
    ax3.set_title('Temperature Profiles (Heat Spreading)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. Width of heat distribution (FWHM)
    ax4 = axes[1, 1]
    widths = []
    for i in range(len(t)):
        profile = T_excess[:, i]
        if profile.max() > 0:
            half_max = profile.max() / 2
            above_half = profile > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                width = x[indices[-1]] - x[indices[0]]
            else:
                width = 0
        else:
            width = 0
        widths.append(width)

    ax4.plot(t, widths, 'b-', linewidth=2)
    ax4.plot(t_plot, np.sqrt(t_plot) * 0.5, 'g--', alpha=0.5, label='Diffusive ~√t')
    ax4.set_xlabel('Time t (τ)')
    ax4.set_ylabel('FWHM width')
    ax4.set_title('Heat Distribution Width')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"pulse_validation_{run_id}.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved pulse validation:[/green] pulse_validation_{run_id}.png")


def plot_kn_validation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    config: "Experiment1DTransientConfig",
    output_dir: Path,
    run_id: str,
) -> None:
    """Visualizations for Knudsen number study mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    L = config.domain_length
    lambda_mfp = 400  # Approximate MFP for Si at 300K in Angstroms
    Kn = lambda_mfp / L

    # 1. Final temperature profile with Kn info
    ax1 = axes[0, 0]
    T_final = T_pinn[:, -1]
    T_linear = config.T_hot + (config.T_cold - config.T_hot) * x  # Fourier prediction

    ax1.plot(x, T_final, 'b-', linewidth=2, label='PINN (BTE)')
    ax1.plot(x, T_linear, 'r--', linewidth=2, label='Fourier (diffusive)')
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title(f'Final Profile: L={L:.0f}Å, Kn≈{Kn:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Temperature jump at boundaries
    ax2 = axes[0, 1]
    T_jump_hot = config.T_hot - T_pinn[1, :]  # Jump at hot boundary
    T_jump_cold = T_pinn[-2, :] - config.T_cold  # Jump at cold boundary

    ax2.plot(t, T_jump_hot, 'r-', linewidth=2, label='Hot boundary jump')
    ax2.plot(t, T_jump_cold, 'b-', linewidth=2, label='Cold boundary jump')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time t (τ)')
    ax2.set_ylabel('Temperature jump (K)')
    ax2.set_title('Boundary Temperature Jumps (Kn effect)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Effective thermal conductivity
    ax3 = axes[1, 0]
    dx = x[1] - x[0]
    q = -(T_pinn[1:, :] - T_pinn[:-1, :]) / dx
    q_mean = np.mean(q, axis=0)
    dT = config.T_hot - config.T_cold
    kappa_eff = np.abs(q_mean) * 1.0 / dT  # Normalized

    ax3.plot(t, kappa_eff, 'b-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Fourier κ')
    ax3.set_xlabel('Time t (τ)')
    ax3.set_ylabel('κ_eff / κ_bulk')
    ax3.set_title(f'Effective Thermal Conductivity (Kn≈{Kn:.2f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Relaxation characteristic
    ax4 = axes[1, 1]
    T_center = T_pinn[len(x)//2, :]
    T_steady = (config.T_hot + config.T_cold) / 2
    deviation = np.abs(T_center - T_steady)
    deviation_norm = deviation / deviation[0] if deviation[0] > 0 else deviation

    ax4.semilogy(t, deviation_norm + 1e-10, 'b-', linewidth=2)
    ax4.set_xlabel('Time t (τ)')
    ax4.set_ylabel('|T - T_steady| / |T₀ - T_steady|')
    ax4.set_title('Relaxation Dynamics')
    ax4.grid(True, alpha=0.3)

    # Estimate relaxation time
    half_idx = np.argmax(deviation_norm < 0.5) if np.any(deviation_norm < 0.5) else -1
    if half_idx > 0:
        ax4.axvline(x=t[half_idx], color='r', linestyle='--',
                    label=f'τ_relax ≈ {t[half_idx]:.2f}τ')
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"kn_validation_{run_id}.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved Kn validation:[/green] kn_validation_{run_id}.png")
    console.print(f"  Knudsen number: Kn ≈ {Kn:.3f}")


def plot_timescale_validation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    T_analytical: npt.NDArray[np.float64],
    config: "Experiment1DTransientConfig",
    output_dir: Path,
    run_id: str,
) -> None:
    """Visualizations for time scale study mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PINN vs Fourier error over time
    ax1 = axes[0, 0]
    error = np.sqrt(np.mean((T_pinn - T_analytical)**2, axis=0))
    ax1.semilogy(t, error + 1e-10, 'b-', linewidth=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='t = τ')
    ax1.set_xlabel('Time t (τ)')
    ax1.set_ylabel('RMSE PINN vs Fourier (K)')
    ax1.set_title('BTE-Fourier Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Relative error (should decrease at long times)
    ax2 = axes[0, 1]
    T_range = np.max(T_pinn) - np.min(T_pinn)
    rel_error = error / (T_range + 1e-10) * 100
    ax2.plot(t, rel_error, 'b-', linewidth=2)
    ax2.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_xlabel('Time t (τ)')
    ax2.set_ylabel('Relative error (%)')
    ax2.set_title('Relative Error (BTE vs Fourier)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Short vs long time profiles
    ax3 = axes[1, 0]
    early_idx = max(1, int(0.1 / config.t_max * len(t)))
    late_idx = -1

    ax3.plot(x, T_pinn[:, early_idx], 'b-', linewidth=2,
             label=f'PINN t={t[early_idx]:.2f}τ')
    ax3.plot(x, T_analytical[:, early_idx], 'b--', linewidth=2,
             label=f'Fourier t={t[early_idx]:.2f}τ')
    ax3.plot(x, T_pinn[:, late_idx], 'r-', linewidth=2,
             label=f'PINN t={t[late_idx]:.1f}τ')
    ax3.plot(x, T_analytical[:, late_idx], 'r--', linewidth=2,
             label=f'Fourier t={t[late_idx]:.1f}τ')

    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Temperature (K)')
    ax3.set_title('Early vs Late Time Profiles')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Time to reach steady state
    ax4 = axes[1, 1]
    T_steady = config.T_hot + (config.T_cold - config.T_hot) * x
    steady_deviation = np.sqrt(np.mean((T_pinn - T_steady[:, np.newaxis])**2, axis=0))

    ax4.semilogy(t, steady_deviation + 1e-10, 'b-', linewidth=2, label='PINN')
    ax4.semilogy(t, np.sqrt(np.mean((T_analytical - T_steady[:, np.newaxis])**2, axis=0)) + 1e-10,
                 'r--', linewidth=2, label='Fourier')
    ax4.set_xlabel('Time t (τ)')
    ax4.set_ylabel('RMSE from steady state (K)')
    ax4.set_title(f'Approach to Steady (t_max={config.t_max}τ)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"timescale_validation_{run_id}.png", dpi=150)
    plt.close()
    console.print(f"[green]Saved timescale validation:[/green] timescale_validation_{run_id}.png")


def plot_validation(
    x: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    T_pinn: npt.NDArray[np.float64],
    output_dir: Path,
    run_id: str,
) -> None:
    """Visualizations for validation mode."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Initial condition: T(x, 0) should match cos(2πx)
    ax1 = axes[0, 0]
    t_idx_0 = 0
    T_ic_pinn = T_pinn[:, t_idx_0]
    T_ic_exact = np.cos(2 * np.pi * x)
    ax1.plot(x, T_ic_exact, 'k--', label='Exact: cos(2πx)', linewidth=2)
    ax1.plot(x, T_ic_pinn, 'r-', label='PINN', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('T')
    ax1.set_title('Initial Condition (t=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Compute IC error
    ic_error = np.sqrt(np.mean((T_ic_pinn - T_ic_exact)**2))
    ax1.text(0.02, 0.02, f'RMSE: {ic_error:.4f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom')

    # 2. Temperature decay over time (should decay exponentially)
    ax2 = axes[0, 1]
    n_profiles = 6
    t_indices = np.linspace(0, len(t) - 1, n_profiles, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for idx, color in zip(t_indices, colors):
        ax2.plot(x, T_pinn[:, idx], '-', color=color, linewidth=2,
                 label=f't={t[idx]:.2f}τ')
    ax2.set_xlabel('x')
    ax2.set_ylabel('T')
    ax2.set_title('Temperature Evolution')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Amplitude decay: max(|T|) vs time (should decay)
    ax3 = axes[1, 0]
    T_amplitude = np.max(np.abs(T_pinn), axis=0)
    ax3.semilogy(t, T_amplitude, 'b-', linewidth=2, label='PINN')

    A0 = T_amplitude[0]
    tau_decay_diffusive = 1.0 / (2 * np.pi)**2
    ax3.semilogy(t, A0 * np.exp(-t / tau_decay_diffusive), 'g--', linewidth=2,
                 label=f'Diffusive: exp(-t/{tau_decay_diffusive:.3f})')
    ax3.semilogy(t, A0 * np.exp(-t / 0.5), 'r--', linewidth=2,
                 label='exp(-t/0.5)')

    ax3.set_xlabel('t (τ)')
    ax3.set_ylabel('max|T|')
    ax3.set_title('Amplitude Decay (log scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Periodic BC check: T(0,t) vs T(1,t)
    ax4 = axes[1, 1]
    T_left = T_pinn[0, :]
    T_right = T_pinn[-1, :]
    bc_error = np.abs(T_left - T_right)

    ax4.plot(t, T_left, 'b-', linewidth=2, label='T(x=0)')
    ax4.plot(t, T_right, 'r--', linewidth=2, label='T(x=1)')
    ax4.set_xlabel('t (τ)')
    ax4.set_ylabel('T')
    ax4.set_title('Periodic BC Check: T(0) vs T(1)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Secondary y-axis for error
    ax4_twin = ax4.twinx()
    ax4_twin.semilogy(t, bc_error + 1e-10, 'g-', alpha=0.5, linewidth=1)
    ax4_twin.set_ylabel('|T(0) - T(1)|', color='g')
    ax4_twin.tick_params(axis='y', labelcolor='g')

    plt.tight_layout()
    plt.savefig(output_dir / f"validation_{run_id}.png", dpi=150)
    plt.close()

    # Print summary
    console.print(f"\n[bold]Validation Summary:[/bold]")
    console.print(f"  IC RMSE: {ic_error:.4f}")
    console.print(f"  Final amplitude: {T_amplitude[-1]:.4f} (initial: {T_amplitude[0]:.4f})")
    console.print(f"  Decay ratio: {T_amplitude[-1]/T_amplitude[0]:.4f}")
    console.print(f"  Mean BC error: {np.mean(bc_error):.6f}")
    console.print(f"[green]Saved validation:[/green] validation_{run_id}.png")
