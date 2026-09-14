"""Visualization utilities for transient transient BTE trainers."""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize


def create_temperature_animation(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    output_path: Path,
    Lt: float = 3.0,
    fps: int = 10,
    title: str = "Temperature Evolution",
) -> None:
    """Create animated GIF of temperature evolution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Auto-detect amplitude from initial data
    T_ic = T[0, :]
    amplitude = (np.max(T_ic) - np.min(T_ic)) / 2
    if amplitude < 0.1:
        amplitude = 1.0  # Fallback for normalized small ΔT

    # Initial frame
    line, = ax.plot(x, T[0, :], 'b-', linewidth=2, label='PINN')
    exact_line, = ax.plot(x, amplitude * np.cos(2 * np.pi * x), 'k--', linewidth=1.5,
                          alpha=0.5, label=f'{amplitude:.0f}K·cos(2πx)' if amplitude > 1 else 'cos(2πx)')

    ax.set_xlim(0, 1)
    y_max = amplitude * 1.2
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('x (normalized)', fontsize=12)
    ax.set_ylabel('T [K]' if amplitude > 1 else 'T', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)
    ax.set_title(title, fontsize=14)

    def update(frame):
        line.set_ydata(T[frame, :])
        t_physical = t[frame] * Lt
        time_text.set_text(f't = {t_physical:.3f}τ')
        return line, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(t), interval=1000 // fps, blit=True
    )

    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animation: {output_path}")


def plot_amplitude_decay(
    t: np.ndarray,
    T: np.ndarray,
    Lt: float,
    eta: float,
    output_path: Path,
    include_analytical: bool = True,
) -> dict:
    """Plot amplitude decay with optional analytical comparison."""
    # Compute amplitude from PINN results
    amplitude_pinn = np.max(T, axis=1) - np.min(T, axis=1)
    amplitude_pinn_normalized = amplitude_pinn / amplitude_pinn[0]

    # Physical time
    t_physical = t * Lt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Amplitude decay
    ax = axes[0]
    ax.plot(t_physical, amplitude_pinn_normalized, 'b-', linewidth=2, label='PINN')

    if include_analytical:
        # Compute analytical solution using integral equation
        A_analytical = compute_analytical_amplitude(t_physical, eta)
        ax.plot(t_physical, A_analytical, 'r--', linewidth=2, label='Analytical')

    ax.set_xlabel('t/τ', fontsize=12)
    ax.set_ylabel('A/A₀ (normalized amplitude)', fontsize=12)
    ax.set_title(f'Amplitude Decay (η={eta:.2f}, Kn={eta/(2*np.pi):.3f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Right: Log scale
    ax = axes[1]
    ax.semilogy(t_physical, amplitude_pinn_normalized, 'b-', linewidth=2, label='PINN')
    if include_analytical:
        ax.semilogy(t_physical, A_analytical, 'r--', linewidth=2, label='Analytical')

    ax.set_xlabel('t/τ', fontsize=12)
    ax.set_ylabel('A/A₀ (log scale)', fontsize=12)
    ax.set_title('Amplitude Decay (log scale)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved amplitude plot: {output_path}")

    # Metrics
    metrics = {
        'initial_amplitude': amplitude_pinn[0],
        'final_amplitude': amplitude_pinn[-1],
        'decay_ratio': amplitude_pinn[-1] / amplitude_pinn[0],
    }

    if include_analytical:
        rmse = np.sqrt(np.mean((amplitude_pinn_normalized - A_analytical) ** 2))
        metrics['analytical_rmse'] = rmse

    return metrics


def compute_analytical_amplitude(t: np.ndarray, eta: float) -> np.ndarray:
    """Compute analytical amplitude decay for TTG."""
    n = len(t)
    A = np.zeros(n)
    A[0] = 1.0

    dt = t[1] - t[0] if n > 1 else 0.01

    for i in range(1, n):
        ti = t[i]
        # Free term: sinc(ξt) * exp(-t)
        if eta * ti == 0:
            sinc_term = 1.0
        else:
            sinc_term = np.sin(eta * ti) / (eta * ti)
        free_term = sinc_term * np.exp(-ti)

        # Trapezoidal integral over known points j = 0..i-1
        integral = 0.0
        for j in range(i):
            delta_t = ti - t[j]
            if eta * delta_t == 0:
                sinc_delta = 1.0
            else:
                sinc_delta = np.sin(eta * delta_t) / (eta * delta_t)
            integrand = A[j] * sinc_delta * np.exp(-delta_t)
            if j == 0:
                integral += 0.5 * integrand * dt
            else:
                integral += integrand * dt

        A[i] = (free_term + integral) / (1.0 - 0.5 * dt)

    return A


def plot_temperature_snapshots(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    Lt: float,
    output_path: Path,
    time_fractions: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
) -> None:
    """Plot temperature snapshots at different times."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(time_fractions)))

    for i, frac in enumerate(time_fractions):
        idx = int(frac * (len(t) - 1))
        t_val = t[idx] * Lt
        ax.plot(x, T[idx, :], color=colors[i], linewidth=2,
                label=f't = {t_val:.2f}τ')

    # Reference: initial condition
    ax.plot(x, np.cos(2 * np.pi * x), 'k--', alpha=0.3, label='cos(2πx)')

    ax.set_xlabel('x (normalized)', fontsize=12)
    ax.set_ylabel('T', fontsize=12)
    ax.set_title('Temperature Profiles at Different Times', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved snapshots: {output_path}")


def plot_temperature_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    Lt: float,
    output_path: Path,
    title: str = "Temperature Field T(x,t)",
) -> None:
    """Plot temperature field as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    X, T_grid = np.meshgrid(x, t * Lt)
    cs = ax.contourf(X, T_grid, T, levels=50, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='T')

    ax.set_xlabel('x (normalized)', fontsize=12)
    ax.set_ylabel('t/τ', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {output_path}")


def plot_boundary_condition_check(
    t: np.ndarray,
    T: np.ndarray,
    Lt: float,
    output_path: Path,
    bc_type: str = "periodic",
) -> dict:
    """Plot boundary condition verification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t_physical = t * Lt

    if bc_type == "periodic":
        # Periodic BC: T(0,t) = T(1,t)
        T_left = T[:, 0]
        T_right = T[:, -1]
        bc_error = np.abs(T_left - T_right)

        # Left: T at boundaries
        ax = axes[0]
        ax.plot(t_physical, T_left, 'b-', label='T(x=0, t)', linewidth=2)
        ax.plot(t_physical, T_right, 'r--', label='T(x=1, t)', linewidth=2)
        ax.set_xlabel('t/τ', fontsize=12)
        ax.set_ylabel('T', fontsize=12)
        ax.set_title('Temperature at Boundaries', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: BC error
        ax = axes[1]
        ax.semilogy(t_physical, bc_error + 1e-10, 'k-', linewidth=2)
        ax.set_xlabel('t/τ', fontsize=12)
        ax.set_ylabel('|T(0,t) - T(1,t)|', fontsize=12)
        ax.set_title(f'Periodic BC Error (mean={np.mean(bc_error):.2e})', fontsize=14)
        ax.grid(True, alpha=0.3)

        metrics = {
            'mean_bc_error': np.mean(bc_error),
            'max_bc_error': np.max(bc_error),
        }

    else:
        # Dirichlet BC
        metrics = {}

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved BC check: {output_path}")

    return metrics


def plot_initial_condition_check(
    x: np.ndarray,
    T: np.ndarray,
    output_path: Path,
    ic_type: str = "cosine",
    amplitude: float = None,
) -> dict:
    """Plot initial condition verification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T_ic = T[0, :]

    # Auto-detect amplitude from data if not provided
    if amplitude is None:
        amplitude = (np.max(T_ic) - np.min(T_ic)) / 2

    if ic_type == "cosine":
        T_exact = amplitude * np.cos(2 * np.pi * x)
    else:
        T_exact = np.zeros_like(x)

    ic_error = np.abs(T_ic - T_exact)

    # Left: IC comparison
    ax = axes[0]
    ax.plot(x, T_exact, 'k-', label='Exact IC', linewidth=2)
    ax.plot(x, T_ic, 'r--', label='PINN T(x, 0)', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('T', fontsize=12)
    ax.set_title('Initial Condition Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: IC error
    ax = axes[1]
    ax.semilogy(x, ic_error + 1e-10, 'k-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('|T_PINN - T_exact|', fontsize=12)
    rmse = np.sqrt(np.mean(ic_error ** 2))
    ax.set_title(f'IC Error (RMSE={rmse:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved IC check: {output_path}")

    metrics = {
        'ic_rmse': rmse,
        'ic_max_error': np.max(ic_error),
    }

    return metrics


def plot_comprehensive_validation(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    eta: float,
    Lt: float,
    output_path: Path,
) -> dict:
    """Generate comprehensive validation figure with multiple panels."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    t_physical = t * Lt
    Kn = eta / (2 * np.pi)

    # 1. Temperature heatmap (top left, spans 2 rows)
    ax1 = fig.add_subplot(gs[0:2, 0])
    X, T_grid = np.meshgrid(x, t_physical)
    cs = ax1.contourf(X, T_grid, T, levels=50, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax1, label='T')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t/τ')
    ax1.set_title(f'T(x,t) - η={eta:.2f}, Kn={Kn:.3f}')

    # 2. Temperature snapshots (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    for i, idx in enumerate(time_indices):
        ax2.plot(x, T[idx, :], color=colors[i], label=f't={t_physical[idx]:.2f}τ')
    ax2.plot(x, np.cos(2*np.pi*x), 'k--', alpha=0.3, label='IC')
    ax2.set_xlabel('x')
    ax2.set_ylabel('T')
    ax2.set_title('T(x) at different times')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. Amplitude decay (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    amplitude = np.max(T, axis=1) - np.min(T, axis=1)
    amplitude_norm = amplitude / amplitude[0]
    A_analytical = compute_analytical_amplitude(t_physical, eta)
    ax3.plot(t_physical, amplitude_norm, 'b-', linewidth=2, label='PINN')
    ax3.plot(t_physical, A_analytical, 'r--', linewidth=2, label='Analytical')
    ax3.set_xlabel('t/τ')
    ax3.set_ylabel('A/A₀')
    ax3.set_title('Amplitude Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. IC check (middle middle)
    ax4 = fig.add_subplot(gs[1, 1])
    T_ic = T[0, :]
    T_exact = np.cos(2 * np.pi * x)
    ax4.plot(x, T_exact, 'k-', label='cos(2πx)', linewidth=2)
    ax4.plot(x, T_ic, 'r--', label='PINN', linewidth=2)
    ic_rmse = np.sqrt(np.mean((T_ic - T_exact) ** 2))
    ax4.set_xlabel('x')
    ax4.set_ylabel('T')
    ax4.set_title(f'Initial Condition (RMSE={ic_rmse:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. BC check (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    bc_error = np.abs(T[:, 0] - T[:, -1])
    ax5.semilogy(t_physical, bc_error + 1e-10, 'k-', linewidth=2)
    ax5.set_xlabel('t/τ')
    ax5.set_ylabel('|T(0) - T(1)|')
    ax5.set_title(f'Periodic BC Error (mean={np.mean(bc_error):.2e})')
    ax5.grid(True, alpha=0.3)

    # 6. T(t) at different x (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    x_indices = [0, len(x)//4, len(x)//2, 3*len(x)//4, -1]
    for idx in x_indices:
        ax6.plot(t_physical, T[:, idx], label=f'x={x[idx]:.2f}')
    ax6.set_xlabel('t/τ')
    ax6.set_ylabel('T')
    ax6.set_title('T(t) at different x')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Amplitude decay (log) (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.semilogy(t_physical, amplitude_norm, 'b-', linewidth=2, label='PINN')
    ax7.semilogy(t_physical, A_analytical, 'r--', linewidth=2, label='Analytical')
    ax7.set_xlabel('t/τ')
    ax7.set_ylabel('A/A₀ (log)')
    ax7.set_title('Amplitude Decay (log scale)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Metrics text (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    amplitude_rmse = np.sqrt(np.mean((amplitude_norm - A_analytical) ** 2))
    bc_mean = np.mean(bc_error)

    metrics_text = (
        f"Validation Metrics\n"
        f"─────────────────\n"
        f"η = {eta:.2f}\n"
        f"Kn = {Kn:.4f}\n"
        f"L = {2*np.pi/eta:.3f}\n"
        f"Lt = {Lt:.1f}τ\n\n"
        f"IC RMSE: {ic_rmse:.6f}\n"
        f"BC Mean Error: {bc_mean:.2e}\n"
        f"Amplitude RMSE: {amplitude_rmse:.6f}\n"
        f"Final Decay: {amplitude_norm[-1]:.4f}\n"
    )
    ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes,
             fontsize=11, family='monospace', verticalalignment='top')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive validation: {output_path}")

    metrics = {
        'ic_rmse': ic_rmse,
        'bc_mean_error': bc_mean,
        'amplitude_rmse': amplitude_rmse,
        'final_decay': amplitude_norm[-1],
    }

    return metrics


def plot_nongray_amplitude_decay(
    t: np.ndarray,
    T: np.ndarray,
    A_analytical: np.ndarray,
    Lt: float,
    L_um: float,
    Kn_eff: float,
    output_path: Path,
) -> float:
    """Plot amplitude decay with nongray analytical comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get PINN amplitude
    T_amplitude = np.max(np.abs(T), axis=1)
    A_pinn = T_amplitude / T_amplitude[0] if T_amplitude[0] > 0 else T_amplitude
    t_phys = t * Lt

    ax.semilogy(t_phys, A_pinn, 'b-', linewidth=2, label='PINN (Nongray)')
    ax.semilogy(t_phys, A_analytical, 'r--', linewidth=2, label='Analytical (Mode-resolved)')

    # Also show single-exponential for reference
    if A_pinn[-1] > 1e-10:
        gamma_eff = -np.log(A_pinn[-1]) / t_phys[-1]
        A_single_exp = np.exp(-gamma_eff * t_phys)
        ax.semilogy(t_phys, A_single_exp, 'g:', linewidth=1, alpha=0.5,
                    label=f'Single exp (γ={gamma_eff:.3f})')

    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('Normalized Amplitude A(t)/A(0)')
    ax.set_title(f'TTG Amplitude Decay - L={L_um:.1f}μm, Kn_eff={Kn_eff:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved amplitude plot: {output_path}")

    rmse = np.sqrt(np.mean((A_pinn - A_analytical) ** 2))
    return rmse


def plot_nongray_comprehensive(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    Lt: float,
    L_um: float,
    Kn_eff: float,
    mode_props: dict,
    A_analytical: np.ndarray,
    output_path: Path,
) -> None:
    """Create comprehensive validation plot for nongray model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    t_phys = t * Lt

    # 1. Temperature heatmap
    ax = axes[0, 0]
    X, Tgrid = np.meshgrid(x, t_phys)
    cs = ax.contourf(X, Tgrid, T, levels=50, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='T')
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('t / τ_ref')
    ax.set_title('T(x, t)')

    # 2. Mode-resolved decay rates
    ax = axes[0, 1]
    gamma = mode_props['gamma'].flatten()
    C = mode_props['C'].flatten()
    Kn_mode = mode_props['Kn_mode'].flatten()

    sort_idx = np.argsort(Kn_mode)
    scatter = ax.scatter(Kn_mode[sort_idx], gamma[sort_idx], c=C[sort_idx], s=50,
                         cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Heat capacity C')
    ax.set_xlabel('Mode Kn = v·τ·q')
    ax.set_ylabel('Decay rate γ (1/s)')
    ax.set_title('Mode-resolved decay rates')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. Amplitude decay comparison
    ax = axes[0, 2]
    T_amplitude = np.max(np.abs(T), axis=1)
    A_pinn = T_amplitude / T_amplitude[0] if T_amplitude[0] > 0 else T_amplitude

    ax.semilogy(t_phys, A_pinn, 'b-', linewidth=2, label='PINN')
    ax.semilogy(t_phys, A_analytical, 'r--', linewidth=2, label='Analytical')
    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('A(t) / A(0)')
    ax.set_title('Amplitude Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Temperature snapshots
    ax = axes[1, 0]
    n_snapshots = 5
    t_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    for idx, color in zip(t_indices, colors):
        ax.plot(x, T[idx, :], '-', color=color, linewidth=2,
                label=f't={t_phys[idx]:.1f}τ')
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Heat capacity spectrum
    ax = axes[1, 1]
    C_norm = C / np.max(C)
    ax.bar(np.arange(len(C_norm)), C_norm, color='steelblue', alpha=0.7)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('C / C_max')
    Nk = mode_props.get('Nk', len(C) // 3)
    Np = mode_props.get('Np', 3)
    ax.set_title(f'Heat capacity spectrum (Nk={Nk}, Np={Np})')
    ax.grid(True, alpha=0.3)

    # 6. Error analysis
    ax = axes[1, 2]
    error = A_pinn - A_analytical
    ax.plot(t_phys, error, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(t_phys, 0, error, alpha=0.3)
    rmse = np.sqrt(np.mean(error**2))
    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('A_PINN - A_analytical')
    ax.set_title(f'Error (RMSE={rmse:.4f})')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Nongray Validation: L={L_um:.1f}μm, Kn_eff={Kn_eff:.4f}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comprehensive validation: {output_path}")


def plot_2d_snapshots(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    Kn: float,
    output_path: Path,
    n_times: int = 6,
    T_ref: Optional[float] = None,
    show_physical: bool = False,
) -> None:
    """Plot 2D temperature snapshots at different times."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    time_indices = np.linspace(0, len(t) - 1, n_times, dtype=int)

    X, Y = np.meshgrid(x, y)

    # Convert to physical temperature if requested
    T_plot = T + T_ref if (show_physical and T_ref is not None) else T
    cbar_label = 'T (K)' if (show_physical and T_ref is not None) else 'T - T_ref (K)'
    cmap = 'hot' if (show_physical and T_ref is not None) else 'RdBu_r'

    for ax, idx in zip(axes.flat, time_indices):
        cs = ax.contourf(X, Y, T_plot[idx, :, :], levels=50, cmap=cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t[idx]:.2f}')
        ax.set_aspect('equal')
        plt.colorbar(cs, ax=ax, shrink=0.8, label=cbar_label)

    title = f'2D Temperature Evolution (Kn={Kn:.3f})'
    if show_physical and T_ref is not None:
        title += f', T_ref={T_ref:.0f}K'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved 2D snapshots: {output_path}")


def plot_2d_centerlines(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    output_path: Path,
    n_times: int = 4,
    T_ref: Optional[float] = None,
    show_physical: bool = False,
) -> None:
    """Plot temperature along centerlines at different times."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    time_indices = np.linspace(0, len(t) - 1, n_times, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_times))

    y_mid = len(y) // 2
    x_mid = len(x) // 2

    # Convert to physical temperature if requested
    T_plot = T + T_ref if (show_physical and T_ref is not None) else T
    y_label = 'T (K)' if (show_physical and T_ref is not None) else 'T - T_ref (K)'

    # T(x, y=0.5) at different times
    ax = axes[0]
    for idx, color in zip(time_indices, colors):
        ax.plot(x, T_plot[idx, y_mid, :], color=color, label=f't={t[idx]:.2f}')
    if show_physical and T_ref is not None:
        ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')
    ax.set_xlabel('x')
    ax.set_ylabel(y_label)
    ax.set_title('T(x, y=0.5) at different times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # T(x=0.5, y) at different times
    ax = axes[1]
    for idx, color in zip(time_indices, colors):
        ax.plot(y, T_plot[idx, :, x_mid], color=color, label=f't={t[idx]:.2f}')
    if show_physical and T_ref is not None:
        ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')
    ax.set_xlabel('y')
    ax.set_ylabel(y_label)
    ax.set_title('T(x=0.5, y) at different times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved centerlines: {output_path}")


def plot_2d_evolution(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    output_path: Path,
    T_ref: Optional[float] = None,
    show_physical: bool = False,
) -> None:
    """Plot temperature evolution at key points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to physical temperature if requested
    T_plot = T + T_ref if (show_physical and T_ref is not None) else T
    y_label = 'T (K)' if (show_physical and T_ref is not None) else 'T - T_ref (K)'

    points = [
        (len(x)//2, len(y)//2, 'center'),
        (len(x)//2, -1, 'top-center'),
        (0, len(y)//2, 'left-mid'),
    ]

    for xi, yi, name in points:
        ax.plot(t, T_plot[:, yi, xi], label=name, linewidth=2)

    if show_physical and T_ref is not None:
        ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')

    ax.set_xlabel('t (normalized)')
    ax.set_ylabel(y_label)
    ax.set_title('Temperature Evolution at Key Points')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved evolution: {output_path}")


def plot_2d_final_state(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    output_path: Path,
) -> None:
    """Plot final temperature field."""
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(x, y)
    cs = ax.contourf(X, Y, T[-1, :, :], levels=50, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Final Temperature Field (t={t[-1]:.2f})')
    ax.set_aspect('equal')
    plt.colorbar(cs, ax=ax, label='T')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved final state: {output_path}")


def plot_loss_history(
    loss_history: list,
    output_dir: Path,
    run_id: str,
    labels: Optional[List[str]] = None,
) -> None:
    """Plot training loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_array = np.array(loss_history)

    if labels is None:
        labels = [f'Loss_{i}' for i in range(loss_array.shape[1] - 1)] + ['Total']

    for i, label in enumerate(labels):
        if i < loss_array.shape[1]:
            ax.semilogy(loss_array[:, i], label=label, alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"loss_{run_id}.png", dpi=150)
    plt.close()
    print(f"  Saved: loss_{run_id}.png")


def create_2d_temperature_animation(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    output_path: Path,
    fps: int = 10,
    title: str = "2D Temperature Evolution",
    T_ref: Optional[float] = None,
    show_physical: bool = False,
) -> None:
    """Create animated GIF of 2D temperature evolution."""
    # Ensure arrays are numpy arrays and have correct shape
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    T = np.asarray(T)

    # Get number of time frames from T
    Nt = T.shape[0]

    # Validate and create time array
    if t is None:
        t = np.linspace(0, 1, Nt)
    else:
        t = np.asarray(t).flatten()
        if len(t) != Nt:
            # Regenerate time array if mismatch
            t = np.linspace(float(t[0]) if len(t) > 0 else 0,
                           float(t[-1]) if len(t) > 0 else 1, Nt)

    # Make sure t is a list for safe indexing
    t_list = list(t)

    # Convert to physical temperature if requested
    T_plot = T + T_ref if (show_physical and T_ref is not None) else T
    cbar_label = 'T (K)' if (show_physical and T_ref is not None) else 'T - T_ref (K)'
    cmap = 'hot' if (show_physical and T_ref is not None) else 'RdBu_r'

    # Create figure with space for colorbar
    fig, ax = plt.subplots(figsize=(9, 7))
    X, Y = np.meshgrid(x, y)

    # Get global min/max for consistent colorscale
    vmin, vmax = T_plot.min(), T_plot.max()
    if abs(vmax - vmin) < 1e-10:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    # Create fixed levels for consistent colorbar
    levels = np.linspace(vmin, vmax, 51)

    # Initial frame with colorbar
    cs = ax.contourf(X, Y, T_plot[0, :, :], levels=levels, cmap=cmap)
    cbar = fig.colorbar(cs, ax=ax, label=cbar_label, shrink=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)

    def update(frame):
        # Clear only the contour collections, not the entire axis
        for c in ax.collections:
            c.remove()
        for txt in ax.texts:
            txt.remove()

        # Draw new contourf with same levels and colormap
        ax.contourf(X, Y, T_plot[frame, :, :], levels=levels, cmap=cmap)

        # Add time label
        t_val = t_list[frame] if frame < len(t_list) else frame
        ax.text(0.02, 0.98, f't = {t_val:.2f}', transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=Nt, interval=1000 // fps, blit=False
    )

    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()
    print(f"Saved 2D animation: {output_path}")


def plot_beta_network(
    net_beta,
    T_ref: float,
    delta_T: float,
    Nk: int,
    Np: int,
    output_path: Path,
    device: str = 'cpu',
) -> None:
    """Visualize β-network predictions across temperature and modes."""
    import torch

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T_norm_range = np.linspace(-1, 1, 50)
    T_physical = T_ref + T_norm_range * (delta_T / 2)

    # 1. β vs T for different k (fixed branch = TA)
    ax = axes[0]
    k_values = np.linspace(0.1, 0.9, 5)
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for k_val, color in zip(k_values, colors):
        beta_vals = []
        for T_norm in T_norm_range:
            inp = torch.FloatTensor([[k_val, 0.0, T_norm]]).to(device)
            with torch.no_grad():
                beta = net_beta(inp).cpu().numpy()[0, 0]
            beta_vals.append(beta)
        ax.plot(T_physical, beta_vals, color=color, linewidth=2, label=f'k={k_val:.2f}')

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('β', fontsize=12)
    ax.set_title('β(T) for TA branch at different k', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')

    # 2. β vs k for different T (fixed branch = TA)
    ax = axes[1]
    k_range = np.linspace(0.05, 0.95, 30)
    T_norm_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_norm_values)))

    for T_norm, color in zip(T_norm_values, colors):
        T_phys = T_ref + T_norm * (delta_T / 2)
        beta_vals = []
        for k_val in k_range:
            inp = torch.FloatTensor([[k_val, 0.0, T_norm]]).to(device)
            with torch.no_grad():
                beta = net_beta(inp).cpu().numpy()[0, 0]
            beta_vals.append(beta)
        ax.plot(k_range, beta_vals, color=color, linewidth=2, label=f'T={T_phys:.0f}K')

    ax.set_xlabel('Normalized k', fontsize=12)
    ax.set_ylabel('β', fontsize=12)
    ax.set_title('β(k) for TA branch at different T', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. β heatmap (k vs T)
    ax = axes[2]
    k_grid = np.linspace(0.05, 0.95, 30)
    T_grid = np.linspace(-1, 1, 30)
    K, TN = np.meshgrid(k_grid, T_grid)

    beta_map = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            inp = torch.FloatTensor([[K[i, j], 0.0, TN[i, j]]]).to(device)
            with torch.no_grad():
                beta_map[i, j] = net_beta(inp).cpu().numpy()[0, 0]

    T_phys_grid = T_ref + T_grid * (delta_T / 2)
    KK, TP = np.meshgrid(k_grid, T_phys_grid)
    cs = ax.contourf(KK, TP, beta_map, levels=30, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='β')
    ax.set_xlabel('Normalized k', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('β(k, T) for TA branch', fontsize=14)
    ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5)

    plt.suptitle(f'β-Network Visualization (T_ref={T_ref}K, ΔT={delta_T}K)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved β-network plot: {output_path}")


def plot_tau_temperature_dependence(
    T_ref: float,
    delta_T: float,
    tau_exponent: float,
    output_path: Path,
) -> None:
    """Visualize temperature-dependent relaxation time τ(T)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T_cold = T_ref - delta_T / 2
    T_hot = T_ref + delta_T / 2

    T_range = np.linspace(T_cold, T_hot, 100)
    tau_ratio = (T_ref / T_range) ** tau_exponent

    # 1. τ(T)/τ_ref vs T
    ax = axes[0]
    ax.plot(T_range, tau_ratio, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='τ_ref')
    ax.axvline(x=T_ref, color='r', linestyle='--', alpha=0.5, label='T_ref')

    # Mark key points
    ax.scatter([T_cold, T_ref, T_hot],
               [(T_ref/T_cold)**tau_exponent, 1.0, (T_ref/T_hot)**tau_exponent],
               c=['blue', 'red', 'orange'], s=100, zorder=5)

    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('τ(T) / τ_ref', fontsize=12)
    ax.set_title(f'τ(T) = τ_ref × (T_ref/T)^{tau_exponent:.1f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(f'T_cold={T_cold:.0f}K\nτ/τ_ref={tau_ratio[0]:.2f}',
                xy=(T_cold, tau_ratio[0]), xytext=(T_cold + 10, tau_ratio[0] + 0.3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'T_hot={T_hot:.0f}K\nτ/τ_ref={tau_ratio[-1]:.2f}',
                xy=(T_hot, tau_ratio[-1]), xytext=(T_hot - 30, tau_ratio[-1] + 0.3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

    # 2. τ(T)/τ_ref vs normalized T
    ax = axes[1]
    T_norm = (T_range - T_ref) / (delta_T / 2)
    ax.plot(T_norm, tau_ratio, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    ax.set_xlabel('Normalized Temperature (T - T_ref) / (ΔT/2)', fontsize=12)
    ax.set_ylabel('τ(T) / τ_ref', fontsize=12)
    ax.set_title('τ(T) in normalized coordinates', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Temperature-Dependent Relaxation Time\nT_ref={T_ref}K, ΔT={delta_T}K, α={tau_exponent}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved τ(T) plot: {output_path}")


def plot_large_dt_temperature_field(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    T_ref: float,
    delta_T: float,
    Lt: float,
    output_path: Path,
) -> None:
    """Plot temperature field in physical units (Kelvin) for large ΔT."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_phys = t * Lt

    # 1. Perturbation heatmap (T - T_ref)
    ax = axes[0, 0]
    X, Tgrid = np.meshgrid(x, t_phys)
    cs = ax.contourf(X, Tgrid, T, levels=50, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='T - T_ref (K)')
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('t / τ_ref')
    ax.set_title(f'Temperature Perturbation (T_ref={T_ref}K)')

    # 2. Physical temperature heatmap
    ax = axes[0, 1]
    T_physical = T + T_ref
    cs = ax.contourf(X, Tgrid, T_physical, levels=50, cmap='hot')
    plt.colorbar(cs, ax=ax, label='T (K)')
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('t / τ_ref')
    ax.set_title('Absolute Temperature Field')

    # 3. Temperature snapshots (physical)
    ax = axes[1, 0]
    n_snapshots = 6
    t_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    for idx, color in zip(t_indices, colors):
        ax.plot(x, T_physical[idx, :], '-', color=color, linewidth=2,
                label=f't={t_phys[idx]:.1f}τ')

    ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')
    ax.axhline(y=T_ref + delta_T/2, color='r', linestyle=':', alpha=0.5)
    ax.axhline(y=T_ref - delta_T/2, color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel('x (normalized)', fontsize=12)
    ax.set_ylabel('T (K)', fontsize=12)
    ax.set_title('Temperature Profiles')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Min/Max/Mean temperature evolution
    ax = axes[1, 1]
    T_min = np.min(T_physical, axis=1)
    T_max = np.max(T_physical, axis=1)
    T_mean = np.mean(T_physical, axis=1)

    ax.fill_between(t_phys, T_min, T_max, alpha=0.3, color='blue', label='Range')
    ax.plot(t_phys, T_mean, 'b-', linewidth=2, label='Mean')
    ax.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')
    ax.set_xlabel('t / τ_ref', fontsize=12)
    ax.set_ylabel('T (K)', fontsize=12)
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Large ΔT Temperature Analysis (T_ref={T_ref}K, ΔT={delta_T}K)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved large ΔT temperature plot: {output_path}")


def plot_large_dt_comprehensive(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    T_ref: float,
    delta_T: float,
    Lt: float,
    Kn_eff: float,
    tau_exponent: float,
    loss_history: list,
    output_path: Path,
    net_beta=None,
    device: str = 'cpu',
) -> dict:
    """Generate comprehensive validation figure for large ΔT trainer."""
    import torch

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    t_phys = t * Lt
    T_physical = T + T_ref
    T_cold = T_ref - delta_T / 2
    T_hot = T_ref + delta_T / 2

    # 1. Temperature heatmap (spans 2 cols)
    ax1 = fig.add_subplot(gs[0, 0:2])
    X, Tgrid = np.meshgrid(x, t_phys)
    cs = ax1.contourf(X, Tgrid, T, levels=50, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax1, label='T - T_ref (K)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t / τ_ref')
    ax1.set_title(f'Temperature Perturbation')

    # 2. Physical temperature profiles
    ax2 = fig.add_subplot(gs[0, 2])
    t_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_indices)))
    for idx, color in zip(t_indices, colors):
        ax2.plot(x, T_physical[idx, :], color=color, linewidth=2,
                 label=f't={t_phys[idx]:.1f}τ')
    ax2.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('T (K)')
    ax2.set_title('T(x) at different times')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Amplitude decay
    ax3 = fig.add_subplot(gs[0, 3])
    amplitude = np.max(T, axis=1) - np.min(T, axis=1)
    amplitude_norm = amplitude / amplitude[0] if amplitude[0] > 0 else amplitude
    ax3.semilogy(t_phys, amplitude_norm, 'b-', linewidth=2, label='PINN')
    ax3.set_xlabel('t / τ_ref')
    ax3.set_ylabel('A / A₀')
    ax3.set_title('Amplitude Decay')
    ax3.grid(True, alpha=0.3)

    # 4. τ(T) dependence
    ax4 = fig.add_subplot(gs[1, 0])
    T_range = np.linspace(T_cold, T_hot, 50)
    tau_ratio = (T_ref / T_range) ** tau_exponent
    ax4.plot(T_range, tau_ratio, 'b-', linewidth=2)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=T_ref, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('T (K)')
    ax4.set_ylabel('τ(T) / τ_ref')
    ax4.set_title(f'τ(T) = τ_ref × (T_ref/T)^{tau_exponent}')
    ax4.grid(True, alpha=0.3)

    # 5. β-network if available
    ax5 = fig.add_subplot(gs[1, 1])
    if net_beta is not None:
        T_norm_range = np.linspace(-1, 1, 30)
        k_values = [0.2, 0.5, 0.8]
        for k_val in k_values:
            beta_vals = []
            for T_norm in T_norm_range:
                inp = torch.FloatTensor([[k_val, 0.0, T_norm]]).to(device)
                with torch.no_grad():
                    beta = net_beta(inp).cpu().numpy()[0, 0]
                beta_vals.append(beta)
            T_phys_plot = T_ref + T_norm_range * (delta_T / 2)
            ax5.plot(T_phys_plot, beta_vals, linewidth=2, label=f'k={k_val}')
        ax5.set_xlabel('T (K)')
        ax5.set_ylabel('β')
        ax5.set_title('β(T) for TA branch')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'β-network\nnot available', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=14)
        ax5.axis('off')

    # 6. IC check
    ax6 = fig.add_subplot(gs[1, 2])
    T_ic = T[0, :]
    T_ic_target = delta_T / 2 * np.cos(2 * np.pi * x)
    ax6.plot(x, T_ic_target, 'k-', label='Target', linewidth=2)
    ax6.plot(x, T_ic, 'r--', label='PINN', linewidth=2)
    ic_rmse = np.sqrt(np.mean((T_ic - T_ic_target) ** 2))
    ax6.set_xlabel('x')
    ax6.set_ylabel('T - T_ref (K)')
    ax6.set_title(f'Initial Condition (RMSE={ic_rmse:.4f}K)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. BC check
    ax7 = fig.add_subplot(gs[1, 3])
    bc_error = np.abs(T[:, 0] - T[:, -1])
    ax7.semilogy(t_phys, bc_error + 1e-10, 'k-', linewidth=2)
    ax7.set_xlabel('t / τ_ref')
    ax7.set_ylabel('|T(0) - T(1)| (K)')
    ax7.set_title(f'Periodic BC Error (mean={np.mean(bc_error):.2e}K)')
    ax7.grid(True, alpha=0.3)

    # 8. Loss history
    ax8 = fig.add_subplot(gs[2, 0:2])
    loss_array = np.array(loss_history)
    labels = ['BTE', 'Energy', 'BC', 'IC_n', 'IC_T', 'Total']
    for i, label in enumerate(labels):
        if i < loss_array.shape[1]:
            ax8.semilogy(loss_array[:, i], label=label, alpha=0.7)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Loss')
    ax8.set_title('Training Loss History')
    ax8.legend(ncol=3)
    ax8.grid(True, alpha=0.3)

    # 9. T evolution at key points
    ax9 = fig.add_subplot(gs[2, 2])
    x_indices = [0, len(x)//4, len(x)//2]
    for xi in x_indices:
        ax9.plot(t_phys, T_physical[:, xi], linewidth=2, label=f'x={x[xi]:.2f}')
    ax9.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5)
    ax9.set_xlabel('t / τ_ref')
    ax9.set_ylabel('T (K)')
    ax9.set_title('T(t) at different x')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Metrics summary
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')

    metrics = {
        'ic_rmse': ic_rmse,
        'bc_mean_error': np.mean(bc_error),
        'final_amplitude': amplitude[-1],
        'decay_ratio': amplitude_norm[-1],
    }

    metrics_text = (
        f"Validation Metrics\n"
        f"─────────────────────\n"
        f"T_ref = {T_ref:.0f} K\n"
        f"ΔT = {delta_T:.0f} K\n"
        f"T range: [{T_cold:.0f}K, {T_hot:.0f}K]\n"
        f"Kn_eff = {Kn_eff:.4f}\n"
        f"τ exponent α = {tau_exponent}\n\n"
        f"IC RMSE: {ic_rmse:.4f} K\n"
        f"BC Mean Error: {np.mean(bc_error):.2e} K\n"
        f"Final Amplitude: {amplitude[-1]:.2f} K\n"
        f"Decay Ratio: {amplitude_norm[-1]:.4f}\n"
        f"Final Loss: {loss_array[-1, -1]:.2e}\n"
    )
    ax10.text(0.1, 0.9, metrics_text, transform=ax10.transAxes,
              fontsize=11, family='monospace', verticalalignment='top')

    plt.suptitle(f'Large ΔT Validation: T_ref={T_ref}K, ΔT={delta_T}K, Kn={Kn_eff:.4f}',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved large ΔT comprehensive validation: {output_path}")

    return metrics


def create_large_dt_animation(
    x: np.ndarray,
    t: np.ndarray,
    T: np.ndarray,
    T_ref: float,
    delta_T: float,
    Lt: float,
    output_path: Path,
    fps: int = 10,
) -> None:
    """Create animated GIF for large ΔT temperature evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T_physical = T + T_ref
    t_phys = t * Lt

    T_cold = T_ref - delta_T / 2
    T_hot = T_ref + delta_T / 2

    # Initial frames
    ax1 = axes[0]
    line1, = ax1.plot(x, T[0, :], 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-delta_T * 0.6, delta_T * 0.6)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('T - T_ref (K)', fontsize=12)
    ax1.set_title('Temperature Perturbation', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    line2, = ax2.plot(x, T_physical[0, :], 'r-', linewidth=2)
    ax2.axhline(y=T_ref, color='k', linestyle='--', alpha=0.5, label='T_ref')
    ax2.axhline(y=T_hot, color='r', linestyle=':', alpha=0.3)
    ax2.axhline(y=T_cold, color='b', linestyle=':', alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(T_cold - 10, T_hot + 10)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('T (K)', fontsize=12)
    ax2.set_title('Physical Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3)

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    plt.suptitle(f'Large ΔT Evolution (T_ref={T_ref}K, ΔT={delta_T}K)', fontsize=14)

    def update(frame):
        line1.set_ydata(T[frame, :])
        line2.set_ydata(T_physical[frame, :])
        time_text.set_text(f't = {t_phys[frame]:.2f} τ_ref')
        return line1, line2, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(t), interval=1000 // fps, blit=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()
    print(f"Saved large ΔT animation: {output_path}")


def generate_large_dt_visualizations(
    trainer,
    results: dict,
    loss_history: list,
    output_dir: Path,
    run_id: str,
) -> dict:
    """Generate all visualizations for large ΔT trainer."""
    x = results['x']
    t = results['t']
    T = results['T']  # Temperature perturbation in K
    T_ref = results.get('T_ref', trainer.T_ref)
    delta_T = results.get('delta_T', trainer.delta_T)
    Lt = trainer.Lt
    Kn_eff = trainer.Kn_eff
    tau_exponent = trainer.tau_exponent

    device = str(trainer.device)

    print(f"\nGenerating large ΔT visualizations for {run_id}...")

    # 1. Temperature heatmap
    plot_temperature_heatmap(
        x, t, T, Lt,
        output_dir / f"heatmap_{run_id}.png",
        title=f"Temperature Perturbation (T_ref={T_ref}K, ΔT={delta_T}K)"
    )

    # 2. Temperature snapshots
    plot_temperature_snapshots(
        x, t, T, Lt,
        output_dir / f"snapshots_{run_id}.png",
    )

    # 3. τ(T) dependence
    plot_tau_temperature_dependence(
        T_ref, delta_T, tau_exponent,
        output_dir / f"tau_T_{run_id}.png",
    )

    # 4. β-network visualization
    try:
        plot_beta_network(
            trainer.net_beta, T_ref, delta_T,
            trainer.Nk, trainer.Np,
            output_dir / f"beta_network_{run_id}.png",
            device=device,
        )
    except Exception as e:
        print(f"  Could not plot β-network: {e}")

    # 5. Large ΔT temperature field (physical units)
    plot_large_dt_temperature_field(
        x, t, T, T_ref, delta_T, Lt,
        output_dir / f"temperature_field_{run_id}.png",
    )

    # 6. Comprehensive validation
    try:
        metrics = plot_large_dt_comprehensive(
            x, t, T, T_ref, delta_T, Lt,
            Kn_eff, tau_exponent, loss_history,
            output_dir / f"comprehensive_{run_id}.png",
            net_beta=trainer.net_beta,
            device=device,
        )
    except Exception as e:
        print(f"  Could not create comprehensive plot: {e}")
        metrics = {}

    # 7. Animation
    try:
        create_large_dt_animation(
            x, t, T, T_ref, delta_T, Lt,
            output_dir / f"animation_{run_id}.gif",
            fps=10,
        )
    except Exception as e:
        print(f"  Could not create animation: {e}")

    # 8. Loss history
    plot_loss_history(
        loss_history,
        output_dir,
        run_id,
        labels=['BTE', 'Energy', 'BC', 'IC_n', 'IC_T', 'Total']
    )

    # 9. BC check
    bc_metrics = plot_boundary_condition_check(
        t, T, Lt,
        output_dir / f"bc_check_{run_id}.png",
        bc_type="periodic",
    )
    print(f"  Mean BC error: {bc_metrics.get('mean_bc_error', 0):.2e} K")

    # 10. IC check
    ic_metrics = plot_initial_condition_check(
        x, T,
        output_dir / f"ic_check_{run_id}.png",
        ic_type="cosine",
    )
    print(f"  IC RMSE: {ic_metrics.get('ic_rmse', 0):.4f} K")

    # Merge metrics
    metrics.update(bc_metrics)
    metrics.update(ic_metrics)

    return metrics
