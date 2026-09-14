"""1D transient thermal-grating runner for the sampled-residual control."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from pinn_bte.config.physics import UM_TO_ANGSTROM, ANGSTROM_TO_UM
from pinn_bte.training.transient import Transient1DNongrayLargeDTTrainer
from pinn_bte.utils.plotting_transient import (
    plot_temperature_heatmap,
    plot_temperature_snapshots,
    plot_boundary_condition_check,
    plot_initial_condition_check,
    plot_loss_history,
    create_temperature_animation,
)
from pinn_bte.utils.logging import console


MODES = {
    "nongray_large_dt": "Mode-resolved with β-network, large ΔT (~100K)",
}


def run_nongray_large_dt(args) -> dict:
    """Run the 1D mode-resolved large-ΔT transient experiment."""
    L = args.L * UM_TO_ANGSTROM  # μm to Å
    output_dir = Path(args.output) / f"1d_nongray_large_dt_{args.L}um_{int(args.delta_T)}K" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Transient1DNongrayLargeDTTrainer(
        L=L,
        T_ref=args.T_ref,
        delta_T=args.delta_T,
        epochs=args.epochs,
        epochs_beta=args.epochs_beta,
        learning_rate=args.lr,
        Nx=args.nx,
        Nt=args.nt,
        Ns=args.ns,
        Nk=args.nk,
        tau_exponent=args.tau_exponent,
        tau_model=args.tau_model,
        n_decay_times=args.n_decay_times,
        device=torch.device(args.device if torch.cuda.is_available() else 'cpu'),
        output_dir=output_dir,
        use_hard_ic=args.hard_ic,
        n_waypoints=args.n_waypoints,
        waypoint_weight=args.waypoint_weight,
        waypoint_shape=args.waypoint_shape,
        dtic_weight=args.dtic_weight,
        waypoint_spacing=args.waypoint_spacing,
        use_curriculum=args.curriculum,
        gamma_mode=args.gamma_mode,
        gamma_weight=args.gamma_weight,
        log_time=args.log_time,
        energy_weight=args.energy_weight,
        energy_moment_c=args.energy_moment_c,
        energy_norm=args.energy_norm,
        moment_weight=args.moment_weight,
        closure_ansatz=args.closure_ansatz,
        closure_tau_typ=args.closure_tau_typ,
        gamma_target_mode=args.gamma_target,
        window_crossing_thresh=args.window_crossing_thresh,
        amp_loss_weight=args.amp_loss_weight,
        use_spectral_t=args.spectral_t,
        spectral_K=args.spectral_K,
        meanrate_norm_scope=args.meanrate_norm_scope,
        signed_a1=args.signed_a1,
        max_principle_guard=args.max_principle_guard,
        max_principle_weight=args.max_principle_weight,
        amp_grid_Nx=args.amp_grid_nx,
        amp_grid_Nt=args.amp_grid_nt,
        arbiter_nmu_floor=args.arbiter_nmu_floor,
        arbiter_window_rule=args.arbiter_window_rule,
        net_arch=args.net_arch,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_symmetric_T=args.symmetric_T,
        subsample_N=args.subsample_N,
        seed=args.seed,
        angular_quad=args.angular_quad,
        waypoint_anneal_from=args.waypoint_anneal_from,
    )

    console.print(f"\n[bold cyan]1D Nongray Large ΔT Transient BTE[/bold cyan]")
    console.print(f"  L = {args.L} μm, ΔT = {args.delta_T} K")
    console.print(f"  T range: [{trainer.T_cold:.0f}K, {trainer.T_hot:.0f}K]")
    console.print(f"  Kn_eff = {trainer.Kn_eff:.4f}")
    console.print(f"  Epochs: {args.epochs} (β: {args.epochs_beta})")
    console.print(f"  Output: {output_dir}")

    loss_history = trainer.train()
    results = trainer.test()

    generate_visualizations(trainer, results, output_dir, trainer.run_id)
    plot_loss_history(loss_history, output_dir, trainer.run_id, labels=trainer.loss_names)
    plot_1d_amplitude_decay(results, trainer, output_dir, trainer.run_id)

    return {'trainer': trainer, 'results': results, 'loss_history': loss_history}


def plot_1d_amplitude_decay(results: dict, trainer, output_dir: Path, run_id: str):
    """Plot amplitude decay A(t) for 1D TTG validation."""
    import matplotlib.pyplot as plt

    x = results['x']
    t = results['t']
    T = results['T']

    Lt = getattr(trainer, 'Lt', 200.0)
    Kn_eff = getattr(trainer, 'Kn_eff', 0.1)
    L_um = getattr(trainer, 'L', 1e4) * ANGSTROM_TO_UM

    Nt_grid = len(t)

    A_pinn = np.zeros(Nt_grid)
    mean_T = np.zeros(Nt_grid)
    for i in range(Nt_grid):
        T_slice = T[i, :]  # spatial profile at time step i
        A_pinn[i] = (T_slice.max() - T_slice.min()) / 2
        mean_T[i] = T_slice.mean()

    t_physical = t * Lt  # in τ_ref units

    A_analytical = results.get('A_analytical', None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Amplitude decay (linear)
    ax = axes[0]
    ax.plot(t_physical, A_pinn, 'b-', lw=2, label='PINN')
    if A_analytical is not None:
        ax.plot(t_physical, A_analytical, 'r--', lw=2, label='Analytical (kinetic)')
    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('Amplitude A(t)')
    ax.set_title(f'Amplitude Decay (L={L_um:.1f}μm, Kn≈{Kn_eff:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Amplitude decay (log)
    ax = axes[1]
    ax.semilogy(t_physical, np.maximum(A_pinn, 1e-4), 'b-', lw=2, label='PINN')
    if A_analytical is not None:
        ax.semilogy(t_physical, np.maximum(A_analytical, 1e-4), 'r--', lw=2, label='Analytical')
    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('Amplitude A(t) [log]')
    ax.set_title('Log scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Energy conservation (mean T)
    ax = axes[2]
    ax.plot(t_physical, mean_T, 'g-', lw=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('t / τ_ref')
    ax.set_ylabel('Mean T(x,t)')
    ax.set_title(f'Energy Conservation (drift={mean_T[-1]:.4f})')
    ax.grid(True, alpha=0.3)

    red_flags = []
    if A_pinn[0] < 0.9:
        red_flags.append(f'IC: A(0)={A_pinn[0]:.3f} < 0.9')
    if abs(mean_T[-1]) > 0.1:
        red_flags.append(f'Drift: mean(T)={mean_T[-1]:.3f}')

    # Check for oscillation: A increases after initial decrease
    dA = np.diff(A_pinn)
    decreased = False
    for i, d in enumerate(dA):
        if d < -0.01:
            decreased = True
        if decreased and d > 0.02:
            red_flags.append(f'Oscillation at t≈{t_physical[i+1]:.0f}τ')
            break

    if red_flags:
        flag_text = 'RED FLAGS:\n' + '\n'.join(f'⚠ {f}' for f in red_flags)
        fig.text(0.5, 0.01, flag_text, ha='center', fontsize=11,
                 color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08 if red_flags else 0, 1, 1])
    save_path = output_dir / f'amplitude_decay_{run_id}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()

    decay_pct = (1 - A_pinn[-1] / A_pinn[0]) * 100 if A_pinn[0] > 0 else 0
    console.print(f"  Amplitude: A(0)={A_pinn[0]:.4f}, A(end)={A_pinn[-1]:.4f} ({decay_pct:.1f}% decay)")
    console.print(f"  Energy drift: {mean_T[-1]:.4f}")
    if red_flags:
        for f in red_flags:
            console.print(f"  [bold red]⚠ {f}[/bold red]")
    console.print(f"  Saved: {save_path}")


def generate_visualizations(trainer, results, output_dir: Path, run_id: str):
    """Generate standard visualizations for 1D trainers."""
    x = results['x']
    t = results['t']
    T = results['T']

    Lt = getattr(trainer, 'Lt', 3.0)
    L_um = getattr(trainer, 'L', UM_TO_ANGSTROM) * ANGSTROM_TO_UM
    Kn_eff = getattr(trainer, 'Kn_eff', getattr(trainer, 'eta', 0.6) / (2 * np.pi))

    console.print("  Generating visualizations...")

    plot_temperature_heatmap(
        x, t, T, Lt,
        output_dir / f"heatmap_{run_id}.png",
        title=f"Temperature Field (L={L_um:.1f}μm, Kn≈{Kn_eff:.3f})"
    )

    plot_temperature_snapshots(
        x, t, T, Lt,
        output_dir / f"snapshots_{run_id}.png"
    )

    plot_boundary_condition_check(
        t, T, Lt,
        output_dir / f"bc_check_{run_id}.png",
        bc_type="periodic"
    )

    A0 = getattr(trainer, 'A0', 1.0)
    plot_initial_condition_check(
        x, T,
        output_dir / f"ic_check_{run_id}.png",
        ic_type="cosine",
        amplitude=A0,
    )

    try:
        create_temperature_animation(
            x, t, T,
            output_dir / f"animation_{run_id}.gif",
            Lt=Lt,
            fps=15,
        )
    except Exception as e:
        console.print(f"  [yellow]Animation skipped: {e}[/yellow]")


MODE_DISPATCH = {
    "nongray_large_dt": run_nongray_large_dt,
}
assert set(MODE_DISPATCH) == set(MODES), (
    f"MODE_DISPATCH and MODES disagree: {set(MODES) ^ set(MODE_DISPATCH)}")

#: Modes whose IC is the cosine grating, so --hard-ic defaults on.
COSINE_IC_MODES = frozenset(MODES)


def main():
    parser = argparse.ArgumentParser(
        description="1D transient PINN-BTE experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<20} {v}" for k, v in MODES.items())
    )

    # Mode
    parser.add_argument("--mode", type=str, default="nongray_large_dt",
                        choices=list(MODES.keys()), help="Experiment mode")

    # Training params
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--epochs-beta", type=int, default=2000, help="β pretraining epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Physics params
    parser.add_argument("--L", type=float, default=1.0, help="Grating period (μm)")
    parser.add_argument("--T-ref", type=float, default=300.0, help="Reference temperature (K)")
    parser.add_argument("--delta-T", type=float, default=100.0, help="Temperature amplitude (K)")
    parser.add_argument("--tau-exponent", type=float, default=3.0, help="τ(T) exponent (power model)")
    parser.add_argument("--tau-model", type=str, default="power", choices=["power", "holland"],
                        help="τ(T) model: 'power' (uniform T^-α) or 'holland' (full branch-resolved τ(ω,T))")
    parser.add_argument("--n-decay-times", type=float, default=2.0,
                        help="Number of kinetic e-folding times. Under "
                             "--gamma-target dom/dom-crossing-union this is "
                             "the n_dec of the exponential window candidate "
                             "n_dec/γ_dom; the control sweep uses 7 (an "
                             "empirically pinned optimization requirement, "
                             "not a derived physical criterion). Recorded in "
                             "the run's npz.")

    # Hard IC + waypoint + curriculum params.  --hard-ic default depends on
    # the mode; resolved below after parse_args.
    parser.add_argument("--hard-ic", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Use hard IC constraint (T=cos+t·net, n=t·net). "
                             "Default True for cosine-IC modes.")
    parser.add_argument("--n-waypoints", type=int, default=5,
                        help="Number of temporal waypoints")
    parser.add_argument("--waypoint-weight", type=float, default=10.0,
                        help="Loss weight for waypoint constraints (0 = the data-free control)")
    parser.add_argument("--waypoint-spacing", type=str, default="uniform",
                        choices=["uniform", "log", "random-log"],
                        help="Waypoint time spacing: uniform, log (dense near t=0), or "
                             "random-log (fresh log-uniform times each epoch — continuous "
                             "trace supervision, no unsupervised gaps)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use progressive time curriculum training")

    # Continuous γ-constraint params (replaces waypoints)
    parser.add_argument("--gamma-mode", type=str, default="none",
                        choices=["none", "formula", "network"],
                        help="γ-constraint mode: none=waypoints, formula=analytical, network=learned")
    parser.add_argument("--gamma-weight", type=float, default=1.0,
                        help="Loss weight for continuous γ-constraint")

    # Time collocation
    parser.add_argument("--log-time", action="store_true", default=False,
                        help="Use log-spaced time collocation (denser near t=0)")

    # Loss balancing
    parser.add_argument("--energy-weight", type=str, default="auto",
                        help="Energy conservation weight: 'auto' (Kn-adaptive), or float (0=disable)")
    parser.add_argument("--energy-moment-c", action="store_true",
                        help="Carry the modal heat capacity c_m = beta*dfdT*T_scale "
                             "on the energy residual's storage term, making it the "
                             "EXACT dΩ moment of the BTE residual. The legacy "
                             "residual carries 1 instead and therefore does not "
                             "vanish on a solution of the BTE.")
    parser.add_argument("--energy-norm", type=str, default="legacy",
                        choices=["legacy", "moment"],
                        help="Energy-loss normaliser. 'legacy': output_scale^2 (the "
                             "CE amplitude of the unknown n). 'moment': the moment's "
                             "own scale, which makes the term the squared RELATIVE "
                             "conservation violation (dimensionless, O(1), "
                             "Kn-independent) and forces w_E=1 under "
                             "--energy-weight auto. Requires --energy-moment-c.")
    parser.add_argument("--amp-loss-weight", type=float, default=10.0,
                        help="Cosine-amplitude physics constraint weight (A(t)≥0 + dA/dt≤0). "
                             "0 disables.")
    parser.add_argument("--spectral-t", action="store_true",
                        help="Use spectral cosine-mode decomposition for T_norm. "
                             "T(x,t)=Σ a_k(t)cos(2πkx/L); a_1>0 monotone by construction. "
                             "Auto-disables --hard-ic.")
    parser.add_argument("--spectral-K", type=int, default=5,
                        help="Number of cosine modes when --spectral-t (default 5)")
    parser.add_argument("--signed-a1", action="store_true",
                        help="Release the spectral a_1 head's positivity/"
                             "monotonicity constraint via an extra raw channel "
                             "(a_1=(1+t·g_s)·exp(-t·softplus(g_0)); a_1(0)=1 kept exactly) "
                             "so the head can represent the DOM anti-phase grating "
                             "inversion. Requires --spectral-t. Default off.")
    parser.add_argument("--max-principle-guard", action="store_true",
                        help="Add the maximum-principle guard "
                             "mean_breach(relu(|T_norm|-1)^2) on the amplitude grid, "
                             "i.e. |T-T_ref| <= A0 = ΔT/2. The bound is the run's own "
                             "IC amplitude (no new constant) and the term is exactly "
                             "zero on any field respecting the maximum principle.")
    parser.add_argument("--amp-grid-nx", type=int, default=81,
                        help="x-resolution of the amplitude grid")
    parser.add_argument("--amp-grid-nt", type=int, default=81,
                        help="t-resolution of the amplitude grid (the guard bounds |T| "
                             "at these nodes only)")
    parser.add_argument("--max-principle-weight", type=float, default=None,
                        help="Weight of --max-principle-guard. Default (None) is DERIVED: "
                             "0.25 x --waypoint-weight.")
    parser.add_argument("--arbiter-nmu-floor", type=int, default=0,
                        help="Floor for the mu-quadrature of the trainer's DOM-reference "
                             "calls (gamma targets, shape curve, crossing window). The "
                             "fixed Nmu=32 default carries a ~+1%% gamma bias at the "
                             "ballistic period; 0 = off.")
    parser.add_argument("--arbiter-window-rule", type=str, default="full",
                        choices=["full", "legacy-earlystop"],
                        help="Integration-window rule of the DOM reference's gamma. "
                             "'full' (default) is what new runs must use. "
                             "'legacy-earlystop' replays the first-crossing truncation "
                             "bit-exactly, so a run recorded under it can be reproduced "
                             "under ITS OWN convention. Recorded in the npz as "
                             "`arbiter_window_rule`.")

    # Architecture
    parser.add_argument("--moment-weight", type=float, default=0.0,
                        help="μ² angular-moment weak-closure weight (0=off)")
    parser.add_argument("--waypoint-shape", type=str, default="rate",
                        choices=["rate", "dom-curve", "dom-curve-proj", "dom-rate-t", "dom-hybrid",
                                 "dom-hybrid-meanrate"],
                        help="waypoint target form: 'rate' (dT/dt=-gamma*T, exponential), "
                             "'dom-curve' (pointwise value-match of the signed DOM amplitude "
                             "trace; correct non-exponential shape at ballistic L), "
                             "'dom-curve-proj' (match only the a1 cosine projection), "
                             "'dom-rate-t' (da1/dt = -gamma_dom(t)*a1 on the projection: "
                             "locally satisfiable rate form with the full DOM kinetic "
                             "shape), or 'dom-hybrid-meanrate' (scale-free log-rate "
                             "residual normalized by the DOM mean window rate — "
                             "window-invariant; rate masked where |A_dom|<thresh, signed "
                             "DOM-relative value anchor)")
    parser.add_argument("--dtic-weight", type=float, default=60.0,
                        help="dT/dt-IC loss weight (0 disables)")
    parser.add_argument("--closure-ansatz", action="store_true",
                        help="Chapman–Enskog closure ansatz: bake the diffusive (Fourier) flux into "
                             "n_eff so γ can EMERGE without the analytical-rate waypoint (opt-in)")
    parser.add_argument("--closure-tau-typ", type=float, default=None,
                        help="closure ramp m(t) timescale in τ_ref units (default: 1·τ_ref)")
    parser.add_argument("--gamma-target", type=str, default="approx",
                        choices=["approx", "exact", "dom", "dom-crossing",
                                 "dom-crossing-union"],
                        help="waypoint/IC γ-target: 'approx'=gray 1/(1+Kn²) | 'exact'=linear-BTE "
                             "dispersion eigenvalue | 'dom'=DOM integral γ_eff (universal, "
                             "all-L incl. ballistic) | 'dom-crossing'=DOM γ_eff with the time "
                             "window sized by the last |A(t)|>=thresh·A0 crossing of the DOM "
                             "trace | 'dom-crossing-union'=the union rule "
                             "t_max = max(t_thresh, --n-decay-times/γ_dom)")
    parser.add_argument("--window-crossing-thresh", type=float, default=0.01,
                        help="crossing window threshold (fraction of A0)")
    parser.add_argument("--meanrate-norm-scope", type=str, default="rate",
                        choices=["rate", "whole", "none"],
                        help="which terms of the 'dom-hybrid-meanrate' waypoint loss the "
                             "reference window-mean pressure <P> divides: 'rate' (default) "
                             "= rate/<P> + anchor | 'whole' = (rate + anchor)/<P> | "
                             "'none' = rate + anchor, no division")
    parser.add_argument("--hidden-dim", type=int, default=40,
                        help="f^neq net (net_n) width — capacity knob for the BTE residual")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="f^neq net (net_n) depth")
    parser.add_argument("--net-arch", type=str, default="mlp", choices=["mlp", "pirate"],
                        help="Network architecture: mlp (standard) or pirate (adaptive residual gates)")
    parser.add_argument("--waypoint-anneal-from", type=float, default=0.0,
                        help="if >0: cosine-anneal waypoint+dtic weights to 0 over the last "
                             "(1-x) fraction of training (scaffold-removal emergence test)")
    parser.add_argument("--angular-quad", type=str, default="gl", choices=["gl", "uniform"],
                        help="mu quadrature: gl (Gauss-Legendre) | uniform (full-sphere "
                             "Omega_x marginal: dense grazing-angle sampling)")
    parser.add_argument("--symmetric-T", action="store_true",
                        help="Symmetrize T(x,t)=(raw(x)+raw(1-x))/2 for periodic BC by construction")

    # Mesh params
    parser.add_argument("--nx", type=int, default=60, help="Spatial points")
    parser.add_argument("--nt", type=int, default=60, help="Time points")
    parser.add_argument("--ns", type=int, default=16, help="Angular points")
    parser.add_argument("--nk", type=int, default=10, help="Frequency bands")
    parser.add_argument("--subsample-N", type=int, default=None,
                        help="Random subset of interior points per epoch (None=all)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default="runs/transient", help="Output directory")

    args = parser.parse_args()

    # Smart default for --hard-ic: enable for cosine-mode TTG runs.
    if args.hard_ic is None:
        args.hard_ic = args.mode in COSINE_IC_MODES

    console.print(f"\n[bold]Mode:[/bold] {args.mode} - {MODES[args.mode]}")

    MODE_DISPATCH[args.mode](args)

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
