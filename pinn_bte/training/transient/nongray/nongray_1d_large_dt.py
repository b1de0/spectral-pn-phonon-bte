"""Nongray 1D transient BTE trainer for LARGE temperature differences."""

import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from pinn_bte.models import Net, PirateNet
from pinn_bte.physics import compute_phonon_properties
from pinn_bte.physics import compute_relaxation_time_torch
from pinn_bte.physics.ttg_dom import (
    WINDOW_RULE_FULL,
    _WINDOW_RULES as ARBITER_WINDOW_RULES,
)
from pinn_bte.config.physics import (
    LATTICE_CONSTANT, HBAR, HKB, PRINT_FREQUENCY,
    VELOCITY_SCALE, FREQUENCY_SCALE, ANGSTROM_TO_UM,
    LOSS_NORM_EPSILON, energy_weight as _energy_weight,
)
from pinn_bte.utils.plotting_1d_transient import analytical_ttg_nongray
from pinn_bte.training.core.wandb_mixin import WandbMixin
from pinn_bte.training.transient.nongray.mode_utils import precompute_modes
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


class Transient1DNongrayLargeDTTrainer(WandbMixin):
    """Trainer for 1D nongray transient BTE with LARGE ΔT."""

    WAYPOINT_SHAPES = ('rate', 'dom-curve', 'dom-curve-proj', 'dom-rate-t',
                       'dom-hybrid', 'dom-hybrid-meanrate')

    DOM_TRACE_SHAPES = tuple(s for s in WAYPOINT_SHAPES if s != 'rate')

    DOM_VALUE_SHAPES = tuple(s for s in DOM_TRACE_SHAPES
                             if s.startswith('dom-curve'))

    DOM_RATE_SHAPES = tuple(filter(
        lambda s, _val=DOM_VALUE_SHAPES: s not in _val, DOM_TRACE_SHAPES))

    PLAIN_RATE_SHAPES = tuple(filter(
        lambda s, _dom=DOM_TRACE_SHAPES: s not in _dom, WAYPOINT_SHAPES))

    SIGNED_A1_UNSAFE_SHAPES = ('rate', 'dom-rate-t', 'dom-hybrid')

    GAMMA_TARGET_MODES = ('approx', 'exact', 'dom', 'dom-crossing',
                          'dom-crossing-union')

    #: Modes whose window comes from a PHYSICAL rate rather than the gray
    #: analytic decay probe.
    PHYSICAL_WINDOW_MODES = tuple(m for m in GAMMA_TARGET_MODES
                                  if m != 'approx')
    #: Modes whose gamma-target is the DOM integral rate (ttg_decay_spectral),
    #: the universal all-L ground truth, rather than the eigenvalue.
    DOM_GAMMA_MODES = tuple(m for m in GAMMA_TARGET_MODES
                            if m.startswith('dom'))
    #: Modes that run the 1%-crossing search (ttg_window_crossing).
    CROSSING_WINDOW_MODES = tuple(m for m in DOM_GAMMA_MODES if 'crossing' in m)
    #: Modes that UNION the crossing window with the n_decay/gamma one.
    UNION_WINDOW_MODES = tuple(m for m in CROSSING_WINDOW_MODES
                               if m.endswith('-union'))

    LT_RULES = ('gray', 'crossing', 'n_decay', 'floor')

    CLAMP_OCCUPANCY_BAR = 0.5
    CLAMP_OCCUPANCY_EPOCHS = 200

    WAYPOINT_SPACING_RANGE = {
        'uniform':    (0.1,   0.8,  False),
        'log':        (0.01,  0.85, True),
        'random-log': (1e-3,  0.9,  True),
    }

    WAYPOINT_SPACINGS = tuple(WAYPOINT_SPACING_RANGE)

    RANDOM_WAYPOINT_SPACINGS = tuple(s for s in WAYPOINT_SPACINGS
                                     if s.startswith('random-'))

    MEANRATE_NORM_SCOPES = ('rate', 'whole', 'none')

    MEANRATE_ANCHOR_FRACTION = 0.25

    MAX_PRINCIPLE_MAX_DT_TAU = 0.2

    CONTINUUM_EXCESS_PCT_PER_TAU2 = 1.9

    MAX_PRINCIPLE_BAR_FRAC = 1.01

    MAX_PRINCIPLE_TRIGGER_FRAC = 1.001
    MAX_PRINCIPLE_TRIGGER_EPOCHS = 200

    MAX_PRINCIPLE_MAX_DT_FRAC = 1.0 / 320.0

    MAX_PRINCIPLE_NODES_PER_FEATURE = 30.0

    SUP_GRID_CHUNK_POINTS = 1 << 18

    def __init__(
        self,
        L: float = 1e4,  # Domain length in Angstroms (1 μm = 10000 Å)
        T_ref: float = 300.0,  # Reference temperature (K)
        delta_T: float = 100.0,  # Temperature amplitude (K) - LARGE!
        epochs: int = 30000,
        epochs_beta: int = 5000,  # Epochs for β pretraining
        learning_rate: float = 1e-3,
        Nx: int = 60,
        Nt: int = 60,
        Ns: int = 20,  # Angular quadrature points
        Nk: int = 10,  # Frequency bands
        Np: int = 3,  # Phonon branches (TA1, TA2, LA)
        tau_exponent: float = 3.0,  # τ(T) = τ_ref × (T_ref/T)^α  (power-law model)
        tau_model: str = "power",  # 'power' (uniform T^-α) | 'holland' (full branch-resolved τ(ω,T))
        dopant_concentration: float = 0.0,
        n_decay_times: float = 5.0,  # Number of kinetic e-folding times to simulate
        device: Optional[torch.device] = None,
        output_dir: Optional[Path] = None,
        # Ballistic regime fix parameters
        use_hard_ic: bool = False,
        n_waypoints: int = 5,
        waypoint_weight: float = 10.0,
        waypoint_shape: str = 'rate',  # 'rate' (dT/dt=-gamma*T, imposes an
        dtic_weight: float = 60.0,
        ic_is_structural: bool = False,
        waypoint_spacing: str = 'uniform',
        angular_quad: str = 'gl',
        waypoint_anneal_from: float = 0.0,
        use_curriculum: bool = False,
        curriculum_stages: Optional[List[Tuple[float, float]]] = None,
        # Continuous γ-constraint parameters
        gamma_mode: str = 'none',  # 'none' | 'formula' | 'network'
        gamma_weight: float = 1.0,
        gamma_target_mode: str = 'approx',
        # Threshold for the 'dom-crossing' window (fraction of A0).
        window_crossing_thresh: float = 0.01,
        meanrate_norm_scope: str = 'rate',
        # Time collocation
        log_time: bool = True,  # Log-spaced time points (dense near t=0)
        # Loss balancing
        energy_weight: Union[str, float] = "auto",  # 'auto' (Kn-adaptive) or float (0=disable)
        energy_moment_c: bool = False,
        energy_norm: str = "legacy",
        moment_weight: float = 0.0,
        closure_ansatz: bool = False,
        closure_tau_typ: Optional[float] = None,  # ramp m(t) timescale (τ_ref units; None ⇒ 1·τ_ref)
        # Cosine-amplitude physics constraint: A(t) >= amp_floor and
        # dA/dt <= 0. amp_loss_weight=0 disables.
        amp_loss_weight: float = 10.0,
        amp_loss_floor: float = 0.5,  # Kelvin
        amp_grid_Nx: int = 81,
        amp_grid_Nt: int = 81,
        use_spectral_t: bool = False,
        spectral_K: int = 5,
        spectral_hidden: int = 32,
        spectral_layers: int = 4,
        signed_a1: bool = False,
        max_principle_guard: bool = False,
        # Weight of that guard. None ⇒ derived (see _max_principle_weight).
        max_principle_weight: Optional[float] = None,
        arbiter_nmu_floor: int = 0,
        arbiter_window_rule: str = WINDOW_RULE_FULL,
        # Architecture
        net_arch: str = "mlp",  # 'mlp' or 'pirate'
        hidden_dim: int = 40,  # f^neq net (net_n) width — capacity knob for the BTE residual
        num_layers: int = 8,   # f^neq net (net_n) depth
        use_symmetric_T: bool = False,  # Symmetrize T(x,t)=(raw(x)+raw(1-x))/2 for periodic BC
        # Spatial batching for large grids
        subsample_N: Optional[int] = None,  # Random subset of interior points per epoch (None=all)
        # Reproducibility
        seed: Optional[int] = None,
    ):
        self.L = L
        self.T_ref = T_ref
        self.delta_T = delta_T
        self.tau_exponent = tau_exponent
        self.tau_model = tau_model
        if dopant_concentration < 0:
            raise ValueError(
                f"dopant_concentration={dopant_concentration} is negative; "
                f"Matthiessen's rule would lengthen tau")
        self.dopant_concentration = float(dopant_concentration)
        self.n_decay_times = n_decay_times

        # Hard IC + waypoint + curriculum configuration
        self.use_hard_ic = use_hard_ic
        self.n_waypoints = n_waypoints
        self.waypoint_weight = waypoint_weight
        if waypoint_shape not in self.WAYPOINT_SHAPES:
            raise ValueError(f"unknown waypoint_shape {waypoint_shape!r}")
        self.waypoint_shape = waypoint_shape
        self.dtic_weight = dtic_weight
        self.ic_is_structural = ic_is_structural
        if waypoint_spacing not in self.WAYPOINT_SPACINGS:
            raise ValueError(
                f"unknown waypoint_spacing {waypoint_spacing!r}; expected one "
                f"of {self.WAYPOINT_SPACINGS}. (This flag used to have NO "
                f"validation: an unknown value silently trained the 'uniform' "
                f"law AND normalized by the 'uniform' <P>.)")
        self.waypoint_spacing = waypoint_spacing
        if angular_quad not in ('gl', 'uniform'):
            raise ValueError(
                f"angular_quad must be 'gl' or 'uniform', got {angular_quad!r}")
        self.angular_quad = angular_quad
        self.waypoint_anneal_from = waypoint_anneal_from
        self.use_curriculum = use_curriculum
        self.curriculum_stages = curriculum_stages or [(0.3, 0.4), (0.6, 0.3), (1.0, 0.3)]

        # Continuous γ-constraint configuration
        self.gamma_mode = gamma_mode
        self.gamma_weight = gamma_weight
        if gamma_target_mode not in self.GAMMA_TARGET_MODES:
            raise ValueError(
                f"unknown gamma_target_mode {gamma_target_mode!r}; "
                f"expected one of {self.GAMMA_TARGET_MODES}")
        self.gamma_target_mode = gamma_target_mode
        self.window_crossing_thresh = window_crossing_thresh
        if meanrate_norm_scope not in self.MEANRATE_NORM_SCOPES:
            raise ValueError(
                f"unknown meanrate_norm_scope {meanrate_norm_scope!r}; "
                f"expected one of {self.MEANRATE_NORM_SCOPES}")
        self.meanrate_norm_scope = meanrate_norm_scope
        # Window provenance, filled by _setup_mesh (NaN = candidate not
        # computed for this mode).
        self.Lt_rule = 'gray'
        self.Lt_candidate_crossing_tau = float('nan')
        self.Lt_candidate_ndecay_tau = float('nan')
        self.window_gamma_hz = float('nan')

        # Time collocation
        self.log_time = log_time

        # Loss balancing
        self.energy_weight = energy_weight
        self.energy_moment_c = bool(energy_moment_c)
        if energy_norm not in ('legacy', 'moment'):
            raise ValueError(
                f"energy_norm must be 'legacy' or 'moment', got {energy_norm!r}")
        self.energy_norm = energy_norm
        if self.energy_norm == 'moment' and not self.energy_moment_c:
            raise ValueError(
                "energy_norm='moment' requires energy_moment_c=True — Sₘ = "
                "4π·c_m^lin·γ̂ is the scale of the EXACT ∫dΩ moment, and the "
                "legacy residual (which carries 1 in place of c_m) is not it.")
        self.moment_weight = moment_weight
        self.closure_ansatz = closure_ansatz
        self.closure_tau_typ = closure_tau_typ
        self.amp_loss_weight = amp_loss_weight
        self.amp_loss_floor = amp_loss_floor
        self.amp_grid_Nx = amp_grid_Nx
        self.amp_grid_Nt = amp_grid_Nt
        # Spectral T_norm
        self.use_spectral_t = use_spectral_t
        self.spectral_K = spectral_K
        self.spectral_hidden = spectral_hidden
        self.spectral_layers = spectral_layers
        if signed_a1 and not use_spectral_t:
            raise ValueError(
                "signed_a1=True releases the spectral a_1 head's positivity "
                "constraint and requires use_spectral_t=True")
        if signed_a1 and waypoint_shape in self.SIGNED_A1_UNSAFE_SHAPES:
            raise ValueError(
                f"signed_a1=True with waypoint_shape={waypoint_shape!r} is "
                "unguarded: that objective does not charge the dead field "
                "(see SIGNED_A1_UNSAFE_SHAPES), while "
                "signed_a1 removes the structural positivity that made the "
                "collapse unreachable. Use 'dom-hybrid-meanrate' (charges it "
                "in both terms) or keep signed_a1=False.")
        self.signed_a1 = signed_a1
        self.max_principle_guard = max_principle_guard
        self._max_principle_weight_arg = max_principle_weight
        self.arbiter_nmu_floor = int(arbiter_nmu_floor)
        if arbiter_window_rule not in ARBITER_WINDOW_RULES:
            raise ValueError(
                f"unknown arbiter_window_rule {arbiter_window_rule!r}; "
                f"expected one of {ARBITER_WINDOW_RULES}")
        self.arbiter_window_rule = str(arbiter_window_rule)
        if self.use_spectral_t:
            if use_hard_ic:
                logger.info(
                    "use_spectral_t=True: forcing use_hard_ic=False "
                    "(spectral architecture handles IC by construction)."
                )
            self.use_hard_ic = False

        # Architecture
        self.net_arch = net_arch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_symmetric_T = use_symmetric_T
        self.subsample_N = subsample_N

        # Temperature range
        self.T_hot = T_ref + delta_T / 2  # e.g., 350K
        self.T_cold = T_ref - delta_T / 2  # e.g., 250K

        # Mesh parameters
        self.Nx = Nx
        self.Nt = Nt
        self.Ns = Ns
        self.Nk = Nk
        self.Np = Np

        # Training parameters
        self.epochs = epochs
        self.epochs_beta = epochs_beta
        self.learning_rate = learning_rate

        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output directory handling
        self.run_id = f"{time.strftime('%Y%m%d_%H%M%S')}"
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            dir_name = self.output_dir.name
            if len(dir_name) == 15 and dir_name[8] == "_":
                self.run_id = dir_name
        else:
            base_dir = Path("runs/transient")
            self.output_dir = (
                base_dir
                / f"1d_nongray_large_dt_{int(L * ANGSTROM_TO_UM)}um_{int(delta_T)}K"
                / self.run_id
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility: set seeds before any random operations
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Random seed set: {seed}")

        # Prepare physics data
        self._prepare_data()

        # Initialize networks
        self._init_networks()

        # Compute numerical dT/dt(x,0) target with T-dependent τ
        self._compute_dTdt_initial()

        # Loss names
        self.loss_names = ['bte', 'energy_cons', 'periodic_bc', 'ic_n', 'ic_T', 'dt_ic', 'waypoint', 'global_energy', 'amp_phys']
        if self.max_principle_guard:
            self.loss_names.append('max_principle')
            logger.info(
                f"Maximum-principle guard ENABLED: "
                f"mean_breach(relu(|T_norm|-1)²) on the {self.amp_grid_Nt}×"
                f"{self.amp_grid_Nx} amplitude grid, weight "
                f"{self._max_principle_weight:.4g} "
                f"(|T-T_ref| <= A0 = {self.delta_T / 2:.1f} K)")
            dt_node = self._max_principle_dt_tau
            dt_frac = 1.0 / max(self.amp_grid_Nt - 1, 1)
            logger.info(
                "Guard grid: dt' = 1/%d (portable criterion <= 1/%d: %s), "
                "dt = %.4g τ_ref (absolute criterion <= %.2g τ_ref: %s)",
                self.amp_grid_Nt - 1,
                int(round(1.0 / self.MAX_PRINCIPLE_MAX_DT_FRAC)),
                "MET" if dt_frac <= self.MAX_PRINCIPLE_MAX_DT_FRAC else "NOT met",
                dt_node, self.MAX_PRINCIPLE_MAX_DT_TAU,
                "MET" if dt_node <= self.MAX_PRINCIPLE_MAX_DT_TAU else "NOT met")
            if dt_frac > self.MAX_PRINCIPLE_MAX_DT_FRAC:
                logger.warning(
                    "maximum-principle guard UNDER-RESOLVED (portable form): "
                    "dt' = 1/%d > 1/%d (amp_grid_Nt=%d). A collocation bound "
                    "leaves a spike BETWEEN nodes; measured on the round-3 L=1 "
                    "head the continuous sup then sits ~%.2f%% of A0 above the "
                    "bound while every saved frame reads exactly A0. Pass "
                    "--amp-grid-nt >= %d.",
                    self.amp_grid_Nt - 1,
                    int(round(1.0 / self.MAX_PRINCIPLE_MAX_DT_FRAC)),
                    self.amp_grid_Nt,
                    self.CONTINUUM_EXCESS_PCT_PER_TAU2 * dt_node ** 2,
                    int(np.ceil(1.0 / self.MAX_PRINCIPLE_MAX_DT_FRAC)) + 1)
            # ... and the PHYSICAL criterion, checked against this rung's own
            # arbiter so a uniform --amp-grid-nt verifies itself per arm.
            t_half = self._dom_half_decay_tau()
            if np.isfinite(t_half) and t_half > 0:
                nodes = (self.amp_grid_Nt - 1) * t_half / self.Lt
                need = int(np.ceil(self.MAX_PRINCIPLE_NODES_PER_FEATURE
                                   * self.Lt / t_half)) + 1
                (logger.info if nodes >= self.MAX_PRINCIPLE_NODES_PER_FEATURE
                 else logger.warning)(
                    "Guard grid: %.1f nodes across the arbiter half-decay "
                    "(%.4g τ_ref = %.4g of the window); criterion >= %.0f, "
                    "which needs amp_grid_Nt >= %d at this rung: %s",
                    nodes, t_half, t_half / self.Lt,
                    self.MAX_PRINCIPLE_NODES_PER_FEATURE, need,
                    "MET" if nodes >= self.MAX_PRINCIPLE_NODES_PER_FEATURE
                    else "NOT met")
            if dt_node > self.MAX_PRINCIPLE_MAX_DT_TAU:
                logger.info(
                    "Guard grid does NOT resolve the ABSOLUTE dt <= %.2g τ_ref "
                    "form (would need amp_grid_Nt >= %d at Lt = %.4g τ). This "
                    "is EXPECTED and unfixable as a TRAINING grid at the "
                    "diffusive rungs (L=10/100 µm need ~2.1e4 / ~2.0e6 nodes "
                    "per epoch); the guard there is a coarse safety net, and "
                    "F-MP1 is NOT scored on it — the falsifier uses the "
                    "test-time sup grid (_max_principle_sup), which refines to "
                    "all three criteria independently of this one.",
                    self.MAX_PRINCIPLE_MAX_DT_TAU,
                    int(np.ceil(self.Lt / self.MAX_PRINCIPLE_MAX_DT_TAU)) + 1,
                    self.Lt)

        # Log hard IC configuration
        if self.use_hard_ic:
            logger.info(
                f"Hard IC enabled: T = cos(2πx) + t·net_T, n = t·net_n | "
                f"Curriculum: {self.use_curriculum}"
            )
        if self.gamma_mode != 'none':
            logger.info(
                f"Continuous γ-constraint: mode={self.gamma_mode}, weight={self.gamma_weight}"
            )
        else:
            logger.info(
                f"Waypoints: {self.n_waypoints} ({self.waypoint_spacing}), weight={self.waypoint_weight}, shape={self.waypoint_shape}"
            )

        if self.log_time:
            logger.info(f"Log-spaced time collocation: {self.Nt} points, 50% in [0, 0.1·Lt]")
        if self.energy_weight == "auto":
            ew_val = _energy_weight(self.Kn_grating)
            logger.info(f"Auto energy weight: {ew_val:.3f} (grating Kn={self.Kn_grating:.4f}, Kn_eff={self.Kn_eff:.4f})")
        elif float(self.energy_weight) != 1.0:
            logger.info(f"Energy conservation weight: {self.energy_weight}")

        # β pretrained flag
        self.beta_pretrained = False

        # Store BC/IC/heat source types
        self.bc_type = BCType.PERIODIC
        self.ic_type = ICType.COSINE
        self.heat_source_type = HeatSourceType.NONE

        # Create composable components via factories
        self.bc = create_bc(BCType.PERIODIC, include_distribution=False)
        self.ic = create_ic(ICType.COSINE, amplitude=1.0)
        self.heat_source = create_heat_source(HeatSourceType.NONE)

        # Initialize W&B with standardized tags
        _L_um = str(L * ANGSTROM_TO_UM)
        _dT = str(int(delta_T))
        self.__init_wandb__(
            experiment_name=get_transient_wandb_group('1d', 'nongray', 'large_dt'),
            tags=get_transient_wandb_tags(
                dimension='1d',
                model='nongray',
                bc_left=BCType.PERIODIC,
                bc_right=BCType.PERIODIC,
                ic_type=ICType.COSINE,
                heat_source=HeatSourceType.NONE,
                delta_T_regime='large_dt',
                L_um=_L_um,
                delta_T_K=_dT,
                has_beta_network='true',
            ),
            run_name=build_transient_run_name(
                '1d', 'nongray', 'large_dt', L_um=_L_um, delta_T_K=_dT,
            ),
        )

    def _prepare_data(self):
        """Prepare phonon properties, compute normalization scales, and precompute
        expanded tensors.
        """
        Nx, Nt, Ns, Nk, Np = self.Nx, self.Nt, self.Ns, self.Nk, self.Np
        T_ref = self.T_ref

        # Spatial and time grids (normalized to [0, 1])
        xm = np.linspace(0, 1, Nx)
        if self.log_time:
            # Log-spaced: dense near t=0 where physics changes fastest
            # logspace(-2, 0) gives [0.01, ..., 1.0], prepend t=0
            tm = np.concatenate([[0.0], np.logspace(-2, 0, Nt - 1)])
        else:
            tm = np.linspace(0, 1, Nt)
        x_grid, t_grid = np.meshgrid(xm, tm)
        x = x_grid.reshape(-1, 1)  # (Nx*Nt, 1)
        t = t_grid.reshape(-1, 1)  # (Nx*Nt, 1)

        if self.angular_quad == 'uniform':
            mu = -1.0 + (2 * np.arange(Ns) + 1.0) / Ns
            w = np.full(Ns, 2.0 / Ns)
        else:
            mu, w = np.polynomial.legendre.leggauss(Ns)
        mu = mu.reshape(-1, 1)
        w = w.reshape(-1, 1) * 2 * np.pi

        # Frequency bands (normalized wavevector)
        k_norm = np.linspace(0.05, 0.95, Nk).reshape(-1, 1)
        k_physical = k_norm * (np.pi * 2 / LATTICE_CONSTANT)

        # Phonon branches: TA1, TA2 (degenerate), LA
        branch = np.vstack([
            np.zeros_like(k_norm),  # TA1
            np.zeros_like(k_norm),  # TA2
            np.ones_like(k_norm),   # LA
        ])

        # Compute phonon properties at reference temperature
        k_tiled = np.tile(k_physical, (Np, 1))
        omega, vk, D, tau, dfdT = compute_phonon_properties(
            k_tiled, branch, T_ref,
            dopant_concentration=self.dopant_concentration,
        )

        # Reshape to (Np, Nk)
        omega = omega.reshape(Np, Nk)
        vk = vk.reshape(Np, Nk)  # In units of 10^13 Å/s
        D = D.reshape(Np, Nk)
        tau = tau.reshape(Np, Nk)  # In seconds
        dfdT = dfdT.reshape(Np, Nk)

        # === COMPUTE REFERENCE SCALES ===
        C = HBAR * FREQUENCY_SCALE * omega * D * dfdT  # Modal heat capacity
        v_ref = np.sqrt(np.sum(C * vk**2) / np.sum(C))
        tau_ref = np.sum(C) / np.sum(C / tau)

        # Effective Knudsen number
        Kn_eff = v_ref * VELOCITY_SCALE * tau_ref / self.L

        # Store reference scales
        self.v_ref = v_ref
        self.tau_ref = tau_ref
        self.Kn_eff = Kn_eff
        self.Kn_grating = 2.0 * np.pi * Kn_eff

        # Normalized velocities and relaxation times
        vk_norm = vk / v_ref
        tau_norm = tau / tau_ref

        threshold = np.exp(-self.n_decay_times)
        # Estimate t_probe range from slowest mode decay rate
        q = 2 * np.pi / self.L
        v_phys = vk.flatten() * VELOCITY_SCALE  # Å/s
        tau_flat = tau.flatten()  # s
        Kn_modes = v_phys * tau_flat * q
        gamma_modes = (v_phys * q) ** 2 * tau_flat / (3.0 * (1 + Kn_modes ** 2))  # 1/s; 1/3=<mu^2>
        gamma_slowest = gamma_modes.min()
        t_needed_tau = self.n_decay_times / (gamma_slowest * tau_ref)  # τ_ref units
        t_probe_max = max(500.0, 2 * t_needed_tau)
        t_probe = np.linspace(0, t_probe_max, 5000)
        # vk is in paper units (10^13 Å/s); analytical function needs Å/s
        A_analytical = analytical_ttg_nongray(
            t_probe, vk * VELOCITY_SCALE, tau, C, self.L, tau_ref
        )
        # Find first time where A < threshold
        below = np.where(A_analytical < threshold)[0]
        if len(below) > 0:
            Lt_analytical = t_probe[below[0]]
        else:
            Lt_analytical = t_probe[-1]
        self.Lt = max(Lt_analytical, 3.0)  # Floor at 3τ (gray)

        if self.gamma_target_mode in self.PHYSICAL_WINDOW_MODES:
            g_phys, Lt_sel = None, None
            if self.gamma_target_mode in self.DOM_GAMMA_MODES:
                from pinn_bte.physics.ttg_dispersion import shipped_mode_source
                from pinn_bte.physics.ttg_dom import ttg_decay_spectral
                with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                    g_phys = ttg_decay_spectral(self.L * ANGSTROM_TO_UM, '1d',
                                                Nk=self.Nk, T_ref=self.T_ref,
                                                window_rule=self.arbiter_window_rule,
                                                **self._nmu_kw)
            else:
                from pinn_bte.physics.ttg_dispersion import (
                    dominant_gamma, shipped_mode_source)
                with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                    gd, ok = dominant_gamma(self.L * ANGSTROM_TO_UM, '1d',
                                            Nk=self.Nk, T_ref=self.T_ref)
                g_phys = gd.real if ok else None  # ballistic eigenvalue absent → keep gray
            if g_phys and g_phys > 0:
                self.window_gamma_hz = float(g_phys)
                self.Lt_candidate_ndecay_tau = self.n_decay_times / (g_phys * tau_ref)
            if self.gamma_target_mode in self.CROSSING_WINDOW_MODES:
                from pinn_bte.physics.ttg_dispersion import shipped_mode_source
                from pinn_bte.physics.ttg_dom import ttg_window_crossing
                with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                    t_cross_s = ttg_window_crossing(
                        self.L * ANGSTROM_TO_UM, '1d', Nk=self.Nk,
                        thresh=self.window_crossing_thresh, T_ref=self.T_ref,
                        window_rule=self.arbiter_window_rule, **self._nmu_kw)
                self.Lt_candidate_crossing_tau = t_cross_s / tau_ref
                Lt_sel, self.Lt_rule = self.Lt_candidate_crossing_tau, 'crossing'
                if (self.gamma_target_mode in self.UNION_WINDOW_MODES
                        and self.Lt_candidate_ndecay_tau > Lt_sel):
                    Lt_sel, self.Lt_rule = self.Lt_candidate_ndecay_tau, 'n_decay'
                self.Lt = max(Lt_sel, 0.2)
                logger.info(
                    # NB: no square brackets — the rich console handler eats
                    # them as markup and the winning rule vanishes from the log.
                    f"Lt from {self.gamma_target_mode} rule={self.Lt_rule}: "
                    f"{self.Lt:.4f} τ_ref = {self.Lt * tau_ref:.3e} s "
                    f"(t_1% at {self.window_crossing_thresh:g}·A0: "
                    f"{self.Lt_candidate_crossing_tau:.4f} τ_ref; "
                    f"n_dec={self.n_decay_times:g}/γ_dom: "
                    f"{self.Lt_candidate_ndecay_tau:.4f} τ_ref)")
            elif g_phys and g_phys > 0:
                Lt_sel = self.Lt_candidate_ndecay_tau
                self.Lt = max(Lt_sel, 0.2)
                self.Lt_rule = 'n_decay'
                logger.info(f"Lt from {self.gamma_target_mode} γ: {self.Lt:.3f} τ_ref "
                            f"(gray Lt was {max(Lt_analytical, 3.0):.3f})")
            if Lt_sel is not None and self.Lt > Lt_sel:
                self.Lt_rule = 'floor'   # the 0.2 tau numerical floor bound
        logger.info(
            f"Analytical Lt: A(Lt)={threshold:.3f} threshold, "
            f"Lt_analytical={Lt_analytical:.1f}τ, Lt_final={self.Lt:.1f}τ"
        )

        if self.energy_norm == 'moment' and not np.isfinite(self.window_gamma_hz):
            raise ValueError(
                f"energy_norm='moment' needs the window rate γ̂, but "
                f"window_gamma_hz is not set under gamma_target_mode="
                f"{self.gamma_target_mode!r} (Lt_rule={self.Lt_rule!r}). Use a "
                f"γ-target in PHYSICAL_WINDOW_MODES (e.g. 'dom-crossing-union').")
        if self.energy_norm == 'moment':
            logger.info(
                f"Energy normalisation: MOMENT (derivation 2). "
                f"γ̂={self.window_gamma_hz * tau_ref:.6g}/τ_ref, w_E=1 "
                f"(Kn-independent by construction; the heuristic would have "
                f"given {_energy_weight(self.Kn_grating):.3e} at grating "
                f"Kn={self.Kn_grating:.4f})")

        # Store raw phonon data for β training
        self.omega_raw = omega
        self.dfdT_raw = dfdT
        self.k_physical = k_physical

        # Initial/boundary samples
        xi = np.linspace(0, 1, Nx).reshape(-1, 1)
        tb = np.linspace(0, 1, Nt + 2)[1:Nt + 1].reshape(-1, 1)

        # Store data
        self.data = {
            'x': torch.FloatTensor(x).to(self.device),  # (Nx*Nt, 1) meshed
            't': torch.FloatTensor(t).to(self.device),  # (Nx*Nt, 1) meshed
            'mu': torch.FloatTensor(mu).to(self.device),
            'w': torch.FloatTensor(w).to(self.device),
            'k_norm': torch.FloatTensor(k_norm).to(self.device),
            'k_physical': torch.FloatTensor(k_physical).to(self.device),
            'omega': torch.FloatTensor(omega).to(self.device),
            'vk_norm': torch.FloatTensor(vk_norm).to(self.device),
            'tau_norm': torch.FloatTensor(tau_norm).to(self.device),
            'tau_ref': torch.FloatTensor(tau.reshape(Np, Nk)).to(self.device),
            'dfdT': torch.FloatTensor(dfdT).to(self.device),
            'C': torch.FloatTensor(C).to(self.device),
            'xi': torch.FloatTensor(xi).to(self.device),
            'tb': torch.FloatTensor(tb).to(self.device),
        }

        # === PRECOMPUTE EXPANDED TENSORS FOR VECTORIZED ANGLES ===
        N = Nx * Nt  # Interior grid size
        self._N = N

        # Interior: (N,1) → (N*Ns,1)
        self._x_exp = self.data['x'].repeat(1, Ns).reshape(-1, 1)
        self._t_exp = self.data['t'].repeat(1, Ns).reshape(-1, 1)
        self._mu_exp = self.data['mu'].squeeze().repeat(N).reshape(-1, 1)

        # BC: (Nt,1) → (Nt*Ns,1)
        n_bc = tb.shape[0]
        self._tb_exp = self.data['tb'].repeat(1, Ns).reshape(-1, 1)
        self._mu_bc_exp = self.data['mu'].squeeze().repeat(n_bc).reshape(-1, 1)

        # IC: (Nx,1) → (Nx*Ns,1)
        n_ic = xi.shape[0]
        self._xi_exp = self.data['xi'].repeat(1, Ns).reshape(-1, 1)
        self._mu_ic_exp = self.data['mu'].squeeze().repeat(n_ic).reshape(-1, 1)

        # mu as (N, Ns) for BTE residual computation
        self._mu_2d = self.data['mu'].squeeze().unsqueeze(0).expand(N, -1)

        # Store full grid for subsampling
        self._N_full = N
        self._x_full = self.data['x'].clone()
        self._t_full = self.data['t'].clone()

        Nx_amp = self.amp_grid_Nx
        Nt_amp = self.amp_grid_Nt
        x_amp = np.linspace(0.0, 1.0, Nx_amp).astype(np.float32)
        t_amp = np.linspace(0.0, 1.0, Nt_amp).astype(np.float32)
        xx_amp, tt_amp = np.meshgrid(x_amp, t_amp, indexing='xy')  # (Nt_amp, Nx_amp)
        self._amp_x = torch.from_numpy(xx_amp.reshape(-1, 1)).to(self.device)
        self._amp_t = torch.from_numpy(tt_amp.reshape(-1, 1)).to(self.device)
        self._amp_cos_x = torch.from_numpy(np.cos(2 * np.pi * x_amp)).to(self.device)
        self._amp_dt_norm = 1.0 / (Nt_amp - 1)

        logger.info(f"Large ΔT model: L={self.L * ANGSTROM_TO_UM:.1f}μm, ΔT={self.delta_T}K")
        logger.info(f"Temperature range: [{self.T_cold:.0f}K, {self.T_hot:.0f}K]")
        logger.info(f"Reference scales: v_ref={v_ref:.2e}, τ_ref={tau_ref:.2e}")
        logger.info(f"Effective Kn={Kn_eff:.4f}, Lt={self.Lt:.1f}")
        if self.closure_ansatz:
            _ttyp = self.closure_tau_typ if self.closure_tau_typ is not None else 1.0
            logger.info(f"Chapman–Enskog closure ansatz ENABLED (ramp τ_typ={_ttyp}·τ_ref): "
                        f"diffusive flux baked into n_eff; γ may emerge without the waypoint")
        logger.info(f"Vectorized: ALL {Np*Nk} modes × ALL {Ns} angles per epoch")
        if self.subsample_N is not None and self.subsample_N < N:
            logger.info(f"Subsampling: {self.subsample_N}/{N} interior points per epoch")

    def _subsample_grid(self):
        """Randomly subsample interior collocation points for this epoch."""
        if self.subsample_N is None or self.subsample_N >= self._N_full:
            return

        idx = torch.randperm(self._N_full, device=self.device)[:self.subsample_N]
        self.data['x'] = self._x_full[idx]
        self.data['t'] = self._t_full[idx]

        N = self.subsample_N
        Ns = self.Ns
        self._x_exp = self.data['x'].repeat(1, Ns).reshape(-1, 1)
        self._t_exp = self.data['t'].repeat(1, Ns).reshape(-1, 1)
        self._mu_exp = self.data['mu'].squeeze().repeat(N).reshape(-1, 1)
        self._mu_2d = self.data['mu'].squeeze().unsqueeze(0).expand(N, -1)
        self._N = N

    def _init_networks(self):
        """Initialize neural networks."""
        NetClass = PirateNet if self.net_arch == "pirate" else Net

        if self.use_spectral_t:
            out_dim = self.spectral_K + (1 if self.signed_a1 else 0)
            seq = [nn.Linear(1, self.spectral_hidden), nn.SiLU()]
            for _ in range(self.spectral_layers - 1):
                seq += [nn.Linear(self.spectral_hidden, self.spectral_hidden), nn.SiLU()]
            seq += [nn.Linear(self.spectral_hidden, out_dim)]
            self.net_T = nn.Sequential(*seq).to(self.device)
        else:
            self.net_T = NetClass(
                input_dim=2,
                hidden_dim=40,
                num_layers=8,
            ).to(self.device)

        self.net_n = NetClass(
            input_dim=5,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_checkpointing=True,
        ).to(self.device)

        # β-network: (k_norm, branch, T_norm) → β scaling factor
        self.net_beta = Net(
            input_dim=3,
            hidden_dim=30,
            num_layers=1,
        ).to(self.device)

        # Optimizers
        self.optimizer_T = optim.Adam(
            self.net_T.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )
        self.optimizer_n = optim.Adam(
            self.net_n.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )
        self.optimizer_beta = optim.Adam(
            self.net_beta.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )

        # Weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        self.net_T.apply(init_weights)
        self.net_n.apply(init_weights)
        self.net_beta.apply(init_weights)

        # γ-network (Variant B): small net mapping T_input → γ
        if self.gamma_mode == 'network':
            self.net_gamma = Net(
                input_dim=1,
                hidden_dim=20,
                num_layers=1,
            ).to(self.device)
            self.net_gamma.apply(init_weights)
            self.optimizer_gamma = optim.Adam(
                self.net_gamma.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.99),
                eps=1e-10,
            )

    def _amp_forward(self, net, x):
        """Forward pass with BFloat16 autocast, returning float32 output."""
        if self.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = net(x)
            return out.float()
        return net(x)

    def _T_forward(self, x, t):
        """Temperature network forward."""
        if self.use_spectral_t:
            # net_T: (B, 1) → (B, K [+1 if signed_a1]) raw coefficients g_k(t)
            g = self._amp_forward(self.net_T, t)
            exponent = t * F.softplus(g[:, 0:1])
            cap = self._a1_exponent_cap
            if cap is not None:
                # occupancy accounting for the §12.4 runtime guard
                self._clamp_hits = (getattr(self, '_clamp_hits', 0.0)
                                    + float((exponent > cap).sum()))
                self._clamp_total = (getattr(self, '_clamp_total', 0.0)
                                     + float(exponent.numel()))
                exponent = torch.clamp(exponent, max=cap)
            a_1 = torch.exp(-exponent)
            if self.signed_a1:
                a_1 = (1.0 + t * g[:, self.spectral_K:self.spectral_K + 1]) * a_1
            # a_k(t) = g_k(t)·a_1^(k²)·(1-a_1) for k≥2: damped, bounded,
            # zero at t=0 and t→∞ (see docstring).
            k_h2 = torch.arange(2, self.spectral_K + 1,
                                dtype=x.dtype, device=x.device) ** 2
            # Envelope base |a_1| under signed_a1: pow(negative, float k²)
            # is NaN in torch; flag off uses a_1 itself (identical tensor).
            env = a_1.abs() if self.signed_a1 else a_1
            a_higher = g[:, 1:self.spectral_K] * torch.pow(env, k_h2) * (1.0 - a_1)
            # Cosine basis: shape (B, K)
            k_idx = torch.arange(1, self.spectral_K + 1,
                                 dtype=x.dtype, device=x.device)
            cos_basis = torch.cos(2 * np.pi * k_idx * x)
            a_all = torch.cat([a_1, a_higher], dim=1)
            return (a_all * cos_basis).sum(dim=1, keepdim=True)
        raw = self._amp_forward(self.net_T, torch.cat([x, t], dim=1))
        if self.use_symmetric_T:
            raw_mirror = self._amp_forward(self.net_T, torch.cat([1.0 - x, t], dim=1))
            return (raw + raw_mirror) / 2
        return raw

    def pretrain_beta(self) -> List[float]:
        """Pretrain β-network to approximate nonlinear f^eq(T)."""
        console.print("\n[bold cyan]Phase 1: Pretraining β-network[/bold cyan]")
        console.print(f"  Temperature range: [{self.T_cold:.0f}K, {self.T_hot:.0f}K]")
        console.print(f"  Epochs: {self.epochs_beta}")

        self.net_beta.train()

        # Sample temperature perturbations
        num_temps = 50
        T_norm_samples = np.linspace(-1, 1, num_temps).reshape(-1, 1)
        T_physical = self.T_ref + T_norm_samples * (self.delta_T / 2)

        Nk, Np = self.Nk, self.Np
        k_norm = self.data['k_norm'].cpu().numpy()

        # Prepare training tensors
        all_k = []
        all_branch = []
        all_T_norm = []
        all_df = []
        all_dfdT = []

        for p in range(Np):
            for ik in range(Nk):
                omega_val = self.omega_raw[p, ik]
                dfdT_val = self.dfdT_raw[p, ik]

                for it in range(num_temps):
                    T_val = T_physical[it, 0]
                    T_norm_val = T_norm_samples[it, 0]

                    exp_arg_ref = HKB * omega_val / self.T_ref
                    exp_arg_T = HKB * omega_val / T_val

                    f_ref = 1.0 / (np.exp(min(exp_arg_ref, 50)) - 1 + 1e-30)
                    f_T = 1.0 / (np.exp(min(exp_arg_T, 50)) - 1 + 1e-30)

                    df = f_T - f_ref

                    all_k.append(k_norm[ik, 0])
                    all_branch.append(float(p) / (Np - 1) if Np > 1 else 0.0)
                    all_T_norm.append(T_norm_val)
                    all_df.append(df)
                    all_dfdT.append(dfdT_val)

        # Convert to tensors
        k_tensor = torch.FloatTensor(all_k).reshape(-1, 1).to(self.device)
        branch_tensor = torch.FloatTensor(all_branch).reshape(-1, 1).to(self.device)
        T_norm_tensor = torch.FloatTensor(all_T_norm).reshape(-1, 1).to(self.device)
        df_tensor = torch.FloatTensor(all_df).reshape(-1, 1).to(self.device)
        dfdT_tensor = torch.FloatTensor(all_dfdT).reshape(-1, 1).to(self.device)

        dT_physical = T_norm_tensor * (self.delta_T / 2)

        loss_history = []
        start_time = time.time()

        for epoch in range(self.epochs_beta):
            self.optimizer_beta.zero_grad()

            beta_input = torch.cat([k_tensor, branch_tensor, T_norm_tensor], dim=1)
            beta = self.net_beta(beta_input)

            df_predicted = beta * dT_physical * dfdT_tensor

            df_safe = df_tensor + torch.sign(df_tensor) * 1e-30
            residual = df_predicted / df_safe - 1

            loss = torch.mean(residual ** 2)

            loss.backward()
            self.optimizer_beta.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if epoch % 500 == 0:
                elapsed = time.time() - start_time
                console.print(
                    f"  {elapsed:6.1f}s | Epoch {epoch:5d} | "
                    f"β Loss = {loss_val:.2e}"
                )

        # Freeze β-network
        for param in self.net_beta.parameters():
            param.requires_grad = False
        self.net_beta.eval()
        self.beta_pretrained = True

        console.print(f"  [green]β pretraining complete![/green] Final loss: {loss_history[-1]:.2e}")

        # Save β network
        torch.save(
            self.net_beta.state_dict(),
            self.output_dir / f"net_beta_{self.run_id}.pt"
        )

        # === γ-network pretraining (Variant B only) ===
        if self.gamma_mode == 'network':
            console.print("\n[bold cyan]Phase 1b: Pretraining γ-network[/bold cyan]")
            n_gamma_samples = 200
            T_gamma_samples = torch.linspace(self.T_cold, self.T_hot, n_gamma_samples).to(self.device)
            T_norm_samples = (T_gamma_samples - self.T_ref) / (self.delta_T / 2)
            gamma_targets = self._compute_gamma_normalized(T_norm_samples.reshape(-1, 1))

            gamma_epochs = 500
            for epoch in range(gamma_epochs):
                self.optimizer_gamma.zero_grad()
                T_input = ((T_gamma_samples - self.T_cold) / self.delta_T).reshape(-1, 1)
                gamma_pred = F.softplus(self.net_gamma(T_input))
                loss_gamma = torch.mean((gamma_pred - gamma_targets) ** 2)
                loss_gamma.backward()
                self.optimizer_gamma.step()
                if epoch % 100 == 0:
                    console.print(f"  γ pretrain epoch {epoch:4d} | loss={loss_gamma.item():.2e}")
            console.print(f"  [green]γ pretraining complete![/green] Final loss: {loss_gamma.item():.2e}")
            torch.save(
                self.net_gamma.state_dict(),
                self.output_dir / f"net_gamma_{self.run_id}.pt"
            )

        return loss_history

    def _holland_branch(self, p: int) -> float:
        """Dispersion branch for the Holland model: LA (last index) -> 1, else TA
        -> 0.
        """
        return 1.0 if p == (self.Np - 1) else 0.0

    def _tau_holland_ratio(self, T_physical: torch.Tensor, p: int, ik: int) -> torch.Tensor:
        k_phys = float(self.k_physical[ik, 0]) if self.k_physical.ndim == 2 \
            else float(self.k_physical[ik])
        branch = self._holland_branch(p)
        k_t = torch.full_like(T_physical, k_phys)
        b_t = torch.full_like(T_physical, branch)
        tref_t = torch.full_like(T_physical, self.T_ref)
        tau_T = compute_relaxation_time_torch(
            k_t, b_t, T_physical,
            dopant_concentration=self.dopant_concentration)
        tau_ref = compute_relaxation_time_torch(
            k_t, b_t, tref_t,
            dopant_concentration=self.dopant_concentration)
        return tau_T / tau_ref

    def compute_tau_T(self, T_norm: torch.Tensor, tau_ref_val: float,
                      p: Optional[int] = None, ik: Optional[int] = None) -> torch.Tensor:
        """Compute temperature-dependent relaxation time."""
        T_physical = self.T_ref + T_norm * (self.delta_T / 2)
        T_physical = torch.clamp(T_physical, min=100.0, max=500.0)
        if self.tau_model == "holland" and p is not None and ik is not None:
            tau_ratio = self._tau_holland_ratio(T_physical, p, ik)
        else:
            tau_ratio = (self.T_ref / T_physical) ** self.tau_exponent
        return tau_ref_val * tau_ratio

    def _compute_dTdt_initial(self):
        """Compute numerical dT/dt(x,0) accounting for T-dependent τ."""
        Nk, Np = self.Nk, self.Np

        # IC spatial points (same as training IC points)
        x_ic = np.linspace(0, 1, self.Nx)
        T_ic_norm = np.cos(2 * np.pi * x_ic)  # T_norm at t=0
        T_ic_phys = self.T_ref + T_ic_norm * (self.delta_T / 2)  # Physical T

        # Phonon properties at T_ref (stored from _prepare_data)
        vk = self.data['vk_norm'].cpu().numpy() * self.v_ref  # Back to paper units
        tau = self.data['tau_norm'].cpu().numpy() * self.tau_ref  # Back to seconds

        C = self.data['C'].cpu().numpy()  # (Np, Nk)

        q = 2 * np.pi / self.L  # TTG wavevector in Å⁻¹

        # For each spatial point, compute mode-resolved γ with local τ(T)
        dTdt_target = np.zeros(len(x_ic))

        for ix, T_phys in enumerate(T_ic_phys):
            gamma_weighted_sum = 0.0
            C_sum = 0.0
            for p in range(Np):
                for ik in range(Nk):
                    v_phys = vk[p, ik] * VELOCITY_SCALE  # Å/s
                    if self.tau_model == "holland":
                        Tt = torch.tensor([[float(T_phys)]], device=self.device)
                        tau_local = tau[p, ik] * float(self._tau_holland_ratio(Tt, p, ik).item())
                    else:
                        tau_local = tau[p, ik] * (self.T_ref / T_phys) ** self.tau_exponent
                    Kn_mode = v_phys * tau_local * q
                    gamma_mode = (v_phys * q) ** 2 * tau_local / (3.0 * (1 + Kn_mode ** 2))  # 1/3=<mu^2>
                    gamma_weighted_sum += C[p, ik] * gamma_mode
                    C_sum += C[p, ik]
            gamma_eff = gamma_weighted_sum / C_sum  # Effective γ at this x
            dTdt_target[ix] = -gamma_eff * T_ic_norm[ix] * (self.delta_T / 2)  # Physical dT/dt (K/s)

        if self.gamma_target_mode in self.PHYSICAL_WINDOW_MODES:
            helper = (self._exact_gamma_no13 if self.gamma_target_mode == 'exact'
                      else self._dom_gamma_no13)
            gamma_eff_arr = helper(np.asarray(T_ic_phys, dtype=float))
            dTdt_target = -gamma_eff_arr * np.asarray(T_ic_norm, dtype=float) * (self.delta_T / 2)

        if self.waypoint_shape in self.DOM_TRACE_SHAPES:
            tt_s, A_s = self._dom_shape_curve()
            slope_s = (A_s[1] - A_s[0]) / (tt_s[1] - tt_s[0])
            dTdt_target = slope_s * np.asarray(
                np.cos(2 * np.pi * x_ic), dtype=float) * (self.delta_T / 2)

        T_scale = self.delta_T / 2
        dTdt_target_norm = dTdt_target / T_scale * (self.Lt * self.tau_ref)

        # Store as tensor for use in loss
        self.dTdt_initial = torch.FloatTensor(
            dTdt_target_norm.reshape(-1, 1)
        ).to(self.device)

        logger.info(
            f"dT/dt IC (large ΔT): target range [{dTdt_target_norm.min():.3f}, "
            f"{dTdt_target_norm.max():.3f}] in [0,1] time"
        )

    @property
    def _nmu_kw(self):
        return ({'nmu_floor': self.arbiter_nmu_floor}
                if self.arbiter_nmu_floor else {})

    def _exact_gamma_no13(self, T_phys_np):
        """Exact dominant-eigenvalue γ (no-1/3 scale, Hz) interpolated at T_phys."""
        import numpy as _np
        if getattr(self, '_eg_T', None) is None:
            from pinn_bte.physics.ttg_dispersion import (
                dominant_gamma, shipped_mode_source)
            L_um = self.L * ANGSTROM_TO_UM
            Tg = _np.linspace(self.T_cold, self.T_hot, 41)
            gg = _np.empty_like(Tg)
            with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                for i, T in enumerate(Tg):
                    g, ok = dominant_gamma(L_um, '1d', Nk=self.Nk, T_ref=float(T))
                    gg[i] = g.real if ok else _np.nan  # physical (carries 1/3); trainer is physical now
            if _np.any(_np.isnan(gg)):  # ballistic holes — fall back to nearest valid
                good = ~_np.isnan(gg)
                gg = _np.interp(Tg, Tg[good], gg[good])
            self._eg_T, self._eg_g = Tg, gg
            logger.info(
                f"Exact-eigenvalue γ-target: grid [{Tg[0]:.0f},{Tg[-1]:.0f}]K, "
                f"γ_no13 ∈ [{gg.min():.2e},{gg.max():.2e}] Hz (replaces 1/(1+Kn²))"
            )
        return _np.interp(_np.asarray(T_phys_np, dtype=float), self._eg_T, self._eg_g)

    def _dom_gamma_no13(self, T_phys_np):
        """DOM integral γ_eff (no-1/3 scale, Hz) interpolated at T_phys — the
        UNIVERSAL target.
        """
        import numpy as _np
        if getattr(self, '_dg_T', None) is None:
            from pinn_bte.physics.ttg_dispersion import shipped_mode_source
            from pinn_bte.physics.ttg_dom import ttg_decay_spectral
            L_um = self.L * ANGSTROM_TO_UM
            Tg = _np.linspace(self.T_cold, self.T_hot, 21)
            with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                gg = _np.array([
                    ttg_decay_spectral(L_um, '1d', Nk=self.Nk, T_ref=float(T),
                                       window_rule=self.arbiter_window_rule,
                                       **self._nmu_kw)
                    for T in Tg])  # physical (carries 1/3)
            self._dg_T, self._dg_g = Tg, gg
            logger.info(
                f"DOM γ-target (universal, all-L): grid [{Tg[0]:.0f},{Tg[-1]:.0f}]K, "
                f"γ_no13 ∈ [{gg.min():.2e},{gg.max():.2e}] Hz (replaces 1/(1+Kn²))"
            )
        return _np.interp(_np.asarray(T_phys_np, dtype=float), self._dg_T, self._dg_g)

    def _compute_gamma_normalized(self, T_norm: torch.Tensor) -> torch.Tensor:
        """Compute analytical dimensionless γ_eff at each point."""
        Nk, Np = self.Nk, self.Np

        T_phys = self.T_ref + T_norm.detach() * (self.delta_T / 2)
        T_phys = torch.clamp(T_phys, min=100.0, max=500.0)

        if self.gamma_target_mode in self.PHYSICAL_WINDOW_MODES:
            helper = (self._exact_gamma_no13 if self.gamma_target_mode == 'exact'
                      else self._dom_gamma_no13)
            gp = helper(T_phys.cpu().numpy())  # physical Hz (no-1/3 scale)
            gp = torch.as_tensor(gp, dtype=T_norm.dtype, device=T_norm.device).reshape(T_norm.shape)
            return gp * (self.Lt * self.tau_ref)  # dimensionless

        vk = self.data['vk_norm'].cpu().numpy() * self.v_ref
        tau = self.data['tau_norm'].cpu().numpy() * self.tau_ref
        C = self.data['C'].cpu().numpy()

        q = 2 * np.pi / self.L

        gamma_sum = torch.zeros_like(T_phys)
        C_total = 0.0

        for p in range(Np):
            for ik in range(Nk):
                v_phys = vk[p, ik] * VELOCITY_SCALE
                tau_ref_mode = tau[p, ik]
                C_mode = C[p, ik]

                if self.tau_model == "holland":
                    tau_local = tau_ref_mode * self._tau_holland_ratio(T_phys, p, ik)
                else:
                    tau_local = tau_ref_mode * (self.T_ref / T_phys) ** self.tau_exponent
                Kn_mode = v_phys * tau_local * q
                gamma_mode = (v_phys * q) ** 2 * tau_local / (3.0 * (1 + Kn_mode ** 2))  # 1/3=<mu^2>

                gamma_sum += C_mode * gamma_mode
                C_total += C_mode

        gamma_phys = gamma_sum / C_total  # Physical γ (1/s)
        return gamma_phys * (self.Lt * self.tau_ref)  # Dimensionless

    def _dom_shape_curve(self):
        """Cached SIGNED DOM amplitude trace (t_s, A/A0) covering the run window,
        used as the waypoint/dt-IC SHAPE target when waypoint_shape ==
        'dom-curve'.
        """
        if getattr(self, '_dshape', None) is None:
            from pinn_bte.physics.ttg_dispersion import shipped_mode_source
            from pinn_bte.physics.ttg_dom import ttg_amplitude_curve
            L_um = self.L * ANGSTROM_TO_UM
            t_end = 1.05 * self.Lt * self.tau_ref
            with shipped_mode_source(Nk=self.Nk, T_ref=self.T_ref):
                tt, A = ttg_amplitude_curve(L_um, '1d', Nk=self.Nk,
                                            T_ref=self.T_ref, t_end_s=t_end,
                                            **self._nmu_kw)
            self._dshape = (tt, A)
            logger.info(
                f"DOM shape curve: {len(tt)} pts over {t_end:.3e}s, "
                f"min A/A0 = {A.min():+.4f} (non-monotone at ballistic L)")
        return self._dshape

    def _wp_proj_grid(self, n: int = 64):
        if getattr(self, '_x_proj', None) is None:
            self._x_proj = ((torch.arange(n, device=self.device, dtype=torch.float32)
                             + 0.5) / n).reshape(-1, 1)
        return self._x_proj

    def _dom_gamma_of_t(self, t_frac):
        """Instantaneous DOM decay rate gamma_n(t) = -A'/A in NORMALIZED time
        units, interpolated at fractional times t_frac of the run window.
        """
        if getattr(self, '_dgamma', None) is None:
            tt_s, A_s = self._dom_shape_curve()
            gam_s = -np.gradient(A_s, tt_s) / np.clip(A_s, 1e-6, None)  # 1/s
            self._dgamma = (tt_s, gam_s * self.Lt * self.tau_ref)
        tt_s, gam_n = self._dgamma
        return np.interp(np.asarray(t_frac) * self.Lt * self.tau_ref, tt_s, gam_n)

    def _dom_gamma_of_t_signed(self, t_frac):
        """Signed-trace local log-rate for 'dom-hybrid-meanrate': gamma_n(t) =
        -A'/A with the SIGNED A (|A| floored at 1e-6 preserving sign).
        """
        if getattr(self, '_dgamma_signed', None) is None:
            tt_s, A_s = self._dom_shape_curve()
            den = np.where(np.abs(A_s) < 1e-6,
                           np.where(A_s < 0, -1e-6, 1e-6), A_s)
            gam_s = -np.gradient(A_s, tt_s) / den  # 1/s
            self._dgamma_signed = (tt_s, gam_s * self.Lt * self.tau_ref)
        tt_s, gam_n = self._dgamma_signed
        return np.interp(np.asarray(t_frac) * self.Lt * self.tau_ref, tt_s, gam_n)

    def _dom_phi_max(self):
        """Under the dom-crossing criterion this is ln(1/thresh) = ln 100 BY
        DEFINITION (the criterion is self-normalizing). Cached.
        """
        if getattr(self, '_phi_max', None) is None:
            tt_s, A_s = self._dom_shape_curve()
            A_end = np.interp(self.Lt * self.tau_ref, tt_s, A_s)
            self._phi_max = float(-np.log(max(abs(float(A_end)), 1e-12)))
        return self._phi_max

    @property
    def _a1_exponent_cap(self):
        if self.waypoint_shape != 'dom-hybrid-meanrate':
            return None
        if getattr(self, '_a1_cap', None) is None:
            self._a1_cap = 2.0 * self._dom_phi_max()
        return self._a1_cap

    @property
    def _a1_structural_floor(self):
        """The value anchor's log clamp uses it so head and anchor go flat at the
        SAME amplitude and the collapse basin keeps a restoring gradient
        everywhere the head can reach.
        """
        return float(np.exp(-self._a1_exponent_cap))

    def _wp_placement_quadrature(self, n=4096):
        lo, hi, geom = self.WAYPOINT_SPACING_RANGE[self.waypoint_spacing]
        return (np.geomspace(lo, hi, n) if geom else np.linspace(lo, hi, n))

    def _draw_waypoint_times(self, t_max_frac):
        """The per-epoch waypoint times, in window fraction, on the SAME law."""
        lo, hi, geom = self.WAYPOINT_SPACING_RANGE[self.waypoint_spacing]
        if self.waypoint_spacing in self.RANDOM_WAYPOINT_SPACINGS:
            u = np.random.rand(self.n_waypoints)
            return lo * (hi / lo) ** u * t_max_frac
        if geom:
            return np.geomspace(lo * t_max_frac, hi * t_max_frac,
                                self.n_waypoints)
        return np.linspace(lo * t_max_frac, hi * t_max_frac, self.n_waypoints)

    @property
    def _meanrate_pressure_norm(self):
        if getattr(self, '_pnorm', None) is None:
            t = self._wp_placement_quadrature()
            P = (self._dom_gamma_of_t_signed(t) / self._dom_phi_max()) ** 2
            val = float(np.mean(np.where(self._wp_rate_mask(t), P, 0.0)))
            if not np.isfinite(val) or val <= 0.0:
                logger.warning(
                    "meanrate pressure normalization degenerate (<P>=%r); "
                    "falling back to 1.0 (no normalization)", val)
                val = 1.0
            self._pnorm = val
            logger.info(
                f"Meanrate pressure normalization <P> = {val:.4f} "
                f"(arbiter window-mean; uniform rate error eps costs eps²)")
        return self._pnorm

    @staticmethod
    def _max_principle_loss(T_norm: torch.Tensor) -> torch.Tensor:
        """Maximum-principle guard (report §15.4 item 1, the §12.4 structural
        amplitude guard with a measured target).
        """
        v = torch.relu(T_norm.abs() - 1.0)
        n_breach = (v > 0).sum()
        if n_breach == 0:
            return v.sum() * 0.0           # exact zero, graph-connected
        return (v ** 2).sum() / n_breach

    @property
    def _max_principle_dt_tau(self) -> float:
        """Amplitude-grid time step in tau_ref — the resolution at which the guard
        actually bounds the field (see MAX_PRINCIPLE_MAX_DT_TAU).
        """
        return float(self.Lt) / max(self.amp_grid_Nt - 1, 1)

    def _dom_half_decay_tau(self) -> float:
        """The arbiter's fastest feature: the first time |A_dom| <= 0.5, in
        tau_ref.
        """
        tt_s, A_s = self._dom_shape_curve()
        below = np.nonzero(np.abs(A_s) <= 0.5)[0]
        if not len(below):
            return float('nan')
        return float(tt_s[below[0]]) / self.tau_ref

    def _max_principle_sup(self, chunk_points: Optional[int] = None) -> dict:
        A0 = self.delta_T / 2
        Lt = float(self.Lt)
        t_half = self._dom_half_decay_tau()
        nt_feature = (int(np.ceil(self.MAX_PRINCIPLE_NODES_PER_FEATURE
                                  * Lt / t_half)) + 1
                      if np.isfinite(t_half) and t_half > 0 else 0)
        Nx = self.amp_grid_Nx
        Nt = max(self.amp_grid_Nt,
                 int(np.ceil(1.0 / self.MAX_PRINCIPLE_MAX_DT_FRAC)) + 1,
                 int(np.ceil(Lt / self.MAX_PRINCIPLE_MAX_DT_TAU)) + 1,
                 nt_feature)
        dt_tau = Lt / max(Nt - 1, 1)
        dt_frac = 1.0 / max(Nt - 1, 1)
        x_row = torch.linspace(0.0, 1.0, Nx, device=self.device).reshape(1, -1)
        chunk_t = max(1, (chunk_points or self.SUP_GRID_CHUNK_POINTS) // Nx)
        node = 0.0
        with torch.no_grad():
            for i0 in range(0, Nt, chunk_t):
                n = min(chunk_t, Nt - i0)
                t_col = (torch.arange(i0, i0 + n, device=self.device,
                                      dtype=torch.float32)
                         / max(Nt - 1, 1)).reshape(-1, 1)
                xx = x_row.expand(n, Nx).reshape(-1, 1)
                tt = t_col.expand(n, Nx).reshape(-1, 1)
                T = self._T_forward(xx, tt)
                if self.use_hard_ic:
                    # under the hard-IC ansatz _T_forward returns the RAW head
                    T = torch.cos(2 * np.pi * xx) + tt * T
                node = max(node, float(T.abs().max()))
        node_K = node * A0
        guard_active = bool(self.max_principle_guard) and node_K >= A0
        correction_K = (self.CONTINUUM_EXCESS_PCT_PER_TAU2 * dt_tau ** 2
                        * A0 / 100.0) if guard_active else 0.0
        return dict(max_abs_T_nodes_K=node_K,
                    max_abs_T_sup_K=node_K + correction_K,
                    sup_correction_K=correction_K,
                    sup_correction_applied=guard_active,
                    sup_grid_Nt=Nt, sup_grid_Nx=Nx, sup_grid_dt_tau=dt_tau,
                    sup_grid_dt_frac=dt_frac,
                    sup_grid_t_half_tau=t_half,
                    sup_grid_nodes_per_feature=(
                        (Nt - 1) * t_half / Lt if np.isfinite(t_half) else float('nan')),
                    sup_grid_admissible=bool(
                        dt_frac <= self.MAX_PRINCIPLE_MAX_DT_FRAC
                        and (not np.isfinite(t_half) or (Nt - 1) * t_half / Lt
                             >= self.MAX_PRINCIPLE_NODES_PER_FEATURE)))

    @property
    def _max_principle_weight(self) -> float:
        """Price of the maximum-principle guard, DERIVED from the objective's own
        pinned price for a relative amplitude error rather than picked.
        """
        if self._max_principle_weight_arg is not None:
            return float(self._max_principle_weight_arg)
        return self.MEANRATE_ANCHOR_FRACTION * self.waypoint_weight

    def _check_max_principle_trigger(self, epoch):
        mx = getattr(self, '_max_abs_T_last', None)
        if mx is None:
            return
        if mx > self.MAX_PRINCIPLE_TRIGGER_FRAC * self.delta_T / 2:
            self._mp_trigger_streak = getattr(self, '_mp_trigger_streak', 0) + 1
        else:
            self._mp_trigger_streak = 0
        self._mp_trigger_max_streak = max(
            getattr(self, '_mp_trigger_max_streak', 0), self._mp_trigger_streak)
        if self._mp_trigger_streak == self.MAX_PRINCIPLE_TRIGGER_EPOCHS:
            self._mp_trigger_fired = True
            logger.warning(
                "maximum-principle ESCALATION TRIGGER: raw node max |T| = "
                "%.4f K > %.4f K for %d consecutive epochs (through epoch %d). "
                "The guard is doing corrective work, not sitting inactive. "
                "REPORTED ONLY — escalate the weight by hand on the measured "
                "ladder (%.4g -> 1.28e3 -> 1.30e4) and re-check the 50%%-decay "
                "lobe against the Richardson ladder if you do.",
                mx, self.MAX_PRINCIPLE_TRIGGER_FRAC * self.delta_T / 2,
                self.MAX_PRINCIPLE_TRIGGER_EPOCHS, epoch,
                self._max_principle_weight)

    def _clamp_occupancy(self):
        total = getattr(self, '_clamp_total', 0.0)
        frac = (getattr(self, '_clamp_hits', 0.0) / total) if total else 0.0
        self._clamp_hits = self._clamp_total = 0.0
        return float(frac)

    def _check_clamp_occupancy(self, epoch):
        """Aborts after CLAMP_OCCUPANCY_EPOCHS consecutive epochs above
        CLAMP_OCCUPANCY_BAR. No-op unless the clamp is active.
        """
        if self._a1_exponent_cap is None:
            return 0.0
        frac = self._clamp_occupancy()
        self._clamp_occupancy_last = frac
        if frac > self.CLAMP_OCCUPANCY_BAR:
            self._clamp_streak = getattr(self, '_clamp_streak', 0) + 1
        else:
            self._clamp_streak = 0
        if self._clamp_streak >= self.CLAMP_OCCUPANCY_EPOCHS:
            raise RuntimeError(
                f"a_1 clamp occupancy {frac:.2f} > {self.CLAMP_OCCUPANCY_BAR} "
                f"for {self._clamp_streak} consecutive epochs (through epoch "
                f"{epoch}): the head is living at the 2·phi_max exponent cap, "
                "i.e. running away to the dead field. Optimization guard bar, "
                "not a physics constant (ruling §12.4).")
        return frac

    def _wp_rate_mask(self, t_frac):
        tt_s, A_s = self._dom_shape_curve()
        A_t = np.interp(np.asarray(t_frac) * self.Lt * self.tau_ref, tt_s, A_s)
        return np.abs(A_t) >= self.window_crossing_thresh

    @staticmethod
    def _meanrate_residual(da1_dt, a1, gamma_n, phi_max, a1_floor):
        signed_floor = torch.where(a1 >= 0, torch.full_like(a1, a1_floor),
                                   torch.full_like(a1, -a1_floor))
        denom = torch.where(a1.abs() >= a1_floor, a1, signed_floor)
        return (da1_dt / denom + gamma_n) / phi_max

    @staticmethod
    def _meanrate_value_anchor(a1, A_t, thresh, a1_floor=None):
        A = float(A_t)
        if abs(A) >= thresh:
            floor = thresh ** 2 if a1_floor is None else float(a1_floor)
            return 0.25 * (torch.log(torch.clamp(a1.abs(), min=floor))
                           - float(np.log(abs(A)))) ** 2
        return 0.25 * ((a1 - A) / thresh) ** 2

    def _meanrate_wp_loss(self, a1, da1_dt, gamma_n, A_t, rate_active):
        """Per-waypoint 'dom-hybrid-meanrate' objective: masked rate residual
        squared + value anchor.
        """
        phi_max = self._dom_phi_max()
        thresh = self.window_crossing_thresh
        scope = self.meanrate_norm_scope
        pnorm = 1.0 if scope == 'none' else self._meanrate_pressure_norm
        whole = scope == 'whole'
        loss = 0.0
        if rate_active:
            loss = loss + self._meanrate_residual(
                da1_dt, a1, float(gamma_n), phi_max, thresh) ** 2 / pnorm
        anchor = self._meanrate_value_anchor(
            a1, float(A_t), thresh, self._a1_structural_floor)
        return loss + (anchor / pnorm if whole else anchor)

    def _compute_waypoint_dTdt(self, T_norm_pred: torch.Tensor) -> torch.Tensor:
        """Compute dT/dt targets at waypoint times using current T predictions."""
        gamma_norm = self._compute_gamma_normalized(T_norm_pred)
        return -gamma_norm * T_norm_pred.detach()

    @staticmethod
    def _closure_ramp(t, Lt, tau_typ):
        a = Lt / tau_typ
        e = torch.exp(-a * t)
        return 1.0 - e, a * e

    @staticmethod
    def _closure_ansatz_terms(tau_c, vs_c_3d, mu_3d, Kn_eff, beta_c, df_c_3d, T_scale,
                              dT_dx_c, dT_dxx_c, dT_dxt_c, m_t, dm_dt):
        """Chapman--Enskog diffusive closure for n_eff and its x,t derivatives."""
        coeff = -(tau_c * vs_c_3d * mu_3d * Kn_eff * beta_c * df_c_3d * T_scale)
        closure = coeff * dT_dx_c * m_t
        dclosure_dx = coeff * dT_dxx_c * m_t
        dclosure_dt = coeff * (dT_dxt_c * m_t + dT_dx_c * dm_dt)
        return closure, dclosure_dx, dclosure_dt

    def _phonon_coupling_source(self, x, t, T_ph_norm):
        """Normalized isotropic (l=0 only, μ-independent) energy-exchange rate
        s(x,t), shape (N, 1), or None.
        """
        return None

    def _coupling_src_bte(self, beta_c, df_c_3d, src_col):
        """Per-mode BTE-residual source term: β_m·(df/dT)_m·T_scale·s — the
        equilibrium-shaped (capacity-partition) projection of the isotropic
        source onto mode m, in bte_res units.
        """
        return beta_c * df_c_3d * (self.delta_T / 2) * src_col

    def _coupling_src_energy(self, w_sum, src_col, c_col=1.0):
        """Exact angular moment of the isotropic source in energy_res units:
        w_sum·c_m·s (∫dΩ = 4π). Same pin test as _coupling_src_bte.
        """
        return w_sum * c_col * src_col

    def compute_loss_and_backward(self) -> Tuple[float, ...]:
        """Compute all losses with per-mode gradient accumulation."""
        if not self.beta_pretrained:
            raise RuntimeError("β-network must be pretrained before BTE training!")

        data = self.data
        Nx, Nt, Ns, Nk, Np = self.Nx, self.Nt, self.Ns, self.Nk, self.Np
        N = self._N  # may differ from Nx*Nt when subsampling
        Kn_eff = self.Kn_eff
        Lt = self.Lt
        T_scale = self.delta_T / 2

        w = data['w'].float()      # (Ns, 1)
        vk_norm = data['vk_norm']  # (Np, Nk)
        tau_norm = data['tau_norm']  # (Np, Nk)
        dfdT = data['dfdT']        # (Np, Nk)

        n_total_modes = Np * Nk
        w_sum = w.sum()
        modes = precompute_modes(vk_norm, tau_norm, data['k_norm'], Nk, Np, Kn_eff)

        # Compute energy weight once
        if self.energy_weight == "auto":
            ew = 1.0 if self.energy_norm == 'moment' else _energy_weight(self.Kn_grating)
        else:
            ew = float(self.energy_weight)

        # Pre-fetch BC and IC data
        tb = data['tb']  # (Nt, 1)
        n_bc = tb.shape[0]
        xi = data['xi']
        n_ic = xi.shape[0]

        # === PERIODIC BC: T periodicity (skip if symmetric — satisfied by construction) ===
        if self.use_symmetric_T or self.use_spectral_t:
            # Spectral T = Σ a_k(t)·cos(2πk·x); cos is periodic with
            # period 1, so T(0,t) = T(1,t) = Σ a_k(t) by construction.
            loss_bc_val = 0.0
        else:
            T_left = self._amp_forward(self.net_T, torch.cat([
                torch.zeros(n_bc, 1, device=self.device), tb
            ], dim=1))
            T_right = self._amp_forward(self.net_T, torch.cat([
                torch.ones(n_bc, 1, device=self.device), tb
            ], dim=1))
            loss_bc_T = torch.mean((T_left - T_right) ** 2)
            loss_bc_T.backward()
            loss_bc_val = loss_bc_T.item()

        if self.use_hard_ic or self.ic_is_structural:
            loss_ic_T_val = 0.0
        else:
            T_ic = self._T_forward(xi, torch.zeros(n_ic, 1, device=self.device))
            T_ic_target = torch.cos(2 * np.pi * xi)
            loss_ic_T = torch.mean((T_ic - T_ic_target) ** 2)
            (60.0 * loss_ic_T).backward()
            loss_ic_T_val = loss_ic_T.item()

        if self.dtic_weight == 0.0 or self.ic_is_structural:
            loss_dt_ic_val = 0.0
        else:
            if self.use_hard_ic:
                T_raw_at_0 = self._T_forward(xi.detach(), torch.zeros(n_ic, 1, device=self.device))
                loss_dt_ic = torch.mean((T_raw_at_0 - self.dTdt_initial) ** 2)
            else:
                t_ic_dt = torch.zeros(n_ic, 1, device=self.device, requires_grad=True)
                T_for_dt = self._T_forward(xi.detach(), t_ic_dt)
                dT_dt_ic = torch.autograd.grad(
                    T_for_dt, t_ic_dt, torch.ones_like(T_for_dt), create_graph=True
                )[0]
                loss_dt_ic = torch.mean((dT_dt_ic - self.dTdt_initial) ** 2)
            (self.dtic_weight * getattr(self, '_wp_anneal_factor', 1.0)
             * loss_dt_ic).backward()
            loss_dt_ic_val = loss_dt_ic.item()

        # === γ-CONSTRAINT or WAYPOINTS (outside mode loop) ===
        if self.ic_is_structural:
            loss_wp_val = 0.0
        elif self.gamma_mode != 'none':
            x_g = data['x'].detach()
            t_g = data['t'].clone().requires_grad_(True)
            T_raw_g = self._T_forward(x_g, t_g)
            if self.use_hard_ic:
                T_g = torch.cos(2 * np.pi * x_g) + t_g * T_raw_g
            else:
                T_g = T_raw_g
            dT_dt_g = torch.autograd.grad(
                T_g, t_g, torch.ones_like(T_g), create_graph=True
            )[0]

            if self.gamma_mode == 'formula':
                gamma_norm = self._compute_gamma_normalized(T_g)
            else:
                T_phys_g = self.T_ref + T_g.detach() * (self.delta_T / 2)
                T_input_g = (T_phys_g - self.T_cold) / self.delta_T
                gamma_norm = F.softplus(self._amp_forward(self.net_gamma, T_input_g))

            target_dTdt = -gamma_norm * T_g.detach()
            loss_gamma_raw = torch.mean((dT_dt_g - target_dTdt) ** 2)
            (self.gamma_weight * loss_gamma_raw).backward()
            loss_wp_val = loss_gamma_raw.item()

            if self.gamma_mode == 'network':
                T_ic_norm = torch.cos(2 * np.pi * xi.detach())
                gamma_analytical = self._compute_gamma_normalized(T_ic_norm)
                T_ic_phys = self.T_ref + T_ic_norm * (self.delta_T / 2)
                T_ic_input = (T_ic_phys - self.T_cold) / self.delta_T
                gamma_pred_ic = F.softplus(self._amp_forward(self.net_gamma, T_ic_input))
                loss_gamma_reg = torch.mean((gamma_pred_ic - gamma_analytical) ** 2)
                loss_gamma_reg.backward()
        else:
            t_max_frac = getattr(self, '_current_t_max_frac', 1.0)
            t_wp_np = self._draw_waypoint_times(t_max_frac)
            t_wp_vals = torch.from_numpy(t_wp_np).float().to(self.device)
            loss_wp_total = 0.0
            if self.waypoint_shape in self.DOM_RATE_SHAPES:
                meanrate = self.waypoint_shape == 'dom-hybrid-meanrate'
                gam_wp = (self._dom_gamma_of_t_signed(t_wp_np) if meanrate
                          else self._dom_gamma_of_t(t_wp_np))  # normalized-time units
                if self.waypoint_shape == 'dom-hybrid' or meanrate:
                    tt_s, A_s = self._dom_shape_curve()
                    A_wp = np.interp(t_wp_np * self.Lt * self.tau_ref, tt_s, A_s)
                else:
                    A_wp = np.zeros_like(t_wp_np)  # unused
                if meanrate:
                    rate_mask = self._wp_rate_mask(t_wp_np)
                x_proj = self._wp_proj_grid()
                cos_proj = torch.cos(2 * np.pi * x_proj)
                for i_wp, (t_val, g_t, A_t) in enumerate(
                        zip(t_wp_vals, gam_wp, A_wp)):
                    t_wp = torch.full_like(x_proj, t_val.item()).requires_grad_(True)
                    T_wp = self._T_forward(x_proj, t_wp)
                    if self.use_hard_ic:
                        T_wp = cos_proj + t_wp * T_wp
                    a1 = 2.0 * torch.mean(T_wp * cos_proj)
                    da1_dt = torch.autograd.grad(a1, t_wp, create_graph=True)[0].sum()
                    if meanrate:
                        loss_wp_total += self._meanrate_wp_loss(
                            a1, da1_dt, float(g_t), float(A_t),
                            bool(rate_mask[i_wp]))
                        continue
                    target = -float(g_t) * a1.detach()
                    loss_wp_total += (da1_dt - target) ** 2 / (abs(float(g_t)) + 1.0)
                    if self.waypoint_shape == 'dom-hybrid':
                        a1c = torch.clamp(a1, min=1e-3)
                        logA = float(np.log(max(float(A_t), 1e-3)))
                        loss_wp_total += 0.25 * (torch.log(a1c) - logA) ** 2
            elif self.waypoint_shape in self.DOM_VALUE_SHAPES:
                tt_s, A_s = self._dom_shape_curve()
                A_wp = np.interp(t_wp_np * self.Lt * self.tau_ref, tt_s, A_s)
                if self.waypoint_shape == 'dom-curve-proj':
                    x_proj = self._wp_proj_grid()
                    cos_proj = torch.cos(2 * np.pi * x_proj)
                    for t_val, A_t in zip(t_wp_vals, A_wp):
                        t_wp = torch.full_like(x_proj, t_val.item())
                        T_wp = self._T_forward(x_proj, t_wp)
                        if self.use_hard_ic:
                            T_wp = cos_proj + t_wp * T_wp
                        a1_wp = 2.0 * torch.mean(T_wp * cos_proj)
                        loss_wp_total += (a1_wp - float(A_t)) ** 2
                else:
                    cos_xi = torch.cos(2 * np.pi * xi.detach())
                    for t_val, A_t in zip(t_wp_vals, A_wp):
                        t_wp = torch.full((n_ic, 1), t_val.item(), device=self.device)
                        T_wp = self._T_forward(xi.detach(), t_wp)
                        if self.use_hard_ic:
                            T_wp = torch.cos(2 * np.pi * xi.detach()) + t_wp * T_wp
                        loss_wp_total += torch.mean((T_wp - float(A_t) * cos_xi) ** 2)
            else:
                if self.waypoint_shape not in self.PLAIN_RATE_SHAPES:
                    raise RuntimeError(
                        f"waypoint_shape {self.waypoint_shape!r} reached the "
                        f"plain-exponential waypoint branch but is not in "
                        f"PLAIN_RATE_SHAPES {self.PLAIN_RATE_SHAPES}; it is "
                        f"missing from DOM_RATE_SHAPES / DOM_VALUE_SHAPES. "
                        f"Classify it in WAYPOINT_SHAPES' derived sets.")
                for t_val in t_wp_vals:
                    t_wp = torch.full((n_ic, 1), t_val.item(), device=self.device, requires_grad=True)
                    T_raw_wp = self._T_forward(xi.detach(), t_wp)
                    if self.use_hard_ic:
                        IC_x_wp = torch.cos(2 * np.pi * xi.detach())
                        T_wp = IC_x_wp + t_wp * T_raw_wp
                    else:
                        T_wp = T_raw_wp
                    dT_dt_wp = torch.autograd.grad(
                        T_wp, t_wp, torch.ones_like(T_wp), create_graph=True
                    )[0]
                    target_wp = self._compute_waypoint_dTdt(T_wp)
                    loss_wp_total += torch.mean((dT_dt_wp - target_wp) ** 2)
            loss_wp_total /= self.n_waypoints
            (self.waypoint_weight * getattr(self, '_wp_anneal_factor', 1.0)
             * loss_wp_total).backward()
            loss_wp_val = loss_wp_total.item()

        # === SINGLE T FORWARD (computed once, reused across all modes) ===
        x_t = data['x'].detach().clone().requires_grad_(True)
        t_t = data['t'].detach().clone().requires_grad_(True)
        T_raw = self._T_forward(x_t, t_t)

        if self.use_hard_ic:
            IC_x = torch.cos(2 * np.pi * x_t)
            T_field = IC_x + t_t * T_raw
        else:
            T_field = T_raw

        dT_dx = torch.autograd.grad(T_field, x_t, torch.ones_like(T_field),
                                     create_graph=True, retain_graph=True)[0]
        dT_dt = torch.autograd.grad(T_field, t_t, torch.ones_like(T_field),
                                     create_graph=True, retain_graph=True)[0]

        coupling_src = self._phonon_coupling_source(x_t, t_t, T_field)
        src_col = (None if coupling_src is None
                   else coupling_src.reshape(1, N, 1))

        # Closure ansatz needs the second T-derivatives (analytic under spectral_t,
        # here via autograd on dT_dx) and a flux-establishment ramp m(t).
        if self.closure_ansatz:
            dT_dxx = torch.autograd.grad(dT_dx, x_t, torch.ones_like(dT_dx),
                                          create_graph=True, retain_graph=True)[0]
            dT_dxt = torch.autograd.grad(dT_dx, t_t, torch.ones_like(dT_dx),
                                          create_graph=True, retain_graph=True)[0]
            tau_typ = self.closure_tau_typ if self.closure_tau_typ is not None else 1.0
            m_t_col, dm_dt_col = self._closure_ramp(t_t, self.Lt, tau_typ)

        M = n_total_modes
        NNs = N * Ns  # points per mode
        mode_chunk = min(M, getattr(self, 'mode_chunk', 6))

        # Pre-compute per-mode vectors (float32)
        k_vec = torch.tensor([md.k_val for md in modes], device=self.device, dtype=torch.float32)
        p_vec = torch.tensor([md.p_val for md in modes], device=self.device, dtype=torch.float32)
        output_scales = torch.tensor([md.output_scale for md in modes], device=self.device, dtype=torch.float32)
        scale_sq_vec = torch.tensor([md.scale_sq for md in modes], device=self.device, dtype=torch.float32)
        v_star_vec = torch.tensor([md.v_star for md in modes], device=self.device, dtype=torch.float32)
        dfdT_vec = torch.tensor([dfdT[md.p, md.ik].item() for md in modes], device=self.device, dtype=torch.float32)

        if self.energy_norm == 'moment':
            g_hat = float(self.window_gamma_hz) * self.tau_ref  # γ̂, per τ_ref
            energy_norm_vec = torch.clamp(
                (4.0 * np.pi * dfdT_vec * T_scale * g_hat) ** 2,
                min=LOSS_NORM_EPSILON)
        else:
            energy_norm_vec = scale_sq_vec

        # Batched beta for all modes — (M*N, 3) → (M, N, 1) [detached, no graph]
        T_rep = T_field.repeat(M, 1)
        k_beta = k_vec.repeat_interleave(N).unsqueeze(1)
        p_beta = p_vec.repeat_interleave(N).unsqueeze(1)
        beta_all = self._amp_forward(self.net_beta, torch.cat([k_beta, p_beta, T_rep], dim=1)).detach()
        beta_all = beta_all.reshape(M, N, 1).expand(-1, -1, Ns)

        # tau(T) for all modes — (M, N, 1) [detached]
        T_det = T_field.detach()
        tau_all = torch.stack([
            self.compute_tau_T(T_det, md.tau_star_ref, md.p, md.ik) for md in modes
        ], dim=0).expand(-1, -1, Ns)

        tb_base = self._tb_exp
        mu_bc_base = self._mu_bc_exp
        n_bc_pts = n_bc * Ns
        bc_t = tb_base.repeat(M, 1)
        bc_mu = mu_bc_base.repeat(M, 1)
        bc_k = k_vec.repeat_interleave(n_bc_pts).unsqueeze(1)
        bc_p = p_vec.repeat_interleave(n_bc_pts).unsqueeze(1)
        bc_scale = output_scales.repeat_interleave(n_bc_pts).unsqueeze(1)

        n_left_all = self._amp_forward(self.net_n,
            torch.cat([torch.zeros(M * n_bc_pts, 1, device=self.device),
                       bc_t, bc_mu, bc_k, bc_p], dim=1)) * bc_scale
        n_right_all = self._amp_forward(self.net_n,
            torch.cat([torch.ones(M * n_bc_pts, 1, device=self.device),
                       bc_t, bc_mu, bc_k, bc_p], dim=1)) * bc_scale

        n_left_3d = n_left_all.reshape(M, n_bc, Ns)
        n_right_3d = n_right_all.reshape(M, n_bc, Ns)
        mean_left = torch.matmul(n_left_3d, w) / (4 * np.pi)
        mean_right = torch.matmul(n_right_3d, w) / (4 * np.pi)
        bc_diff = (n_left_3d - mean_left) - (n_right_3d - mean_right)
        loss_bc_n_per_mode = torch.mean(bc_diff ** 2, dim=(1, 2)) / scale_sq_vec
        loss_bc_n_total_t = loss_bc_n_per_mode.mean()
        loss_bc_n_total_t.backward()
        loss_bc_n_total = loss_bc_n_total_t.item()

        # Batched IC for n (if soft IC)
        loss_ic_n_total = 0.0
        if not self.use_hard_ic:
            n_ic_pts = n_ic * Ns
            ic_x = self._xi_exp.repeat(M, 1)
            ic_t_z = torch.zeros(M * n_ic_pts, 1, device=self.device)
            ic_mu = self._mu_ic_exp.repeat(M, 1)
            ic_k = k_vec.repeat_interleave(n_ic_pts).unsqueeze(1)
            ic_p = p_vec.repeat_interleave(n_ic_pts).unsqueeze(1)
            ic_scale = output_scales.repeat_interleave(n_ic_pts).unsqueeze(1)

            n_ic_all = self._amp_forward(self.net_n,
                torch.cat([ic_x, ic_t_z, ic_mu, ic_k, ic_p], dim=1)) * ic_scale
            n_ic_3d = n_ic_all.reshape(M, n_ic, Ns)
            mean_ic = torch.matmul(n_ic_3d, w) / (4 * np.pi)
            n_ic_proj = n_ic_3d - mean_ic
            loss_ic_n_per_mode = torch.mean(n_ic_proj ** 2, dim=(1, 2)) / scale_sq_vec
            loss_ic_n_total_t = loss_ic_n_per_mode.mean()
            (10.0 * loss_ic_n_total_t).backward()
            loss_ic_n_total = loss_ic_n_total_t.item()

        loss_bte_total = 0.0
        loss_energy_total = 0.0
        # DIAGNOSTIC (opt-in via self._diag_every>0; default off = zero overhead):
        # measure isolated per-term gradient norms on net_n every _diag_every epochs.
        _diag_every = getattr(self, '_diag_every', 0)
        _measuring = _diag_every > 0 and (getattr(self, '_diag_epoch', 0) % _diag_every == 0)
        if _measuring:
            _gn_params = [p for p in self.net_n.parameters() if p.requires_grad]
            _gn_bte_acc = [torch.zeros_like(p) for p in _gn_params]
            _gn_en_acc = [torch.zeros_like(p) for p in _gn_params]
        x_base = self._x_exp
        t_base = self._t_exp
        mu_base = self._mu_exp

        n_chunks = (M + mode_chunk - 1) // mode_chunk
        for chunk_idx in range(n_chunks):
            c_start = chunk_idx * mode_chunk
            c_end = min(c_start + mode_chunk, M)
            C = c_end - c_start  # chunk size

            # Slice per-mode vectors for this chunk
            k_c = k_vec[c_start:c_end]
            p_c = p_vec[c_start:c_end]
            os_c = output_scales[c_start:c_end]
            ss_c = scale_sq_vec[c_start:c_end]
            en_norm_c = energy_norm_vec[c_start:c_end]
            vs_c = v_star_vec[c_start:c_end]
            df_c = dfdT_vec[c_start:c_end]
            beta_c = beta_all[c_start:c_end]  # (C, N, Ns)
            tau_c = tau_all[c_start:c_end]  # (C, N, Ns)

            # Batched net_n: (C*NNs, 5)
            x_n = x_base.repeat(C, 1).detach().clone().requires_grad_(True)
            t_n = t_base.repeat(C, 1).detach().clone().requires_grad_(True)
            mu_n = mu_base.repeat(C, 1)
            k_n = k_c.repeat_interleave(NNs).unsqueeze(1)
            p_n = p_c.repeat_interleave(NNs).unsqueeze(1)
            scale_n = os_c.repeat_interleave(NNs).unsqueeze(1)

            n_out = self._amp_forward(self.net_n, torch.cat([x_n, t_n, mu_n, k_n, p_n], dim=1)) * scale_n
            dn_dx = torch.autograd.grad(n_out, x_n, torch.ones_like(n_out),
                                         create_graph=True, retain_graph=True)[0]
            dn_dt = torch.autograd.grad(n_out, t_n, torch.ones_like(n_out),
                                         create_graph=True)[0]

            # Reshape to (C, N, Ns)
            n_3d = n_out.reshape(C, N, Ns)
            dn_dx_3d = dn_dx.reshape(C, N, Ns)
            dn_dt_3d = dn_dt.reshape(C, N, Ns)

            # Zero-mean projection
            mean_n = torch.matmul(n_3d, w) / (4 * np.pi)
            mean_dn_dx = torch.matmul(dn_dx_3d, w) / (4 * np.pi)
            mean_dn_dt = torch.matmul(dn_dt_3d, w) / (4 * np.pi)
            n_proj = n_3d - mean_n
            dn_dx_proj = dn_dx_3d - mean_dn_dx
            dn_dt_proj = dn_dt_3d - mean_dn_dt

            # Hard IC
            if self.use_hard_ic:
                t_3d = t_t.unsqueeze(0).expand(C, -1, Ns)
                n_eff = t_3d * n_proj
                dn_eff_dx = t_3d * dn_dx_proj
                dn_eff_dt = n_proj + t_3d * dn_dt_proj
            else:
                n_eff = n_proj
                dn_eff_dx = dn_dx_proj
                dn_eff_dt = dn_dt_proj

            # BTE residual — (C, N, Ns)
            dT_dx_c = dT_dx.unsqueeze(0).expand(C, -1, Ns)
            dT_dt_c = dT_dt.unsqueeze(0).expand(C, -1, Ns)
            df_c_3d = df_c.reshape(C, 1, 1)
            vs_c_3d = vs_c.reshape(C, 1, 1)
            mu_3d = self._mu_2d.unsqueeze(0).expand(C, -1, -1)

            if self.closure_ansatz:
                dT_dxx_c = dT_dxx.unsqueeze(0).expand(C, -1, Ns)
                dT_dxt_c = dT_dxt.unsqueeze(0).expand(C, -1, Ns)
                m_t = m_t_col.reshape(1, N, 1)
                dm_dt = dm_dt_col.reshape(1, N, 1)
                closure, dclosure_dx, dclosure_dt = self._closure_ansatz_terms(
                    tau_c, vs_c_3d, mu_3d, Kn_eff, beta_c, df_c_3d, T_scale,
                    dT_dx_c, dT_dxx_c, dT_dxt_c, m_t, dm_dt)
                n_eff = n_eff + closure
                dn_eff_dx = dn_eff_dx + dclosure_dx
                dn_eff_dt = dn_eff_dt + dclosure_dt

            df_dx_c = dn_eff_dx + beta_c * df_c_3d * T_scale * dT_dx_c
            df_dt_c = dn_eff_dt + beta_c * df_c_3d * T_scale * dT_dt_c

            bte_res = df_dt_c / Lt + vs_c_3d * mu_3d * df_dx_c * Kn_eff + n_eff / tau_c
            if src_col is not None:
                bte_res = bte_res - self._coupling_src_bte(beta_c, df_c_3d,
                                                           src_col)
            loss_bte_c = torch.mean(bte_res ** 2, dim=(1, 2)) / ss_c

            # Energy conservation — (C, N, 1)
            dq_x = torch.matmul(mu_3d * dn_eff_dx, w)
            c_col = (beta_c[:, :, :1] * df_c_3d * T_scale
                     if self.energy_moment_c else 1.0)
            energy_res = (dT_dt.unsqueeze(0) * c_col * w_sum / Lt
                          + vs_c_3d.squeeze(-1).unsqueeze(-1) * Kn_eff * dq_x)
            if src_col is not None:
                energy_res = energy_res - (
                    self._coupling_src_energy(w_sum, src_col, c_col)
                    if self.energy_moment_c
                    else self._coupling_src_energy(w_sum, src_col))
            loss_energy_c = torch.mean(energy_res ** 2, dim=(1, 2)) / en_norm_c

            _mw = getattr(self, 'moment_weight', 0.0)
            loss_moment_c = (torch.mean(torch.matmul(mu_3d ** 2 * bte_res, w) ** 2,
                                        dim=(1, 2)) / ss_c) if _mw > 0 else None

            if _measuring:
                _gb = torch.autograd.grad(loss_bte_c.mean() / n_chunks, _gn_params,
                                          retain_graph=True, allow_unused=True)
                _ge = torch.autograd.grad(ew * loss_energy_c.mean() / n_chunks, _gn_params,
                                          retain_graph=True, allow_unused=True)
                for _i in range(len(_gn_params)):
                    if _gb[_i] is not None:
                        _gn_bte_acc[_i] += _gb[_i].detach()
                    if _ge[_i] is not None:
                        _gn_en_acc[_i] += _ge[_i].detach()

            # Backward for this chunk (retain T graph for remaining chunks)
            is_last_chunk = (chunk_idx == n_chunks - 1)
            _terms = loss_bte_c.mean() + ew * loss_energy_c.mean()
            if loss_moment_c is not None:
                _terms = _terms + _mw * loss_moment_c.mean()
            chunk_loss = _terms / n_chunks
            chunk_loss.backward(retain_graph=not is_last_chunk)

            loss_bte_total += loss_bte_c.mean().item() / n_chunks
            loss_energy_total += loss_energy_c.mean().item() / n_chunks

        if _measuring:
            _gn_bte = float(torch.sqrt(sum((g ** 2).sum() for g in _gn_bte_acc)))
            _gn_en = float(torch.sqrt(sum((g ** 2).sum() for g in _gn_en_acc)))
            _dot = float(sum((gb * ge).sum() for gb, ge in zip(_gn_bte_acc, _gn_en_acc)))
            _cos = _dot / (_gn_bte * _gn_en + 1e-30)
            if not hasattr(self, '_diag_history'):
                self._diag_history = []
            self._diag_history.append((getattr(self, '_diag_epoch', 0), _gn_bte, _gn_en, _cos))
        if _diag_every > 0:
            self._diag_epoch = getattr(self, '_diag_epoch', 0) + 1

        loss_bc_val += loss_bc_n_total

        # === GLOBAL ENERGY CONSERVATION ===
        T_ge_raw = self._T_forward(data['x'], data['t'])
        if self.use_hard_ic:
            # T_hard = cos(2πx) + t·net_T; mean(cos) = 0, so mean(T_hard) = mean(t·net_T)
            T_ge = torch.cos(2 * np.pi * data['x']) + data['t'] * T_ge_raw
        else:
            T_ge = T_ge_raw
        loss_ge = torch.mean(T_ge) ** 2
        loss_ge.backward()
        loss_ge_val = loss_ge.item()

        loss_amp_val = 0.0
        if self.amp_loss_weight > 0 and not self.use_spectral_t:
            T_amp_raw = self._T_forward(self._amp_x, self._amp_t)
            if self.use_hard_ic:
                T_amp = torch.cos(2 * np.pi * self._amp_x) + self._amp_t * T_amp_raw
            else:
                T_amp = T_amp_raw
            T_amp_grid = T_amp.reshape(self.amp_grid_Nt, self.amp_grid_Nx)
            # Cosine projection (signed; matches test-time `compute_amplitude_1d`)
            A_t = 2.0 * (T_amp_grid * self._amp_cos_x.unsqueeze(0)).mean(dim=1)
            # Barrier above 0: penalise A < amp_floor so we leave headroom
            # for between-sample drift instead of merely A>=0 strictly at samples
            amp_floor = self.amp_loss_floor
            loss_amp_pos = torch.mean(torch.relu(amp_floor - A_t) ** 2)
            dA_dt = (A_t[1:] - A_t[:-1]) / self._amp_dt_norm
            loss_amp_mono = torch.mean(torch.relu(dA_dt) ** 2)
            loss_amp = self.amp_loss_weight * (loss_amp_pos + loss_amp_mono)
            loss_amp.backward()
            loss_amp_val = loss_amp.item()

        out = (
            loss_bte_total,
            loss_energy_total,
            loss_bc_val,
            loss_ic_n_total,
            loss_ic_T_val,
            loss_dt_ic_val,
            loss_wp_val,
            loss_ge_val,
            loss_amp_val,
        )
        if not self.max_principle_guard:
            return out

        T_mp = self._T_forward(self._amp_x, self._amp_t)
        if self.use_hard_ic:
            # same composition as the amplitude constraint above: under the
            # hard-IC ansatz `_T_forward` returns the RAW head, not the field.
            T_mp = torch.cos(2 * np.pi * self._amp_x) + self._amp_t * T_mp
        loss_mp_raw = self._max_principle_loss(T_mp)
        loss_mp = self._max_principle_weight * loss_mp_raw
        # A feasible field gives an exactly-zero term with an exactly-zero
        # gradient; skipping the backward keeps that literal (and cheap).
        if loss_mp_raw.detach().item() > 0.0:
            loss_mp.backward()
        # collapse-guard diagnostics: max|T| per epoch, in Kelvin
        self._max_abs_T_last = float(T_mp.detach().abs().max()) * self.delta_T / 2
        return out + (loss_mp.item(),)

    def train(self) -> List[List[float]]:
        """Run full training: β pretraining + BTE training."""
        log_file = setup_file_logging(self.output_dir, self.run_id)
        logger.info(f"Logging to: {log_file}")

        with self.wandb_run():
            # Phase 1: Pretrain β
            beta_loss = self.pretrain_beta()

            # Phase 2: Train BTE
            return self._training_loop()

    def _update_time_domain(self, t_max_frac: float):
        """Regenerate collocation grid for new time domain [0, t_max_frac]."""
        Nx, Nt, Ns = self.Nx, self.Nt, self.Ns

        tm = np.linspace(0, t_max_frac, Nt)
        xm = np.linspace(0, 1, Nx)
        x_grid, t_grid = np.meshgrid(xm, tm)
        x = x_grid.reshape(-1, 1)
        t = t_grid.reshape(-1, 1)
        N = Nx * Nt

        self.data['x'] = torch.FloatTensor(x).to(self.device)
        self.data['t'] = torch.FloatTensor(t).to(self.device)

        # Regenerate expanded tensors
        self._N = N
        self._N_full = N
        self._x_full = self.data['x'].clone()
        self._t_full = self.data['t'].clone()
        self._x_exp = self.data['x'].repeat(1, Ns).reshape(-1, 1)
        self._t_exp = self.data['t'].repeat(1, Ns).reshape(-1, 1)
        self._mu_exp = self.data['mu'].squeeze().repeat(N).reshape(-1, 1)
        self._mu_2d = self.data['mu'].squeeze().unsqueeze(0).expand(N, -1)

        # Update boundary time samples
        tb = np.linspace(0, t_max_frac, Nt + 2)[1:Nt + 1].reshape(-1, 1)
        self.data['tb'] = torch.FloatTensor(tb).to(self.device)
        n_bc = tb.shape[0]
        self._tb_exp = self.data['tb'].repeat(1, Ns).reshape(-1, 1)
        self._mu_bc_exp = self.data['mu'].squeeze().repeat(n_bc).reshape(-1, 1)

        self._current_t_max_frac = t_max_frac
        logger.info(f"Curriculum: updated time domain to [0, {t_max_frac:.2f}]")

    def _reset_optimizers(self):
        """Reset optimizer state (Adam moments) between curriculum stages."""
        self.optimizer_T = optim.Adam(
            self.net_T.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )
        self.optimizer_n = optim.Adam(
            self.net_n.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            eps=1e-10,
        )
        if self.gamma_mode == 'network':
            self.optimizer_gamma = optim.Adam(
                self.net_gamma.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.99),
                eps=1e-10,
            )

    def _training_loop(self) -> List[List[float]]:
        """BTE training loop with per-mode gradient accumulation."""
        log_training_start(
            mode="Transient_Nongray_LargeDT",
            num_epochs=self.epochs,
            device=str(self.device),
            run_id=self.run_id,
        )

        console.print(f"\n[bold cyan]Phase 2: BTE Training[/bold cyan]")
        console.print(f"  L = {self.L * ANGSTROM_TO_UM:.1f} μm, ΔT = {self.delta_T}K")
        console.print(f"  T range: [{self.T_cold:.0f}K, {self.T_hot:.0f}K]")
        console.print(f"  Kn_eff = {self.Kn_eff:.4f}")
        console.print(f"  τ(T) model = {self.tau_model}"
                      + (f" (α = {self.tau_exponent})" if self.tau_model == "power"
                         else " (full branch-resolved Holland--Callaway)"))
        if self.use_hard_ic:
            console.print(f"  Hard IC: T = cos(2πx) + t·net_T, n = t·net_n")
        if self.gamma_mode != 'none':
            console.print(f"  γ-constraint: mode={self.gamma_mode}, weight={self.gamma_weight}")
        else:
            console.print(f"  Waypoints: {self.n_waypoints} ({self.waypoint_spacing}), weight={self.waypoint_weight}")
        if self.use_curriculum:
            console.print(f"  Curriculum: {self.curriculum_stages}")
        console.print(f"  ALL {self.Np*self.Nk} modes × ALL {self.Ns} angles (vectorized)")
        console.print(f"  Epochs: {self.epochs}")

        self.net_T.train()
        self.net_n.train()

        loss_history = []
        min_loss = float('inf')
        start_time = time.time()

        # LR scheduling
        scheduler_T = CosineAnnealingLR(self.optimizer_T, T_max=self.epochs, eta_min=1e-6)
        scheduler_n = CosineAnnealingLR(self.optimizer_n, T_max=self.epochs, eta_min=1e-6)

        # Build epoch schedule: [(start_epoch, end_epoch, t_max_frac), ...]
        if self.use_curriculum:
            stages = []
            epoch_offset = 0
            for i, (t_frac, epoch_frac) in enumerate(self.curriculum_stages):
                n_epochs = int(self.epochs * epoch_frac)
                if i == len(self.curriculum_stages) - 1:
                    n_epochs = self.epochs - epoch_offset  # Last stage gets remainder
                stages.append((epoch_offset, epoch_offset + n_epochs, t_frac))
                epoch_offset += n_epochs
        else:
            stages = [(0, self.epochs, 1.0)]

        global_epoch = 0
        for stage_idx, (stage_start, stage_end, t_frac) in enumerate(stages):
            if self.use_curriculum:
                console.print(
                    f"\n  [bold yellow]Curriculum Stage {stage_idx + 1}/{len(stages)}: "
                    f"t ∈ [0, {t_frac:.1f}], epochs {stage_start}-{stage_end}[/bold yellow]"
                )
                self._update_time_domain(t_frac)
                if stage_idx > 0:
                    self._reset_optimizers()

            for epoch in range(stage_start, stage_end):
                self.optimizer_T.zero_grad()
                self.optimizer_n.zero_grad()
                if self.gamma_mode == 'network':
                    self.optimizer_gamma.zero_grad()

                self._subsample_grid()
                if self.waypoint_anneal_from > 0.0:
                    e0 = self.waypoint_anneal_from * self.epochs
                    if epoch <= e0:
                        self._wp_anneal_factor = 1.0
                    else:
                        frac = (epoch - e0) / max(self.epochs - e0, 1.0)
                        self._wp_anneal_factor = 0.5 * (1.0 + np.cos(np.pi * min(frac, 1.0)))
                loss_values = self.compute_loss_and_backward()
                torch.nn.utils.clip_grad_norm_(self.net_T.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.net_n.parameters(), max_norm=1.0)
                if self.gamma_mode == 'network':
                    torch.nn.utils.clip_grad_norm_(self.net_gamma.parameters(), max_norm=1.0)
                self.optimizer_T.step()
                self.optimizer_n.step()
                scheduler_T.step()
                scheduler_n.step()
                if self.gamma_mode == 'network':
                    self.optimizer_gamma.step()

                clamp_frac = self._check_clamp_occupancy(epoch)

                total_loss = sum(loss_values)
                loss_vals = list(loss_values) + [total_loss]
                loss_history.append(loss_vals)

                max_abs_T = getattr(self, '_max_abs_T_last', None)
                self._check_max_principle_trigger(epoch)

                if hasattr(self, '_wandb_run_active') and self._wandb_run_active:
                    if max_abs_T is not None:
                        self.log_metric('max_abs_T_K', max_abs_T, step=epoch)
                    self.log_metrics({
                        'loss_bte': loss_vals[0],
                        'loss_energy': loss_vals[1],
                        'loss_bc': loss_vals[2],
                        'loss_ic_n': loss_vals[3],
                        'loss_ic_T': loss_vals[4],
                        'loss_dt_ic': loss_vals[5],
                        'loss_waypoint': loss_vals[6],
                        'loss_global_energy': loss_vals[7],
                        'loss_amp_phys': loss_vals[8],
                        'loss_total': loss_vals[-1],
                    }, step=epoch)
                    if self.net_arch == "pirate" and epoch % 500 == 0:
                        for name, net in [("T", self.net_T), ("n", self.net_n)]:
                            if hasattr(net, "alphas"):
                                for i, a in enumerate(net.alphas):
                                    self.log_metric(f"gate/{name}/layer_{i}", torch.sigmoid(a).item(), step=epoch)

                if epoch % PRINT_FREQUENCY == 0:
                    elapsed = time.time() - start_time
                    stage_tag = f"S{stage_idx + 1} " if self.use_curriculum else ""
                    console.print(
                        f"{elapsed:8.1f}s | {stage_tag}Epoch {epoch:5d} | "
                        f"BTE={loss_vals[0]:.2e} | Energy={loss_vals[1]:.2e} | "
                        f"BC={loss_vals[2]:.2e} | IC_n={loss_vals[3]:.2e} | "
                        f"IC_T={loss_vals[4]:.2e} | dT={loss_vals[5]:.2e} | "
                        f"WP={loss_vals[6]:.2e} | GE={loss_vals[7]:.2e} | "
                        f"Amp={loss_vals[8]:.2e} | Total={loss_vals[-1]:.2e}"
                        + (f" | clamp={clamp_frac:.2f}"
                           if self._a1_exponent_cap is not None else "")
                        + (f" | MP={loss_vals[9]:.2e} maxT={max_abs_T:.4f}K"
                           if max_abs_T is not None else "")
                    )

                if total_loss < min_loss:
                    min_loss = total_loss
                    self._save_models()

        # Save final models and loss history
        elapsed_time = time.time() - start_time
        self._save_models()

        loss_header = ("BTE Energy_Cons Periodic_BC IC_n IC_T dT_IC Waypoint "
                       "Global_Energy Total")
        if self.max_principle_guard:
            loss_header = ("BTE Energy_Cons Periodic_BC IC_n IC_T dT_IC "
                           "Waypoint Global_Energy Amp_Phys Max_Principle Total")
        np.savetxt(
            self.output_dir / f"loss_{self.run_id}.txt",
            np.array(loss_history),
            fmt='%.6e',
            header=loss_header,
        )

        model_files = [
            f"net_T_{self.run_id}.pt",
            f"net_n_{self.run_id}.pt",
            f"net_beta_{self.run_id}.pt",
        ]
        if self.gamma_mode == 'network':
            model_files.append(f"net_gamma_{self.run_id}.pt")
        log_training_complete(
            elapsed_time=elapsed_time,
            output_path=str(self.output_dir),
            model_files=model_files,
        )

        return loss_history

    def _save_models(self):
        """Save model checkpoints."""
        torch.save(self.net_T.state_dict(), self.output_dir / f"net_T_{self.run_id}.pt")
        torch.save(self.net_n.state_dict(), self.output_dir / f"net_n_{self.run_id}.pt")
        if self.gamma_mode == 'network':
            torch.save(self.net_gamma.state_dict(), self.output_dir / f"net_gamma_{self.run_id}.pt")

    def test(self, Nx_test: int = 81, Nt_test: int = 81) -> Dict[str, np.ndarray]:
        """Generate test results."""
        self.net_T.eval()

        if self.max_principle_guard and (Nx_test > self.amp_grid_Nx
                                         or Nt_test > self.amp_grid_Nt):
            logger.warning(
                "test grid %d×%d is finer than the maximum-principle guard's "
                "%d×%d amplitude grid: |T| is only constrained at the guard's "
                "nodes, so the saved field may exceed A0 between them.",
                Nt_test, Nx_test, self.amp_grid_Nt, self.amp_grid_Nx)

        x_test = torch.linspace(0, 1, Nx_test).reshape(-1, 1).to(self.device)
        t_test = torch.linspace(0, 1, Nt_test).reshape(-1, 1).to(self.device)

        T_field = np.zeros((Nt_test, Nx_test))

        with torch.no_grad():
            for it, t_val in enumerate(t_test):
                t_expanded = t_val.expand(Nx_test, 1)
                if self.use_spectral_t:
                    # Use _T_forward — already returns final T_norm
                    T_norm = self._T_forward(x_test, t_expanded)
                else:
                    T_raw = self.net_T(torch.cat([x_test, t_expanded], dim=1))
                    if self.use_hard_ic:
                        IC_x = torch.cos(2 * np.pi * x_test)
                        T_norm = IC_x + t_expanded * T_raw
                    else:
                        T_norm = T_raw
                T_field[it, :] = (T_norm.cpu().numpy().flatten() * self.delta_T / 2)

        # Compute effective energy weight for metadata
        if self.energy_weight == "auto":
            # must mirror compute_loss_and_backward exactly, or the sidecar
            # would report a weight the run never used
            if self.energy_norm == 'moment':
                ew_effective, ew_label = 1.0, "auto(moment→1)"
            else:
                ew_effective, ew_label = _energy_weight(self.Kn_grating), "auto"
        else:
            ew_effective = float(self.energy_weight)
            ew_label = str(self.energy_weight)

        x_np = x_test.cpu().numpy().flatten()
        n_keep = Nx_test - 1 if np.isclose(x_np[-1] - x_np[0], 1.0) else Nx_test
        cos_x = np.cos(2 * np.pi * x_np[:n_keep])
        amp_t = 2.0 * np.mean(T_field[:, :n_keep] * cos_x[None, :], axis=1)
        A0_proj, A0_expect = float(amp_t[0]), self.delta_T / 2
        max_amp_after_t0 = float(np.max(np.abs(amp_t[1:]))) if len(amp_t) > 1 else 0.0
        amp_floor = self.window_crossing_thresh * abs(A0_expect)
        ic_is_structural_cosine = (not self.ic_is_structural
                                   and (self.use_spectral_t or self.use_hard_ic))

        results = {
            'x': x_test.cpu().numpy().flatten(),
            't': t_test.cpu().numpy().flatten(),
            'T': T_field,
            'A0_projected': A0_proj,
            'max_amp_after_t0': max_amp_after_t0,
            'T_ref': self.T_ref,
            'delta_T': self.delta_T,
            'Lt': self.Lt,
            'Lt_seconds': self.Lt * self.tau_ref,
            'gamma_target_mode': self.gamma_target_mode,
            'window_crossing_thresh': self.window_crossing_thresh,
            'n_decay_times': self.n_decay_times,
            'Lt_rule': self.Lt_rule,
            'Lt_candidate_crossing_tau': self.Lt_candidate_crossing_tau,
            'Lt_candidate_ndecay_tau': self.Lt_candidate_ndecay_tau,
            'window_gamma_hz': self.window_gamma_hz,
            'waypoint_shape': self.waypoint_shape,
            'n_waypoints': self.n_waypoints,
            'waypoint_weight': self.waypoint_weight,
            'waypoint_spacing': self.waypoint_spacing,
            'meanrate_norm_scope': self.meanrate_norm_scope,
            'signed_a1': self.signed_a1,
            'arbiter_nmu_floor': self.arbiter_nmu_floor,
            'arbiter_window_rule': self.arbiter_window_rule,
            'max_principle_guard': self.max_principle_guard,
            'max_principle_weight': self._max_principle_weight,
            'max_abs_T_K': float(np.abs(T_field).max()),
            # The guard's own grid: it bounds |T| at THESE nodes, so the
            # falsifier is only as strong as this resolution.
            'amp_grid_Nx': self.amp_grid_Nx,
            'amp_grid_Nt': self.amp_grid_Nt,
            'amp_grid_dt_tau': self._max_principle_dt_tau,
            'amp_grid_dt_frac': 1.0 / max(self.amp_grid_Nt - 1, 1),
            'max_principle_trigger_K': (self.MAX_PRINCIPLE_TRIGGER_FRAC
                                        * self.delta_T / 2),
            'max_principle_trigger_fired': getattr(self, '_mp_trigger_fired',
                                                   False),
            'max_principle_trigger_max_streak': getattr(
                self, '_mp_trigger_max_streak', 0),
            'max_principle_bar_K': self.MAX_PRINCIPLE_BAR_FRAC * self.delta_T / 2,
            **self._max_principle_sup(),
            'Nk': self.Nk,
            'tau_ref': self.tau_ref,
            'Kn_eff': self.Kn_eff,
            'v_ref': self.v_ref,
            'L_um': self.L * ANGSTROM_TO_UM,
            'energy_weight_effective': ew_effective,
            'energy_weight_label': ew_label,
            'energy_moment_c': self.energy_moment_c,
            'energy_norm': self.energy_norm,
            'net_arch': getattr(self, 'net_arch', 'mlp'),
        }

        np.savez(
            self.output_dir / f"{self.run_id}_results.npz",
            **results
        )
        console.print(f"Results saved to: {self.output_dir / f'{self.run_id}_results.npz'}")

        # Raise AFTER writing, so a degenerate run is still diagnosable.
        if max_amp_after_t0 < amp_floor:
            raise RuntimeError(
                f"DEGENERATE FIELD: max|A(t>0)| = {max_amp_after_t0:.4g} K is "
                f"below the window criterion's own floor "
                f"{self.window_crossing_thresh:.3g}·A0 = {amp_floor:.4g} K — "
                f"the run is the zero solution (results still written to "
                f"{self.output_dir}). Ruling §12.2: no pin may silently be a "
                "dead field.")
        if ic_is_structural_cosine and abs(A0_proj / A0_expect - 1.0) > 0.01:
            raise RuntimeError(
                f"DEGENERATE IC: projected A0 = {A0_proj:.4g} K deviates "
                f"{abs(A0_proj / A0_expect - 1.0):.2%} from the cosine-IC "
                f"amplitude {A0_expect:.4g} K, which this head enforces BY "
                f"CONSTRUCTION (results still written to {self.output_dir}).")

        return results
