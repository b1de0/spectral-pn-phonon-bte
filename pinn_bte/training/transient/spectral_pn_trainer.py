"""Training harness for the spectral-PN transient-thermal-grating solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pinn_bte.models.spectral_pn import (
    PNCoefficientNet,
    pn_amplitude,
    pn_residual,
)
from pinn_bte.physics.ttg_dom import WINDOW_RULE_FULL
from pinn_bte.physics.decay_metrics import (
    DENOMINATOR_RULE_ARBITER_WINDOW,
    DENOMINATOR_RULE_RULED,
    DENOMINATOR_RULE_SAME_WINDOW,
    DENOMINATOR_RULES,
    GAMMA_ESTIMATOR_RULED,
    amplitude_sign_report,
    denominator_provenance,
    integral_gamma,
)

ANGSTROM_PER_UM = 1e4

BLENDS = ("banded", "ugks")

SETTLE_WINDOW_DECLARED = 5000
SETTLE_TOL_DECLARED = 0.005


def gamma_settled(epochs, gammas, window, tol) -> bool:
    """Has the reported rate stopped moving?"""
    ep = np.asarray(epochs, dtype=float)
    g = np.asarray(gammas, dtype=float)
    if ep.size == 0 or ep[-1] < window:
        return False                      # window not yet full
    sel = ep >= ep[-1] - window
    if sel.sum() < 3:
        return False                      # too sparsely sampled to judge
    seg = g[sel]
    return bool(seg.max() - seg.min() < tol)


def _settle_plan(cfg) -> tuple:
    """Resolve (epochs_cap, settle_on) and refuse the silent-failure configs."""
    settle_on = cfg.settle_window > 0
    cap = cfg.epochs_cap or cfg.epochs
    if settle_on and cfg.gamma_log_every <= 0:
        raise ValueError(
            "settle_window > 0 requires gamma_log_every > 0: the stopping rule "
            "reads gamma_history, and without the logger it can never fire, so "
            "every arm would run to epochs_cap and the truncation would be "
            "indistinguishable from a converged result.")
    if cap < cfg.epochs:
        raise ValueError(
            f"epochs_cap={cap} is below epochs={cfg.epochs}; `epochs` is the "
            "MINIMUM budget when the stopping rule is on, so a smaller cap "
            "would cut the run below its own declared floor.")
    return cap, settle_on


@dataclass
class SpectralPNConfig:
    """One spectral-PN training run. Times/rates share the tau unit system of the
    mode set (seconds for the Si set, tau_gray for gray tests).
    """
    v: np.ndarray
    tau: np.ndarray
    C: np.ndarray
    q: float
    t_end: float
    L_max: int = 8
    width: int = 64
    depth: int = 4
    emb_dim: int = 8
    epochs: int = 20000
    lr: float = 2e-3
    lr_cosine: bool = True
    lr_decay_epochs: Optional[int] = None
    settle_window: int = 0
    settle_tol: float = 0.0
    epochs_cap: int = 0
    nt: int = 1200
    t_grid: str = "log-mix"          # "uniform" | "log-mix"
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"           # "float64" | "float32"
    t_ref: Optional[np.ndarray] = None
    A_ref: Optional[np.ndarray] = None
    gamma_ref: Optional[float] = None  # 1/t-unit; the ARBITER-WINDOW rate
    gamma_ref_rule: str = DENOMINATOR_RULE_RULED
    label: str = "spectral-pn"
    outdir: Optional[Path] = None
    run_id: Optional[str] = None
    log_every: int = 500
    gamma_log_every: int = 0
    component_log_every: int = 0
    snapshot_selection: str = "final"
    extra_meta: Optional[dict] = None   # unit metadata for the npz
                                        # (tau_ref/Nk/L_um for Si cases)
    ap_scaling: bool = False            # asymptotic-preserving l>=1 row
                                        # weighting (deep-diffusive fix,
                                        # see pn_residual)
    slow_head: bool = False             # slow-manifold factorized A_hat
    duhamel: bool = False               # exact collisionless part in the
                                        # ansatz + truncation-edge residual
                                        # correction (deep-ballistic fix)
    formulation: str = "plain"          # "plain" | "ce" (Chapman-Enskog
    blend: str = "banded"
    macro_row: str = "on"
    residual_weight_floor: Optional[float] = None
    gauge: str = "additive"
    closure_pin: bool = False
    xi_split: float = 1.0               # xihybrid only: modes with
                                        # xi_m = q v tau below it get the
                                        # CE structure
    xi_duh: float = 10.0                # xihybrid only: modes at/above it
                                        # get the Duhamel base; the middle
                                        # band [xi_split, xi_duh) is plain
    shared_ahat: bool = False
    carrier_energy: bool = False
    carrier_row: bool = False           # + carrier-consistency row: the
    Nk: int = 20
    T_ref: float = 300.0


WINDOW_MODES = ('n_decay', 'crossing', 'crossing-union')
#: Modes that run the 1%-crossing search.
CROSSING_WINDOW_MODES = tuple(m for m in WINDOW_MODES if 'crossing' in m)
#: Modes that UNION the crossing window with the n_decay/gamma one.
UNION_WINDOW_MODES = tuple(m for m in CROSSING_WINDOW_MODES
                           if m.endswith('-union'))
WINDOW_RULES = ('n_decay', 'crossing')

WINDOW_CROSSING_THRESH = 0.01


@dataclass
class WindowChoice:
    """All times SECONDS."""
    t_end: float
    rule: str
    t_candidate_ndecay_s: float
    t_candidate_crossing_s: float
    gamma_dom_hz: float
    mode: str
    crossing_thresh: float


def select_window(L_um: float, Nk: int = 20, T_ref: float = 300.0,
                  n_decay_times: float = 7.0, geometry: str = "1d",
                  window_mode: str = "n_decay",
                  window_crossing_thresh: float = WINDOW_CROSSING_THRESH,
                  window_rule: str | None = None,
                  ) -> WindowChoice:
    """Pick the PN evaluation window. See `build_si_case` for the rules."""
    from pinn_bte.physics.ttg_dom import ttg_decay_spectral, ttg_window_crossing

    rule_kw = {} if window_rule is None else {"window_rule": window_rule}

    if window_mode not in WINDOW_MODES:
        raise ValueError(f"window_mode={window_mode!r} (expected one of "
                         f"{WINDOW_MODES})")

    gamma_dom_hz = ttg_decay_spectral(L_um, geometry, Nk=Nk, T_ref=T_ref,
                                      **rule_kw)
    if not np.isfinite(gamma_dom_hz) or gamma_dom_hz <= 0.0:
        raise ValueError(
            f"select_window(L={L_um}um, {geometry}, Nk={Nk}): the arbiter "
            f"returned a non-finite or non-positive gamma_dom "
            f"({gamma_dom_hz!r}); every window rule below is a function of it.")
    # Candidate 1: the exponential n_decay/gamma window (the shipped rule).
    t_ndecay = n_decay_times / gamma_dom_hz
    t_crossing = float('nan')
    t_end, rule = t_ndecay, 'n_decay'
    if window_mode in CROSSING_WINDOW_MODES:
        t_crossing = ttg_window_crossing(L_um, geometry, Nk=Nk,
                                         thresh=window_crossing_thresh,
                                         T_ref=T_ref, **rule_kw)
        if not np.isfinite(t_crossing):
            raise ValueError(
                f"select_window(L={L_um}um, {geometry}, Nk={Nk}, "
                f"mode={window_mode!r}): the 1%-crossing search returned a "
                f"non-finite window ({t_crossing!r}). Refusing to select a "
                f"window that no comparison can reject.")
        t_end, rule = t_crossing, 'crossing'
        if window_mode in UNION_WINDOW_MODES and t_ndecay > t_end:
            t_end, rule = t_ndecay, 'n_decay'
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError(
            f"select_window(L={L_um}um, {geometry}, Nk={Nk}, "
            f"mode={window_mode!r}): non-finite or non-positive window "
            f"t_end={t_end!r} (rule={rule!r}, gamma_dom={gamma_dom_hz!r}).")
    return WindowChoice(t_end=t_end, rule=rule, t_candidate_ndecay_s=t_ndecay,
                        t_candidate_crossing_s=t_crossing,
                        gamma_dom_hz=gamma_dom_hz, mode=window_mode,
                        crossing_thresh=window_crossing_thresh)


@dataclass
class SiCase:
    """Full-Si TTG case: modes + window + DOM reference (analyzer units)."""
    v: np.ndarray
    tau: np.ndarray
    C: np.ndarray
    q: float
    t_end: float
    t_ref: np.ndarray
    A_ref: np.ndarray
    gamma_dom_hz: float
    tau_ref: float
    L_um: float = field(default=1.0)
    Nk: int = field(default=20)
    window_mode: str = field(default='n_decay')
    window_rule: str = field(default='n_decay')
    t_candidate_ndecay_s: float = field(default=float('nan'))
    t_candidate_crossing_s: float = field(default=float('nan'))
    window_crossing_thresh: float = field(default=WINDOW_CROSSING_THRESH)
    arbiter_window_rule: str = field(default=WINDOW_RULE_FULL)

    def window_provenance(self) -> dict:
        return {
            'window_mode': self.window_mode,
            'Lt_rule': self.window_rule,
            'Lt_candidate_crossing_tau': self.t_candidate_crossing_s / self.tau_ref,
            'Lt_candidate_ndecay_tau': self.t_candidate_ndecay_s / self.tau_ref,
            't_candidate_crossing_s': self.t_candidate_crossing_s,
            't_candidate_ndecay_s': self.t_candidate_ndecay_s,
            'window_crossing_thresh': self.window_crossing_thresh,
            'arbiter_window_rule': self.arbiter_window_rule,
        }

    def comb_provenance(self) -> dict:
        from pinn_bte.physics.ttg_dispersion import comb_fingerprint
        return comb_fingerprint(self.v, self.tau, self.C)


def build_si_case(L_um: float, Nk: int = 20, T_ref: float = 300.0,
                  n_decay_times: float = 7.0, geometry: str = "1d",
                  window_mode: str = "n_decay",
                  window_crossing_thresh: float = WINDOW_CROSSING_THRESH,
                  window_rule: str | None = None,
                  ) -> SiCase:
    """Mode set + DOM window/reference for a Si TTG at grating period L_um."""
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    from pinn_bte.physics.ttg_dom import ttg_amplitude_curve

    win = select_window(L_um, Nk=Nk, T_ref=T_ref,
                        n_decay_times=n_decay_times, geometry=geometry,
                        window_mode=window_mode,
                        window_crossing_thresh=window_crossing_thresh,
                        window_rule=window_rule)
    v, tau, C = phonon_modes(Nk, T_ref)
    q = 2.0 * np.pi / (L_um * ANGSTROM_PER_UM)
    t_ref, A_ref = ttg_amplitude_curve(L_um, geometry, Nk=Nk, T_ref=T_ref,
                                       t_end_s=win.t_end)
    tau_ref = float(np.sum(C) / np.sum(C / tau))
    return SiCase(v=v, tau=tau, C=C, q=q, t_end=win.t_end,
                  t_ref=t_ref, A_ref=A_ref, gamma_dom_hz=win.gamma_dom_hz,
                  tau_ref=tau_ref, L_um=L_um, Nk=Nk,
                  window_mode=win.mode, window_rule=win.rule,
                  t_candidate_ndecay_s=win.t_candidate_ndecay_s,
                  t_candidate_crossing_s=win.t_candidate_crossing_s,
                  window_crossing_thresh=win.crossing_thresh,
                  arbiter_window_rule=(WINDOW_RULE_FULL if window_rule is None
                                       else window_rule))


def _collocation_times(cfg: SpectralPNConfig) -> np.ndarray:
    if cfg.t_grid == "uniform":
        return np.linspace(0.0, cfg.t_end, cfg.nt)
    if cfg.t_grid == "log-mix":
        # half uniform coverage + half early-decade density (the ballistic
        # drop happens in the first percent of the window at small L)
        n_uni = cfg.nt // 2
        n_log = cfg.nt - n_uni
        uni = np.linspace(0.0, cfg.t_end, n_uni)
        log = np.geomspace(cfg.t_end * 1e-4, cfg.t_end, n_log)
        return np.unique(np.concatenate([uni, log]))
    raise ValueError(f"unknown t_grid: {cfg.t_grid}")


def _integral_gamma(t: np.ndarray, A: np.ndarray,
                    estimator: str = GAMMA_ESTIMATOR_RULED) -> float:
    return integral_gamma(t, A, estimator=estimator)


def _sign_provenance(t: np.ndarray, A: np.ndarray) -> dict:
    r = amplitude_sign_report(t, A)
    return dict(
        gamma_estimator=GAMMA_ESTIMATOR_RULED,
        gamma_eff_clip=r.gamma_clip,
        gamma_clip_minus_absA_rel=r.clip_minus_absA_rel,
        amp_sign_change=r.sign_change,
        amp_A_min=r.A_min,
        amp_n_negative=r.n_negative,
        amp_n_samples=r.n_samples,
        amp_negative_time_frac=r.negative_time_fraction,
        amp_negative_sample_frac=r.negative_sample_fraction,
    )


def _select_gamma_ref(cfg) -> tuple:
    if cfg.gamma_ref_rule not in DENOMINATOR_RULES:
        raise ValueError(
            f"gamma_ref_rule={cfg.gamma_ref_rule!r} not in {DENOMINATOR_RULES}")
    if cfg.A_ref is None:
        return None, {}
    if cfg.gamma_ref_rule == DENOMINATOR_RULE_SAME_WINDOW:
        gamma_ref = integral_gamma(cfg.t_ref, cfg.A_ref,
                                   estimator=GAMMA_ESTIMATOR_RULED)
    elif cfg.gamma_ref_rule == DENOMINATOR_RULE_ARBITER_WINDOW:
        if cfg.gamma_ref is None:
            raise ValueError(
                f"gamma_ref_rule={DENOMINATOR_RULE_ARBITER_WINDOW!r} needs an "
                f"explicit cfg.gamma_ref (the arbiter's own n_decay rate); "
                f"none was supplied")
        gamma_ref = float(cfg.gamma_ref)
    else:
        raise ValueError(
            f"gamma_ref_rule={cfg.gamma_ref_rule!r} is declared vocabulary but "
            f"no trainer path computes it; refusing to guess a denominator")
    prov = denominator_provenance(
        estimator=GAMMA_ESTIMATOR_RULED, rule=cfg.gamma_ref_rule,
        t_end_s=float(cfg.t_end), n_samples=int(np.asarray(cfg.t_ref).size),
        gamma_ref_hz=gamma_ref,
        gamma_ref_arbiter_window_hz=(float(cfg.gamma_ref)
                                     if cfg.gamma_ref is not None
                                     else float("nan")))
    return gamma_ref, prov


SHAPE_NAMES = ("rmse", "max_abs_dA", "int_signed_dA", "int_abs_dA",
               "A_min", "max_dA_fwd")


def _shape_metrics(t: np.ndarray, A: np.ndarray, A_ref_i: np.ndarray) -> tuple:
    """SHAPE of A(t) vs the reference — the reading gamma_ratio cannot give."""
    dA = A - A_ref_i
    return (float(np.sqrt(np.mean(dA ** 2))),
            float(np.abs(dA).max()),
            float(np.trapezoid(dA, t)),
            float(np.trapezoid(np.abs(dA), t)),
            float(A.min()),
            float(np.diff(A).max()))


def _head_names(net) -> list:
    """The net's REAL top-level parameter groups ('which network'). Derived from
    net carries qnet/q2net.
    """
    return sorted({n.split(".")[0] for n, _ in net.named_parameters()})


def _grad_norms(net, names: list) -> list:
    """Per-head L2 gradient norm."""
    with torch.no_grad():
        sq = {g: 0.0 for g in names}
        for n, p in net.named_parameters():
            if p.grad is not None:
                sq[n.split(".")[0]] += float(p.grad.pow(2).sum())
    return [sq[g] ** 0.5 for g in names]


def _component_mses(comps) -> tuple:
    """(names, per-term MSE, weights) from ((name, tensor, weight), ...)."""
    with torch.no_grad():
        names = [c[0] for c in comps]
        mses = [float((c[1] ** 2).mean()) for c in comps]
        weights = [float(c[2]) for c in comps]
    return names, mses, weights


class _MinLossSnapshot:

    def __init__(self, cfg: SpectralPNConfig):
        self.enabled = cfg.snapshot_selection == "min-loss"
        self._log_every = cfg.log_every
        self._gamma_every = cfg.gamma_log_every
        self._last = cfg.epochs - 1
        self.best_epoch = -1
        self.best_loss = float("inf")
        self._state = None

    def consider(self, it: int, loss: torch.Tensor, net) -> None:
        """Call between backward() and opt.step(); inert unless enabled."""
        if not self.enabled:
            return
        if not (it % self._log_every == 0 or it == self._last
                or (self._gamma_every > 0 and it % self._gamma_every == 0)):
            return
        loss_val = float(loss.detach())
        if loss_val < self.best_loss:  # strict: ties keep the EARLIEST best
            self.best_loss = loss_val
            self.best_epoch = it
            self._state = {k: t.detach().clone()
                           for k, t in net.state_dict().items()}

    def finalize(self, net, loss_final: float) -> tuple:
        """Restore the shipped snapshot; return its (epoch, logged loss)."""
        if self.enabled and self._state is not None:
            net.load_state_dict(self._state)
            return self.best_epoch, self.best_loss
        return self._last, loss_final


def train_spectral_pn(cfg: SpectralPNConfig) -> dict:
    if cfg.snapshot_selection not in ("final", "min-loss"):
        raise ValueError(
            f"snapshot_selection={cfg.snapshot_selection!r} "
            "(expected 'final' or 'min-loss')")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dtype = {"float64": torch.float64, "float32": torch.float32}[cfg.dtype]
    device = torch.device(cfg.device)

    v = torch.as_tensor(np.asarray(cfg.v, dtype=float), dtype=dtype, device=device)
    tau = torch.as_tensor(np.asarray(cfg.tau, dtype=float), dtype=dtype, device=device)
    C = torch.as_tensor(np.asarray(cfg.C, dtype=float), dtype=dtype, device=device)

    if cfg.blend not in BLENDS:
        raise ValueError(f"blend={cfg.blend!r} (expected one of {BLENDS})")
    if cfg.blend != "banded":
        assert cfg.formulation == "xihybrid", \
            "blend='ugks' requires formulation='xihybrid'"
        assert not (cfg.shared_ahat or cfg.carrier_energy or cfg.carrier_row), \
            ("blend='ugks' replaces the carrier machinery (drop "
             "--shared-ahat/--carrier-energy/--carrier-row)")
    if cfg.macro_row not in ("on", "off"):
        raise ValueError(f"macro_row={cfg.macro_row!r} "
                         "(expected 'on' or 'off')")
    if cfg.macro_row == "off":
        assert cfg.blend == "ugks", \
            ("macro_row='off' is the ugks-path ablation "
             "(requires blend='ugks')")
    if cfg.closure_pin:
        if cfg.formulation != "xihybrid" or cfg.blend != "ugks":
            raise ValueError(
                "closure_pin=True requires formulation='xihybrid' and "
                "blend='ugks' (the shim acts on the ugks l=0 slaved slot); "
                f"got formulation={cfg.formulation!r}, blend={cfg.blend!r}")

    gamma_kin = (cfg.q ** 2 * float(np.sum(np.asarray(cfg.C) *
                                           np.asarray(cfg.v) ** 2 *
                                           np.asarray(cfg.tau)))
                 / 3.0 / float(np.sum(cfg.C)))
    if cfg.formulation == "ce":
        assert not (cfg.duhamel or cfg.ap_scaling or cfg.slow_head), \
            "formulation='ce' subsumes ap/slow and excludes duhamel"
        from pinn_bte.models.spectral_pn import CEPNCoefficientNet
        net = CEPNCoefficientNet(n_modes=len(cfg.v), L_max=cfg.L_max,
                                 gamma_kin=gamma_kin, v=cfg.v, tau=cfg.tau,
                                 q=cfg.q, width=cfg.width, depth=cfg.depth,
                                 emb_dim=cfg.emb_dim,
                                 ).to(device=device, dtype=dtype)
    elif cfg.formulation == "xihybrid":
        assert not (cfg.duhamel or cfg.ap_scaling or cfg.slow_head), \
            "formulation='xihybrid' subsumes ap/slow/duhamel"
        from pinn_bte.models.spectral_pn import XiHybridPNCoefficientNet
        net = XiHybridPNCoefficientNet(
            n_modes=len(cfg.v), L_max=cfg.L_max, gamma_kin=gamma_kin,
            v=cfg.v, tau=cfg.tau, q=cfg.q, width=cfg.width,
            depth=cfg.depth, emb_dim=cfg.emb_dim, xi_split=cfg.xi_split,
            xi_duh=cfg.xi_duh, shared_ahat=cfg.shared_ahat, C=cfg.C,
            carrier_energy=cfg.carrier_energy, blend=cfg.blend,
            macro_row=cfg.macro_row, closure_pin=cfg.closure_pin,
            gauge=cfg.gauge,
        ).to(device=device, dtype=dtype)
    elif cfg.formulation == "plain":
        net = PNCoefficientNet(n_modes=len(cfg.v), L_max=cfg.L_max,
                               width=cfg.width, depth=cfg.depth,
                               emb_dim=cfg.emb_dim,
                               gamma_slow=(gamma_kin if cfg.slow_head
                                           else None),
                               duhamel=cfg.duhamel,
                               ).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"formulation={cfg.formulation!r}")
    init_prov = {"init_from": None, "init_from_sha256": None, "data_free_on_rate": True}
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    decay_horizon = cfg.lr_decay_epochs or cfg.epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=decay_horizon, eta_min=cfg.lr * 0.05)
        if cfg.lr_cosine else None)
    epochs_cap, settle_on = _settle_plan(cfg)

    t_coll = torch.as_tensor(_collocation_times(cfg), dtype=dtype, device=device)

    t_eval = np.linspace(0.0, cfg.t_end, max(801, cfg.nt))
    te = torch.as_tensor(t_eval, dtype=dtype, device=device)

    def _amplitude() -> np.ndarray:
        with torch.no_grad():
            A_t = pn_amplitude(net, te, tau, C, v=v, q=cfg.q)
        return A_t.cpu().numpy()

    gamma_ref, gamma_ref_prov = _select_gamma_ref(cfg)
    A_ref_i = None
    if cfg.A_ref is not None:
        A_ref_i = np.interp(t_eval, cfg.t_ref, cfg.A_ref)

    def _loss(tc, want_components: bool = False):
        """The loss expression is IDENTICAL to the shipped one; want_components
        only additionally reports the (detached) per-term means.
        """
        if cfg.formulation == "ce":
            res, macro = pn_residual(net, tc, v, tau, C, q=cfg.q,
                                     ap_scaling=True, return_macro=True,
                                     weight_floor=cfg.residual_weight_floor)
            loss = (res ** 2).mean() + (macro ** 2).mean()
            comps = (("res", res, 1.0), ("macro", macro, 1.0))
        elif cfg.formulation == "xihybrid" and cfg.carrier_row:
            res, carrier = pn_residual(net, tc, v, tau, C, q=cfg.q,
                                       ap_scaling=True,
                                       return_carrier=True,
                                       weight_floor=cfg.residual_weight_floor)
            loss = (res ** 2).mean() + (carrier ** 2).mean()
            comps = (("res", res, 1.0), ("carrier", carrier, 1.0))
        elif (cfg.formulation == "xihybrid"
              and getattr(net, "macro_licensed", False)):
            res, macro = pn_residual(net, tc, v, tau, C, q=cfg.q,
                                     ap_scaling=True, return_macro=True,
                                     weight_floor=cfg.residual_weight_floor)
            loss = (res ** 2).mean() + (macro ** 2).mean()
            comps = (("res", res, 1.0), ("macro", macro, 1.0))
        else:
            res = pn_residual(net, tc, v, tau, C, q=cfg.q,
                              ap_scaling=(cfg.ap_scaling or
                                          cfg.formulation == "xihybrid"),
                              weight_floor=cfg.residual_weight_floor)
            loss = (res ** 2).mean()
            comps = (("res", res, 1.0),)
        return (loss, _component_mses(comps) if want_components else None)

    loss_initial = None
    history = []
    gamma_history = []
    shape_history = []
    log_gamma = cfg.gamma_log_every > 0 and gamma_ref is not None
    component_history, component_names, component_weights = [], [], []
    gradnorm_names = _head_names(net) if cfg.component_log_every > 0 else []
    snapshot = _MinLossSnapshot(cfg)
    settled, epochs_to_settle, it = False, None, -1
    for it in range(epochs_cap):
        hard_last = (it == epochs_cap - 1)
        log_comp = cfg.component_log_every > 0 and (
            it % cfg.component_log_every == 0 or hard_last)
        opt.zero_grad()
        loss, comps = _loss(t_coll, want_components=log_comp)
        loss.backward()
        # pre-step, like loss_val below: the snapshot pairs the state with the
        # loss it produced (pure reads — cannot perturb the trajectory)
        snapshot.consider(it, loss, net)
        if log_comp:
            # read AFTER backward (grads exist) and BEFORE opt.step(): this is
            # the gradient that produced this step, and reading is inert
            lr_now = opt.param_groups[0]["lr"]
            grad_norms = _grad_norms(net, gradnorm_names)
        opt.step()
        if sched is not None and sched.last_epoch < decay_horizon:
            sched.step()
        loss_val = float(loss.detach())
        if log_comp:
            component_names, component_mses, component_weights = comps
            component_history.append(
                (it, lr_now, loss_val, *component_mses, *grad_norms))
        if loss_initial is None:
            loss_initial = loss_val
        if it % cfg.log_every == 0 or hard_last:
            history.append((it, loss_val))
        if log_gamma and (it % cfg.gamma_log_every == 0 or hard_last):
            # ONE amplitude evaluation feeds BOTH the rate and the shape: the
            # forward pass is the expensive part and gamma already paid for it.
            A_now = _amplitude()
            gamma_history.append(
                (it, _integral_gamma(t_eval, A_now) / gamma_ref, loss_val))
            shape_history.append((it, *_shape_metrics(t_eval, A_now, A_ref_i)))
            # cfg.epochs is the FLOOR: the rule may not fire below it however
            # flat the curve looks early.
            if settle_on and (it + 1) >= cfg.epochs:
                _gh = np.asarray(gamma_history, dtype=float)
                if gamma_settled(_gh[:, 0], _gh[:, 1],
                                 cfg.settle_window, cfg.settle_tol):
                    settled, epochs_to_settle = True, it + 1
                    break
    epochs_run = it + 1
    loss_final = loss_val
    best_epoch, best_loss = snapshot.finalize(net, loss_final)

    # --- evaluation on a dense uniform grid (at the SHIPPED snapshot) ------
    A = _amplitude()

    metrics = dict(loss_initial=loss_initial, loss_final=loss_final,
                   history=history, gamma_history=gamma_history,
                   shape_history=shape_history,
                   shape_names=list(SHAPE_NAMES),
                   component_history=component_history,
                   component_names=component_names,
                   component_weights=component_weights,
                   gradnorm_names=gradnorm_names,
                   snapshot_selection=cfg.snapshot_selection,
                   init_provenance=init_prov,
                   best_epoch=best_epoch, best_loss=best_loss,
                   settle_window=cfg.settle_window,
                   settle_tol=cfg.settle_tol,
                   epochs_cap=epochs_cap,
                   lr_decay_epochs=cfg.lr_decay_epochs,
                   settled=settled,
                   epochs_to_settle=epochs_to_settle,
                   epochs_run=epochs_run,
                   t=t_eval, A=A)
    gamma_eff = _integral_gamma(t_eval, A)
    metrics["gamma_eff"] = gamma_eff
    sign_prov = _sign_provenance(t_eval, A)
    metrics.update(sign_prov)
    metrics.update(gamma_ref_prov)

    if cfg.A_ref is not None:
        shape = _shape_metrics(t_eval, A, A_ref_i)
        metrics.update(
            zip(SHAPE_NAMES, shape),          # rmse/max_abs_dA + the shape set
            gamma_ratio=gamma_eff / gamma_ref,
            A_ref_interp=A_ref_i,
        )

    if cfg.outdir is not None:
        run_id = cfg.run_id or "spn"
        outdir = Path(cfg.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / f"{run_id}_results.npz",
            t=t_eval, A=A,
            t_ref=(cfg.t_ref if cfg.t_ref is not None else np.zeros(0)),
            A_ref=(cfg.A_ref if cfg.A_ref is not None else np.zeros(0)),
            loss_history=np.asarray(history, dtype=float),
            gamma_history=np.asarray(gamma_history,
                                     dtype=float).reshape(-1, 3),
            gamma_log_every=cfg.gamma_log_every,
            shape_history=np.asarray(shape_history, dtype=float).reshape(
                -1, 1 + len(SHAPE_NAMES)),
            shape_names=np.asarray(SHAPE_NAMES, dtype=object),
            component_history=np.asarray(
                component_history, dtype=float).reshape(
                    -1, 3 + len(component_names) + len(gradnorm_names)),
            component_names=np.asarray(component_names, dtype=object),
            component_weights=np.asarray(component_weights, dtype=float),
            gradnorm_names=np.asarray(gradnorm_names, dtype=object),
            component_log_every=cfg.component_log_every,
            gamma_eff=gamma_eff,
            gamma_ratio=metrics.get("gamma_ratio", np.nan),
            rmse=metrics.get("rmse", np.nan),
            max_abs_dA=metrics.get("max_abs_dA", np.nan),
            int_signed_dA=metrics.get("int_signed_dA", np.nan),
            int_abs_dA=metrics.get("int_abs_dA", np.nan),
            A_min=metrics.get("A_min", np.nan),
            max_dA_fwd=metrics.get("max_dA_fwd", np.nan),
            **sign_prov,
            **gamma_ref_prov,
            L_max=cfg.L_max, label=cfg.label,
            q=cfg.q, t_end=cfg.t_end, width=cfg.width, depth=cfg.depth,
            epochs=cfg.epochs, lr=cfg.lr, nt=cfg.nt,
            epochs_run=epochs_run,
            epochs_to_settle=(-1 if epochs_to_settle is None
                              else epochs_to_settle),
            settled=settled,
            settle_window=cfg.settle_window, settle_tol=cfg.settle_tol,
            epochs_cap=epochs_cap,
            lr_decay_epochs=(-1 if cfg.lr_decay_epochs is None
                             else cfg.lr_decay_epochs),
            nt_actual=len(t_coll), seed=cfg.seed,
            dtype=cfg.dtype, t_grid=cfg.t_grid,
            ap_scaling=cfg.ap_scaling, slow_head=cfg.slow_head,
            duhamel=cfg.duhamel, formulation=cfg.formulation,
            blend=cfg.blend,
            macro_row=cfg.macro_row,
            closure_pin=cfg.closure_pin,
            **init_prov,
            xi_split=cfg.xi_split, xi_duh=cfg.xi_duh,
            shared_ahat=cfg.shared_ahat,
            carrier_energy=cfg.carrier_energy, carrier_row=cfg.carrier_row,
            snapshot_selection=cfg.snapshot_selection,
            best_epoch=best_epoch, best_loss=best_loss,
            final_loss=loss_final,
            **(cfg.extra_meta or {}),
        )
        torch.save(net.state_dict(), outdir / f"net_pn_{run_id}.pt")
        metrics["run_dir"] = outdir

    return metrics
