"""The macro-row ablation of the ugks blend: default byte-identical, 'off'
removes exactly the macro term.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import (
    XiHybridPNCoefficientNet,
    pn_residual,
)
from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    _collocation_times,
    build_si_case,
    train_spectral_pn,
)

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.usefixtures("default_float64")


# --------------------------------------------------------------- fixtures
def _gray_three():
    v = torch.tensor([0.1, 3.0, 30.0])
    tau = torch.tensor([1.0, 1.0, 1.0])
    C = torch.tensor([1.0, 1.0, 1.0])
    return v, tau, C, 1.0


def _gamma_kin(v, tau, C, q):
    return float(q ** 2 * torch.sum(C * v ** 2 * tau) / 3.0 / torch.sum(C))


def _net(v, tau, C, q, seed=0, blend="ugks", L_max=4, width=16, depth=2,
         **kw):
    torch.manual_seed(seed)
    return XiHybridPNCoefficientNet(
        n_modes=len(v), L_max=L_max, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q, width=width, depth=depth,
        blend=blend, **kw).double()


def _tiny_cfg(**overrides):
    cfg = dict(
        v=np.array([0.1, 3.0, 30.0]), tau=np.ones(3), C=np.ones(3),
        q=1.0, t_end=6.0, L_max=4, width=16, depth=2,
        epochs=3, lr=2e-3, nt=41, seed=0, device="cpu", dtype="float64",
        formulation="xihybrid", blend="ugks", label="macro-row-test",
    )
    cfg.update(overrides)
    return SpectralPNConfig(**cfg)


def _manual_net_and_inputs(cfg):
    """Replicate the trainer's exact seeding + net construction + collocation
    grid, so manually composed losses are bitwise comparable to its output.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    v = torch.as_tensor(np.asarray(cfg.v, dtype=float), dtype=torch.float64)
    tau = torch.as_tensor(np.asarray(cfg.tau, dtype=float),
                          dtype=torch.float64)
    C = torch.as_tensor(np.asarray(cfg.C, dtype=float), dtype=torch.float64)
    gamma_kin = (cfg.q ** 2 * float(np.sum(np.asarray(cfg.C) *
                                           np.asarray(cfg.v) ** 2 *
                                           np.asarray(cfg.tau)))
                 / 3.0 / float(np.sum(cfg.C)))
    net = XiHybridPNCoefficientNet(
        n_modes=len(cfg.v), L_max=cfg.L_max, gamma_kin=gamma_kin,
        v=cfg.v, tau=cfg.tau, q=cfg.q, width=cfg.width,
        depth=cfg.depth, emb_dim=cfg.emb_dim, xi_split=cfg.xi_split,
        xi_duh=cfg.xi_duh, blend=cfg.blend, macro_row=cfg.macro_row,
    ).to(dtype=torch.float64)
    t_coll = torch.as_tensor(_collocation_times(cfg), dtype=torch.float64)
    return net, t_coll, v, tau, C


@pytest.mark.parametrize("blend", ["ugks", "banded"])
def test_macro_row_default_is_byte_identical(blend):
    v, tau, C, q = _gray_three()
    net0 = _net(v, tau, C, q, seed=3, blend=blend)
    net1 = _net(v, tau, C, q, seed=3, blend=blend, macro_row="on")
    assert net0.macro_licensed == net1.macro_licensed
    t = torch.linspace(0.0, 4.0, 17, dtype=torch.float64)
    c0, s0 = net0.coefficients(t, v, tau, C, q)
    c1, s1 = net1.coefficients(t, v, tau, C, q)
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    r0, m0 = pn_residual(net0, t, v, tau, C, q, ap_scaling=True,
                         return_macro=True)
    r1, m1 = pn_residual(net1, t, v, tau, C, q, ap_scaling=True,
                         return_macro=True)
    assert torch.equal(r0, r1) and torch.equal(m0, m1)


def test_default_ugks_loss_is_res_plus_macro():
    cfg = _tiny_cfg(epochs=1)
    res_out = train_spectral_pn(cfg)
    net, t_coll, v, tau, C = _manual_net_and_inputs(cfg)
    res, macro = pn_residual(net, t_coll, v, tau, C, q=cfg.q,
                             ap_scaling=True, return_macro=True)
    expected = float(((res ** 2).mean() + (macro ** 2).mean()).detach())
    assert res_out["loss_initial"] == expected


def test_macro_row_off_unlicenses_the_row():
    v, tau, C, q = _gray_three()
    assert _net(v, tau, C, q).macro_licensed          # ugks default: ON
    off = _net(v, tau, C, q, macro_row="off")
    assert not off.macro_licensed
    assert off.macro_row == "off"


def test_macro_row_off_is_res_only_exactly():
    """Trainer loss under 'off' == the res-only value computed independently
    (bitwise), and the on/off gap at init is EXACTLY (macro^2).mean().
    """
    cfg_off = _tiny_cfg(epochs=1, macro_row="off")
    loss_off = train_spectral_pn(cfg_off)["loss_initial"]
    net, t_coll, v, tau, C = _manual_net_and_inputs(cfg_off)
    res, macro = pn_residual(net, t_coll, v, tau, C, q=cfg_off.q,
                             ap_scaling=True, return_macro=True)
    assert loss_off == float((res ** 2).mean().detach())
    loss_on = train_spectral_pn(_tiny_cfg(epochs=1))["loss_initial"]
    macro_mse = float((macro ** 2).mean().detach())
    assert macro_mse > 0.0, "macro term trivially zero — the test is blind"
    assert loss_on == loss_off + macro_mse


def test_macro_row_off_component_decomposition():
    """M1b: the logged loss decomposition shows the row gone — ['res'] under 'off'
    vs ['res', 'macro'] under 'on'.
    """
    on = train_spectral_pn(_tiny_cfg(epochs=1, component_log_every=1))
    assert on["component_names"] == ["res", "macro"]
    off = train_spectral_pn(_tiny_cfg(epochs=1, component_log_every=1,
                                      macro_row="off"))
    assert off["component_names"] == ["res"]


def test_macro_row_off_gradient_is_res_only():
    v, tau, C, q = _gray_three()
    net_off = _net(v, tau, C, q, seed=7, macro_row="off")
    net_on = _net(v, tau, C, q, seed=7)
    for p0, p1 in zip(net_off.parameters(), net_on.parameters()):
        assert torch.equal(p0, p1), "seeding drifted — grads not comparable"
    t = torch.linspace(0.0, 4.0, 17, dtype=torch.float64)

    (pn_residual(net_off, t, v, tau, C, q, ap_scaling=True) ** 2
     ).mean().backward()
    res_on = pn_residual(net_on, t, v, tau, C, q, ap_scaling=True)
    (res_on ** 2).mean().backward()
    grads_off = [p.grad.clone() for p in net_off.parameters()]
    grads_res_only = [p.grad.clone() for p in net_on.parameters()]
    for g0, g1 in zip(grads_off, grads_res_only):
        assert torch.equal(g0, g1)

    net_on.zero_grad()
    res, macro = pn_residual(net_on, t, v, tau, C, q, ap_scaling=True,
                             return_macro=True)
    ((res ** 2).mean() + (macro ** 2).mean()).backward()
    grads_full = [p.grad.clone() for p in net_on.parameters()]
    assert any(not torch.equal(g0, g1)
               for g0, g1 in zip(grads_res_only, grads_full)), \
        "macro row carries no gradient at all — the exclusion is untestable"


def test_macro_row_off_training_matches_manual_res_only_loop():
    """M2b end-to-end: a 3-epoch trainer run under 'off' == a manual Adam+cosine
    res-only loop, bitwise at every logged epoch.
    """
    cfg = _tiny_cfg(epochs=3, log_every=1, macro_row="off")
    history = train_spectral_pn(cfg)["history"]

    net, t_coll, v, tau, C = _manual_net_and_inputs(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    manual = []
    for it in range(cfg.epochs):
        opt.zero_grad()
        loss = (pn_residual(net, t_coll, v, tau, C, q=cfg.q,
                            ap_scaling=True) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        manual.append((it, float(loss.detach())))
    assert history == manual


def test_macro_row_invalid_value_rejected():
    v, tau, C, q = _gray_three()
    with pytest.raises(ValueError, match="macro_row"):
        _net(v, tau, C, q, macro_row="maybe")
    with pytest.raises(ValueError, match="macro_row"):
        train_spectral_pn(_tiny_cfg(macro_row="maybe"))


def test_macro_row_off_requires_ugks_net():
    v, tau, C, q = _gray_three()
    with pytest.raises(ValueError, match="ugks"):
        _net(v, tau, C, q, blend="banded", macro_row="off")


def test_cli_macro_row_flag():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_spectral_pn_cli_macro_test",
        REPO / "experiments" / "transient" / "run_spectral_pn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_args(["--L", "1.0"]).macro_row == "on"
    args = mod.parse_args(["--L", "1.0", "--formulation", "xihybrid",
                           "--blend", "ugks", "--macro-row", "off"])
    assert args.macro_row == "off"
    with pytest.raises(SystemExit):            # off requires --blend ugks
        mod.parse_args(["--L", "1.0", "--macro-row", "off"])
    with pytest.raises(SystemExit):            # xihybrid alone is not enough
        mod.parse_args(["--L", "1.0", "--formulation", "xihybrid",
                        "--macro-row", "off"])


def test_macro_row_recorded_in_npz(tmp_path):
    """Every CLI param goes in the artifact (the unrecorded-subsample OOM lesson):
    macro_row lands in the 1q results npz, both values.
    """
    train_spectral_pn(_tiny_cfg(epochs=1, macro_row="off",
                                outdir=tmp_path, run_id="moff"))
    assert str(np.load(tmp_path / "moff_results.npz",
                       allow_pickle=True)["macro_row"]) == "off"
    train_spectral_pn(_tiny_cfg(epochs=1, outdir=tmp_path, run_id="mon"))
    assert str(np.load(tmp_path / "mon_results.npz",
                       allow_pickle=True)["macro_row"]) == "on"


