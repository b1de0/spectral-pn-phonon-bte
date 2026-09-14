"""Per-term loss decomposition logging: opt-in, byte-identical trajectory,
components reconstruct the aggregate.
"""
import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    build_si_case,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")


@pytest.fixture(scope="module")
def case():
    return build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")


def _cfg_1q(case, **kw):
    base = dict(
        v=case.v, tau=case.tau, C=case.C, q=case.q, t_end=case.t_end,
        L_max=4, width=16, depth=2, emb_dim=4, epochs=8, lr=2e-3,
        lr_cosine=True, nt=60, t_grid="uniform", seed=42, device="cpu",
        dtype="float64", t_ref=case.t_ref, A_ref=case.A_ref,
        gamma_ref=case.gamma_dom_hz, Nk=6, T_ref=300.0, log_every=1,
        formulation="xihybrid", shared_ahat=True,
        carrier_energy=True, carrier_row=True, xi_split=1.0, xi_duh=10.0,
    )
    base.update(kw)
    return SpectralPNConfig(**base)


CFGS = {"1q": _cfg_1q}
EXPECTED_HEADS = {"1q": ["emb", "slow", "trunk"]}
EXPECTED_COMPONENTS = {"1q": ["res", "carrier"]}


@pytest.mark.parametrize("path", ["1q"])
def test_component_log_defaults_off(case, path):
    cfg = CFGS[path](case)
    assert cfg.component_log_every == 0
    r = train_spectral_pn(cfg)
    assert r["component_history"] == []


# ============================== (b) the logger does not perturb training
@pytest.mark.parametrize("path", ["1q"])
def test_component_logging_leaves_trajectory_byte_identical(case, path):
    """THE non-negotiable: reading .grad and the per-term means must not move a
    single bit of the run.
    """
    r_off = train_spectral_pn(CFGS[path](case, component_log_every=0))
    r_on = train_spectral_pn(CFGS[path](case, component_log_every=1))

    assert r_on["history"] == r_off["history"], (
        f"[{path}] loss trajectory perturbed by the logger:\n"
        f"  off={r_off['history']}\n  on ={r_on['history']}")
    assert r_on["loss_final"] == r_off["loss_final"], \
        f"[{path}] {r_on['loss_final']!r} vs {r_off['loss_final']!r}"
    assert r_on["gamma_eff"] == r_off["gamma_eff"], \
        f"[{path}] {r_on['gamma_eff']!r} vs {r_off['gamma_eff']!r}"
    assert r_on["gamma_ratio"] == r_off["gamma_ratio"], \
        f"[{path}] {r_on['gamma_ratio']!r} vs {r_off['gamma_ratio']!r}"
    assert np.array_equal(r_on["A"], r_off["A"]), f"[{path}] published A moved"


@pytest.mark.parametrize("path", ["1q"])
def test_component_logging_consumes_no_rng(case, path):
    """These loops draw no RNG after init, so the byte-identity test above is
    STRUCTURALLY BLIND to a stray draw (proven: inserting torch.rand(1) leaves
    it green).
    """
    def _rng_state_after(component_log_every):
        train_spectral_pn(CFGS[path](case, component_log_every=component_log_every))
        return torch.random.get_rng_state()

    assert torch.equal(_rng_state_after(1), _rng_state_after(0)), \
        f"[{path}] the component logger advanced the global torch RNG"


@pytest.mark.parametrize("path", ["1q"])
def test_component_logging_leaves_gamma_curve_byte_identical(case, path):
    r_off = train_spectral_pn(CFGS[path](case, gamma_log_every=1))
    r_on = train_spectral_pn(CFGS[path](case, gamma_log_every=1,
                                        component_log_every=1))
    assert r_on["gamma_history"] == r_off["gamma_history"]


# ============================================ (c) THE FLAGSHIP: sum consistency
@pytest.mark.parametrize("path", ["1q"])
def test_components_reconstruct_the_aggregate_loss(case, path):
    """The decomposition must ACCOUNT FOR the aggregate, weights included —
    otherwise a term could be silently omitted and the split would still look
    plausible.
    """
    cfg = CFGS[path](case, component_log_every=1)
    r = train_spectral_pn(cfg)
    ch = np.asarray(r["component_history"], dtype=float)
    names = r["component_names"]
    w = np.asarray(r["component_weights"], dtype=float)

    assert names == EXPECTED_COMPONENTS[path], names
    loss_col = ch[:, 2]
    mse = ch[:, 3:3 + len(names)]
    recon = mse @ w
    rel = np.abs(recon - loss_col) / np.abs(loss_col)
    assert rel.max() < 1e-12, (
        f"[{path}] components do not reconstruct the loss: max rel {rel.max():.3e}\n"
        f"  names={names} weights={w.tolist()}\n"
        f"  recon={recon[:3]} loss={loss_col[:3]}")


@pytest.mark.parametrize("path", ["1q"])
def test_gradnorm_columns_name_the_real_param_groups(case, path):
    """'Which net is still learning' is only answerable if the head names are the
    net's ACTUAL parameter groups.
    """
    r = train_spectral_pn(CFGS[path](case, component_log_every=1))
    assert r["gradnorm_names"] == EXPECTED_HEADS[path], r["gradnorm_names"]

    ch = np.asarray(r["component_history"], dtype=float)
    n_c = len(r["component_names"])
    gn = ch[:, 3 + n_c:]
    assert gn.shape == (8, len(EXPECTED_HEADS[path])), gn.shape
    assert np.isfinite(gn).all(), "non-finite grad norm"
    assert (gn[0] > 0).all(), f"[{path}] a head had zero gradient at init: {gn[0]}"


@pytest.mark.parametrize("path", ["1q"])
def test_lr_column_tracks_the_cosine_schedule(case, path):
    cfg = CFGS[path](case, component_log_every=1)
    ch = np.asarray(train_spectral_pn(cfg)["component_history"], dtype=float)
    lr = ch[:, 1]
    assert lr[0] == cfg.lr, f"{lr[0]!r} != {cfg.lr!r}"
    assert (np.diff(lr) < 0).all(), f"cosine LR should decay monotonically: {lr}"


@pytest.mark.parametrize("path", ["1q"])
def test_component_history_loss_column_matches_the_loss_history(case, path):
    r = train_spectral_pn(CFGS[path](case, component_log_every=1))
    ch = np.asarray(r["component_history"], dtype=float)
    losses = np.asarray(r["history"], dtype=float)
    assert np.array_equal(ch[:, 0], losses[:, 0])
    assert np.array_equal(ch[:, 2], losses[:, 1])


@pytest.mark.parametrize("path", ["1q"])
def test_component_log_cadence_is_honoured(case, path):
    cfg = CFGS[path](case, epochs=8, component_log_every=3)
    ch = np.asarray(train_spectral_pn(cfg)["component_history"], dtype=float)
    assert list(ch[:, 0]) == [0, 3, 6, 7]


# ============================================== (e) provenance in the npz
@pytest.mark.parametrize("path", ["1q"])
def test_component_curve_and_flag_recorded_in_npz(tmp_path, case, path):
    cfg = CFGS[path](case, component_log_every=2, outdir=tmp_path,
                     run_id=f"cl_{path}")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / f"cl_{path}_results.npz", allow_pickle=True)
    assert int(data["component_log_every"]) == 2
    ch = data["component_history"]
    n_c, n_h = len(EXPECTED_COMPONENTS[path]), len(EXPECTED_HEADS[path])
    assert ch.ndim == 2 and ch.shape[1] == 3 + n_c + n_h, ch.shape
    assert list(data["component_names"]) == EXPECTED_COMPONENTS[path]
    assert list(data["gradnorm_names"]) == EXPECTED_HEADS[path]
    assert len(data["component_weights"]) == n_c
    # the gamma logger's keys and every pre-existing key survive untouched
    for k in ("t", "A", "loss_history", "gamma_history", "gamma_log_every",
              "gamma_eff", "gamma_ratio", "rmse"):
        assert k in data.files, k


@pytest.mark.parametrize("path", ["1q"])
def test_component_log_off_records_empty_curve(tmp_path, case, path):
    cfg = CFGS[path](case, outdir=tmp_path, run_id=f"cloff_{path}")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / f"cloff_{path}_results.npz", allow_pickle=True)
    assert int(data["component_log_every"]) == 0
    assert data["component_history"].shape[0] == 0


# ======================================= the runner exposes it on BOTH paths
@pytest.mark.parametrize("path", ["1q"])
def test_runner_accepts_component_log_every(path):
    from experiments.transient.run_spectral_pn import parse_args
    args = parse_args(["--L", "1.0", 
                       "--component-log-every", "250"])
    assert args.component_log_every == 250


def test_runner_rejects_negative_component_log_every():
    from experiments.transient.run_spectral_pn import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--L", "1.0", "--component-log-every", "-1"])
