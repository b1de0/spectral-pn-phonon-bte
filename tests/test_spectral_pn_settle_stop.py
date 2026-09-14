"""The answer-blind STOPPING RULE and the decay-then-hold LR schedule."""
import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SETTLE_TOL_DECLARED,
    SETTLE_WINDOW_DECLARED,
    SpectralPNConfig,
    build_si_case,
    gamma_settled,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")


@pytest.fixture(scope="module")
def case():
    return build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")


def _cfg(case, **kw):
    base = dict(
        v=case.v, tau=case.tau, C=case.C, q=case.q, t_end=case.t_end,
        L_max=4, width=16, depth=2, emb_dim=4, epochs=8, lr=2e-3,
        lr_cosine=True, nt=60, t_grid="uniform", seed=42, device="cpu",
        dtype="float64", t_ref=case.t_ref, A_ref=case.A_ref,
        gamma_ref=case.gamma_dom_hz, Nk=6, T_ref=300.0, log_every=1,
        formulation="xihybrid", carrier_row=True, shared_ahat=True,
    )
    base.update(kw)
    return SpectralPNConfig(**base)


def test_declared_bar_is_the_one_the_gate_names():
    """The bar lives in ONE place."""
    assert SETTLE_WINDOW_DECLARED == 5000
    assert SETTLE_TOL_DECLARED == 0.005


def test_settled_when_flat_within_tolerance():
    ep = np.arange(0, 20001, 250, dtype=float)
    g = np.full_like(ep, 0.997)
    assert gamma_settled(ep, g, window=5000, tol=0.005)


def test_not_settled_while_still_moving():
    ep = np.arange(0, 20001, 250, dtype=float)
    # a 0.02 drift across the trailing window -- four times the tolerance
    g = 0.90 + 0.10 * ep / ep[-1]
    assert not gamma_settled(ep, g, window=5000, tol=0.005)


def test_window_must_be_full_before_the_rule_may_fire():
    """A short history is trivially 'flat'."""
    ep = np.array([0.0, 250.0, 500.0])
    g = np.array([0.997, 0.997, 0.997])
    assert not gamma_settled(ep, g, window=5000, tol=0.005)


def test_rule_reads_only_epochs_and_gamma():
    """ANSWER-BLINDNESS, mechanically."""
    import inspect
    params = set(inspect.signature(gamma_settled).parameters)
    assert params == {"epochs", "gammas", "window", "tol"}


def test_spread_is_max_minus_min_not_a_slope():
    ep = np.arange(0, 20001, 250, dtype=float)
    g = 0.99 + 0.004 * np.sin(ep / 400.0)   # zero net slope, 0.008 spread
    assert not gamma_settled(ep, g, window=5000, tol=0.005)


def _lr_trace(cfg):
    out = train_spectral_pn(cfg)
    ch = np.asarray(out["component_history"], dtype=float)
    return ch[:, 0].astype(int), ch[:, 1]


def test_default_schedule_is_untouched(case):
    """lr_decay_epochs=None must reproduce CosineAnnealingLR(T_max=epochs)
    exactly. The 20 000-epoch ladder is pinned paper evidence; this option may
    not perturb it.
    """
    it, lr = _lr_trace(_cfg(case, epochs=12, component_log_every=1))
    net = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=12, eta_min=2e-3 * 0.05)
    want = []
    for _ in range(12):
        want.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    np.testing.assert_allclose(lr, want, rtol=0, atol=0)


def test_decay_then_hold_reaches_eta_min_and_stays(case):
    """After the decay horizon the LR must HOLD at eta_min."""
    eta_min = 2e-3 * 0.05
    it, lr = _lr_trace(_cfg(case, epochs=6, epochs_cap=18,
                            lr_decay_epochs=6, settle_window=0,
                            component_log_every=1))
    assert len(lr) == 18
    assert lr[-1] == pytest.approx(eta_min, rel=1e-12)
    held = lr[6:]
    np.testing.assert_allclose(held, eta_min, rtol=1e-12)
    assert held.max() <= lr[0]          # never climbs back


def test_hold_phase_is_strictly_non_increasing(case):
    it, lr = _lr_trace(_cfg(case, epochs=4, epochs_cap=16,
                            lr_decay_epochs=4, settle_window=0,
                            component_log_every=1))
    assert np.all(np.diff(lr) <= 1e-15)


def test_settle_off_runs_exactly_epochs(case):
    """(a) The rule OFF is the current behaviour, to the epoch."""
    out = train_spectral_pn(_cfg(case, epochs=9, gamma_log_every=1))
    assert out["epochs_run"] == 9
    assert out["settled"] is False
    assert out["epochs_to_settle"] is None


def test_epochs_is_a_MINIMUM_when_the_rule_is_on(case):
    """The rule may never fire before the declared minimum budget, however flat
    the curve looks early. `epochs` keeps its meaning: the floor.
    """
    out = train_spectral_pn(_cfg(
        case, epochs=10, epochs_cap=20, lr_decay_epochs=10,
        settle_window=2, settle_tol=1e9,   # a tolerance nothing can fail
        gamma_log_every=1))
    assert out["epochs_run"] >= 10
    assert out["epochs_to_settle"] >= 10


def test_cap_is_honoured_and_non_settling_is_reported(case):
    """An arm that never settles must stop at the cap and SAY SO. A silent
    truncation reads as 'converged' to every downstream consumer.
    """
    out = train_spectral_pn(_cfg(
        case, epochs=4, epochs_cap=11, lr_decay_epochs=4,
        settle_window=2, settle_tol=0.0,   # unsatisfiable
        gamma_log_every=1))
    assert out["epochs_run"] == 11
    assert out["settled"] is False
    assert out["epochs_to_settle"] is None


def test_settling_stops_early_and_records_the_epoch(case):
    out = train_spectral_pn(_cfg(
        case, epochs=4, epochs_cap=40, lr_decay_epochs=4,
        settle_window=2, settle_tol=1e9,
        gamma_log_every=1))
    assert out["settled"] is True
    assert out["epochs_run"] < 40
    assert out["epochs_run"] == out["epochs_to_settle"]


def test_rule_parameters_travel_with_the_result(case):
    out = train_spectral_pn(_cfg(
        case, epochs=4, epochs_cap=12, lr_decay_epochs=4,
        settle_window=2, settle_tol=1e9, gamma_log_every=1))
    for k in ("settle_window", "settle_tol", "epochs_cap",
              "lr_decay_epochs", "settled", "epochs_to_settle", "epochs_run"):
        assert k in out, f"{k} missing from the result dict"
    assert out["settle_window"] == 2
    assert out["epochs_cap"] == 12
    assert out["lr_decay_epochs"] == 4


def test_settle_requires_the_gamma_logger(case):
    """The rule reads gamma_history. Requesting it without the logger would
    silently never fire -- an arm that runs to the cap and looks like a physics
    result. Refuse instead.
    """
    with pytest.raises(ValueError, match="gamma_log_every"):
        train_spectral_pn(_cfg(
            case, epochs=4, epochs_cap=12, lr_decay_epochs=4,
            settle_window=2, settle_tol=0.005, gamma_log_every=0))


def test_cap_below_epochs_is_refused(case):
    with pytest.raises(ValueError, match="epochs_cap"):
        train_spectral_pn(_cfg(
            case, epochs=10, epochs_cap=5, lr_decay_epochs=10,
            settle_window=2, settle_tol=0.005, gamma_log_every=1))
