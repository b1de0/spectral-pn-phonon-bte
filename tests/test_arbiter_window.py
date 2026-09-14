"""The spectral DOM reference must integrate the window it declares."""
from __future__ import annotations

import numpy as np
import pytest

from pinn_bte.physics.ttg_dom import (
    STEPS_PER_DECAY_DEFAULT,
    WINDOW_RULE_FULL,
    WINDOW_RULE_LEGACY,
    ttg_amplitude_curve,
    ttg_decay_spectral,
)


@pytest.fixture(autouse=True)
def _pre_adoption_comb():
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    with shipped_mode_source():
        yield


PINNED_GAMMA_DOM_1D = 479047664690.1412
PINNED_GAMMA_DOM_2D = 473686241012.6454
#: tests/test_p077_retrain_flags.py / test_p077_2d_wiring.py fixture rungs.
LEGACY_GAMMA_L01_NK3_1D = 13713527553.840748
LEGACY_GAMMA_L01_NK3_2D = 13713498139.326948


def _diag(L_um, geometry, **kw):
    g, d = ttg_decay_spectral(L_um, geometry, return_diag=True, **kw)
    return g, d


class TestSameWindowAcrossGeometries:
    """The '1d' and '2d' arms are the same physics; they must share a window."""

    @pytest.mark.parametrize(
        "L_um, Nk",
        [(0.01, 20),   # the PN headline ballistic rung -- production Nk
         (0.1, 3),     # the test-fixture rung: two sign changes inside the window
         (1.0, 20)],
    )
    def test_realised_window_is_bit_identical(self, L_um, Nk):
        _, d1 = _diag(L_um, "1d", Nk=Nk)
        _, d2 = _diag(L_um, "2d", Nk=Nk)
        assert d1["t_realized_s"] == d2["t_realized_s"], (
            f"L={L_um} Nk={Nk}: 1d and 2d arbiters score on different windows "
            f"({d1['t_realized_s']:.6e} vs {d2['t_realized_s']:.6e}, "
            f"{100 * (d2['t_realized_s'] / d1['t_realized_s'] - 1):+.2f}%)")
        assert d1["n_steps_used"] == d2["n_steps_used"]
        assert d1["dt_s"] == d2["dt_s"]

    def test_legacy_rule_is_where_the_geometries_diverged(self):
        """Mutation control: the defect is real and this rung exhibits it."""
        _, d1 = _diag(0.01, "1d", Nk=20, window_rule=WINDOW_RULE_LEGACY)
        _, d2 = _diag(0.01, "2d", Nk=20, window_rule=WINDOW_RULE_LEGACY)
        assert d1["n_steps_used"] == 11603 and d2["n_steps_used"] == 12000
        rel = d2["t_realized_s"] / d1["t_realized_s"] - 1
        assert 0.033 < rel < 0.035, f"expected the measured 3.4% split, got {rel:.5f}"


class TestRealisedWindowEqualsDeclaredWindow:
    @pytest.mark.parametrize(
        "L_um, geometry, Nk",
        [(0.01, "1d", 20), (0.01, "2d", 20), (0.1, "2d", 10),
         (0.1, "1d", 3), (1.0, "1d", 20), (100.0, "1d", 20)])
    def test_full_rule_runs_the_declared_window(self, L_um, geometry, Nk):
        _, d = _diag(L_um, geometry, Nk=Nk)
        assert d["window_rule"] == WINDOW_RULE_FULL
        assert d["n_steps_used"] == d["nsteps"]
        assert d["t_realized_s"] == d["t_declared_s"]
        assert d["t_declared_s"] == pytest.approx(
            d["n_decay"] / d["gamma_dominant_hz"], rel=1e-12)


class TestTheTailTheOldStopDropped:
    """Quantified: WHAT the truncation removed, and which way it biased gamma."""

    LIVE_TRACE_RUNGS = [
        (0.01, "1d", 20, 11603, 18, 19.5),
        (0.1, "1d", 3, 6359, 1, 414.1),
        (0.1, "2d", 10, 11077, 4, 127.7),
        (0.01, "1d", 10, 10218, 3, 303.1),
    ]

    @pytest.mark.parametrize("L_um, geometry, Nk, step, to_cross, over",
                             LIVE_TRACE_RUNGS)
    def test_stop_fired_just_before_a_zero_crossing_on_a_live_trace(
            self, L_um, geometry, Nk, step, to_cross, over):
        _, d = _diag(L_um, geometry, Nk=Nk, window_rule=WINDOW_RULE_LEGACY)
        assert d["n_steps_used"] == step
        _, full = _diag(L_um, geometry, Nk=Nk)
        A, sgn = full["A"], np.sign(full["A_signed"])
        crossings = np.nonzero(sgn[1:] != sgn[:-1])[0]
        assert crossings.size, "this rung is supposed to change sign"
        nxt = int(min(c - step for c in crossings if c >= step))
        assert nxt == to_cross, (
            f"stop fired {nxt} steps before the sign change, expected {to_cross}")
        tail_peak = float(np.max(A[step + 1:])) / 1e-4
        assert tail_peak == pytest.approx(over, abs=0.15), (
            f"post-stop |A| peaks at {tail_peak:.1f}x the 1e-4 stop threshold")

    @pytest.mark.parametrize("L_um, Nk", [(1.0, 20), (10.0, 20), (100.0, 20)])
    def test_on_sign_definite_traces_the_tail_is_a_small_exponential_remainder(
            self, L_um, Nk):
        g_legacy, dl = _diag(L_um, "1d", Nk=Nk, window_rule=WINDOW_RULE_LEGACY)
        g_full, df = _diag(L_um, "1d", Nk=Nk)
        assert np.all(df["A_signed"] > 0), "rung is supposed to be sign-definite"
        assert dl["n_steps_used"] / dl["nsteps"] > 0.75
        rel = g_legacy / g_full - 1.0
        assert 8.0e-5 < rel < 9.5e-5, (
            f"L={L_um}: dropped-tail bias {rel:.3e} (denominator: the "
            f"full-window gamma) outside the measured 0.0089-0.0094% band")

    @pytest.mark.parametrize(
        "L_um, geometry, Nk",
        [(0.01, "1d", 20), (0.1, "1d", 3), (0.1, "2d", 10), (0.01, "1d", 10),
         (1.0, "1d", 20), (100.0, "1d", 20)])
    def test_truncation_always_biased_gamma_high(self, L_um, geometry, Nk):
        g_legacy = ttg_decay_spectral(L_um, geometry, Nk=Nk,
                                      window_rule=WINDOW_RULE_LEGACY)
        g_full = ttg_decay_spectral(L_um, geometry, Nk=Nk)
        assert g_legacy >= g_full


class TestLegacyReplayIsBitExact:
    """Every pre-fix pinned artefact stays replayable, exactly."""

    @pytest.mark.parametrize("L_um, geometry, Nk, expected", [
        (0.01, "1d", 20, PINNED_GAMMA_DOM_1D),
        (0.01, "2d", 20, PINNED_GAMMA_DOM_2D),
        (0.1, "1d", 3, LEGACY_GAMMA_L01_NK3_1D),
        (0.1, "2d", 3, LEGACY_GAMMA_L01_NK3_2D),
    ])
    def test_legacy_rule_reproduces_the_shipped_value(self, L_um, geometry, Nk,
                                                      expected):
        assert float(ttg_decay_spectral(
            L_um, geometry, Nk=Nk, window_rule=WINDOW_RULE_LEGACY)) == expected

    def test_default_is_the_full_window_not_the_legacy_one(self):
        """Mutation control: the two rules must NOT be silently the same."""
        assert (float(ttg_decay_spectral(0.1, "1d", Nk=3))
                != LEGACY_GAMMA_L01_NK3_1D)

    def test_unknown_window_rule_raises(self):
        with pytest.raises(ValueError, match="window_rule"):
            ttg_decay_spectral(1.0, "1d", Nk=3, window_rule="nope")


class TestStepsPerDecayIsOptInAndByteIdenticalOff:
    """dt must be pinnable independently of the window length."""

    @pytest.mark.parametrize("L_um, geometry, Nk",
                             [(0.1, "1d", 3), (1.0, "1d", 20)])
    def test_off_is_byte_identical(self, L_um, geometry, Nk):
        a = float(ttg_decay_spectral(L_um, geometry, Nk=Nk))
        b = float(ttg_decay_spectral(L_um, geometry, Nk=Nk, steps_per_decay=None))
        assert a == b

    def test_default_config_is_the_documented_step_density(self):
        _, d0 = _diag(1.0, "1d", Nk=3)
        _, d1 = _diag(1.0, "1d", Nk=3,
                      steps_per_decay=STEPS_PER_DECAY_DEFAULT)
        assert d0["nsteps"] == d1["nsteps"] == 12000
        assert d0["dt_s"] == d1["dt_s"]

    def test_on_pins_dt_across_window_lengths(self):
        spd = STEPS_PER_DECAY_DEFAULT
        _, a = _diag(1.0, "1d", Nk=3, n_decay=12, steps_per_decay=spd)
        _, b = _diag(1.0, "1d", Nk=3, n_decay=24, steps_per_decay=spd)
        assert a["dt_s"] == b["dt_s"]
        assert b["nsteps"] == 24000

    def test_off_the_window_silently_coarsens_dt(self):
        """Mutation control: this is the hazard the knob exists to remove."""
        _, a = _diag(1.0, "1d", Nk=3, n_decay=12)
        _, b = _diag(1.0, "1d", Nk=3, n_decay=24)
        assert b["dt_s"] == pytest.approx(2.0 * a["dt_s"], rel=1e-12)


class TestAmplitudeCurveStepsPerDecay:
    def test_off_is_byte_identical(self):
        t0, a0 = ttg_amplitude_curve(0.1, "1d", Nk=3)
        t1, a1 = ttg_amplitude_curve(0.1, "1d", Nk=3, steps_per_decay=None)
        assert np.array_equal(t0, t1) and np.array_equal(a0, a1)

    def test_on_pins_dt_across_t_end(self):
        spd = STEPS_PER_DECAY_DEFAULT
        t_short, _ = ttg_amplitude_curve(0.1, "1d", Nk=3, n_decay=7,
                                         steps_per_decay=spd)
        t_long, _ = ttg_amplitude_curve(0.1, "1d", Nk=3, n_decay=21,
                                        steps_per_decay=spd)
        dt_s = t_short[1] - t_short[0]
        dt_l = t_long[1] - t_long[0]
        assert dt_l == pytest.approx(dt_s, rel=1e-12)
        assert len(t_long) == 3 * (len(t_short) - 1) + 1
