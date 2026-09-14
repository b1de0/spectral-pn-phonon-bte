"""The UNION window rule on the PN path."""
from functools import lru_cache

import numpy as np
import pytest

from pinn_bte.physics.ttg_dispersion import shipped_mode_source
from pinn_bte.physics.ttg_dom import WINDOW_RULE_FULL
from pinn_bte.training.transient.spectral_pn_trainer import (
    CROSSING_WINDOW_MODES,
    UNION_WINDOW_MODES,
    WINDOW_MODES,
    WINDOW_RULES,
    build_si_case,
    select_window,
)

@pytest.fixture(autouse=True)
def _pre_adoption_mode_set():
    with shipped_mode_source():
        yield


#: PN's nine production rungs (um), 1D, Nk=20, n_decay_times=7, thresh 0.01.
PN_RUNGS = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)

#: MEASURED t_1% / (7/gamma_dom) at those periods.
#: DERIVED below into the short/clean partition; never hand-list the partition.
H171_RATIO = {0.01: 3.347, 0.1: 1.476, 0.3: 0.654, 0.5: 0.635, 0.7: 0.635,
              1.0: 0.633, 1.5: 0.633, 10.0: 0.654, 100.0: 0.658}

#: The rungs where the crossing OUTRUNS the exponential window (union bites)...
SHORT_RUNGS = tuple(L for L in PN_RUNGS if H171_RATIO[L] > 1.0)
#: ...and its complement, where the union rule must be a bit-exact no-op.
CLEAN_RUNGS = tuple(L for L in PN_RUNGS if L not in SHORT_RUNGS)

#: The `t_end` stored in the pinned L=100 run's npz. Equality, not proximity.
PINNED_L100_T_END = 2.4055102818217088e-05
LEGACY_RULE = "legacy-earlystop"


@lru_cache(maxsize=None)
def _win(L_um, window_mode):
    """PRODUCTION config (Nk=20, 1D, n_dec=7, thresh 0.01) — never a reduced one:
    a reduced-fidelity gate false-greens.
    """
    return select_window(L_um, Nk=20, n_decay_times=7.0, geometry="1d",
                         window_mode=window_mode)


@lru_cache(maxsize=None)
def _win_legacy(L_um, window_mode):
    """Same, replayed under the pinned runs' own reference window rule."""
    return select_window(L_um, Nk=20, n_decay_times=7.0, geometry="1d",
                         window_mode=window_mode, window_rule=LEGACY_RULE)


def test_h171_partition_is_exactly_two_rungs():
    """The premise of a TWO-ARM fix, asserted rather than assumed."""
    assert SHORT_RUNGS == (0.01, 0.1)
    assert len(CLEAN_RUNGS) == 7


@pytest.mark.slow
@pytest.mark.parametrize("L_um", CLEAN_RUNGS)
def test_union_reproduces_legacy_bit_for_bit_at_clean_rungs(L_um):
    """`==` on float64, not isclose."""
    legacy = _win(L_um, "n_decay")
    union = _win(L_um, "crossing-union")
    assert union.t_end == legacy.t_end, (
        f"L={L_um}: union window {union.t_end!r} != legacy {legacy.t_end!r} "
        f"— the ladder would become MIXED")
    assert union.rule == "n_decay"


@pytest.mark.slow
def test_union_matches_the_pinned_L100_t_end():
    assert _win_legacy(100.0, "crossing-union").t_end == PINNED_L100_T_END
    # ...and the RULED window is the same choice, one truncation removed.
    assert _win(100.0, "crossing-union").rule == "n_decay"
    assert _win(100.0, "crossing-union").t_end == pytest.approx(
        PINNED_L100_T_END, rel=1.0e-4)


@pytest.mark.slow
def test_legacy_default_is_unchanged():
    """The shipped default must stay the pre-union rule so pins reproduce."""
    win = select_window(100.0, Nk=20, n_decay_times=7.0, geometry="1d",
                        window_rule=LEGACY_RULE)
    assert win.mode == "n_decay" and win.rule == "n_decay"
    assert win.t_end == PINNED_L100_T_END
    assert win.t_end == 7.0 / win.gamma_dom_hz
    # The RULE is what this test guards; it must hold under EITHER reference
    # window convention, so assert the identity on the default one too.
    ruled = select_window(100.0, Nk=20, n_decay_times=7.0, geometry="1d")
    assert ruled.mode == "n_decay" and ruled.rule == "n_decay"
    assert ruled.t_end == 7.0 / ruled.gamma_dom_hz


@pytest.mark.slow
@pytest.mark.parametrize("L_um", SHORT_RUNGS)
def test_union_extends_the_two_short_rungs(L_um):
    legacy = _win(L_um, "n_decay")
    union = _win(L_um, "crossing-union")
    assert union.t_end > legacy.t_end
    assert union.rule == "crossing"
    ratio = union.t_end / legacy.t_end
    # O(dt) grid resolution of the crossing search — 1% is generous for a
    # 12k-36k step trace and still refutes a wrong branch.
    assert ratio == pytest.approx(H171_RATIO[L_um], rel=0.01)


@pytest.mark.slow
def test_crossing_mode_is_not_the_union_mode():
    """Bare 'crossing' must NOT take the max — else the two modes are one."""
    bare = _win(1.0, "crossing")
    legacy = _win(1.0, "n_decay")
    assert bare.t_end < legacy.t_end
    assert bare.rule == "crossing"


def test_every_window_mode_is_dispatchable():
    """EXHAUSTIVE ENUMERATION over the vocabulary tuple itself."""
    assert len(WINDOW_MODES) == len(set(WINDOW_MODES))
    for mode in WINDOW_MODES:
        # build_si_case, not select_window: this covers the WHOLE path, so a
        # mode the selector handles but the case builder chokes on fails here.
        case = build_si_case(1.0, Nk=8, n_decay_times=7.0, window_mode=mode)
        assert np.isfinite(case.t_end) and case.t_end > 0.0, mode
        assert case.window_rule in WINDOW_RULES, (mode, case.window_rule)
        assert case.window_mode == mode
        assert np.isfinite(case.t_candidate_ndecay_s)
        assert case.t_ref[-1] > 0.0
        # the reference trace must actually SPAN the chosen window
        assert case.t_ref[-1] == pytest.approx(case.t_end, rel=0.06)


def test_window_mode_sublists_are_derived_not_hand_listed():
    assert CROSSING_WINDOW_MODES == tuple(m for m in WINDOW_MODES
                                          if "crossing" in m)
    assert UNION_WINDOW_MODES == tuple(m for m in CROSSING_WINDOW_MODES
                                       if m.endswith("-union"))
    assert set(UNION_WINDOW_MODES) <= set(CROSSING_WINDOW_MODES) <= set(WINDOW_MODES)
    # A non-crossing mode must not carry a crossing candidate.
    assert np.isnan(_win(1.0, "n_decay").t_candidate_crossing_s)


def test_unknown_window_mode_raises():
    for fn in (select_window, build_si_case):
        with pytest.raises(ValueError, match="window_mode"):
            fn(1.0, Nk=8, window_mode="n-decay")


def test_a_non_finite_crossing_raises_instead_of_skipping_the_union(monkeypatch):
    """MUTATION-CONTROLLED: inject the NaN the guard is written against."""
    import pinn_bte.physics.ttg_dom as dom
    monkeypatch.setattr(dom, "ttg_window_crossing", lambda *a, **k: float("nan"))
    for mode in CROSSING_WINDOW_MODES:
        with pytest.raises(ValueError, match="non-finite"):
            select_window(1.0, Nk=8, n_decay_times=7.0, geometry="1d",
                          window_mode=mode)


def test_a_non_finite_gamma_raises_under_every_mode(monkeypatch):
    """The other way a NaN can reach `t_end`: through gamma_dom. 'n_decay' never
    touches the crossing search, so the crossing guard alone would let this one
    through.
    """
    import pinn_bte.physics.ttg_dom as dom
    monkeypatch.setattr(dom, "ttg_decay_spectral", lambda *a, **k: float("nan"))
    for mode in WINDOW_MODES:
        with pytest.raises(ValueError, match="non-finite"):
            select_window(1.0, Nk=8, n_decay_times=7.0, geometry="1d",
                          window_mode=mode)


def test_the_nan_crossing_candidate_under_n_decay_stays_a_sentinel(monkeypatch):
    """THE BENIGN CASE MUST STAY BENIGN."""
    import pinn_bte.physics.ttg_dom as dom
    calls = []

    def _forbidden(*a, **k):
        calls.append((a, k))
        raise AssertionError("crossing search called under window_mode='n_decay'")

    monkeypatch.setattr(dom, "ttg_window_crossing", _forbidden)
    w = select_window(0.01, Nk=20, n_decay_times=7.0, geometry="1d",
                      window_mode="n_decay", window_rule=LEGACY_RULE)
    assert calls == []
    assert np.isnan(w.t_candidate_crossing_s)
    assert w.rule == "n_decay"
    assert w.t_end == 1.4612324651509903e-11
    # Same guard on the RULED reference window: still zero crossing calls.
    w2 = select_window(0.01, Nk=20, n_decay_times=7.0, geometry="1d",
                       window_mode="n_decay")
    assert calls == []
    assert np.isnan(w2.t_candidate_crossing_s) and w2.rule == "n_decay"
    assert w2.t_end == 1.4616373349110644e-11


def test_provenance_uses_the_lobe_vocabulary():
    case = build_si_case(0.01, Nk=8, n_decay_times=7.0, geometry="1d",
                         window_mode="crossing-union")
    prov = case.window_provenance()
    assert set(prov) == {"window_mode", "Lt_rule", "Lt_candidate_crossing_tau",
                         "Lt_candidate_ndecay_tau", "t_candidate_crossing_s",
                         "t_candidate_ndecay_s", "window_crossing_thresh",
                         "arbiter_window_rule"}
    assert prov["Lt_rule"] in WINDOW_RULES
    assert prov["arbiter_window_rule"] == WINDOW_RULE_FULL
    assert prov["Lt_candidate_crossing_tau"] == pytest.approx(
        case.t_candidate_crossing_s / case.tau_ref)
    assert prov["Lt_candidate_ndecay_tau"] == pytest.approx(
        case.t_candidate_ndecay_s / case.tau_ref)


def test_provenance_survives_an_npz_round_trip(tmp_path):
    """The runner ships this dict into the npz via `extra_meta`."""
    case = build_si_case(1.0, Nk=8, n_decay_times=7.0, geometry="1d")
    prov = case.window_provenance()
    np.savez(tmp_path / "p.npz", **prov)
    back = np.load(tmp_path / "p.npz", allow_pickle=True)
    assert set(back.files) == set(prov)
    assert str(back["window_mode"]) == "n_decay"
    assert str(back["Lt_rule"]) == "n_decay"
    assert float(back["Lt_candidate_ndecay_tau"]) == pytest.approx(
        case.t_end / case.tau_ref)
    assert np.isnan(float(back["Lt_candidate_crossing_tau"]))
