"""scripts/score_run.py dispatches the field-run DOM denominator by the npz's
own mode_source field, never by a name default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import score_run  # noqa: E402

MLP = REPO / "data" / "runs" / "mlp_control" / "1d_nongray_large_dt_1.0um_100K"

PIRATE_TS = "20260704_134526"
PIRATE_GR_OWN = 0.541623
PIRATE_GR_JOINT_MISSCORE = 0.4719


class _FakeNpz:
    """The minimal npz surface `score_run` touches: `.files` + indexing."""

    def __init__(self, **fields):
        self._fields = dict(fields)

    @property
    def files(self):
        return list(self._fields)

    def __getitem__(self, key):
        return self._fields[key]


class TestOwnModeSourceIsReadFromTheArtifact:

    def test_a_declared_field_is_used_verbatim(self):
        d = _FakeNpz(mode_source="joint")
        assert score_run.own_mode_source(d) == "joint"

    def test_an_absent_field_is_a_pre_adoption_shipped_declaration(self):
        d = _FakeNpz(T=np.zeros((2, 2)))
        assert score_run.own_mode_source(d) == "shipped"

    def test_no_flag_resolves_to_the_runs_own_kind(self):
        assert score_run.resolve_mode_source(
            _FakeNpz(mode_source="joint"), None) == "joint"
        assert score_run.resolve_mode_source(
            _FakeNpz(T=np.zeros((2, 2))), None) == "shipped"

    def test_a_flag_contradicting_a_declared_field_is_refused_not_mixed(self):
        """A ratio across two mode sets reproduces no published number."""
        with pytest.raises(SystemExit, match="contradicts"):
            score_run.resolve_mode_source(
                _FakeNpz(mode_source="joint"), "shipped")

    def test_a_cross_comb_request_on_a_legacy_run_is_labelled_diagnostic(
            self, capsys):
        got = score_run.resolve_mode_source(_FakeNpz(T=np.zeros((2, 2))),
                                            "joint")
        assert got == "joint"
        assert "DIAGNOSTIC" in capsys.readouterr().out

    def test_the_cli_default_is_auto_never_a_name(self):
        ap = score_run.build_parser()
        assert ap.parse_args([str(MLP / PIRATE_TS)]).mode_source is None


class TestPirateNetArmScoresOnItsOwnReference:

    def test_auto_dispatch_reproduces_the_own_reference_score(self):
        """End to end on the archived artifact: no flag, shipped denominator,
        0.5416 -- not the 0.4719 the joint name default produced.
        """
        _path, d = score_run._load(MLP / PIRATE_TS)
        ms = score_run.resolve_mode_source(d, None)
        assert ms == "shipped"
        res = score_run.score_field_run(d, ms)
        assert res["gamma_ratio"] == pytest.approx(PIRATE_GR_OWN, abs=1e-6)
        assert res["gamma_ratio"] != pytest.approx(PIRATE_GR_JOINT_MISSCORE,
                                                   abs=1e-3)
        assert res["mode_source"] == "shipped"
