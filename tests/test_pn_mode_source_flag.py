"""`run_spectral_pn.py --mode-source` — the adopted mode set switch."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.transient.run_spectral_pn import (
    MODE_SOURCE_DEFAULT, MODE_SOURCES, mode_source_ctx, parse_args)
from pinn_bte.training.transient.spectral_pn_trainer import build_si_case


def _case(mode_source: str, L: float = 1.0):
    with mode_source_ctx(mode_source, Nk=20):
        return build_si_case(L, Nk=20, n_decay_times=7.0, geometry="1d")


def test_vocabulary_is_declared():
    assert MODE_SOURCES == ("shipped", "joint", "joint-sumrule")


def test_joint_is_the_default_and_the_cli_agrees():
    assert MODE_SOURCE_DEFAULT == "joint"
    assert parse_args(["--L", "1.0"]).mode_source == "joint"


def test_shipped_reproduces_the_legacy_comb_bit_for_bit():
    """The pre-adoption weight must stay exactly reachable, or every number
    published under it becomes unfalsifiable.
    """
    from pinn_bte.physics.ttg_dispersion import phonon_modes

    got = _case("shipped")
    legacy = phonon_modes(Nk=20, T_ref=300.0, measure=False)
    for a, b in zip((got.v, got.tau, got.C), legacy):
        assert np.array_equal(a, b)


def test_joint_changes_the_comb_and_the_arbiter_target():
    ref, got = _case("shipped"), _case("joint")
    assert len(ref.v) == 60 and len(got.v) == 61      # + the optical reservoir
    # the DOM target must MOVE -- otherwise the switch cannot discriminate
    assert abs(got.gamma_dom_hz / ref.gamma_dom_hz - 1.0) > 0.05


def test_the_module_default_is_the_intermediate_no_run_should_use():
    bare = build_si_case(1.0, Nk=20, n_decay_times=7.0, geometry="1d")
    assert len(bare.v) == 60                       # acoustic only
    assert not np.array_equal(bare.C, _case("shipped").C)


def test_context_restores_the_namespaces_byte_identical():
    import pinn_bte.physics.ttg_dispersion as disp
    import pinn_bte.physics.ttg_dom as dom

    saved = (disp.phonon_modes, dom.phonon_modes)
    before = disp.phonon_modes(20, 300.0)
    with mode_source_ctx("joint", Nk=20):
        assert disp.phonon_modes is not saved[0]
    assert disp.phonon_modes is saved[0] and dom.phonon_modes is saved[1]
    for x, y in zip(before, disp.phonon_modes(20, 300.0)):
        assert np.array_equal(x, y)


def test_unknown_mode_source_is_refused_not_guessed():
    with pytest.raises(ValueError, match="mode_source"):
        with mode_source_ctx("fullbz", Nk=20):
            pass


def test_flag_is_recorded_in_the_run_artifact(tmp_path, monkeypatch):
    """--mode-source must land in extra_meta, not just in the shell history."""
    import experiments.transient.run_spectral_pn as runner

    captured = {}

    class _Stop(Exception):
        pass

    def fake_train(cfg):  # capture the config, skip the 20k-epoch solve
        captured["meta"] = dict(cfg.extra_meta)
        captured["n_modes"] = len(cfg.v)
        raise _Stop

    monkeypatch.setattr(runner, "train_spectral_pn", fake_train)
    for ms in MODE_SOURCES:
        with pytest.raises(_Stop):
            runner.main(["--L", "1.0", "--epochs", "1", "--device", "cpu",
                         "--outdir", str(tmp_path / ms), "--mode-source", ms])
        assert captured["meta"]["mode_source"] == ms
        assert captured["n_modes"] == (61 if ms in ("joint", "joint-sumrule") else 60)
