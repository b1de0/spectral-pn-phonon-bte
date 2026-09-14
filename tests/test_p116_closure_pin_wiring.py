"""Wiring of the closure-pin shim flag: config default, CLI vocabulary, path
guards and npz provenance.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    train_spectral_pn,
)

REPO = Path(__file__).resolve().parents[1]


def _runner():
    """Import experiments/transient/run_spectral_pn.py by path (the runner is a
    script, not an installed module).
    """
    path = REPO / "experiments" / "transient" / "run_spectral_pn.py"
    spec = importlib.util.spec_from_file_location("p116_run_spectral_pn", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The minimal command line that reaches the shim's path. Kept in one place so
#: the guard tests below negate exactly one field at a time.
_UGKS_CLI = ["--L", "1.0", "--formulation", "xihybrid", "--blend", "ugks"]


def _tiny(**overrides):
    """A 1-mode problem the trainer accepts. Nothing here is physics -- these
    tests are about plumbing, and they must not need a DOM build.
    """
    cfg = dict(v=np.array([1.0]), tau=np.array([1.0]), C=np.array([1.0]),
               q=0.6, t_end=2.0, L_max=2, width=8, depth=1,
               epochs=2, nt=12, seed=0, device="cpu", dtype="float64")
    cfg.update(overrides)
    return SpectralPNConfig(**cfg)


def test_config_field_defaults_to_false_and_is_a_bool():
    cfg = _tiny()
    assert cfg.closure_pin is False
    assert isinstance(cfg.closure_pin, bool)


def test_cli_default_is_off_and_maps_to_false():
    args = _runner().parse_args(["--L", "1.0"])
    assert args.closure_pin == "off"


@pytest.mark.parametrize("value,expected", [("off", False), ("on", True)])
def test_cli_value_maps_to_the_bool(value, expected):
    args = _runner().parse_args(_UGKS_CLI + ["--closure-pin", value])
    assert args.closure_pin == value
    assert (args.closure_pin == "on") is expected


def test_cli_vocabulary_is_closed():
    """`true`/`1`/`yes` must not quietly become anything."""
    with pytest.raises(SystemExit):
        _runner().parse_args(_UGKS_CLI + ["--closure-pin", "true"])


@pytest.mark.parametrize("cli", [
    ["--L", "1.0"],                                        # plain / banded
    ["--L", "1.0", "--formulation", "ce"],
    ["--L", "1.0", "--formulation", "xihybrid"],           # banded blend
])
def test_cli_refuses_the_flag_off_its_own_path(cli):
    with pytest.raises(SystemExit):
        _runner().parse_args(cli + ["--closure-pin", "on"])


@pytest.mark.parametrize("kw", [
    dict(formulation="plain"),
    dict(formulation="ce"),
    dict(formulation="xihybrid", blend="banded"),
])
def test_trainer_refuses_the_flag_off_its_own_path(kw):
    with pytest.raises(ValueError, match="closure_pin"):
        train_spectral_pn(_tiny(closure_pin=True, **kw))


def test_the_guard_is_not_vacuous_with_the_flag_off():
    """Same configs, flag OFF -> no guard fires (a guard that refused everything
    would pass the tests above while breaking every shipped run).
    """
    train_spectral_pn(_tiny(formulation="plain"))
    train_spectral_pn(_tiny(formulation="xihybrid", blend="banded"))


@pytest.mark.parametrize("flag", [False, True])
def test_npz_records_closure_pin(tmp_path, flag):
    cfg = _tiny(formulation="xihybrid", blend="ugks",
                closure_pin=flag, outdir=tmp_path, run_id="p116w")
    train_spectral_pn(cfg)
    d = np.load(tmp_path / "p116w_results.npz", allow_pickle=False)
    assert "closure_pin" in d.files, \
        ("the npz lost its closure-pin discriminator: a shimmed arm and its control "
         "differ in NO other config field")
    assert bool(d["closure_pin"]) is flag


def test_flag_actually_changes_the_solve(tmp_path):
    """NON-VACUITY of the provenance key: if `on` and `off` produced the same
    artifact, recording the flag would be recording nothing.
    """
    out = {}
    for flag in (False, True):
        d = tmp_path / f"pin_{flag}"
        train_spectral_pn(_tiny(formulation="xihybrid", blend="ugks",
                                closure_pin=flag,
                                epochs=3, outdir=d, run_id="p116n"))
        out[flag] = np.load(d / "p116n_results.npz")["A"]
    assert not np.array_equal(out[False], out[True])
