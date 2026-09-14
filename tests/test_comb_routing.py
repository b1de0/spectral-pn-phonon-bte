"""pinn_bte.physics.comb_routing : one place that turns a comb name into."""
import numpy as np
import pytest

import pinn_bte.physics.ttg_dispersion as td
from pinn_bte.physics import comb_routing as cr
from pinn_bte.physics.optical_reservoir import joint_modes
from pinn_bte.physics.ttg_dispersion import sum_rule_grid

def test_grid_is_empty_for_joint_and_sum_rule_for_sumrule():
    assert cr.comb_grid("joint", 20) == {}
    assert cr.comb_grid("legacy", 20) == {}
    assert cr.comb_grid("joint-sumrule", 20) == sum_rule_grid(20)
    with pytest.raises(ValueError):
        cr.comb_grid("bare", 20)

def test_modes_match_the_trainers_construction():
    v, tau, C = cr.comb_modes("joint-sumrule", 20)
    v2, tau2, C2 = joint_modes(Nk=20, T_ref=300.0, **sum_rule_grid(20))
    assert np.array_equal(C, C2) and np.array_equal(v, v2) and np.array_equal(tau, tau2)
    assert len(C) == 61
    v, tau, C = cr.comb_modes("joint", 20)
    assert np.array_equal(C, joint_modes(Nk=20, T_ref=300.0)[2])
    v, tau, C = cr.comb_modes("legacy", 20)
    assert len(C) == 60

def test_context_routes_the_phonon_modes_namespace():
    want = float(np.sum(joint_modes(Nk=20, T_ref=300.0, **sum_rule_grid(20))[2]))
    # through the MODULE attribute -- a by-name import reads the bare third comb
    with cr.comb_context("joint-sumrule", 20, 300.0):
        got = float(np.sum(td.phonon_modes(20, 300.0)[2]))
    assert got == pytest.approx(want, rel=1e-12)
    with cr.comb_context("joint", 20, 300.0):
        got_joint = float(np.sum(td.phonon_modes(20, 300.0)[2]))
    assert got_joint != pytest.approx(want, rel=1e-6)
    with cr.comb_context("legacy", 20, 300.0):
        assert len(td.phonon_modes(20, 300.0)[2]) == 60

def test_context_agrees_with_the_trainers_mode_source_ctx():
    from experiments.transient.run_spectral_pn import mode_source_ctx
    with mode_source_ctx("joint-sumrule", 20, 300.0):
        a = np.asarray(td.phonon_modes(20, 300.0)[2])
    with cr.comb_context("joint-sumrule", 20, 300.0):
        b = np.asarray(td.phonon_modes(20, 300.0)[2])
    assert np.array_equal(a, b)

def test_pinned_comb_reads_the_shared_file(tmp_path):
    p = tmp_path / "ladders.yaml"
    p.write_text("ladder_source: x\ncombs: {x: joint-sumrule}\n")
    assert cr.pinned_comb(p) == ("x", "joint-sumrule")
    src, comb = cr.pinned_comb()
    assert comb in cr.COMBS
