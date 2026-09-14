"""The 2q/Hq arbiter must not be half on one comb and half on another."""

import numpy as np
import pytest

from pinn_bte.physics.optical_reservoir import joint_mode_source
from pinn_bte.physics.ttg_pnhq import solve_pnHq_decay

KW = dict(H=2, A0_K=75.0, Nk=8, L_max=4, T_ref=300.0, n_decay=1.0,
          n_phi=8, ic_2c=-1.0, ic_2s=0.0, T_lo=140.0, T_hi=460.0, nT=8)


def _rate(**over):
    kw = dict(KW); kw.update(over)
    return float(np.atleast_1d(solve_pnHq_decay(100.0, **kw))[0])


def test_runs_under_the_joint_mode_source():
    """The regression: this raised ValueError before the fix."""
    with joint_mode_source(Nk=KW["Nk"], T_ref=KW["T_ref"]):
        g = _rate(tau_model="frozen")
    assert np.isfinite(g) and g > 0.0


def test_the_comb_actually_changes_the_answer():
    """A fix that made it run by quietly staying bare would pass the test above
    and be worthless. The joint comb carries a 43% optical reservoir; the rate
    MUST move.
    """
    bare = _rate(tau_model="frozen")
    with joint_mode_source(Nk=KW["Nk"], T_ref=KW["T_ref"]):
        joint = _rate(tau_model="frozen")
    assert abs(joint / bare - 1.0) > 0.01, (bare, joint)


def test_local_and_frozen_both_run_on_joint():
    with joint_mode_source(Nk=KW["Nk"], T_ref=KW["T_ref"]):
        gl = _rate(tau_model="local")
        gf = _rate(tau_model="frozen")
    assert np.isfinite(gl) and np.isfinite(gf)
    assert gl > 0.0 and gf > 0.0
