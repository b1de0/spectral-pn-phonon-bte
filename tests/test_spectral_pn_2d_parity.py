"""1D/2D parity of the ugks composition: case-level identity and bitwise
geometry independence of the coefficients.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import build_si_case


def test_ugks_case_level_1d_2d_identity():
    c1 = build_si_case(1.0, Nk=10, geometry="1d")
    c2 = build_si_case(1.0, Nk=10, geometry="2d")
    # mode set and wavenumber: bitwise — geometry cannot touch them
    assert np.array_equal(c1.v, c2.v)
    assert np.array_equal(c1.tau, c2.tau)
    assert np.array_equal(c1.C, c2.C)
    assert c1.q == c2.q
    # reference marginal identity (the banded-era bars, unchanged)
    assert np.isclose(c2.gamma_dom_hz, c1.gamma_dom_hz, rtol=1e-4)
    assert np.abs(np.interp(c1.t_ref, c2.t_ref, c2.A_ref)
                  - c1.A_ref).max() < 1e-5


def _gamma_kin(v, tau, C, q):
    return float(q ** 2 * np.sum(C * v ** 2 * tau) / 3.0 / np.sum(C))


def _seeded_ugks_net(v, tau, C, q, seed=42):
    from pinn_bte.models.spectral_pn import XiHybridPNCoefficientNet
    torch.manual_seed(seed)
    return XiHybridPNCoefficientNet(
        n_modes=len(v), L_max=4, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v, tau=tau, q=q, width=16, depth=2, blend="ugks").double()


def test_ugks_coefficients_geometry_independent_bitwise():
    c1 = build_si_case(1.0, Nk=10, geometry="1d")
    c2 = build_si_case(1.0, Nk=10, geometry="2d")
    t = torch.linspace(0.0, 0.8 * min(c1.t_end, c2.t_end), 33,
                       dtype=torch.float64).reshape(-1, 1)

    def _tensors(c):
        return (torch.tensor(c.v, dtype=torch.float64),
                torch.tensor(c.tau, dtype=torch.float64),
                torch.tensor(c.C, dtype=torch.float64))

    v1, tau1, C1 = _tensors(c1)
    v2, tau2, C2 = _tensors(c2)
    net1 = _seeded_ugks_net(c1.v, c1.tau, c1.C, c1.q)
    net2 = _seeded_ugks_net(c2.v, c2.tau, c2.C, c2.q)
    with torch.no_grad():
        out1 = net1.coefficients(t, v1, tau1, C1, c1.q)
        out2 = net2.coefficients(t, v2, tau2, C2, c2.q)
    for a, b in zip(out1, out2):
        assert torch.equal(a, b), "ugks coefficients must be geometry-independent"

    # FIRING MUTATION: the equality must be falsifiable — perturb tau
    net3 = _seeded_ugks_net(c2.v, c2.tau * 1.01, c2.C, c2.q)
    with torch.no_grad():
        out3 = net3.coefficients(t, v2, tau2 * 1.01, C2, c2.q)
    assert not all(torch.equal(a, b) for a, b in zip(out1, out3)),\
        "mutation did not fire — the bitwise test is vacuous"


def test_cli_smoke_ugks_2d(tmp_path):
    import importlib
    mod = importlib.import_module("experiments.transient.run_spectral_pn")
    run_dir = mod.main(["--L", "1.0", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float64",
                        "--geometry", "2d",
                        "--formulation", "xihybrid", "--blend", "ugks",
                        "--outdir", str(tmp_path), "--log-every", "1"])
    d = np.load(next(Path(run_dir).glob("*_results.npz")), allow_pickle=True)
    assert str(d["geometry"]) == "2d"
    assert str(d["blend"]) == "ugks"
