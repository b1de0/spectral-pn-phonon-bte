"""Comb name -> grid kwargs / modes / context manager, in ONE place so that no
consumer re-implements the joint vs joint-sumrule distinction."""
from __future__ import annotations

import contextlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LADDERS = REPO / "data" / "ladders.yaml"
COMBS = ("legacy", "joint", "joint-sumrule")

def _check(comb: str) -> None:
    if comb not in COMBS:
        raise ValueError(f"comb={comb!r} not in {COMBS}")

def comb_grid(comb: str, Nk: int) -> dict:
    """Grid kwargs for `joint_modes` / `joint_mode_source`: {} unless sum-rule."""
    _check(comb)
    if comb == "joint-sumrule":
        from pinn_bte.physics.ttg_dispersion import sum_rule_grid
        return sum_rule_grid(Nk)
    return {}

def comb_modes(comb: str, Nk: int, T_ref: float = 300.0):
    """(v [Ang/s], tau [s], C [J m^-3 K^-1]) of the named comb, built exactly as
    the trainer builds it (`run_spectral_pn.mode_source_ctx`)."""
    _check(comb)
    if comb == "legacy":
        import pinn_bte.physics.ttg_dispersion as td
        with td.shipped_mode_source(Nk=Nk, T_ref=T_ref):
            return td.phonon_modes(Nk, T_ref)
    from pinn_bte.physics.optical_reservoir import joint_modes
    return joint_modes(Nk=Nk, T_ref=T_ref, **comb_grid(comb, Nk))

@contextlib.contextmanager
def comb_context(comb: str, Nk: int = 20, T_ref: float = 300.0):
    """Route BOTH `phonon_modes` namespaces (ttg_dispersion + ttg_dom) to the comb."""
    _check(comb)
    if comb == "legacy":
        from pinn_bte.physics.ttg_dispersion import shipped_mode_source
        with shipped_mode_source(Nk=Nk, T_ref=T_ref):
            yield
        return
    from pinn_bte.physics.optical_reservoir import joint_mode_source
    with joint_mode_source(Nk=Nk, T_ref=T_ref, **comb_grid(comb, Nk)):
        yield

def pinned_comb(ladders_path: Path = LADDERS) -> tuple[str, str]:
    """(ladder_source, comb) from the one shared declaration, data/ladders.yaml."""
    raw = yaml.safe_load(Path(ladders_path).read_text())
    src = raw["ladder_source"]
    return src, raw["combs"][src]
