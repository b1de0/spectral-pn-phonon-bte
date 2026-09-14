"""Silicon's optical-branch heat capacity, as a slow reservoir for the TTG comb."""
from __future__ import annotations

import contextlib

import numpy as np

from pinn_bte.config.physics import LATTICE_CONSTANT
from pinn_bte.physics.ttg_dispersion import phonon_modes

KB_SI = 1.380649e-23      # J/K
HBAR_SI = 1.0545718e-34   # J s
ANG_PER_M = 1e10

# rho c_p of silicon at 300 K: 2329 kg/m^3 x 700 J/(kg K).
SI_RHO_CP_300K = 1.63e6   # J/(m^3 K)

# Diamond structure: 4 primitive cells (2 atoms each) per conventional cube a^3.
PRIMITIVE_CELLS_PER_M3 = 4.0 / (LATTICE_CONSTANT * 1e-10) ** 3
N_OPTICAL_BRANCHES = 3

OMEGA_OPT_THZ_DEFAULT = 14.0     # BZ-averaged, Nilsson & Nelin PRB 6 3777 (1972)
V_OPT_MS_DEFAULT = 1200.0        # branch-averaged |dw/dk| on Gamma->X, same ref
TAU_OPT_S_DEFAULT = 3.7e-12      # Raman FWHM 1.45 cm^-1, Menendez & Cardona 1984

def einstein_capacity(omega_rad_s: float, T: float) -> float:
    """Heat capacity of ONE harmonic mode, J/K: kB x^2 e^x / (e^x - 1)^2."""
    x = HBAR_SI * omega_rad_s / (KB_SI * T)
    if x < 1e-6:
        return KB_SI
    e = np.exp(x)
    return float(KB_SI * x ** 2 * e / (e - 1.0) ** 2)

#: Zone-centre frequency the ANHARMONIC Bose factor is evaluated at.  The
#: Raman line Menendez & Cardona measure is the Gamma-point mode (520.5 cm^-1
#: = 15.6 THz), NOT the 14.0 THz zone average the lumped mode carries -- the
#: measurement fixes the decay channel, the average fixes the mode's energy.
#: Declared rather than silently reusing OMEGA_OPT_THZ_DEFAULT, because the two
#: are different physical quantities that happen to be close.
OMEGA_RAMAN_THZ = 15.6

#: Anchor temperature: tau_opt_of_T(TAU_OPT_ANCHOR_K) == TAU_OPT_S_DEFAULT
#: EXACTLY, so every number computed at 300 K is unchanged by this function.
TAU_OPT_ANCHOR_K = 300.0

def _klemens_factor(T: float, omega_THz: float = OMEGA_RAMAN_THZ) -> float:
    """1 + 2/(exp(hbar w0 / 2 kB T) - 1) -- three-phonon decay into two equal
    halves (Klemens).  This is the "anharmonic effects" of the title of
    Menendez & Cardona, PRB 29, 2051 (1984), the source already cited for the
    300 K linewidth."""
    HBAR_J_S = 1.054571817e-34
    KB_J_K = 1.380649e-23
    x = HBAR_J_S * (2.0 * np.pi * omega_THz * 1e12) / (2.0 * KB_J_K * float(T))
    return 1.0 + 2.0 / np.expm1(x)

def tau_opt_of_T(T: float) -> float:
    """Optical-reservoir lifetime at temperature ``T`` (seconds)."""
    return TAU_OPT_S_DEFAULT * (_klemens_factor(TAU_OPT_ANCHOR_K)
                                / _klemens_factor(float(T)))

def optical_reservoir_modes(T_ref: float = 300.0,
                            omega_opt_THz: float = OMEGA_OPT_THZ_DEFAULT,
                            v_opt_ms: float = V_OPT_MS_DEFAULT,
                            tau_opt_s: float | None = None):
    """One lumped optical mode in the ``phonon_modes`` interface.

    Returns ``(v [Angstrom/s], tau [s], C [J/(m^3 K) per bin])``, length 1.
    """
    omega = 2.0 * np.pi * omega_opt_THz * 1e12
    C = (N_OPTICAL_BRANCHES * PRIMITIVE_CELLS_PER_M3
         * einstein_capacity(omega, T_ref))
    # means "use the anharmonic law"; an explicit value is honoured so the
    # sensitivity study and any provenance re-run can still hold it fixed.
    tau = tau_opt_of_T(T_ref) if tau_opt_s is None else float(tau_opt_s)
    return (np.array([v_opt_ms * ANG_PER_M]),
            np.array([tau]),
            np.array([C]))

def joint_modes(Nk: int = 20, T_ref: float = 300.0,
                omega_opt_THz: float = OMEGA_OPT_THZ_DEFAULT,
                v_opt_ms: float = V_OPT_MS_DEFAULT,
                tau_opt_s: float | None = None, **grid_kwargs):
    """Measure-corrected acoustic comb + optical reservoir, concatenated."""
    # is written out even though it is now also the bare default, because this
    # is the one call whose meaning must NOT follow a future default flip --
    # and because it is why `optical_reservoir` is deliberately absent from
    # `ttg_dispersion._MODE_SOURCE_NAMESPACES` (patching it would make the
    # joint source return the legacy weight inside a legacy context).
    v_ac, tau_ac, C_ac = phonon_modes(Nk=Nk, T_ref=T_ref, measure=True,
                                      **grid_kwargs)
    v_op, tau_op, C_op = optical_reservoir_modes(
        T_ref, omega_opt_THz, v_opt_ms, tau_opt_s)
    return (np.concatenate([v_ac, v_op]),
            np.concatenate([tau_ac, tau_op]),
            np.concatenate([C_ac, C_op]))

@contextlib.contextmanager
def joint_mode_source(Nk: int = 20, T_ref: float = 300.0, **kwargs):
    """Route BOTH ``phonon_modes`` namespaces to the joint comb, then restore."""
    import importlib

    import pinn_bte.physics.ttg_dispersion as _disp
    from pinn_bte.physics.ttg_dispersion import _MODE_SOURCE_NAMESPACES

    # Every namespace that binds `phonon_modes` by name is routed together:
    # patching a subset leaves two solvers on two combs.
    _EXCLUDED = ()
    _targets = [_disp] + [importlib.import_module(m)
                          for m in _MODE_SOURCE_NAMESPACES if m not in _EXCLUDED]

    modes = joint_modes(Nk=Nk, T_ref=T_ref, **kwargs)
    saved = [(m, m.phonon_modes) for m in _targets]

    # Memoised per (Nk, T).  A full 160-temperature `_tables` sweep costs
    # 4.9 ms of rebuilds, measured -- honouring the argument is free.
    _cache = {float(T_ref): modes}

    def _injected(_nk_arg=None, T_arg=None, *_a, **_k):
        # whole point.  Nk is a DISCRETISATION choice the context makes; a
        # consumer that re-calls with its own Nk must not silently switch
        # combs mid-computation -- that is what
        # `test_injected_source_ignores_caller_Nk_drift` protects, and it is
        # right.  T is a PHYSICAL argument that `ttg_dom._tables` sweeps ON
        # without weakening the protection that test encodes.
        T = float(T_ref) if T_arg is None else float(T_arg)
        if T not in _cache:
            _cache[T] = joint_modes(Nk=Nk, T_ref=T, **kwargs)
        return _cache[T]

    for _m in _targets:
        _m.phonon_modes = _injected
    try:
        yield modes
    finally:
        for _m, _fn in saved:
            _m.phonon_modes = _fn
