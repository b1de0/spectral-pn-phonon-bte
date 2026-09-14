""": the per-mode optical-thickness spectrum and the composition weight."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The production spectral resolution and reference temperature of every
NK = 20
T_REF = 300.0
from pinn_bte.physics.comb_routing import comb_context, comb_modes, pinned_comb  # noqa: E402
#: The comb every pinned run is trained and scored under, from the ONE
LADDER_SOURCE, COMB = pinned_comb()

#: Angstroms per micrometre -- the unit hop between the mode set (Ang/s) and
#: the grating period (um).  Mirrors ttg_dispersion.ANGSTROM_PER_UM.
ANGSTROM_PER_UM = 1.0e4

#: The nine pinned grating periods of the 1D ladder, in um.
LADDER_UM = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)

#: The three periods the figure draws in full: one per decade, bracketing the
#: crossover.  All three are pinned ladder rungs.
PANEL_UM = (0.1, 1.0, 10.0)

#: The two hard cuts of the RETIRED band-routed predecessor
#: (spectral_pn.py:480-481 defaults xi_split, xi_duh).  Drawn to show WHERE a
#: hard cut lands relative to the spectrum -- which is a function of the
#: period, NOT a constant: either cut lands in the interior only at the
#: slow-collective periods, and outside the spectrum entirely at the ballistic
#: and diffusive ones.  Measured by `cut_position` and asserted in
#: tests/test_p101_mode_spectrum.py; see that test for the three drawn cases.
RETIRED_CUTS = (1.0, 10.0)

def modes(Nk: int = NK, T_ref: float = T_REF):
    """(v [Ang/s], tau [s], C) for the 61-mode JOINT production comb."""
    return comb_modes(COMB, Nk, T_ref)

def q_of(L_um: float) -> float:
    """Grating wavenumber [1/Ang] -- the trainer's own construction."""
    return 2.0 * np.pi / (L_um * ANGSTROM_PER_UM)

def xi_of(L_um: float, Nk: int = NK, T_ref: float = T_REF) -> np.ndarray:
    """Per-mode optical thickness xi_m = q v_m tau_m."""
    v, tau, _ = modes(Nk, T_ref)
    return q_of(L_um) * v * tau

def kernels(xi, y=0.0):
    """K0, K1 through the PRODUCTION kernel (torch, the atan2 branch).

    Called with the same routine the trainer calls, so a change to the shipped
    kernel moves this figure instead of leaving it behind.
    """
    import torch
    xi_t = torch.as_tensor(np.asarray(xi, dtype=float), dtype=torch.float64)
    y_t = torch.as_tensor(np.broadcast_to(np.asarray(y, dtype=float), xi_t.shape).copy(),
                          dtype=torch.float64)
    from pinn_bte.models.spectral_pn import ugks_resolvent_kernels
    K0, K1 = ugks_resolvent_kernels(y_t, xi_t)
    return K0.numpy(), K1.numpy()

def k0_full(L_um: float, gamma_hz: float, Nk: int = NK,
            T_ref: float = T_REF) -> np.ndarray:
    """The y-DEPENDENT kernel, for measuring the size of the y=0 simplification.

    Not used by the figure; used by the report and by the module self-test so
    that "the y dependence is a correction, not a threshold" is a measured
    statement.
    """
    _, tau, _ = modes(Nk, T_ref)
    return kernels(xi_of(L_um, Nk, T_ref), gamma_hz * tau)[0]

def slaved_share_ydep(L_um: float, kind: str = "C", Nk: int = NK,
                      T_ref: float = T_REF) -> float:
    """`slaved_share` with the y_m DEPENDENCE PUT BACK, at the arbiter's own effective collective rate."""
    from pinn_bte.physics.ttg_dom import ttg_decay_spectral

    with comb_context(COMB, Nk, T_ref):
        g = float(ttg_decay_spectral(L_um, "1d", Nk=Nk, T_ref=T_ref))
    return float(np.sum(weights(kind, Nk, T_ref) * k0_full(L_um, g, Nk, T_ref)))

def weights(kind: str, Nk: int = NK, T_ref: float = T_REF) -> np.ndarray:
    """Normalised per-mode weight.

    'C'        heat capacity C_m -- the per-mode prefactor that appears
               explicitly in Eqs. (ugks-c)/(ugks-s), and the standard
               MFP-spectroscopy weighting.
    'closure'  C_m/tau_m -- the weight with which mode m enters the OBSERVED
               amplitude A(t) through the closure moment (spectral_pn.py:336).
    """
    _, tau, C = modes(Nk, T_ref)
    w = {"C": C, "closure": C / tau}[kind]
    return w / w.sum()

def reservoir_share(Nk: int = NK, T_ref: float = T_REF) -> float:
    """The optical reservoir's share of the joint set's total heat capacity."""
    v, tau, C = modes(Nk, T_ref)
    i_res = int(np.argmin(v * tau))
    assert i_res == v.size - 1, f"reservoir expected last, argmin(MFP)={i_res}"
    assert abs(v[i_res] * tau[i_res] - 44.4) < 1e-6, "reservoir MFP != 4.44 nm"
    return float(C[i_res] / C.sum())

def slaved_share(L_um: float, kind: str = "C", Nk: int = NK,
                 T_ref: float = T_REF) -> float:
    """Weighted mean of K0(0, xi_m): the share of the composition that rides
    the collective amplitude rather than its own collisionless carrier.

    0 = every mode is carried entirely by its exact free-streaming kernel
    (free-molecular limit); 1 = every mode is fully slaved to the single
    collective amplitude (Chapman-Enskog limit).  Both ends are limits of ONE
    formula, reached continuously.
    """
    K0, _ = kernels(xi_of(L_um, Nk, T_ref))
    return float(np.sum(weights(kind, Nk, T_ref) * K0))

def xi_geomean(L_um: float, kind: str = "C", Nk: int = NK,
               T_ref: float = T_REF) -> float:
    """Weighted geometric mean of xi_m: WHERE the period places the spectrum.

    Exactly proportional to 1/L, because xi_m = q v_m tau_m and only q moves
    with the period -- the spectrum translates rigidly on a log axis, one
    decade per decade of L.
    """
    lx = np.log(xi_of(L_um, Nk, T_ref))
    return float(np.exp(np.sum(weights(kind, Nk, T_ref) * lx)))

def xi_span_decades(Nk: int = NK, T_ref: float = T_REF) -> float:
    """log10(xi_max/xi_min) of the mode set -- independent of L."""
    xi = xi_of(1.0, Nk, T_ref)
    return float(np.log10(xi.max() / xi.min()))

def share_above_cut(L_um: float, cut: float, kind: str = "C",
                    Nk: int = NK, T_ref: float = T_REF) -> float:
    """Weight fraction with xi_m >= cut -- what a HARD ROUTING CUT would send
    to the collisionless band.  Used to show that a cut bisects the spectrum
    instead of selecting a regime."""
    xi = xi_of(L_um, Nk, T_ref)
    return float(np.sum(weights(kind, Nk, T_ref)[xi >= cut]))

def cut_position(L_um: float, cut: float, Nk: int = NK,
                 T_ref: float = T_REF) -> str:
    """WHERE a hard routing cut lands relative to the spectrum at this period."""
    xi = xi_of(L_um, Nk, T_ref)
    n_above = int((xi >= cut).sum())
    if n_above == 0:
        return "above"
    if n_above == xi.size:
        return "below"
    return "interior"

def xi_weighted_span_decades(kind: str = "C", p_lo: float = 0.05,
                             p_hi: float = 0.95, Nk: int = NK,
                             T_ref: float = T_REF) -> float:
    """The span in log10(xi) between two WEIGHTED quantiles of the mode set."""
    lx = np.log10(xi_of(1.0, Nk, T_ref))          # L cancels: a rigid shift
    w = weights(kind, Nk, T_ref)
    order = np.argsort(lx)
    lxs, ws = lx[order], w[order]
    cdf = np.cumsum(ws) - 0.5 * ws
    return float(np.interp(p_hi, cdf, lxs) - np.interp(p_lo, cdf, lxs))

def spectral_density(L_um: float, kind: str = "C", grid=None, bw: float = 0.13,
                     Nk: int = NK, T_ref: float = T_REF):
    """Weighted density of the mode set over log10(xi), for drawing only."""
    if grid is None:
        grid = np.linspace(-4.0, 5.0, 1200)
    lx = np.log10(xi_of(L_um, Nk, T_ref))
    w = weights(kind, Nk, T_ref)
    d = np.exp(-0.5 * ((grid[:, None] - lx[None, :]) / bw) ** 2) / (
        bw * np.sqrt(2.0 * np.pi))
    return grid, d @ w

def support(L_um: float, Nk: int = NK, T_ref: float = T_REF) -> tuple:
    """(xi_min, xi_max) of the mode set at this period -- the true support the
    drawn density is clipped to, so no fill appears where no mode is."""
    xi = xi_of(L_um, Nk, T_ref)
    return float(xi.min()), float(xi.max())

# --------------------------------------------------------------------------- #
# Independent re-derivations -- the G11 side.  Different code, same number.
# --------------------------------------------------------------------------- #
def k0_by_quadrature(xi) -> np.ndarray:
    """K0(0, xi) = Re <1/(1 + i xi mu)>_mu by ADAPTIVE QUADRATURE.

    Independent of both the shipped atan2 kernel and the closed form: it
    integrates the DEFINING angular moment.  The integrand has width ~1/xi
    about mu = 0 and xi reaches 6.5e3 on this mode set, so `quad` is given the
    singular-ish point explicitly rather than trusted to find it.
    """
    from scipy.integrate import quad
    out = []
    for x in np.atleast_1d(np.asarray(xi, dtype=float)):
        val, _ = quad(lambda m, x=x: 0.5 / (1.0 + (x * m) ** 2),
                      -1.0, 1.0, points=[0.0], limit=400,
                      epsabs=1e-13, epsrel=1e-13)
        out.append(val)
    return np.asarray(out)

def k0_full_by_quadrature(y, xi) -> np.ndarray:
    """K0(y, xi) = Re <1/(1 - y + i xi mu)>_mu by ADAPTIVE QUADRATURE.

    The y-dependent counterpart of `k0_by_quadrature`: the real part of the
    defining moment is (1/2) int (1-y) / ((1-y)^2 + (xi mu)^2) dmu.  Forks the
    shipped atan2 branch on the axis where it matters most -- y > 1, where the
    resolvent changes sign and a branch error would be invisible in the y = 0
    checks.
    """
    from scipy.integrate import quad
    y = np.atleast_1d(np.asarray(y, dtype=float))
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    out = []
    for yi, xii in zip(np.broadcast_to(y, xi.shape), xi):
        a = 1.0 - float(yi)
        val, _ = quad(lambda m, a=a, x=float(xii): 0.5 * a / (a * a + (x * m) ** 2),
                      -1.0, 1.0, points=[0.0], limit=400,
                      epsabs=1e-13, epsrel=1e-13)
        out.append(val)
    return np.asarray(out)

def case_from_trainer(L_um: float, Nk: int = NK, T_ref: float = T_REF):
    """(v, tau, C, q) through the PRODUCTION case builder."""
    from pinn_bte.training.transient.spectral_pn_trainer import build_si_case
    with comb_context(COMB, Nk, T_ref):
        c = build_si_case(L_um, Nk=Nk, T_ref=T_ref)
    return c.v, c.tau, c.C, c.q

def self_test() -> bool:
    """Eight invariants, printed with their numbers, not asserted silently."""
    v, tau, C = modes()
    assert v.shape == (61,), v.shape

    # 1. the production kernel IS the closed form, at every mode, every rung
    worst = 0.0
    for L in LADDER_UM:
        xi = xi_of(L)
        worst = max(worst, float(np.abs(kernels(xi)[0] - np.arctan(xi) / xi).max()))
    print(f"  K0(prod) vs arctan(xi)/xi, max abs over 9x60 : {worst:.3e}")
    assert worst < 1e-12

    # 2. and it is the defining angular moment
    xi = xi_of(1.0)
    dq = float(np.abs(kernels(xi)[0] - k0_by_quadrature(xi)).max())
    print(f"  K0(prod) vs adaptive quadrature, max abs      : {dq:.3e}")
    assert dq < 1e-10

    # 3. rigid translation: xi_m(L) * L is L-independent
    ratios = [xi_geomean(L) * L for L in LADDER_UM]
    print(f"  xi_geomean(L) * L, spread over the ladder     : "
          f"{max(ratios) - min(ratios):.3e}")
    assert max(ratios) - min(ratios) < 1e-9

    # 4. both asymptotic limits are reached by the SAME formula.  The bound at
    #    the free-molecular end is 1e-2, not the legacy 1e-3: the optical
    #    reservoir's 4.4 nm MFP means its xi at L = 1e-4 um is ~279, where
    #    K0 ~ 5.6e-3 -- the limit is reached, one decade of L later.
    print(f"  slaved share at L=1e-4 um / 1e4 um           : "
          f"{slaved_share(1e-4):.6f} / {slaved_share(1e4):.6f}")
    assert slaved_share(1e-4) < 1e-2 and slaved_share(1e4) > 1 - 1e-5

    # 5. monotone in L, under BOTH weightings -- no kink, no threshold
    for kind in ("C", "closure"):
        f = [slaved_share(L, kind) for L in LADDER_UM]
        assert all(b > a for a, b in zip(f, f[1:])), (kind, f)
        print(f"  slaved share ({kind:7s}) over the ladder      : "
              + " ".join(f"{x:.3f}" for x in f))

    # 6. a hard cut's position is a FUNCTION OF THE PERIOD, not a constant.
    #    This is the invariant the retired "either cut falls inside the
    #    spectrum at every period" wording violated, so it is asserted, not
    #    the xi = 1 rule is now INTERIOR -- it slices between the optical
    #    reservoir (43.1% of C, below) and all 60 acoustic modes (above);
    #    on the legacy acoustic comb it sat below the whole spectrum.  The
    # "below" case retreats to the undrawn L = 0.01 um, and "above" still
    #    holds at (L=10, xi=10).
    pos = {(L, c): cut_position(L, c) for L in PANEL_UM for c in RETIRED_CUTS}
    print("  hard-cut position vs the spectrum               : "
          + " ".join(f"L={L:g}/xi={c:g}:{pos[(L, c)]}"
                      for L in PANEL_UM for c in RETIRED_CUTS))
    assert pos[(1.0, 1.0)] == "interior" and pos[(1.0, 10.0)] == "interior"
    assert pos[(0.1, 1.0)] == "interior" and pos[(10.0, 10.0)] == "above"
    assert cut_position(0.01, 1.0) == "below"

    # 7. the drawn width and the SET width are different numbers
    print(f"  xi span: unweighted / C-weighted 5-95%        : "
          f"{xi_span_decades():.4f} / {xi_weighted_span_decades():.4f} decades")
    assert xi_weighted_span_decades() < xi_span_decades()

    # 8. the trainer's own case builder gives the same four arrays
    v2, tau2, C2, q2 = case_from_trainer(1.0)
    d = max(float(np.abs(v - v2).max()), float(np.abs(tau - tau2).max()),
            float(np.abs(C - C2).max()), abs(q_of(1.0) - q2))
    print(f"  build_si_case(1.0) vs this module, max abs    : {d:.3e}")
    assert d == 0.0
    return True

def _report() -> None:
    print("mode spectrum -- dispersion + Holland tau only, no training\n")
    print(f"  {len(modes()[0])} modes (Nk={NK}, T_ref={T_REF} K), "
          f"xi span {xi_span_decades():.3f} decades at every period "
          f"(C-weighted 5-95%: {xi_weighted_span_decades():.3f})")
    print(f"  optical reservoir: {reservoir_share():.4%} of total C "
          f"(acoustic complement {1.0 - reservoir_share():.4%}) -- the "
          f"stem-and-fill split of fig:mode_spectrum panel (a)\n")
    hdr = (f"  {'L [um]':>8} {'xi_gm(C)':>10} {'slaved(C)':>10} "
           f"{'slaved(cl)':>11} {'C above xi=1':>13} {'C above xi=10':>14}")
    print(hdr)
    print(" " + "-" * (len(hdr) - 2))
    for L in LADDER_UM:
        print(f"  {L:8g} {xi_geomean(L):10.4g} {slaved_share(L):10.4f} "
              f"{slaved_share(L, 'closure'):11.4f} "
              f"{share_above_cut(L, 1.0):13.4f} {share_above_cut(L, 10.0):14.4f}")
    print("\n  WHERE EACH RETIRED HARD CUT LANDS -- a function of the period,")
    print("  which is why no single sentence covers all three:")
    for L in PANEL_UM:
        print(f"    L={L:6g} " + " ".join(
            f"xi={c:g}: {cut_position(L, c):8s} "
            f"({int((xi_of(L) >= c).sum()):2d}/{xi_of(L).size} modes above, "
            f"{share_above_cut(L, c):.4%} of C)" for c in RETIRED_CUTS))

    print("\n  y = gamma_hat tau_m correction, DOM reference's own effective")
    print("  rate on the SAME joint comb (the rate and the weights must not")
    print("  mix combs inside one number).  QUOTE THE WHOLE LADDER, NOT THE")
    print("  MILDEST RUNG: it is non-monotone, and at L = 0.01 um it leaves")
    print("  [0,1] and flips sign.")
    from pinn_bte.physics.ttg_dom import ttg_decay_spectral
    w = weights("C")
    for L in LADDER_UM:
        with comb_context(COMB, NK, T_REF):
            g = float(ttg_decay_spectral(L, "1d", Nk=NK, T_ref=T_REF))
        mark = "  <- DRAWN" if L in PANEL_UM else ""
        print(f"    L={L:6g}  <K0(0,xi)>={slaved_share(L):7.4f} "
              f"<K0(y,xi)>={float(np.sum(w * k0_full(L, g))):7.4f} "
              f"(gamma_DOM = {g:.4e} Hz){mark}")
    print("\n  self-test:")
    self_test()
    print("\n  OK")

if __name__ == "__main__":
    _report()
