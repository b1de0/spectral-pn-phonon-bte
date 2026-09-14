"""Angle-exact Volterra cross-validation of the spectral-AP DOM TTG solver."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinn_bte.physics import ttg_dom


def volterra_trace(v, tau, C, q, t_end, N):
    """Solve the spectral Volterra equation on a uniform N+1-point grid (implicit
    trapezoid; O(N^2) via a running dot product).
    """
    t = np.linspace(0.0, t_end, N + 1)
    dt = t[1] - t[0]
    D = np.sum(C / tau)
    x = np.outer(t, q * v)
    sinc = np.ones_like(x)
    nz = x != 0
    sinc[nz] = np.sin(x[nz]) / x[nz]
    expt = np.exp(-np.outer(t, 1.0 / tau))
    F = (sinc * expt) @ (C / tau) / D
    K = (sinc * expt) @ (C / tau ** 2) / D
    A = np.empty(N + 1)
    A[0] = 1.0
    denom = 1.0 - 0.5 * dt * K[0]
    for n in range(1, N + 1):
        s = 0.5 * K[n] * A[0]
        if n > 1:
            s += K[1:n][::-1] @ A[1:n]
        A[n] = (F[n] + dt * s) / denom
    return t, A


def auto_N(v, tau, q, t_end, pts_per_period=30, pts_per_tau=8,
           n_min=8000, n_max=400_000):
    """Resolve both the fastest sinc oscillation and the sharpest kernel decay."""
    wmax = q * np.max(v)
    n_osc = pts_per_period * wmax * t_end / (2 * np.pi)
    n_tau = pts_per_tau * t_end / np.min(tau)
    return int(min(max(n_min, n_osc, n_tau), n_max))


def validate(L_values=(0.01, 0.1, 1.0), Nk=20, T_ref=300.0, tol=2e-3):
    """Compare ttg_amplitude_curve against the angle-exact Volterra reference."""
    ok = True
    v, tau, C = ttg_dom.phonon_modes(Nk, T_ref)
    for L in L_values:
        q = 2 * np.pi / (L * 1e4)
        td, Ad = ttg_dom.ttg_amplitude_curve(L, "1d", Nk=Nk, T_ref=T_ref)
        N = auto_N(v, tau, q, td[-1])
        if L >= 1.0:            # kernel spike (small tau) needs extra points
            N = max(N, 40_000)
        tv, Av = volterra_trace(v, tau, C, q, td[-1], N)
        d = float(np.abs(np.interp(tv, td, Ad) - Av).max())
        status = "PASS" if d < tol else "FAIL"
        ok &= d < tol
        print(f"  L={L:5.2f} um: max|DOM - Volterra| = {d:.2e}  "
              f"minA_exact = {Av.min():+.4f}  [{status}]")
    return ok


def self_test():
    """Gray single-mode limit must reproduce the exact Zhou/Volterra reference."""
    from pinn_bte.utils.plotting_transient import compute_analytical_amplitude
    ok = True
    for xi in (0.6, 3.0, 10.0):
        t, A = volterra_trace(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                              xi, 10.0, 6000)
        d = float(np.abs(A - compute_analytical_amplitude(t, xi)).max())
        ok &= d < 1e-10
        print(f"  gray xi={xi:4.1f}: max|dA| vs exact reference = {d:.1e}"
              f"  [{'PASS' if d < 1e-10 else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    print("Self-test (gray limit vs compute_analytical_amplitude):")
    ok = self_test()
    print("DOM vs angle-exact spectral Volterra:")
    ok &= validate()
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
