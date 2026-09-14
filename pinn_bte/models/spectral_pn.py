"""Spectral-PN core for the transient-thermal-grating phonon BTE: Legendre
streaming matrix, generator and matrix-exponential solve, Duhamel free part,
UGKS resolvent kernels, and the plain / Chapman-Enskog / xi-hybrid
coefficient networks with their residuals.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def legendre_streaming_matrix(L_max: int) -> np.ndarray:
    """T[l, j] such that the P_l-coefficient of mu * sum_j a_j P_j is [T a]_l."""
    T = np.zeros((L_max + 1, L_max + 1))
    for l in range(L_max + 1):
        if l - 1 >= 0:
            T[l, l - 1] = l / (2 * l - 1)
        if l + 1 <= L_max:
            T[l, l + 1] = (l + 1) / (2 * l + 3)
    return T


def _pn_rhs(y, v, tau, C, q, T, D):
    """RHS of the PN ODE system. y = (c, s), each (M, L+1)."""
    c, s = y
    dT_c = np.sum(c[:, 0] / tau) / D
    dT_s = np.sum(s[:, 0] / tau) / D
    src_c = np.zeros_like(c); src_c[:, 0] = C * dT_c
    src_s = np.zeros_like(s); src_s[:, 0] = C * dT_s
    vq = (v * q)[:, None]
    dc = -vq * (s @ T.T) - (c - src_c) / tau[:, None]
    ds = +vq * (c @ T.T) - (s - src_s) / tau[:, None]
    return dc, ds


def pn_generator(v, tau, C, q, L_max, inv_tau_N=None) -> np.ndarray:
    """Generator A of the linear PN system on y = [c.ravel(), s.ravel()] (each (M,
    L_max+1) row-major). Includes the rank-2 energy closure coupling.
    """
    v = np.asarray(v, dtype=float)
    tau = np.asarray(tau, dtype=float)
    C = np.asarray(C, dtype=float)
    M, L1 = len(v), L_max + 1
    n = M * L1
    T = legendre_streaming_matrix(L_max)
    D = np.sum(C / tau)

    A = np.zeros((2 * n, 2 * n))
    idx = lambda m, l: m * L1 + l
    for m in range(M):
        vq = v[m] * q
        for l in range(L1):
            i_c, i_s = idx(m, l), n + idx(m, l)
            A[i_c, i_c] -= 1.0 / tau[m]
            A[i_s, i_s] -= 1.0 / tau[m]
            for j in range(L1):
                if T[l, j] != 0.0:
                    A[i_c, n + idx(m, j)] -= vq * T[l, j]
                    A[i_s, idx(m, j)] += vq * T[l, j]
        # closure source into l=0: + (C_m/tau_m) * dT, dT = sum_j c_{j,0}/tau_j / D
        for j in range(M):
            A[idx(m, 0), idx(j, 0)] += (C[m] / tau[m]) * (1.0 / tau[j]) / D
            A[n + idx(m, 0), n + idx(j, 0)] += (C[m] / tau[m]) * (1.0 / tau[j]) / D

    if inv_tau_N is not None and L_max >= 1:
        inv_tau_N = np.asarray(inv_tau_N, dtype=float)
        Dp = np.sum(C * v ** 2 * inv_tau_N)
        if Dp > 0.0:                          # else pure-RTA: leave A untouched
            for m in range(M):
                if inv_tau_N[m] == 0.0:
                    continue
                src_m = (C[m] * v[m] * inv_tau_N[m]) / Dp
                for j in range(M):
                    if inv_tau_N[j] == 0.0:
                        continue
                    coup = src_m * (v[j] * inv_tau_N[j])
                    A[idx(m, 1), idx(j, 1)] += coup
                    A[n + idx(m, 1), n + idx(j, 1)] += coup
    return A


def solve_pn_ode(v, tau, C, q, t, L_max, return_state: bool = False):
    """Reference (non-neural) solve of the PN system, for validating the residual
    assembly and quantifying the required L_max.
    """
    from scipy.linalg import expm

    v = np.asarray(v, dtype=float)
    tau = np.asarray(tau, dtype=float)
    C = np.asarray(C, dtype=float)
    M, L1 = len(v), L_max + 1
    n = M * L1
    D = np.sum(C / tau)

    dts = np.diff(t)
    assert np.allclose(dts, dts[0], rtol=1e-8), "uniform t grid required"
    dt = float(dts[0])

    A = pn_generator(v, tau, C, q, L_max)

    y = np.zeros(2 * n)
    for m in range(M):
        y[m * L1] = C[m]

    E = expm(A * dt)
    traj = np.empty((len(t), 2 * n))
    traj[0] = y
    for k in range(1, len(t)):
        y = E @ y
        traj[k] = y

    c_traj = traj[:, :n].reshape(len(t), M, L1)
    s_traj = traj[:, n:].reshape(len(t), M, L1)
    A_amp = (c_traj[:, :, 0] / tau[None, :]).sum(axis=1) / D
    A_amp = A_amp / A_amp[0]
    if return_state:
        return A_amp, (c_traj, s_traj)
    return A_amp


_GL_NODES = 128
                  # exact to machine precision for phase arguments
                  # x = v q t up to ~200 (l <= ~20)

_DUHAMEL_PROJ_CACHE: dict = {}


def _duhamel_proj(L_top: int):
    hit = _DUHAMEL_PROJ_CACHE.get(L_top)
    if hit is None:
        mu, w = np.polynomial.legendre.leggauss(_GL_NODES)
        P = np.stack([np.polynomial.legendre.Legendre.basis(l)(mu)
                      for l in range(L_top + 1)])       # (L1, Nmu)
        # projection weights: (2l+1)/2 * w * P_l(mu_k)
        proj = (2 * np.arange(L_top + 1)[:, None] + 1) / 2.0 * (w[None, :] * P)
        hit = (mu, proj)
        _DUHAMEL_PROJ_CACHE[L_top] = hit
    return hit


def duhamel_free_coefficients(t, v, tau, C, q, L_top, ic_amp=None):
    """Legendre coefficients (up to l = L_top) of the exact collisionless part of
    the cosine-IC TTG evolution,
    """
    mu, proj = _duhamel_proj(L_top)
    proj_t = torch.as_tensor(proj, dtype=t.dtype, device=t.device)
    mu_t = torch.as_tensor(mu, dtype=t.dtype, device=t.device)

    tt = t.reshape(-1, 1, 1)                             # (Nt, 1, 1)
    x = tt * (q * v).reshape(1, -1, 1)                   # (Nt, M, 1)
    phase = x * mu_t.reshape(1, 1, -1)                   # (Nt, M, Nmu)
    damp = torch.exp(-tt * (1.0 / tau).reshape(1, -1, 1))
    a_ic = C if ic_amp is None else ic_amp
    amp = a_ic.reshape(1, -1, 1) * damp                  # (Nt, M, 1)
    c_free = amp * torch.einsum("tmk,lk->tml", torch.cos(phase), proj_t)
    s_free = amp * torch.einsum("tmk,lk->tml", torch.sin(phase), proj_t)
    return c_free, s_free


def ugks_resolvent_kernels(y: torch.Tensor, xi: torch.Tensor):
    a = 1.0 - y
    b = xi.reshape(*([1] * (y.dim() - 1)), -1).expand_as(a)
    small = b.abs() < 1e-10
    one = torch.ones_like(b)
    b_s = torch.where(small, one, b)
    a_s = torch.where(small, a, one)
    K0_main = torch.atan2(2.0 * a * b_s, a * a - b_s * b_s) / (2.0 * b_s)
    K1_main = 3.0 * (1.0 - a * K0_main) / b_s
    K0 = torch.where(small, 1.0 / a_s, K0_main)
    K1 = torch.where(small, b / (a_s * a_s), K1_main)
    return K0, K1


class PNCoefficientNet(nn.Module):
    """a_{m,l}(t) with the exact IC pinned by construction."""

    def __init__(self, n_modes: int, L_max: int, width: int = 64,
                 depth: int = 4, emb_dim: int = 8,
                 gamma_slow: float = None, duhamel: bool = False):
        super().__init__()
        self.n_modes, self.L_max = n_modes, L_max
        self.gamma_slow = gamma_slow
        self.duhamel = duhamel
        self.emb = nn.Embedding(n_modes, emb_dim)
        layers = [nn.Linear(3 + emb_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2 * (L_max + 1))]
        self.trunk = nn.Sequential(*layers)
        if gamma_slow is not None:
            self.slow = nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                                      nn.Linear(32, 32), nn.SiLU(),
                                      nn.Linear(32, 1))

    def coefficients(self, t: torch.Tensor, v: torch.Tensor,
                     tau: torch.Tensor, C: torch.Tensor,
                     q: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (c, s), each (Nt, M, L+1)."""
        Nt, M = t.shape[0], self.n_modes
        tt = t.reshape(-1, 1, 1)
        z1 = tt * (1.0 / tau).reshape(1, -1, 1)          # t / tau_m
        z2 = tt * (q * v).reshape(1, -1, 1)              # t * q v_m
        feats = torch.cat([
            torch.log1p(z1), torch.log1p(z2),
            torch.tanh(0.1 * z2),                        # bounded phase cue
        ], dim=2)                                        # (Nt, M, 3)
        emb = self.emb.weight.unsqueeze(0).expand(Nt, -1, -1)
        out = self.trunk(torch.cat([feats, emb], dim=2))  # (Nt, M, 2(L+1))
        raw_c, raw_s = out.chunk(2, dim=2)
        r = (1.0 / tau + q * v).reshape(1, -1, 1)
        gate = 1.0 - torch.exp(-tt * r)
        if self.duhamel:
            base_c, base_s = duhamel_free_coefficients(
                t, v, tau, C, q, self.L_max)
            c = base_c + gate * raw_c * C.reshape(1, -1, 1)
            s = base_s + gate * raw_s * C.reshape(1, -1, 1)
            return c, s
        ic = torch.zeros(1, M, self.L_max + 1, dtype=out.dtype,
                         device=out.device)
        ic[0, :, 0] = C
        if self.gamma_slow is not None:
            ts = tt[:, 0, :] * self.gamma_slow          # (Nt, 1) slow time
            gate_slow = 1.0 - torch.exp(-ts)
            A_hat = 1.0 + gate_slow * self.slow(torch.log1p(ts))
            ic = ic * A_hat.unsqueeze(2)
        c = ic + gate * raw_c * C.reshape(1, -1, 1)
        s = gate * raw_s * C.reshape(1, -1, 1)
        return c, s


def _closure(c, tau, C):
    D = torch.sum(C / tau)
    return torch.sum(c[..., 0] / tau.reshape(1, -1), dim=1) / D


def rate_corridor(v, tau, q):
    """Physical bounds on the collective TTG decay rate, from the mode set alone.
    Returns (gamma_lo, gamma_hi) in the mode set's own rate units.
    """
    v = np.asarray(v, dtype=float)
    tau = np.asarray(tau, dtype=float)
    xi = q * v * tau
    S = 3.0 * (xi - np.arctan(xi)) / xi ** 3
    lo = float(((q ** 2) * v ** 2 * tau * S / 3.0).min())
    hi = float((1.0 / tau + q * v).max())
    return lo, hi


def residual_time_weight(amplitude, floor):
    """Per-time weight that makes a given RELATIVE error cost the same at every
    time: `w = 1 / max(|A(t)|, floor)`.
    """
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor!r}: it is the "
                         "maximum amplification and it is what keeps the "
                         "weight finite through the ballistic zero crossing")
    return 1.0 / torch.clamp(amplitude.detach().abs(), min=floor)


_CLOSURE_PIN_DELTA = 1e-11


def closure_pin_shim(c0: torch.Tensor, w: torch.Tensor, tau: torch.Tensor,
                     C: torch.Tensor, A_hat: torch.Tensor,
                     delta: float = _CLOSURE_PIN_DELTA) -> torch.Tensor:
    D = torch.sum(C / tau)
    tau_row = tau.reshape(1, -1)
    A_un = torch.sum(c0 / tau_row, dim=1) / D        # == _closure(c, tau, C)
    S = torch.sum(w / tau_row, dim=1) / D
    sigma = (A_hat.reshape(-1) - A_un) * (S / (S * S + delta * delta))
    return sigma.reshape(-1, 1) * w


class CEPNCoefficientNet(nn.Module):

    def __init__(self, n_modes: int, L_max: int, gamma_kin: float,
                 v, tau, q: float, width: int = 64, depth: int = 4,
                 emb_dim: int = 8):
        super().__init__()
        self.n_modes, self.L_max = n_modes, L_max
        self.gamma_kin = gamma_kin
        self.emb = nn.Embedding(n_modes, emb_dim)
        layers = [nn.Linear(3 + emb_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2 * (L_max + 1))]
        self.trunk = nn.Sequential(*layers)
        self.slow = nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                                  nn.Linear(32, 32), nn.SiLU(),
                                  nn.Linear(32, 1))
        v_t = torch.as_tensor(np.asarray(v, dtype=float))
        tau_t = torch.as_tensor(np.asarray(tau, dtype=float))
        xi = q * v_t * tau_t
        self.register_buffer("wce", 1.0 / (1.0 + xi ** 2))
        self.register_buffer("epsl", torch.clamp(xi, 1e-3, 1.0))
        self.register_buffer("u_closure", 1.0 / tau_t)
        self.register_buffer(
            "T_stream",
            torch.as_tensor(legendre_streaming_matrix(L_max)))

    def coefficients(self, t: torch.Tensor, v, tau, C, q: float):
        Nt, M = t.shape[0], self.n_modes
        tt = t.reshape(-1, 1, 1)
        z1 = tt * (1.0 / tau).reshape(1, -1, 1)
        z2 = tt * (q * v).reshape(1, -1, 1)
        feats = torch.cat([torch.log1p(z1), torch.log1p(z2),
                           torch.tanh(0.1 * z2)], dim=2)
        emb = self.emb.weight.unsqueeze(0).expand(Nt, -1, -1)
        out = self.trunk(torch.cat([feats, emb], dim=2))
        raw_c, raw_s = out.chunk(2, dim=2)
        r = (1.0 / tau + q * v).reshape(1, -1, 1)
        gate = 1.0 - torch.exp(-tt * r)

        ts = tt[:, 0, :] * self.gamma_kin
        A_hat = 1.0 + (1.0 - torch.exp(-ts)) * self.slow(torch.log1p(ts))

        g0 = gate[..., 0] * raw_c[..., 0] * C.reshape(1, -1)
        for wvec in (self.u_closure, torch.ones_like(self.u_closure)):
            wn = wvec / torch.sqrt((wvec * wvec).sum())
            g0 = g0 - wn.reshape(1, -1) * (g0 * wn.reshape(1, -1)
                                           ).sum(1, keepdim=True)

        corr_scale = (self.epsl.reshape(1, -1, 1) * C.reshape(1, -1, 1))
        corr_c = gate * raw_c * corr_scale
        corr_s = gate * raw_s * corr_scale

        c = torch.zeros(Nt, M, self.L_max + 1,
                        dtype=out.dtype, device=out.device)
        s = torch.zeros_like(c)
        c[..., 0] = C.reshape(1, -1) * A_hat + g0
        vqt = (v * q * tau).reshape(1, -1)
        w = self.wce.reshape(1, -1)
        for l in range(1, self.L_max + 1):
            Tl = float(self.T_stream[l, l - 1])
            s[..., l] = w * (vqt * Tl * c[..., l - 1]) + corr_s[..., l]
            c[..., l] = w * (-vqt * Tl * s[..., l - 1]) + corr_c[..., l]
        return c, s


class XiHybridPNCoefficientNet(nn.Module):

    _BLEND_DELTA = 1e-4   # smooth reciprocal 1/A -> A/(A^2 + delta^2): exact
    _BLEND_YMAX = 50.0    # smooth range bound y -> Y tanh(y/Y) on the kernel

    def __init__(self, n_modes: int, L_max: int, gamma_kin: float,
                 v, tau, q: float, width: int = 64, depth: int = 4,
                 emb_dim: int = 8, xi_split: float = 1.0,
                 xi_duh: float = 10.0, shared_ahat: bool = False,
                 C=None, carrier_energy: bool = False, k_harmonic: int = 1,
                 blend: str = "banded", macro_row: str = "on",
                 closure_pin: bool = False, gauge: str = "additive"):
        super().__init__()
        if gauge not in ("additive", "positive", "corridor"):
            raise ValueError(f"gauge={gauge!r} (expected 'additive', "
                             "'positive' or 'corridor')")
        self.gauge = gauge
        if gauge == "corridor":
            lo, hi = rate_corridor(v, tau, q)
            self.register_buffer("_g_lo", torch.tensor(lo))
            self.register_buffer("_g_log_span", torch.tensor(np.log(hi / lo)))
        if blend not in ("banded", "ugks"):
            raise ValueError(f"blend={blend!r} (expected 'banded' or 'ugks')")
        if macro_row not in ("on", "off"):
            raise ValueError(f"macro_row={macro_row!r} "
                             "(expected 'on' or 'off')")
        if macro_row == "off" and blend != "ugks":
            raise ValueError("macro_row='off' requires blend='ugks' "
                             "(the ablation is a ugks-path "
                             "instrument; banded licensing is a derived "
                             "physical condition)")
        self.macro_row = macro_row
        if closure_pin and blend != "ugks":
            raise ValueError(
                "closure_pin=True requires blend='ugks': the l=0 slaved slot "
                "the shim corrects (and the A_hat gauge it pins to) exist "
                "only on the ugks path — the banded path's l=0 structure is "
                "the carrier machinery, which this repair does not touch")
        if closure_pin and int(k_harmonic) != 1:
            raise ValueError(
                "closure_pin=True requires k_harmonic=1: `_closure` is the "
                "SINGLE-harmonic closure moment, so pinning it inside a 2q "
                "block would pin the wrong object. The {DC,q,2q} stack "
                "(spectral_pn_ugks_2q) has its own slaved machinery and its "
                "own per-harmonic macro rows; this repair is NOT half-applied "
                "there — that is a separate PLANS entry")
        self.closure_pin = bool(closure_pin)
        self.blend = blend
        if blend == "ugks":
            assert L_max >= 1, "blend='ugks' slaves the l=1 flux slot"
            assert not shared_ahat and not carrier_energy, \
                ("blend='ugks' replaces the carrier machinery (A_hat is the "
                 "global amplitude by construction)")
        self.n_modes, self.L_max = n_modes, L_max
        self.gamma_kin = gamma_kin
        self.xi_split = xi_split
        self.xi_duh = xi_duh
        self.k_harmonic = int(k_harmonic)
        self.carrier_energy = carrier_energy
        self.shared_ahat = shared_ahat
        self.emb = nn.Embedding(n_modes, emb_dim)
        layers = [nn.Linear(3 + emb_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2 * (L_max + 1))]
        self.trunk = nn.Sequential(*layers)
        self.slow = nn.Sequential(nn.Linear(1, 32), nn.SiLU(),
                                  nn.Linear(32, 32), nn.SiLU(),
                                  nn.Linear(32, 1))
        v_t = torch.as_tensor(np.asarray(v, dtype=float))
        tau_t = torch.as_tensor(np.asarray(tau, dtype=float))
        xi = (self.k_harmonic * q) * v_t * tau_t
        ce = xi < xi_split
        duh = xi >= max(xi_duh, xi_split)   # CE takes precedence below split
        self.register_buffer("ce_mask", ce)
        self.register_buffer("duhamel_mask", duh)
        if shared_ahat and C is not None:
            C_t = torch.as_tensor(np.asarray(C, dtype=float))
            slow = ~duh
            self.f_C_slow = (float(C_t[slow].sum() / C_t.sum())
                             if bool(slow.any()) else 0.0)
            self.shared_active = self.f_C_slow > 0.9
        else:
            self.f_C_slow = float("nan")
            self.shared_active = bool(shared_ahat)
        self.macro_licensed = bool(self.shared_active
                                   and bool(ce.all()))
        if blend == "ugks":
            self.macro_licensed = True
            if macro_row == "off":
                self.macro_licensed = False
        # carrier subset (static): the modes whose c0 rides A_hat — the
        # slow bands under an active shared carrier, else the CE band.
        slow = ~duh
        use_shared = self.shared_active and bool(slow.any())
        self.register_buffer("carrier_mask", slow if use_shared else ce)
        self.register_buffer("wce", 1.0 / (1.0 + xi ** 2))
        self.register_buffer("epsl", torch.clamp(xi, 1e-3, 1.0))
        self.register_buffer("u_closure", 1.0 / tau_t)
        self.register_buffer(
            "T_stream",
            torch.as_tensor(legendre_streaming_matrix(L_max)))

    def _blend_amplitude(self, t2: torch.Tensor):
        """(A_hat, dA_hat/dt) from the slow head, each (Nt, 1); A_hat(0) = 1
        exactly (the kinetic-diffusive gate).
        """
        ts = t2 * self.gamma_kin
        h = torch.log1p(ts)
        dh = 1.0 / (1.0 + ts)                       # d log1p(ts) / d ts
        for layer in self.slow:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                dh = dh.matmul(layer.weight.t())
            elif isinstance(layer, nn.SiLU):
                sig = torch.sigmoid(h)              # pre-activation h
                dh = dh * (sig * (1.0 + h * (1.0 - sig)))
                h = h * sig
            else:                                    # pragma: no cover
                raise NotImplementedError(
                    "blend amplitude derivative: unsupported slow-head "
                    f"layer {type(layer).__name__}")
        if self.gauge == "corridor":
            sig = torch.sigmoid(h)
            gamma = self._g_lo * torch.exp(self._g_log_span * sig)
            dgamma_dts = gamma * self._g_log_span * sig * (1.0 - sig) * dh
            t_phys = ts / self.gamma_kin
            expo = torch.clamp(gamma * t_phys, max=700.0)
            A_hat = torch.exp(-expo)
            # d/dt = gamma_kin * d/dts, and d(gamma*t)/dts = gamma/gamma_kin +
            # t*dgamma/dts
            dA_dt = -A_hat * (gamma + t_phys * dgamma_dts * self.gamma_kin)
            return A_hat, dA_dt
        if self.gauge == "positive":
            phi = torch.nn.functional.softplus(h)
            dphi = torch.sigmoid(h) * dh          # d softplus(h)/dts
            expo = torch.clamp(ts * phi, max=700.0)
            A_hat = torch.exp(-expo)
            dA_dt = -self.gamma_kin * A_hat * (phi + ts * dphi)
            return A_hat, dA_dt
        e = torch.exp(-ts)
        A_hat = 1.0 + (1.0 - e) * h
        dA_dt = self.gamma_kin * (e * h + (1.0 - e) * dh)
        return A_hat, dA_dt

    def _coefficients_ugks(self, t: torch.Tensor, v, tau, C, q: float):
        assert self.k_harmonic >= 1, \
            ("blend='ugks' requires k_harmonic >= 1: DC (h=0) is never a "
             "ugks block (use the plain DC trunk)")
        Nt, M = t.shape[0], self.n_modes
        kq = self.k_harmonic * q
        tt = t.reshape(-1, 1, 1)
        z1 = tt * (1.0 / tau).reshape(1, -1, 1)
        z2 = tt * (kq * v).reshape(1, -1, 1)
        feats = torch.cat([torch.log1p(z1), torch.log1p(z2),
                           torch.tanh(0.1 * z2)], dim=2)
        emb = self.emb.weight.unsqueeze(0).expand(Nt, -1, -1)
        out = self.trunk(torch.cat([feats, emb], dim=2))
        raw_c, raw_s = out.chunk(2, dim=2)
        r = (1.0 / tau + kq * v).reshape(1, -1, 1)
        gate = 1.0 - torch.exp(-tt * r)

        corr_scale = self.epsl.reshape(1, -1, 1) * C.reshape(1, -1, 1)
        corr_c = gate * raw_c * corr_scale
        corr_s = gate * raw_s * corr_scale

        # exact collisionless free part — GLOBAL (all modes)
        free_c, free_s = duhamel_free_coefficients(t, v, tau, C, kq,
                                                   self.L_max)

        A_hat, dA_dt = self._blend_amplitude(tt[:, 0, :])       # (Nt, 1)
        inv_A = A_hat / (A_hat * A_hat + self._BLEND_DELTA ** 2)
        gamma_hat = -dA_dt * inv_A                               # (Nt, 1)
        y = gamma_hat * tau.reshape(1, -1)                       # (Nt, M)
        y = self._BLEND_YMAX * torch.tanh(y / self._BLEND_YMAX)
        K0, K1 = ugks_resolvent_kernels(y, kq * v * tau)

        growth = 1.0 - torch.exp(-z1[..., 0])                    # (Nt, M)
        slave = growth * C.reshape(1, -1) * A_hat                # (Nt, M)
        sc = (slave * K0).unsqueeze(-1)
        ss = (slave * K1).unsqueeze(-1)
        c = free_c + corr_c + torch.cat(
            [sc, sc.new_zeros(Nt, M, self.L_max)], dim=-1)
        s = free_s + corr_s + torch.cat(
            [ss.new_zeros(Nt, M, 1), ss,
             ss.new_zeros(Nt, M, self.L_max - 1)], dim=-1)
        if self.closure_pin:
            shim = self._closure_pin_shim(c[..., 0], growth, tau, C, A_hat)
            c = c + torch.cat(
                [shim.unsqueeze(-1), shim.new_zeros(Nt, M, self.L_max)],
                dim=-1)
        return c, s

    def _closure_pin_shim(self, c0, growth, tau, C, A_hat):
        """The shim block (Nt, M): direction w_m = growth_m * C_m, normalised by
        the CLOSURE's own 1/tau weighting (closure_pin_shim).
        """
        return closure_pin_shim(c0, growth * C.reshape(1, -1), tau, C, A_hat,
                                _CLOSURE_PIN_DELTA)

    def coefficients(self, t: torch.Tensor, v, tau, C, q: float):
        if self.blend == "ugks":
            return self._coefficients_ugks(t, v, tau, C, q)
        Nt, M = t.shape[0], self.n_modes
        kq = self.k_harmonic * q      # block wavenumber (k_harmonic=1 => q)
        tt = t.reshape(-1, 1, 1)
        z1 = tt * (1.0 / tau).reshape(1, -1, 1)
        z2 = tt * (kq * v).reshape(1, -1, 1)
        feats = torch.cat([torch.log1p(z1), torch.log1p(z2),
                           torch.tanh(0.1 * z2)], dim=2)
        emb = self.emb.weight.unsqueeze(0).expand(Nt, -1, -1)
        out = self.trunk(torch.cat([feats, emb], dim=2))
        raw_c, raw_s = out.chunk(2, dim=2)
        r = (1.0 / tau + kq * v).reshape(1, -1, 1)
        gate = 1.0 - torch.exp(-tt * r)

        corr_c = gate * raw_c * C.reshape(1, -1, 1)
        corr_s = gate * raw_s * C.reshape(1, -1, 1)
        ic = torch.zeros(Nt, M, self.L_max + 1, dtype=out.dtype,
                         device=out.device)
        ic[..., 0] = C.reshape(1, -1)
        c_kin = ic + corr_c
        s_kin = corr_s
        if bool(self.duhamel_mask.any()):
            base_c, base_s = duhamel_free_coefficients(t, v, tau, C, kq,
                                                       self.L_max)
            dm = self.duhamel_mask.reshape(1, -1, 1)
            c_kin = torch.where(dm, base_c + corr_c, c_kin)
            s_kin = torch.where(dm, base_s + corr_s, s_kin)
        slow_band = ~self.duhamel_mask
        use_shared = self.shared_active and bool(slow_band.any())
        if not bool(self.ce_mask.any()) and not use_shared:
            return c_kin, s_kin

        subset = self.carrier_mask
        ts = tt[:, 0, :] * self.gamma_kin
        A_hat = 1.0 + (1.0 - torch.exp(-ts)) * self.slow(torch.log1p(ts))
        g0 = gate[..., 0] * raw_c[..., 0] * C.reshape(1, -1)
        cem = subset.to(out.dtype)
        ones = torch.ones_like(self.u_closure)
        projs = (ones,) if self.carrier_energy else (self.u_closure, ones)
        for wvec in projs:
            wm = wvec * cem
            wn = wm / torch.sqrt((wm * wm).sum())
            g0 = g0 - wn.reshape(1, -1) * (g0 * wn.reshape(1, -1)
                                           ).sum(1, keepdim=True)
        c0_shared = C.reshape(1, -1) * A_hat + g0            # (Nt, M)
        if use_shared:
            # plain-band modes ride the carrier on l=0; l>=1 stays free
            plain_band = slow_band & ~self.ce_mask
            pm = plain_band.reshape(1, -1)
            c_kin = torch.cat(
                [torch.where(pm, c0_shared, c_kin[..., 0]).unsqueeze(-1),
                 c_kin[..., 1:]], dim=-1)
            if not bool(self.ce_mask.any()):
                return c_kin, s_kin

        corr_scale = (self.epsl.reshape(1, -1, 1) * C.reshape(1, -1, 1))
        corr_c = gate * raw_c * corr_scale
        corr_s = gate * raw_s * corr_scale

        c_ce = torch.zeros(Nt, M, self.L_max + 1,
                           dtype=out.dtype, device=out.device)
        s_ce = torch.zeros_like(c_ce)
        c_ce[..., 0] = c0_shared
        vqt = (v * kq * tau).reshape(1, -1)
        w = self.wce.reshape(1, -1)
        for l in range(1, self.L_max + 1):
            Tl = float(self.T_stream[l, l - 1])
            s_ce[..., l] = w * (vqt * Tl * c_ce[..., l - 1]) + corr_s[..., l]
            c_ce[..., l] = w * (-vqt * Tl * s_ce[..., l - 1]) + corr_c[..., l]

        sel = self.ce_mask.reshape(1, -1, 1)
        return torch.where(sel, c_ce, c_kin), torch.where(sel, s_ce, s_kin)


def pn_amplitude(net: PNCoefficientNet, t, tau, C, v, q):
    c, _ = net.coefficients(t, v, tau, C, q)
    return _closure(c, tau, C)


def pn_residual(net: PNCoefficientNet, t, v, tau, C, q,
                ap_scaling: bool = False, return_macro: bool = False,
                return_carrier: bool = False, weight_floor=None):
    """Dimensionless PN residual on the collocation times t (1D tensor)."""
    from torch.func import jvp

    def fn(tt):
        c, s = net.coefficients(tt, v, tau, C, q)
        return torch.stack([c, s])

    y, dy = jvp(fn, (t,), (torch.ones_like(t),))
    c, s = y[0], y[1]
    dc, ds = dy[0], dy[1]

    T = torch.tensor(legendre_streaming_matrix(net.L_max),
                     dtype=c.dtype, device=c.device)
    dT_c = _closure(c, tau, C)
    dT_s = _closure(s, tau, C)
    src_c = torch.zeros_like(c); src_c[..., 0] = C.reshape(1, -1) * dT_c[:, None]
    src_s = torch.zeros_like(s); src_s[..., 0] = C.reshape(1, -1) * dT_s[:, None]

    vq = (v * q).reshape(1, -1, 1)
    inv_tau = (1.0 / tau).reshape(1, -1, 1)
    R_c = dc + vq * torch.einsum("lj,tmj->tml", T, s) + (c - src_c) * inv_tau
    R_s = ds - vq * torch.einsum("lj,tmj->tml", T, c) + (s - src_s) * inv_tau

    duh_mask = getattr(net, "duhamel_mask", None)
    if duh_mask is None and getattr(net, "duhamel", False):
        duh_mask = torch.ones(len(v), dtype=torch.bool, device=R_c.device)
    if getattr(net, "blend", "banded") == "ugks":
        duh_mask = torch.ones(len(v), dtype=torch.bool, device=R_c.device)
    if duh_mask is not None and bool(duh_mask.any()):
        with torch.no_grad():
            c_hi, s_hi = duhamel_free_coefficients(t, v, tau, C, q,
                                                   net.L_max + 1)
        T_edge = float(legendre_streaming_matrix(net.L_max + 1)
                       [net.L_max, net.L_max + 1])
        vq_flat = (v * q).reshape(1, -1) * duh_mask.to(R_c.dtype).reshape(1, -1)
        R_c[..., net.L_max] = (R_c[..., net.L_max]
                               + vq_flat * T_edge * s_hi[..., net.L_max + 1])
        R_s[..., net.L_max] = (R_s[..., net.L_max]
                               - vq_flat * T_edge * c_hi[..., net.L_max + 1])

    scale = ((1.0 / tau + q * v) * C).reshape(1, -1, 1)
    if ap_scaling:
        M, L1 = len(v), net.L_max + 1
        eps = torch.clamp(q * v * tau, 1e-3, 1.0).reshape(1, -1, 1)
        lmask = torch.ones(1, M, L1, dtype=scale.dtype, device=scale.device)
        lmask[..., 1:] = eps.expand(1, M, L1 - 1)
        scale = scale * lmask
    res = torch.cat([R_c / scale, R_s / scale], dim=2)
    if weight_floor is not None:
        res = res * residual_time_weight(dT_c, weight_floor).reshape(-1, 1, 1)
    if return_carrier:
        sub = getattr(net, "carrier_mask", None)
        sub = (torch.ones(len(v), dtype=C.dtype, device=R_c.device)
               if sub is None else sub.to(C.dtype))
        if float(torch.sum(C * sub)) == 0.0:
            carrier = torch.zeros(R_c.shape[0], 2, dtype=R_c.dtype,
                                  device=R_c.device)
            return res, carrier
        gamma_S = (q ** 2 * torch.sum(C * v ** 2 * tau * sub) / 3.0
                   / torch.sum(C * sub))
        subr = sub.reshape(1, -1)
        carrier = torch.stack([(R_c[..., 0] * subr).sum(dim=1),
                               (R_s[..., 0] * subr).sum(dim=1)],
                              dim=1) / (gamma_S * torch.sum(C * sub))
        return res, carrier
    if not return_macro:
        return res
    gamma_kin = q ** 2 * torch.sum(C * v ** 2 * tau) / 3.0 / torch.sum(C)
    macro = torch.stack([R_c[..., 0].sum(dim=1),
                         R_s[..., 0].sum(dim=1)], dim=1) / (gamma_kin
                                                            * torch.sum(C))
    return res, macro
