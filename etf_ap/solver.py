"""Backward HJB-QVI solver: implicit OU + implicit Poisson fills + QVI projection.

Calendar-time HJB on the continuation region:

    dV/dt + L^{delta*, nu*} V - phi ((q^E)^2 + (q^I)^2) = 0,   V(T) = 0

An *explicit* treatment of the fill generator is unstable once A Δt ≳ 1
(the intensity at a tight quote).  Quotes are frozen from V^{n+1}; the
resulting linear jump operator is inverted by Gauss–Seidel on the inventory
lattice, each slice an x-tridiagonal OU solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid, build_grid
from .hamiltonian import hedge_hamiltonian, intensity, optimal_ask_spread, optimal_bid_spread
from .params import ModelParams


@dataclass
class Solution:
    params: ModelParams
    grid: Grid
    V: np.ndarray  # (n_x, n_qe, n_qi, n_t+1)
    delta_b: np.ndarray
    delta_a: np.ndarray
    nu: np.ndarray
    region: np.ndarray  # 0 continuation, +1 create, -1 redeem
    residual: np.ndarray


def _inventory_jumps(V: np.ndarray, grid: Grid):
    dplus = np.zeros_like(V)
    dminus = np.zeros_like(V)
    dqi = np.zeros_like(V)
    can_buy = np.zeros(V.shape, dtype=bool)
    can_sell = np.zeros(V.shape, dtype=bool)
    dplus[:, :-1, :] = V[:, 1:, :] - V[:, :-1, :]
    dminus[:, 1:, :] = V[:, :-1, :] - V[:, 1:, :]
    can_buy[:, :-1, :] = True
    can_sell[:, 1:, :] = True
    dqi[:, :, 1:-1] = 0.5 * (V[:, :, 2:] - V[:, :, :-2])
    dqi[:, :, 0] = V[:, :, 1] - V[:, :, 0]
    dqi[:, :, -1] = V[:, :, -1] - V[:, :, -2]
    return dplus, dminus, dqi, can_buy, can_sell


# Alias used by analytics / tests.
_inventory_jumps = _inventory_jumps


def _ou_tridiagonal(grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = grid.params
    x = grid.x
    dx, dt = p.dx, p.dt
    n = x.size
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    mu = -p.kappa * x
    alpha = p.sigma_x**2 / (2.0 * dx * dx)
    for i in range(n):
        am = abs(mu[i]) / dx
        if i == 0:
            diag[0] = 1.0 / dt + 2.0 * alpha + am
            upper[0] = -2.0 * alpha - am
            continue
        if i == n - 1:
            diag[-1] = 1.0 / dt + 2.0 * alpha + am
            lower[-1] = -2.0 * alpha - am
            continue
        if mu[i] >= 0.0:
            adv_i, adv_im1, adv_ip1 = mu[i] / dx, -mu[i] / dx, 0.0
        else:
            adv_i, adv_im1, adv_ip1 = -mu[i] / dx, 0.0, mu[i] / dx
        lower[i] = -adv_im1 - alpha
        diag[i] = -adv_i + 2.0 * alpha + 1.0 / dt
        upper[i] = -adv_ip1 - alpha
    return lower, diag, upper


def _solve_tridiag(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = rhs.shape[0]
    d = rhs.copy()
    extra_shape = d.shape[1:]
    a = np.broadcast_to(lower.reshape((n,) + (1,) * len(extra_shape)), d.shape).copy()
    c = np.broadcast_to(upper.reshape((n,) + (1,) * len(extra_shape)), d.shape).copy()
    if diag.ndim == 1:
        b = np.broadcast_to(diag.reshape((n,) + (1,) * len(extra_shape)), d.shape).copy()
    else:
        b = diag.copy()
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]
    x = np.empty_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _qvi_project(V: np.ndarray, grid: Grid, n_sweeps: int = 4) -> tuple[np.ndarray, np.ndarray]:
    p = grid.params
    K = p.K
    orig = V.copy()
    out = V.copy()
    n_qe, n_qi = grid.n_qe, grid.n_qi
    for _ in range(n_sweeps):
        create = np.full_like(out, -np.inf)
        redeem = np.full_like(out, -np.inf)
        if K < n_qe and K < n_qi:
            create[:, : n_qe - K, K:] = out[:, K:, : n_qi - K] - p.fee
            redeem[:, K:, : n_qi - K] = out[:, : n_qe - K, K:] - p.fee
        out = np.maximum(np.maximum(out, create), redeem)
    create = np.full_like(out, -np.inf)
    redeem = np.full_like(out, -np.inf)
    if K < n_qe and K < n_qi:
        create[:, : n_qe - K, K:] = out[:, K:, : n_qi - K] - p.fee
        redeem[:, K:, : n_qi - K] = out[:, : n_qe - K, K:] - p.fee
    # Classify vs the *pre-projection* value so V = M V does not look like continuation.
    stacked = np.stack([orig, create, redeem], axis=0)
    choice = np.argmax(stacked, axis=0)
    region = np.where(choice == 1, 1, np.where(choice == 2, -1, 0)).astype(np.int8)
    return out, region


def _controls(V_slice: np.ndarray, grid: Grid):
    p = grid.params
    dplus, dminus, dqi, can_buy, can_sell = _inventory_jumps(V_slice, grid)
    db = np.where(can_buy, optimal_bid_spread(dplus, grid.x, p), p.delta_max)
    da = np.where(can_sell, optimal_ask_spread(dminus, grid.x, p), p.delta_max)
    lb = np.where(can_buy, intensity(db, p), 0.0)
    la = np.where(can_sell, intensity(da, p), 0.0)
    cash_b = db - grid.x[:, None, None]
    cash_a = da + grid.x[:, None, None]
    hh, nu = hedge_hamiltonian(dqi, p)
    je = grid.qe[None, :, None].astype(float)
    ki = grid.qi[None, None, :].astype(float)
    penalty = p.phi * (je**2 + ki**2)
    return db, da, nu, lb, la, cash_b, cash_a, hh, penalty


def _implicit_continuation(V_next: np.ndarray, grid: Grid, lower, diag, upper, n_sweeps: int = 5):
    p = grid.params
    dt = p.dt
    db, da, nu, lb, la, cash_b, cash_a, hh, penalty = _controls(V_next, grid)
    base = V_next / dt + lb * cash_b + la * cash_a + hh - penalty
    U = V_next.copy()
    extra = lb + la
    for _ in range(n_sweeps):
        nbr = np.zeros_like(U)
        nbr[:, :-1, :] += lb[:, :-1, :] * U[:, 1:, :]
        nbr[:, 1:, :] += la[:, 1:, :] * U[:, :-1, :]
        diag3 = diag[:, None, None] + extra
        U = _solve_tridiag(lower, diag3, upper, base + nbr)
    return U, db, da, nu


def solve(params: ModelParams | None = None, verbose: bool = True) -> Solution:
    params = params or ModelParams()
    grid = build_grid(params)
    nx, nqe, nqi, nt = grid.n_x, grid.n_qe, grid.n_qi, params.n_t

    V = np.zeros((nx, nqe, nqi, nt + 1))
    delta_b = np.zeros((nx, nqe, nqi, nt + 1))
    delta_a = np.zeros((nx, nqe, nqi, nt + 1))
    nu = np.zeros((nx, nqe, nqi, nt + 1))
    region = np.zeros((nx, nqe, nqi, nt + 1), dtype=np.int8)
    residual = np.zeros(nt)

    lower, diag, upper = _ou_tridiagonal(grid)

    for n in range(nt - 1, -1, -1):
        v_cont, db, da, nv = _implicit_continuation(V[:, :, :, n + 1], grid, lower, diag, upper)
        v_proj, reg = _qvi_project(v_cont, grid)
        db, da, nv, *_ = _controls(v_proj, grid)
        V[:, :, :, n] = v_proj
        delta_b[:, :, :, n] = db
        delta_a[:, :, :, n] = da
        nu[:, :, :, n] = nv
        region[:, :, :, n] = reg
        residual[n] = float(np.max(np.abs(v_proj - v_cont)))
        if verbose and (n % max(nt // 8, 1) == 0 or n == 0):
            vmax = float(np.max(np.abs(v_proj)))
            print(f"  t={grid.t[n]:.3f}  ||V||_inf={vmax:.6f}  QVI gap={residual[n]:.4e}")

    return Solution(
        params=params,
        grid=grid,
        V=V,
        delta_b=delta_b,
        delta_a=delta_a,
        nu=nu,
        region=region,
        residual=residual,
    )
