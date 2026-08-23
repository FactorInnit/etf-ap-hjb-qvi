"""Extracted feedback-control policies and Monte Carlo evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ModelParams
from .solver import Solution


@dataclass
class PathBatch:
    t: np.ndarray
    x: np.ndarray
    qe: np.ndarray
    qi: np.ndarray
    cash: np.ndarray
    wealth: np.ndarray
    n_create: np.ndarray
    n_redeem: np.ndarray
    n_bid: np.ndarray
    n_ask: np.ndarray


def _nearest_index(grid_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = grid_x[1] - grid_x[0]
    idx = np.rint((x - grid_x[0]) / dx).astype(int)
    return np.clip(idx, 0, grid_x.size - 1)


def simulate(
    sol: Solution,
    n_paths: int = 4000,
    seed: int = 7,
    qe0: int = 0,
    qi0: int = 0,
    x0: float = 0.0,
    policy: str = "optimal",
) -> PathBatch:
    """Simulate controlled AP wealth.

    policy:
        optimal   - HJB feedback (spreads, hedge rate, QVI impulses)
        mm_only   - optimal quotes + hedge, impulses disabled
        naive_arb - fixed half-spreads, create/redeem when |x| exceeds a barrier
        hold      - no trading
    """
    rng = np.random.default_rng(seed)
    p = sol.params
    g = sol.grid
    nt = p.n_t
    dt = p.dt

    x = np.full(n_paths, x0, dtype=float)
    qe = np.full(n_paths, qe0, dtype=int)
    qi = np.full(n_paths, qi0, dtype=int)
    cash = np.zeros(n_paths)
    n_create = np.zeros(n_paths, dtype=int)
    n_redeem = np.zeros(n_paths, dtype=int)
    n_bid = np.zeros(n_paths, dtype=int)
    n_ask = np.zeros(n_paths, dtype=int)

    x_hist = np.zeros((nt + 1, n_paths))
    qe_hist = np.zeros((nt + 1, n_paths), dtype=int)
    qi_hist = np.zeros((nt + 1, n_paths), dtype=int)
    cash_hist = np.zeros((nt + 1, n_paths))
    wealth_hist = np.zeros((nt + 1, n_paths))
    x_hist[0] = x
    qe_hist[0] = qe
    qi_hist[0] = qi

    naive_barrier = 0.45 * p.x_max
    naive_spread = 1.0 / p.k_lambda

    for n in range(nt):
        ix = _nearest_index(g.x, x)
        jq = np.clip(qe - int(g.qe[0]), 0, g.n_qe - 1)
        kq = np.clip(qi - int(g.qi[0]), 0, g.n_qi - 1)

        if policy == "hold":
            db = np.full(n_paths, p.delta_max)
            da = np.full(n_paths, p.delta_max)
            nu = np.zeros(n_paths)
            do_c = np.zeros(n_paths, dtype=bool)
            do_r = np.zeros(n_paths, dtype=bool)
        elif policy == "naive_arb":
            db = np.full(n_paths, naive_spread)
            da = np.full(n_paths, naive_spread)
            nu = np.clip(-qi.astype(float) * 0.25, -p.nu_max, p.nu_max)
            do_c = (x < -naive_barrier) & (qi >= p.K) & (qe + p.K <= p.q_e_max)
            do_r = (x > naive_barrier) & (qe >= p.K) & (qi + p.K <= p.q_i_max)
        elif policy == "mm_only":
            db = sol.delta_b[ix, jq, kq, n]
            da = sol.delta_a[ix, jq, kq, n]
            nu = sol.nu[ix, jq, kq, n]
            do_c = np.zeros(n_paths, dtype=bool)
            do_r = np.zeros(n_paths, dtype=bool)
        else:
            db = sol.delta_b[ix, jq, kq, n]
            da = sol.delta_a[ix, jq, kq, n]
            nu = sol.nu[ix, jq, kq, n]
            reg = sol.region[ix, jq, kq, n]
            do_c = (reg == 1) & (qi >= p.K) & (qe + p.K <= p.q_e_max)
            do_r = (reg == -1) & (qe >= p.K) & (qi + p.K <= p.q_i_max)

        if np.any(do_c):
            qe[do_c] += p.K
            qi[do_c] -= p.K
            cash[do_c] -= p.fee
            n_create[do_c] += 1
        if np.any(do_r):
            qe[do_r] -= p.K
            qi[do_r] += p.K
            cash[do_r] -= p.fee
            n_redeem[do_r] += 1

        if policy == "hold":
            hit_b = np.zeros(n_paths, dtype=bool)
            hit_a = np.zeros(n_paths, dtype=bool)
        else:
            lam_b = p.A * np.exp(-p.k_lambda * db)
            lam_a = p.A * np.exp(-p.k_lambda * da)
            hit_b = rng.random(n_paths) < (1.0 - np.exp(-lam_b * dt))
            hit_a = rng.random(n_paths) < (1.0 - np.exp(-lam_a * dt))
            hit_b &= qe < p.q_e_max
            hit_a &= qe > -p.q_e_max

        cash[hit_b] += db[hit_b] - x[hit_b]
        qe[hit_b] += 1
        n_bid[hit_b] += 1
        cash[hit_a] += da[hit_a] + x[hit_a]
        qe[hit_a] -= 1
        n_ask[hit_a] += 1

        cash -= p.eta * nu * nu * dt
        qi = np.clip(np.rint(qi.astype(float) + nu * dt).astype(int), -p.q_i_max, p.q_i_max)

        x = x + (-p.kappa * x) * dt + p.sigma_x * np.sqrt(dt) * rng.standard_normal(n_paths)
        x = np.clip(x, -p.x_max, p.x_max)

        x_hist[n + 1] = x
        qe_hist[n + 1] = qe
        qi_hist[n + 1] = qi
        cash_hist[n + 1] = cash
        wealth_hist[n + 1] = cash + qe.astype(float) * x

    wealth_hist[0] = cash_hist[0] + qe_hist[0].astype(float) * x_hist[0]
    return PathBatch(
        t=g.t,
        x=x_hist,
        qe=qe_hist,
        qi=qi_hist,
        cash=cash_hist,
        wealth=wealth_hist,
        n_create=n_create,
        n_redeem=n_redeem,
        n_bid=n_bid,
        n_ask=n_ask,
    )


def summarize(batch: PathBatch) -> dict[str, float]:
    w = batch.wealth[-1]
    return {
        "mean_wealth": float(np.mean(w)),
        "std_wealth": float(np.std(w)),
        "sharpe_like": float(np.mean(w) / (np.std(w) + 1e-12)),
        "p05": float(np.quantile(w, 0.05)),
        "p50": float(np.quantile(w, 0.50)),
        "p95": float(np.quantile(w, 0.95)),
        "mean_creates": float(np.mean(batch.n_create)),
        "mean_redeems": float(np.mean(batch.n_redeem)),
        "mean_bid_fills": float(np.mean(batch.n_bid)),
        "mean_ask_fills": float(np.mean(batch.n_ask)),
        "mean_|qE|_T": float(np.mean(np.abs(batch.qe[-1]))),
        "mean_|qI|_T": float(np.mean(np.abs(batch.qi[-1]))),
    }


PathBatch = PathBatch
