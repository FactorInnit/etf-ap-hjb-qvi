"""Closed-form Hamiltonians for quoting and hedging.

The first-order conditions below are the economically consistent versions of
Theorems 1-2 in ``docs/03-proofs.md``.  In particular the fill P&L signs are

    bid (AP buys 1 ETF):   cash increment =  delta^b - X
    ask (AP sells 1 ETF):  cash increment =  delta^a + X

so that a cheap ETF (X < 0) is attractive on the bid and a rich ETF (X > 0)
is attractive on the ask.  The posted Hamiltonian that used (Delta V + x - delta)
implies a negative post-optimality fill value of -1/k and is rejected.
"""

from __future__ import annotations

import numpy as np

from .params import ModelParams


def optimal_bid_spread(delta_plus_v: np.ndarray, x: np.ndarray, p: ModelParams) -> np.ndarray:
    """delta^{b*} = 1/k + x - Delta^+ V, clipped to [delta_min, delta_max]."""
    raw = (1.0 / p.k_lambda) + x[:, None, None] - delta_plus_v
    return np.clip(raw, p.delta_min, p.delta_max)


def optimal_ask_spread(delta_minus_v: np.ndarray, x: np.ndarray, p: ModelParams) -> np.ndarray:
    """delta^{a*} = 1/k - x - Delta^- V, clipped to [delta_min, delta_max]."""
    raw = (1.0 / p.k_lambda) - x[:, None, None] - delta_minus_v
    return np.clip(raw, p.delta_min, p.delta_max)


def intensity(delta: np.ndarray, p: ModelParams) -> np.ndarray:
    return p.A * np.exp(-p.k_lambda * delta)


def fill_hamiltonian_bid(delta_plus_v: np.ndarray, x: np.ndarray, p: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns (H^b, delta^{b*})."""
    delta = optimal_bid_spread(delta_plus_v, x, p)
    cash = delta - x[:, None, None]
    h = intensity(delta, p) * (delta_plus_v + cash)
    return h, delta


def fill_hamiltonian_ask(delta_minus_v: np.ndarray, x: np.ndarray, p: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns (H^a, delta^{a*})."""
    delta = optimal_ask_spread(delta_minus_v, x, p)
    cash = delta + x[:, None, None]
    h = intensity(delta, p) * (delta_minus_v + cash)
    return h, delta


def optimal_hedge_rate(marginal_qi: np.ndarray, p: ModelParams) -> np.ndarray:
    """nu* = clip( m / (2 eta), +/- nu_max ) from max_nu [nu * m - eta nu^2]."""
    nu = marginal_qi / (2.0 * p.eta)
    return np.clip(nu, -p.nu_max, p.nu_max)


def hedge_hamiltonian(marginal_qi: np.ndarray, p: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    nu = optimal_hedge_rate(marginal_qi, p)
    h = nu * marginal_qi - p.eta * nu * nu
    return h, nu
