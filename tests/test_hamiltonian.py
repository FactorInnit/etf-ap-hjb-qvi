"""Closed-form identities that do not require the PDE solver."""

import numpy as np

from etf_ap.hamiltonian import (
    fill_hamiltonian_ask,
    fill_hamiltonian_bid,
    hedge_hamiltonian,
    intensity,
    optimal_ask_spread,
    optimal_bid_spread,
    optimal_hedge_rate,
)
from etf_ap.params import ModelParams


def test_bid_foc_matches_finite_difference():
    p = ModelParams()
    x = np.array([-0.1, 0.0, 0.12])
    dplus = np.zeros((3, 1, 1))
    dplus[:, 0, 0] = np.array([-0.04, 0.0, 0.03])
    h, delta = fill_hamiltonian_bid(dplus, x, p)

    def h_of(d):
        cash = d - x[:, None, None]
        return intensity(d, p) * (dplus + cash)

    eps = 1e-6
    dplus_fd = (h_of(delta + eps) - h_of(delta - eps)) / (2 * eps)
    interior = (delta > p.delta_min + 1e-8) & (delta < p.delta_max - 1e-8)
    assert np.all(np.abs(dplus_fd[interior]) < 5e-5)


def test_ask_foc():
    p = ModelParams()
    x = np.array([-0.08, 0.05])
    dminus = np.zeros((2, 1, 1))
    dminus[:, 0, 0] = np.array([0.02, -0.01])
    h, delta = fill_hamiltonian_ask(dminus, x, p)

    def h_of(d):
        cash = d + x[:, None, None]
        return intensity(d, p) * (dminus + cash)

    eps = 1e-6
    dh = (h_of(delta + eps) - h_of(delta - eps)) / (2 * eps)
    interior = (delta > p.delta_min + 1e-8) & (delta < p.delta_max - 1e-8)
    assert np.all(np.abs(dh[interior]) < 5e-5)


def test_spread_signs_are_economic():
    p = ModelParams()
    x = np.linspace(-0.2, 0.2, 9)
    dplus = np.zeros((9, 1, 1))
    dminus = np.zeros((9, 1, 1))
    db = optimal_bid_spread(dplus, x, p).ravel()
    da = optimal_ask_spread(dminus, x, p).ravel()
    # cheaper ETF (smaller x) -> tighter bid; richer ETF -> tighter ask
    assert db[0] < db[-1]
    assert da[-1] < da[0]


def test_hedge_foc():
    p = ModelParams(eta=0.05)
    m = np.array([-0.4, 0.0, 0.25])
    h, nu = hedge_hamiltonian(m, p)
    assert np.allclose(nu, np.clip(m / (2 * p.eta), -p.nu_max, p.nu_max))
    # second derivative of nu*m - eta nu^2 is -2 eta < 0 (maximum)
    eps = 1e-5
    h_plus = (nu + eps) * m - p.eta * (nu + eps) ** 2
    h_minus = (nu - eps) * m - p.eta * (nu - eps) ** 2
    assert np.all(h + 1e-12 >= h_plus)
    assert np.all(h + 1e-12 >= h_minus)


def test_unclipped_bid_formula():
    p = ModelParams()
    x = 0.03
    dplus = -0.02
    expected = 1.0 / p.k_lambda + x - dplus
    got = optimal_bid_spread(np.array([[[dplus]]]), np.array([x]), p)
    assert abs(float(got.ravel()[0]) - expected) < 1e-12
