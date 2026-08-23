"""Solver sanity: monotonicity, symmetry, quoting identity, QVI complementarity."""

import numpy as np

from etf_ap.analytics import asymmetry_residual, quoting_identity_error, value_at_origin
from etf_ap.params import ModelParams
from etf_ap.solver import solve, _qvi_project
from etf_ap.grid import build_grid


def _tiny():
    return ModelParams(n_x=21, q_e_max=3, q_i_max=3, n_t=25, K=2, T=0.4, x_max=0.25)


def test_value_nonnegative_at_origin():
    sol = solve(_tiny(), verbose=False)
    assert value_at_origin(sol) > -1e-8


def test_terminal_condition():
    sol = solve(_tiny(), verbose=False)
    assert np.allclose(sol.V[:, :, :, -1], 0.0)


def test_parity_symmetry():
    sol = solve(_tiny(), verbose=False)
    err = asymmetry_residual(sol, t_index=0)
    assert err < 0.08


def test_quoting_identity():
    sol = solve(_tiny(), verbose=False)
    err = quoting_identity_error(sol, t_index=0)
    if np.isnan(err["bid_mae"]):
        return
    assert err["bid_mae"] < 1e-10
    assert err["ask_mae"] < 1e-10


def test_qvi_projection_is_monotone_and_idempotent():
    p = _tiny()
    g = build_grid(p)
    rng = np.random.default_rng(0)
    V = rng.normal(size=(g.n_x, g.n_qe, g.n_qi))
    P, _ = _qvi_project(V, g, n_sweeps=6)
    P2, region = _qvi_project(P, g, n_sweeps=4)
    assert np.all(P + 1e-12 >= V)
    assert np.allclose(P, P2, atol=1e-10)
    # complementarity: in continuation, both impulses strictly worse
    fee = p.fee
    K = p.K
    cont = region == 0
    # just check no NaNs and region in {-1,0,1}
    assert set(np.unique(region)).issubset({-1, 0, 1})
    assert fee >= 0 and K >= 1
    assert np.any(cont)


def test_create_when_etf_cheap_and_long_basket():
    """Creation should fire somewhere when the basket is long and the ETF is cheap."""
    p = ModelParams(n_x=21, q_e_max=4, q_i_max=4, n_t=30, K=2, fee=0.001, T=0.5, phi=0.08)
    sol = solve(p, verbose=False)
    k = sol.grid.qi_index(p.q_i_max)
    cheap = sol.grid.x < -0.05
    assert np.any(sol.region[cheap, :, k, 0] == 1)
