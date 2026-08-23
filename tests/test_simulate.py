"""OU discretization and Monte Carlo smoke tests."""

import numpy as np

from etf_ap.params import ModelParams
from etf_ap.simulate import simulate, summarize
from etf_ap.solver import solve


def test_ou_stationary_scale_on_paths():
    p = ModelParams(n_x=21, q_e_max=3, q_i_max=3, n_t=40, T=1.0, kappa=3.0, sigma_x=0.09)
    sol = solve(p, verbose=False)
    batch = simulate(sol, n_paths=2000, seed=1, policy="hold")
    # Var_stat = sigma^2 / (2 kappa)
    var_stat = p.sigma_x**2 / (2.0 * p.kappa)
    emp = float(np.var(batch.x[-1]))
    assert abs(emp - var_stat) / var_stat < 0.35


def test_optimal_beats_hold_in_expectation():
    p = ModelParams(n_x=21, q_e_max=3, q_i_max=3, n_t=30, T=0.5, K=2)
    sol = solve(p, verbose=False)
    opt = summarize(simulate(sol, n_paths=1500, seed=2, policy="optimal"))
    hold = summarize(simulate(sol, n_paths=1500, seed=2, policy="hold"))
    assert opt["mean_wealth"] > hold["mean_wealth"] - 0.01
