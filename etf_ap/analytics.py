"""Diagnostics: free boundaries, smooth pasting, grid convergence, identities."""

from __future__ import annotations

import numpy as np

from .hamiltonian import optimal_ask_spread, optimal_bid_spread
from .solver import Solution, _inventory_jumps


def slice_at(sol: Solution, t_frac: float = 0.0, qi: int = 0) -> dict[str, np.ndarray]:
    g = sol.grid
    n = int(round(t_frac * (g.n_t - 1)))
    n = int(np.clip(n, 0, sol.V.shape[-1] - 1))
    k = g.qi_index(qi)
    return {
        "t": g.t[n],
        "x": g.x,
        "qe": g.qe,
        "V": sol.V[:, :, k, n],
        "delta_b": sol.delta_b[:, :, k, n],
        "delta_a": sol.delta_a[:, :, k, n],
        "nu": sol.nu[:, :, k, n],
        "region": sol.region[:, :, k, n],
    }


def free_boundary_x(sol: Solution, t_index: int, qi: int) -> dict[str, np.ndarray]:
    """For each qE, the most conservative |x| still in continuation, plus impulse flags."""
    k = sol.grid.qi_index(qi)
    region = sol.region[:, :, k, t_index]
    qe = sol.grid.qe
    x = sol.grid.x
    create_edge = np.full(qe.size, np.nan)
    redeem_edge = np.full(qe.size, np.nan)
    for j in range(qe.size):
        c = np.where(region[:, j] == 1)[0]
        r = np.where(region[:, j] == -1)[0]
        if c.size:
            create_edge[j] = x[c].max()  # create when ETF is cheap (x negative): least negative in I_c
            create_edge[j] = x[c].min() if np.all(x[c] < 0) else x[c[np.argmin(np.abs(x[c]))]]
        if r.size:
            redeem_edge[j] = x[r].min() if np.all(x[r] > 0) else x[r[np.argmin(np.abs(x[r]))]]
    return {"qe": qe.astype(float), "create_x": create_edge, "redeem_x": redeem_edge}


def smooth_pasting_error(sol: Solution, t_index: int = 0, qi: int = 0) -> dict[str, float]:
    """Compare V_x across the continuation/impulse interface (Theorem 2)."""
    g = sol.grid
    k = g.qi_index(qi)
    V = sol.V[:, :, k, t_index]
    region = sol.region[:, :, k, t_index]
    dx = g.params.dx
    vx = np.gradient(V, dx, axis=0)
    errs = []
    for j in range(g.n_qe):
        r = region[:, j]
        # interfaces: continuation next to create/redeem
        for s in (1, -1):
            interface = np.where((r[:-1] == 0) & (r[1:] == s))[0]
            interface2 = np.where((r[:-1] == s) & (r[1:] == 0))[0]
            for i in np.concatenate([interface, interface2]):
                errs.append(abs(vx[i, j] - vx[min(i + 1, vx.shape[0] - 1), j]))
    arr = np.array(errs) if errs else np.array([np.nan])
    return {
        "n_interfaces": float(arr.size if errs else 0),
        "mean_abs_jump_Vx": float(np.nanmean(arr)),
        "max_abs_jump_Vx": float(np.nanmax(arr)),
    }


def quoting_identity_error(sol: Solution, t_index: int = 0) -> dict[str, float]:
    """Check delta* vs closed form on the continuation region (unclipped)."""
    p = sol.params
    V = sol.V[:, :, :, t_index]
    dplus, dminus, _, can_buy, can_sell = _inventory_jumps(V, sol.grid)
    db_theory = optimal_bid_spread(dplus, sol.grid.x, p)
    da_theory = optimal_ask_spread(dminus, sol.grid.x, p)
    cont = sol.region[:, :, :, t_index] == 0
    bid_m = cont & can_buy & (db_theory > p.delta_min + 1e-12) & (db_theory < p.delta_max - 1e-12)
    ask_m = cont & can_sell & (da_theory > p.delta_min + 1e-12) & (da_theory < p.delta_max - 1e-12)
    if not np.any(bid_m) or not np.any(ask_m):
        return {"bid_mae": np.nan, "ask_mae": np.nan}
    bid_mae = float(np.mean(np.abs(sol.delta_b[:, :, :, t_index][bid_m] - db_theory[bid_m])))
    ask_mae = float(np.mean(np.abs(sol.delta_a[:, :, :, t_index][ask_m] - da_theory[ask_m])))
    return {"bid_mae": bid_mae, "ask_mae": ask_mae}


def value_at_origin(sol: Solution) -> float:
    g = sol.grid
    return float(sol.V[g.i0, g.qe_index(0), g.qi_index(0), 0])


def asymmetry_residual(sol: Solution, t_index: int = 0) -> float:
    """Under (x, qE, qI) -> (-x, -qE, -qI) the value should be even.

    V(t, x, qE, qI) = V(t, -x, -qE, -qI) because creation/redemption and
    bid/ask are dual.  Returns relative L_inf residual.
    """
    V = sol.V[:, :, :, t_index]
    flipped = V[::-1, ::-1, ::-1]
    denom = np.max(np.abs(V)) + 1e-12
    return float(np.max(np.abs(V - flipped)) / denom)


quoting_identity_error = quoting_identity_error
smooth_pasting_error = smooth_pasting_error
