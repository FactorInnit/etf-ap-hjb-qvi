"""Grid-refinement study for the HJB-QVI value at the origin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from etf_ap.analytics import value_at_origin
from etf_ap.params import ModelParams
from etf_ap.solver import solve


def main() -> None:
    rows = []
    for n_x, n_t in ((21, 40), (41, 80), (61, 120)):
        p = ModelParams(n_x=n_x, n_t=n_t, q_e_max=4, q_i_max=4, K=2, T=0.75)
        sol = solve(p, verbose=False)
        rows.append(
            {
                "n_x": n_x,
                "n_t": n_t,
                "dx": p.dx,
                "dt": p.dt,
                "V0": value_at_origin(sol),
                "V_inf": float(np.max(np.abs(sol.V[:, :, :, 0]))),
            }
        )
        print(rows[-1])

    v = [r["V0"] for r in rows]
    r12 = abs(v[1] - v[0])
    r23 = abs(v[2] - v[1])
    order = float(np.log(r12 / max(r23, 1e-16)) / np.log(2.0))
    out = {"rows": rows, "empirical_order_V0": order}
    Path("results").mkdir(exist_ok=True)
    Path("results/convergence.json").write_text(json.dumps(out, indent=2))
    print("empirical order ~", order)


if __name__ == "__main__":
    main()
