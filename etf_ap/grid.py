"""State grid for (x, q^E, q^I). Time is stored as an extra axis on V."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ModelParams


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    qe: np.ndarray
    qi: np.ndarray
    t: np.ndarray
    params: ModelParams

    @property
    def n_x(self) -> int:
        return self.x.size

    @property
    def n_qe(self) -> int:
        return self.qe.size

    @property
    def n_qi(self) -> int:
        return self.qi.size

    @property
    def n_t(self) -> int:
        return self.t.size

    @property
    def i0(self) -> int:
        """Index of x = 0."""
        return int(np.argmin(np.abs(self.x)))

    def qe_index(self, q: int) -> int:
        return int(q - self.qe[0])

    def qi_index(self, q: int) -> int:
        return int(q - self.qi[0])

    def in_qe(self, q: int) -> bool:
        return self.qe[0] <= q <= self.qe[-1]

    def in_qi(self, q: int) -> bool:
        return self.qi[0] <= q <= self.qi[-1]


def build_grid(params: ModelParams) -> Grid:
    x = np.linspace(-params.x_max, params.x_max, params.n_x)
    qe = np.arange(-params.q_e_max, params.q_e_max + 1, dtype=int)
    qi = np.arange(-params.q_i_max, params.q_i_max + 1, dtype=int)
    t = np.linspace(0.0, params.T, params.n_t + 1)
    return Grid(x=x, qe=qe, qi=qi, t=t, params=params)
