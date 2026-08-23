"""Model parameters for the Authorized Participant HJB-QVI problem."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelParams:
    """Structural parameters of the ETF AP control problem.

    Spread dynamics
        dX_t = -kappa * X_t dt + sigma_x dW_t

    Fill intensity (Avellaneda-Stoikov / Guéant)
        lambda(delta) = A * exp(-k_lambda * delta)

    Temporary impact on the basket trading rate nu
        psi(nu) = eta * nu^2

    Inventory running penalty
        phi * ((q^E)^2 + (q^I)^2)
    """

    kappa: float = 2.0
    sigma_x: float = 0.08
    A: float = 140.0
    k_lambda: float = 12.0
    eta: float = 0.02
    phi: float = 0.015
    fee: float = 0.08
    K: int = 4
    T: float = 1.0
    x_max: float = 0.35
    n_x: int = 81
    q_e_max: int = 8
    q_i_max: int = 8
    n_t: int = 160
    delta_min: float = 1e-4
    delta_max: float = 0.50
    nu_max: float = 25.0

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.sigma_x <= 0:
            raise ValueError("OU parameters must be positive")
        if self.A <= 0 or self.k_lambda <= 0:
            raise ValueError("intensity parameters must be positive")
        if self.eta <= 0 or self.phi < 0 or self.fee < 0:
            raise ValueError("cost parameters must be non-negative with eta > 0")
        if self.K <= 0:
            raise ValueError("creation unit K must be a positive integer")
        if self.n_x < 5 or self.n_x % 2 == 0:
            raise ValueError("n_x must be odd and >= 5 so that x=0 is on the grid")
        if self.n_t < 2:
            raise ValueError("n_t must be >= 2")

    @property
    def dx(self) -> float:
        return 2.0 * self.x_max / (self.n_x - 1)

    @property
    def dt(self) -> float:
        return self.T / self.n_t

    def paper(self) -> ModelParams:
        """Faster grid used for CI / first-pass figures."""
        return replace(self, n_x=41, q_e_max=5, q_i_max=5, n_t=80, K=3)

    def fine(self) -> ModelParams:
        """Denser grid for convergence tables."""
        return replace(self, n_x=121, q_e_max=8, q_i_max=8, n_t=240)


def default_params() -> ModelParams:
    return ModelParams()


default_params = default_params
