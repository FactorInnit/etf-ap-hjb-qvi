# 1. Market structure and state dynamics

## 1.1 Authorized participants and the ETF basis

An exchange-traded fund is a listed claim on a basket of constituents.  Let
$\mathbf{S}_t \in \mathbb{R}^d$ be the constituent mid prices and
$\mathbf{w} \in \mathbb{R}^d$ the (fixed, intra-day) creation-unit weights.
The net asset value of one creation unit, expressed per ETF share, is

$$
I_t = \mathbf{w}^\top \mathbf{S}_t.
$$

Write $S_t^E$ for the ETF mid.  The **mispricing spread** (ETF basis)

$$
X_t = S_t^E - I_t
$$

is the only continuous risk factor that an Authorized Participant (AP) cannot
kill by a simultaneous long-short in the ETF and the basket: a long ETF / short
basket position has mark-to-market $X_t$ per share, and conversely.

Creations and redemptions are the institutional mechanism that *does* kill
$X$.  In a creation the AP delivers $K$ baskets to the issuer and receives
$K$ ETF shares; in a redemption the swap is reversed.  Both operations cost a
fixed fee $C_{\mathrm{fee}}$ (wire, creation fee, tracking error, and the
opportunity cost of locking capital for the settlement window).  They are
therefore **impulse controls**, not continuous trades.

Market making in the listed ETF is a **regular control**: the AP posts bid and
ask quotes and is filled by a Poisson flow of end-investors.  Hedging in the
constituent basket is a second regular control, with temporary market impact.

The AP is not a noise trader.  The three controls — quotes, hedge rate, and
create/redeem — are one joint stochastic control problem.  That is the object
of this repository.

## 1.2 Stochastic basis

We take the reduced-form specification

$$
dX_t = -\kappa X_t \, dt + \sigma_X \, dW_t^X, \qquad \kappa>0,\ \sigma_X>0.
$$

This is the unique (weak) solution of an Ornstein–Uhlenbeck SDE; the
invariant law is $\mathcal{N}(0, \sigma_X^2/(2\kappa))$.  Mean reversion
$\kappa$ is the reduced-form footprint of *other* APs, statistical arb
desks, and the issuer's own AP channel.  It is **not** an assumption that
the basis is a traded OU in isolation: it is the residual after the
common factor $I_t$ has been projected out.

Two modelling choices are deliberate.

1. $I_t$ itself is not a state.  Marking the basket at NAV and the ETF at
   $I_t + X_t$ makes the common factor a pure translation of cash.  With
   no discounting and linear terminal wealth, cash and $I$ drop out of the
   value function (see §2.3 of `02-hjb-qvi.md`).
2. $X$ is a diffusion, not a jump-diffusion.  Overnight gaps and NAV
   staleness are real; they are omitted so that the QVI analysis and the
   finite-difference scheme are comparable to the analytic first-order
   conditions.  Extending $X$ to a Lévy driver is a one-line change of
   the generator and is left as a structured exercise.

## 1.3 Inventory and cash

The AP's **physical** state at time $t$ is

$$
(X_t,\ q_t^E,\ q_t^I,\ Y_t) \in \mathbb{R} \times \mathbb{Z} \times \mathbb{Z} \times \mathbb{R},
$$

where $q^E$ is ETF shares, $q^I$ is basket units (in ETF-share
equivalents), and $Y$ is cash.

**Continuous quoting.**  Half-spreads $(\delta_t^b, \delta_t^a)$ are posted
around the ETF mid.  Bid and ask fills are independent Poisson processes
$N^{b}, N^{a}$ with intensities

$$
\lambda(\delta) = A e^{-k \delta}, \qquad A>0,\ k>0,
$$

the standard exponential-intensity microstructure of Avellaneda–Stoikov and
Guéant–Lehalle–Fernandez-Tapia.  On a bid fill the AP **buys** one ETF share;
on an ask fill the AP **sells** one ETF share.

Marking the acquired (resp. sold) share at NAV, the instantaneous cash
increments are

$$
\begin{aligned}
\text{bid fill:}&\quad
\Delta Y = -(S^E - \delta^b) + I = \delta^b - X,\\
\text{ask fill:}&\quad
\Delta Y = +(S^E + \delta^a) - I = \delta^a + X.
\end{aligned}
$$

These signs are economically forced: a cheap ETF ($X<0$) makes buying
profitable even at a thin bid; a rich ETF ($X>0$) makes selling profitable
even at a thin ask.  A Hamiltonian written with the opposite cash sign
produces $\delta^\star = R + 1/k$ and a strictly negative post-optimality
fill value $-1/k$, which cannot be a maximum of a market-making objective.
The derivation is in `03-proofs.md`, Theorem 1.

**Continuous hedging.**  The AP trades the basket at rate $\nu_t \in \mathbb{R}$
with temporary quadratic impact $\psi(\nu)=\eta\nu^2$.  In the interior of
the inventory grid this is the generator term $\nu \partial_{q^I} V - \eta\nu^2$.
On the integer lattice we replace $\partial_{q^I}$ by a centered difference.

**Impulses.**  A sequence of stopping times and sizes $(\tau_j, \xi_j)_{j\ge 1}$
with $\xi_j \in \{+K, -K\}$ implements

$$
\begin{aligned}
\text{creation }(+K):&\quad
(q^E, q^I) \mapsto (q^E+K,\ q^I-K),\quad Y \mapsto Y - C_{\mathrm{fee}},\\
\text{redemption }(-K):&\quad
(q^E, q^I) \mapsto (q^E-K,\ q^I+K),\quad Y \mapsto Y - C_{\mathrm{fee}}.
\end{aligned}
$$

Creations are feasible only when the AP holds the basket; redemptions only
when the AP holds the ETF.  The numerical QVI projection enforces this by
refusing jumps that leave the computational inventory box.

## 1.4 Objective

Over a finite horizon $[0,T]$ (the close, or a slice of the trading day)
the AP maximises expected marked wealth minus a running inventory penalty:

$$
\mathbb{E}\left[
  Y_T + q_T^E S_T^E + q_T^I I_T
  - \phi \int_0^T \left( (q_s^E)^2 + (q_s^I)^2 \right)\, ds
\right].
$$

The quadratic penalty is the reduced form of overnight inventory aversion
and of a VaR / hard-limit shadow price.  It is **not** a risk-neutral
pricing kernel; the whole problem is an optimal-control problem for a
principal with inventory preferences, in the Carr–Jaimungal–Roşu tradition.

The marked terminal wealth expands as

$$
Y_T + q_T^E (I_T + X_T) + q_T^I I_T
= \underbrace{\left(Y_T + (q_T^E+q_T^I)I_T\right)}_{\text{NAV-marked book}}
  + q_T^E X_T.
$$

Because every fill and every hedge is already NAV-marked in cash, the
NAV-marked book is exactly the integral of the cash increments.  The
reduced value function therefore depends on $(t,x,q^E,q^I)$ only, with
terminal condition $V(T,\cdot)=0$ after cash has been accumulated in the
Hamiltonian.  The leftover $q^E X_T$ is priced by the diffusion of $X$
through the generator, not as an extra terminal lump (doing both would
double-count).

## 1.5 What “optimality” means here

The control is admissible if

- $\delta^b,\delta^a$ are predictable and valued in $[\delta_{\min},\delta_{\max}]$,
- $\nu$ is progressively measurable and essentially bounded,
- $(\tau_j)$ is an increasing sequence of stopping times, $\xi_j\in\{+K,-K\}$,
- inventories remain in a finite box $[-Q_E,Q_E]\times[-Q_I,Q_I]$
  (computational, but also a hard desk limit).

Existence of an optimal control for this QVI is standard under linear growth
of the Hamiltonian and a strictly positive fee (Bensoussan–Lions, Øksendal–Sulem).
Uniqueness of the value function in the viscosity sense follows from a
comparison theorem for locally bounded viscosity solutions of HJB-QVIs with
bounded jumps; we do not reprove it.  What we *do* prove, in closed form,
are the first-order conditions for the two regular controls, and the
smooth-pasting condition on the impulse free boundary.  The solver is a
constructive approximation of that value function.
