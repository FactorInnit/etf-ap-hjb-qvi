# 2. HJB-QVI formulation

## 2.1 Generator of the controlled diffusion-plus-jumps

On the continuation region the state is a piecewise-deterministic Markov
process: $X$ diffuses, $q^I$ is advected by $\nu$, and $q^E$ jumps
on Poisson fills.  For a test function $v(t,x,q^E,q^I)$ the infinitesimal
generator of the **regular** controls is

$$
\begin{aligned}
\mathcal{L}^{\boldsymbol{\delta},\nu} v
&= -\kappa x\, \partial_x v
   + \tfrac12 \sigma_X^2 \partial_{xx} v
   + \nu\, \partial_{q^I} v
   - \eta\nu^2 \\
&\quad + \lambda(\delta^b)\Delta^b v
   + \lambda(\delta^a)\Delta^a v,
\end{aligned}
$$

where the fill operators **include cash**:

$$
\begin{aligned}
\Delta^b v
&= v(t,x,q^E+1,q^I) - v(t,x,q^E,q^I) + (\delta^b - x),\\
\Delta^a v
&= v(t,x,q^E-1,q^I) - v(t,x,q^E,q^I) + (\delta^a + x).
\end{aligned}
$$

(The discrete inventory derivative $\partial_{q^I} v$ is a centered
difference on the computational grid.)

## 2.2 Impulse operators

$$
\begin{aligned}
\mathcal{M}_c v(t,x,q^E,q^I)
&= v(t,x,q^E+K,q^I-K) - C_{\mathrm{fee}},\\
\mathcal{M}_r v(t,x,q^E,q^I)
&= v(t,x,q^E-K,q^I+K) - C_{\mathrm{fee}}.
\end{aligned}
$$

They are non-local in inventory and local in $(t,x)$.  Because a creation
does not change $X$ instantaneously — the AP has swapped two claims that
differ by $X$, and the cash is only the fee — the mispricing is *transferred
from the basis into the inventory mix*, not cancelled in the SDE.  Cancelling
the basis in P&L terms is exactly the statement that $q^E X + q^I \cdot 0$
becomes $(q^E+K)X + (q^I-K)\cdot 0$ while the NAV-marked books offset; the
change in marked wealth is $K X - C_{\mathrm{fee}}$ plus the change in
continuation value.  That $KX$ is already inside $v(\cdot,q^E+K,\cdot)$
through future quoting and mean reversion, which is why $\mathcal{M}_c$
does not contain an extra $KX$ term.

## 2.3 Value function and QVI

$$
V(t,x,q^E,q^I)
= \sup_{(\boldsymbol{\delta},\nu,\boldsymbol{\tau},\boldsymbol{\xi})}
  \mathbb{E}_{t,x,q^E,q^I}
  \Biggl[
    \int_t^T \Bigl(
        \text{cash rate of }\mathcal{L}
        - \phi\bigl((q_s^E)^2+(q_s^I)^2\bigr)
    \Bigr) ds
  \Biggr],
$$

with $V(T,\cdot)\equiv 0$.  Cash of impulses is applied at $\tau_j$.

The dynamic programming principle for combined regular and impulse control
yields the Hamilton–Jacobi–Bellman quasi-variational inequality

$$
\max\Bigl\{
  \partial_t V + \sup_{\delta,\nu}\mathcal{L}^{\boldsymbol{\delta},\nu} V
    - \phi\bigl((q^E)^2+(q^I)^2\bigr),
  \ \mathcal{M}_c V - V,
  \ \mathcal{M}_r V - V
\Bigr\} = 0
$$

on $[0,T)\times\mathbb{R}\times\mathbb{Z}^2$, in the viscosity sense, with
$V(T,\cdot)=0$.

The three arguments of the $\max$ are mutually complementary:

| region | name | characterisation |
|---|---|---|
| $\mathcal{C}$ | continuation | $V > \max(\mathcal{M}_c V,\mathcal{M}_r V)$ and the HJB holds as an equality |
| $\mathcal{I}_c$ | create | $V = \mathcal{M}_c V \ge \mathcal{M}_r V$ |
| $\mathcal{I}_r$ | redeem | $V = \mathcal{M}_r V \ge \mathcal{M}_c V$ |

On $\mathcal{I}_c$ the AP creates immediately; the HJB inequality is
dropped.  A strictly positive fee plus the inventory box prevent infinite
impulse cascades.

## 2.4 Hamiltonian reduction

The supremum over regular controls separates:

$$
\sup_{\delta,\nu}\mathcal{L}^{\boldsymbol{\delta},\nu} V
= \mathcal{G}^{\mathrm{OU}} V
  + \mathcal{H}^b(\Delta_{q^E}^+ V, x)
  + \mathcal{H}^a(\Delta_{q^E}^- V, x)
  + \mathcal{H}^\nu(\partial_{q^I} V),
$$

where $\mathcal{G}^{\mathrm{OU}} = -\kappa x\partial_x + \tfrac12\sigma_X^2\partial_{xx}$
and the three Hamiltonians are maximised in closed form (Theorems 1–2 in
`03-proofs.md`):

$$
\begin{aligned}
\delta^{b\star}
&= \frac{1}{k} + x - \Delta_{q^E}^+ V,\\
\delta^{a\star}
&= \frac{1}{k} - x - \Delta_{q^E}^- V,\\
\nu^\star
&= \frac{1}{2\eta}\,\partial_{q^I} V,
\end{aligned}
$$

then clipped to the admissible box.  Substituting back produces a
**semilinear** parabolic QVI, which is what the IMEX scheme solves.

## 2.5 Duality / parity

The map

$$
(x,q^E,q^I,\delta^b,\delta^a,\nu,\xi)
\;\mapsto\;
(-x,-q^E,-q^I,\delta^a,\delta^b,-\nu,-\xi)
$$

leaves the objective invariant.  Consequently

$$
V(t,x,q^E,q^I) = V(t,-x,-q^E,-q^I).
$$

This is a hard numerical diagnostic: a solver bug that treats bids and asks
asymmetrically, or that offsets the $x$-grid, shows up as a parity residual.
The test `test_parity_symmetry` enforces it.
