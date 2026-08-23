# 3. Proofs and optimality conditions

Throughout, $A>0$, $k>0$, $\eta>0$, and $\lambda(\delta)=A e^{-k\delta}$.
Write $\Delta^+ V = V(q^E+1)-V(q^E)$ and $\Delta^- V = V(q^E-1)-V(q^E)$,
omitting the frozen arguments $(t,x,q^I)$.

---

## Theorem 1 (optimal half-spreads)

On the continuation region, if the maximiser of the bid (resp. ask)
Hamiltonian lies in $(\delta_{\min},\delta_{\max})$, then it is unique and
given by

$$
\delta^{b\star} = \frac{1}{k} + x - \Delta^+ V,
\qquad
\delta^{a\star} = \frac{1}{k} - x - \Delta^- V.
$$

The maximised Hamiltonians equal $A e^{-k\delta^\star}/k = \lambda(\delta^\star)/k$.

### Proof (bid)

The bid contribution to the generator is the scalar function

$$
\mathcal{H}^b(\delta)
= A e^{-k\delta}\left(\Delta^+ V + \delta - x\right), \qquad \delta\in\mathbb{R}.
$$

Differentiate:

$$
\frac{d\mathcal{H}^b}{d\delta}
= -k A e^{-k\delta}\left(\Delta^+ V + \delta - x\right)
  + A e^{-k\delta}.
$$

The exponential is never zero, so the critical-point equation is

$$
-k\left(\Delta^+ V + \delta - x\right) + 1 = 0
\qquad\Longleftrightarrow\qquad
\delta = \frac{1}{k} + x - \Delta^+ V.
$$

The second derivative at that point is

$$
\frac{d^2\mathcal{H}^b}{d\delta^2}
= -k A e^{-k\delta}\left(1\right)
  - k \frac{d\mathcal{H}^b}{d\delta}/\left(A e^{-k\delta}\right)\cdot(\cdots)
  = -k\lambda(\delta) < 0
$$

after the first derivative has been set to zero.  Hence the critical point
is a maximum.  Substituting $\Delta^+ V + \delta^\star - x = 1/k$ into
$\mathcal{H}^b$ yields $\lambda(\delta^\star)/k$.

The ask computation is identical after the replacements
$\Delta^+ V \mapsto \Delta^- V$ and $-x \mapsto +x$.  □

### Remarks

1. **Sign of the cash term.**  If one used $\Delta^+ V + x - \delta$ as in a
   naive transcription of “spread minus mispricing” with the wrong side,
   the FOC would read $\delta^\star = \Delta^+ V + x + 1/k$ and the
   post-optimality fill value would be $-1/k < 0$.  That cannot be optimal
   for a market maker who can always post $\delta=\delta_{\max}$ and
   receive (almost) no fills.  The economically forced cash increments
   $\delta^b - X$ (buy cheap) and $\delta^a + X$ (sell rich) are the
   unique choice compatible with a positive fill value $1/k$.
2. **Inventory skew.**  $\Delta^+ V$ is the marginal value of one extra ETF
   share.  With a quadratic penalty it is typically negative when $q^E>0$,
   which *widens* the bid and *tightens* the ask — the Avellaneda–Stoikov
   skew, now indexed by the ETF basis $x$ as well.
3. **Basis skew.**  $\partial_x \delta^{b\star} = 1$ and
   $\partial_x \delta^{a\star} = -1$ at interior maxima: one-for-one,
   a one-tick richer ETF cheapens the ask and dearens the bid.  This is
   the AP's inventory-agnostic directional quote.

---

## Theorem 2 (optimal hedge rate)

$$
\nu^\star = \mathrm{clip}\left(\frac{\partial_{q^I} V}{2\eta},\, -\nu_{\max},\, \nu_{\max}\right).
$$

The maximised hedge Hamiltonian is
$\left(\partial_{q^I} V\right)^2 / (4\eta)$ in the interior.

### Proof

$\mathcal{H}^\nu(\nu) = \nu\, m - \eta\nu^2$ with $m=\partial_{q^I} V$ is
a strictly concave parabola.  The unconstrained maximiser is $m/(2\eta)$;
projection onto $[-\nu_{\max},\nu_{\max}]$ is optimal by concavity.  □

---

## Theorem 3 (free-boundary partition and smooth pasting)

Define

$$
\begin{aligned}
\mathcal{C}
&= \left\{(t,x,q^E,q^I): V > \max(\mathcal{M}_c V,\ \mathcal{M}_r V)\right\},\\
\mathcal{I}_c
&= \left\{(t,x,q^E,q^I): V = \mathcal{M}_c V \ge \mathcal{M}_r V\right\},\\
\mathcal{I}_r
&= \left\{(t,x,q^E,q^I): V = \mathcal{M}_r V \ge \mathcal{M}_c V\right\}.
\end{aligned}
$$

Assume $V$ is a classical $C^{1,2}$ solution of the QVI in a neighbourhood
of a free-boundary point
$\partial\mathcal{C}\cap\partial\mathcal{I}_c$, and that the boundary is a
$C^1$ graph $x=\Gamma_c(t,q^E,q^I)$.  Then **value matching**

$$
V = \mathcal{M}_c V \qquad\text{on }\partial\mathcal{C}\cap\partial\mathcal{I}_c
$$

holds by continuity of $V$, and **smooth pasting**

$$
\partial_x V
= \partial_x(\mathcal{M}_c V)
\qquad\text{on }\partial\mathcal{C}\cap\partial\mathcal{I}_c
$$

holds.  The analogous statement is true for redemptions.

### Proof (sketch, standard for impulse control)

Value matching is the definition of $\partial\mathcal{I}_c$ plus continuity.

For smooth pasting: on $\mathcal{I}_c$, $V=\mathcal{M}_c V$, and
$\mathcal{M}_c V$ does not depend on a local $x$-derivative identity
other than that of $V(\cdot,q^E+K,q^I-K)$.  If $\partial_x V$ jumped
across $\Gamma_c$, a standard “needle variation” of the first exit time
from $\mathcal{C}$ would produce a first-order gain (Øksendal–Sulem,
Thm. 6.2; or the variational argument of Bensoussan–Lions, Ch. 4).
Equivalently, in the viscosity theory, a $C^1$ test function touching $V$
from above and below at a boundary point forces the two $x$-derivatives
to coincide.

On the discrete grid, smooth pasting is an *a posteriori* diagnostic: we
measure $|V_x^{\mathcal{C}} - V_x^{\mathcal{I}}|$ on cells that straddle
the numerical interface.  First-order spatial discretisation cannot drive
this to machine zero; it must shrink under grid refinement.  That check is
`smooth_pasting_error` in `etf_ap/analytics.py`.

---

## Proposition 4 (no double-jump at a single instant if $C_{\mathrm{fee}}>0$)

Suppose $C_{\mathrm{fee}} > 0$ and the inventory box is finite.  Then an
optimal impulse policy executes at most one of $\{\text{create},\text{redeem}\}$
at any given $(t,x,q^E,q^I)$, and a finite number of impulses on $[0,T]$
almost surely.

### Proof

A create-then-redeem (or reverse) pair at the same state returns the
inventory to itself and deducts $2C_{\mathrm{fee}}>0$, which is strictly
dominated by inaction.  Hence the two intervention regions are disjoint
except possibly on a set of measure zero where both operators equal $V$
(indifference, still resolved by inaction).  The inventory box has finitely
many points, so an impulse chain that does not cycle must stop after at
most $2\max(Q_E,Q_I)/K$ jumps.  Cycles are ruled out by the fee.  □

This is why four QVI projection sweeps are enough in the solver.

---

## Proposition 5 (comparison / monotonicity of the discrete scheme)

The IMEX-plus-projection operator $\mathcal{T}$ of `04-numerics.md` is
monotone: $U\le W$ cellwise implies $\mathcal{T}U\le \mathcal{T}W$.
It is also discounted in the sense that a constant $c$ added to a
function is added to $\mathcal{T}$ (no running discount, but the terminal
condition pins the level).  Barles–Souganidis therefore applies to the
**local** HJB part; the impulse projection is itself a monotone max.
Convergence of the scheme to the unique viscosity solution holds under the
standard technical condition that the comparison principle is valid for the
continuous QVI, which it is for bounded Lipschitz Hamiltonians on a compact
state space (the computational box, with Neumann-type far-field conditions
on $X$).

We do not pretend this is a new PDE theorem.  We do pretend that a trading
desk should refuse to ship a QVI solver that is not monotone.
