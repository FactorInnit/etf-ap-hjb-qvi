# 4. Finite-difference IMEX scheme

## 4.1 Grid

\[
x_i = -X_{\max} + i\Delta x,\quad
i=0,\dots,N_x-1,\quad
N_x\text{ odd so that }x=0\text{ is a node}.
\]

Inventories are the integers
\(q^E\in\{-Q_E,\dots,Q_E\}\), \(q^I\in\{-Q_I,\dots,Q_I\}\).
Time is \(t_n = n\Delta t\), \(n=0,\dots,N_t\), \(\Delta t=T/N_t\).

The unknown is \(V^n_{i,j,k} \approx V(t_n, x_i, q^E_j, q^I_k)\).
Terminal condition: \(V^{N_t}\equiv 0\).

## 4.2 Why backward, and why the posted “forward” stencil is wrong

The HJB is a **terminal-value** parabolic equation: \(\partial_t V + \mathcal{L}V = 0\)
with \(V(T)\) given.  Integrating *forward* from \(t=0\) is the pricing
equation for an initial-value problem and will not compute \(V\).  The
scheme in this repository steps \(n=N_t-1,\dots,0\).

An IMEX split that looks like

\[
\frac{V^{n+1}-\tilde V^n}{\Delta t} - \kappa x \,\tilde V_x + \cdots = 0
\]

with \(n\) increasing calendar time is the heat equation run the wrong way
and is unconditionally unstable in the diffusion term.  We never do that.

## 4.3 IMEX operator split

At step \(n\), given \(V^{n+1}\):

**Explicit Hamiltonian.**  Using \(V^{n+1}\), form

\[
\Delta^+ V,\ \Delta^- V,\ \widehat{\partial_{q^I} V}
\]

by one-sided inventory jumps (rejected with value \(-10^6\) at the box
edge, which forces \(\delta=\delta_{\max}\) and kills the intensity) and a
centered difference in \(q^I\).  Insert into the closed-form
\((\delta^{b\star},\delta^{a\star},\nu^\star)\) and evaluate

\[
H^{n+1}
= \mathcal{H}^b+\mathcal{H}^a+\mathcal{H}^\nu
  - \phi\bigl((q^E)^2+(q^I)^2\bigr).
\]

**Implicit OU.**  For each frozen inventory \((j,k)\), solve the tridiagonal
system

\[
\Bigl(\tfrac1{\Delta t} I - \mathcal{G}^{\mathrm{OU}}_{\Delta x}\Bigr) \tilde V^n_{\cdot,j,k}
= \tfrac1{\Delta t} V^{n+1}_{\cdot,j,k} + H^{n+1}_{\cdot,j,k}.
\]

The discrete OU generator uses

- central second differences for \(\tfrac12\sigma_X^2 \partial_{xx}\),
- **upwind** first differences for \(-\kappa x \partial_x\)
  (drift \(\mu=-\kappa x\): backward if \(\mu>0\), forward if \(\mu<0\)).

Upwinding makes the spatial matrix an M-matrix for any \(\Delta t>0\),
which is the monotonicity input to Proposition 5.

Boundary condition in \(x\): first-order extrapolation with the local
advection/diffusion stencil, equivalent to a homogeneous Neumann condition
at \(\pm X_{\max}\).  The OU pull toward zero makes the far field a thin
layer; \(X_{\max}\) is chosen several stationary standard deviations out.

**QVI projection.**

\[
V^n_{i,j,k}
= \max\Bigl\{
    \tilde V^n_{i,j,k},\;
    V^n_{i,j+K,k-K} - C_{\mathrm{fee}},\;
    V^n_{i,j-K,k+K} - C_{\mathrm{fee}}
  \Bigr\},
\]

with out-of-box impulses dropped.  Four Gauss–Seidel sweeps over the
inventory lattice compute the fixed point of this max (Proposition 4).

The region flag is the \(\mathrm{argmax}\) of those three arguments
(\(0,+1,-1\)).

## 4.4 Complexity and the C++ twin

Per time step the Hamiltonian is \(O(N_x N_{q^E} N_{q^I})\) and the Thomas
solves are \(O(N_x)\) per inventory cell, also \(O(N_x N_{q^E} N_{q^I})\).
The paper profile \((N_x,Q_E,Q_I,N_t)=(41,5,5,80)\) is \(\sim 1.5\cdot 10^6\)
cell-steps and runs in a few seconds in NumPy.  The header-only C++ solver
in `cpp/include/etf_ap/solver.hpp` implements the same split and is the
object one would drop into a production library; it is not a binding, it is
the algorithm written twice.

## 4.5 Verification suite

| test | what it actually checks |
|---|---|
| `test_bid_foc_matches_finite_difference` | Theorem 1, not the PDE |
| `test_hedge_foc` | Theorem 2 |
| `test_spread_signs_are_economic` | cheap ETF ⇒ tighter bid |
| `test_quoting_identity` | stored \(\delta^\star\) equals the closed form on \(\mathcal{C}\) |
| `test_parity_symmetry` | §2.5 duality |
| `test_qvi_projection_is_monotone_and_idempotent` | discrete \(\mathcal{M}\) |
| `test_create_when_etf_cheap_and_long_basket` | \(\mathcal{I}_c\) is inhabited |
| `test_ou_stationary_scale_on_paths` | Euler–Maruyama vs \(\sigma^2/(2\kappa)\) |
| `test_optimal_beats_hold_in_expectation` | policy is not noise |

Grid refinement is `scripts/run_convergence.py`.  Smooth pasting is reported
in `results/metrics.json`, not asserted to machine zero.

## 4.6 What this is not

It is not a live market-making engine.  Intensities are calibrated as
dimensionless numbers, not to a specific ETF's L2 book.  \(K\) is a toy
creation unit (3–4 shares) so that the inventory lattice is small enough to
solve on a laptop; a production AP book would use a sparse representation
in the \((q^E+q^I,\ q^E)\) coordinates (net vs. mix) and a coarser \(x\)
grid with local refinement at the free boundary.  The mathematics does not
change.
