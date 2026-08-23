# 5. Experiments

All commands assume the repository root and `pip install -e .`.

## 5.1 Paper profile

```bash
python scripts/run_experiments.py --profile paper --paths 4000
python scripts/run_convergence.py
pytest -q
```

The paper profile is $N_x=41$, $Q_E=Q_I=5$, $N_t=80$, $K=3$, $T=1$.
It is small enough to be a unit of conversational currency in an interview
and large enough that the QVI regions are not grid artefacts.

Default and fine profiles live on `ModelParams.default` / `.fine()`.

## 5.2 What to look at, in order

1. **`figures/value_function.png`**
   $V(0,x,q^E,0)$.  It should be even in the joint flip
   $(x,q^E)\mapsto(-x,-q^E)$, increasing as $|x|$ grows at $q^E=0$
   (the option to capture the basis), and decreasing in $|q^E|$ at $x=0$
   (inventory penalty).
2. **`figures/optimal_spreads.png`**
   Bid widens when $X>0$ or $q^E>0$; ask widens when $X<0$ or $q^E<0$.
   If you do not see this, the cash-term sign is wrong and Theorem 1 has
   been implemented as the rejected Hamiltonian.
3. **`figures/qvi_regions.png`** and **`figures/free_boundary.png`**
   Creation occupies the south-west of the $(q^E,X)$ plane (long basket,
   cheap ETF).  Redemption occupies the north-east.  The continuation
   band around $X=0$ is the no-trade region in the impulse coordinate —
   you still quote.
4. **`figures/impulse_inventory.png`**
   Three slices in $X$.  At $X=0$ the impulse regions should nearly
   vanish except at extreme opposite inventories, where a create/redeem
   is a pure inventory-mix cleanup paying the fee to flatten the penalty.
5. **`figures/mc_wealth.png`**
   Four policies, same Brownian and Poisson seeds:
   - `optimal` — full HJB feedback including QVI,
   - `mm_only` — quotes + hedge, no create/redeem,
   - `naive_arb` — constant half-spreads and a fixed $|X|$ barrier,
   - `hold` — do nothing.
   The ordering of mean wealth should be
   $\text{optimal} \ge \text{mm_only} \ge \text{naive_arb} \gg \text{hold}$
   up to Monte Carlo error.  `hold` has mean near zero and variance from
   the unused basis.
6. **`results/metrics.json`**
   Quoting-identity MAE should be $\sim 10^{-16}$ (the spreads *are* the
   formula).  Parity residual should be a few percent on the paper grid
   and fall under refinement.  Smooth-pasting $|\Delta V_x|$ is a grid
   quantity; quote it, do not worship it.

## 5.3 How a recruiter should stress this

Ask what happens if $C_{\mathrm{fee}}\to 0$.  The continuation band in
$X$ collapses; the scheme remains well-posed because of the inventory
box, but the economic model becomes “always flatten the mix”.  Ask what
happens if $\kappa\to\infty$.  The basis vanishes and the problem
degenerates to a two-asset Avellaneda–Stoikov MM with a costly inventory
swap.  Ask what happens if $\eta\to 0$.  The hedge Hamiltonian explodes
and $q^I$ is slammed to the minimiser of the penalty given $q^E$, i.e.
the AP continuously replicates a creation without paying the fee — which
is exactly why temporary impact has to be there.

Those three degeneracies are the right oral exam.  The pictures in
`results/figures/` are the right written exam.

## 5.4 Calibration note (honest)

$(A,k,\kappa,\sigma_X,\eta,\phi,C_{\mathrm{fee}})$ are not estimated from
a tape in this repository.  A desk-facing sequel would

- estimate $(A,k)$ from fill rates vs. queue position / posted spread
  on the ETF,
- estimate $(\kappa,\sigma_X)$ from the mid-minus-IOPV residual at 1s,
- estimate $\eta$ from the basket's implementation shortfall,
- take $C_{\mathrm{fee}}$ from the prospectus plus borrow,
- take $\phi$ from the desk's overnight limit shadow price.

The control problem is well-posed for any positive such tuple; the code
does not pretend the tuple used in the figures is SPY.
