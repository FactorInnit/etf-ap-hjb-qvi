# Experiment metrics

Profile `paper`. Value at the origin V(0,0,0,0) = **7.9028**.

| policy | mean wealth | std | 5% | 95% | creates | redeems |
|---|---:|---:|---:|---:|---:|---:|
| optimal | 6.2588 | 0.5650 | 5.3499 | 7.2040 | 0.013 | 0.226 |
| mm_only | 6.2225 | 0.5722 | 5.2631 | 7.1600 | 0.000 | 0.000 |
| naive_arb | 6.0538 | 0.6137 | 5.0350 | 7.0506 | 0.000 | 0.000 |
| hold | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |

Quoting identity MAE (bid/ask): 0.00e+00 / 0.00e+00.
Parity residual: 2.18e-13.

Monte Carlo uses 2000 paths, common seed. Ranking is optimal > mm_only > naive_arb >> hold.
