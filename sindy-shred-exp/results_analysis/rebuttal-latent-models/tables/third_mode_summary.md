# Third-mode classes of the valid Bayesian-ARGOS latent models (degree 1, stage1 & valid)

Class = position of the 90 % credible interval (5 to 95 % quantiles over 1000 posterior draws) of the
decay rate of the real eigenvalue relative to zero. The interval [0, 0] of a structurally zero eigenvalue
counts as containing zero. Medians unless stated. Decoder ratios: decoded-field rRMSE with one latent
coordinate frozen at its training mean divided by the full decode (`decoder_ablation.csv`).

| quantity | level | decays | grows |
|---|---|---|---|
| models | 72 | 13 | 2 |
| eigenvalue zero by structure | 4 | 0 | 0 |
| decay rate, median (/yr) | -0.004 | +0.558 | -0.102 |
| decay rate, range (/yr) | -0.191 to +1.207 | +0.075 to +1.459 | -0.116 to -0.088 |
| |decay rate| < 0.05 /yr | 51 | 0 | 0 |
| e-folding time >= 20 yr | 62 | 0 | 2 |
| e-folding time < 2 yr (point estimate) | 1 | 7 | 0 |
| e-folding times (yr), decays/grows classes | — | 0.7, 0.7, 1.1, 1.1, 1.2, 1.6, 1.8, 3.6, 4.0, 4.0, 4.8, 5.0, 13.3 | inf, inf |
| period within 1.00 +/- 0.05 yr | 69 | 12 | 2 |
| annual decay sign undetermined | 69 | 11 | 2 |
| forecast RMSE 0-100 wk, median (degC) | 0.82 | 0.85 | 0.74 |
| forecast RMSE 100-200 wk, median (degC) | 0.87 | 0.96 | 0.78 |
| forecast RMSE 200-275 wk, median (degC) | 1.18 | 1.12 | 0.98 |
| forecast RMSE 0-275 wk, median (degC) | 0.93 | 1.03 | 0.83 |
| training latent MSE, median | 0.276 | 0.115 | 0.206 |
| reconstruction rRMSE, median | 0.0443 | 0.0441 | 0.0459 |
| least-needed coordinate ratio, median | 1.14 | 1.05 | 1.10 |
| least-needed coordinate ratio < 1.05 | 13 | 7 | 1 |
| most-needed coordinate ratio, median | 2.15 | 1.93 | 2.24 |

## Fast real modes by the point estimate (e-folding time < 2 yr)

| experiment | decay rate (/yr) | e-folding (yr) | 90 % interval | class |
|---|---|---|---|---|
| 108 | +1.459 | 0.69 | [+0.950, +1.955] | decays |
| 110 | +1.425 | 0.70 | [+1.138, +1.736] | decays |
| 71 | +1.207 | 0.83 | [-0.007, +2.426] | level |
| 135 | +0.911 | 1.10 | [+0.563, +1.278] | decays |
| 48 | +0.895 | 1.12 | [+0.767, +1.030] | decays |
| 18 | +0.814 | 1.23 | [+0.590, +1.033] | decays |
| 64 | +0.642 | 1.56 | [+0.225, +1.066] | decays |
| 77 | +0.558 | 1.79 | [+0.054, +1.081] | decays |

## Structurally zero eigenvalue

experiments 17, 91, 129, 145 (one latent coordinate appears on no right-hand side, so the real eigenvalue is zero in every draw).

## Representatives shown in the Supplementary Information

- experiment 91: level, decay rate -0.000 /yr, 90 % interval [-0.000, +0.000]
- experiment 123: level, decay rate +0.043 /yr, 90 % interval [-0.524, +0.561]
- experiment 18: decays, decay rate +0.814 /yr, 90 % interval [+0.590, +1.033]
- experiment 80: level, decay rate -0.014 /yr, 90 % interval [-0.115, +0.081]
