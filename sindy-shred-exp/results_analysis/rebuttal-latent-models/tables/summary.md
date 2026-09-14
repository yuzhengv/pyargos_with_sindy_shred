# Identified latent models: 150 experiments

Stage 1 (rRMSE <= 0.05, eval mode): 116 of 150 experiments pass.

| model | runs | stage1 | stage1 & valid | spectrally stable (stage1 & valid) | affine-linear | mean terms | runs with nonlinear terms |
|---|---|---|---|---|---|---|---|
| ba_deg1 | 150 | 116 | 87 | 31 | 87 | 10.68 | 0 |
| sindy_deg1 | 150 | 116 | 67 | 39 | 67 | 8.33 | 0 |
| ba_deg3 | 150 | 116 | 51 | 0 | 0 | 39.92 | 51 |
| sindy_deg3 | 150 | 116 | 44 | 0 | 0 | 40.34 | 44 |

Paired subset (stage1, both degree-1 models valid): 53 experiments.

Stage 1 evaluated with one dropout-active draw instead (as the submitted batch did): 107 pass; passing under both: 107.

**Validity and spectral stability.** Invalid = training latent MSE > 1 on some dimension (or a non-finite simulation); an affine-linear model with an eigenvalue of positive real part grows exponentially.

- ba_deg1: 29 invalid stage1 runs; max real eigenvalue > 0 in 29 of them (median +0.326 /yr); non-finite simulations 0
- sindy_deg1: 49 invalid stage1 runs; max real eigenvalue > 0 in 44 of them (median +0.249 /yr); non-finite simulations 0
- ba_deg3: 65 invalid stage1 runs; non-finite simulations 41
- sindy_deg3: 72 invalid stage1 runs; non-finite simulations 60
- ba_deg3: of 51 valid runs, test-period forecast non-finite in 14, test latent MSE > 1 in 7, decoded RMSE > 5 degC in 4 (validity is judged on the training period only)
- sindy_deg3: of 44 valid runs, test-period forecast non-finite in 7, test latent MSE > 1 in 3, decoded RMSE > 5 degC in 1 (validity is judged on the training period only)

**Mode structure** of the stage1 & valid affine-linear models:

- ba_deg1 (n = 87): one oscillatory pair + one real mode in 87 runs; period within 1.00 +/- 0.05 yr in 83; annual mode decaying (real part < 0) in 56; real mode decaying in 43, fast (e-folding < 2 yr) in 8, near-neutral (|rate| < 0.05 /yr) in 51
- sindy_deg1 (n = 67): one oscillatory pair + one real mode in 67 runs; period within 1.00 +/- 0.05 yr in 53; annual mode decaying (real part < 0) in 50; real mode decaying in 47, fast (e-folding < 2 yr) in 13, near-neutral (|rate| < 0.05 /yr) in 27
- ba_deg1 posterior: annual mode decaying with P > 0.95 in 5 runs, growing with P > 0.95 in 1; real mode decaying with P > 0.95 (q05 > 0) in 13 of 87

**period (yr)**, median [5 %, 95 %] over stage1 & valid runs:

- ba_deg1: 1.0174 [0.9975, 1.0431] (n = 87)
- sindy_deg1: 1.0303 [1.0033, 1.0748] (n = 67)

**annual-mode decay rate (/yr; negative = growing)**, median [5 %, 95 %] over stage1 & valid runs:

- ba_deg1: 0.0044 [-0.0116, 0.0743] (n = 87)
- sindy_deg1: 0.0196 [-0.0347, 0.3049] (n = 67)

**real-mode decay rate (/yr; negative = growing)**, median [5 %, 95 %] over stage1 & valid runs:

- ba_deg1: 0.0000 [-0.0853, 0.8708] (n = 87)
- sindy_deg1: 0.0585 [-0.0614, 1.5241] (n = 67)

**decoded-forecast RMSE 0-275 (degC)**, median [5 %, 95 %] over stage1 & valid runs:

- ba_deg1: 0.9294 [0.7378, 1.8520] (n = 87)
- sindy_deg1: 1.1859 [0.7544, 2.1488] (n = 67)
- ba_deg3: 1.8746 [0.8056, 364959.7656] (n = 37)
- sindy_deg3: 1.4739 [0.8066, 3.1519] (n = 37)

Bayesian-ARGOS degree 1: posterior probability that the annual mode is decaying, median over runs 0.53; runs with P > 0.95: 5 of 87.

**Paired comparison** on the 53 experiments where both degree-1 models are valid: decoded-forecast RMSE 0-275 median Bayesian-ARGOS 0.929 vs SINDy 1.171 degC; Bayesian-ARGOS lower in 46. Stage1 runs valid for Bayesian-ARGOS only: 34; valid for SINDy only: 14.

**Term frequency** (share of stage1 & valid runs in which the term is identified, per equation):


ba_deg1 (n = 87):

| equation   |    1 |   x1 |   x2 |   x3 |
|:-----------|-----:|-----:|-----:|-----:|
| dz1/dt     | 0.83 | 0.91 | 0.94 | 0.94 |
| dz2/dt     | 0.78 | 0.97 | 0.85 | 0.94 |
| dz3/dt     | 0.83 | 0.94 | 0.84 | 0.91 |

sindy_deg1 (n = 67):

| equation   |    1 |   x1 |   x2 |   x3 |
|:-----------|-----:|-----:|-----:|-----:|
| dz1/dt     | 0.28 | 0.9  | 0.93 | 0.94 |
| dz2/dt     | 0.21 | 0.99 | 0.67 | 0.9  |
| dz3/dt     | 0.1  | 0.9  | 0.73 | 0.79 |

ba_deg3 (n = 51):

| equation   |    1 |   x1 |   x1^2 |   x1^2x2 |   x1^2x3 |   x1^3 |   x1x2 |   x1x2^2 |   x1x2x3 |   x1x3 |   x1x3^2 |   x2 |   x2^2 |   x2^2x3 |   x2^3 |   x2x3 |   x2x3^2 |   x3 |   x3^2 |   x3^3 |
|:-----------|-----:|-----:|-------:|---------:|---------:|-------:|-------:|---------:|---------:|-------:|---------:|-----:|-------:|---------:|-------:|-------:|---------:|-----:|-------:|-------:|
| dz1/dt     | 0.88 | 0.84 |   0.51 |     0.75 |     0.75 |   0.73 |   0.63 |     0.67 |     0.76 |   0.63 |     0.75 | 0.84 |   0.57 |     0.65 |   0.57 |   0.57 |     0.8  | 0.84 |   0.53 |   0.39 |
| dz2/dt     | 0.75 | 0.98 |   0.45 |     0.63 |     0.73 |   0.67 |   0.65 |     0.8  |     0.69 |   0.69 |     0.78 | 0.69 |   0.59 |     0.71 |   0.67 |   0.65 |     0.57 | 0.8  |   0.49 |   0.57 |
| dz3/dt     | 0.67 | 0.92 |   0.55 |     0.63 |     0.69 |   0.61 |   0.73 |     0.55 |     0.73 |   0.57 |     0.69 | 0.71 |   0.45 |     0.71 |   0.53 |   0.71 |     0.63 | 0.78 |   0.57 |   0.35 |

sindy_deg3 (n = 44):

| equation   |    1 |   x1 |   x1^2 |   x1^2x2 |   x1^2x3 |   x1^3 |   x1x2 |   x1x2^2 |   x1x2x3 |   x1x3 |   x1x3^2 |   x2 |   x2^2 |   x2^2x3 |   x2^3 |   x2x3 |   x2x3^2 |   x3 |   x3^2 |   x3^3 |
|:-----------|-----:|-----:|-------:|---------:|---------:|-------:|-------:|---------:|---------:|-------:|---------:|-----:|-------:|---------:|-------:|-------:|---------:|-----:|-------:|-------:|
| dz1/dt     | 0.52 | 0.86 |   0.59 |     0.75 |     0.75 |   0.73 |   0.68 |     0.68 |     0.84 |   0.52 |     0.73 | 0.95 |   0.57 |     0.64 |   0.57 |   0.66 |     0.7  | 0.89 |   0.59 |   0.48 |
| dz2/dt     | 0.5  | 0.95 |   0.52 |     0.73 |     0.75 |   0.8  |   0.7  |     0.84 |     0.73 |   0.57 |     0.77 | 0.75 |   0.55 |     0.73 |   0.75 |   0.5  |     0.64 | 0.82 |   0.5  |   0.57 |
| dz3/dt     | 0.41 | 0.84 |   0.45 |     0.55 |     0.75 |   0.59 |   0.64 |     0.66 |     0.82 |   0.68 |     0.75 | 0.68 |   0.52 |     0.66 |   0.7  |   0.5  |     0.7  | 0.75 |   0.73 |   0.59 |

## Decoder ablation: is the third latent coordinate needed for the field?

Stage1 & valid Bayesian-ARGOS runs: n = 87. Decoded test-field rRMSE with one coordinate frozen at its training mean, relative to the full decode:
- least-needed coordinate: median ratio 1.12 (5-95 %: 1.00-1.37); runs where freezing it costs < 10 %: 34; < 25 %: 68
- most-needed coordinate: median ratio 2.14
- runs in which the identified model leaves one coordinate out of some equation: 44; in those, that coordinate is also the decoder's least-needed one in 21
