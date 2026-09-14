# Identified latent models across the 150 SST experiments (Referee 1, Point 8)

> **Status:** batch run 2026-09-12 (two jobs, 08:57–11:20 UTC for 136
> experiments; the 14 whose weights arrived later were run 13:05–13:20 UTC):
> all 150 experiments logged. Aggregated tables in `tables/`; the numbers below
> come from `tables/summary.md`.

Referee 1's Point 8 asks, beyond the hemisphere question, why the SST latent
library is affine-linear and whether higher-order terms were considered, why a
3D latent system is sought when the fast mode decays in about a year, whether
the method merely rediscovers the annual cycle, and why the posterior error bars
do not enter the projections or the modal solution. The submitted batch behind
Fig. 8a recorded only latent MSEs, reconstruction rRMSE and forecast metrics, so
the 150 identified systems themselves were not kept. This folder logs them.

## What is recorded per experiment (`run_experiment.py --iteration <it>`)

Follows `sindy-shred-exp/analysis_script_ba.py` / `analysis_script_sindy.py`
(same split, scaler, network in eval mode, latent normalisation, simulation
horizons and metric definitions) and adds:

| Item | Where |
|---|---|
| identified equations, Bayesian-ARGOS and SINDy (STLSQ 0.8, alpha 0.05, finite differences), each with the manuscript's degree-1 library **and** a degree-3 polynomial library | `model_<tag>.csv`, `summary.json` |
| Bayesian-ARGOS posterior mean / sd / 90 % HDI of every candidate term, screened-out terms marked absent | `posterior_ba_deg<d>.csv` |
| E-SINDy coefficients learnt inside the network (ensemble tensor and its mean) | `esindy_coefficients.npy`, `_mean.csv` |
| for every affine-linear model: `A`, `b`, equilibrium, eigenvalues; per mode period, decay rate, half-life, e-folding time; spectral stability | `modes_<tag>.csv`, `summary.json` |
| for Bayesian-ARGOS degree 1: the same modal quantities over 1000 posterior draws, with 5/50/95 % quantiles, P(stable), P(annual mode decaying) | `modes_draws_ba_deg1.csv`, `summary.json` |
| selection quantities: network test rRMSE (eval mode, and one dropout-active draw as the submitted batch computed it), training latent MSE per dimension, validity flag, test latent MSE, decoded-forecast RMSE over the Fig. 8a horizons (0–100, 100–200, 200–275, 0–275 weeks) | `summary.json` |

Tags: `ba_deg1`, `ba_deg3`, `sindy_deg1`, `sindy_deg3`. Simulations that
diverge or fail are recorded as non-finite and flagged invalid, never dropped.

## Selection flags (`collect.py`)

The manuscript's two-stage selection, as implemented in
`results_analysis/performance_assessment.py` and `analysis-functions/`:

1. **stage1** — network test reconstruction rRMSE ≤ 0.05. The submitted batch
   evaluated this with decoder dropout active (see `rebuttal-sst-forecast-figs/`);
   here it is evaluated in eval mode, with the dropout-draw value kept for
   reference, so the change in the passing set can be stated.
2. **valid** — per model, training latent MSE ≤ 1 on every dimension and the
   simulation finite. This produced the "82 valid cases" / "64 valid cases" of
   the submitted Fig. 8a.

`collect.py` writes `tables/experiments.csv` (one row per experiment × model,
flags as columns), `tables/terms.csv` (one row per identified term) and
`tables/summary.md` (counts per subset; distributions of period, slow-mode
decay, fast-mode e-folding time and forecast RMSE over the stage1 & valid runs;
the paired subset where both degree-1 models are valid; posterior probability
that the annual mode decays; term frequencies per equation). Nothing is
filtered in the CSVs.

## Running

```bash
conda activate env_dgx_spark            # pysindy 2.1.0 was added to this env on 2026-09-12
cd sindy-shred-exp/results_analysis/rebuttal-latent-models
python run_experiment.py --iteration 18 --exp-root ~/Projects/pyargos_with_sindy_shred/sindy-shred-exp
python collect.py
python make_fig8a.py                    # Fig. 8a from tables/experiments.csv -> figures/rmse_comparison_violin_plot.pdf
python run_decoder_ablation.py --iteration 18 --exp-root ...   # freeze each latent coordinate at decode time (~20 s)
python collect_ablation.py              # -> tables/decoder_ablation.csv, section appended to tables/summary.md
```

### Running the batch

```bash
./run_batch.sh ~/Projects/pyargos_with_sindy_shred/sindy-shred-exp 0 149 2   # EXP_ROOT FIRST LAST JOBS
```

Per experiment: about 1 min for `ba_deg1`, a few minutes for `ba_deg3`
(larger library, more HMC), seconds for SINDy. With two parallel jobs on the
DGX Spark the full batch is a matter of hours; logs go to `logs/`, results to
`results/exp_<it>/`, and `collect.py` runs at the end. The data, sensor
locations and weights are git-ignored and live only in the `~/Projects`
checkout, read through `--exp-root`; nothing is written there.

## Note on the first pass

The `~/Projects` checkout initially lacked `model_<it>.pth` for experiments
32, 35, 40, 47, 49, 51, 80, 91, 96, 98, 102, 105, 114, 142 (their runs are the
FAILED lines in `logs/batch.log`); the files were copied in the same day and
those 14 were rerun singly (`logs/batch_missing14.log`) before `collect.py`.

## Notes

* Bayesian-ARGOS run-to-run HMC noise moves the third decimal of the
  coefficients and, through it, the small real part of the annual-mode
  eigenvalue (its decay rate); the per-draw quantiles are the honest statement
  of that quantity.
* Equations are in the normalised latent coordinates ([-1, 1] on the training
  min/max) used throughout the manuscript.
* Fig. 8a is regenerated by `make_fig8a.py` from `experiments.csv` (eval-mode
  decoding; 87 Bayesian-ARGOS and 67 SINDy valid cases against 82 and 64 in the
  submitted panel; mean RMSE per period Bayesian-ARGOS 0.921 / 1.014 / 1.297 °C,
  SINDy 0.994 / 1.221 / 1.617 °C). Staged as `imgs/rmse_comparison_violin_plot_eval.pdf`
  in the manuscript repository for the deferred swap.

## Test on experiment 18

`python run_experiment.py --iteration 18`, 186 s on the DGX Spark (`ba_deg3`
takes most of it). Stage 1: rRMSE 0.0462 in eval mode (0.0467 for one dropout
draw), pass.

| model | valid | terms (nonlinear) | train latent MSE | forecast RMSE 0–275 wk (°C) | period (yr) | annual-mode decay (/yr) | fast-mode e-folding (yr) |
|---|---|---|---|---|---|---|---|
| ba_deg1 | yes | 10 (0) | 0.052 | 0.760 | 1.007 | +0.012 | 1.23 |
| sindy_deg1 | yes | 8 (0) | 0.136 | 0.863 | 1.011 | +0.005 | 17.1 |
| ba_deg3 | yes | 41 (32) | 0.822 | 2.492 | – | – | – |
| sindy_deg3 | no (diverges) | 45 (36) | – | – | – | – | – |

Bayesian-ARGOS degree 1, over 1000 posterior draws: period 1.007 [0.997, 1.017] yr;
annual-mode decay rate 0.011 [−0.052, +0.070] /yr, P(decaying) = 0.60; fast-mode
decay rate 0.81 [0.59, 1.02] /yr. So on this experiment the annual mode's
"small decay" is not distinguishable from zero (the manuscript's 0.015 /yr and
46-year half-life are point estimates inside that interval), whereas the fast
mode's one-year e-folding time is well determined. With the degree-3 library
both methods return dense cubic models (the polynomial features of a
near-sinusoidal trajectory are strongly collinear): Bayesian-ARGOS keeps 32
nonlinear terms and its decoded forecast degrades from 0.76 to 2.49 °C RMSE,
SINDy's diverges. The network's own E-SINDy layer (degree-3 library, ensemble
mean) is likewise dense, 41 of 60 coefficients nonzero. Whether these hold
across the 150 experiments is what the batch will show.

## Results across the 150 logged experiments

Selection: 116 pass stage 1 in eval mode (107 with a dropout-active draw, a
subset of the 116). Of those, 87 Bayesian-ARGOS and 67 SINDy degree-1 models
are valid (submitted Fig. 8a: 82 and 64 of 150). 53 experiments are valid for
both (the paired subset).

**Validity is spectral instability.** Every invalid Bayesian-ARGOS degree-1
model (29 of 29) and 44 of 49 invalid SINDy models have an eigenvalue with
positive real part (median +0.33 and +0.25 /yr): the simulation grows, the
training latent MSE exceeds 1. No degree-1 simulation was non-finite.

**Mode structure (degree 1, valid runs).** Every model has one oscillatory
pair and one real mode. Period: Bayesian-ARGOS median 1.017 yr, 83 of 87
within 1.00 ± 0.05 yr; SINDy 1.030 yr, 53 of 67. Annual-mode decay rate:
Bayesian-ARGOS median +0.004 [-0.012, +0.074] /yr, decaying in 56 of 87 runs,
and over the posterior P(decaying) > 0.95 in only 5 runs (P < 0.05 in 1): the
annual mode is neutrally stable within its uncertainty. The real mode is *not*
generally fast: median decay +0.000 [-0.09, +0.87] /yr, near-neutral
(|rate| < 0.05 /yr) in 51 of 87 runs, decaying with e-folding < 2 yr in 8, and
clearly decaying over the posterior in 13. Experiment 18's fast mode (e-folding
1.2 yr) is therefore an exception; the typical third direction is a
quasi-static (near-zero eigenvalue) component. SINDy: real mode near-neutral in
27 of 67, fast in 13.

**Forecast (degree 1).** Decoded RMSE 0–275 weeks, valid runs: Bayesian-ARGOS
median 0.93 °C [0.74, 1.85], SINDy 1.19 °C [0.75, 2.15]. Paired subset (53):
0.93 vs 1.17 °C, Bayesian-ARGOS lower in 46 of 53. Stage-1 runs valid for
Bayesian-ARGOS only: 34; for SINDy only: 14.

**Degree-3 library.** Both methods return dense cubic models in every valid
run (about 40 terms, all with nonlinear terms; no run returned a sparse or
linear model). Validity drops to 51 (Bayesian-ARGOS) and 44 (SINDy) of 116,
with 41 and 60 non-finite training simulations among the invalid ones, and the
test-period forecast of the valid models is worse and often unstable: median
decoded RMSE 1.87 and 1.47 °C, non-finite in 14 and 7, above 5 °C in 4 and 1.
The polynomial features of a near-sinusoidal trajectory are strongly
collinear; the affine-linear library is the parsimonious choice and the better
forecaster. The network's own E-SINDy layer (degree-3 library) is likewise
dense (`esindy_coefficients_mean.csv` per experiment).

**Term frequency (degree 1, valid runs).** Bayesian-ARGOS identifies each
linear term in 84–97 % of runs and the intercept in 78–83 %; SINDy identifies
the intercept in only 10–28 % (STLSQ at threshold 0.8 removes small
constants), which is where the equilibrium offset of the latent trajectory goes.


## Decoder ablation (`run_decoder_ablation.py`, `collect_ablation.py`), 2026-09-15

For the referee's "why not a 2-D latent space": each experiment's own test latent
trajectory is decoded with one coordinate frozen at its training mean, and the
decoded-field error is compared with the full decode (`tables/decoder_ablation.csv`;
section at the end of `tables/summary.md`). Over the 87 valid runs the least-needed
coordinate costs a median 12 % in rRMSE when frozen (5–95 %: 0–37 %; < 5 % in 21
runs), the most-needed 114 %; in 19 runs all three cost ≥ 25 %. The coordinate the
identified model leaves out of equations is the decoder's least-needed one in 21 of
44 such runs (chance level). Reading: the decoder uses all three coordinates in most
latent spaces, a quarter have a nearly redundant one, and dynamical triviality of a
coordinate says nothing about its role in the representation. Exp 91 (the proposed
highlighted case): freezing its dynamically slaved $z_1$ costs 117 %, $z_2$ 25 %,
$z_3$ 12 %.
