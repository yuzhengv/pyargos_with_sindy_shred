# Posterior-predictive bands for the SST forecast (Referee 1, Point 8, "error bars")

> **Status:** done 2026-10-06 (exp 91, 1,000 draws, 515 s). Framing decided the same
> day: the posterior applies in full to the *latent* model, so the Fig. 8b band is
> kept (`imgs/comparison_plot_ba_91_bands.pdf` in the manuscript repository); the
> decoded bands are an error decomposition for the SI (`imgs/rmse_lead_time_91.pdf`,
> coverage stated), not predictive intervals; the Fig. S9 bands are not used in the
> manuscript (figure kept here). The manuscript swap is deferred to the final
> amendment step; the submitted figure files are unchanged.

Referee 1 wrote: "Bayesian-ARGOS presumably returns error bars on the model
coefficients, but those don't factor into generating projections or the
analytic modal solution." The modal solution is covered by
`../rebuttal-latent-models/` (eigenvalues of 1,000 posterior draws of the
coefficient matrix, per experiment). This folder covers the projections: it
propagates the coefficient posterior of the highlighted experiment (exp 91,
the model of Eq. 5 in the revision) through the identified latent system and
the SINDy-SHRED decoder, and draws credible bands on the panels the manuscript
shows (Fig. 8b latent trajectories, Fig. S9 sensor forecasts) and on the
decoded RMSE as a function of lead time.

## What is propagated, and what is not

* The pipeline up to the forecast is `results_analysis/analysis_script_ba.py`
  step by step (data split, scaler, network in eval mode, latent normalisation,
  Bayesian-ARGOS with the degree-1 library and 90 % credible intervals); the
  experiment loader and decoder pass are imported from
  `../rebuttal-latent-models/run_experiment.py`, the pipeline modules are used
  unchanged, and no package code is modified.
* Draws: 1,000 joint draws of the coefficients from the HMC posterior of each
  equation (the three equations are fitted independently, so their draws are
  paired at random). Each draw is simulated over the training period from the
  training initial condition (Fig. 8b band) and over the test period from the
  test initial condition, decoded to the 44,219-point field, and scored by its
  global RMSE at every lead time and at the 18 held-out sensors of Fig. S9 (the
  sensor sample is read from `../rebuttal-sst-forecast-figs/results/`, so the
  panels are the staged S9 panels).
* Two draw sets: `identified` (terms whose 90 % interval excludes zero, i.e. the
  structure of Eq. 5, other terms fixed at zero; used for the figures) and
  `allterms` (every term that survived the screening step, whatever its
  interval; a sensitivity check). For exp 91 the screening left exactly the
  identified terms, so the two sets coincide.
* **The bands carry coefficient uncertainty only, conditional on the screened
  support.** The encoder and decoder weights are fixed, and the regression noise
  `sigma` of the latent equations is not injected as process noise. The bands
  therefore answer "how much of the forecast error is attributable to the
  uncertainty of the identified coefficients", not "where will the field be with
  90 % probability". Coverage is reported so that this is visible.
* Bands are pointwise 5–95 % quantiles over draws. "Point model" is the
  posterior-mean model used for the submitted and staged figures;
  "mean-of-draws forecast" is the average of the decoded forecasts over draws.

## Scripts

| File | What it does | Runtime (DGX Spark) |
|---|---|---|
| `posterior_forecast.py` | one experiment: Bayesian-ARGOS fit, point forecast, posterior draws, simulation, decoding, quantiles, coverage, figures | ~10 min for 1,000 draws (Bayesian-ARGOS itself ~6 s) |
| `plot_bands.py` | band-aware versions of the Fig. 8b and Fig. S9 plotting blocks, plus the RMSE-versus-lead-time panel | seconds |
| `run_all.sh` | runs exp 91 with 1,000 draws, log in `results/log_91.txt` | ~10 min |

```bash
conda activate env_dgx_spark
cd sindy-shred-exp/results_analysis/rebuttal-posterior-forecast
./run_all.sh ~/Projects/pyargos_with_sindy_shred/sindy-shred-exp   # checkout holding Data/ and exp-1/
python posterior_forecast.py --iteration 18 --exp-root ...          # any other experiment
```

Outputs: `results/identified_model_91.csv`, `latent_bands_91.npz`,
`sensor_bands_91.npz`, `rmse_lead_time_91.csv` (per week),
`rmse_horizons_91.csv` (t = 50…250 with field coverage and band width),
`rmse_draws_91_<set>.csv`, `sensor_stats_91.csv`, `run_91.json`;
`figures/comparison_plot_ba_91_bands.pdf` (Fig. 8b with band),
`latent_forecast_91_bands.pdf` (test-period latent forecast with band),
`multi_sensor_forecast_91_bands.pdf` (Fig. S9 with bands and per-panel coverage),
`rmse_lead_time_91.pdf`.

## Results (exp 91, 1,000 draws, both draw sets identical)

**Identified model** (posterior mean; sd of every coefficient 0.02–0.06, i.e.
0.5–3 % of its value; from `../rebuttal-latent-models/results/exp_91/posterior_ba_deg1.csv`):

```
dz1/dt = -0.980 +8.166 z2 -4.199 z3
dz2/dt = +0.550 -3.413 z2 +6.152 z3
dz3/dt = +0.953 -8.189 z2 +3.426 z3
```

Posterior of the modes (same source, 1,000 draws): period 1.010 yr [1.003, 1.018];
annual-mode decay rate −0.006 /yr [−0.054, +0.038] (negative = growing), so the
sign of the decay is undetermined; the third eigenvalue is exactly zero in every
draw (z1 is absent from all three equations).

**Latent bands (Fig. 8b, training period; `latent_bands_91.npz`).** The 90 %
band is narrow at first and widens with time, because the small uncertainty in
the period accumulates as phase drift and the decay rate's sign is undetermined.
Half-width of the band on z1 / z2 / z3, in normalised latent units (the
oscillation amplitude is about 1):

| week | 52 | 104 | 208 | 1040 (end of training) |
|---|---|---|---|---|
| training-period simulation | 0.07 / 0.03 / 0.04 | 0.14 / 0.07 / 0.08 | 0.28 / 0.13 / 0.16 | 1.48 / 0.79 / 0.99 |
| test-period forecast (week from the test start) | 0.08 / 0.04 / 0.05 | 0.15 / 0.08 / 0.09 | 0.31 / 0.17 / 0.19 | 0.45 / 0.26 / 0.29 at week 317 |

**Decoded RMSE against lead time (`rmse_horizons_91.csv`, `rmse_lead_time_91.csv`).**
Global RMSE (°C) over the 44,219 ocean points. "Point model" is the posterior-mean
model (the staged Fig. 8c values); quantiles are over the 1,000 draws; coverage is
the fraction of ocean points whose observed value lies inside the 5–95 % band of
the decoded draws at that week.

| t (weeks) | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|
| point model | 0.671 | 0.740 | 0.671 | 0.718 | 0.846 |
| mean-of-draws forecast | 0.672 | 0.740 | 0.670 | 0.712 | 0.860 |
| draws, 5 % | 0.671 | 0.739 | 0.663 | 0.687 | 0.827 |
| draws, median | 0.678 | 0.753 | 0.700 | 0.784 | 0.931 |
| draws, 95 % | 0.719 | 0.867 | 0.996 | 1.290 | 1.578 |
| band width (°C, mean over points) | 0.27 | 0.45 | 0.73 | 1.13 | 1.38 |
| field coverage of the 90 % band | 0.14 | 0.20 | 0.34 | 0.48 | 0.51 |

Over the whole test period the point model's RMSE averages 0.72 °C, the median
draw 0.76 °C and the 95 % draw 1.04 °C. The band of RMSE values is narrower than
0.2 °C for the first 39 weeks and exceeds 0.5 °C from week 138.

**Sensor bands (Fig. S9, `sensor_stats_91.csv`).** Mean per-panel RMSE of the
point forecast 0.104 (normalised units; the staged S9 value is 0.104); mean
coverage of the 90 % band over the 250 forecast weeks 0.38 (0.12–0.76 across
panels).

**Reading.** (i) The coefficient posterior is tight, yet its effect on the
latent trajectory compounds with time: it is negligible for the first year and
comparable to the oscillation amplitude after twenty, because a period
uncertainty of ±0.7 % is a phase uncertainty of ±1.5 months after 20 years, and
the sign of the decay rate is undetermined. This band lives entirely in the
latent space, where the uncertainty is Bayesian-ARGOS's own, so it is a correct
posterior band and goes into Fig. 8b. (ii) Carried through the decoder, the same
draws measure the share of the forecast error attributable to the identified
coefficients: negligible in year one, comparable to the error itself by year
five. The posterior-mean model lies at the low edge of the RMSE distribution at
long lead times, so the point model's flat 0.67–0.85 °C error understates what
the posterior as a whole predicts. (iii) Coverage far below 90 % at short lead
times is expected and is the reason the decoded bands are *not* predictive
intervals: the field error has two further sources the pipeline does not
quantify, the slowly varying non-periodic content of the latent trajectory that
an autonomous linear model does not carry (its scale is the regression noise
`sigma`, not injected here), and the reconstruction error of the decoder, a
deterministic network with fixed weights and no posterior. SINDy-SHRED
quantifies neither. The decoded RMSE-against-lead-time panel is therefore
presented in the SI as an error decomposition with its coverage stated; the
sensor-level bands (`multi_sensor_forecast_91_bands.pdf`) are kept here for
reference only, because a shaded band around a sensor forecast invites the
predictive-interval reading. All intervals are conditional on the screened
support.
