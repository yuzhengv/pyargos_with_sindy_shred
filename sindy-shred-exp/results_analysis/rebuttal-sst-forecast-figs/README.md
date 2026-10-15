# SST forecast figures regenerated from the fixed pipeline (Referee 1, Point 8)

> **Status:** done 2026-09-12. `figures/global_comparison_18.pdf` is the revised
> panel c of Fig. 8 (full globe); `figures/comparison_plot_ba_18.pdf` (Fig. 8b)
> and `figures/multi_sensor_forecast_18.pdf` (Fig. S9) are the same panels
> regenerated after the two fixes below. The submitted files
> (`results_analysis/imgs-2/zoomed_comparison_18.pdf`, `comparison_plot_ba_18.pdf`,
> and `multi_sensor_forecast_18.pdf` in the manuscript repository) are unchanged
> on disk. The swap in the manuscript is deferred to the final amendment step.
>
> **Update 2026-10-14:** the revision figures are regenerated in the house style
> (`../plot-style/house.mplstyle`; plotting standard, blocks M9 and S-F, decisions
> D1, D3, D4, D8) from the stored arrays by the six scripts added to the table
> below, into `figures/house/`: `sst_application.pdf` (revised Fig. 9, experiment
> 91: RMSE violins, latent trajectories with the 90 % posterior band, global maps),
> `multi_sensor_forecast_91.pdf` (Supplementary Fig. S16),
> `latent_space_{123,18,80}.pdf` (Supplementary latent-space figures) and
> `sindy_shred_integration.pdf` (revised Fig. 8, the schematic with the experiment
> 91 equations). Copies under the same names sit in the manuscript repository
> (`imgs/`, `si-plots/`, `reply-to-reviewers/Figures/`). The earlier figures in
> `figures/` are unchanged.

Referee 1 asked why "the SST dataset suddenly only covers the Eastern Hemisphere"
in Fig. 8 while Fig. 7 shows the whole planet. Nothing in the pipeline is
hemispheric: training, the 250 random sensors, the decoded output and the RMSE
distribution of Fig. 8a all use the full 180 × 360 NOAA grid (44,219 ocean
points). The submitted maps came from `plot_zoomed_comparison` in
`results_analysis/analysis_script_ba.py` called with `lon_range = (0, 180)`,
i.e. longitude columns 0–180 (0–180°E) — a display crop, undeclared in the
caption, and the RMSE printed on each panel was computed on that crop. The
SINDy-SHRED paper (Gao et al., PNAS 2026, Fig. 4C) crops the same dataset the
same way, with "we crop the global map for better visualization" as the only
stated reason.

Regenerating the panel surfaced two bugs in the SST pipeline, fixed in this
repository on 2026-09-12:

1. **Inference with dropout active.** None of the SST analysis scripts
   (`analysis_script_ba.py`, `analysis_script_sindy.py`, root and
   `results_analysis/` copies) called `model.eval()`, so the decoder's dropout
   (p = 0.1) was live at inference and every decode was a fresh random draw. The
   submitted Fig. 8c RMSEs, Fig. S9 trajectories and the Fig. 8a distributions
   carry that noise; both forecasters were affected alike. **Fix:** `shred.eval()`
   after `load_state_dict` in all four scripts. The GRU has no dropout, so the
   latent trajectories, the identified equations and Fig. 8b are unaffected
   (checked: latent output identical in train and eval mode).
2. **`sigma_log__` in the identified model.** With bambi 0.16 / pymc 5.28 the
   posterior carries the sampler's transformed site `sigma_log__`;
   `extract_identified_model` in `pyargos/src/argos_bayesian_argos.py` dropped
   only the exact name `sigma`, so the site leaked into the identified model and
   `simulate()` failed with `NameError: name 's' is not defined`. **Fix:** the
   filter is now `not name.startswith("sigma")` (both copies of the method). No
   numerical change for runs made under the older environment.

Fig. 8a (150-experiment RMSE distributions, `exp-ba/` and `exp-sindy/`) was
produced with dropout active and has **not** been rerun; that is a separate
decision (about 150 × 2 analysis runs).

## Scripts

All scripts follow `results_analysis/analysis_script_ba.py` step by step (data
split, scaler, network, latent normalisation, Bayesian-ARGOS design matrix,
forecast, decoder pass) and reuse the pipeline modules unchanged
(`src.sindy_shred`, `utils.processdata`, `pyargos.utils.argos_utils`,
`pyargos.src.argos_bayesian_argos`). The plotting functions are the originals,
copied verbatim.

| File | What it does | Runtime (DGX Spark) |
|---|---|---|
| `regenerate_figs.py` | one experiment: identifies the latent model, simulates the training period (Fig. 8b), forecasts and decodes the test period, writes per-horizon RMSE on the 0–180°E crop and on the globe, the map panels (Fig. 8c), and the 18-sensor panel (Fig. S9); `--dropout-active` reproduces the submitted behaviour | ~1.5 min |
| `plot_fields.py` | `plot_zoomed_comparison` + CLI over `results/fields_<it>.npz`; `--lon 0,180` is the submitted crop, `0,360` the globe | seconds |
| `plot_latent.py` | Fig. 8b block + CLI over `results/latent_train_<it>.npz` | seconds |
| `plot_sensors.py` | `plot_multiple_sensor_predictions` + CLI over `results/sensor_trajectories_<it>.npz` | seconds |
| `run_all.sh` | eval-mode run (the revision figures) then a `--dropout-active` run (comparison only, `_dropout` suffix on every output) | ~3 min |
| `make_readme_results.py` | rewrites the Results section below from `results/` | seconds |
| `make_fig9_panels.py` | revised Fig. 9 panels in the house style: a, RMSE violins from `../rebuttal-latent-models/tables/experiments.csv` (degree-1, stage1 and valid: 87 Bayesian-ARGOS and 67 SINDy experiments; means 0.921/1.014/1.297 and 0.994/1.221/1.617 °C); b, experiment 91 latent trajectories with the 90 % band from `../rebuttal-posterior-forecast/results/latent_bands_91.npz`; c, global maps from `results/fields_91.npz`, RMSE checked against `results/rmse_91.csv` (0.67/0.74/0.67/0.72/0.84 °C); canvases 8.3, 9.7 and 18 in wide (k = 2.5), saved to `figures/house/fig9{a,b,c}.pdf` | 3 s |
| `compose_fig9.sh` | generates a standalone TeX file and includes the three panels at scale 0.4 (a left, b right, c below) in a 183 mm box with a 1 mm border: `figures/house/sst_application.pdf`, 185 x 163 mm | seconds |
| `make_si_sensor_fig.py` | Supplementary Fig. S16 (18 held-out sensors of experiment 91, eval mode, `results/sensor_trajectories_91.npz`) in the house style: 3 x 6 grid, one vertical scale (full data range), RMSE per panel checked against `results/sensor_rmse_91.csv` (mean 0.104, normalised units); canvas 24.3 x 12.2 in for a final width of 247 mm (sidewaysfigure) | 1 s |
| `make_si_latent_figs.py` | Supplementary latent-space panels, 18 in canvases: experiments 123 and 18 (a, GRU latent trajectory against the posterior-mean simulation over the training period; b, global maps with RMSE from `results/rmse_<it>.csv`) and experiment 80 (overlay only, quasi-static third coordinate) | 6 s |
| `compose_si_latent.sh` | composes them at scale 0.4 into `figures/house/latent_space_<it>.pdf` (183 mm plus a 1 mm border) | seconds |
| `compose_fig8_schematic.sh` | revised Fig. 8: covers the three old equation lines in the "Latent Dynamical System" box of `schematic-plots/pyargos_with_sindy_shred.pdf` with the box fill colour and typesets the experiment 91 posterior means (two decimals) in sans-serif mathematics (TikZ, helvet, sfmath, Helvetica minus sign): `figures/house/sindy_shred_integration.pdf` | seconds |

```bash
conda activate env_dgx_spark
cd sindy-shred-exp/results_analysis/rebuttal-sst-forecast-figs
./run_all.sh ~/Projects/pyargos_with_sindy_shred/sindy-shred-exp   # checkout holding Data/ and exp-1/
python plot_fields.py --lon 0,180 --out figures/crop_comparison_18.pdf   # redraw without re-running
python make_readme_results.py                                            # refresh the Results tables
```

House-style revision figures (2026-10-14; no data root needed, stored arrays only):

```bash
python make_fig9_panels.py && ./compose_fig9.sh        # figures/house/sst_application.pdf (revised Fig. 9)
python make_si_sensor_fig.py                           # figures/house/multi_sensor_forecast_91.pdf (Fig. S16)
python make_si_latent_figs.py && ./compose_si_latent.sh   # figures/house/latent_space_{123,18,80}.pdf
./compose_fig8_schematic.sh <manuscript-repo>/schematic-plots/pyargos_with_sindy_shred.pdf \
    figures/house/sindy_shred_integration.pdf          # revised Fig. 8
```

`Data/SST_data.mat`, `exp-1/sensor-location/sensor_location_18.npy` and
`exp-1/saved-model/model_18.pth` are git-ignored and live only in the
`~/Projects` checkout; pass that folder as `--exp-root`. The scripts read it and
write nothing there.

Fig. S9 note: the original draws the 18 held-out sensor columns from the global
NumPy RNG (seeded 0 at the top of the script) *after* the Bayesian-ARGOS run,
and the sampler consumes that RNG, so a fresh seed-0 draw gives a different set.
The submitted figure's sample turned out to be that stream advanced by one
further draw than this script reaches (its per-panel RMSEs match the submitted
ones panel by panel only with that shift), so `--sensor-offset 1` is the
default and reproduces the submitted sample; the script prints the sample it
used. The S9 series and RMSE are in the network's normalised [0, 1] units, as in
the submitted figure.

## Candidate illustrative cases (2026-09-14)

`regenerate_figs.py --iteration <it>` was also run for candidate replacements of
the highlighted case, chosen from `rebuttal-latent-models/tables/experiments.csv`
(criteria in the Point 8 draft): exps 5, 83, 114, 80, 21, 85, 123, and the sparse
candidates 91, 17, 53, 135 (exp 91 recommended: 9 of 12 terms, clean overlay, flat
error, zero third eigenvalue; staged as `*_91*`). Files
`figures/*_<it>.pdf`. Exp 5 (typical third mode but growing error, 0.64 → 1.58 °C)
is withdrawn; exp 114 (same latent fit and flat error as exp 18, near-neutral third
mode; global RMSE 0.857 / 0.850 / 0.691 / 0.706 / 0.842 °C at t = 50–250) and exp 123 (cleanest Fig. 8b among the typical
candidates; global RMSE printed on its panels) are the candidates; both are staged in the manuscript repository as
`imgs/global_comparison_{114,123}.pdf`, `imgs/comparison_plot_ba_{114,123}_eval.pdf`,
`si-plots/multi_sensor_forecast_{114,123}_eval.pdf`. Exp 80's Fig. 8b is a clear
illustration of a quasi-static third latent direction (candidate SI figure). Exp 18 for reference:
0.708 / 0.756 / 0.704 / 0.733 / 0.879 °C.

## Results

**Identified latent model** (experiment 18, eval mode). The manuscript's Eq. 6
rounds the same values to two decimals; repeated HMC runs move the third decimal.

```
dz1/dt = +0.295 +5.786 z1 -1.649 z2 +10.477 z3
dz2/dt = -0.179 -7.850 z1 -8.261 z3
dz3/dt = -0.064 -8.606 z1 -6.619 z3
```

**Fig. 8b.** Unchanged in substance: the GRU latent trajectory is identical in
train and eval mode and the identified model matches, so
`figures/comparison_plot_ba_18.pdf` differs from the submitted panel only by
HMC noise in the third decimal of the coefficients.

**Fig. 8c.** RMSE (°C) of the decoded forecast against the observed field:

| t (weeks) | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|
| submitted panel (crop 0–180°E, dropout active) | 0.866 | 0.779 | 0.833 | 0.756 | 0.804 |
| eval mode, crop 0–180°E | 0.727 | 0.766 | 0.780 | 0.742 | 0.762 |
| **eval mode, global (revised Fig. 8c)** | **0.708** | **0.756** | **0.704** | **0.733** | **0.879** |
| dropout active, global (one draw) | 0.789 | 0.761 | 0.733 | 0.765 | 0.864 |

Cropped and global values differ by up to 0.12 °C at a given horizon, in either
direction, so the submitted annotations were not representative of the global
error. Dropout at inference moves a single decode by up to ~0.07 °C.

**Fig. S9.** Per-sensor RMSE in the network's normalised units. The sensor
sample is the submitted one (see the `--sensor-offset` note above); the
dropout-active rerun matches the submitted values to within one draw's noise,
and the eval-mode values are the revised figure's.

| panel | sensor column | submitted (dropout active) | dropout active, this rerun | **eval mode (revised S9)** |
|---|---|---|---|---|
| 0 | 1304 | 0.0123 | 0.0122 | **0.0123** |
| 1 | 38340 | 0.1610 | 0.1594 | **0.1590** |
| 2 | 32807 | 0.1350 | 0.1355 | **0.1363** |
| 3 | 16768 | 0.1350 | 0.1348 | **0.1329** |
| 4 | 30449 | 0.1020 | 0.1030 | **0.1025** |
| 5 | 35625 | 0.1160 | 0.1171 | **0.1154** |
| 6 | 28929 | 0.0927 | 0.0923 | **0.0922** |
| 7 | 21934 | 0.0821 | 0.0817 | **0.0801** |
| 8 | 22190 | 0.0877 | 0.0889 | **0.0868** |
| 9 | 23804 | 0.1860 | 0.1874 | **0.1855** |
| 10 | 20406 | 0.1990 | 0.1998 | **0.1984** |
| 11 | 21156 | 0.0920 | 0.0927 | **0.0902** |
| 12 | 26508 | 0.1090 | 0.1107 | **0.1104** |
| 13 | 9166 | 0.0669 | 0.0638 | **0.0651** |
| 14 | 35599 | 0.0920 | 0.0916 | **0.0912** |
| 15 | 19737 | 0.1300 | 0.1304 | **0.1296** |
| 16 | 24613 | 0.0975 | 0.0959 | **0.0917** |
| 17 | 7490 | 0.0863 | 0.0824 | **0.0845** |
| mean | | 0.1101 | 0.1100 | **0.1091** |

Files: `results/rmse_18.csv`, `results/sensor_rmse_18.csv`, `results/identified_model_18.csv`;
`_dropout` variants for the comparison run; fields and trajectories in the `.npz` files (~2 MB each).
