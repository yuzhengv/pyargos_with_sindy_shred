# Latent dimension 2 versus 3 (Referee 1, Point 8: "why not find a 2D latent system directly?")

> **Status:** done 2026-09-18. Twenty r = 2 networks trained (experiments 0–19,
> ~160 s each on the DGX Spark), identified and compared with their r = 3
> counterparts; `tables/summary.md`, `tables/latent_dim_comparison.csv`. Result: the
> two-dimensional latent space reconstructs the field worse and yields worse
> decoded forecasts; see "Results". 2026-11-09: the filter-passing convention of
> the 150-experiment batch added (validity among networks passing the filter,
> medians over the valid models); the Supplementary Information reports that
> convention, under which the valid models of the two dimensions forecast alike.

Referee 1 asks why a three-dimensional latent system is sought when the third
mode is short-lived. The 150-experiment log (`../rebuttal-latent-models/`)
shows that the third mode is in fact near-neutral in most latent spaces and that
the decoder uses all three coordinates in most of them (decoder ablation), but
none of that says how a network trained with a *two*-dimensional latent space
would do. This folder answers that directly.

## Design

- Paired comparison on the sensor sets of experiments 0–19 (chosen as the first
  twenty; 14 of their r = 3 networks pass the stage-1 reconstruction filter and
  16 yield a valid Bayesian-ARGOS model). For each, a network with latent
  dimension r = 2 is trained by `train_latent_dim.py` exactly as
  `sindy-shred-exp/training_script.py` trained the r = 3 networks (same split,
  scaler, sensor set, widths, 600 epochs, thresholding schedule, seed 0); only
  `hidden_size` and the E-SINDy library size change.
- Sample size: 20 pairs. A paired rank test on a continuous outcome
  (reconstruction rRMSE, decoded-forecast RMSE) can detect a median shift of
  about 0.6 standard deviations at n = 20 (about 0.9 at n = 10); the binary
  outcomes (stage-1 pass, valid model) are reported as paired counts and are the
  less powerful part of the comparison.
- Outcomes per pair, r = 2 against the existing r = 3 network: test-set
  reconstruction rRMSE (eval mode) and stage-1 pass; Bayesian-ARGOS degree-1
  identification on the r = 2 latent trajectory via
  `../rebuttal-latent-models/run_experiment.py --latent-dim 2 --model-path ...`
  (validity, eigenvalues, decoded-forecast RMSE over the Fig. 8a horizons); the
  same for SINDy at the pipeline threshold.

## Files

| file | role |
|---|---|
| `train_latent_dim.py` | retrain SINDy-SHRED for one experiment's sensor set with `--latent-dim r`; writes `models/model_<it>_r<r>.pth`, `results/train_<it>_r<r>.json` |
| `run_batch.sh` | 20 trainings (two at a time), then identification for each, then `collect.py` |
| `collect.py` | paired table `tables/latent_dim_comparison.csv` and `tables/summary.md` |

```bash
conda activate env_dgx_spark
cd sindy-shred-exp/results_analysis/rebuttal-latent-dim2
./run_batch.sh ~/Projects/pyargos_with_sindy_shred/sindy-shred-exp 0 19 2
```

The data and sensor locations are read from the `~/Projects` checkout; nothing
is written there, and the existing r = 3 models are not touched.

## Results (20 paired sensor sets; `tables/summary.md`)

| outcome | r = 2 | r = 3 | paired comparison |
|---|---|---|---|
| test reconstruction rRMSE (eval mode), median | 0.053 | 0.046 | r = 2 lower in 5 of 20; Wilcoxon p = 0.027 |
| stage-1 pass (rRMSE ≤ 0.05) | 9 of 20 | 14 of 20 | both 7; r = 2 only 2; r = 3 only 7 |
| Bayesian-ARGOS valid model | 17 | 16 | validity does not discriminate |
| Bayesian-ARGOS decoded-forecast RMSE 0–275 wk, pairs valid for both (n = 13), median | 1.35 °C | 0.93 °C | r = 2 lower in 3 of 13; p = 0.013 |
| Bayesian-ARGOS identified period, median | 1.033 yr | 1.017 yr | r = 2 off by > 0.1 yr in 3 of 20 (5.2, 2.7, 2.8 yr in exps 19, 7, 8) |
| SINDy valid model | 16 | 16 | – |
| SINDy decoded-forecast RMSE, pairs valid for both (n = 13), median | 1.74 °C | 1.44 °C | r = 2 lower in 5 of 13; p = 0.15 |

The rows above identify the latent model for all 20 networks, whether or not they
pass the stage-1 filter. Under the filter-passing convention of the 150-experiment
batch (validity counted among the networks that pass the filter, medians over the
valid models; `tables/summary.md`, second block), which the Supplementary
Information (Table S4) reports since 2026-11-09:

| outcome | r = 2 | r = 3 | comparison |
|---|---|---|---|
| networks passing the filter | 9 of 20 | 14 of 20 | – |
| Bayesian-ARGOS valid model among them | 7 of 9 | 11 of 14 | – |
| Bayesian-ARGOS decoded-forecast RMSE 0–275 wk, medians over the valid passing models | 0.86 °C (n = 7) | 0.93 °C (n = 11) | Mann-Whitney U p = 0.48; 5 pairs valid on both sides, too few for a paired test |
| Bayesian-ARGOS identified period, valid passing models | median 1.010 yr, all within 0.025 yr of one year | median 1.018 yr, all within 0.036 yr | the off-period r = 2 models (exps 7, 8, 19) all fail the filter |
| SINDy valid model among them | 8 of 9 | 10 of 14 | – |
| SINDy decoded-forecast RMSE, medians over the valid passing models | 1.12 °C (n = 8) | 1.41 °C (n = 10) | p = 0.20 |

Reading. With two latent coordinates the network can only encode the annual
oscillation (its phase and amplitude); the slowly varying level that the third
coordinate carries in the r = 3 networks has nowhere to go, so the decoded field
is reconstructed less accurately (fewer latent spaces pass the 0.05 filter) and
the latent trajectory is less well described by an autonomous linear model
(decoded forecasts 0.4–0.6 °C worse at the median, and in three of twenty cases
the identified oscillation is no longer annual). The identification step is not
the limitation: the r = 2 models are valid as often as the r = 3 ones, and under
the filter-passing convention the valid r = 2 models also forecast as accurately as
the valid r = 3 ones (the worse forecasts and the off-period models above belong to
r = 2 networks that fail the filter), so the limitation is the reconstruction of the
field by the two-dimensional representation. This is
the direct answer to "why not find a 2D latent system directly": a 2-D latent
space is a worse representation of the field, and the identification inherits
that. Caveats: n = 20; the r = 3 networks were trained on the HPC and the r = 2
ones here, with the same seed and schedule but different hardware; a single
training seed per pair.

Training logs were removed after the run (per-epoch loss prints); the batch log
`logs/batch_r2.log` and the per-network `results/train_<it>_r2.json` (with the
validation-error trace) are kept. The 20 r = 2 weight files are in `models/` (1.4 GB) and are git-ignored (rule added
to the repository's `.gitignore`); they are regenerable with `run_batch.sh`.
