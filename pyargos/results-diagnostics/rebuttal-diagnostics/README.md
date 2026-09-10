# Rebuttal diagnostics (Referee 1 Points 6–7, Referee 2 minor point 2)

> **Status:** prototype. The numbers used in the reply come from the R re-implementation in
> `code-repos/Bayesian-ARGOS/ResultsAnalysis/rebuttal-diagnostics/` (same experiments, the
> manuscript's own pipeline). This folder remains the home of the HMC equivalence check
> (`exp3_hmc_check.py`), the stored-results post-processing (`exp4`), and the shared
> `make_figures.py` (`--res <folder>` works for both result sets; with `--tables-only` it writes the
> summary tables while the R folder's `make_figures.R` draws the figures in the manuscript's house style). The "pipeline-selected
> support" residual rows produced here reflect the Python screening and are superseded by
> the R run, which finds no residual structure at any finite SNR.

New evidence for the revision of *Fast and principled equation discovery from
chaos to climate* (COMMSPHYS-26-0738-T). The referees asked (i) why VIF is
reported per term when collinearity is collective, (ii) whether the Aizawa
collinearity comes from the system or from the candidate library, (iii) what
the smooth curve in Fig. 3d is, and (iv) for a quantitative measure of
heteroscedasticity / coherent residual structure at high SNR.

All scripts re-use the manuscript pipeline unchanged (`utils.argos_simulator`,
`utils.argos_utils.build_design_matrix`, `src.argos_sparse_regression`,
`src.argos_bayesian_argos`), so the numbers are comparable with Fig. 3.
Unlike the original experiments the noise draw is seeded, so every table can be
regenerated exactly.

| Script | What it does | Runtime (20 trials, 10 workers, DGX Spark) |
|---|---|---|
| `common.py` | systems, seeded simulation, library, screening passes, VIF, Belsley diagnostics, residual statistics | – |
| `exp1_collinearity.py` | per-term VIF and condition number on the full / degree-trimmed / selected / ground-truth designs over the Fig. 3b grids; on-attractor vs box vs multi-trajectory sampling; Belsley near-dependency groups | ~10 min |
| `exp2_heteroscedasticity.py` | Breusch–Pagan, White, decile variance ratio, lag-1 autocorrelation and Durbin–Watson of residuals vs SNR (Aizawa x3-dot, Rössler x1-dot); residual variance decomposition into derivative-estimation, smoothing and coefficient error | ~5 min |
| `exp3_hmc_check.py` | full Bayesian-ARGOS (HMC) runs at SNR 50 / 60 / ∞ confirming that residuals about the posterior-predictive mean equal the OLS residuals on the same support; PSIS-LOO k-hat | ~6 min |
| `exp4_screening_substitution.py` | reads the per-trial supports exported by the R script (below): per-term admission at screening vs retention after HMC, the spurious substitutes selected when a true term is lost, and VIF of the true terms inside the admitted support | seconds |
| `exp5_ols_weight_fix_ab.py` | A/B of the screening with the original OLS-weight branch (design-faithful to R) versus the standardised alternative used in AgenticARGOS; in-memory only, repository code untouched | ~2 min |
| `make_figures.py` | `figures/*.pdf` and `tables/summary_tables.md` | seconds |

```bash
conda activate env_dgx_spark      # or the pyargos environment.yml
cd pyargos/results-diagnostics/rebuttal-diagnostics
python exp1_collinearity.py --trials 20 --jobs 10
python exp2_heteroscedasticity.py --trials 20 --jobs 10
python exp3_hmc_check.py
python make_figures.py
```

Definitions used throughout:

* **VIF_j = 1 / (1 − R_j²)**, R_j² from regressing library column *j* on all
  the other columns of the same design plus an intercept (centered VIF, the
  `statsmodels` convention used for Fig. 3b).
* **Condition number κ** and **variance-decomposition proportions** follow
  Belsley, Kuh & Welsch (1980): columns (with the intercept) scaled to unit
  length, κ = σ_max/σ_min; a near dependency is a singular direction with
  condition index > 30 in which two or more columns carry > 50 % of their
  coefficient variance.
* **Residual statistics** are computed on OLS residuals of the ground-truth
  support (exp2) and on the HMC posterior-predictive residuals (exp3).

Outputs are written to `results/`, `figures/` and `tables/` next to the
scripts.

## Companion R analysis

`code-repos/Bayesian-ARGOS/ResultsAnalysis/rebuttal-diagnostics/aizawa_screening_vs_hmc.R`
re-reads the stored 100-trial benchmark results for the Aizawa x3-dot equation and
tabulates, per n and SNR, the size of the adaptive-lasso pass-2 support that enters
HMC, the fraction of spurious candidates that survive the 90 % credible-interval
rule, and the exact-recovery success rate.

## Headline findings

* True terms are lost at the adaptive-lasso screening stage, never at the HMC
  stage (admission and retention rates coincide), by substitution within a
  near-dependent monomial chain (x2²x3 → x2²x3², x1²x3 → x1²x3²); VIF inside the
  admitted support reaches 80–8000 at n = 1e5 without destabilising HMC.

* The seven ground-truth terms of x3-dot are well conditioned on the attractor
  (VIF ≤ 9, κ ≈ 11); the degree-trimmed library used for Fig. 3b has κ ≈ 400 and
  the full library κ ≈ 900, dominated by monomial chains such as {x3², x3³, x3⁴}.
  The collinearity is a property of the candidate library, not of the equation.
* Sampling the same monomials on 50 short transient trajectories reduces κ of the
  full library by an order of magnitude; uniform box samples only halve it.
* With the correct support the residuals are homoscedastic at every finite SNR
  (Breusch–Pagan R² < 0.001, decile variance ratio ≈ 1.4); structure appears only
  for a mis-selected support (unmasked as noise falls) or at SNR = ∞, where the
  residual is the Savitzky–Golay truncation error (σ ≈ 1e-5).
* The Fig. 3d smooth curve is a cubic `np.polyfit`; `make_figures.py` replaces it
  with a binned mean ± 2 sd band.

## Caveat

The Python screening (`argos_alasso`) is weaker than the R implementation on this
equation: it never returned the full ground-truth support in these runs (it drops
x1³x3), whereas the stored R results contain that term in > 90 % of trials at
50–60 dB. Design-based collinearity numbers are unaffected; the "selected support"
residual statistics illustrate the mechanism but should not be read as the
benchmark success rates. The no-intercept raw-x OLS weights are the intended design (R does the
same); the standardised alternative used in AgenticARGOS was tested in exp5 and
does not help (it loses x1³x3 more often), so it should not be adopted here. The
gap is elsewhere (lasso path / CV / post-selection differences between glmnet
and adelie).
