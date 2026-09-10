#!/usr/bin/env bash
# Regenerates every result, figure and table used in the reply to Referee 1 Point 6.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY exp1_collinearity.py --trials 20 --trials-n 10 --jobs 16
$PY exp2_heteroscedasticity.py --trials 20 --jobs 16
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS   # let the HMC chains use the cores
$PY exp3_hmc_check.py
$PY exp4_screening_substitution.py   # needs aizawa_eq3_supports.csv from the R script
$PY make_figures.py
