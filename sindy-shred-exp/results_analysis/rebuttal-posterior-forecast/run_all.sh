#!/usr/bin/env bash
# Posterior-predictive bands for the highlighted SST experiment (Referee 1, Point 8, "error bars").
# Usage: ./run_all.sh <exp-root>   (checkout holding Data/SST_data.mat and exp-1/)
set -euo pipefail
EXP_ROOT="${1:-$HOME/Projects/pyargos_with_sindy_shred/sindy-shred-exp}"
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/anaconda3/envs/env_dgx_spark/bin/python}"
export OMP_NUM_THREADS=1
"$PY" posterior_forecast.py --exp-root "$EXP_ROOT" --iteration 91 --n-draws 1000 2>&1 | tee results/log_91.txt
