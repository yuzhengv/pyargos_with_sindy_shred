#!/usr/bin/env bash
# Log the identified latent models for all 150 SINDy-SHRED experiments (both forecasters, degrees 1 and 3).
# Usage: ./run_batch.sh [EXP_ROOT] [FIRST] [LAST] [JOBS]
set -uo pipefail
cd "$(dirname "$0")"
EXP_ROOT="${1:-$HOME/Projects/pyargos_with_sindy_shred/sindy-shred-exp}"
FIRST="${2:-0}"; LAST="${3:-149}"; JOBS="${4:-2}"
mkdir -p logs
seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c \
  'python run_experiment.py --iteration {} --exp-root "'"$EXP_ROOT"'" > logs/exp_{}.log 2>&1 && echo "exp {} ok" || echo "exp {} FAILED"'
python collect.py
