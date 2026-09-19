#!/usr/bin/env bash
# Latent dimension 2 vs 3: retrain r = 2 networks on the sensor sets of experiments FIRST..LAST,
# identify the latent dynamics on each with the batch script, then aggregate.
# Usage: ./run_batch.sh [EXP_ROOT] [FIRST] [LAST] [JOBS] [LATENT_DIM]
set -uo pipefail
cd "$(dirname "$0")"
EXP_ROOT="${1:-$HOME/Projects/pyargos_with_sindy_shred/sindy-shred-exp}"
FIRST="${2:-0}"; LAST="${3:-19}"; JOBS="${4:-2}"; R="${5:-2}"
mkdir -p logs models results
seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c \
  'python train_latent_dim.py --iteration {} --latent-dim '"$R"' --exp-root "'"$EXP_ROOT"'" > logs/train_{}_r'"$R"'.log 2>&1 && echo "train {} ok" || echo "train {} FAILED"'
seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c \
  'python ../rebuttal-latent-models/run_experiment.py --iteration {} --latent-dim '"$R"' --model-path models/model_{}_r'"$R"'.pth --degrees 1 --out results/identification_r'"$R"' --exp-root "'"$EXP_ROOT"'" > logs/ident_{}_r'"$R"'.log 2>&1 && echo "ident {} ok" || echo "ident {} FAILED"'
python collect.py --latent-dim "$R"
