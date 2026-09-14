#!/usr/bin/env bash
# Regenerate Fig. 8b, Fig. 8c (full globe) and Fig. S9 for experiment 18 (Referee 1, Point 8).
# Needs a checkout with Data/SST_data.mat, exp-1/sensor-location/ and exp-1/saved-model/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")"
EXP_ROOT="${1:-$HOME/Projects/pyargos_with_sindy_shred/sindy-shred-exp}"
python regenerate_figs.py --exp-root "$EXP_ROOT"                    # eval mode: the revision figures
python regenerate_figs.py --exp-root "$EXP_ROOT" --dropout-active   # submitted behaviour, for comparison only
