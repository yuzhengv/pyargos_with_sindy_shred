#!/usr/bin/env bash
set -euo pipefail
# DGX Spark (aarch64, GB10, CUDA 13) environment for pyargos_with_sindy_shred.
#
# Follows AgenticARGOS/environment/create_uv_vllm_env.sh. Two things differ
# from the shipped environment.yml:
#   * torch comes from the cu130 index (the cu124 index has no aarch64 wheels);
#   * adelie has no aarch64 wheel and upstream fails to compile here, so it is
#     built from the patched source tree in AgenticARGOS with --no-deps (its
#     numpy<2 cap conflicts with the rest of the stack).
# jax/bayeux/flowmc/gurobipy/torchdyn from environment.yml are dropped: nothing
# in this repo imports them.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/" && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT/.venv-spark}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
ADELIE_SRC="${ADELIE_SRC:-$HOME/Projects/AgenticARGOS/environment/adelie}"
EIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR:-/usr/include/eigen3}"

if [[ ! -f "$ADELIE_SRC/setup.py" ]]; then
  echo "Patched adelie source not found at $ADELIE_SRC (set ADELIE_SRC)" >&2
  exit 1
fi
if [[ ! -d "$EIGEN3_INCLUDE_DIR/Eigen" ]]; then
  echo "Eigen headers not found at $EIGEN3_INCLUDE_DIR (apt install libeigen3-dev, or set EIGEN3_INCLUDE_DIR)" >&2
  exit 1
fi

echo "=== pyargos_with_sindy_shred DGX Spark env ==="
echo "venv:    $VENV_PATH"
echo "python:  $PYTHON_VERSION"
echo "adelie:  $ADELIE_SRC"
echo "eigen:   $EIGEN3_INCLUDE_DIR"

# --managed-python: never anaconda's interpreter (its RPATH pulls an old
# libstdc++ that breaks JIT-compiled CUDA kernels).
rm -rf "$VENV_PATH"
uv venv --python "$PYTHON_VERSION" --managed-python "$VENV_PATH"
PYTHON_BIN="$VENV_PATH/bin/python"

echo "--- core deps"
uv pip install --python "$PYTHON_BIN" \
  --index-strategy unsafe-best-match \
  --extra-index-url "$PYTORCH_INDEX_URL" \
  -r "$ROOT/requirements.spark.txt"

echo "--- adelie from patched source"
uv pip install --python "$PYTHON_BIN" setuptools wheel pybind11
export CPLUS_INCLUDE_PATH="${EIGEN3_INCLUDE_DIR}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
rm -rf "$ADELIE_SRC/build" "$ADELIE_SRC/dist"
find "$ADELIE_SRC" -name "*.so" -delete
uv pip install --python "$PYTHON_BIN" --no-build-isolation --no-deps "$ADELIE_SRC"

echo "--- import check"
# Run from the venv dir so the adelie source tree never shadows the install.
(cd "$VENV_PATH" && "$PYTHON_BIN" -c "
import adelie, arviz, bambi, pymc, pytensor, nutpie, pysindy, torch, dill, seaborn
print(f'  torch:   {torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'  adelie:  {adelie.__version__}')
print(f'  pysindy: {pysindy.__version__}')
print('  all imports OK')
")

echo
echo "Activate with:  source $VENV_PATH/bin/activate"
