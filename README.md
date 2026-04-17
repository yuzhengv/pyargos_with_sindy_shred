# PyARGOS with SINDy-SHRED

This is the code repository for the manuscript **Fast and principled equation discovery from chaos to climate** ([arXiv:2604.11929](https://arxiv.org/abs/2604.11929)). It contains the Python implementation of Bayesian-ARGOS, its integration with SINDy-SHRED for equation discovery from spatiotemporal sensor data, and diagnostic analyses.
For the R version of Bayesian-ARGOS and the related performance evaluation, please see the companion repository [Bayesian-ARGOS](https://github.com/yuzhengv/Bayesian-ARGOS).

## Repository Structure

### `pyargos/` — Bayesian-ARGOS

The core Bayesian-ARGOS implementation for sparse identification of dynamical systems.

- **`src/`** — Core algorithms
  - `argos_bayesian_argos.py` — Main `BayesianArgos` class orchestrating the identification pipeline
  - `argos_sparse_regression.py` — Frequentist regression module
  - `argos_bayesian_regression.py` — Bayesian model fitting via Bambi
  - `adelie_custom.py` — Custom grouped elastic net solver using ADELIE
  - `bambi_prior_custom.py` — Custom prior specification for Bayesian regression
  - `argos_standardize.py` — Data standardization utilities
- **`utils/`** — Supporting utilities
  - `argos_simulator.py` — ODE simulation and synthetic data generation (Lorenz, Rossler, Aizawa, Dadras, etc.)
  - `argos_utils.py` — Design matrix construction and Savitzky-Golay smoothing
- **`results-diagnostics/`** — Diagnostic analyses including variance inflation factor (VIF), leave-one-out cross-validation (LOO-CV), posterior predictive checks, and process analysis
- **`simulate-trajectories/`** — Trajectory visualization scripts
- `test_bayesian_argos.py` — Example script demonstrating the Bayesian-ARGOS pipeline

### `sindy-shred-exp/` — SINDy-SHRED Experiments

Integration of Bayesian-ARGOS and SINDy-SHRED architecture for equation discovery from spatiotemporal data (e.g., sea surface temperature).

- **`src/`**
  - `sindy_shred.py` — SINDy-SHRED model architecture
  - `sindy.py` — SINDy library functions for feature library construction
- **`utils/`** — Data processing utilities
  - `processdata.py` — Data loading, preprocessing, and QR-pivoting sensor placement
  - `sindy_utils.py` — SINDy fitting and simulation helpers
- **Training and analysis scripts**
  - `training_script.py` — Main training script for the SINDy-SHRED model
  - `sst_sindy_shred.py` — Sea surface temperature (SST) application pipeline
  - `analysis_script_ba.py` — Post-training analysis with Bayesian-ARGOS
  - `analysis_script_sindy.py` — Post-training analysis with SINDy
- **`results_analysis/`** — Result post-processing, performance assessment, and figure generation
- **`exp-1/`, `exp-ba/`, `exp-sindy/`** — Saved experimental results (trained models, MSE metrics, comparison plots)
- Shell scripts (`run_*.sh`, `submit_*.sh`) for HPC/SLURM job submission

## Installation

Create the conda environment from the provided environment file:

```bash
# For Linux with CUDA 12.4
conda env create -f environment.yml

# For macOS
conda env create -f environment_mac.yml
```

Then activate the environment:

```bash
conda activate pyargos-shred-dev
```

### Key Dependencies

- Python >= 3.10, NumPy, SciPy, scikit-learn, pandas
- PyTorch (with CUDA support on Linux)
- JAX
- Adelie (for adaptive lasso regression)
- Bambi, ArviZ (Bayesian inference)
- PySINDy

<!--  -->
