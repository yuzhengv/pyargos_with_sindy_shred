# %%
import os
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# import utils.argos_simulator as ags
# import utils.argos_utils as agu
# from src.argos_bayesian_argos import BayesianArgos

# ! -------------------- Aizawa with snr 50 observations--------------------
# %%
with open(
    "../../data/snr-analysis-aizawa-50-cprior/aizawa_model_snr_50.dill", "rb"
) as f:
    model_n = dill.load(f)

model = model_n.results["equation_3"]["model"]
results = model_n.results["equation_3"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

# %%
y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean

# Store data from infinite SNR case
pp_mean_shorter = pp_mean.copy()
residuals_shorter = residuals.copy()

# ! -------------------- Aizawa with snr 60 observations--------------------
# %%
with open(
    "../../data/snr-analysis-aizawa-50-cprior/aizawa_model_snr_60.dill", "rb"
) as f:
    model_n = dill.load(f)

model = model_n.results["equation_3"]["model"]
results = model_n.results["equation_3"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

# %%
y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean

# %%
plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "serif",
        # "font.serif": ["Times New Roman", "Computer Modern Roman"],
        "font.serif": ["sans-serif", "DejaVu Serif", "serif"],
        "text.usetex": False,
        "axes.linewidth": 1.2,
        "axes.labelweight": "normal",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }
)

# %%
# Create subplot layout: 1 row, 2 columns
fig_residuals, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.5, 6.75))

# Plot infinite SNR case (left subplot)
ax1.scatter(pp_mean_shorter, residuals_shorter, s=1, color="#116DA9", alpha=0.7)
z_inf = np.polyfit(pp_mean_shorter, residuals_shorter, 3)
p_inf = np.poly1d(z_inf)
ax1.plot(
    pp_mean_shorter,
    p_inf(pp_mean_shorter),
    color="#634564",
    linestyle="-",
    alpha=0.8,
    linewidth=1.5,
)
ax1.axhline(0, color="#9F0000", linestyle="--", alpha=0.6)
ax1.set_xlabel("Posterior Predictive Mean", fontsize=22)
ax1.set_ylabel("Residuals", fontsize=22)
ax1.set_title(r"SNR = $50$", fontsize=26)

ax2.scatter(pp_mean, residuals, s=1, color="#116DA9", alpha=0.7)
z = np.polyfit(pp_mean, residuals, 3)
p = np.poly1d(z)
ax2.plot(
    pp_mean,
    p(pp_mean),
    color="#634564",
    linestyle="-",
    alpha=0.8,
    linewidth=1.5,
)
ax2.axhline(0, color="#9F0000", linestyle="--", alpha=0.6)
ax2.set_xlabel("Posterior Predictive Mean", fontsize=22)
ax2.set_ylabel("Residuals", fontsize=22)
ax2.set_title(r"SNR = $60$", fontsize=26)

fig_residuals.tight_layout()

# %%
fig_residuals.savefig(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "figures",
        "diagnostics-cprior",
        "aizawa_50_50_60_ppmean_vs_residuals.pdf",
    ),
    dpi=600,
)

# %%
