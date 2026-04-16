# %%
import os
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ! -------------------- Load Left Panel System's Data --------------------
# %%
with open(
    "../../data/snr-analysis-rossler-cprior/rossler_model_snr_61.dill", "rb"
) as f:
    model_n = dill.load(f)

model = model_n.results["equation_1"]["model"]
results = model_n.results["equation_1"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean

pp_mean_left = pp_mean.copy()
residuals_left = residuals.copy()

# ! -------------------- Load Right Panel System's Data --------------------
# %%
with open(
    "../../data/snr-analysis-rossler-cprior/rossler_model_snr_62.dill", "rb"
) as f:
    model_n = dill.load(f)

model = model_n.results["equation_1"]["model"]
results = model_n.results["equation_1"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

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
fig_residuals, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10.5))

# Plot infinite SNR case (left subplot)
ax1.scatter(pp_mean_left, residuals_left, s=1, color="#116DA9", alpha=0.7)
z_inf = np.polyfit(pp_mean_left, residuals_left, 3)
p_inf = np.poly1d(z_inf)
ax1.plot(
    pp_mean_left,
    p_inf(pp_mean_left),
    color="#634564",
    linestyle="-",
    alpha=0.8,
    linewidth=1.5,
)
ax1.axhline(0, color="#9F0000", linestyle="--", alpha=0.6)
ax1.set_xlabel("Posterior Predictive Mean", fontsize=22)
ax1.set_ylabel("Residuals", fontsize=22)
ax1.set_title(r"SNR = $61$", fontsize=26)

# Plot SNR=61 case (right subplot)
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
ax2.set_title(r"SNR = $\infty$", fontsize=26)

fig_residuals.tight_layout()

# Add label "b" to top left corner
fig_residuals.text(
    0.02,
    0.98,
    "c",
    transform=fig_residuals.transFigure,
    fontsize=32,
    fontweight="bold",
    va="top",
    ha="left",
)

# %%
fig_residuals.savefig(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "figures",
        "diagnostics-cprior",
        "rossler_61_62_ppmean_vs_residuals.svg",
    ),
    dpi=300,
)

# %%
