# %%
import os
import sys

import arviz as az
import dill  # Better serialization than pickle
import matplotlib.pyplot as plt

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


# ! -------------------- Deal with 10^4 observations --------------------
# %%
with open("../data/dadras_model_n_104.dill", "rb") as f:
    model_n = dill.load(f)

model = model_n.results["equation_1"]["model"]
results = model_n.results["equation_1"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

loo_104 = az.loo(
    results, pointwise=True
)  # pointwise=True gives per-observation stuff
khat_104 = loo_104.pareto_k

# ! -------------------- Deal with 10^5 observations --------------------
# %%
with open("../data/dadras_model_n_105.dill", "rb") as f:
    model_n = dill.load(f)

model = model_n.results["equation_1"]["model"]
results = model_n.results["equation_1"]["results"]
del model_n  # Free up memory
model.predict(results, kind="response")

loo_105 = az.loo(
    results, pointwise=True
)  # pointwise=True gives per-observation stuff
khat_105 = loo_105.pareto_k


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
# Create combined plot with 1x2 layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot for 10^4 observations (left subplot)
n_points_104 = len(khat_104)
x_104 = range(n_points_104)
# Plot points < 0.7 in blue
ax1.scatter(
    [i for i in x_104 if khat_104[i] < 0.7],
    [khat_104[i] for i in x_104 if khat_104[i] < 0.7],
    color="#116DA9",
    label="k < 0.7",
    s=8,  # smaller point size
    marker="o",  # circle marker
    alpha=0.6,  # slight transparency
)
# Plot points >= 0.7 in red
ax1.scatter(
    [i for i in x_104 if khat_104[i] >= 0.7],
    [khat_104[i] for i in x_104 if khat_104[i] >= 0.7],
    color="#B03C2B",
    label="k ≥ 0.7",
    s=14,  # slightly larger than the blue points
    marker="^",  # triangle marker
    alpha=0.9,  # more visible
)
ax1.axhline(y=0.7, color="#9F0000", linestyle="--", alpha=0.5)
ax1.set_xlabel("Observations", fontsize=22)
ax1.set_ylabel("Pareto k", fontsize=22, labelpad=12)
ax1.tick_params(
    axis="y", pad=20
)  # adjust distance between ticks and tick labels

# ax1.set_title("Pareto k Diagnostic Values ($10^4$ observations)", fontsize=26)
ax1.set_title("$n = 10^4$", fontsize=26)
ax1.legend()

# Plot for 10^5 observations (right subplot)
n_points_105 = len(khat_105)
x_105 = range(n_points_105)
# Plot points < 0.7 in blue
ax2.scatter(
    [i for i in x_105 if khat_105[i] < 0.7],
    [khat_105[i] for i in x_105 if khat_105[i] < 0.7],
    color="#116DA9",
    label="k < 0.7",
    s=8,  # smaller point size
    marker="o",  # circle marker
    alpha=0.6,  # slight transparency
)
# Plot points >= 0.7 in red
ax2.scatter(
    [i for i in x_105 if khat_105[i] >= 0.7],
    [khat_105[i] for i in x_105 if khat_105[i] >= 0.7],
    color="#B03C2B",
    label="k ≥ 0.7",
    s=14,  # slightly larger than the blue points
    marker="^",  # triangle marker
    alpha=0.9,  # more visible
)
ax2.axhline(y=0.7, color="#9F0000", linestyle="--", alpha=0.5)
ax2.set_xlabel("Observations", fontsize=22)
ax2.set_ylabel(
    "Pareto k",
    fontsize=22,
    # labelpad=12
)  # increase/decrease labelpad
# ax2.tick_params(
#     axis="y",
#     pad=10
# )  # adjust distance between ticks and tick labels
ax2.set_title("$n = 10^5$", fontsize=26)
ax2.legend(markerscale=2.0)

plt.tight_layout()

fig.text(
    0.02,
    0.98,
    "b",
    transform=fig.transFigure,
    fontsize=32,
    fontweight="bold",
    va="top",
    ha="left",
)

# %%
fig.savefig(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "results-diagnostics",
        "figures",
        "loo_plot.pdf",
    ),
    dpi=600,
)

# %%
