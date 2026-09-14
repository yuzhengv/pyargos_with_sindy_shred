"""Fig. 8b: GRU latent trajectory vs Bayesian-ARGOS simulation.

The plotting block of `results_analysis/analysis_script_ba.py` (training
period, `comparison_plot_ba_<it>`), unchanged except that it takes NumPy
arrays and a file name.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_latent_comparison(gru_outs, x_sim, save_path, latent_dim=3):
    min_length = min(len(gru_outs), len(x_sim))
    fig, axes = plt.subplots(
        latent_dim, 1, figsize=(12, latent_dim * 3.5), dpi=600, sharex=True, constrained_layout=True,
    )
    if latent_dim == 1:
        axes = [axes]
    gru_color = "#6a51a3"
    ba_color = "#505ed5"
    for i in range(latent_dim):
        axes[i].plot(gru_outs[:min_length, i], linewidth=2, color=gru_color, alpha=0.8, label="GRU Output")
        axes[i].plot(x_sim[:min_length, i], linestyle="--", linewidth=2, color=ba_color, alpha=0.8,
                     label="Bayesian-ARGOS Simulation")
        axes[i].set_ylabel(f"$z_{{{i + 1}}}$", fontsize=30, fontweight="bold")
        axes[i].grid(True, alpha=0.3, linestyle="--")
        axes[i].tick_params(axis="both", which="major", labelsize=24)
    axes[-1].set_xlabel("Time Index (Weeks)", fontsize=30)
    fig.text(-0.01, 1.03, "b", fontsize=44, fontweight="bold", ha="left", va="top")
    legend_lines = [
        Line2D([0], [0], color=gru_color, lw=2, label="GRU Output"),
        Line2D([0], [0], color=ba_color, lw=2, linestyle="--", label="Bayesian-ARGOS Simulation"),
    ]
    fig.legend(handles=legend_lines, loc="upper center", bbox_to_anchor=(0.54, 0), ncol=2, fontsize=26)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent", default="results/latent_train_18.npz")
    p.add_argument("--out", default="figures/comparison_plot_ba_18.pdf")
    a = p.parse_args()
    d = np.load(a.latent)
    plot_latent_comparison(d["gru"], d["sim"], a.out, latent_dim=d["gru"].shape[1])
    print("saved", a.out)


if __name__ == "__main__":
    main()
