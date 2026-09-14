"""Fig. 8a regenerated from the batch tables (eval-mode decoding, fixed pipeline).

Draws the violin plot of decoded-forecast RMSE per time period exactly as
`results_analysis/performance_assessment.py` does (same seaborn call, colours,
mean markers, quadratic guide line, legend with valid-case counts), but from
`tables/experiments.csv` written by `collect.py`: degree-1 models, stage1 &
valid runs, time ranges (0, 100), (100, 200), (200, 275) weeks.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
RANGES = ["(0, 100)", "(100, 200)", "(200, 275)"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default=os.path.join(HERE, "tables", "experiments.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "figures", "rmse_comparison_violin_plot.pdf"))
    a = p.parse_args()
    df = pd.read_csv(a.table)
    rows = []
    for tag, name in [("ba_deg1", "Bayesian-ARGOS"), ("sindy_deg1", "SINDy")]:
        g = df[(df.model == tag) & df.stage1 & df.valid]
        for _, r in g.iterrows():
            for rng in RANGES:
                s, e = rng.strip("()").split(", ")
                rows.append({"Time_Range": rng, "RMSE": r[f"RMSE_{s}_{e}"], "Forecaster": name, "Index": int(r.iteration)})
    combined_rmse_df = pd.DataFrame(rows)
    counts = combined_rmse_df.groupby("Forecaster").Index.nunique()

    # ---- plotting block from performance_assessment.py (unchanged) ----
    plt.figure(figsize=(13.5, 13), dpi=600)
    ax = sns.violinplot(
        data=combined_rmse_df, x="Time_Range", y="RMSE", hue="Forecaster", split=False,
        palette={"Bayesian-ARGOS": "#505ed5", "SINDy": "#8eb63d"}, alpha=0.6,
    )
    ax.tick_params(axis="both", which="major", labelsize=24)
    means_df = combined_rmse_df.groupby(["Time_Range", "Forecaster"])["RMSE"].mean().reset_index()
    time_ranges = combined_rmse_df["Time_Range"].unique()
    x_positions = range(len(time_ranges))
    for forecaster in ["Bayesian-ARGOS", "SINDy"]:
        forecaster_means = means_df[means_df["Forecaster"] == forecaster]["RMSE"].values
        if forecaster == "Bayesian-ARGOS":
            color, offset = "#505ed5", -0.2
        else:
            color, offset = "#8eb63d", 0.2
        x_pos_adj = [x + offset for x in x_positions[: len(forecaster_means)]]
        z = np.polyfit(x_pos_adj, forecaster_means, 2)
        pfit = np.poly1d(z)
        plt.plot(x_pos_adj, pfit(x_pos_adj), color=color, linestyle="--", linewidth=2.5, alpha=1)
        plt.scatter(x_pos_adj, forecaster_means, color=color, s=100, zorder=3, marker="D", edgecolors="black", linewidth=1)
        for x_pos, mean_val in zip(x_pos_adj, forecaster_means):
            plt.annotate(f"{mean_val:.4f}", (x_pos, mean_val), textcoords="offset points", xytext=(0, 15),
                         ha="center", va="bottom", fontsize=22, color="black",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color))
    plt.text(-0.075, 1, "a", transform=ax.transAxes, fontsize=46, fontweight="bold", va="bottom", ha="right")
    plt.xlabel("Time Period", fontsize=30)
    plt.ylabel("RMSE", fontsize=30)
    custom_label_map = {
        "Bayesian-ARGOS": f"Bayesian-ARGOS ({int(counts['Bayesian-ARGOS'])} valid cases)",
        "SINDy": f"SINDy ({int(counts['SINDy'])} valid cases)",
    }
    handles, labels = ax.get_legend_handles_labels()
    labels = [custom_label_map.get(label, label) for label in labels]
    ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2, fontsize=26.25, title=None)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=600, bbox_inches="tight")
    print("saved", a.out)
    print(means_df.pivot(index="Time_Range", columns="Forecaster", values="RMSE").round(4).to_string())
    print("valid cases:", counts.to_dict())


if __name__ == "__main__":
    main()
