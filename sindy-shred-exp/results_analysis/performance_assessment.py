# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "analysis-functions"))

import analysis_functions as af

# %%
dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
model_dir = os.path.join(dirpath, "exp-1")
exp_seneor_dir = os.path.join(model_dir, "sensor-location")
exp_saved_model_dir = os.path.join(model_dir, "saved-model")

# Path settings for SINDy experiment
exp_sindy_dir = os.path.join(dirpath, "exp-sindy")
exp_compare_sindy_dir = os.path.join(exp_sindy_dir, "comparison-plots-sindy")
recon_rmse_test_sindy_dir = os.path.join(exp_sindy_dir, "recon-rmse-test")
latent_mse_train_sindy_dir = os.path.join(
    exp_sindy_dir, "latent-mse-sindy-train"
)
latent_mse_test_sindy_dir = os.path.join(exp_sindy_dir, "latent-mse-sindy-test")
recon_metrics_via_sindy_dir = os.path.join(
    exp_sindy_dir, "recon-via-forecaster"
)

# Path settings for Pyargos experiment
exp_ba_dir = os.path.join(dirpath, "exp-ba")
exp_compare_ba_dir = os.path.join(exp_ba_dir, "comparison-plots-ba")
recon_rmse_test_ba_dir = os.path.join(exp_ba_dir, "recon-rmse-test")
latent_mse_train_ba_dir = os.path.join(exp_ba_dir, "latent-mse-ba-train")
latent_mse_test_ba_dir = os.path.join(exp_ba_dir, "latent-mse-ba-test")
recon_metrics_via_ba_dir = os.path.join(exp_ba_dir, "recon-via-forecaster")

# %%
# Filter out the results whose RMSE of the reconstruction results obtained from SINDy-SHRED network
# on the test set is larger than 0.05
recon_rmse_test = af.get_recon_rmse_test(recon_rmse_test_ba_dir)

(
    filtered_results_on_recon_rmse,
    retained_file_numbers_on_recon_rmse,
    filtering_stats_on_recon_rmse,
) = af.filter_recon_rmse_test_results(recon_rmse_test)

# ! -------------------- BA Results --------------------
# %%
# Load MSE results into DataFrame
latent_mse_train_ba = af.load_latent_mse_to_dataframe(
    latent_mse_train_ba_dir, data_type="train", method="ba"
)

latent_mse_test_ba = af.load_latent_mse_to_dataframe(
    latent_mse_test_ba_dir, data_type="test", method="ba"
)

# Filter the MSE dataframe
latent_mse_df_filtered_ba, latent_mse_df_valid_ba = (
    af.filter_the_latent_mse_based_on_recon_rmse(
        latent_mse_train_ba, retained_file_numbers_on_recon_rmse
    )
)
af.display_performance_metrics(latent_mse_df_filtered_ba)

# %%
# Calculate the reconstruction mse via the latent forecaster based on the index of latent_mse_df_filtered_ba
forecaster_metrics_df_filtered_ba, forecaster_metrics_df_ba = (
    af.load_and_filter_forecaster_metrics(
        dir_path=recon_metrics_via_ba_dir,
        method="ba",
        valid_indices=latent_mse_df_filtered_ba.index,
    )
)

# %%
# Create a DataFrame with the filtered forecaster metrics for the following plotting
ba_RMSE_normalized_df, ba_RMSE_normalized_plot_df = (
    af.extract_forecaster_metrics_columns(
        forecaster_metric_df=forecaster_metrics_df_filtered_ba,
        metric_pattern="RMSE_normalized",
        forecater="Bayesian-ARGOS",
        excluded_ranges=["(0, 275)"],  # Add more tuples as needed
    )
)

# %%
# Get summary of different metrics for the reconstruction via forecaster
metric_df_via_ba = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_ba, method="ba"
)

nrmse_metric_df_via_ba = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_ba,
    column_filter="RMSE_normalized",
    exclude_normalized=False,
    method="ba",
)

# %%
# ! -------------------- Sindy Results --------------------
# Load MSE results into DataFrame
latent_mse_train_sindy = af.load_latent_mse_to_dataframe(
    latent_mse_train_sindy_dir, data_type="train", method="sindy"
)

latent_mse_test_sindy = af.load_latent_mse_to_dataframe(
    latent_mse_test_sindy_dir, data_type="test", method="sindy"
)

# Filter the MSE dataframe
latent_mse_df_filtered_sindy, latent_mse_df_valid_sindy = (
    af.filter_the_latent_mse_based_on_recon_rmse(
        latent_mse_train_sindy, retained_file_numbers_on_recon_rmse
    )
)
af.display_performance_metrics(latent_mse_df_filtered_sindy)

# %%
# Calculate the reconstruction mse via the latent forecaster based on the index of latent_mse_df_filtered_ba
forecaster_metrics_df_filtered_sindy, forecaster_metrics_df_sindy = (
    af.load_and_filter_forecaster_metrics(
        dir_path=recon_metrics_via_sindy_dir,
        method="sindy",
        valid_indices=latent_mse_df_filtered_sindy.index,
    )
)

# %%
# Create a DataFrame with the filtered forecaster metrics for the following plotting
sindy_RMSE_normalized_df, sindy_RMSE_normalized_plot_df = (
    af.extract_forecaster_metrics_columns(
        forecaster_metric_df=forecaster_metrics_df_filtered_sindy,
        metric_pattern="RMSE_normalized",
        forecater="SINDy",
        excluded_ranges=["(0, 275)"],  # Add more tuples as needed
    )
)
# %%
metric_df_via_sindy = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_sindy, method="sindy"
)

nrmse_metric_df_via_sindy = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_sindy,
    column_filter="RMSE_normalized",
    exclude_normalized=False,
    method="sindy",
)

# ! -------------------- Make comparison between the metrics of SINDy and Bayesian ARGOS --------------------
# %%
# Combine both dataframes for comparison plotting
combined_rmse_df = pd.concat(
    [ba_RMSE_normalized_plot_df, sindy_RMSE_normalized_plot_df],
    ignore_index=True,
)

# %%
# Create violin plot with regression lines
plt.figure(figsize=(12, 8.9), dpi=600)

# Create the violin plot
ax = sns.violinplot(
    data=combined_rmse_df,
    x="Time_Range",
    y="RMSE_normalized",
    hue="Forecaster",
    split=False,
    # inner="quart",
    # inner_kws=dict(box_width=15, whis_width=2, color=".8"),
    palette={
        "Bayesian-ARGOS": "#505ed5",
        "SINDy": "#8eb63d",
    },  # RGBA format with custom alpha
    alpha=0.6,  # Add alpha parameter to control transparency (0.0 to 1.0)
)

ax.tick_params(axis="both", which="major", labelsize=14)

# Calculate means for regression lines
means_df = (
    combined_rmse_df.groupby(["Time_Range", "Forecaster"])["RMSE_normalized"]
    .mean()
    .reset_index()
)

# Create numeric x-positions for regression lines
time_ranges = combined_rmse_df["Time_Range"].unique()
x_positions = range(len(time_ranges))

# Plot regression lines for each forecaster
for forecaster in ["Bayesian-ARGOS", "SINDy"]:
    forecaster_means = means_df[means_df["Forecaster"] == forecaster][
        "RMSE_normalized"
    ].values
    if forecaster == "Bayesian-ARGOS":
        color = "#505ed5"
        offset = -0.2
    else:
        color = "#8eb63d"
        offset = 0.2

    # Adjust x positions slightly for better visualization
    x_pos_adj = [x + offset for x in x_positions[: len(forecaster_means)]]

    # Plot regression line
    z = np.polyfit(x_pos_adj, forecaster_means, 2)
    p = np.poly1d(z)
    plt.plot(
        x_pos_adj,
        p(x_pos_adj),
        color=color,
        linestyle="--",
        linewidth=2.5,
        # label=f"{forecaster.upper()} trend",
        alpha=1,
    )

    # Plot mean points
    plt.scatter(
        x_pos_adj,
        forecaster_means,
        color=color,
        s=100,
        zorder=3,
        marker="D",
        edgecolors="black",
        linewidth=1,
    )

    # Add text annotations for mean values
    for i, (x_pos, mean_val) in enumerate(zip(x_pos_adj, forecaster_means)):
        plt.annotate(
            f"{mean_val:.4f}",
            (x_pos, mean_val),
            textcoords="offset points",
            xytext=(0, 15),  # 15 points above the point
            ha="center",
            va="bottom",
            fontsize=20,
            # fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor=color,
            ),
        )

# plt.title(
#     "Normalized RMSE Comparison: Bayesian-ARGOS versus SINDy Forecasters",
#     fontsize=14,
#     # fontweight="bold",
# )

# Add subplot label "a" to upper left corner
plt.text(
    -0.05,
    1.03,
    "a",
    transform=ax.transAxes,
    fontsize=32.5,
    fontweight="bold",
    va="bottom",
    ha="right",
)

plt.xlabel("Time Period", fontsize=18.75)
plt.ylabel("Normalized RMSE", fontsize=18.75)
# Customize legend labels with valid-case counts
ba_valid_cases = int(ba_RMSE_normalized_plot_df["Index"].nunique())
sindy_valid_cases = int(sindy_RMSE_normalized_plot_df["Index"].nunique())
custom_label_map = {
    "Bayesian-ARGOS": f"Bayesian-ARGOS ({ba_valid_cases} valid cases)",
    "SINDy": f"SINDy ({sindy_valid_cases} valid cases)",
}
handles, labels = ax.get_legend_handles_labels()
labels = [custom_label_map.get(label, label) for label in labels]
ax.legend(
    handles,
    labels,
    bbox_to_anchor=(0.5, -0.1),
    loc="upper center",
    ncol=2,
    fontsize=20,
    title=None,
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plot_save_path = os.path.join(
    dirpath, "results_analysis/imgs/nrmse_comparison_violin_plot.png"
)
plt.savefig(plot_save_path, dpi=600, bbox_inches="tight")

# Print summary statistics
print("Summary Statistics by Forecaster and Time Range:")
print(
    combined_rmse_df.groupby(["Forecaster", "Time_Range"])[
        "RMSE_normalized"
    ].agg(["mean", "std", "median", "count"])
)
plt.show()

# : -------------------- Make plot based on RMSE --------------------
# %%
# Filter out the results whose RMSE of the reconstruction results obtained from SINDy-SHRED network
# on the test set is larger than 0.05
recon_rmse_test = af.get_recon_rmse_test(recon_rmse_test_ba_dir)

(
    filtered_results_on_recon_rmse,
    retained_file_numbers_on_recon_rmse,
    filtering_stats_on_recon_rmse,
) = af.filter_recon_rmse_test_results(recon_rmse_test)

# ! -------------------- BA Results --------------------
# %%
# Load MSE results into DataFrame
latent_mse_train_ba = af.load_latent_mse_to_dataframe(
    latent_mse_train_ba_dir, data_type="train", method="ba"
)

latent_mse_test_ba = af.load_latent_mse_to_dataframe(
    latent_mse_test_ba_dir, data_type="test", method="ba"
)

# Filter the MSE dataframe
latent_mse_df_filtered_ba, latent_mse_df_valid_ba = (
    af.filter_the_latent_mse_based_on_recon_rmse(
        latent_mse_train_ba, retained_file_numbers_on_recon_rmse
    )
)
af.display_performance_metrics(latent_mse_df_filtered_ba)

# %%
# Calculate the reconstruction mse via the latent forecaster based on the index of latent_mse_df_filtered_ba
forecaster_metrics_df_filtered_ba, forecaster_metrics_df_ba = (
    af.load_and_filter_forecaster_metrics(
        dir_path=recon_metrics_via_ba_dir,
        method="ba",
        valid_indices=latent_mse_df_filtered_ba.index,
    )
)

# %%
# Create a DataFrame with the filtered forecaster metrics for the following plotting
ba_RMSE_df, ba_RMSE_plot_df = af.extract_forecaster_metrics_columns(
    forecaster_metric_df=forecaster_metrics_df_filtered_ba,
    metric_pattern="RMSE",
    forecater="Bayesian-ARGOS",
    excluded_ranges=["(0, 275)"],  # Add more tuples as needed
)

# %%
# Get summary of different metrics for the reconstruction via forecaster
metric_df_via_ba = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_ba, method="ba"
)

rmse_metric_df_via_ba = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_ba,
    column_filter="RMSE",
    exclude_normalized=True,
    method="ba",
)

# %%
# ! -------------------- Sindy Results --------------------
# Load MSE results into DataFrame
latent_mse_train_sindy = af.load_latent_mse_to_dataframe(
    latent_mse_train_sindy_dir, data_type="train", method="sindy"
)

latent_mse_test_sindy = af.load_latent_mse_to_dataframe(
    latent_mse_test_sindy_dir, data_type="test", method="sindy"
)

# Filter the MSE dataframe
latent_mse_df_filtered_sindy, latent_mse_df_valid_sindy = (
    af.filter_the_latent_mse_based_on_recon_rmse(
        latent_mse_train_sindy, retained_file_numbers_on_recon_rmse
    )
)
af.display_performance_metrics(latent_mse_df_filtered_sindy)

# %%
# Calculate the reconstruction mse via the latent forecaster based on the index of latent_mse_df_filtered_ba
forecaster_metrics_df_filtered_sindy, forecaster_metrics_df_sindy = (
    af.load_and_filter_forecaster_metrics(
        dir_path=recon_metrics_via_sindy_dir,
        method="sindy",
        valid_indices=latent_mse_df_filtered_sindy.index,
    )
)

# %%
# Create a DataFrame with the filtered forecaster metrics for the following plotting
sindy_RMSE_df, sindy_RMSE_plot_df = af.extract_forecaster_metrics_columns(
    forecaster_metric_df=forecaster_metrics_df_filtered_sindy,
    metric_pattern="RMSE",
    forecater="SINDy",
    excluded_ranges=["(0, 275)"],  # Add more tuples as needed
)
# %%
metric_df_via_sindy = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_sindy, method="sindy"
)

rmse_metric_df_via_sindy = af.get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered_sindy,
    column_filter="RMSE",
    exclude_normalized=True,
    method="sindy",
)

# ! -------------------- Make comparison between the metrics of SINDy and Bayesian ARGOS --------------------
# %%
# Combine both dataframes for comparison plotting
combined_rmse_df = pd.concat(
    [ba_RMSE_plot_df, sindy_RMSE_plot_df],
    ignore_index=True,
)

# %%
# Create violin plot with regression lines
plt.figure(figsize=(13.5, 13), dpi=600)

# Create the violin plot
ax = sns.violinplot(
    data=combined_rmse_df,
    x="Time_Range",
    y="RMSE",
    hue="Forecaster",
    split=False,
    # inner="quart",
    # inner_kws=dict(box_width=15, whis_width=2, color=".8"),
    palette={
        "Bayesian-ARGOS": "#505ed5",
        "SINDy": "#8eb63d",
    },  # RGBA format with custom alpha
    alpha=0.6,  # Add alpha parameter to control transparency (0.0 to 1.0)
)

ax.tick_params(axis="both", which="major", labelsize=24)

#
# Calculate means for regression lines
means_df = (
    combined_rmse_df.groupby(["Time_Range", "Forecaster"])["RMSE"]
    .mean()
    .reset_index()
)

# Create numeric x-positions for regression lines
time_ranges = combined_rmse_df["Time_Range"].unique()
x_positions = range(len(time_ranges))

# Plot regression lines for each forecaster
for forecaster in ["Bayesian-ARGOS", "SINDy"]:
    forecaster_means = means_df[means_df["Forecaster"] == forecaster][
        "RMSE"
    ].values
    if forecaster == "Bayesian-ARGOS":
        color = "#505ed5"
        offset = -0.2
    else:
        color = "#8eb63d"
        offset = 0.2

    # Adjust x positions slightly for better visualization
    x_pos_adj = [x + offset for x in x_positions[: len(forecaster_means)]]

    # Plot regression line
    z = np.polyfit(x_pos_adj, forecaster_means, 2)
    p = np.poly1d(z)
    plt.plot(
        x_pos_adj,
        p(x_pos_adj),
        color=color,
        linestyle="--",
        linewidth=2.5,
        # label=f"{forecaster.upper()} trend",
        alpha=1,
    )

    # Plot mean points
    plt.scatter(
        x_pos_adj,
        forecaster_means,
        color=color,
        s=100,
        zorder=3,
        marker="D",
        edgecolors="black",
        linewidth=1,
    )

    # Add text annotations for mean values
    for i, (x_pos, mean_val) in enumerate(zip(x_pos_adj, forecaster_means)):
        plt.annotate(
            f"{mean_val:.4f}",
            (x_pos, mean_val),
            textcoords="offset points",
            xytext=(0, 15),  # 15 points above the point
            ha="center",
            va="bottom",
            fontsize=22,
            # fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor=color,
            ),
        )

# plt.title(
#     "RMSE Comparison: Bayesian-ARGOS versus SINDy Forecasters",
#     fontsize=14,
#     # fontweight="bold",
# )

# Add subplot label "a" to upper left corner
plt.text(
    -0.075,
    1,
    "a",
    transform=ax.transAxes,
    fontsize=46,
    fontweight="bold",
    va="bottom",
    ha="right",
)

plt.xlabel("Time Period", fontsize=30)
plt.ylabel("RMSE", fontsize=30)
# Customize legend labels with valid-case counts
ba_valid_cases = int(ba_RMSE_plot_df["Index"].nunique())
sindy_valid_cases = int(sindy_RMSE_plot_df["Index"].nunique())
custom_label_map = {
    "Bayesian-ARGOS": f"Bayesian-ARGOS ({ba_valid_cases} valid cases)",
    "SINDy": f"SINDy ({sindy_valid_cases} valid cases)",
}
handles, labels = ax.get_legend_handles_labels()
labels = [custom_label_map.get(label, label) for label in labels]
ax.legend(
    handles,
    labels,
    bbox_to_anchor=(0.5, -0.1),
    loc="upper center",
    ncol=2,
    fontsize=26.25,
    title=None,
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plot_save_path = os.path.join(
    dirpath, "results_analysis/imgs/rmse_comparison_violin_plot.pdf"
)
plt.savefig(plot_save_path, dpi=600, bbox_inches="tight")

# Print summary statistics
print("Summary Statistics by Forecaster and Time Range:")
print(
    combined_rmse_df.groupby(["Forecaster", "Time_Range"])["RMSE"].agg(
        ["mean", "std", "median", "count"]
    )
)
plt.show()

# %%
