# %%
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src.sindy as sindy
import src.sindy_shred as sindy_shred
from utils.processdata import TimeSeriesDataset, load_data_with_path

import pyargos.utils.argos_simulator as ags
import pyargos.utils.argos_utils as agu
from pyargos.src.argos_bayesian_argos import BayesianArgos

# %%
torch.cuda.get_device_name()

# %%
# Parse command line arguments using argparse

# parser = argparse.ArgumentParser(
#     description="Run training with specified iteration"
# )
# parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
# parser.add_argument(
#     "--num_sensors", type=int, default=250, help="Number of sensors"
# )

# args = parser.parse_args()
# iteration = args.iteration
# num_sensors = args.num_sensors

# print(f"Running iteration {iteration}")

# %%
iteration = 18
num_sensors = 250

# %%
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(dirpath, "exp-1")
exp_dir = os.path.join(dirpath, "exp-ba")
exp_sensor_dir = os.path.join(model_dir, "sensor-location")
exp_saved_model_dir = os.path.join(model_dir, "saved-model")
exp_compare_dir = os.path.join(exp_dir, "comparison-plots-ba")
recon_rmse_test_dir = os.path.join(exp_dir, "recon-rmse-test")
latent_mse_train_dir = os.path.join(exp_dir, "latent-mse-ba-train")
latent_mse_test_dir = os.path.join(exp_dir, "latent-mse-ba-test")
recon_metrics_via_forecaster_dir = os.path.join(exp_dir, "recon-via-forecaster")
output_image_dir = os.path.join(dirpath, "results_analysis/imgs-2")
# %%
lags = 52
load_X = load_data_with_path("SST", "../Data/SST_data.mat")
load_X.shape
n = load_X.shape[0]
m = load_X.shape[1]

# Save sensor locations with iteration number in filename
sensor_file_name = f"sensor_location_{iteration}.npy"
sensor_file_path = os.path.join(exp_sensor_dir, sensor_file_name)
sensor_locations = np.load(sensor_file_path)

# %%
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# %%
train_indices = np.arange(0, 1000)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
valid_indices = valid_test_indices[:30]
test_indices = valid_test_indices[30:]
# %%
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i : i + lags, sensor_locations]

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
train_data_in = torch.tensor(
    all_data_in[train_indices], dtype=torch.float32
).to(device)
valid_data_in = torch.tensor(
    all_data_in[valid_indices], dtype=torch.float32
).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(
    device
)

train_data_out = torch.tensor(
    transformed_X[train_indices + lags - 1], dtype=torch.float32
).to(device)
valid_data_out = torch.tensor(
    transformed_X[valid_indices + lags - 1], dtype=torch.float32
).to(device)
test_data_out = torch.tensor(
    transformed_X[test_indices + lags - 1], dtype=torch.float32
).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

# %%
# - Show the original trajectories with academic publication formatting
sub_index_X = transformed_X[:, sensor_locations]

# Set up the figure with proper dimensions for academic publications
fig, axes = plt.subplots(10, 1, figsize=(12.5, 36), dpi=600)
# fig.suptitle('Normalized SST Time Series at Selected Sensor Locations',
#              fontsize=16, fontweight='bold', y=0.98)

# Define a professional color palette
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i in range(10):
    axes[i].plot(
        sub_index_X[:, i],
        linewidth=1.5,
        color=colors[i],
        alpha=0.8,
        label=f"Sensor {sensor_locations[i]}",
    )

    # Add proper labels and formatting
    # axes[i].set_ylabel('Normalized SST', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3, linestyle="--")
    # axes[i].legend(loc='upper right', fontsize=10)

    # Format tick labels
    axes[i].tick_params(axis="both", which="major", labelsize=22)

    # Add subplot title
    axes[i].set_title(
        f"Sensor Location {sensor_locations[i]}", fontsize=24, pad=10
    )

# Only add x-label to the bottom subplot
axes[-1].set_xlabel("Time Index", fontsize=24, fontweight="bold")

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save the figure with publication quality
# sensor_trajectories_file = f"sensor_trajectories_{iteration}.png"
# sensor_trajectories_path = os.path.join(
#     output_image_dir, sensor_trajectories_file
# )
# plt.savefig(
#     sensor_trajectories_path,
#     dpi=600,
#     bbox_inches="tight",
#     facecolor="white",
#     edgecolor="none",
# )
plt.show()

# %%
# Plot and save each of the 10 observed sensor trajectories individually
sensors_obs_dir = os.path.join(output_image_dir, "sensors_obs")
os.makedirs(sensors_obs_dir, exist_ok=True)

for i in range(10):
    fig, ax = plt.subplots(figsize=(12.5, 3.6), dpi=600)
    ax.plot(
        sub_index_X[:, i],
        linewidth=1.8,
        color=colors[i],
        alpha=0.9,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_title(f"Sensor Location {sensor_locations[i]}", fontsize=24, pad=10)
    ax.set_xlabel("Time Index", fontsize=22, fontweight="bold")
    ax.set_ylabel("Normalized SST", fontsize=22)
    fig.tight_layout()

    sensor_obs_file = (
        f"sensor_obs_iter_{iteration}_loc_{sensor_locations[i]}.svg"
    )
    sensor_obs_path = os.path.join(sensors_obs_dir, sensor_obs_file)
    fig.savefig(
        sensor_obs_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


# %%
latent_dim = 3
poly_order = 3
include_sine = False
library_dim = sindy.library_size(latent_dim, poly_order, include_sine, True)

# %%
shred = sindy_shred.SINDy_SHRED(
    num_sensors,
    m,
    hidden_size=latent_dim,
    hidden_layers=2,
    l1=350,
    l2=400,
    dropout=0.1,
    library_dim=library_dim,
    poly_order=poly_order,
    include_sine=include_sine,
    dt=1 / 52.0 * 0.1,
    layer_norm=False,
).to(device)

# %%
model_file_name = f"model_{iteration}.pth"
model_file_path = os.path.join(exp_saved_model_dir, model_file_name)

shred.load_state_dict(torch.load(model_file_path))

# %%
# ! -------------------- Analysis based on sequence model and forecaster --------------------
# : Calculate the reconstruction rmse on the test dataset of the SINDy-SHRED network
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
test_rmse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(
    test_ground_truth
)

# recon_rmse_test_file_name = f"recon_rmse_test_{iteration}.npy"
# recon_rmse_test_file_path = os.path.join(
#     recon_rmse_test_dir, recon_rmse_test_file_name
# )
# np.save(recon_rmse_test_file_path, test_rmse)

# %%
gru_outs, sindy_outs = shred.gru_outputs(test_dataset.X, sindy=True)

# Set up the figure with proper dimensions for academic publications
fig, axes = plt.subplots(
    latent_dim, 1, figsize=(18, latent_dim * 3.35), dpi=600
)

# Handle single subplot case
if latent_dim == 1:
    axes = [axes]

# Define a professional color palette
colors = plt.cm.winter(np.linspace(0, 0.6, latent_dim))

for i in range(latent_dim):
    axes[i].plot(
        gru_outs[1:, 0, i].detach().cpu().numpy(),
        linewidth=1.5,
        color=colors[i],
        alpha=0.8,
    )

    # Add proper labels and formatting
    # axes[i].set_ylabel(
    #     f"Latent Dimension {i + 1}", fontsize=22, fontweight="bold"
    # )
    axes[i].grid(True, alpha=0.3, linestyle="--")

    # Format tick labels
    axes[i].tick_params(axis="both", which="major", labelsize=28)

    # Add subplot title
    axes[i].set_title(f"Latent Dimension {i + 1}", fontsize=30, pad=10)

# Only add x-label to the bottom subplot
axes[-1].set_xlabel("Time Index", fontsize=30, fontweight="bold")

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure with publication quality
gru_outputs_file = f"gru_outputs_{iteration}.svg"
gru_outputs_path = os.path.join(output_image_dir, gru_outputs_file)
plt.savefig(
    gru_outputs_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

# %%
# : Calculate MSE for each simulated trajectory in the latent space obtained from the identified equations
# - Here use the train_dataset
gru_outs, sindy_outs = shred.gru_outputs(train_dataset.X, sindy=True)
gru_outs = gru_outs[:, 0, :]

gru_outs[:, 0] = (gru_outs[:, 0] - torch.min(gru_outs[:, 0])) / (
    torch.max(gru_outs[:, 0]) - torch.min(gru_outs[:, 0])
)
gru_outs[:, 1] = (gru_outs[:, 1] - torch.min(gru_outs[:, 1])) / (
    torch.max(gru_outs[:, 1]) - torch.min(gru_outs[:, 1])
)
gru_outs[:, 2] = (gru_outs[:, 2] - torch.min(gru_outs[:, 2])) / (
    torch.max(gru_outs[:, 2]) - torch.min(gru_outs[:, 2])
)

gru_outs = 2 * gru_outs - 1

x = gru_outs.detach().cpu().numpy()

# %%
design_matrix = agu.build_design_matrix(
    x_t=x,
    dt=1 / 52.0,
    library_degree=1,
    sg_poly_order=4,
    library_type="poly",
)

model_ba = BayesianArgos(design_matrix=design_matrix, accelerator=True)
model_ba.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
model_ba.get_identified_model_from_all_equations()

# %%
init_cond = np.zeros(latent_dim)
init_cond[:latent_dim] = gru_outs[0, :].detach().cpu().numpy()
x_sim = model_ba.simulate(n=1040, dt=1 / 52.0, init_conditions=init_cond)

# Compare lengths and use the minimum length for plotting
gru_length = len(gru_outs)
sim_length = len(x_sim)
min_length = min(gru_length, sim_length)

# Set up the figure with proper dimensions for academic publications
fig, axes = plt.subplots(
    latent_dim,
    1,
    figsize=(12, latent_dim * 3.5),
    dpi=600,
    sharex=True,
    constrained_layout=True,
)

# Handle single subplot case
if latent_dim == 1:
    axes = [axes]

# Define a professional color palette
gru_color = "#6a51a3"  # Professional blue
ba_color = "#505ed5"  # Academic red

for i in range(latent_dim):
    axes[i].plot(
        gru_outs[:min_length, i].detach().cpu().numpy(),
        linewidth=2,
        color=gru_color,
        alpha=0.8,
        label="GRU Output",
    )
    axes[i].plot(
        x_sim[:min_length, i],
        linestyle="--",
        linewidth=2,
        color=ba_color,
        alpha=0.8,
        label="Bayesian-ARGOS Simulation",
    )

    # Add proper labels and formatting
    axes[i].set_ylabel(
        f"$z_{{{i + 1}}}$", fontsize=30, fontweight="bold"
    )  # Use LaTeX-style formatting for mathematical notation
    axes[i].grid(True, alpha=0.3, linestyle="--")
    axes[i].tick_params(axis="both", which="major", labelsize=24)
    # axes[i].legend(loc="upper right", fontsize=10)  # removed per-axes legend

# Only add x-label to the bottom subplot
axes[-1].set_xlabel("Time Index (Weeks)", fontsize=30)

# Add a panel tag "b" at the top-left of the figure
fig.text(-0.01, 1.03, "b", fontsize=44, fontweight="bold", ha="left", va="top")

# Single legend at bottom of the figure
legend_lines = [
    Line2D([0], [0], color=gru_color, lw=2, label="GRU Output"),
    Line2D(
        [0],
        [0],
        color=ba_color,
        lw=2,
        linestyle="--",
        label="Bayesian-ARGOS Simulation",
    ),
]
fig.legend(
    handles=legend_lines,
    loc="upper center",
    bbox_to_anchor=(0.54, 0),
    ncol=2,
    # frameon=False,
    fontsize=26,
)

# Adjust layout for better spacing (reserve bottom for legend)
# fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.tight_layout()

# Save the figure with publication quality
plot_file_name = f"comparison_plot_ba_{iteration}.svg"
plot_file_path = os.path.join(output_image_dir, plot_file_name)
plt.savefig(
    plot_file_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

# %%
ags.plot_3d_trajectory(
    x_sim,
    "Latent_Simulated",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

# %%
# - Calculate MSE for each simulated trajectory in the latent space obtained from the identified equations
gru_data = [
    gru_outs[:min_length, i].detach().cpu().numpy() for i in range(latent_dim)
]
sim_data = [x_sim[:min_length, i] for i in range(latent_dim)]
mse_values = [
    np.mean((gru_data[i] - sim_data[i]) ** 2) for i in range(latent_dim)
]

# Create MSE dictionary in a single step
latent_mse_train_dict = {
    f"MSE for dimension {i}": mse_values[i] for i in range(latent_dim)
}
latent_mse_train_dict["Overall_MSE"] = np.mean(mse_values)

latent_mse_train_file_name = f"latent_mse_ba_train_{iteration}.npy"
latent_mse_train_file_path = os.path.join(
    latent_mse_train_dir, latent_mse_train_file_name
)
# np.save(latent_mse_train_file_path, latent_mse_train_dict)

# %%
# ! -------------------- Include Decoder --------------------
# - Reload data from the entire training dataset and do the transformation work
gru_outs_train, _ = shred.gru_outputs(train_dataset.X, sindy=True)
gru_outs_train = gru_outs_train[:, 0, :]
gru_outs_val, _ = shred.gru_outputs(valid_dataset.X, sindy=True)
gru_outs_val = gru_outs_val[:, 0, :]
gru_outs_test, _ = shred.gru_outputs(test_dataset.X, sindy=True)
# samples reduced from 318 to 317 due to the usage of sindy=True
gru_outs_test = gru_outs_test[:, 0, :]

gru_outs_all = np.zeros((1345, 3))
gru_outs_all[:999, :] = gru_outs_train.detach().cpu().numpy()
gru_outs_all[999:1028, :] = gru_outs_val.detach().cpu().numpy()
gru_outs_all[1028:, :] = gru_outs_test.detach().cpu().numpy()

gru_outs_numpy = gru_outs_train.detach().cpu().numpy()

gru_outs_all[:, 0] = (gru_outs_all[:, 0] - np.min(gru_outs_numpy[:, 0])) / (
    np.max(gru_outs_numpy[:, 0]) - np.min(gru_outs_numpy[:, 0])
)
gru_outs_all[:, 1] = (gru_outs_all[:, 1] - np.min(gru_outs_numpy[:, 1])) / (
    np.max(gru_outs_numpy[:, 1]) - np.min(gru_outs_numpy[:, 1])
)
gru_outs_all[:, 2] = (gru_outs_all[:, 2] - np.min(gru_outs_numpy[:, 2])) / (
    np.max(gru_outs_numpy[:, 2]) - np.min(gru_outs_numpy[:, 2])
)

gru_outs_all = 2 * gru_outs_all - 1

# %%
# : Caulate the MSE for the latent space trajectories of the test dataset
###############Normalization###############
gru_outs_test_np = gru_outs_test.detach().cpu().numpy()

gru_outs_test_np[:, 0] = (
    gru_outs_test_np[:, 0] - np.min(gru_outs_numpy[:, 0])
) / (np.max(gru_outs_numpy[:, 0]) - np.min(gru_outs_numpy[:, 0]))
gru_outs_test_np[:, 1] = (
    gru_outs_test_np[:, 1] - np.min(gru_outs_numpy[:, 1])
) / (np.max(gru_outs_numpy[:, 1]) - np.min(gru_outs_numpy[:, 1]))
gru_outs_test_np[:, 2] = (
    gru_outs_test_np[:, 2] - np.min(gru_outs_numpy[:, 2])
) / (np.max(gru_outs_numpy[:, 2]) - np.min(gru_outs_numpy[:, 2]))

gru_outs_test_np = 2 * gru_outs_test_np - 1  # Transform to [-1, 1]

################Forward simulation with the model################
init_cond = np.zeros(latent_dim)
init_cond[:latent_dim] = gru_outs_test_np[0, :]
x_sim_test = model_ba.simulate(n=317, dt=1 / 52.0, init_conditions=init_cond)

latent_mse_test_values = [
    np.mean((gru_outs_test_np[:, i] - x_sim_test[:, i]) ** 2)
    for i in range(latent_dim)
]

# Create MSE dictionary in a single step
latent_mse_test_dict = {
    f"MSE for dimension {i}": latent_mse_test_values[i]
    for i in range(latent_dim)
}
latent_mse_test_dict["Overall_MSE"] = np.mean(latent_mse_test_values)

# latent_mse_test_file_name = f"latent_mse_ba_test_{iteration}.npy"
# latent_mse_test_file_path = os.path.join(
#     latent_mse_test_dir, latent_mse_test_file_name
# )
# np.save(latent_mse_test_file_path, latent_mse_test_dict)

# %%
# Set up the figure with proper dimensions for academic publications
fig, axes = plt.subplots(
    latent_dim,
    1,
    figsize=(12, latent_dim * 3.5),
    dpi=600,
    sharex=True,
    constrained_layout=True,
)

# Handle single subplot case
if latent_dim == 1:
    axes = [axes]

# Consistent color scheme with prior plots
gru_color = "#6a51a3"
ba_color = "#505ed5"

for i in range(latent_dim):
    axes[i].plot(
        gru_outs_test_np[:, i],
        linewidth=2,
        color=gru_color,
        alpha=0.8,
        label="GRU Output",
    )
    axes[i].plot(
        x_sim_test[:, i],
        linestyle="--",
        linewidth=2,
        color=ba_color,
        alpha=0.8,
        label="Bayesian-ARGOS",
    )
    # Styling
    axes[i].grid(True, alpha=0.3, linestyle="--")
    axes[i].tick_params(axis="both", which="major", labelsize=24)
    # axes[i].set_title(
    #     f"Latent Dimension {i + 1}",
    #     fontsize=11,
    #     pad=10,
    # )
    axes[i].set_ylabel(f"$z_{{{i + 1}}}$", fontsize=30, fontweight="bold")

# Only add x-label to the bottom subplot
axes[-1].set_xlabel("Time Index (Weeks)", fontsize=30)

# Add a panel tag "b" at the top-left of the figure
fig.text(-0.01, 1.03, "b", fontsize=44, fontweight="bold", ha="left", va="top")

# Single legend at bottom of the figure
legend_lines = [
    Line2D([0], [0], color=gru_color, lw=2, label="GRU Output"),
    Line2D(
        [0],
        [0],
        color=ba_color,
        lw=2,
        linestyle="--",
        label="Bayesian-ARGOS Simulation",
    ),
]
fig.legend(
    handles=legend_lines,
    loc="upper center",
    bbox_to_anchor=(0.54, 0),
    ncol=2,
    # frameon=False,
    fontsize=26,
)

# Adjust layout for better spacing
plt.tight_layout()

comparison_test_file = f"comparison_plot_ba_test_{iteration}.svg"
comparison_test_path = os.path.join(output_image_dir, comparison_test_file)
plt.savefig(
    comparison_test_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

# %%
###############Predict back in the pixel space###############

# Step 1: Revert the normalization to original scale
gru_outs_test_np = (gru_outs_test_np + 1) / 2  # Revert from [-1, 1] to [0, 1]

for i in range(3):  # Assuming 3 latent dimensions for this example
    gru_outs_test_np[:, i] = gru_outs_test_np[:, i] * (
        np.max(gru_outs_numpy[:, i]) - np.min(gru_outs_numpy[:, i])
    ) + np.min(gru_outs_numpy[:, i])

# Step 2: Decoder reconstruction using the decoder model
latent_pred = torch.FloatTensor(gru_outs_test_np).cuda()

# Pass through the decoder
output = shred.linear1(latent_pred)
output = shred.dropout(output)
output = torch.nn.functional.relu(output)
output = shred.linear2(output)
output = shred.dropout(output)
output = torch.nn.functional.relu(output)
output = shred.linear3(output)

output_np = (
    output.detach().cpu().numpy()
)  # Reconstructed spatial temporal field from the sequence model
# (317, 44219)

# %%
# : -------------------- Do the plots related to SST data --------------------
###############Plotting settings###############
load_X = loadmat("../Data/SST_data.mat")["Z"].T
mean_X = np.mean(load_X, axis=0)
sst_locs = np.where(mean_X != 0)[0]
reconstructed_data = np.zeros_like(load_X[0, :])
reconstructed_data[sst_locs] = output_np[0, :]  # Assuming first timestep
reshaped_reconstructed = reconstructed_data.reshape(180, 360)


# %%
# - Reconstructed SST map via the simulation through Bayesian-ARGOS
def reconstruct_and_plot_sindy(
    x_sim_test,
    gru_outs_numpy,
    decoder_model,
    sst_data_path,
    timesteps,
    *,
    cmap="coolwarm",
    colorbar_color=None,  # customize tick/outline color (e.g. '#333333')
    cbar_label=None,  # optional colorbar label
    fig_dpi=600,
    save_dir=None,  # optional directory to save figures
    filename_prefix="reconstructed_sst",
):
    """
    Reconstructs and visualizes SST maps from simulated latent trajectories.

    Added style controls to match earlier publication-quality figures.

    Args:
        timesteps (list[int]): Timesteps to plot.
        cmap (str | Colormap): Colormap for the image.
        colorbar_color (str | None): Color for colorbar ticks/outline.
        cbar_label (str | None): Label for colorbar.
        fig_dpi (int): Figure DPI.
        save_dir (str | None): If provided, saves each figure.
        filename_prefix (str): Prefix for saved figure filenames.
    """
    # Step 1: Reverse Min-Max scaling for SINDy-simulated data (x_sim_test)
    x_sim_test = np.array(x_sim_test)  # Ensure it's a numpy array if needed

    # Revert the scaling from [-1, 1] back to [0, 1]
    x_sim_test = (x_sim_test + 1) / 2

    # Perform the Min-Max reverse transformation using the original min/max values
    for i in range(3):  # Assuming 3 latent dimensions, need to be updated
        x_sim_test[:, i] = x_sim_test[:, i] * (
            np.max(gru_outs_numpy[:, i]) - np.min(gru_outs_numpy[:, i])
        ) + np.min(gru_outs_numpy[:, i])

    # Perform the decoder reconstruction using the transformed SINDy-simulated data
    latent_pred_forecaster = torch.FloatTensor(
        x_sim_test
    ).cuda()  # Convert to torch tensor for reconstruction

    # Pass the SINDy-simulated latent space data through the decoder
    output_forecaster = decoder_model.linear1(latent_pred_forecaster)
    output_forecaster = decoder_model.dropout(output_forecaster)
    output_forecaster = torch.nn.functional.relu(output_forecaster)
    output_forecaster = decoder_model.linear2(output_forecaster)
    output_forecaster = decoder_model.dropout(output_forecaster)
    output_forecaster = torch.nn.functional.relu(output_forecaster)
    output_forecaster = decoder_model.linear3(output_forecaster)

    # Detach and convert reconstructed data
    output_forecaster_np = output_forecaster.detach().cpu().numpy()
    load_X = loadmat(sst_data_path)["Z"].T
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Pre-compute global vmin/vmax for consistent color scaling (optional)
    vmin = np.min(output_forecaster_np[timesteps, :])
    vmax = np.max(output_forecaster_np[timesteps, :])

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    for t in timesteps:
        reconstructed_data_forecaster = np.full_like(
            load_X[0, :], np.nan, dtype=float
        )
        reconstructed_data_forecaster[sst_locs] = output_forecaster_np[t, :]
        reshaped_reconstructed_forecaster = (
            reconstructed_data_forecaster.reshape(180, 360)
        )

        # Publication-style figure
        fig, ax = plt.subplots(figsize=(12, 6), dpi=fig_dpi)
        im = ax.imshow(
            reshaped_reconstructed_forecaster,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        if cbar_label:
            cbar.set_label(cbar_label, fontsize=20)
            cbar.ax.tick_params(labelsize=20)
        if colorbar_color:
            cbar.outline.set_edgecolor(colorbar_color)
            cbar.ax.yaxis.set_tick_params(color=colorbar_color)
            for tick in cbar.ax.get_yticklabels():
                tick.set_color(colorbar_color)

        # Clean axis (map-style) but consistent typography
        ax.set_title(f"SST Reconstruction (t = {t})", fontsize=24, pad=8)
        ax.set_axis_off()

        fig.tight_layout()

        if save_dir is not None:
            fig_path = os.path.join(save_dir, f"{filename_prefix}_t{t}.svg")
            fig.savefig(
                fig_path,
                dpi=fig_dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        plt.show()

    return output_forecaster_np


# Set up output directory
plot_dir_name = f"decoded_full_field_{iteration}"
plot_dir_path = os.path.join(output_image_dir, plot_dir_name)

# Example usage (custom styling) – leave original call compatible:
timesteps = [50, 100, 150]
output_forecaster_np = reconstruct_and_plot_sindy(
    x_sim_test,
    gru_outs_numpy,
    shred,
    "../Data/SST_data.mat",
    timesteps,
    cmap="coolwarm",  # main image colormap
    # colorbar_color="#222222",  # ticks/outline color
    cbar_label="Normalized SST",
    fig_dpi=600,
    save_dir=plot_dir_path,
    filename_prefix=f"reconstructed_sst_iter{iteration}",
)


# %%
def plot_zoomed_comparison(
    real_data,
    reconstructed_data_via_forecaster,
    sst_locs,
    timesteps,
    lat_range,
    lon_range,
    diff_scale=1.0,
    *,
    cmap="coolwarm",
    diff_cmap="RdBu_r",  # diverging colormap
    figsize=None,
    fig_dpi=600,
    add_colorbar=True,
    annotate_stats=True,
    font_scale=1.0,
    background_value=np.nan,
    suptitle=None,
    robust=True,
    robust_pct=(1, 99),
    save=False,  # NEW: whether to save the figure
    save_dir=None,  # NEW: directory to save into
    filename="zoomed_comparison.pdf",  # NEW: filename
):
    """
    Publication-quality comparison of observed vs. model SST fields (zoomed) and their differences.
    Keeps original latitude orientation, removes geo ticks/labels.

    Args:
        real_data: The true SST data.
        reconstructed_data_via_forecaster: The reconstructed SST data from the forecaster.
        sst_locs: The locations of the SST data.
        timesteps: The timesteps to plot.
        lat_range: The latitude range to zoom in on.
        lon_range: The longitude range to zoom in on.
        diff_scale: Scaling factor for the difference plots.
        cmap: Colormap for the SST plots.
        diff_cmap: Colormap for the difference plots.
        figsize: Figure size.
        fig_dpi: Figure DPI.
        add_colorbar: Whether to add a colorbar to the plots.
        annotate_stats: Whether to annotate the plots with statistics.
        font_scale: Font size scale factor.
        background_value: Value for the background of the plots.
        suptitle: Super title for the figure.
        robust: Whether to use robust scaling for the color limits.
        robust_pct: Percentiles for robust scaling.
        save (bool): If True, save the generated figure.
        save_dir (str | None): Directory to save figure when save=True.
        filename (str): Output filename when save=True.
    Returns:
        dict: Global stats; includes 'saved_path' key if figure was saved.
    """
    # Inverse transform
    real_data_inv = sc.inverse_transform(real_data)
    recon_data_inv = sc.inverse_transform(reconstructed_data_via_forecaster)

    # Helper to rebuild the 2D field
    def build_field(vector):
        field = np.full(180 * 360, background_value, dtype=float)
        field[sst_locs] = vector
        field = field.reshape(180, 360)
        return field[lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]]

    real_stack = np.stack(
        [build_field(real_data_inv[t, :]) for t in timesteps], axis=0
    )
    model_stack = np.stack(
        [build_field(recon_data_inv[t, :]) for t in timesteps], axis=0
    )
    diff_stack = real_stack - model_stack

    # Robust color limits for main data
    if robust:
        all_data = np.concatenate([real_stack.ravel(), model_stack.ravel()])
        data_min = np.nanpercentile(all_data, robust_pct[0])
        data_max = np.nanpercentile(all_data, robust_pct[1])
    else:
        data_min = np.nanmin([real_stack, model_stack])
        data_max = np.nanmax([real_stack, model_stack])

    # Symmetric diff limits
    if robust:
        m = np.nanpercentile(np.abs(diff_stack), robust_pct[1])
        diff_lim = float(m) * diff_scale
    else:
        diff_lim = np.nanmax(np.abs(diff_stack)) * diff_scale
    diff_norm = TwoSlopeNorm(vmin=-diff_lim, vcenter=0.0, vmax=diff_lim)

    # Figure layout
    n_cols = len(timesteps)
    if figsize is None:
        figsize = (max(6.5, 2.4 * n_cols), 8)

    fig, axes = plt.subplots(
        3, n_cols, figsize=figsize, dpi=fig_dpi, constrained_layout=True
    )
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Observations", "Predictions", "Differences"]

    for j, t in enumerate(timesteps):
        im0 = axes[0, j].imshow(
            real_stack[j],
            cmap=cmap,
            norm=Normalize(vmin=data_min, vmax=data_max),
            aspect="equal",
            interpolation="nearest",
        )
        im1 = axes[1, j].imshow(
            model_stack[j],
            cmap=cmap,
            norm=Normalize(vmin=data_min, vmax=data_max),
            aspect="equal",
            interpolation="nearest",
        )
        im2 = axes[2, j].imshow(
            diff_stack[j],
            cmap=diff_cmap,
            norm=diff_norm,
            aspect="equal",
            interpolation="nearest",
        )

        # Titles & labels
        axes[0, j].set_title(f"t = {t}", fontsize=18 * font_scale, pad=6)
        for r in range(3):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(row_titles[r], fontsize=18 * font_scale)

        if annotate_stats:
            rmse = float(np.nanmean((diff_stack[j]) ** 2)) ** 0.5
            axes[2, j].text(
                0.02,
                0.02,
                f"RMSE={rmse:.3f}",
                fontsize=16 * font_scale,
                color="black",
                ha="left",
                va="bottom",
                transform=axes[2, j].transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    linewidth=0.5,
                ),
            )

    if add_colorbar:
        # Let tight_layout finish so we can read final axes positions
        # plt.tight_layout(w_pad=0.9, h_pad=0.9)
        fig.subplots_adjust(right=0.88)
        fig.canvas.draw()  # ensures positions are finalized

        # union of all axes in rows 0–1 (SST) and row 2 (diffs)
        row01_boxes = [ax.get_position(fig) for ax in axes[0, :]] + [
            ax.get_position(fig) for ax in axes[1, :]
        ]
        row2_boxes = [ax.get_position(fig) for ax in axes[2, :]]

        box_row01 = Bbox.union(row01_boxes)
        box_row2 = Bbox.union(row2_boxes)

        # spacing to the right of the panels (figure fraction)
        pad = 0.02
        width = 0.01

        # create colorbar axes that match the row heights
        shrink = 0.95
        shrink_2 = 0.9
        cax0 = fig.add_axes(
            [
                box_row01.x1 + pad,
                box_row01.y0 + (1 - shrink) * box_row01.height / 2,
                width,
                shrink * box_row01.height,
            ]
        )
        cax1 = fig.add_axes(
            [
                box_row2.x1 + pad,
                box_row2.y0 + (1 - shrink_2) * box_row2.height / 2,
                width,
                shrink_2 * box_row2.height,
            ]
        )

        # draw colorbars
        cb0 = fig.colorbar(im0, cax=cax0)
        cb1 = fig.colorbar(im2, cax=cax1)

        # styling
        cb0.ax.tick_params(labelsize=16 * font_scale)
        cb1.ax.tick_params(labelsize=16 * font_scale)
        cb0.set_label("SST (°C)", fontsize=18 * font_scale)
        cb1.set_label("Difference (°C)", fontsize=18 * font_scale)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10 * font_scale, y=0.995)

    fig.text(
        -0.01, 1.03, "c", fontsize=28, fontweight="bold", ha="left", va="top"
    )
    # plt.tight_layout(w_pad=0.6, h_pad=0.6)

    saved_path = None
    if save:
        if save_dir is None:
            raise ValueError("save=True but save_dir is None.")
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, filename)
        fig.savefig(saved_path, dpi=fig_dpi, bbox_inches="tight")

    plt.show()

    # Global stats
    global_rmse = float(np.nanmean(diff_stack**2) ** 0.5)
    global_mae = float(np.nanmean(np.abs(diff_stack)))
    result = {"RMSE": global_rmse, "MAE": global_mae}
    # if saved_path:
    #     result["saved_path"] = saved_path
    return result


timesteps = [50, 100, 150, 200, 250]  # Define the timesteps you want to compare
lat_range = (
    0,
    180,
)  # Define the latitude range to zoom in (adjust based on your grid)
lon_range = (
    0,
    180,
)  # Define the longitude range to zoom in (adjust based on your grid)
plot_zoomed_comparison(
    test_dataset.Y.detach().cpu().numpy(),
    output_forecaster_np,
    sst_locs,
    timesteps,
    lat_range,
    lon_range,
    diff_scale=1,
    save=True,
    filename=f"zoomed_comparison_{iteration}.pdf",
    save_dir=output_image_dir,
)


# %%
def plot_multiple_sensor_predictions(
    real_data,
    forecaster_data,
    sensor_locations,
    sensor_indices,
    num_train=52,
    num_pred=250,
    rows=3,
    cols=6,
    *,
    # Restored original colors
    train_color="#6a51a3",
    pred_color="#505ed5",
    train_label="Observation",
    pred_label="Forecast",
    figsize=None,
    dpi=600,
    lw=1.6,
    font_scale=1,
    robust=True,
    robust_pct=(2, 98),
    share_ylim=True,
    panel_labels=False,
    panel_label_style=dict(x=0.02, y=0.92, fontsize=9, weight="bold"),
    vertical_line_style=dict(color="gray", linestyle=":", linewidth=1.0),
    grid=True,
    grid_alpha=0.15,
    grid_linestyle="--",
    show_rmse=True,
    rmse_box_style=dict(
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", lw=0.6),
        fontsize=16,
        x=0.98,
        y=0.06,
        ha="right",
        va="bottom",
    ),
    xlabel="Time (weeks)",
    ylabel="Value",
    sci_yaxis=True,
    sci_limits=(-2, 3),
    tick_count_x=5,
    tick_count_y=5,
    tight=True,
    metadata_title="Observed vs Forecasted Trajectories",
    save_path=None,
):
    import os
    import string

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

    # Style tweaks
    rc = mpl.rcParams
    rc["font.size"] = 16 * font_scale
    rc["axes.titlesize"] = 18 * font_scale
    rc["axes.labelsize"] = 16 * font_scale
    rc["xtick.labelsize"] = 16 * font_scale
    rc["ytick.labelsize"] = 16 * font_scale
    rc["axes.linewidth"] = 1
    rc["figure.facecolor"] = "white"
    rc["axes.facecolor"] = "white"
    rc["legend.frameon"] = True

    total_needed = len(sensor_indices)
    max_panels = rows * cols
    if total_needed > max_panels:
        sensor_indices = sensor_indices[:max_panels]

    if figsize is None:
        figsize = (cols * 4, rows * 4)

    T_window = num_train + num_pred
    rmse_per_sensor = {}
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, dpi=dpi, sharex=True, sharey=share_ylim
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)
    y_collect = []

    y_formatter = ScalarFormatter(useMathText=True)
    if sci_yaxis:
        y_formatter.set_powerlimits(sci_limits)

    for plot_idx, sensor_idx in enumerate(sensor_indices):
        r, c = divmod(plot_idx, cols)
        ax = axes[r, c]
        sensor_col = int(sensor_locations[sensor_idx])
        obs_series = np.asarray(real_data[:T_window, sensor_col]).astype(float)
        model_series = np.asarray(
            forecaster_data[:T_window, sensor_col]
        ).astype(float)

        obs_hist = obs_series[:num_train]
        obs_fore = obs_series[num_train:]
        model_fore = model_series[num_train : num_train + num_pred]

        x_hist = np.arange(num_train)
        x_fore = np.arange(num_train, num_train + len(obs_fore))

        ax.plot(x_hist, obs_hist, color=train_color, lw=lw, alpha=0.9)
        ax.plot(x_fore, obs_fore, color=train_color, lw=lw, alpha=0.35)
        ax.plot(
            x_fore[: len(model_fore)],
            model_fore,
            color=pred_color,
            lw=lw,
            linestyle="--",
            alpha=0.95,
        )

        ax.axvline(x=num_train, **vertical_line_style)
        ax.set_title(f"Sensor {sensor_idx}", pad=3.5)

        if panel_labels:
            label = f"({string.ascii_lowercase[plot_idx]})"
            ax.text(
                panel_label_style.get("x", 0.02),
                panel_label_style.get("y", 0.92),
                label,
                transform=ax.transAxes,
                fontsize=panel_label_style.get("fontsize", 9) * font_scale,
                weight=panel_label_style.get("weight", "bold"),
                ha="left",
                va="center",
            )

        if grid:
            ax.grid(
                True, which="major", linestyle=grid_linestyle, alpha=grid_alpha
            )
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=tick_count_x, integer=True)
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_count_y))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both", length=3.5, width=0.7, pad=2.5)
        ax.tick_params(which="minor", length=2.0, width=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if sci_yaxis:
            ax.yaxis.set_major_formatter(y_formatter)

        if share_ylim:
            if len(obs_hist):
                y_collect.append(obs_hist[np.isfinite(obs_hist)])
            if len(obs_fore):
                y_collect.append(obs_fore[np.isfinite(obs_fore)])
            if len(model_fore):
                y_collect.append(model_fore[np.isfinite(model_fore)])

        min_len = min(len(model_fore), len(obs_fore))
        if min_len > 0:
            valid = np.isfinite(model_fore[:min_len]) & np.isfinite(
                obs_fore[:min_len]
            )
            if valid.any():
                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                model_fore[:min_len][valid]
                                - obs_fore[:min_len][valid]
                            )
                            ** 2
                        )
                    )
                )
                rmse_per_sensor[sensor_idx] = rmse
                if show_rmse:
                    ax.text(
                        rmse_box_style["x"],
                        rmse_box_style["y"],
                        f"RMSE: {rmse:.3g}",
                        transform=ax.transAxes,
                        fontsize=rmse_box_style["fontsize"] * font_scale,
                        ha=rmse_box_style["ha"],
                        va=rmse_box_style["va"],
                        bbox=rmse_box_style["bbox"],
                    )
            else:
                rmse_per_sensor[sensor_idx] = float("nan")

    for leftover in range(len(sensor_indices), max_panels):
        r, c = divmod(leftover, cols)
        fig.delaxes(axes[r, c])

    if share_ylim and len(y_collect):
        all_vals = np.concatenate([arr for arr in y_collect if arr.size > 0])
        if all_vals.size > 0:
            if robust:
                vmin = np.nanpercentile(all_vals, robust_pct[0])
                vmax = np.nanpercentile(all_vals, robust_pct[1])
            else:
                vmin = np.nanmin(all_vals)
                vmax = np.nanmax(all_vals)
            if np.isfinite(vmin) and np.isfinite(vmax):
                pad = 0.04 * (
                    vmax - vmin if vmax > vmin else max(abs(vmax), 1.0)
                )
                for ax in fig.axes:
                    if ax.has_data():
                        ax.set_ylim(vmin - pad, vmax + pad)

    for r in range(rows):
        for c in range(cols):
            try:
                ax = axes[r, c]
            except Exception:
                continue
            if ax not in fig.axes or not ax.has_data():
                continue
            if r == rows - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)

    # Legend at bottom
    legend_lines = [
        Line2D([0], [0], color=train_color, lw=2, label=train_label),
        Line2D(
            [0], [0], color=pred_color, lw=2, linestyle="--", label=pred_label
        ),
    ]
    fig.legend(
        handles=legend_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=20 * font_scale,
        handlelength=2.2,
        columnspacing=1.8,
    )

    if tight:
        fig.tight_layout(rect=[0, 0.03, 1, 1])  # extra bottom space for legend

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        fig.savefig(
            save_path,
            dpi=dpi if ext in {".png", ".jpg", ".tif"} else None,
            bbox_inches="tight",
            metadata=dict(Title=metadata_title),
        )

    valid_vals = [v for v in rmse_per_sensor.values() if np.isfinite(v)]
    stats = {
        "per_sensor_rmse": rmse_per_sensor,
        "mean_rmse": float(np.mean(valid_vals)) if valid_vals else np.nan,
    }
    return fig, axes, stats


sensor_locations_test = np.random.randint(1, 40000, size=18)
sensor_indices = list(range(0, 18, 1))

fig, axes, stats = plot_multiple_sensor_predictions(
    test_dataset.Y.detach().cpu().numpy(),
    output_forecaster_np,
    sensor_locations_test,
    sensor_indices,
    rows=3,
    cols=6,
    save_path=f"imgs/multi_sensor_forecast_{iteration}.pdf",
)

# %%
