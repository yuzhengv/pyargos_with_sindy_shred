# %%
import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import src.sindy as sindy
import src.sindy_shred as sindy_shred
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from utils.processdata import TimeSeriesDataset, load_data

# %%
# Parse command line arguments using argparse

parser = argparse.ArgumentParser(
    description="Run training with specified iteration"
)
parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
parser.add_argument(
    "--num_sensors", type=int, default=250, help="Number of sensors"
)

args = parser.parse_args()
iteration = args.iteration
num_sensors = args.num_sensors

print(f"Running iteration {iteration}")

# %%
# # Test settings
# iteration = 24
# num_sensors = 250
# %%
dirpath = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(dirpath, "exp-1")
exp_dir = os.path.join(dirpath, "exp-sindy")
exp_seneor_dir = os.path.join(model_dir, "sensor-location")
exp_saved_model_dir = os.path.join(model_dir, "saved-model")
exp_compare_dir = os.path.join(exp_dir, "comparison-plots-sindy")
recon_rmse_test_dir = os.path.join(exp_dir, "recon-rmse-test")
latent_mse_train_dir = os.path.join(exp_dir, "latent-mse-sindy-train")
latent_mse_test_dir = os.path.join(exp_dir, "latent-mse-sindy-test")
recon_metrics_via_forecaster_dir = os.path.join(exp_dir, "recon-via-forecaster")

# %%
lags = 52
load_X = load_data("SST")
load_X.shape
n = load_X.shape[0]
m = load_X.shape[1]

# Save sensor locations with iteration number in filename
sensor_file_name = f"sensor_location_{iteration}.npy"
sensor_file_path = os.path.join(exp_seneor_dir, sensor_file_name)
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

recon_rmse_test_file_name = f"recon_rmse_test_{iteration}.npy"
recon_rmse_test_file_path = os.path.join(
    recon_rmse_test_dir, recon_rmse_test_file_name
)
np.save(recon_rmse_test_file_path, test_rmse)

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
poly_order = 1
threshold = 0.05

differentiation_method = ps.differentiation.FiniteDifference()

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.8, alpha=0.05),
    differentiation_method=differentiation_method,
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)

model.fit(x, t=1 / 52.0, ensemble=False)

# %%
t_train = np.arange(0, 20, 1 / 52.0)
init_cond = np.zeros(latent_dim)
init_cond[:latent_dim] = gru_outs[0, :].detach().cpu().numpy()
x_sim = model.simulate(init_cond, t_train)

# Compare lengths and use the minimum length for plotting
gru_length = len(gru_outs)
sim_length = len(x_sim)
min_length = min(gru_length, sim_length)

fig, ax = plt.subplots(latent_dim)
for i in range(latent_dim):
    ax[i].plot(gru_outs[:min_length, i].detach().cpu().numpy())
    ax[i].plot(
        x_sim[:min_length, i],
        "--",
        color="#ae3729",
        label="SINDy",
    )
# Save the plot
plot_file_name = f"comparison_plot_sindy_{iteration}.png"
plot_file_path = os.path.join(exp_compare_dir, plot_file_name)
plt.savefig(plot_file_path)
plt.close(fig)

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

latent_mse_train_file_name = f"latent_mse_sindy_train_{iteration}.npy"
latent_mse_train_file_path = os.path.join(
    latent_mse_train_dir, latent_mse_train_file_name
)
np.save(latent_mse_train_file_path, latent_mse_train_dict)

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
t_test = np.arange(0, 317 * 1 / 52.0, 1 / 52.0)
init_cond = np.zeros(latent_dim)
init_cond[:latent_dim] = gru_outs_test_np[0, :]
x_sim_test = model.simulate(init_cond, t_test)

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

latent_mse_test_file_name = f"latent_mse_sindy_test_{iteration}.npy"
latent_mse_test_file_path = os.path.join(
    latent_mse_test_dir, latent_mse_test_file_name
)
np.save(latent_mse_test_file_path, latent_mse_test_dict)

# %%
# Plotting for each latent dimension: True vs SINDy
# fig, ax = plt.subplots(latent_dim, figsize=(10, latent_dim * 3))
# for i in range(latent_dim):
#     ax[i].plot(gru_outs_test_np[:, i], label="True Signal", linewidth=2)
#     ax[i].plot(
#         x_sim_test[:, i], "k--", label="SINDy Approximation", linewidth=2
#     )
#     ax[i].legend()
#     ax[i].set_ylabel(f"z_{i + 1}")
#     ax[i].set_xlabel("Time")

# plt.tight_layout()
# plt.show()

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
# : Reconstructed spatial temporal field from the latent forecaster
decoder_model = shred

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
# Detach and convert the reconstructed data back to numpy for visualization
output_forecaster_np = output_forecaster.detach().cpu().numpy()

# %%
# Calculate MSE between SINDy-reconstructed output and test ground truth for different time ranges
# First, inverse transform the predictions to original scale
forecaster_recons = sc.inverse_transform(output_forecaster_np)
test_ground_truth_np = test_ground_truth

# Define time ranges to evaluate
time_ranges = [(0, 50), (100, 200), (200, 275)]
results = {}

for start, end in time_ranges:
    # Ensure the ranges are within bounds
    if end > len(forecaster_recons) or end > len(test_ground_truth_np):
        continue

    range_label = f"{start}_{end}"
    current_forecaster = forecaster_recons[start:end, :]
    current_truth = test_ground_truth_np[start:end, :]

    # Calculate MSE
    mse = np.mean((current_forecaster - current_truth) ** 2)

    # Calculate normalized RMSE
    rmse = np.linalg.norm(current_forecaster - current_truth) / np.linalg.norm(
        current_truth
    )

    # Store results
    results[f"MSE_{range_label}"] = mse
    results[f"RMSE_normalized_{range_label}"] = rmse

    print(f"Time range [{start}:{end}]:")
    print(
        f"  MSE between forecaster reconstruction and ground truth: {mse:.6f}"
    )
    print(f"  Normalized RMSE: {rmse:.6f}")


# Save the results
forecaster_metrics_file_name = f"forecaster_metrics_test_{iteration}.npy"
forecaster_metrics_file_path = os.path.join(
    recon_metrics_via_forecaster_dir, forecaster_metrics_file_name
)
np.save(forecaster_metrics_file_path, results)
