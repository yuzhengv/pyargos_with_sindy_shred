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
from pysindy.differentiation import FiniteDifference
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
# Test settings
# iteration = 24
# num_sensors = 250
# %%
dirpath = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.join(dirpath, "exp-1")
exp_seneor_dir = os.path.join(exp_dir, "sensor-location")
exp_saved_model_dir = os.path.join(exp_dir, "saved-model")
exp_compare_dir = os.path.join(exp_dir, "comparison-plots")
test_rmse_dir = os.path.join(exp_dir, "test-rmse")
mse_train_dir = os.path.join(exp_dir, "mse-pysindy-train")
# %%
lags = 52
load_X = load_data("SST")
load_X.shape
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Save sensor locations with iteration number in filename
sensor_file_name = f"sensor_location_{iteration}.npy"
sensor_file_path = os.path.join(exp_seneor_dir, sensor_file_name)
np.save(sensor_file_path, sensor_locations)
# sensor_locations = np.load(sensor_file_path)

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

validation_errors = sindy_shred.fit(
    shred,
    train_dataset,
    valid_dataset,
    batch_size=128,
    num_epochs=600,
    lr=1e-3,
    verbose=True,
    threshold=0.25,
    patience=5,
    sindy_regularization=10.0,
    optimizer="AdamW",
    thres_epoch=100,
)

# %%
model_file_name = f"model_{iteration}.pth"
model_file_path = os.path.join(exp_saved_model_dir, model_file_name)

torch.save(shred.state_dict(), model_file_path)
# shred.load_state_dict(torch.load(model_file_path))

# %%
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
test_rmse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(
    test_ground_truth
)

test_rmse_file_name = f"test_rmse_{iteration}.npy"
test_rmse_file_path = os.path.join(test_rmse_dir, test_rmse_file_name)
np.save(test_rmse_file_path, test_rmse)

# %%
gru_outs, sindy_outs = shred.gru_outputs(test_dataset.X, sindy=True)

# %%
# - Here use the train_dataset
gru_outs, sindy_outs = shred.gru_outputs(train_dataset.X, sindy=True)
gru_outs = gru_outs[:, 0, :]

# %%
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
    ax[i].plot(x_sim[:min_length, i], "m--", label="model")

# Save the plot
plot_file_name = f"comparison_plot_{iteration}.png"
plot_file_path = os.path.join(exp_compare_dir, plot_file_name)
plt.savefig(plot_file_path)
plt.close(fig)

# Calculate MSE for each dimension using list comprehension
gru_data = [
    gru_outs[:min_length, i].detach().cpu().numpy() for i in range(latent_dim)
]
sim_data = [x_sim[:min_length, i] for i in range(latent_dim)]
mse_values = [
    np.mean((gru_data[i] - sim_data[i]) ** 2) for i in range(latent_dim)
]

# Create MSE dictionary in a single step
mse_train_dict = {
    f"MSE for dimension {i}": mse_values[i] for i in range(latent_dim)
}
mse_train_dict["Overall_MSE"] = np.mean(mse_values)

mse_train_file_name = f"mse_pysidny_train_{iteration}.npy"
mse_pysindy_file_path = os.path.join(mse_train_dir, mse_train_file_name)
np.save(mse_pysindy_file_path, mse_train_dict)
# np.load(mse_pysindy_file_path, allow_pickle=True).item()
