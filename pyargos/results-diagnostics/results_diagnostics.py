# %%
import os
import sys

import arviz as az
import dill  # Better serialization than pickle
import matplotlib.pyplot as plt
import numpy as np

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import utils.argos_simulator as ags
import utils.argos_utils as agu
from src.argos_bayesian_argos import BayesianArgos

# %%
# az.style.use(["arviz-white", "arviz-bluish"])

# ! -------------------- Rossler with inf snr--------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-10, 10),
    x2_range=(-10, 10),
    x3_range=(0, 20),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-1, -1], [1, 0.2], [0.2, 1, -5.7]],
    variable_names=[["x2", "x3"], ["x1", "x2"], ["", "x1x3", "x3"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=np.inf,
)

# %%
fig = ags.plot_trajectory_3d(x_t, "Rossler")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)
design_matrix["sorted_feature_names"].shape
# %%
model_rossler = BayesianArgos(design_matrix=design_matrix, accelerator=True)
model_rossler.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
)

model_rossler.get_identified_model_from_all_equations()

# %%
model = model_rossler.results["equation_1"]["model"]
results = model_rossler.results["equation_1"]["results"]
model.predict(results, kind="response")

# az.plot_ppc(
#     results,
#     num_pp_samples=200,
#     figsize=(10, 10),
# )

y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean

# %%
fig_residuals, ax_residuals = plt.subplots(figsize=(12, 8), dpi=300)
ax_residuals.scatter(pp_mean, residuals, s=1, color="#116DA9", alpha=0.7)
# Add regression line
z = np.polyfit(pp_mean, residuals, 3)
p = np.poly1d(z)
ax_residuals.plot(
    pp_mean,
    p(pp_mean),
    color="#634564",
    linestyle="-",
    alpha=0.8,
    linewidth=1,
    # label=f"Regression line (slope={z[0]:.4f})",
)
ax_residuals.axhline(0, color="#9F0000", linestyle="--", alpha=0.6)
ax_residuals.set_xlabel("Posterior Predictive Mean")
ax_residuals.set_ylabel("Residuals")
# ax_residuals.legend()
fig_residuals.tight_layout()


# ! -------------------- Rossler with 49 snr--------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-10, 10),
    x2_range=(-10, 10),
    x3_range=(0, 20),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-1, -1], [1, 0.2], [0.2, 1, -5.7]],
    variable_names=[["x2", "x3"], ["x1", "x2"], ["", "x1x3", "x3"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

# %%
fig = ags.plot_trajectory_3d(x_t, "Rossler")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_rossler = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_rossler.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
)

model_rossler.get_identified_model_from_all_equations()

# %%
model = model_rossler.results["equation_1"]["model"]
results = model_rossler.results["equation_1"]["results"]
model.predict(results, kind="response")

az.plot_ppc(
    results,
    num_pp_samples=200,
    figsize=(12, 10),
)

y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean
plt.figure(figsize=(12, 8), dpi=300)
plt.scatter(pp_mean, residuals, s=1, alpha=0.5)
plt.axhline(0, color="#9F0000", linestyle="--", alpha=0.5)
plt.xlabel("Posterior Predictive Mean")
plt.ylabel("Residuals")
plt.tight_layout()


# ! -------------------- Dadras --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-4, 4),
    x2_range=(-4, 4),
    x3_range=(-4, 4),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, -3, 2.7], [1.7, -1, 1], [2, -9]],
    variable_names=[["x2", "x1", "x2x3"], ["x2", "x1x3", "x3"], ["x1x2", "x3"]],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)


# fig = ags.plot_trajectory_3d(x_t, "Dadras")
# plt.close(fig)

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_dadras = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_dadras.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# Save model_dadras object to file
# Save model_compare object to file using dill
with open(f"../data/dadras_model_n_10_50.dill", "wb") as f:
    dill.dump(model_dadras, f)

# Example of how to load it back
# with open("dadras_model_snr.dill", "rb") as f:
#     model_compare = dill.load(f)


# %%
model_dadras.get_identified_model_from_all_equations()

# %%
model = model_dadras.results["equation_1"]["model"]
results = model_dadras.results["equation_1"]["results"]
model.predict(results, kind="response")

# %%
loo = az.loo(
    results, pointwise=True
)  # pointwise=True gives per-observation stuff
khat = loo.pareto_k
# az.plot_khat(khat)  # interactive Bokeh / Matplotlib backend
# %%
# Plot khat values with highlighting for values >= 0.7
plt.figure(figsize=(12, 8), dpi=300)
n_points = len(khat)
x = range(n_points)
# Plot points < 0.7 in blue
plt.scatter(
    [i for i in x if khat[i] < 0.7],
    [khat[i] for i in x if khat[i] < 0.7],
    color="#116DA9",
    label="k < 0.7",
    s=20,  # smaller point size
    marker="o",  # circle marker
    alpha=0.7,  # slight transparency
)
# Plot points >= 0.7 in red
plt.scatter(
    [i for i in x if khat[i] >= 0.7],
    [khat[i] for i in x if khat[i] >= 0.7],
    color="#B03C2B",
    label="k ≥ 0.7",
    s=30,  # slightly larger than the blue points
    marker="^",  # triangle marker
    alpha=0.9,  # more visible
)
plt.axhline(y=0.7, color="#9F0000", linestyle="--", alpha=0.5)
plt.xlabel("Data Point")
plt.ylabel("Pareto k")
plt.title("Pareto k Diagnostic Values")
plt.legend()
plt.tight_layout()

# Original ArviZ plot for reference
# az.plot_khat(khat)

# ! -------------------- Aizawa --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    x3_range=(-1, 2),
    num_columns=3,
)

# Test 10 ** 3.7 and 10 ** 5
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [-3.5, -0.7, 1],
        [3.5, -0.7, 1],
        [0.95, 0.65, 0.1, -1 / 3, -0.25, -1, -0.25, -1],
    ],
    variable_names=[
        ["x2", "x1", "x1x3"],
        ["x1", "x2", "x2x3"],
        ["x3", "", "x1^3x3", "x3^3", "x1^2x3", "x1^2", "x2^2x3", "x2^2"],
    ],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=np.inf,
)

# fig = ags.plot_3d_trajectory(x_t, "Aizawa", save_figure=False)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_aizawa = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_aizawa.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# Save model_aizawa object to file
with open("../data/snr-analysis/aizawa_model_snr_62.dill", "wb") as f:
    dill.dump(model_aizawa, f)

# %%
# ! -------------------- ChenLee --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(5, 20),
    x2_range=(-20, 20),
    x3_range=(-20, 0),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[5, -1], [-10, 1], [-3.8, 1 / 3]],
    variable_names=[["x1", "x2x3"], ["x2", "x1x3"], ["x3", "x1x2"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=61,
)

# fig = ags.plot_3d_trajectory(x_t, "ChenLee", save_figure=False)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_chenlee = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_chenlee.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# Save model_chenlee object to file
with open("../data/snr-analysis/chenlee_model_snr_61.dill", "wb") as f:
    dill.dump(model_chenlee, f)


# %%
