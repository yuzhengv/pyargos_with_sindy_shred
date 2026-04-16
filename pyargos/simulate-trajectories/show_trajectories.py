# %%
import os
import sys

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import importlib

import utils.argos_simulator as ags

# %%
importlib.reload(ags)

# %%
# ! -------------------- Lorenz --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-15, 15),
    x2_range=(-15, 15),
    x3_range=(10, 40),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-10, 10], [28, -1, -1], [1, -8 / 3]],
    variable_names=[["x1", "x2"], ["x1", "x2", "x1x3"], ["x1x2", "x3"]],
    n=10**4.5,
    dt=0.001,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Lorenz", show_colorbar=False, output_format="pdf")
# ! -------------------------------------------------

# %%
# ! -------------------- Chen Lee --------------------
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
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "ChenLee", show_colorbar=False)

ags.plot_trajectory_colorbar_only(
    n_steps=10**5, system_name="ChenLee", color_bar_ticks=25000
)

# ! -------------------------------------------------

# %%
# ! -------------------- Thomas --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(-1, 1),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, -0.208186], [1, -0.208186], [1, -0.208186]],
    variable_names=[["sin(x2)", "x1"], ["sin(x3)", "x2"], ["sin(x1)", "x3"]],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Thomas", show_colorbar=False)
# ! -------------------------------------------------

# %%
# ! -------------------- Rossler --------------------
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
    n=10**4.5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Rossler", show_colorbar=False)
# ! -------------------------------------------------

# %%
# ! -------------------- Halvorsen --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-4, 4),
    x2_range=(-4, 4),
    x3_range=(-4, 4),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [-1.89, -4, -4, -1],
        [-1.89, -4, -4, -1],
        [-1.89, -4, -4, -1],
    ],
    variable_names=[
        ["x1", "x2", "x3", "x2^2"],
        ["x2", "x3", "x1", "x3^2"],
        ["x3", "x1", "x2", "x1^2"],
    ],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

ags.plot_3d_trajectory(
    x_t,
    "Halvorsen",
    show_colorbar=True,
    color_bar_ticks=25000,
)
# ! -------------------------------------------------

# %%
# ! -------------------- Dadras --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-4, 4),
    x2_range=(-4, 4),
    x3_range=(-4, 4),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [1, -3, 2.7],
        [1.7, -1, 1],
        [2, -9],
    ],
    variable_names=[
        ["x2", "x1", "x2x3"],
        ["x2", "x1x3", "x3"],
        ["x1x2", "x3"],
    ],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[30].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Dadras", show_colorbar=True, color_bar_ticks=25000)
# ! -------------------------------------------------

# %%
# ! -------------------- Sprott --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(-1, 1),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, 2.07, 1], [1, -1.79, 1], [1, -1, -1]],
    variable_names=[
        ["x2", "x1x2", "x1x3"],
        ["", "x1^2", "x2x3"],
        ["x1", "x1^2", "x2^2"],
    ],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Sprott", show_colorbar=False)
# ! ---------------------------------------------------

# ----------------- Put in the appendix ---------------
# %%
# ! -------------------- Linear_3d --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(10**-1, 10**3),
    x2_range=(10**-1, 10**3),
    x3_range=(10**-1, 10**3),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-0.1, 2], [-2, -0.1], [-0.3]],
    variable_names=[["x1", "x2"], ["x1", "x2"], ["x3"]],
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[30].values,
    snr=49,
)

ags.plot_3d_trajectory(x_t, "Linear_3d", show_colorbar=False)
# ! -------------------------------------------------

# %%
# ! -------------------- Aizawa --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    x3_range=(-1, 2),
    num_columns=3,
)

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
    n=10**5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[75].values,
    snr=49,
)

ags.plot_3d_trajectory(
    x_t,
    "Aizawa",
    show_colorbar=True,
    color_bar_ticks=25000,
)
# ! -------------------------------------------------

# %%
