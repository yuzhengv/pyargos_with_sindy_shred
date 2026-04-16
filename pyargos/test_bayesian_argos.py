# %%
import os
import sys

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import utils.argos_simulator as ags
import utils.argos_utils as agu
from src.argos_bayesian_argos import BayesianArgos

# ! -------------------- Lorenz --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-15, 15),
    x2_range=(-15, 15),
    x3_range=(10, 40),
    num_columns=3,
)

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-10, 10], [28, -1, -1], [1, -8 / 3]],
    variable_names=[["x1", "x2"], ["x1", "x2", "x1x3"], ["x1x2", "x3"]],
    n=10**3.5,
    dt=0.001,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)


# %%
fig = ags.plot_3d_trajectory(x_t, "Lorenz", save_figure=False)

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.001,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(
    design_matrix=design_matrix, custom_prior=True, accelerator=True
)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
print(model_compare.results["equation_3"]["model"])
model_compare.results["equation_3"]["model"].plot_priors()
# %%
model_compare.get_frequentist_results()
# %%
model_compare.get_identified_model_from_all_equations()
variable_coeff_identified, variable_names_identified = (
    model_compare.expressions_for_simulation()
)

# ! -------------------- Rossler --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-10, 10),
    x2_range=(-10, 10),
    x3_range=(0, 20),
    num_columns=3,
)

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-1, -1], [1, 0.2], [0.2, 1, -5.7]],
    variable_names=[["x2", "x3"], ["x1", "x2"], ["", "x1x3", "x3"]],
    n=10**3.5,
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
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="comparison",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
)


# %%
model_compare.get_frequentist_results()
model_compare.get_identified_model_from_all_equations()
x_sim = model_compare.simulate(
    n=10**3.5, dt=0.01, init_conditions=initial_value_df.iloc[95].values
)
x_sim.shape
fig = ags.plot_trajectory_3d(x_sim, "Rossler_identified")

# ! -------------------- Thomas --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(-1, 1),
    num_columns=3,
)

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, -0.208186], [1, -0.208186], [1, -0.208186]],
    variable_names=[["sin(x2)", "x1"], ["sin(x3)", "x2"], ["sin(x1)", "x3"]],
    n=10**3.5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)


# %%
fig = ags.plot_trajectory_3d(x_t, "Thomas")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly_four",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="comparison",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
)

# %%
model_compare.get_frequentist_results()
model_compare.get_identified_model_from_all_equations()
model_compare.expressions_for_simulation()
x_sim = model_compare.simulate(
    n=10**3.5, dt=0.01, init_conditions=initial_value_df.iloc[95].values
)
x_sim.shape
fig = ags.plot_trajectory_3d(x_sim, "Thomas_identified")

# ! -------------------- Maxwell Bloch --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(0, 10),
    x2_range=(0, 10),
    x3_range=(0, 10),
    num_columns=3,
)

# x1_range=(0, 5),
# x2_range=(-2, 5),
# x3_range=(-1, 3), under 60%

# x <- runif(1, min = 0, max = 5)
# y <- runif(1, min = 0, max = 5)
# z <- runif(1, min = 0, max = 3) Not good
# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-0.1, 0.1], [0.21, -0.21], [-1.054, -0.34, 1.394]],
    variable_names=[["x1", "x2"], ["x1x3", "x2"], ["x1x2", "x3", ""]],
    n=10**3,
    dt=0.01,
    init_conditions=initial_value_df.iloc[33].values,
    snr=49,
)


# %%
fig = ags.plot_trajectory_3d(x_t, "Maxwell Bloch")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
# %%
model_compare.get_identified_model_from_all_equations()

# %%
model_compare.get_identified_model_from_all_equations()
variable_coeff_identified, variable_names_identified = (
    model_compare.expressions_for_simulation()
)

# ! -------------------- Linear 3d --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(10**-1, 10**3),
    x2_range=(10**-1, 10**3),
    x3_range=(10**-1, 10**3),
    num_columns=3,
)

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-0.1, 2], [-2, -0.1], [-0.3]],
    variable_names=[["x1", "x2"], ["x1", "x2"], ["x3"]],
    n=10**4,
    dt=0.01,
    init_conditions=initial_value_df.iloc[65].values,
    snr=49,
)


# %%
fig = ags.plot_trajectory_3d(x_t, "Linear 3d")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
# %%
model_compare.get_frequentist_results()
# %%
model_compare.get_identified_model_from_all_equations()
variable_coeff_identified, variable_names_identified = (
    model_compare.expressions_for_simulation()
)

# ! -------------------- Chen lee --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(5, 20),
    x2_range=(-20, 20),
    x3_range=(-20, 0),
    num_columns=3,
)

# x1_range=(-20, 20),
# x2_range=(-20, 20),
# x3_range=(-20, 20),

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[5, -1], [-10, 1], [-3.8, 1 / 3]],
    variable_names=[["x1", "x2x3"], ["x2", "x1x3"], ["x3", "x1x2"]],
    n=10**3.5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=49,
)


# %%
fig = ags.plot_trajectory_3d(x_t, "Chen Lee")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
# %%
# model_compare.get_frequentist_results()
model_compare.get_identified_model_from_all_equations()
# variable_coeff_identified, variable_names_identified = (
#     model_compare.expressions_for_simulation()
# )

# ! -------------------- Aizawa attractor --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(0.0, 1),
    num_columns=3,
)

# x1_range=(-2, 2),
# x2_range=(-2, 2),
# x3_range=(-1, 2), # Larger values (e.g., >2) or very negative ones (e.g., < -1) tend to destabilize the system due to cubic and nonlinear terms.

# x1_range=(-0.5, 0.5),
# x2_range=(0, 0.3),
# x3_range=(-0.1, 0.2),

# x1_range=(-1.5, 1.5),
# x2_range=(-1.5, 1.5),
# x3_range=(0.0, 2.0),

# x1_range=(-1.5, 1.5),
# x2_range=(-1.5, 1.5),
# x3_range=(0.0, 2.0), # Firstly suggested by gpt

# x1_range=(-0.5, 0.5),
# x2_range=(-0.5, 0.5),
# x3_range=(0.0, 0.3), # (prefered)

# x1_range=(-1, 1),
# x2_range=(-1, 1),
# x3_range=(0.0, 2),

# x1_range=(-1, 1),
# x2_range=(-1, 1),
# x3_range=(0.0, 1), # Second choice (60)

# x1_range=(-1.5, 1.5),
# x2_range=(-1.5, 1.5),
# x3_range=(0.0, 1.5),# Better than (-0.5, 0.5) highest success rate reached 68%
# %%
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
    n=10**3.5,
    dt=0.01,
    init_conditions=initial_value_df.iloc[75].values,
    snr=49,
)


# %%
fig = ags.plot_trajectory_3d(x_t, " Aizawa attractor")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=True)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
# %%
# model_compare.get_frequentist_results()
model_compare.get_identified_model_from_all_equations()
# variable_coeff_identified, variable_names_identified = (
#     model_compare.expressions_for_simulation()
# )

# ! -------------------- noise --------------------
# %%
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-15, 15),
    x2_range=(-15, 15),
    x3_range=(10, 40),
    num_columns=3,
)

# %%
x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-10, 10], [28, -1, -1], [1, -8 / 3]],
    variable_names=[["x1", "x2"], ["x1", "x2", "x1x3"], ["x1x2", "x3"]],
    n=10**3.4,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

x_n = ags.generate_white_noise(
    n=10**3.4,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=-17,
)

# %%
fig = ags.plot_trajectory_3d(x_t, "Lorenz")

# %%
design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.001,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

# %%
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=False)
model_compare.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)
# %%
model_compare.get_frequentist_results()
# %%
model_compare.get_identified_model_from_all_equations()
variable_coeff_identified, variable_names_identified = (
    model_compare.expressions_for_simulation()
)
