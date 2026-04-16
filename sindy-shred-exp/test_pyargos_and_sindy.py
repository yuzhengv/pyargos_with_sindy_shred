# %%
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

# import pysindy as ps
# from sklearn.preprocessing import MinMaxScaler
# from utils.processdata import TimeSeriesDataset, load_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle

import pyargos.utils.argos_simulator as ags
import pyargos.utils.argos_utils as agu
from pyargos.src.argos_bayesian_argos import BayesianArgos

# %%
save_dir = "saved-latent-space"
x = np.load(f"{save_dir}/latent_space_data.npy")

# %%
# poly_order = 2
# threshold = 0.05

# differentiation_method = ps.differentiation.FiniteDifference()
# # differentiation_method = ps.differentiation.SmoothedFiniteDifference()

# model = ps.SINDy(
#     optimizer=ps.STLSQ(threshold=0.8, alpha=0.05),
#     differentiation_method=differentiation_method,
#     feature_library=ps.PolynomialLibrary(degree=poly_order),
# )

# # model = ps.SINDy(
# #     optimizer=ps.MIOSR(group_sparsity=(2,2,2), alpha=5000),
# #     differentiation_method=differentiation_method,
# #     feature_library=ps.PolynomialLibrary(degree=poly_order),
# # )

# model.fit(x, t=1 / 52.0, ensemble=False)
# model.print()

# ! -------------------- Test bayesian_argos --------------------
# # %%
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
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[93].values,
    snr=49,
)

# # %%
# fig = ags.plot_trajectory_3d(x_t, "Lorenz")

# # %%
# design_matrix = agu.build_design_matrix(
#     x_t=x_t,
#     dt=0.01,
#     library_degree=5,
#     sg_poly_order=4,
#     library_type="poly",
# )
# ! --------------------  --------------------
# %%
design_matrix = agu.build_design_matrix(
    x_t=x,
    dt=1 / 52.0 * 0.1,
    library_degree=3,
    sg_poly_order=4,
    library_type="poly",
)

# %%
design_matrix["sorted_theta"].shape
design_matrix["sorted_feature_names"].shape

# %%
# Try with limited parallel processing
model_compare = BayesianArgos(design_matrix=design_matrix, accelerator=True)
model_results = model_compare.run(
    mode="comparison",
    parallel="yes",
    ci_level=0.95,
)

# %%
model_compare.get_identified_model_from_all_equations()
variable_coeff_identified, variable_names_identified = (
    model_compare.expressions_for_simulation()
)

# %%
# Save model_results dictionary to pickle file
print("Saving model results...")
with open("model_results/model_results_dict.pkl", "wb") as f:
    pickle.dump(model_results, f)

with open("model_results/coefficients.pkl", "wb") as f:
    pickle.dump(variable_coeff_identified, f)

with open("model_results/names.pkl", "wb") as f:
    pickle.dump(variable_names_identified, f)
