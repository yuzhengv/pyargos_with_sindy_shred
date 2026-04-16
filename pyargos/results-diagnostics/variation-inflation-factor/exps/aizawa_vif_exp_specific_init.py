# %%
import os
import sys

import dill  # Better serialization than pickle
import numpy as np

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import utils.argos_simulator as ags
import utils.argos_utils as agu
from src.argos_bayesian_argos import BayesianArgos

n_obs_list = [
    2,
    2.2,
    2.4,
    2.6,
    2.8,
    3,
    3.2,
    3.4,
    3.6,
    3.8,
    4,
    4.2,
    4.4,
    4.6,
    4.8,
    5,
]
# snr_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
system_name = "aizawa"

for n_obs in n_obs_list:
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
        n=10**n_obs,
        dt=0.01,
        init_conditions=np.array([-0.3, 0.2, 0.1]),
        snr=49,
    )

    design_matrix = agu.build_design_matrix(
        x_t=x_t,
        dt=0.01,
        library_degree=5,
        sg_poly_order=4,
        library_type="poly",
    )

    model = BayesianArgos(
        design_matrix=design_matrix, custom_prior=True, accelerator=False
    )
    model.run(
        mode="straight",
        parallel="yes",
        ci_level=0.90,
    )

    # Save model object to file
    with open(
        f"../../../data/n-analysis-aizawa-init/{system_name}_model_n_10_{int(n_obs * 10)}.dill",
        "wb",
    ) as f:
        dill.dump(model, f)
    print(f"Saved {system_name}_model_n_10_{int(n_obs * 10)}.dill")

# %%

# for snr_value in snr_list:
#     x_t = ags.generate_noisy_dynamical_systems(
#         variable_coeff=[
#             [-3.5, -0.7, 1],
#             [3.5, -0.7, 1],
#             [0.95, 0.65, 0.1, -1 / 3, -0.25, -1, -0.25, -1],
#         ],
#         variable_names=[
#             ["x2", "x1", "x1x3"],
#             ["x1", "x2", "x2x3"],
#             ["x3", "", "x1^3x3", "x3^3", "x1^2x3", "x1^2", "x2^2x3", "x2^2"],
#         ],
#         n=5000,
#         dt=0.01,
#         init_conditions=initial_value_df.iloc[35].values,
#         snr=snr_value,
#     )

#     design_matrix = agu.build_design_matrix(
#         x_t=x_t,
#         dt=0.01,
#         library_degree=5,
#         sg_poly_order=4,
#         library_type="poly",
#     )

#     model = BayesianArgos(design_matrix=design_matrix, accelerator=False)
#     model.run(
#         mode="straight",
#         parallel="yes",
#         ci_level=0.90,
#     )

#     snr_value = 62 if snr_value == np.inf else snr_value

#     # Save model object to file
#     with open(
#         f"../../../data/snr-analysis-aizawa-35/{system_name}_model_snr_{int(snr_value)}.dill",
#         "wb",
#     ) as f:
#         dill.dump(model, f)
#     print(f"Saved {system_name}_model_snr_{int(snr_value)}.dill")

# 95
