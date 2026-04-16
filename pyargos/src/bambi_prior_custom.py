import bambi as bmb
import numpy as np
import pandas as pd
from formulae import design_matrices


def customised_priors_for_bambi(
    formula, data, family="gaussian", link="identity"
):
    dm = design_matrices(formula, data)

    # response vector
    y = np.asarray(dm.response).squeeze()

    # rstanarm's s_y and m_y rules
    if family == "gaussian":
        s_y = np.std(y, ddof=1)
        m_y = np.mean(y) if link == "identity" else 0.0
    else:
        s_y = 1.0
        m_y = 0.0

    X = dm.common.as_dataframe()

    priors = {
        "Intercept": bmb.Prior(
            "Normal", mu=m_y, sigma=2.5 * s_y, auto_scale=False
        ),
        # Only relevant when the model has a sigma-like auxiliary parameter (e.g., Gaussian)
        "sigma": bmb.Prior("Exponential", lam=1.0 / s_y, auto_scale=False),
    }

    for col in X.columns:
        if col == "Intercept":
            continue
        s_x = np.std(X[col].to_numpy(), ddof=1)
        if s_x == 0 or np.isnan(s_x):
            continue
        priors[col] = bmb.Prior(
            "Normal", mu=0.0, sigma=2.5 * s_y / s_x, auto_scale=False
        )

    return priors
