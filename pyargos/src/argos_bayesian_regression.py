import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from .bambi_prior_custom import customised_priors_for_bambi


def bayesian_regression_cores_setting():
    """
    Detect the number of CPU cores available.

    Returns
    -------
    int
        Number of CPU cores available.
    """
    try:
        import multiprocessing

        cpu_cores = multiprocessing.cpu_count()

        if cpu_cores > 4:
            cores = 4
        else:
            cores = cpu_cores

        return cores

    except Exception:
        return 1


def fit_bayesian_model(
    data,
    design_matrix,
    coef_nonzero,
    custom_prior=True,
    accelerator=False,
    draws=1000,
    cores=None,
):
    """
    Fit a Bayesian regression model using bambi based on selected features.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array with target in first column
    design_matrix : dict
        Dictionary containing design matrix information
    coef_nonzero : list or numpy.ndarray
        Boolean array indicating which coefficients are non-zero
    draws : int, optional
        Number of draws for MCMC sampling, default=2000

    Returns:
    --------
    model : bambi.Model
        The fitted Bayesian model
    results : InferenceData
        The arviz InferenceData object with sampling results
    results_summary : pd.DataFrame
        Summary of the Bayesian inference results
    """
    if cores is None:
        cores = bayesian_regression_cores_setting()

    # Create a DataFrame for the bayesian regression task
    target = data[:, 0]
    # Convert coef_nonzero to boolean array if it's not already
    coef_nonzero = np.array(coef_nonzero).astype(bool)

    # Create DataFrame for the model
    model_data = pd.DataFrame({"target": target})
    if len(coef_nonzero) > 1:
        selected_features = design_matrix["sorted_theta"][:, coef_nonzero[1:]]
        feature_names = [
            design_matrix["sorted_feature_names"][i]
            for i, selected in enumerate(coef_nonzero[1:])
            if selected
        ]
        for i, name in enumerate(feature_names):
            model_data[name] = selected_features[:, i]

    # Define the formula for bayesian regression based on the condition
    if coef_nonzero[0] and np.any(coef_nonzero[1:]):
        # Intercept and coefficients
        formula = "target ~ 1 + " + " + ".join(feature_names)
    elif coef_nonzero[0] and not np.any(coef_nonzero[1:]):
        # Intercept only
        formula = "target ~ 1"
    elif not coef_nonzero[0] and np.any(coef_nonzero[1:]):
        # No intercept, with coefficients
        formula = "target ~ 0 + " + " + ".join(feature_names)
    else:
        # Default case - include all features with intercept
        feature_names = design_matrix["sorted_feature_names"]
        model_data = pd.DataFrame({"target": target})
        for i, name in enumerate(feature_names):
            model_data[name] = design_matrix["sorted_theta"][:, i]
        formula = "target ~ 1 + " + " + ".join(feature_names)

    # Check and fix formula by enclosing feature names with spaces or special characters in backticks
    parts = formula.split("~")
    if len(parts) == 2:
        lhs, rhs = parts
        rhs_terms = [term.strip() for term in rhs.split("+")]

        # Process each term on the right hand side
        fixed_rhs_terms = []
        for term in rhs_terms:
            term = term.strip()
            # Skip intercept term "1" or "0"
            if term in ["0", "1"]:
                fixed_rhs_terms.append(term)
                continue

            # If term contains spaces or special characters and is not already enclosed in backticks
            if (" " in term or any(c in term for c in "+-*/^")) and not (
                term.startswith("`") and term.endswith("`")
            ):
                # Enclose in backticks
                term = f"`{term}`"
            fixed_rhs_terms.append(term)

        # Reconstruct the formula
        formula = f"{lhs.strip()} ~ {' + '.join(fixed_rhs_terms)}"

    # formula should be in the format formula = "target ~ 1 + x1 + `x1 x3`"

    if custom_prior:
        priors = customised_priors_for_bambi(formula, model_data)
        model = bmb.Model(
            formula,
            model_data,
            family="gaussian",
            link="identity",
            priors=priors,
            center_predictors=True,  # matches rstanarm's "intercept after centering"
            auto_scale=False,  # since you've already scaled explicitly
        )
    else:
        model = bmb.Model(formula, model_data, family="gaussian")

    if not accelerator:
        results = model.fit(
            draws=draws,
            tune=1000,
            cores=cores,
            idata_kwargs=dict(log_likelihood=True),
        )
    else:
        # kwargs = {
        #     "adapt.run": {"num_steps": 500},
        #     "progress_bar": False,
        #     "num_chains": 4,
        #     "num_draws": 500,
        #     "num_adapt_draws": 500,
        # }
        # results = model.fit(
        #     draws=draws, inference_method="blackjax_nuts", cores=cores, **kwargs
        # )
        results = model.fit(
            inference_method="nutpie",
            tune=1000,
            draws=draws,
            # idata_kwargs=dict(log_likelihood=True),
        )
    # results = model.fit(
    #     draws=draws, inference_method="numpyro_hmc", cores=cores
    # )

    # blackjax_nuts tfp_nuts numpyro_hmc

    results_summary = az.summary(results)

    return model, results, results_summary
