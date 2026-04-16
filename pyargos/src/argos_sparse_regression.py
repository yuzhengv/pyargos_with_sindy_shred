import traceback
import warnings

import numpy as np
import statsmodels.api as sm
from adelie import cv_grpnet, grpnet, matrix
from adelie.glm import gaussian
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.tools.tools import add_constant

from .adelie_custom import custom_cv_grpnet
from .argos_standardize import argos_standardize


def detect_cpu_cores():
    """
    Detect the number of CPU cores available.

    Returns
    -------
    int
        Number of CPU cores available.
    """
    try:
        import multiprocessing

        return multiprocessing.cpu_count()

    except Exception:
        return 1


def rescale_coefficients(coef, intercept, mean, scale):
    """Rescale the coefficients and intercept back to the original scale.

    Parameters
    ----------
    coef : array-like
        Coefficients to be rescaled.
    intercept : float
        Intercept to be rescaled.
    mean : array-like
        Mean used to standardize the predictors.
    scale : array-like
        Scale used to standardize the predictors.

    Returns
    -------
    coef_rescaled : array-like
        Rescaled coefficients.
    intercept_rescaled : float
        Rescaled intercept.
    """
    if intercept is not None:
        coef = np.true_divide(coef, scale)
        intercept = intercept - np.dot(coef, mean)

    else:
        coef = np.true_divide(coef, scale)
        intercept = 0

    return coef, intercept


def get_nonzero_terms(coef, design_martix, threshold=1e-10):
    """
    Extract nonzero coefficients and their corresponding feature names.

    Parameters:
    -----------
    coef : numpy.ndarray
        Array of coefficients from regression model
    sorted_feature_names : list or numpy.ndarray
        Names of features corresponding to coefficients
    threshold : float, default=1e-10
        Threshold for considering a coefficient as nonzero

    Returns:
    --------
    dict
        Dictionary mapping feature names to nonzero coefficients
    list
        List of indices of nonzero coefficients
    list
        List of nonzero feature names
    list
        List of nonzero coefficient values
    """
    sorted_feature_names = design_martix["sorted_feature_names"]
    monomial_orders = design_martix["monomial_orders"]

    # Get nonzero coefficients and their feature names
    coef_nonzero_indices = np.where(np.abs(coef) > threshold)[
        0
    ]  # Use threshold for numerical stability
    nonzero_features = sorted_feature_names[coef_nonzero_indices]
    nonzero_coefficients = coef[coef_nonzero_indices]
    nonzero_monomial_orders = monomial_orders[coef_nonzero_indices]

    # # Create a dictionary mapping feature names to coefficients for better interpretation
    # feature_coefficient_dict = dict(
    #     zip(nonzero_features, nonzero_coefficients, nonzero_monomial_orders)
    # )

    # Create a dictionary mapping feature names to coefficients for better interpretation
    feature_coefficient_dict = {}
    for i, feature in enumerate(nonzero_features):
        feature_coefficient_dict[feature] = {
            "coefficient": nonzero_coefficients[i],
            "monomial_order": nonzero_monomial_orders[i],
        }

    return (
        feature_coefficient_dict,
        coef_nonzero_indices,
        nonzero_features,
        nonzero_coefficients,
        nonzero_monomial_orders,
    )


def shrink_design_matrix_based_on_estimate(
    design_matrix, data, initial_estimate
):
    """
    Shrink design matrix based on initial estimate's non-zero terms.

    Args:
        design_matrix (dict): Original design matrix dictionary
        data (ndarray): Data array with derivative column and theta
        initial_estimate (ndarray): Initial coefficient estimates

    Returns:
        tuple: (new_data, new_design_matrix) with reduced features
    """

    def _shrinkage_design_martix(design_matrix, index):
        return {
            "sorted_theta": design_matrix["sorted_theta"][:, :index],
            "sorted_feature_names": design_matrix["sorted_feature_names"][
                :index
            ],
            "monomial_orders": design_matrix["monomial_orders"][:index],
        }

    # Remove first element (intercept) if present
    initial_estimate = initial_estimate[1:]

    # Replace NaN values with 0
    initial_estimate[np.isnan(initial_estimate)] = 0

    # Find the maximum index of non-zero elements
    non_zero_indices = np.where(initial_estimate != 0)[0]
    non_zero_index_max = (
        non_zero_indices.max() if len(non_zero_indices) > 0 else 0
    )

    monomial_orders = design_matrix["monomial_orders"]
    shrinkage_index = np.sum(
        monomial_orders <= monomial_orders[non_zero_index_max]
    )

    # Check if shrinkage_index is NA or equal to the length of monomial_orders
    if np.isnan(shrinkage_index) or shrinkage_index == len(monomial_orders):
        new_data = np.column_stack([data[:, 0], data[:, 1:]])
        new_design_matrix = _shrinkage_design_martix(
            design_matrix, len(monomial_orders)
        )
    else:
        new_data = np.column_stack(
            [data[:, 0], data[:, 1 : shrinkage_index + 1]]
        )
        new_design_matrix = _shrinkage_design_martix(
            design_matrix, shrinkage_index
        )

    return new_data, new_design_matrix, shrinkage_index


def argos_lasso(
    data,
    index=None,
    ols_ps=True,
    n_threads=detect_cpu_cores(),
    analysis_mode=False,
):
    """
    Perform LASSO regression with adaptive lambda selection and optimal model selection using BIC.

    Parameters:
    -----------
    data : numpy.ndarray
        Data array where the first column is the response variable and remaining columns are predictors
    index : array-like, optional
        Indices of observations to use
    standardize_method : str, default="glmnet"
        Method for standardizing predictors ("glmnet" or other)
    ols_ps : bool, default=True
        Whether to use OLS post-selection for coefficient estimation

    Returns:
    --------
    numpy.ndarray
        Array of regression coefficients
    """
    try:
        # - Internal process the input data
        if index is None:
            index = np.arange(data.shape[0])

        # - Extract predictors and response and standardize predictors
        # Extract predictors
        X = data[index, 1:]

        # Standardize the predictors
        scaler = StandardScaler().fit(X)
        mean = scaler.mean_
        scale = scaler.scale_
        X_matrix = scaler.transform(X)

        # Extract response, reshape to 1D array and create GLM for regression
        y = data[index, 0].reshape(-1)
        glm = gaussian(y)

        # - Initial lasso to build up a more suitable lambda grid
        # Perform cross-validated lasso
        lasso_init = cv_grpnet(
            X_matrix,
            glm,
            alpha=1,
            n_folds=10,
            intercept=True,
            progress_bar=False,
            n_threads=n_threads,
        )
        # Get lambda.min (best lambda) from CV results
        best_lambda = lasso_init.lmdas[lasso_init.best_idx]

        # - Second lasso if finer grid exits
        # Build up candicate lambda grid
        if best_lambda == lasso_init.lmdas[-1]:
            # If best lambda is at end of path, create finer grid
            lower_bound_grid = best_lambda / 10
            upper_bound_grid = min(lasso_init.lmdas[0], 1.1 * best_lambda)
            lambda_grid = np.linspace(upper_bound_grid, lower_bound_grid, 100)

            # Use our custom CV function instead of cv_grpnet
            lasso_second = custom_cv_grpnet(
                X=X_matrix,
                glm=glm,
                alpha=1,
                n_folds=10,
                lmda_path=lambda_grid,
                intercept=True,
                progress_bar=False,
                n_threads=n_threads,
            )
            best_lambda = lasso_second.lmdas[lasso_second.best_idx]

        # - Get the result using the best lambda value
        lasso_best = grpnet(
            X_matrix,
            glm,
            alpha=1,
            intercept=True,
            lmda_path=[best_lambda],
            n_threads=n_threads,
        )

        coef = lasso_best.betas.toarray().flatten()
        intercept = lasso_best.intercepts.flatten()

        # Rescale back the coefficients and intercept
        coef, intercept = rescale_coefficients(coef, intercept, mean, scale)
        coef_with_intercept = np.concatenate([intercept, coef])

        # - Apply thresholding to identify non-zero coefficients and build up the candidate model list
        threshold_sequence = np.power(10.0, np.arange(-8, 2))
        lasso_final_coefficients_list = []

        for threshold in threshold_sequence:
            thresholded_coef = np.where(
                np.abs(coef_with_intercept) <= threshold, 0, coef_with_intercept
            )
            if not np.all(thresholded_coef == 0):
                lasso_final_coefficients_list.append(thresholded_coef)

        if not lasso_final_coefficients_list:
            return np.full(data.shape[1], np.nan)

        # - Refit the candidate models only with the selected terms using OLS and select the best model based on BIC
        # Refit the candidate models using OLS
        ols_list = []
        bic_list = []
        for coef_candidate in lasso_final_coefficients_list:
            coef_nonzero = coef_candidate != 0

            if np.sum(coef_nonzero) > 0:
                # Handle different cases based on intercept
                if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                    # Case with intercept and some predictors
                    selected_X = X[:, coef_nonzero[1:]]
                    ols_model = sm.OLS(y, add_constant(selected_X)).fit()
                elif coef_nonzero[0]:
                    # Case with only intercept
                    ols_model = sm.OLS(y, np.ones((y.shape[0], 1))).fit()
                else:
                    # Case with no intercept but some predictors
                    selected_X = X[:, coef_nonzero[1:]]
                    ols_model = sm.OLS(y, selected_X).fit()

                ols_list.append(ols_model)
                bic_list.append(ols_model.bic)
            else:
                # No non-zero coefficients
                ols_list.append(None)
                bic_list.append(np.inf)

        # Find best model based on BIC
        best_model_idx = np.argmin(bic_list)

        # Extract coefficients from the best model
        lasso_ols_coefficients_list = []
        for i, coef_candidate in enumerate(lasso_final_coefficients_list):
            coef_nonzero = coef_candidate != 0
            vect_coef = np.zeros(data.shape[1])

            if ols_list[i] is not None:
                # Map OLS coefficients back to original dimensions
                if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                    # With intercept and predictors
                    vect_coef[0] = ols_list[i].params[0]  # Intercept
                    vect_coef[1:][coef_nonzero[1:]] = ols_list[i].params[1:]
                elif coef_nonzero[0]:
                    # Only intercept
                    vect_coef[0] = ols_list[i].params[0]
                else:
                    # No intercept, only predictors
                    vect_coef[1:][coef_nonzero[1:]] = ols_list[i].params

            lasso_ols_coefficients_list.append(vect_coef)

        # - Deliver the final coefficients
        lasso_final_coefficients = lasso_final_coefficients_list[best_model_idx]
        lasso_ols_coefficients = lasso_ols_coefficients_list[best_model_idx]

        coef_out = (
            lasso_ols_coefficients if ols_ps else lasso_final_coefficients
        )
        if analysis_mode:
            extra1 = lasso_init
            extra2 = lasso_second if "lasso_second" in locals() else None
            return coef_out, extra1, extra2
        return coef_out

    except Exception as e:
        warnings.warn(f"Error in lasso: {str(e)}")
        traceback.print_exc()
        return np.full(data.shape[1], np.nan)


def argos_alasso(
    data,
    index=None,
    weights_method="ridge",
    ols_ps=True,
    n_threads=detect_cpu_cores(),
    analysis_mode=False,
):
    """
    Perform adaptive lasso regression on the given data.

    Parameters
    ----------
    data : array-like
        The input data with the first column as the response variable
        and the remaining columns as predictors.
    index : array-like, optional
        Indices to select rows from the data. Default is None (use all rows).
    weights_method : str, optional
        Method to calculate initial weights. Either "ols" or "ridge". Default is "ridge".
    ols_ps : bool, optional
        Whether to use OLS post-selection for the final model. Default is True.

    Returns
    -------
    array-like
        The estimated coefficients (with intercept as the first element).
    """
    try:
        # - Internal process the input data
        if index is None:
            index = np.arange(data.shape[0])

        # - Extract predictors and response and standardize predictors
        # Extract predictors
        X = data[index, 1:]

        # Standardize the predictors
        scaler = StandardScaler().fit(X)
        mean = scaler.mean_
        scale = scaler.scale_
        X_matrix = scaler.transform(X)

        # Extract response, reshape to 1D array and create GLM for regression
        y = data[index, 0].reshape(-1)
        glm = gaussian(y)

        # - Create weights based on OLS or ridge regression
        if weights_method == "ols":
            # Create OLS weights
            try:
                ols_model = LinearRegression(fit_intercept=False).fit(X, y)
                weight = ols_model.coef_
                # Replace any NaNs with 0
                weight = np.nan_to_num(weight, nan=0.0)
            except Exception:
                # Fallback to ridge if OLS fails
                weights_method = "ridge"

        # Create weights based on ridge regression
        if weights_method == "ridge":
            # Create ridge weights using adelie
            ridge_init = cv_grpnet(
                X_matrix,
                glm,
                alpha=0,
                n_folds=10,
                intercept=True,
                progress_bar=False,
                n_threads=n_threads,
            )
            best_lambda = ridge_init.lmdas[ridge_init.best_idx]

            # Get a higher resolution lamda grid if the best lambda is the smallest
            if best_lambda == ridge_init.lmdas[-1]:
                lower_bound_grid = best_lambda / 10
                upper_bound_grid = min(ridge_init.lmdas[0], 1.1 * best_lambda)
                lambda_grid = np.linspace(
                    upper_bound_grid, lower_bound_grid, 100
                )

                # Refit with the new grid
                ridge_new = custom_cv_grpnet(
                    X_matrix,
                    glm,
                    alpha=0,
                    n_folds=10,
                    lmda_path=lambda_grid,
                    intercept=True,
                    progress_bar=False,
                    n_threads=n_threads,
                )
                best_lambda = ridge_new.lmdas[ridge_new.best_idx]

            # Fit the ridge model with the best lambda
            ridge_model = grpnet(
                X_matrix,
                glm,
                alpha=0,
                intercept=True,
                lmda_path=[best_lambda],
                n_threads=n_threads,
            )
            ridge_coef = ridge_model.betas.toarray().flatten()

            # Replace any NaNs with 0
            ridge_coef = np.nan_to_num(ridge_coef, nan=0.0)
            ridge_coef = rescale_coefficients(ridge_coef, None, mean, scale)[0]

            # Use only the feature weights (excluding intercept)
            weight = (
                ridge_coef if ridge_coef.shape[0] > 1 else np.ones(X.shape[1])
            )

        # - Perform adaptive lasso
        # Create penalty factors for adaptive lasso
        # Use 1/|weight| as penalty factors
        penalty_factors = 1.0 / np.abs(weight)

        # Handle division by zero
        penalty_factors = np.where(
            np.isinf(penalty_factors), np.finfo(float).max, penalty_factors
        )

        # Perform cross-validated adaptive lasso
        alasso_init = cv_grpnet(
            X_matrix,
            glm,
            alpha=1,
            n_folds=10,
            penalty=penalty_factors,
            intercept=True,
            progress_bar=False,
            n_threads=n_threads,
        )

        best_lambda = alasso_init.lmdas[alasso_init.best_idx]

        if best_lambda == alasso_init.lmdas[-1]:
            lower_bound_grid = best_lambda / 10
            upper_bound_grid = min(alasso_init.lmdas[0], 1.1 * best_lambda)
            lambda_grid = np.linspace(upper_bound_grid, lower_bound_grid, 100)
            if analysis_mode:
                lambda_grid_analysis = lambda_grid

            # Refit with the new grid
            alasso_second = custom_cv_grpnet(
                X_matrix,
                glm,
                alpha=1,
                n_folds=10,
                lmda_path=lambda_grid,
                penalty=penalty_factors,
                intercept=True,
                progress_bar=False,
                n_threads=n_threads,
            )
            best_lambda = alasso_second.lmdas[alasso_second.best_idx]

        # Fit the ridge model with the best lambda
        alasso_model = grpnet(
            X_matrix,
            glm,
            alpha=1,
            penalty=penalty_factors,
            intercept=True,
            lmda_path=[best_lambda],
            n_threads=n_threads,
        )

        # - Below code only for analysis purpose to get the full path of lambda and coefficients for visualization
        if analysis_mode:
            if lambda_grid_analysis is not None:
                print("use_high_resolution_lambda_grid")
                print("best lambda from analysis grid:", best_lambda)
                lambda_grid_analysis = np.linspace(
                    upper_bound_grid, lower_bound_grid / 2.5, 100
                )
                alasso_model_analysis = grpnet(
                    X_matrix,
                    glm,
                    alpha=1,
                    penalty=penalty_factors,
                    intercept=True,
                    lmda_path=lambda_grid_analysis,
                    early_exit=False,
                    n_threads=n_threads,
                )
                # early_exit directly affect whether the full lambda path or the best lambda will be stored in the model object
            else:
                print("use_default_lambda_grid")
                print("best lambda from analysis grid:", best_lambda)
                alasso_model_analysis = grpnet(
                    X_matrix,
                    glm,
                    alpha=1,
                    penalty=penalty_factors,
                    intercept=True,
                    early_exit=False,
                    n_threads=n_threads,
                )
        else:
            pass
        # - --------------------------------------------------------------------

        alasso_coef = alasso_model.betas.toarray().flatten()
        alasso_intercept = alasso_model.intercepts.flatten()

        # Rescale back the coefficients and intercept
        alasso_coef, alasso_intercept = rescale_coefficients(
            alasso_coef, alasso_intercept, mean, scale
        )
        alasso_coef_with_intercept = np.concatenate(
            [alasso_intercept, alasso_coef]
        )

        # - Apply thresholding to identify non-zero coefficients and build up the candidate model list
        threshold_sequence = np.power(10.0, np.arange(-8, 2))
        alasso_final_coefficients_list = []

        for threshold in threshold_sequence:
            thresholded_coef = np.where(
                np.abs(alasso_coef_with_intercept) <= threshold,
                0,
                alasso_coef_with_intercept,
            )
            if not np.all(thresholded_coef == 0):
                alasso_final_coefficients_list.append(thresholded_coef)

        if not alasso_final_coefficients_list:
            return np.full(data.shape[1], np.nan)

        # - Refit the candidate models only with the selected terms using OLS and select the best model based on BIC
        # Refit the candidate models using OLS
        ols_list = []
        bic_list = []
        for coef_candidate in alasso_final_coefficients_list:
            coef_nonzero = coef_candidate != 0

            if np.sum(coef_nonzero) > 0:
                # Handle different cases based on intercept
                if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                    # Case with intercept and some predictors
                    selected_X = X[:, coef_nonzero[1:]]
                    ols_model = sm.OLS(y, add_constant(selected_X)).fit()
                elif coef_nonzero[0]:
                    # Case with only intercept
                    ols_model = sm.OLS(y, np.ones((y.shape[0], 1))).fit()
                else:
                    # Case with no intercept but some predictors
                    selected_X = X[:, coef_nonzero[1:]]
                    ols_model = sm.OLS(y, selected_X).fit()

                ols_list.append(ols_model)
                bic_list.append(ols_model.bic)
            else:
                # No non-zero coefficients
                ols_list.append(None)
                bic_list.append(np.inf)

        # Find best model based on BIC
        best_model_idx = np.argmin(bic_list)

        # - Extract coefficients from the best model fitted by OLS
        alasso_ols_coefficients_list = []
        for i, coef_candidate in enumerate(alasso_final_coefficients_list):
            coef_nonzero = coef_candidate != 0
            vect_coef = np.zeros(data.shape[1])

            if ols_list[i] is not None:
                # Map OLS coefficients back to original dimensions
                if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                    # With intercept and predictors
                    vect_coef[0] = ols_list[i].params[0]  # Intercept
                    vect_coef[1:][coef_nonzero[1:]] = ols_list[i].params[1:]
                elif coef_nonzero[0]:
                    # Only intercept
                    vect_coef[0] = ols_list[i].params[0]
                else:
                    # No intercept, only predictors
                    vect_coef[1:][coef_nonzero[1:]] = ols_list[i].params

            alasso_ols_coefficients_list.append(vect_coef)

        alasso_final_coefficients = alasso_final_coefficients_list[
            best_model_idx
        ]
        alasso_ols_coefficients = alasso_ols_coefficients_list[best_model_idx]

        coef_out = (
            alasso_ols_coefficients if ols_ps else alasso_final_coefficients
        )
        if analysis_mode:
            extra1 = alasso_init
            extra2 = alasso_second if "alasso_second" in locals() else None
            extra3 = alasso_model_analysis
            return coef_out, extra1, extra2, extra3
        return coef_out

    except Exception as e:
        warnings.warn(f"Error in argos_alasso: {str(e)}")
        traceback.print_exc()
        return np.full(data.shape[1], np.nan)


def multi_stage_regression(
    design_matrix,
    data,
    weights_method_init="ridge",
    weights_method_final="ridge",
    analysis_mode=False,
):
    """
    Perform a two-stage sparse regression by first identifying non-zero terms
    and then refining the model on a reduced feature set.

    Args:
        design_matrix (dict): The design matrix with features
        data (ndarray): Data array with derivative column and theta
        state_var_deriv (int): State variable derivative to model (1-based index)

    Returns:
        tuple: (design_matrix, data, final_estimate, coef_nonzero, ols_model)
    """
    # - Run sparse regression on the original design matrix
    if not analysis_mode:
        initial_estimate = argos_alasso(
            data, weights_method=weights_method_init, ols_ps=True
        )

        # - Rebuild design matrix based on the previous build design matrix with theta order up to
        # - the last non-zero term
        data, design_matrix, shrinkage_index = (
            shrink_design_matrix_based_on_estimate(
                design_matrix=design_matrix,
                data=data,
                initial_estimate=initial_estimate,
            )
        )

        # - Run sparse regression on the shrinked design matrix
        final_estimate = argos_alasso(
            data, weights_method=weights_method_final, ols_ps=True
        )

        # - Run OLS regression on the final design matrix
        y = data[:, 0]
        X = data[:, 1:]

        final_estimate[np.isnan(final_estimate)] = 0

        coef_nonzero = final_estimate != 0

        if np.sum(coef_nonzero) > 0:
            # Handle different cases based on intercept
            if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                # Case with intercept and some predictors
                selected_X = X[:, coef_nonzero[1:]]
                ols_model = sm.OLS(y, sm.add_constant(selected_X)).fit()
            elif coef_nonzero[0]:
                # Case with only intercept
                ols_model = sm.OLS(y, np.ones((y.shape[0], 1))).fit()
            else:
                # Case with no intercept but some predictors
                selected_X = X[:, coef_nonzero[1:]]
                ols_model = sm.OLS(y, selected_X).fit()
        else:
            ols_model = None

        return design_matrix, data, final_estimate, coef_nonzero, ols_model

    else:
        (
            initial_estimate,
            init_alasso_first,
            init_alasso_second,
            init_alasso_model_analysis,
        ) = argos_alasso(
            data,
            weights_method=weights_method_init,
            ols_ps=True,
            analysis_mode=True,
        )

        # - Rebuild design matrix based on the previous build design matrix with theta order up to
        # - the last non-zero term
        data, design_matrix, shrinkage_index = (
            shrink_design_matrix_based_on_estimate(
                design_matrix=design_matrix,
                data=data,
                initial_estimate=initial_estimate,
            )
        )

        # - Run sparse regression on the shrinked design matrix
        (
            final_estimate,
            final_alasso_first,
            final_alasso_second,
            final_alasso_model_analysis,
        ) = argos_alasso(
            data,
            weights_method=weights_method_final,
            ols_ps=True,
            analysis_mode=True,
        )

        # - Run OLS regression on the final design matrix
        y = data[:, 0]
        X = data[:, 1:]

        final_estimate[np.isnan(final_estimate)] = 0

        coef_nonzero = final_estimate != 0

        if np.sum(coef_nonzero) > 0:
            # Handle different cases based on intercept
            if coef_nonzero[0] and np.any(coef_nonzero[1:]):
                # Case with intercept and some predictors
                selected_X = X[:, coef_nonzero[1:]]
                ols_model = sm.OLS(y, sm.add_constant(selected_X)).fit()
            elif coef_nonzero[0]:
                # Case with only intercept
                ols_model = sm.OLS(y, np.ones((y.shape[0], 1))).fit()
            else:
                # Case with no intercept but some predictors
                selected_X = X[:, coef_nonzero[1:]]
                ols_model = sm.OLS(y, selected_X).fit()
        else:
            ols_model = None

        return (
            design_matrix,
            data,
            final_estimate,
            coef_nonzero,
            ols_model,
            init_alasso_first,
            init_alasso_second,
            final_alasso_first,
            final_alasso_second,
            init_alasso_model_analysis,
            final_alasso_model_analysis,
        )
