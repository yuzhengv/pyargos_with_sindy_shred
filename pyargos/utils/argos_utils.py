import traceback
import warnings

import numpy as np
import statsmodels.api as sm
from adelie import cv_grpnet, grpnet
from adelie.diagnostic import diagnostic
from adelie.glm import gaussian
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tools.tools import add_constant


def sg_optimal_combination(x_t, dt=1, polyorder=None):
    """
    Find optimal Savitzky-Golay filter parameters by evaluating combinations
    of polynomial orders and window lengths.

    Parameters
    ----------
    x_t : numpy.ndarray
        A numeric vector or one-column matrix. The data to be smoothed.
    dt : float, optional
        A numeric scalar. The time-step interval of the data. Default is 1.
    polyorder : int, optional
        A numeric scalar. The order of the polynomial to be used in the
        Savitzky-Golay filter. If not specified, 4 will be used by default.

    Returns
    -------
    dict
        A dictionary with three elements:
        - sg_combinations: a matrix where each row represents a combination of
                          polynomial order and window length tried.
        - sg_order_wl: a vector of length 2 with the optimal polynomial order and
                       window length.
        - mse_df: a vector with the mean squared error of the differences between
                 the original data and the smoothed data for each combination.
    """
    # If polyorder is not specified, use 4
    if polyorder is None:
        polyorder = 4

    # Ensure x_t is a 2D array
    x_t = np.asarray(x_t).reshape(-1, 1)

    # Calculate window length maximum based on the data size
    wl_max = round(x_t.shape[0] * 0.05)
    # Ensure window length is odd (required by Savitzky-Golay)
    wl_max = wl_max + 1 if wl_max % 2 == 0 else wl_max

    # If the window length is too small, just use the minimum values
    if wl_max < 13:
        sg_combinations = np.array([[4, 13]])
    else:
        # Create window lengths (must be odd)
        if wl_max > 101:
            window_length = np.arange(5, 102, 2)
        else:
            if wl_max % 2 == 0:
                wl_max = wl_max + 1
            window_length = np.arange(5, wl_max + 1, 2)

        # Create combinations of polynomial order and window length
        sg_combinations = []
        for wl in window_length:
            if wl > polyorder + 7 - polyorder % 2:
                sg_combinations.append([polyorder, wl])

        sg_combinations = np.array(sg_combinations)

        # If no valid combinations, use default
        if len(sg_combinations) == 0:
            sg_combinations = np.array([[4, 13]])

    # Calculate MSE for each combination
    mse_values = []
    for i in range(sg_combinations.shape[0]):
        poly_order = sg_combinations[i, 0]
        window_length = sg_combinations[i, 1]

        # Apply Savitzky-Golay filter
        try:
            x_t_smoothed = savgol_filter(
                x_t.flatten(),
                window_length=int(window_length),
                polyorder=int(poly_order),
                deriv=0,
                delta=dt,
            )

            # Calculate MSE between original and smoothed data
            mse = mean_squared_error(x_t.flatten(), x_t_smoothed)
            mse_values.append(mse)
        except Exception as e:
            # If filtering fails, use a high MSE value
            warnings.warn(f"Error in Savitzky-Golay filter: {str(e)}")
            mse_values.append(float("inf"))

    # Convert to numpy array
    mse_values = np.array(mse_values)

    # Find the best combination (lowest MSE)
    best_idx = np.argmin(mse_values)
    sg_order_wl = sg_combinations[best_idx]

    return {
        "sg_combinations": sg_combinations,
        "sg_order_wl": sg_order_wl,
        "mse_df": mse_values,
    }


def build_design_matrix(
    x_t, dt=1, sg_poly_order=4, library_degree=5, library_type="poly"
):
    """
    Build design matrix from data after smoothing and derivative estimation.

    Parameters
    ----------
    x_t : numpy.ndarray
        Matrix of observations.
    dt : float, optional
        Time step (default is 1).
    sg_poly_order : int, optional
        Polynomial order for Savitzky-Golay Filter.
    library_degree : int, optional
        Degree of polynomial library (default is 5).
    library_type : str, optional
        Type of library to use. Can be one of "poly", "four", or "poly_four".

    Returns
    -------
    dict
        A dictionary with three elements:
        - sorted_theta: A matrix with sorted polynomial/trigonometric terms.
        - monomial_orders: A vector indicating the order of each polynomial term.
        - xdot_filtered: A matrix with derivative terms (dependent variable).
    """
    # Convert x_t to numpy array if it's not already
    x_t = np.asarray(x_t)

    # Filter x_t and compute derivatives
    num_columns = x_t.shape[1]
    x_filtered = []
    xdot_filtered = []

    for i in range(num_columns):
        # Only filter non-zero columns
        if np.any(x_t[:, i]):
            # Find optimal filter parameters
            sg_combination = sg_optimal_combination(
                x_t[:, i], dt, polyorder=sg_poly_order
            )
            sg_order = int(sg_combination["sg_order_wl"][0])
            sg_window = int(sg_combination["sg_order_wl"][1])

            # Apply filter for smoothed values
            x_smoothed = savgol_filter(
                x_t[:, i],
                window_length=sg_window,
                polyorder=sg_order,
                deriv=0,
                delta=dt,
            )
            x_filtered.append(x_smoothed)

            # Apply filter for derivatives
            x_deriv = savgol_filter(
                x_t[:, i],
                window_length=sg_window,
                polyorder=sg_order,
                deriv=1,
                delta=dt,
            )
            xdot_filtered.append(x_deriv)

    # Combine filtered data and derivatives
    x_filtered = np.column_stack(x_filtered)
    xdot_filtered = np.column_stack(xdot_filtered)

    # Name the derivative columns
    # xdot_column_names = [f"xdot{i+1}" for i in range(xdot_filtered.shape[1])]

    # Build library based on library_type
    if library_type in ["poly", "poly_four"]:
        # Polynomial expansion
        poly = PolynomialFeatures(degree=library_degree, include_bias=False)
        theta = poly.fit_transform(x_filtered)

        # Get feature names from polynomial features
        feature_names = poly.get_feature_names_out(
            input_features=[f"x{i + 1}" for i in range(x_filtered.shape[1])]
        )

        # Get monomial orders
        monomial_orders = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            # Sum the powers in each term
            powers = name.split()
            order = 0
            for power in powers:
                if "^" in power:
                    var, exp = power.split("^")
                    order += int(exp)
                elif power.startswith("x"):
                    # Variables without exponent have power 1
                    order += 1
            monomial_orders[i] = order

        # Sort by monomial order (Optional)
        # sort_idx = np.argsort(monomial_orders)
        # sorted_theta = theta[:, sort_idx]
        # sorted_feature_names = [feature_names[i] for i in sort_idx]
        # monomial_orders = monomial_orders[sort_idx]

        sorted_theta = theta
        sorted_feature_names = feature_names
        monomial_orders = monomial_orders

    if library_type in ["four", "poly_four"]:
        # Add trigonometric functions
        trig_functions = []
        trig_names = []
        for i in range(min(3, x_filtered.shape[1])):  # Up to 3 variables
            sin_term = np.sin(x_filtered[:, i])
            cos_term = np.cos(x_filtered[:, i])
            trig_functions.extend([sin_term, cos_term])
            trig_names.extend([f"sin_x{i + 1}", f"cos_x{i + 1}"])

        trig_functions = np.column_stack(trig_functions)
        trig_names = np.array(trig_names)

        # Each trig function has order 1
        trig_orders = np.ones(trig_names.shape[0])

        if library_type == "four":
            sorted_theta = trig_functions
            sorted_feature_names = trig_names
            monomial_orders = trig_orders
        else:  # poly_four
            # Combine poly and trig
            sorted_theta = np.column_stack([trig_functions, sorted_theta])
            sorted_feature_names = np.concatenate(
                [trig_names, sorted_feature_names]
            )
            monomial_orders = np.concatenate([trig_orders, monomial_orders])

    return {
        "sorted_theta": sorted_theta,
        "sorted_feature_names": sorted_feature_names,
        "monomial_orders": monomial_orders,
        "xdot_filtered": xdot_filtered,
        "x_filtered": x_filtered,
    }


class DesignMatrix:
    def __init__(
        self, x_t, dt=1, sg_poly_order=4, library_degree=5, library_type="poly"
    ):
        self.x_t = x_t
        self.dt = dt
        self.sg_poly_order = sg_poly_order
        self.library_degree = library_degree
        self.library_type = library_type

        # Build design matrix
        self.design_matrix = build_design_matrix(
            x_t, dt, sg_poly_order, library_degree, library_type
        )

    def get_design_matrix(self):
        return self.design_matrix

    def get_x_filtered(self):
        return self.design_matrix["x_filtered"]

    def get_xdot_filtered(self):
        return self.design_matrix["xdot_filtered"]

    def get_sorted_theta(self):
        return self.design_matrix["sorted_theta"]

    def get_sorted_feature_names(self):
        return self.design_matrix["sorted_feature_names"]

    def get_monomial_orders(self):
        return self.design_matrix["monomial_orders"]

    def get_x_t(self):
        return self.x_t

    def get_dt(self):
        return self.dt

    def get_sg_poly_order(self):
        return self.sg_poly_order

    def get_library_degree(self):
        return self.library_degree

    def get_library_type(self):
        return self.library_type
