import multiprocessing

import arviz as az
import numpy as np
import pandas as pd

from pyargos.utils.argos_simulator import solve_ode_odeint

from .argos_bayesian_regression import fit_bayesian_model
from .argos_sparse_regression import get_nonzero_terms, multi_stage_regression


class BayesianArgos:
    """
    A class for Bayesian system identification using the ARGOS pipeline.

    This class provides methods for identifying dynamical systems using
    Bayesian regression techniques, with preprocessing via sparse regression
    methods.
    """

    def __init__(
        self,
        design_matrix=None,
        custom_prior=False,
        draws=1000,
        accelerator=False,
    ):
        """
        Initialize the BayesianArgos instance.

        Parameters:
        -----------
        design_matrix : dict, optional
            Dictionary containing the design matrix and related data
        """
        self.design_matrix = design_matrix
        self.custom_prior = custom_prior
        self.draws = draws
        self.accelerator = accelerator
        self.results = None

    def set_design_matrix(self, design_matrix):
        """
        Set the design matrix for system identification.

        Parameters:
        -----------
        design_matrix : dict
            Dictionary containing the design matrix and related data
        """
        self.design_matrix = design_matrix

    def extract_identified_model(self, results, ci_level=0.95):
        """
        Extract the identified model coefficients from Bayesian regression
        results.

        Parameters:
        -----------
        results : arviz.InferenceData
            Results from Bayesian model fitting
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation (0 < ci_level < 1)

        Returns:
        --------
        identified_df : pd.DataFrame
            DataFrame containing identified variables and their coefficients

        Raises:
        -------
        ValueError
            If results is not an arviz.InferenceData object or ci_level is outside (0,1)
        TypeError
            If the input arguments have incorrect types
        """
        # Validate inputs
        if not isinstance(results, az.InferenceData):
            raise TypeError("results must be an arviz.InferenceData object")

        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")

        # Extract posterior samples and calculate credible intervals
        try:
            posterior_samples = az.extract(results)
            ci = (
                az.hdi(results, hdi_prob=ci_level)
                .to_dataframe()
                .T.reset_index()
            )
        except Exception as e:
            raise ValueError(f"Failed to extract posterior samples: {str(e)}")

        # Get coefficient names and means from posterior
        coef_names = [name for name in ci["index"] if name != "sigma"]
        coef_means = {
            name: float(posterior_samples[name].mean().values)
            for name in coef_names
        }

        # Process coefficients
        identified_model = {}
        for name in coef_names:
            lower_bound = ci.loc[ci["index"] == name, "lower"].values[0]
            upper_bound = ci.loc[ci["index"] == name, "higher"].values[0]
            coef_mean = coef_means[name]

            # Check if CI contains variable and doesn't cross zero
            if lower_bound <= coef_mean <= upper_bound and not (
                lower_bound <= 0 <= upper_bound
            ):
                identified_model[name] = coef_mean
            else:
                identified_model[name] = 0

        # Filter out terms with zero coefficients
        identified_model = {
            name: coef for name, coef in identified_model.items() if coef != 0
        }

        # Convert to DataFrame for easier viewing
        identified_df = pd.DataFrame(
            list(identified_model.items()), columns=["Variable", "Coefficient"]
        )

        # identified_df = identified_df.sort_values(
        #     by="Coefficient", key=abs, ascending=False
        # )

        return identified_df

    def run_with_comparison(
        self,
        state_var_deriv=1,
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
    ):
        """
        Run Bayesian system identification pipeline comparing different weight
        methods.

        Parameters:
        -----------
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        parallel : str, default="yes"
            Whether to use parallel processing ("yes" or "no")
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation (0 < ci_level < 1)

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results

        Raises:
        -------
        ValueError
            If design_matrix is not set or state_var_deriv is invalid
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        # Validate ci_level
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")

        design_matrix = self.design_matrix
        sorted_theta = design_matrix["sorted_theta"]
        xdot = design_matrix["xdot_filtered"]

        # Check if parallel processing is requested and set number of cores
        if parallel != "no" and ncpus is None:
            ncpus = multiprocessing.cpu_count()

        # Create derivative data frames
        num_deriv_columns = xdot.shape[1]
        derivative_data = []

        for i in range(num_deriv_columns):
            deriv_col = xdot[:, i]
            # Combine derivative with sorted_theta
            dot_df = np.column_stack([deriv_col, sorted_theta])
            derivative_data.append(dot_df)

        # Access the desired data frame using the derivative variable
        data = derivative_data[state_var_deriv - 1]  # Convert to 0-based index

        # Run multi-stage regression with ridge weights
        (
            design_matrix_ridge,
            data_ridge,
            final_estimate_ridge,
            coef_nonzero_ridge,
            ols_model_ridge,
        ) = multi_stage_regression(design_matrix=design_matrix, data=data)

        # Run multi-stage regression with OLS weights
        (
            design_matrix_ols,
            data_ols,
            final_estimate_ols,
            coef_nonzero_ols,
            ols_model_ols,
        ) = multi_stage_regression(
            design_matrix=design_matrix,
            data=data,
            weights_method_init="ols",
            weights_method_final="ols",
        )

        # Choose the model with minimum BIC
        min_bic_model = ols_model_ridge.bic <= ols_model_ols.bic
        min_bic_idx = 0 if min_bic_model else 1

        if min_bic_idx == 0:
            design_matrix_final = design_matrix_ridge
            data_final = data_ridge
            final_estimate = final_estimate_ridge
            coef_nonzero = coef_nonzero_ridge
            ols_model = ols_model_ridge
        else:
            design_matrix_final = design_matrix_ols
            data_final = data_ols
            final_estimate = final_estimate_ols
            coef_nonzero = coef_nonzero_ols
            ols_model = ols_model_ols

        frequentist_regression_results = get_nonzero_terms(
            final_estimate[1:], design_matrix_final
        )

        # Fit Bayesian model
        model, results, results_summary = fit_bayesian_model(
            data_final,
            design_matrix_final,
            coef_nonzero,
            custom_prior=self.custom_prior,
            draws=self.draws,
            accelerator=self.accelerator,
        )

        # Extract identified model
        identified_model_df = self.extract_identified_model(
            results, ci_level=ci_level
        )

        self.results = {
            "model": model,
            "results": results,
            "results_summary": results_summary,
            "identified_model_df": identified_model_df,
            "design_matrix": design_matrix_final,
            "data": data_final,
            "final_estimate": final_estimate,
            "coef_nonzero": coef_nonzero,
            "ols_model": ols_model,
            "frequentist_regression_results": frequentist_regression_results,
        }

        return self.results

    def run_straight(
        self,
        state_var_deriv=1,
        weights_method_init="ridge",
        weights_method_final="ols",
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
    ):
        """
        Run Bayesian system identification pipeline with specified weight
        methods.

        Parameters:
        -----------
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        weights_method_init : str, default="ridge"
            Method for computing initial weights
        weights_method_final : str, default="ols"
            Method for computing final weights
        parallel : str, default="yes"
            Whether to use parallel processing
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        design_matrix = self.design_matrix
        sorted_theta = design_matrix["sorted_theta"]
        xdot = design_matrix["xdot_filtered"]

        # Check if parallel processing is requested and set number of cores
        if parallel != "no" and ncpus is None:
            ncpus = multiprocessing.cpu_count()

        # Create derivative data frames
        num_deriv_columns = xdot.shape[1]
        derivative_data = []

        for i in range(num_deriv_columns):
            deriv_col = xdot[:, i]
            # Combine derivative with sorted_theta
            dot_df = np.column_stack([deriv_col, sorted_theta])
            derivative_data.append(dot_df)

        # Access the desired data frame using the derivative variable
        data = derivative_data[state_var_deriv - 1]  # Convert to 0-based index

        # Run multi-stage regression with specified weights
        (
            design_matrix_final,
            data_final,
            final_estimate,
            coef_nonzero,
            ols_model,
        ) = multi_stage_regression(
            design_matrix=design_matrix,
            data=data,
            weights_method_init=weights_method_init,
            weights_method_final=weights_method_final,
        )

        frequentist_regression_results = get_nonzero_terms(
            final_estimate[1:], design_matrix_final
        )

        # Fit Bayesian model
        model, results, results_summary = fit_bayesian_model(
            data_final,
            design_matrix_final,
            coef_nonzero,
            custom_prior=self.custom_prior,
            draws=self.draws,
            accelerator=self.accelerator,
        )

        # Extract identified model
        identified_model_df = self.extract_identified_model(
            results, ci_level=ci_level
        )

        self.results = {
            "model": model,
            "results": results,
            "results_summary": results_summary,
            "identified_model_df": identified_model_df,
            "design_matrix": design_matrix_final,
            "data": data_final,
            "final_estimate": final_estimate,
            "coef_nonzero": coef_nonzero,
            "ols_model": ols_model,
            "frequentist_regression_results": frequentist_regression_results,
        }

        return self.results

    def run_each_equation(
        self,
        mode="comparison",
        state_var_deriv=1,
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
        weights_method_init=None,
        weights_method_final=None,
    ):
        """
        Run the Bayesian system identification pipeline with specified mode.

        Parameters:
        -----------
        mode : str, default="comparison"
            Mode for running the pipeline. Either "comparison" or "straight"
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        parallel : str, default="yes"
            Whether to use parallel processing ("yes" or "no")
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation
        weights_method_init : str or None, default=None
            Method for computing initial weights (only for "straight" mode)
        weights_method_final : str or None, default=None
            Method for computing final weights (only for "straight" mode)

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results

        Raises:
        -------
        ValueError
            If design_matrix is not set, mode is invalid, or state_var_deriv is invalid
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        # Validate mode
        if mode not in ["comparison", "straight"]:
            raise ValueError("Invalid mode. Choose 'comparison' or 'straight'.")

        # Validate state_var_deriv
        num_deriv_columns = self.design_matrix["xdot_filtered"].shape[1]
        if not 1 <= state_var_deriv <= num_deriv_columns:
            raise ValueError(
                f"state_var_deriv must be between 1 and {num_deriv_columns}"
            )

        if mode == "comparison":
            return self.run_with_comparison(
                state_var_deriv=state_var_deriv,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
            )
        elif mode == "straight":
            if weights_method_init is None:
                weights_method_init = "ridge"
            if weights_method_final is None:
                weights_method_final = "ols"
            return self.run_straight(
                state_var_deriv=state_var_deriv,
                weights_method_init=weights_method_init,
                weights_method_final=weights_method_final,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
            )
        else:
            raise ValueError("Invalid mode. Choose 'comparison' or 'straight'.")

    def run(
        self,
        mode="comparison",
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
        weights_method_init=None,
        weights_method_final=None,
    ):
        """
        Run Bayesian system identification for all equations in the system.

        This method iterates through all state derivative equations and
        identifies them using either comparison or straight mode.

        Parameters:
        -----------
        mode : str, default="comparison"
            Mode for running the pipeline. Either "comparison" or "straight"
        parallel : str, default="yes"
            Whether to use parallel processing
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation
        weights_method_init : str or None, default=None
            Method for computing initial weights (only for "straight" mode)
        weights_method_final : str or None, default=None
            Method for computing final weights (only for "straight" mode)

        Returns:
        --------
        dict
            Dictionary with identified models for each equation
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        design_matrix = self.design_matrix
        xdot = design_matrix["xdot_filtered"]

        num_deriv_columns = xdot.shape[1]

        # Create a dictionary to store all results from each equation
        all_results = {}

        for i in range(num_deriv_columns):
            # Run analysis for this equation
            equation_result = self.run_each_equation(
                mode=mode,
                state_var_deriv=i + 1,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
                weights_method_init=weights_method_init,
                weights_method_final=weights_method_final,
            )

            # Store the result with a key indicating which equation it corresponds to
            all_results[f"equation_{i + 1}"] = equation_result

        # Store all results in the instance
        self.results = all_results

        return self.results

    def get_identified_model_from_each_equation(self, equation_num=1):
        """
        Get the identified model dataframe for a specific equation.

        Parameters:
        -----------
        equation_num : int, default=1
            The equation number to retrieve (1-based index)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing identified variables and their coefficients,
            or None if run() has not been called yet or the equation doesn't exist.
        """
        if self.results is None:
            return None

        # Check if results is structured as individual equation or multiple equations
        if any(key.startswith("equation_") for key in self.results.keys()):
            # Results from run() method with multiple equations
            eq_key = f"equation_{equation_num}"
            if eq_key not in self.results:
                return None
            return self.results[eq_key].get("identified_model_df")
        else:
            # Results from single equation analysis
            return self.results.get("identified_model_df")

    def get_identified_model_from_all_equations(self):
        """
        Get the identified models from all equations after running the model.

        This method extracts the identified model DataFrames from the results
        of running the full system identification pipeline.

        Returns:
        --------
        dict or None
            Dictionary containing DataFrames with identified variables and their
            coefficients for each equation, or None if run() has not been called yet
            or if the results don't contain multiple equations.
        """
        if self.results is None:
            return None

        # Check if results is a dictionary of equation results
        if not any(key.startswith("equation_") for key in self.results.keys()):
            # If we only have results for a single equation,
            # wrap it in a dictionary with a single key
            if (
                isinstance(self.results, dict)
                and "identified_model_df" in self.results
            ):
                return {"equation_1": self.results["identified_model_df"]}
            return None

        identified_models = {}
        for eq_key, eq_results in self.results.items():
            if (
                isinstance(eq_results, dict)
                and "identified_model_df" in eq_results
            ):
                identified_models[eq_key] = eq_results["identified_model_df"]

        return identified_models

    def get_frequentist_results_from_each_equation(self, equation_num=1):
        """
        Get the frequentist regression results for a specific equation.

        Parameters:
        -----------
        equation_num : int, default=1
            The equation number to retrieve (1-based index)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing identified variables and their coefficients from
            frequentist regression, or None if run() has not been called yet or
            the equation doesn't exist.
        """
        if self.results is None:
            return None

        # Check if results is structured as individual equation or multiple equations
        if any(key.startswith("equation_") for key in self.results.keys()):
            # Results from run() method with multiple equations
            eq_key = f"equation_{equation_num}"
            if eq_key not in self.results:
                return None
            return self.results[eq_key].get("frequentist_regression_results")
        else:
            # Results from single equation analysis
            return self.results.get("frequentist_regression_results")

    def get_frequentist_results(self):
        """
        Get the frequentist regression results from all equations.

        This method extracts the frequentist regression result DataFrames from
        the results of running the system identification pipeline.

        Returns:
        --------
        dict or None
            Dictionary containing DataFrames with identified variables and their
            coefficients for each equation from frequentist regression, or None
            if run() has not been called yet or if the results don't contain
            multiple equations.
        """
        if self.results is None:
            return None

        # Check if results is a dictionary of equation results
        if not any(key.startswith("equation_") for key in self.results.keys()):
            # If we only have results for a single equation,
            # wrap it in a dictionary with a single key
            if (
                isinstance(self.results, dict)
                and "frequentist_regression_results" in self.results
            ):
                return {
                    "equation_1": self.results["frequentist_regression_results"]
                }
            return None

        frequentist_regression_results = {}
        for eq_key, eq_results in self.results.items():
            if (
                isinstance(eq_results, dict)
                and "frequentist_regression_results" in eq_results
            ):
                frequentist_regression_results[eq_key] = eq_results[
                    "frequentist_regression_results"
                ]

        return frequentist_regression_results

    def expressions_for_simulation(self, model_result=None):
        """Process result from get_identified_model_from_all_equations().

        Parses the identified model coefficients and variable names into a format
        that's easier to use for system representation.

        Parameters:
        -----------
        model_result : dict or None, default=None
            Dictionary with equation results (Variable and Coefficient).
            If None, uses results from get_identified_model_from_all_equations()

        Returns:
        --------
        tuple
            Tuple containing two elements:
            - variable_coeff: Nested list of coefficients by equation
            - variable_names: Nested list of variable names by equation
            ('Intercept' becomes '')

        Notes:
        ------
        This method requires that run() has been called previously to generate model results.
        """
        if model_result is None:
            model_result = self.get_identified_model_from_all_equations()
            if model_result is None:
                raise ValueError(
                    "No model results available. Run the model first."
                )

        def format_trig_functions(var_name):
            """
            Format trigonometric function patterns in variable names.
            Converts "sin_x" to "sin(x)" and "cos_x" to "cos(x)".

            Parameters:
            -----------
            var_name : str
                Variable name to check and format

            Returns:
            --------
            str
                Formatted variable name
            """
            # Check for sin_ or cos_ pattern
            if "sin_" in var_name:
                var_name = var_name.replace("sin_", "sin(") + ")"
            elif "cos_" in var_name:
                var_name = var_name.replace("cos_", "cos(") + ")"
            return var_name

        variable_coeff = []
        variable_names = []
        for eq_key in sorted(model_result.keys()):
            equation_df = model_result[eq_key]
            eq_coeffs = []
            eq_names = []
            for _, row in equation_df.sort_index().iterrows():
                var_name = row["Variable"]
                coeff = row["Coefficient"]
                if var_name == "Intercept":
                    var_name = ""
                    if coeff == 0:
                        continue  # Skip this variable if it's an intercept with zero coefficient

                # Remove spaces from variable names
                var_name = var_name.replace(" ", "")

                # Format trigonometric functions if present
                var_name = format_trig_functions(var_name)

                eq_coeffs.append(coeff)
                eq_names.append(var_name)
            variable_coeff.append(eq_coeffs)
            variable_names.append(eq_names)
        return variable_coeff, variable_names

    def simulate(
        self,
        variable_coeff=None,
        variable_names=None,
        n=5000,
        dt=0.01,
        init_conditions=np.array([1, 2, 3]),
    ):
        """
        Simulate the identified dynamical system using the extracted coefficients.

        Parameters:
        -----------
        variable_coeff : list of list of float, optional
            Nested list of coefficients by equation. If None, extracted from model results.
        variable_names : list of list of str, optional
            Nested list of variable names by equation. If None, extracted from model results.
        n : int, default=5000
            Number of time steps for simulation
        dt : float, default=0.01
            Time step size
        init_conditions : list or array, optional
            Initial conditions for the simulation. If None, zeros are used.

        Returns:
        --------
        numpy.ndarray
            Time series data from simulating the identified system

        Raises:
        -------
        ValueError
            If variable coefficients or names aren't available and model hasn't been run
        """
        if variable_coeff is None or variable_names is None:
            variable_coeff, variable_names = self.expressions_for_simulation()
            if variable_coeff is None or variable_names is None:
                raise ValueError(
                    "No expressions available. Run the model first."
                )

        t = np.arange(0, float(n) * dt, dt)
        x_simulation = solve_ode_odeint(
            variable_coeff, variable_names, init_conditions, t
        )
        return x_simulation


# For backward compatibility, keep the standalone functions
def extract_identified_model_from_azInference(results, ci_level=0.95):
    """
    Extract the identified model coefficients from Bayesian regression results.

    This function is kept for backward compatibility.

    Parameters:
    -----------
    results : arviz.InferenceData
        Results from Bayesian model fitting
    ci_level : float, default=0.95
        Credible interval level for coefficient evaluation

    Returns:
    --------
    identified_df : pd.DataFrame
        DataFrame containing identified variables and their coefficients
    """
    model = BayesianArgos()
    return model.extract_identified_model(results, ci_level)


def bayesian_argos_with_comparison(
    design_matrix,
    state_var_deriv=1,
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
):
    """
    Run the Bayesian system identification pipeline with comparison mode.

    This function is kept for backward compatibility.
    """
    model = BayesianArgos(design_matrix)
    return model.run_with_comparison(
        state_var_deriv=state_var_deriv,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
    )


def bayesian_argos_straight(
    design_matrix,
    state_var_deriv=1,
    weights_method_init="ridge",
    weights_method_final="ols",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
):
    """
    Run the Bayesian system identification pipeline with straight mode.

    This function is kept for backward compatibility.
    """
    model = BayesianArgos(design_matrix)
    return model.run_straight(
        state_var_deriv=state_var_deriv,
        weights_method_init=weights_method_init,
        weights_method_final=weights_method_final,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
    )


def bayesian_argos(
    mode="comparison",
    design_matrix=None,
    state_var_deriv=1,
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
    weights_method_init=None,
    weights_method_final=None,
):
    """
    Run the Bayesian system identification pipeline.

    This function is kept for backward compatibility.
    """
    model = BayesianArgos(design_matrix)
    return model.run(
        mode=mode,
        state_var_deriv=state_var_deriv,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
        weights_method_init=weights_method_init,
        weights_method_final=weights_method_final,
    )


class BayesianArgosAnalysis:
    """
    A class for Bayesian system identification using the ARGOS pipeline.

    This class provides methods for identifying dynamical systems using
    Bayesian regression techniques, with preprocessing via sparse regression
    methods.
    """

    def __init__(
        self,
        design_matrix=None,
        custom_prior=False,
        draws=1000,
        accelerator=False,
    ):
        """
        Initialize the BayesianArgos instance.

        Parameters:
        -----------
        design_matrix : dict, optional
            Dictionary containing the design matrix and related data
        """
        self.design_matrix = design_matrix
        self.custom_prior = custom_prior
        self.draws = draws
        self.accelerator = accelerator
        self.results = None

    def set_design_matrix(self, design_matrix):
        """
        Set the design matrix for system identification.

        Parameters:
        -----------
        design_matrix : dict
            Dictionary containing the design matrix and related data
        """
        self.design_matrix = design_matrix

    def extract_identified_model(self, results, ci_level=0.95):
        """
        Extract the identified model coefficients from Bayesian regression
        results.

        Parameters:
        -----------
        results : arviz.InferenceData
            Results from Bayesian model fitting
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation (0 < ci_level < 1)

        Returns:
        --------
        identified_df : pd.DataFrame
            DataFrame containing identified variables and their coefficients

        Raises:
        -------
        ValueError
            If results is not an arviz.InferenceData object or ci_level is outside (0,1)
        TypeError
            If the input arguments have incorrect types
        """
        # Validate inputs
        if not isinstance(results, az.InferenceData):
            raise TypeError("results must be an arviz.InferenceData object")

        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")

        # Extract posterior samples and calculate credible intervals
        try:
            posterior_samples = az.extract(results)
            ci = (
                az.hdi(results, hdi_prob=ci_level)
                .to_dataframe()
                .T.reset_index()
            )
        except Exception as e:
            raise ValueError(f"Failed to extract posterior samples: {str(e)}")

        # Get coefficient names and means from posterior
        coef_names = [name for name in ci["index"] if name != "sigma"]
        coef_means = {
            name: float(posterior_samples[name].mean().values)
            for name in coef_names
        }

        # Process coefficients
        identified_model = {}
        for name in coef_names:
            lower_bound = ci.loc[ci["index"] == name, "lower"].values[0]
            upper_bound = ci.loc[ci["index"] == name, "higher"].values[0]
            coef_mean = coef_means[name]

            # Check if CI contains variable and doesn't cross zero
            if lower_bound <= coef_mean <= upper_bound and not (
                lower_bound <= 0 <= upper_bound
            ):
                identified_model[name] = coef_mean
            else:
                identified_model[name] = 0

        # Filter out terms with zero coefficients
        identified_model = {
            name: coef for name, coef in identified_model.items() if coef != 0
        }

        # Convert to DataFrame for easier viewing
        identified_df = pd.DataFrame(
            list(identified_model.items()), columns=["Variable", "Coefficient"]
        )

        # identified_df = identified_df.sort_values(
        #     by="Coefficient", key=abs, ascending=False
        # )

        return identified_df

    def run_with_comparison(
        self,
        state_var_deriv=1,
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
    ):
        """
        Run Bayesian system identification pipeline comparing different weight
        methods.

        Parameters:
        -----------
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        parallel : str, default="yes"
            Whether to use parallel processing ("yes" or "no")
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation (0 < ci_level < 1)

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results

        Raises:
        -------
        ValueError
            If design_matrix is not set or state_var_deriv is invalid
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        # Validate ci_level
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1")

        design_matrix = self.design_matrix
        sorted_theta = design_matrix["sorted_theta"]
        xdot = design_matrix["xdot_filtered"]

        # Check if parallel processing is requested and set number of cores
        if parallel != "no" and ncpus is None:
            ncpus = multiprocessing.cpu_count()

        # Create derivative data frames
        num_deriv_columns = xdot.shape[1]
        derivative_data = []

        for i in range(num_deriv_columns):
            deriv_col = xdot[:, i]
            # Combine derivative with sorted_theta
            dot_df = np.column_stack([deriv_col, sorted_theta])
            derivative_data.append(dot_df)

        # Access the desired data frame using the derivative variable
        data = derivative_data[state_var_deriv - 1]  # Convert to 0-based index

        # Run multi-stage regression with ridge weights
        (
            design_matrix_ridge,
            data_ridge,
            final_estimate_ridge,
            coef_nonzero_ridge,
            ols_model_ridge,
            init_alasso_first_ridge,
            init_alasso_second_ridge,
            final_alasso_first_ridge,
            final_alasso_second_ridge,
            init_alasso_model_analysis_ridge,
            final_alasso_model_analysis_ridge,
        ) = multi_stage_regression(
            design_matrix=design_matrix, data=data, analysis_mode=True
        )

        # Run multi-stage regression with OLS weights
        (
            design_matrix_ols,
            data_ols,
            final_estimate_ols,
            coef_nonzero_ols,
            ols_model_ols,
            init_alasso_first_ols,
            init_alasso_second_ols,
            final_alasso_first_ols,
            final_alasso_second_ols,
            init_alasso_model_analysis_ols,
            final_alasso_model_analysis_ols,
        ) = multi_stage_regression(
            design_matrix=design_matrix,
            data=data,
            weights_method_init="ols",
            weights_method_final="ols",
            analysis_mode=True,
        )

        # Choose the model with minimum BIC
        min_bic_model = ols_model_ridge.bic <= ols_model_ols.bic
        min_bic_idx = 0 if min_bic_model else 1

        if min_bic_idx == 0:
            design_matrix_final = design_matrix_ridge
            data_final = data_ridge
            final_estimate = final_estimate_ridge
            coef_nonzero = coef_nonzero_ridge
            ols_model = ols_model_ridge
            init_alasso_first_model = init_alasso_first_ridge
            init_alasso_second_model = init_alasso_second_ridge
            final_alasso_first_model = final_alasso_first_ridge
            final_alasso_second_model = final_alasso_second_ridge
            init_alasso_model_analysis = init_alasso_model_analysis_ridge
            final_alasso_model_analysis = final_alasso_model_analysis_ridge
        else:
            design_matrix_final = design_matrix_ols
            data_final = data_ols
            final_estimate = final_estimate_ols
            coef_nonzero = coef_nonzero_ols
            ols_model = ols_model_ols
            init_alasso_first_model = init_alasso_first_ols
            init_alasso_second_model = init_alasso_second_ols
            final_alasso_first_model = final_alasso_first_ols
            final_alasso_second_model = final_alasso_second_ols
            init_alasso_model_analysis = init_alasso_model_analysis_ols
            final_alasso_model_analysis = final_alasso_model_analysis_ols

        frequentist_regression_results = get_nonzero_terms(
            final_estimate[1:], design_matrix_final
        )

        # Fit Bayesian model
        model, results, results_summary = fit_bayesian_model(
            data_final,
            design_matrix_final,
            coef_nonzero,
            custom_prior=self.custom_prior,
            draws=self.draws,
            accelerator=self.accelerator,
        )

        # Extract identified model
        identified_model_df = self.extract_identified_model(
            results, ci_level=ci_level
        )

        self.results = {
            "model": model,
            "results": results,
            "results_summary": results_summary,
            "identified_model_df": identified_model_df,
            "design_matrix": design_matrix_final,
            "data": data_final,
            "final_estimate": final_estimate,
            "coef_nonzero": coef_nonzero,
            "ols_model": ols_model,
            "frequentist_regression_results": frequentist_regression_results,
            "init_alasso_first_model": init_alasso_first_model,
            "init_alasso_second_model": init_alasso_second_model,
            "final_alasso_first_model": final_alasso_first_model,
            "final_alasso_second_model": final_alasso_second_model,
            "init_alasso_model_analysis": init_alasso_model_analysis,
            "final_alasso_model_analysis": final_alasso_model_analysis,
        }

        return self.results

    def run_straight(
        self,
        state_var_deriv=1,
        weights_method_init="ridge",
        weights_method_final="ols",
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
    ):
        """
        Run Bayesian system identification pipeline with specified weight
        methods.

        Parameters:
        -----------
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        weights_method_init : str, default="ridge"
            Method for computing initial weights
        weights_method_final : str, default="ols"
            Method for computing final weights
        parallel : str, default="yes"
            Whether to use parallel processing
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        design_matrix = self.design_matrix
        sorted_theta = design_matrix["sorted_theta"]
        xdot = design_matrix["xdot_filtered"]

        # Check if parallel processing is requested and set number of cores
        if parallel != "no" and ncpus is None:
            ncpus = multiprocessing.cpu_count()

        # Create derivative data frames
        num_deriv_columns = xdot.shape[1]
        derivative_data = []

        for i in range(num_deriv_columns):
            deriv_col = xdot[:, i]
            # Combine derivative with sorted_theta
            dot_df = np.column_stack([deriv_col, sorted_theta])
            derivative_data.append(dot_df)

        # Access the desired data frame using the derivative variable
        data = derivative_data[state_var_deriv - 1]  # Convert to 0-based index

        # Run multi-stage regression with specified weights
        (
            design_matrix_final,
            data_final,
            final_estimate,
            coef_nonzero,
            ols_model,
            init_alasso_first_model,
            init_alasso_second_model,
            final_alasso_first_model,
            final_alasso_second_model,
            init_alasso_model_analysis,
            final_alasso_model_analysis,
        ) = multi_stage_regression(
            design_matrix=design_matrix,
            data=data,
            weights_method_init=weights_method_init,
            weights_method_final=weights_method_final,
            analysis_mode=True,
        )

        frequentist_regression_results = get_nonzero_terms(
            final_estimate[1:], design_matrix_final
        )

        # Fit Bayesian model
        model, results, results_summary = fit_bayesian_model(
            data_final,
            design_matrix_final,
            coef_nonzero,
            custom_prior=self.custom_prior,
            draws=self.draws,
            accelerator=self.accelerator,
        )

        # Extract identified model
        identified_model_df = self.extract_identified_model(
            results, ci_level=ci_level
        )

        self.results = {
            "model": model,
            "results": results,
            "results_summary": results_summary,
            "identified_model_df": identified_model_df,
            "design_matrix": design_matrix_final,
            "data": data_final,
            "final_estimate": final_estimate,
            "coef_nonzero": coef_nonzero,
            "ols_model": ols_model,
            "frequentist_regression_results": frequentist_regression_results,
            "init_alasso_first_model": init_alasso_first_model,
            "init_alasso_second_model": init_alasso_second_model,
            "final_alasso_first_model": final_alasso_first_model,
            "final_alasso_second_model": final_alasso_second_model,
            "init_alasso_model_analysis": init_alasso_model_analysis,
            "final_alasso_model_analysis": final_alasso_model_analysis,
        }

        return self.results

    def run_each_equation(
        self,
        mode="comparison",
        state_var_deriv=1,
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
        weights_method_init=None,
        weights_method_final=None,
    ):
        """
        Run the Bayesian system identification pipeline with specified mode.

        Parameters:
        -----------
        mode : str, default="comparison"
            Mode for running the pipeline. Either "comparison" or "straight"
        state_var_deriv : int, default=1
            Index of state variable derivative to identify (1-based index)
        parallel : str, default="yes"
            Whether to use parallel processing ("yes" or "no")
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation
        weights_method_init : str or None, default=None
            Method for computing initial weights (only for "straight" mode)
        weights_method_final : str or None, default=None
            Method for computing final weights (only for "straight" mode)

        Returns:
        --------
        dict
            Dictionary with identified model and intermediate results

        Raises:
        -------
        ValueError
            If design_matrix is not set, mode is invalid, or state_var_deriv is invalid
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        # Validate mode
        if mode not in ["comparison", "straight"]:
            raise ValueError("Invalid mode. Choose 'comparison' or 'straight'.")

        # Validate state_var_deriv
        num_deriv_columns = self.design_matrix["xdot_filtered"].shape[1]
        if not 1 <= state_var_deriv <= num_deriv_columns:
            raise ValueError(
                f"state_var_deriv must be between 1 and {num_deriv_columns}"
            )

        if mode == "comparison":
            return self.run_with_comparison(
                state_var_deriv=state_var_deriv,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
            )
        elif mode == "straight":
            if weights_method_init is None:
                weights_method_init = "ridge"
            if weights_method_final is None:
                weights_method_final = "ols"
            return self.run_straight(
                state_var_deriv=state_var_deriv,
                weights_method_init=weights_method_init,
                weights_method_final=weights_method_final,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
            )
        else:
            raise ValueError("Invalid mode. Choose 'comparison' or 'straight'.")

    def run(
        self,
        mode="comparison",
        parallel="yes",
        ncpus=None,
        ci_level=0.95,
        weights_method_init=None,
        weights_method_final=None,
    ):
        """
        Run Bayesian system identification for all equations in the system.

        This method iterates through all state derivative equations and
        identifies them using either comparison or straight mode.

        Parameters:
        -----------
        mode : str, default="comparison"
            Mode for running the pipeline. Either "comparison" or "straight"
        parallel : str, default="yes"
            Whether to use parallel processing
        ncpus : int or None, default=None
            Number of CPUs to use for parallel processing
        ci_level : float, default=0.95
            Credible interval level for coefficient evaluation
        weights_method_init : str or None, default=None
            Method for computing initial weights (only for "straight" mode)
        weights_method_final : str or None, default=None
            Method for computing final weights (only for "straight" mode)

        Returns:
        --------
        dict
            Dictionary with identified models for each equation
        """
        if self.design_matrix is None:
            raise ValueError(
                "Design matrix has not been set. Use set_design_matrix first."
            )

        design_matrix = self.design_matrix
        xdot = design_matrix["xdot_filtered"]

        num_deriv_columns = xdot.shape[1]

        # Create a dictionary to store all results from each equation
        all_results = {}

        for i in range(num_deriv_columns):
            # Run analysis for this equation
            equation_result = self.run_each_equation(
                mode=mode,
                state_var_deriv=i + 1,
                parallel=parallel,
                ncpus=ncpus,
                ci_level=ci_level,
                weights_method_init=weights_method_init,
                weights_method_final=weights_method_final,
            )

            # Store the result with a key indicating which equation it corresponds to
            all_results[f"equation_{i + 1}"] = equation_result

        # Store all results in the instance
        self.results = all_results

        return self.results

    def get_identified_model_from_each_equation(self, equation_num=1):
        """
        Get the identified model dataframe for a specific equation.

        Parameters:
        -----------
        equation_num : int, default=1
            The equation number to retrieve (1-based index)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing identified variables and their coefficients,
            or None if run() has not been called yet or the equation doesn't exist.
        """
        if self.results is None:
            return None

        # Check if results is structured as individual equation or multiple equations
        if any(key.startswith("equation_") for key in self.results.keys()):
            # Results from run() method with multiple equations
            eq_key = f"equation_{equation_num}"
            if eq_key not in self.results:
                return None
            return self.results[eq_key].get("identified_model_df")
        else:
            # Results from single equation analysis
            return self.results.get("identified_model_df")

    def get_identified_model_from_all_equations(self):
        """
        Get the identified models from all equations after running the model.

        This method extracts the identified model DataFrames from the results
        of running the full system identification pipeline.

        Returns:
        --------
        dict or None
            Dictionary containing DataFrames with identified variables and their
            coefficients for each equation, or None if run() has not been called yet
            or if the results don't contain multiple equations.
        """
        if self.results is None:
            return None

        # Check if results is a dictionary of equation results
        if not any(key.startswith("equation_") for key in self.results.keys()):
            # If we only have results for a single equation,
            # wrap it in a dictionary with a single key
            if (
                isinstance(self.results, dict)
                and "identified_model_df" in self.results
            ):
                return {"equation_1": self.results["identified_model_df"]}
            return None

        identified_models = {}
        for eq_key, eq_results in self.results.items():
            if (
                isinstance(eq_results, dict)
                and "identified_model_df" in eq_results
            ):
                identified_models[eq_key] = eq_results["identified_model_df"]

        return identified_models

    def get_frequentist_results_from_each_equation(self, equation_num=1):
        """
        Get the frequentist regression results for a specific equation.

        Parameters:
        -----------
        equation_num : int, default=1
            The equation number to retrieve (1-based index)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing identified variables and their coefficients from
            frequentist regression, or None if run() has not been called yet or
            the equation doesn't exist.
        """
        if self.results is None:
            return None

        # Check if results is structured as individual equation or multiple equations
        if any(key.startswith("equation_") for key in self.results.keys()):
            # Results from run() method with multiple equations
            eq_key = f"equation_{equation_num}"
            if eq_key not in self.results:
                return None
            return self.results[eq_key].get("frequentist_regression_results")
        else:
            # Results from single equation analysis
            return self.results.get("frequentist_regression_results")

    def get_frequentist_results(self):
        """
        Get the frequentist regression results from all equations.

        This method extracts the frequentist regression result DataFrames from
        the results of running the system identification pipeline.

        Returns:
        --------
        dict or None
            Dictionary containing DataFrames with identified variables and their
            coefficients for each equation from frequentist regression, or None
            if run() has not been called yet or if the results don't contain
            multiple equations.
        """
        if self.results is None:
            return None

        # Check if results is a dictionary of equation results
        if not any(key.startswith("equation_") for key in self.results.keys()):
            # If we only have results for a single equation,
            # wrap it in a dictionary with a single key
            if (
                isinstance(self.results, dict)
                and "frequentist_regression_results" in self.results
            ):
                return {
                    "equation_1": self.results["frequentist_regression_results"]
                }
            return None

        frequentist_regression_results = {}
        for eq_key, eq_results in self.results.items():
            if (
                isinstance(eq_results, dict)
                and "frequentist_regression_results" in eq_results
            ):
                frequentist_regression_results[eq_key] = eq_results[
                    "frequentist_regression_results"
                ]

        return frequentist_regression_results

    def expressions_for_simulation(self, model_result=None):
        """Process result from get_identified_model_from_all_equations().

        Parses the identified model coefficients and variable names into a format
        that's easier to use for system representation.

        Parameters:
        -----------
        model_result : dict or None, default=None
            Dictionary with equation results (Variable and Coefficient).
            If None, uses results from get_identified_model_from_all_equations()

        Returns:
        --------
        tuple
            Tuple containing two elements:
            - variable_coeff: Nested list of coefficients by equation
            - variable_names: Nested list of variable names by equation
            ('Intercept' becomes '')

        Notes:
        ------
        This method requires that run() has been called previously to generate model results.
        """
        if model_result is None:
            model_result = self.get_identified_model_from_all_equations()
            if model_result is None:
                raise ValueError(
                    "No model results available. Run the model first."
                )

        def format_trig_functions(var_name):
            """
            Format trigonometric function patterns in variable names.
            Converts "sin_x" to "sin(x)" and "cos_x" to "cos(x)".

            Parameters:
            -----------
            var_name : str
                Variable name to check and format

            Returns:
            --------
            str
                Formatted variable name
            """
            # Check for sin_ or cos_ pattern
            if "sin_" in var_name:
                var_name = var_name.replace("sin_", "sin(") + ")"
            elif "cos_" in var_name:
                var_name = var_name.replace("cos_", "cos(") + ")"
            return var_name

        variable_coeff = []
        variable_names = []
        for eq_key in sorted(model_result.keys()):
            equation_df = model_result[eq_key]
            eq_coeffs = []
            eq_names = []
            for _, row in equation_df.sort_index().iterrows():
                var_name = row["Variable"]
                coeff = row["Coefficient"]
                if var_name == "Intercept":
                    var_name = ""
                    if coeff == 0:
                        continue  # Skip this variable if it's an intercept with zero coefficient

                # Remove spaces from variable names
                var_name = var_name.replace(" ", "")

                # Format trigonometric functions if present
                var_name = format_trig_functions(var_name)

                eq_coeffs.append(coeff)
                eq_names.append(var_name)
            variable_coeff.append(eq_coeffs)
            variable_names.append(eq_names)
        return variable_coeff, variable_names

    def simulate(
        self,
        variable_coeff=None,
        variable_names=None,
        n=5000,
        dt=0.01,
        init_conditions=np.array([1, 2, 3]),
    ):
        """
        Simulate the identified dynamical system using the extracted coefficients.

        Parameters:
        -----------
        variable_coeff : list of list of float, optional
            Nested list of coefficients by equation. If None, extracted from model results.
        variable_names : list of list of str, optional
            Nested list of variable names by equation. If None, extracted from model results.
        n : int, default=5000
            Number of time steps for simulation
        dt : float, default=0.01
            Time step size
        init_conditions : list or array, optional
            Initial conditions for the simulation. If None, zeros are used.

        Returns:
        --------
        numpy.ndarray
            Time series data from simulating the identified system

        Raises:
        -------
        ValueError
            If variable coefficients or names aren't available and model hasn't been run
        """
        if variable_coeff is None or variable_names is None:
            variable_coeff, variable_names = self.expressions_for_simulation()
            if variable_coeff is None or variable_names is None:
                raise ValueError(
                    "No expressions available. Run the model first."
                )

        t = np.arange(0, float(n) * dt, dt)
        x_simulation = solve_ode_odeint(
            variable_coeff, variable_names, init_conditions, t
        )
        return x_simulation


# For backward compatibility, keep the standalone functions
def extract_identified_model_from_azInference_analysis(results, ci_level=0.95):
    """
    Extract the identified model coefficients from Bayesian regression results.

    This function is kept for backward compatibility.

    Parameters:
    -----------
    results : arviz.InferenceData
        Results from Bayesian model fitting
    ci_level : float, default=0.95
        Credible interval level for coefficient evaluation

    Returns:
    --------
    identified_df : pd.DataFrame
        DataFrame containing identified variables and their coefficients
    """
    model = BayesianArgosAnalysis()
    return model.extract_identified_model(results, ci_level)


def bayesian_argos_with_comparison_analysis(
    design_matrix,
    state_var_deriv=1,
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
):
    """
    Run the Bayesian system identification pipeline with comparison mode.

    This function is kept for backward compatibility.
    """
    model = BayesianArgosAnalysis(design_matrix)
    return model.run_with_comparison(
        state_var_deriv=state_var_deriv,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
    )


def bayesian_argos_straight_analysis(
    design_matrix,
    state_var_deriv=1,
    weights_method_init="ridge",
    weights_method_final="ols",
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
):
    """
    Run the Bayesian system identification pipeline with straight mode.

    This function is kept for backward compatibility.
    """
    model = BayesianArgosAnalysis(design_matrix)
    return model.run_straight(
        state_var_deriv=state_var_deriv,
        weights_method_init=weights_method_init,
        weights_method_final=weights_method_final,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
    )


def bayesian_argos_analysis(
    mode="comparison",
    design_matrix=None,
    state_var_deriv=1,
    parallel="yes",
    ncpus=None,
    ci_level=0.95,
    weights_method_init=None,
    weights_method_final=None,
):
    """
    Run the Bayesian system identification pipeline.

    This function is kept for backward compatibility.
    """
    model = BayesianArgosAnalysis(design_matrix)
    return model.run_each_equation(
        mode=mode,
        state_var_deriv=state_var_deriv,
        parallel=parallel,
        ncpus=ncpus,
        ci_level=ci_level,
        weights_method_init=weights_method_init,
        weights_method_final=weights_method_final,
    )
