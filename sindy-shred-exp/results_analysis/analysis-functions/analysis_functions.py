# %%
import os
import re

import numpy as np
import pandas as pd


# %%
# Load recon RMSE results on the test dataset
def get_recon_rmse_test(dir_path):
    """
    Load all .npy files from the specified directory and return a dictionary
    with the file number as the key and the loaded numpy array as the value.
    """

    # Load all .npy files from the directory
    def extract_number_from_filename(filename):
        # Use regex to extract the number between 'test_rmse_' and '.npy'
        match = re.search(r"recon_rmse_test_(\d+)\.npy", filename)
        if match:
            return int(match.group(1))
        return None

    results = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            # Extract the number from the filename
            file_number = extract_number_from_filename(filename)
            if file_number is not None:
                # Load the numpy array
                file_path = os.path.join(dir_path, filename)
                data = np.load(file_path)

                # If data is just a single value, extract it directly
                if data.size == 1:
                    value = data.item()
                else:
                    value = data  # Keep as array if it's not a single value

                # Add the pair to results
                results.append((file_number, value))

    # Sort results by file number
    results.sort(key=lambda x: x[0])
    return results


def filter_recon_rmse_test_results(recon_rmse_test, threshold=0.05):
    """
    Filter test RMSE results to keep only those below a specified threshold.

    Parameters:
    -----------
    recon_rmse_test : list of tuples
        List of (file_number, value) tuples containing test RMSE results
    threshold : float, optional
        Maximum RMSE value to keep (default: 0.05)

    Returns:
    --------
    tuple
        (filtered_results, retained_file_numbers, filtering_stats)
        - filtered_results: List of filtered (file_number, value) tuples
        - retained_file_numbers: List of retained file numbers
        - filtering_stats: Dictionary with filtering statistics
    """
    # Filter out elements where RMSE > threshold
    filtered_test_rmse = [
        item for item in recon_rmse_test if item[1] <= threshold
    ]

    # Calculate filtering statistics
    original_count = len(recon_rmse_test)
    filtered_count = len(filtered_test_rmse)
    removed_count = original_count - filtered_count

    # Get the file numbers of retained results
    retained_file_numbers = [item[0] for item in filtered_test_rmse]

    # Create a dictionary with filtering statistics
    filtering_stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": removed_count,
        "threshold": threshold,
    }

    return filtered_test_rmse, retained_file_numbers, filtering_stats


# %%
# Dealing with latent MSE of the trained dataset or the test dataset


def load_latent_mse_to_dataframe(dir_path, data_type="train", method="sindy"):
    """
    Load all .npy files containing MSE dictionaries from the specified directory
    and return a pandas DataFrame with the metrics as columns and file numbers as rows.
    """
    if data_type == "train":
        if method == "sindy":
            pattern = r"latent_mse_sindy_train_(\d+)\.npy"
        elif method == "ba":
            pattern = r"latent_mse_ba_train_(\d+)\.npy"
    elif data_type == "test":
        if method == "sindy":
            pattern = r"latent_mse_sindy_test_(\d+)\.npy"
        elif method == "ba":
            pattern = r"latent_mse_ba_test_(\d+)\.npy"

    # Dictionary to store all data
    all_data = {}

    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            match = re.search(pattern, filename)
            if match:
                file_number = int(match.group(1))
                file_path = os.path.join(dir_path, filename)

                # Load dictionary from .npy file
                latent_mse_dict = np.load(file_path, allow_pickle=True).item()

                # Add file number to the dictionary
                latent_mse_dict["file_number"] = file_number

                # Store in our collection
                all_data[file_number] = latent_mse_dict

    # Convert to DataFrame
    if all_data:
        # Sort by file number
        sorted_data = [all_data[key] for key in sorted(all_data.keys())]
        df = pd.DataFrame(sorted_data)

        # Set file_number as index if desired
        df.set_index("file_number", inplace=True)

        return df
    else:
        return pd.DataFrame()


# %%
# FUnctions to filter the MSE DataFrame based on the retained cases after the filtering
# based on RMSE results
def filter_the_latent_mse_based_on_recon_rmse(
    latent_mse_df, retained_file_numbers_on_recon_rmse
):
    """
    Filter the MSE dataframe to keep only rows where all values are <= 1
    and whose indices are in retained_file_numbers_on_recon_rmse.

    Parameters:
    -----------
    latent_mse_df : pandas DataFrame
        DataFrame containing MSE values from PySINDy training
    retained_file_numbers_on_recon_rmse : list
        List of file numbers to retain

    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    # Retain the cases whose reconstruction RMSE is less than 0.05
    original_row_count = len(latent_mse_df)
    latent_mse_df_valid = latent_mse_df[
        latent_mse_df.index.isin(retained_file_numbers_on_recon_rmse)
    ]
    filtered_out_count = original_row_count - len(latent_mse_df_valid)
    print(f"Original dataframe shape: {original_row_count} rows")
    print(
        f"Filtered out {filtered_out_count} rows based on the recon rmse of the network framework."
    )
    print(f"Filtered dataframe shape: {len(latent_mse_df_valid)} rows")

    # Retain the cases where all MSEs of the identified governing equations are less than or equal to 1
    before_filter_count = len(latent_mse_df_valid)
    # Create a boolean mask where True means all values in that row are <= 1
    mask = (latent_mse_df_valid <= 1).all(axis=1)
    # Apply the mask to keep only rows where all values are <= 1
    latent_mse_df_filtered = latent_mse_df_valid[mask]
    after_filter_count = len(latent_mse_df_filtered)
    # Report how many rows were filtered out by this step
    latent_mse_filtered_count = before_filter_count - after_filter_count
    print(f"\nFiltered dataframe shape before: {before_filter_count} rows")
    print(
        f"Filtered out {latent_mse_filtered_count} rows whose indices are not in retained_file_numbers_on_recon_rmse."
    )
    print(f"Filtered dataframe shape after: {after_filter_count} rows")

    return latent_mse_df_filtered, latent_mse_df_valid


def display_performance_metrics(latent_mse_df_filtered):
    """
    Display performance metrics from test RMSE results and MSE DataFrame.

    Parameters:
    -----------
    test_rmse_results : list of tuples
        List of (file_number, value) tuples containing test RMSE results
    latent_mse_df_filtered : pandas DataFrame
        DataFrame containing MSE values from PySINDy training
    """

    print("\nMSE DataFrame:")
    print(latent_mse_df_filtered.head())

    # Summary of the latent_mse_df_filtered DataFrame
    print("\nSummary of MSE DataFrame:")
    print("Shape:", latent_mse_df_filtered.shape)
    print("\nColumn Names:", list(latent_mse_df_filtered.columns))
    print("\nDescriptive Statistics:")
    print(latent_mse_df_filtered.describe())
    print("\nMissing Values:")
    print(latent_mse_df_filtered.isnull().sum())
    print("\nAverage MSE by Column:")
    for column in latent_mse_df_filtered.columns:
        print(f"{column}: {latent_mse_df_filtered[column].mean():.6f}")


# calculate the reconstruction mse via the latent forecaster based on the index of latent_mse_df_filtered
def load_and_filter_forecaster_metrics(dir_path, method, valid_indices):
    """
    Load forecaster metrics from .npy files and filter based on valid indices.

    Parameters:
    -----------
    dir_path : str
        Directory path containing the forecaster metrics files
    method : str
        Method type ("sindy" or "ba") - currently both use the same pattern
    valid_indices : list or pandas.Index
        List of valid file numbers/indices to retain after filtering

    Returns:
    --------
    tuple
        (forecaster_metrics_df, forecaster_metrics_df_filtered)
        - forecaster_metrics_df: Complete DataFrame with all loaded metrics
        - forecaster_metrics_df_filtered: Filtered DataFrame with only valid indices
    """
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")

    if method not in ["sindy", "ba"]:
        raise ValueError("Method must be either 'sindy' or 'ba'")

    # Pattern is the same for both methods currently
    pattern = r"forecaster_metrics_test_(\d+)\.npy"

    # Dictionary to store all data
    all_data = {}

    try:
        for filename in os.listdir(dir_path):
            if filename.endswith(".npy"):
                match = re.search(pattern, filename)
                if match:
                    file_number = int(match.group(1))
                    file_path = os.path.join(dir_path, filename)

                    # Load dictionary from .npy file
                    recon_via_forecaster_mse_dict = np.load(
                        file_path, allow_pickle=True
                    ).item()

                    # Add file number to the dictionary
                    recon_via_forecaster_mse_dict["file_number"] = file_number

                    # Store in our collection
                    all_data[file_number] = recon_via_forecaster_mse_dict

    except Exception as e:
        raise RuntimeError(f"Error loading files from {dir_path}: {str(e)}")

    # Convert to DataFrame
    if all_data:
        # Sort by file number
        sorted_data = [all_data[key] for key in sorted(all_data.keys())]
        forecaster_metrics_df = pd.DataFrame(sorted_data)

        # Set file_number as index
        forecaster_metrics_df.set_index("file_number", inplace=True)
    else:
        forecaster_metrics_df = pd.DataFrame()
        print(f"Warning: No forecaster metrics files found in {dir_path}")

    # Filter based on valid indices
    if not forecaster_metrics_df.empty and len(valid_indices) > 0:
        forecaster_metrics_df_filtered = forecaster_metrics_df[
            forecaster_metrics_df.index.isin(valid_indices)
        ]
    else:
        forecaster_metrics_df_filtered = pd.DataFrame()

    return forecaster_metrics_df_filtered, forecaster_metrics_df


def get_recon_via_forecaster_metrics(
    forecaster_metrics_df_filtered,
    column_filter=None,
    exclude_normalized=True,
    method="",
):
    """
    Display performance metrics from the filtered forecaster metrics DataFrame.

    Parameters:
    -----------
    forecaster_metrics_df_filtered : pandas DataFrame
        DataFrame containing filtered forecaster metrics
    column_filter : str, optional
        String pattern to filter columns by (keeps columns containing this pattern)
    exclude_normalized : bool, optional
        If True and column_filter is "RMSE", excludes normalized RMSE columns (default: True)
    method : str, optional
        Method name to append to column names (e.g., "ba", "sindy")
    """
    print("\nForecaster Metrics DataFrame:")
    print(forecaster_metrics_df_filtered.head())

    print(f"\nShape: {forecaster_metrics_df_filtered.shape}")
    print(f"Column Names: {list(forecaster_metrics_df_filtered.columns)}")

    print("\nDescriptive Statistics:")
    print(forecaster_metrics_df_filtered.describe())

    print("\nMissing Values:")
    print(forecaster_metrics_df_filtered.isnull().sum())

    print("\nAverage by Column:")
    # Create DataFrame with column means
    metric_dic = {}
    for column in forecaster_metrics_df_filtered.columns:
        metric_dic[column] = [forecaster_metrics_df_filtered[column].mean()]

    metric_df = pd.DataFrame(metric_dic)

    # Filter columns if column_filter is provided
    if column_filter:
        filtered_columns = [
            col for col in metric_df.columns if column_filter in col
        ]

        # If filtering for RMSE and exclude_normalized is True, remove normalized columns
        # But only if the user didn't specifically ask for normalized columns
        if (
            column_filter == "RMSE"
            and exclude_normalized
            and not any(
                "normalized" in column_filter.lower() for _ in [column_filter]
            )
        ):
            filtered_columns = [
                col
                for col in filtered_columns
                if "normalized" not in col.lower()
            ]

        metric_df = metric_df[filtered_columns]

    # Extract time ranges and metric types from column names
    time_ranges = set()
    metric_types = set()

    for col in metric_df.columns:
        # Extract time range pattern (numbers at the end)
        time_match = re.search(r"(\d+)_(\d+)$", col)
        if time_match:
            start, end = time_match.groups()
            time_ranges.add((int(start), int(end)))
            # Extract metric type (everything before the time range)
            metric_type = re.sub(r"_\d+_\d+$", "", col)
            metric_types.add(metric_type)

    # Sort time ranges
    time_ranges = sorted(time_ranges)
    metric_types = sorted(metric_types)

    # Create new dataframe with time ranges as rows and metric types as columns
    if time_ranges and metric_types:
        new_data = {}
        for metric_type in metric_types:
            # Add method suffix to column name if method is provided
            column_name = f"{metric_type}_{method}" if method else metric_type
            new_data[column_name] = []
            for start, end in time_ranges:
                col_name = f"{metric_type}_{start}_{end}"
                if col_name in metric_df.columns:
                    new_data[column_name].append(metric_df[col_name].iloc[0])
                else:
                    new_data[column_name].append(None)

        # Create new dataframe
        time_range_labels = [f"({start}, {end})" for start, end in time_ranges]
        transformed_df = pd.DataFrame(new_data, index=time_range_labels)

        print(transformed_df)
        return transformed_df
    else:
        # Fallback to original behavior if no time ranges found
        metric_df = metric_df.T
        print(metric_df)
        return metric_df


# %%
def extract_forecaster_metrics_columns(
    forecaster_metric_df,
    metric_pattern,
    forecater="ba",
    excluded_ranges=[(0, 275)],  # Add more tuples as needed
):
    """
    Extract columns from DataFrame based on a metric pattern.

    Parameters:
    -----------
    forecaster_metric_df : pandas DataFrame
        The forecaster metrics DataFrame
    metric_pattern : str
        The metric pattern to search for (e.g., 'RMSE_normalized', 'MSE', 'RMSE')
    forecater : str
        The forecaster type (default: "ba")
    excluded_ranges : list of tuples
        Time ranges to exclude from the plot data (default: [(0, 275)])

    Returns:
    --------
    tuple
        (extracted_forecaster_metric_df, plot_df)
        - extracted_forecaster_metric_df: DataFrame containing only matching columns
        - plot_df: DataFrame prepared for plotting tasks
    """
    # Find columns that contain the metric pattern
    if metric_pattern == "RMSE":
        # For RMSE, exclude RMSE_normalized columns
        matching_columns = [
            col
            for col in forecaster_metric_df.columns
            if "RMSE" in col and "RMSE_normalized" not in col
        ]
    elif metric_pattern == "RMSE_normalized":
        # For RMSE_normalized, only include those specific columns
        matching_columns = [
            col
            for col in forecaster_metric_df.columns
            if "RMSE_normalized" in col
        ]
    else:
        # For other patterns, use simple substring matching
        matching_columns = [
            col for col in forecaster_metric_df.columns if metric_pattern in col
        ]

    if not matching_columns:
        print(f"No columns found containing '{metric_pattern}'")
        return pd.DataFrame(), pd.DataFrame()

    # Extract and return the matching columns
    extracted_forecaster_metric_df = forecaster_metric_df[
        matching_columns
    ].copy()

    # Prepare data for plotting tasks
    data_for_plot = []
    for col in extracted_forecaster_metric_df.columns:
        time_match = re.search(r"(\d+)_(\d+)$", col)
        if time_match:
            start, end = time_match.groups()
            metric_type = re.sub(r"_\d+_\d+$", "", col)

            # Process each row for this column
            for idx, value in extracted_forecaster_metric_df[col].items():
                data_for_plot.append(
                    {
                        "Time_Range": f"({start}, {end})",
                        metric_type: value,
                        "Forecaster": forecater,
                        "Index": idx,
                    }
                )

    plot_df = pd.DataFrame(data_for_plot)

    # Exclude multiple time ranges
    if not plot_df.empty:
        plot_df = plot_df[~plot_df["Time_Range"].isin(excluded_ranges)]

    return extracted_forecaster_metric_df, plot_df
