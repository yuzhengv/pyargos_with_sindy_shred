import numpy as np


def argos_standardize(mat, centers=None, scales=None, ddof=1):
    """
    Patched version of adelie's standardize function to fix the np.ndarray issue.

    Creates a standardized matrix by subtracting the mean and dividing by the standard deviation.

    Parameters:
    -----------
    mat : array-like
        Input matrix to standardize
    centers : array-like, optional
        Vector of center values to use for standardization
    scales : array-like, optional
        Vector of scale values to use for standardization
    ddof : int, default=1
        Delta degrees of freedom for standard deviation calculation
    n_threads : int, default=1
        Number of threads to use

    Returns:
    --------
    numpy.ndarray
        Standardized matrix
    """
    # Convert input to numpy array
    mat = np.array(mat, order="F", copy=True)

    # Calculate centers if not provided
    if centers is None:
        centers = np.mean(mat, axis=0)

    # Calculate scales if not provided
    if scales is None:
        scales = np.std(mat, axis=0, ddof=ddof)
        # Avoid division by zero
        scales[scales == 0] = 1.0

    # Standardize the matrix
    result = (mat - centers) / scales

    return result, centers, scales
