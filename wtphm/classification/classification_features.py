import numpy as np
from scipy.ndimage.interpolation import shift


def get_lagged_features(X, y, features_to_lag_inds, steps):
    """
    Returns an array with certain columns as lagged features for classification

    Args
    ----
    X: m*n np.ndarray
        The input features, with m samples and n features
    y: m*1 np.ndarray
        The m target values
    features_to_lag_inds: np.array
        The indices of the columns in `X` which will be lagged
    steps: int
        The number of lagging steps. This means for feature 'B' at time T,
        features will be added to X at T for B@(T-1), B@(T-2)...B@(T-steps).

    Returns
    -------
    X_lagged: np.ndarray
        An array with the original features and lagged features appended. The
        number of samples will necessarily be decreased because there will be
        some samples at the start with NA values for features.
    y_lagged: np.ndarray
        An updated array of target vaues corresponding to the new number of
        samples in `X_lagged`

    """
    # get a slice with columns of features to be lagged
    X_f = X[:, features_to_lag_inds]

    m = X_f.shape[0]
    n = X_f.shape[1]
    n_ = n * steps

    X_lagged = np.zeros((m, n_))

    for i in np.arange(0, steps):
        X_lagged[:, i * n:(i * n) + n] = shift(X_f, [i + 1, 0], cval=np.NaN)

    X_lagged = np.concatenate((X_f, X_lagged), axis=1)

    y_lagged = y[~np.isnan(X_lagged).any(axis=1)]
    X_lagged = X_lagged[~np.isnan(X_lagged).any(axis=1)]

    return X_lagged, y_lagged
