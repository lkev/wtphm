"""
This module contains functions for processing scada data ahead of using it for
fault detection or prognostics. Read more in the :ref:`label_scada` section
of the User Guide.
"""


import numpy as np
from scipy.ndimage.interpolation import shift
import pandas as pd


def label_stoppages(
        scada_data, fault_batches, drop_fault_batches=True,
        label_pre_stop=True, pre_stop_lims=['90 minutes', 0],
        oth_batches_to_drop=None, drop_type=None):
    """Label times in the scada data which occurred during a stoppage and
    leading up to a stoppage as such.

    This adds a column to the passed ``scada_data``, "stoppage", and an
    optional column "pre_stop". "stoppage" is given a 1 if the scada point
    in question occurs during a stoppage, and "pre_stop" is given a 1 in the
    samples leading up to the stoppage. Both are 0 otherwise. These vary under
    different circumstances (see below). It also adds a "batch_id" column.
    For entries with a "pre_stop" or
    "stoppage" column of 1, "batch_id" corresponds to the batch giving it
    that label.

    Args
    ----
    scada_data: pandas.DataFrame
        Full set of SCADA data for the turbine.
    fault_batches: pandas.DataFrame
        The dataframe of batches of fault events, a subset of the output of
        :func:wtphm.batch.get_batch_data`
    drop_fault_batches: bool, default=True
        Whether to drop the scada entries which correspond to the stoppage
        periods covered by ``fault_batches``.
        i.e. not the pre-fault data, but the fault data itself. This is highly
        recommended, as otherwise the stoppages themselves will be kept in the
        returned data, though the "stoppage" column for these entries will be
        labelled as "1", while the fault-free data will be labelled "0".
    label_pre_stop: bool; default=True
        If True, add a column to the returned ``scada_data_l`` for
        "pre_stop". Samples in the time leading up to a stoppage are given
        label 1, and 0 otherwise.
    pre_stop_lims: 2*1 list of ``pd.Timedelta``-compatible strings,\
        default=['90 mins', 0]
        The amount of time before a stoppage to label scada as "pre_stop".
        E.g., by default, "pre_stop" is labelled as 1 in the time between 90
        mins and 0 mins before the stoppage occurs. If ['120 mins', '20 mins']
        is passed, scada samples from 120 minutes before until 20 minutes
        before the stoppage are given the "pre_stop" label 1.
    oth_batches_to_drop: pd.DataFrame, optional; default=None
        Additional batches, independent of dropping the ``fault_batches`` if
        ``drop_fault_batches`` is passed, which should be dropped from the
        scada data. If this is passed, ``drop_type`` must be given a string
        as well.
    drop_type: str, optional; default=None
        Only used when ``oth_batches_to_drop`` has been passed.
        If 'both', the stoppage and pre-stop entries (according to
        pre_stop_lims) corresponding to batches in ``oth_batches_to_drop`` are
        dropped from the scada data.
        If 'stop', only the stoppage entries are dropped
        If 'pre', opnly the pre-stop entries are dropped

    Returns
    -------
    scada_data_l: pd.DataFrame
        The original scada_data dataframe with the "pre_stop", "stoppage"
        and "batch_id" columns added.
    """

    if (oth_batches_to_drop is not None) and (drop_type is None):
        raise ValueError(
            'oth_batches_to_drop has been passed, but no drop_type has been '
            'passed')

    if drop_type:
        if len(oth_batches_to_drop) == 0:
            raise ValueError(
                'drop_type has been passed, but the length of batches to drop '
                'is 0')
        elif oth_batches_to_drop is None:
            raise ValueError(
                'drop_type has been passed, but no oth_batches_to_drop have '
                'been passed')
        else:
            scada_data_l = _drop_batches(
                scada_data.copy(), oth_batches_to_drop, drop_type,
                pre_stop_lims)
    else:
        scada_data_l = scada_data.copy()

    stoppage_ids, stoppage_batch_ids = _get_stoppage_ids(
        fault_batches, scada_data_l)

    # mark times when stoppages were present
    scada_data_l['stoppage'] = 0
    scada_data_l.loc[stoppage_ids, 'stoppage'] = 1

    # give the associated batch id
    scada_data_l['batch_id'] = -1
    for ids in stoppage_batch_ids.items():
        scada_data_l.loc[ids[1], 'batch_id'] = ids[0]

    # label the pre_stop
    if label_pre_stop is True:
        pre_stop_ids, pre_stop_batch_ids = _get_pre_stop_ids(
            fault_batches, scada_data_l, pre_stop_lims)
        scada_data_l['pre_stop'] = 0

        # don't label samples leading up to a fault as pre_stop if those
        # samples themselves occur during a stop
        pre_stop_ids = scada_data_l.loc[pre_stop_ids, 'pre_stop'].loc[
            scada_data_l.stoppage == 0].index
        scada_data_l.loc[pre_stop_ids, 'pre_stop'] = 1

        # give the associated batch id
        for ids in pre_stop_batch_ids.items():
            scada_data_l.loc[ids[1], 'batch_id'] = ids[0]

    # drop the fault stoppage entries
    if drop_fault_batches is True:
        scada_data_l = scada_data_l.drop(stoppage_ids)
    return scada_data_l


def _drop_batches(scada_data, oth_batches_to_drop, drop_type, pre_stop_lims):
    # whether or not to drop scada from certain batches
    if drop_type == 'both':
        drop_ids = _get_stoppage_ids(oth_batches_to_drop, scada_data)[0]
        drop_ids = drop_ids.append(
            _get_pre_stop_ids(oth_batches_to_drop, scada_data,
                              pre_stop_lims)[0])
        scada_data = scada_data.drop(drop_ids)
    elif drop_type == 'stop':
        drop_ids = _get_stoppage_ids(oth_batches_to_drop, scada_data)[0]
        scada_data = scada_data.drop(drop_ids)
    elif drop_type == 'pre':
        drop_ids = _get_pre_stop_ids(oth_batches_to_drop, scada_data,
                                     pre_stop_lims)[0]
        scada_data = scada_data.drop(drop_ids)
    elif drop_type is not None:
        raise ValueError(
            'drop_type must be one of \"both\", \"stop\", \"pre\" or '
            'None')

    return scada_data


def _get_stoppage_ids(fault_batches, scada_data_l):
    stoppage_ids = pd.Int64Index([])
    stoppage_batch_ids = {}
    for b in fault_batches.itertuples():
        start = (b.start_time + pd.Timedelta('5 minutes')).round('10T')
        end = (b.down_end_time + pd.Timedelta('5 minutes')).round('10T')
        cur_stoppage_ids = scada_data_l[
            (scada_data_l.time >= start) & (scada_data_l.time <= end) &
            (scada_data_l.turbine_num == b.turbine_num)].index
        stoppage_ids = stoppage_ids.append(cur_stoppage_ids)
        stoppage_batch_ids[b.Index] = cur_stoppage_ids
    return stoppage_ids, stoppage_batch_ids


def _get_pre_stop_ids(fault_batches, scada_data_l, pre_stop_lims):
    pre_stop_ids = pd.Int64Index([])
    pre_stop_batch_ids = {}
    for b in fault_batches.itertuples():
        start = (b.start_time + pd.Timedelta('5 minutes')).round('10T')
        cur_pre_stop_ids = scada_data_l[
            (scada_data_l.time >= start - pd.Timedelta(pre_stop_lims[0])) &
            (scada_data_l.time < start - pd.Timedelta(pre_stop_lims[1])) &
            (scada_data_l.turbine_num == b.turbine_num)].index
        cur_pre_stop_batch_ids = scada_data_l[
            (scada_data_l.time >= start - pd.Timedelta(pre_stop_lims[0])) &
            (scada_data_l.time < start) &
            (scada_data_l.turbine_num == b.turbine_num)].index
        pre_stop_ids = pre_stop_ids.append(cur_pre_stop_ids)
        pre_stop_batch_ids[b.Index] = cur_pre_stop_batch_ids
    return pre_stop_ids, pre_stop_batch_ids


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
        The indices of the columns in ``X`` which will be lagged
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
        samples in ``X_lagged``

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
