import numpy as np
from sklearn.base import TransformerMixin
# from .batch_clustering import investigate_label
import pandas as pd
from scipy.ndimage.interpolation import shift


def label_stoppages(
        scada_data, fault_batches, drop_fault_batches=True, label_pre_stop=True,
        pre_stop_lims=['90 minutes', 0], batches_to_drop=None, drop_type=None):
    """Label times in the scada data which occurred during a stoppage and
    leading up to a stoppage as such.

    This adds a column to the passed ``scada_data``, ``stoppage``, and an
    optional column ``pre_stop``. ``stoppage`` is given a 1 if the scada point
    in question occurs during a stoppage, and ``pre_stop`` is given a 1 in the
    samples leading up to the stoppage. Both are 0 otherwise. These vary under
    different circumstances (see below).

    Args
    ----
    scada_data: pandas.DataFrame
        Full set of SCADA data for the turbine.
    fault_batches: pandas.DataFrame
        The dataframe holding the indices in event_data and start and end times
        for each batch (each batch related to a stoppage).
    drop_fault_batches: bool, default=True
        Whether to drop the actual entries which correspond to the batches, i.e.
        not the pre-fault data, but the fault data itself. This is highly
        recommended, as otherwise the stoppages themselves will be kept in the
        returned data, though the ``stoppage`` label for these entries will be
        labelled as "1", while the fault-free data will be labelled "0".
    label_pre_stop: bool; default=True
        If True, add a column to the returned ``scada_data_l`` for ``pre_stop``.
        Samples in the time leading up to a stoppage are given label 1, and 0
        otherwise.
    pre_stop_lims: 2*1 list of pd.Timedelta strs, default=['90 mins', 0]
        The amount of time before a stoppage to label scada as ``pre_stop``.
        E.g., by default, ``pre_stop`` is labelled as 1 in the time between 90
        mins and 0 mins before the stoppage occurs. If ['120 mins', '20 mins']
        is passed, scada samples from 120 minutes before until 20 minutes before
        the stoppage are given the ``pre_stop`` label 1.
    batches_to_drop: pd.DataFrame, optional; default=None
        Additional batches which should be dropped from the scada data. If this
        is passed, ``drop_type`` must be given a string as well.
    drop_type: str, optional; default=None
        Only used when ``batches_to_drop`` has been passed.
        If 'both', the stoppage and pre-stop entries (according to
        pre_stop_lims) corresponding to batches in ``batches_to_drop`` are
        dropped from the scada data.
        If 'stop', only the stoppage entries are dropped
        If 'pre', opnly the pre-stop entries are dropped

    Returns
    -------
    scada_data_l: pd.DataFrame
        The original scada_data dataframe with the ``pre_stop``, ``stoppage``
        and ``batch_id`` columns added
    """

    if (batches_to_drop is not None) and (drop_type is None):
        raise ValueError(
            'batches_to_drop has been passed, but no drop_type has been passed')

    if drop_type:
        if len(batches_to_drop) == 0:
            raise ValueError(
                'drop_type has been passed, but the length of batches to drop '
                'is 0')
        elif batches_to_drop is None:
            raise ValueError(
                'drop_type has been passed, but no batches_to_drop have been '
                'passed')
        else:
            scada_data_l = _drop_batches(
                scada_data.copy(), batches_to_drop, drop_type, pre_stop_lims)
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

        # don't label samples leading up to a fault as pre_stop if those samples
        # themselves occur during a stop
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


def _drop_batches(scada_data, batches_to_drop, drop_type, pre_stop_lims):
    # whether or not to drop scada from certain batches
    if drop_type == 'both':
        drop_ids = _get_stoppage_ids(batches_to_drop, scada_data)[0]
        drop_ids = drop_ids.append(
            _get_pre_stop_ids(batches_to_drop, scada_data, pre_stop_lims)[0])
        scada_data = scada_data.drop(drop_ids)
    elif drop_type == 'stop':
        drop_ids = _get_stoppage_ids(batches_to_drop, scada_data)[0]
        scada_data = scada_data.drop(drop_ids)
    elif drop_type == 'pre':
        drop_ids = _get_pre_stop_ids(batches_to_drop, scada_data,
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
    """Returns an array with certain columns as lagged features

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



# def label_scada_data(
#         scada_data, events_data, clusters, batch_groups, cluster_labels,
#         batch_indices, fault_group='all_faults', label_before_fault=True,
#         fault_offset=9, factor=1):
#     """Label the scada data to give the time to failure and fault type at each
#     timestamp.

#     This adds two new columns to the passed `scada_data`, "fault" and "offset".

#     "fault" contains the label of the specific type of fault that was occurring.
#     This refers to a type of stoppage from the pre_stopvious clustering step. Note
#     that if clusters `[1, 2, 4, 6]` were passed as the different fault types to
#     label the scada data with, these would be labelled as `[1, 2, 3, 4]` (0 is
#     fault-free). If `label_before_fault = True`, then times leading up to the
#     fault (the amount of time is determined by `fault_offset`) are also labelled
#     as such. If not, only times when the fault was actually pre_stopsent are labelled
#     as such. Note also, that "fault" here refers to a particular type of
#     cluster, i.e. the fault is actually a specific equence of alarms, so that
#     we are trying to pre_stopdict when a certain sequence of alarms will occur.

#     "offset" is labelled as default 10 while a fault is pre_stopsent, and each
#     10-minute period leading up to the fault is labelled as 9, 8, 7, etc. so
#     that 9 means a fault is likely in the next 10 minutes, 8 20 minutes, 7 30
#     minutes, and so on all the way to 1 which means a fault is likely in 90
#     minutes.

#     Args
#     ----
#     scada_data:pandas.DataFrame
#         Full set of SCADA data for the turbine
#     events_data: pandas.DataFrame
#         Full set of events data for the turbine
#     clusters: list-like
#         This corresponds to the clusters of batch groups which have been found
#         from a pre_stopvious clustering step. Only the clusters with labels in this
#         list will be included in the labelling of the SCADA data. This means
#         that for each "type" of stoppage identified in the clustering step, only
#         the ones mentioned here will be attempted to be pre_stopdicted.
#     batch_groups: nested dictionary
#         The dictionary of indices of faults which occurred during the
#         stoppage. See create_batch_groups() function in this module for details.
#     cluster_labels:  numpy.ndarray
#         The labels for the clusters, with each one corresponding to a feature
#         vector in batch_indices
#     batch_indices: nunmpy.ndarray
#         Indices of batch_groups associated with each feature_array. Obtained
#         from the extract_batch_features() function in this module (see for
#         details).
#     fault_group: string, default=True
#         The fault group in batch_groups to return, i.e. 'rel_faults',
#         'all_faults' or 'pre_stopv_hr'. See create_batch_groups() function in this
#         module for details
#     label_before_fault: Boolean, default=True
#         Whether or not to label the times leading up to the fault (when
#         fault_offset is non-zero) as the fault also. E.g. if fault is pre_stopsent
#         at time T, whether to also label T-1, T-2, etc. with the fault label
#         as well. If not, these times are still labelled as zero.
#     fault_offset: int, default=9
#         This is used to determine how far in advance pre_stopdiction will be done.
#         When a fault is pre_stopsent, the "offset" column is labelled as
#         `fault_offset + 1`. By default, this will be 10, and the points leading
#         up to the fault will be 9, 8, etc. In this way, the fault_offset
#         dictates how far in advance the fault will be pre_stopdicted (9 means 90
#         minutes, 12 means 120 minutes, etc.)
#     factor: int, default=1
#         Used to multiply the factor_label by, so that (e.g.) numbers will be 10,
#         20, 30...100 rather than 1, 2, 3...10. Possibly useful for easier
#         regression

#     Returns
#     -------
#     scada_data: Pandas.DataFrame
#         The original scada_data with the added 'fault' and 'offset' columns

#     """

#     if type(fault_offset) != int:
#         raise TypeError("fault_offset must be an integer")
#     elif type(factor) != int:
#         raise TypeError("factor must be an integer")

#     scada_data = scada_data.copy()

#     scada_data = _label_fault_classes(
#         scada_data, events_data, clusters, batch_groups, cluster_labels,
#         batch_indices, fault_offset, factor)

#     scada_data = _backfill_labels(scada_data, fault_offset, label_before_fault)

#     scada_data['max_offset'] = fault_offset

#     return scada_data


# def _label_fault_classes(
#         scada_data, events_data, clusters, batch_groups, cluster_labels,
#         batch_indices, fault_offset, factor):
#     '''Provides the initial labels for the fault class and offset (i.e. labels
#     them as such when the fault is pre_stopsent)
#     '''

#     scada_data['offset'] = 0
#     scada_data['fault'] = 0
#     batches = {}

#     for c, l in zip(clusters, np.arange(1, len(clusters) + 1)):
#         batches[c] = investigate_label(batch_groups, cluster_labels, c,
#                                        batch_indices, 'all_faults')

#         for j in batches[c]:
#             # for each fault in the batch, label 'fault' during when the fault
#             # was pre_stopsent as l, i.e. the class label for that fault, and
#             # 'offset' as '(fault_offset * factor)'
#             scada_data.loc[
#                 (scada_data.time >= (events_data.loc[j].time_on.min() +
#                                      pd.Timedelta('5 minutes')).round('10T')) &
#                 (scada_data.time <= (events_data.loc[j].time_on.max() +
#                                      pd.Timedelta('5 minutes')).round('10T')) &
#                 (scada_data.turbine_num == events_data.loc[
#                     j, 'turbine_num'].iloc[0]),
#                 ['fault', 'offset']] = [l, (fault_offset + 1) * factor]

#     return scada_data


# def _backfill_labels(scada_data, fault_offset, label_before_fault):
#     '''the below is for filling up the 'offset' column in the rows leading to
#     the fault'''
#     fault_ind = scada_data[scada_data.offset != 0].index
#     offset_vals = np.arange(1, fault_offset + 2)

#     for i in fault_ind:
#         # select entries leading up to the fault, "pre_stopv_rows", intersected with
#         # "fake_rows" to use as a mask to decide which "offset_vals" to use as
#         # the "new_vals"
#         pre_stopv_rows = scada_data.loc[i - fault_offset:i].index
#         fake_rows = np.arange(i - fault_offset, i + 1)
#         new_vals = offset_vals[np.in1d(fake_rows, pre_stopv_rows)]

#         # keep the non-zero "cur_vals" (existing ones) so that the faults
#         # labelled for earlier values of "i" are not overwritten
#         cur_vals = scada_data.loc[pre_stopv_rows, 'offset'].values
#         new_vals[cur_vals != 0] = cur_vals[cur_vals != 0]

#         scada_data.loc[pre_stopv_rows, 'offset'] = new_vals

#         if label_before_fault is True:
#             fault_label = scada_data.loc[i, 'fault']
#             scada_data.loc[pre_stopv_rows, 'fault'] = fault_label

#     return scada_data


# def _backfill_labels2(scada_data, fault_offset, label_before_fault):
#     '''the below is for filling up the 'offset' column in the rows leading to
#     the fault'''
#     fault_ind = scada_data[scada_data.offset != 0].index
#     offset_vals = np.arange(1, fault_offset + 2)
#     new_data = []
#     print('wha?')
#     for i in fault_ind:
#         # select entries leading up to the fault, "pre_stopv_rows", intersected with
#         # "fake_rows" to use as a mask to decide which "offset_vals" to use as
#         # the "new_vals"
#         pre_stopv_rows = scada_data.loc[i - fault_offset:i].index
#         fake_rows = np.arange(i - fault_offset, i + 1)
#         new_vals = offset_vals[np.in1d(fake_rows, pre_stopv_rows)]

#         # keep the non-zero "cur_vals" (existing ones) so that the faults
#         # labelled for earlier values of "i" are not overwritten
#         cur_vals = scada_data.loc[pre_stopv_rows, 'offset'].values
#         new_vals[cur_vals != 0] = cur_vals[cur_vals != 0]

#         # new_data = ()

#         scada_data.loc[pre_stopv_rows, 'offset'] = new_vals

#         # if label_before_fault is True:
#         #     fault_label = scada_data.loc[i, 'fault']
#         #     scada_data.loc[pre_stopv_rows, 'fault'] = fault_label

#     return scada_data


# def balanced_subsample(x, y, maj_class=0, multiple=1):
#     """Creates a balanced training set by randomly undersampling the majority
#     class

#     Args
#     ----
#     x: np.ndarray or pd.DataFrame
#         The training data (features)
#     y: np.ndarray or pd.DataFrame
#         The target values
#     maj_class: int
#         The y-value of the majority class
#     multiple: int
#         Multiple of undersampled majority class values to select. E.g. if set to
#         2, then no. of samples will be 2*number of next largest class
#     """
#     if isinstance(x, pd.DataFrame):
#         x = np.array(x)

#     if isinstance(y, pd.DataFrame):
#         y = np.array(y)

#     class_xs = []

#     # the number of elements in the biggest minority class
#     max_elems = None

#     for yi in np.unique(y):
#         elems = x[y == yi]
#         class_xs.append((yi, elems))
#         if (yi != maj_class) and (max_elems is None or
#                                   elems.shape[0] > max_elems):
#             max_elems = elems.shape[0]
#     xs = []
#     ys = []

#     for ci, this_xs in class_xs:

#         if ci == maj_class:
#             np.random.shuffle(this_xs)

#         x_ = this_xs[:max_elems * multiple]
#         y_ = np.empty(len(x_))
#         y_.fill(ci)

#         xs.append(x_)
#         ys.append(y_)

#     xs = np.concatenate(xs)
#     ys = np.concatenate(ys)

#     return xs, ys


# class BasicFeature(TransformerMixin):

#     def __init__(self, feature_list):
#         self.feature_list = feature_list

#     def fit(self, X, y):
#         return self

#     def transform(self, X):
#         if type(X) is pd.DataFrame:
#             return X[self.feature_list]

