import numpy as np
from .event_batches import investigate_label
import pandas as pd

def label_scada_data2(
        scada_data, events_data, clusters, batch_groups, cluster_labels,
        batch_indices, fault_group='all_faults', label_before_fault=True,
        fault_offset=10, factor=1):
    """Label the scada data to give the time to failure and fault type at each
    timestamp.

    This adds two new columns to the passed `scada_data`, "fault" and "offset".

    "fault" contains the label of the specific type of fault that was occurring.
    This refers to a type of stoppage from the previous clustering step. Note
    that if clusters `[1, 2, 4, 6]` were passed as the different fault types to
    label the scada data with, these would be labelled as `[1, 2, 3, 4]` (0 is
    fault-free). If `label_before_fault = True`, then times leading up to the
    fault (the amount of time is determined by `fault_offset`) are also labelled
    as such. If not, only times when the fault was actually present are labelled
    as such. Note also, that "fault" here refers to a particular type of
    cluster, i.e. the fault is actually a specific equence of alarms, so that
    we are trying to predict when a certain sequence of alarms will occur.

    "offset" is labelled as default 10 while a fault is present, and each
    10-minute period leading up to the fault is labelled as 9, 8, 7, etc. so
    that 9 means a fault is likely in the next 10 minutes, 8 20 minutes, 7 30
    minutes, and so on all the way to 1 which means a fault is likely in 90
    minutes.

    Args
    ----
    scada_data:pandas.DataFrame
        Full set of SCADA data for the turbine
    events_data: pandas.DataFrame
        Full set of events data for the turbine
    clusters: list-like
        This corresponds to the clusters of batch groups which have been found
        from a previous clustering step. Only the clusters with labels in this
        list will be included in the labelling of the SCADA data. This means
        that for each "type" of stoppage identified in the clustering step, only
        the ones mentioned here will be attempted to be predicted.
    batch_groups: nested dictionary
        The dictionary of indices of faults which occurred during the
        stoppage. See create_batch_groups() function in this module for details.
    cluster_labels:  numpy.ndarray
        The labels for the clusters, with each one corresponding to a feature
        vector in batch_indices
    batch_indices: nunmpy.ndarray
        Indices of batch_groups associated with each feature_array. Obtained
        from the extract_batch_features() function in this module (see for
        details).
    fault_group: string, default=True
        The fault group in batch_groups to return, i.e. 'rel_faults',
        'all_faults' or 'prev_hr'. See create_batch_groups() function in this
        module for details
    label_before_fault: Boolean, default=True
        Whether or not to label the times leading up to the fault (when
        fault_offset is non-zero) as the fault also. E.g. if fault is present
        at time T, whether to also label T-1, T-2, etc. with the fault label
        as well. If not, these times are still labelled as zero.
    fault_offset: int, default=10
        This is what the "offset" is labelled as when faults are present (i.e.
        during the fault). It also represents the number of 10-minute data
        points to include leading up to the fault, e.g. if it's 10, then the
        points leading up to the fault will be 10, 9, 8, etc. In this way, the
        fault_offset dictates how far in advance the fault will be predicted (10
        means 90 minutes, 13 means 120 minutes, etc.)
    factor: int, default=1
        Used to multiply the factor_label by, so that (e.g.) numbers will be 10,
        20, 30...100 rather than 1, 2, 3...10. Possibly useful for easier
        regression

    Returns
    -------
    scada_data: Pandas.DataFrame
        The original scada_data with the added 'fault' and 'offset' columns

    """

    if type(fault_offset) != int:
        raise TypeError("fault_offset must be an integer")
    elif type(factor) != int:
        raise TypeError("factor must be an integer")

    scada_data = scada_data.copy()

    scada_data = _label_fault_classes(
        scada_data, events_data, clusters, batch_groups, cluster_labels,
        batch_indices, fault_offset, factor)

    scada_data = _backfill_labels(scada_data, fault_offset, label_before_fault)

    return scada_data


def _label_fault_classes(
        scada_data, events_data, clusters, batch_groups, cluster_labels,
        batch_indices, fault_offset, factor):
    '''Provides the initial labels for the fault class and offset (i.e. labels
    them as such when the fault is present)
    '''

    scada_data['offset'] = 0
    scada_data['fault'] = 0
    batches = {}

    for c, l in zip(clusters, np.arange(1, len(clusters) + 1)):
        batches[c] = investigate_label(batch_groups, cluster_labels, c,
                                       batch_indices, 'all_faults')

        for j in batches[c]:
            # for each fault in the batch, label 'fault' during when the fault
            # was present as l, i.e. the class label for that fault, and
            # 'offset' as '(fault_offset * factor)'
            scada_data.loc[
                (scada_data.time >= (events_data.loc[j].time_on.min() -
                                     pd.Timedelta('5 minutes')).round('10T')) &
                (scada_data.time <= (events_data.loc[j].time_on.max() +
                                     pd.Timedelta('5 minutes')).round('10T')) &
                (scada_data.turbine_num == events_data.loc[
                    j, 'turbine_num'].iloc[0]),
                ['fault', 'offset']] = [l, (fault_offset + 1) * factor]

    return scada_data


def _backfill_labels(scada_data, fault_offset, label_before_fault):
    '''the below is for filling up the 'offset' column in the rows leading to the
    fault'''
    fault_ind = scada_data[scada_data.offset != 0].index
    offset_vals = np.arange(1, fault_offset + 2)

    for i in fault_ind:
        # select entries leading up to the fault, "prev_rows", intersected with
        # "fake_rows" to use as a mask to decide which "offset_vals" to use as
        # the "new_vals"
        prev_rows = scada_data.loc[i - fault_offset:i].index
        fake_rows = np.arange(i - fault_offset, i + 1)
        new_vals = offset_vals[np.in1d(fake_rows, prev_rows)]

        # keep the non-zero "cur_vals" (existing ones) so that the faults
        # labelled for earlier values of "i" are not overwritten
        cur_vals = scada_data.loc[prev_rows, 'offset'].values
        new_vals[cur_vals != 0] = cur_vals[cur_vals != 0]

        scada_data.loc[prev_rows, 'offset'] = new_vals

        if label_before_fault is True:
            fault_label = scada_data.loc[i, 'fault']
            scada_data.loc[prev_rows, 'fault'] = fault_label

    return scada_data
