### DEPRECATED!!!! The contents of this file have now been included in the Batch
### class in batch.py. It is still included here so older notebooks don't break

import pandas as pd
from itertools import chain
import warnings


def rename_code_groups(event_data, fault_codes, code_groups):
    """This is the old name for get_grouped_fault_data"""
    warnings.warn(
        'rename_code_groups has changed name to get_grouped_fault_data. Please '
        'use that name from now on as it will probably be changed in future',
        FutureWarning)
    fault_data = get_grouped_fault_data(event_data, fault_codes, code_groups)

    return fault_data


def get_grouped_fault_data(event_data, fault_codes, code_groups):
    '''Returns a subset of the events dataframe consisting of fault events which
    are to be included for further analysis. Any groups of identical events that
    happen along different blade axes together are renamed to have the same code
    and description

    The "grouping" converts events like "pitch thyristor 1 fault", "pitch
    thyristor 2 fault" and "pitch thyristor 3 fault" with codes 274, 275 and 276
    into "pitch thyristor 1/2/3 fault (original codes 274/275/276)" and gives
    them all the code 274. This is an optional step when creating batches of
    events to avoid faults which happen along a different turbine axis being
    treated as different types of faults.

    Args
    ----
    event_data: the original events/fault data
    code_groups: the different groups of similar codes. Must be in the form:
        [[10, 11, 12], [24, 25], [56, 57, 58]] or [10, 11, 12]
    codes: all other fault codes that will be included in the returned
        fault data dataframe. The codes in code_groups can be repeated here,
        it won't make a difference.

    Returns
    -------
    fault_data: A subset of event_data, including only the codes in
        code_groups and codes, with the codes in code_groups all grouped
        together as one.
    '''
    # warnings.warn(
    #     'fault_codes and code_groups have switched around. The behaviour of '
    #     'this function has changed significantly!! As have all the functions in'
    #     ' this library.')

    # if there are no code_groups passed, then just do the following:
    if len(code_groups) == 0:
        # print('no groups passed')
        fault_data = event_data[event_data.code.isin(fault_codes)].copy()
        return fault_data

    # if code_groups is just one list, i.e. [10, 11, 12], then change it to
    # [[10, 11, 12]]
    if type(code_groups[0]) is int:
        code_groups = [code_groups]

    # try:
    #     if type(code_groups[0]) is int:
    #         code_groups = [code_groups]
    # except (IndexError, KeyError):
    #     raise ValueError(
    #         'Make sure code_groups is a list or list of lists of codes which '
    #         'should be grouped together. If there aren\'t any, then consider '
    #         'that this function isn\'t needed')

    # get all the codes from codes and code_groups and flatten
    all_codes = list(chain.from_iterable(code_groups))
    all_codes = list(chain.from_iterable([all_codes, fault_codes]))

    fault_data = event_data[event_data.code.isin(all_codes)].copy()

    for code_group in code_groups:

        # find the entries in event_data that have a code in code_group
        # (dupes), and drop any duplicated entries.
        dupes = event_data.loc[event_data.code.isin(code_group),
                               ['code', 'description']].drop_duplicates()

        # get a list of strings of words from the first description and from
        # the other desctiptions
        first_desc = dupes.description.unique()[0]
        other_desc = ' '.join(dupes.description.unique()[1:])
        first_desc_words = first_desc.split()
        other_desc_words = other_desc.split()

        # find the symmetric difference between the descriptions (i.e. the axis
        # numbers)
        nums = sorted(list(set(first_desc_words).symmetric_difference(
            set(other_desc_words))))
        nums = '/'.join(nums)

        # find the word (i.e. the axis number) that is in the first description
        # but not the others, and get its index location
        first_num = list(set(first_desc_words).difference(
            set(other_desc_words)))[0]
        index = first_desc_words.index(first_num)

        # put the numbers in at the index of the old one
        new_desc_words = first_desc_words[:]
        new_desc_words[index] = nums

        # put it all together in a new string
        new_desc = '{} (original codes {})'.format(
            ' '.join(new_desc_words), '/'.join([str(i) for i in code_group]))

        fault_data.loc[fault_data.code.isin(code_group), 'code'] = \
            dupes.code.min()
        fault_data.loc[fault_data.code.isin(code_group),
                       'description'] = new_desc

    return fault_data


def get_batches(event_data, fault_data, t_sep_lim='1 hour', df=False):
    """Get the indices of distinct batches of events as they appear in the
    event_data.

    Each batch is a group of fault events. A batch always begins with a fault
    event from one of the codes in fault_data, and ends with status code 6,
    which signifies the turbine returning to normal operation.

    Args
    ----
    event_data: pd.DataFrame
        The full set of events data
    fault_data: pd.DataFrame
        A subset of the events data which contains only the faults to be looked
        at, where certain faults which are similar are grouped together as one
        (mainly pitch faults on different turbine axes). Can be obtained from
        get_fault_data() function in this module
    t_sep_lim: str (must be compatible with pd.Timedelta), default='1 hour'
        If a batch ends, and another batch begins less than t_sep_lim
        afterwards, then the two batches are treated as one, i.e. it accounts
        for the turbine coming back online and immediately faulting again, and
        treats them as the one event
    df: bool, default=True
        Whether or not to return a dataframe with info on each batch

    Returns
    -------
    batch_inds: a nested dictionary of the form:
        {turbine_num: {batch1: {'all_events': Int64Index([10, 11, 12, 13, 14]),
                                'fault_events': Int64Index([11, 12]),
                                'prev_hr': Int64Index([4, 5, 6, 7, 8, 9])},
                        batch2: {....}
                       }
        }
        As seen, batch_inds contains 3 Pandas Int64Index objects associated
        with each batch for every turbine.

        'all_events' is an index of all events which occurred during the
        stoppage.

        'fault_events' refers to only the faults in 'fault_data' which occurred.

        'prev_hr' refers to all events which occurred in the hour leading up to
        the stoppage.
    batch_df: pd.DataFrame (optional)
        DataFrame with the following headings:
        turbine_num: turbine number of the batch
        fault_start_codes: the fault codes present at the first timestamp in the
            batch
        all_start_codes: all event start codes present at the first timestamp
            in the batch
        start_time: start of first event in the batch
        fault_end_time: time_on of the last fault event in the batch
        down_end_time: the time_on of the last event in the batch, i.e. the last
            code 6 event in hte batch
        fault_dur: duration from start of first fault event to start of final
            fault event in the batch
        down_dur: duration of total downtime in the batch, i.e. from start of
            first fault event to start of last code 6 event
        fault_inds: indices in the events data of faults that occurred
        all_inds: indices in the events data of all events that occurred during
            the batchs

    """
    batch_inds = {}
    for t in event_data.turbine_num.unique():
        # has to be -1 as we are adding 1 before batch creation (see below)
        i = -1
        batch_inds[t] = {}
        end_time = pd.Timestamp('Dec 1970')
        fd_t = fault_data[fault_data.turbine_num == t]
        # loop through the fault events
        for f in fd_t.itertuples():
            # if the next fault event is after the largest time_on from events
            # in the previous batch, then create a new batch! If not, then loop
            # through the events until we get to one that starts after the prev.
            # batch ends, i.e. after the previous code 6
            if f.time_on > end_time:
                prev_end_time = end_time

                end_time, all_events, fault_events, prev_hr = _get_batch_info(
                    f, event_data, fd_t, t)

                if f.time_on > prev_end_time + pd.Timedelta(t_sep_lim):
                    # if it's a certain amount of time more since the last one
                    # ended, then we move on to the next i
                    i += 1
                    batch_inds[t][i] = {'fault_events': fault_events,
                                        'all_events': all_events,
                                        'prev_hr': prev_hr}
                else:
                    batch_inds[t][i]['fault_events'].append(fault_events)
                    batch_inds[t][i]['all_events'].append(all_events)
                    batch_inds[t][i]['prev_hr'].append(prev_hr)

    if not df:
        return batch_inds
    else:
        batch_df = _get_batch_df(batch_inds, event_data, fault_data)

        return batch_inds, batch_df


def _get_batch_info(f, event_data, fd_t, t):
    # the new end_time is the next earliest code 6
    end_time = event_data[(event_data.time_on >= f.time_on) &
                          (event_data.code == 6) &
                          (event_data.turbine_num == t)
                          ].time_on.min()

    # end_time is a NaT if there is no other code 6 in the data,
    # in which case the rest of the data is in the same fault batch
    if pd.isnull(end_time):
        end_time = event_data[
            event_data.turbine_num == t].time_on.max()

    # add the all_events and fault_events which occured between the
    # event at the start of this batch and end_time
    all_events = event_data[(event_data.time_on >= f.time_on) &
                            (event_data.time_on <= end_time) &
                            (event_data.turbine_num == t)].index

    fault_events = fd_t[(fd_t.time_on >= f.time_on) &
                        (fd_t.time_on <= end_time)].index

    prev_hr = event_data[(
        event_data.time_on >= (f.time_on - pd.Timedelta(1, 'h'))) &
        (event_data.time_on < f.time_on) &
        (event_data.turbine_num == t)].index

    return end_time, all_events, fault_events, prev_hr


def _get_batch_df(batch_inds, event_data, fault_data):
    data = []

    for t in batch_inds:
        for inds in batch_inds[t].values():
            fault_inds = inds['fault_events']
            all_inds = inds['all_events']

            rel_events = fault_data.loc[fault_inds]
            all_events = event_data.loc[all_inds]

            start_time = rel_events.time_on.min()
            fault_end_time = rel_events.time_on.max()
            down_end_time = all_events.time_on.max()

            fault_start_codes = tuple(sorted(rel_events.loc[
                rel_events.time_on == start_time, 'code'].unique()))
            all_start_codes = tuple(sorted(all_events.loc[
                all_events.time_on == start_time, 'code'].unique()))

            fault_dur = fault_end_time - start_time
            down_dur = down_end_time - start_time

            data.append(
                [t, fault_start_codes, all_start_codes, start_time,
                 fault_end_time, down_end_time, fault_dur, down_dur, fault_inds,
                 all_inds])

    columns = [
        'turbine_num', 'fault_start_codes', 'all_start_codes', 'start_time',
        'fault_end_time', 'down_end_time', 'fault_dur', 'down_dur',
        'fault_inds', 'all_inds']

    batch_df = pd.DataFrame(
        data, columns=columns).dropna().reset_index(drop=True)

    return batch_df


def count_batches(scada_data, batch_inds):
    """Count the total number of batches (i.e. stoppages) in the batch groups

    Args
    ----
    scada_data: Pandas.DataFrame
        The full set of scada data
    batch_inds: nested dictionary
        The dictionary of indices of faults which occurred during the
        stoppage. See create_batch_inds() function in this module for details.

    Returns
    -------
    c: int
        Total number of batches in the batch_inds

    """
    c = 0
    for t in scada_data.turbine_num.unique():
        for i in batch_inds[t].keys():
            c += 1

    return c
