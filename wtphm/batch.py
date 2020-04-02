import pandas as pd
import numpy as np
from itertools import chain
from collections import Counter


def get_grouped_event_data(event_data, code_groups, fault_codes):
    """
    Groups together similar event codes as the same code.

    This returns the events dataframe but with some fault events which have
    different but similar codes and descriptions grouped together and
    relabelled to have the same code and description.

    Example:
    The "grouping" gives the events "pitch thyristor 1 fault" with code 501
    , "pitch thyristor 2 fault" with code 502 and "pitch thyristor 3 fault"
    with code 503 all the same event description and code, i.e. they all
    become "pitch thyristor 1/2/3 fault (original codes 501/502/503)" with
    code 501. This is an optional step before creating batches of events to
    avoid similar faults which happen along different turbine axes being
    treated as different types of faults.

    Args
    ----
    event_data : pandas.DataFrame
        The original events/fault data.

    fault_codes : numpy.ndarray
        All event codes that will be treated as fault events for the batches
    code_groups : list-like, optional, default=None
        Some events with similar codes/descriptions, e.g. identical pitch
        faults that happen along different turbine axes, may be given the same
        code and description so they are treated as the same event code during
        analysis.
        Must be in the form: '[[10, 11, 12], [24, 25], [56, 57, 58]]' or
        '[10, 11, 12]'. If this is passed, then the attributes
        ``grouped_fault_codes`` and ``grouped_event_data`` become available.

    Returns
    -------
    grouped_event_data : pandas.DataFrame
        The ``event_data``, but with codes and descriptions from
        ``code_groups`` changed so that similar ones are identical
    grouped_fault_codes : pandas.DataFrame
        The ``fault_codes``, but with the similar codes in each group
        treated as identical

    """

    # if code_groups is just one list, i.e. [10, 11, 12], then change it to
    # [[10, 11, 12]]
    if type(code_groups[0]) is int:
        code_groups = [code_groups]

    if type(fault_codes) != np.ndarray:
        raise TypeError('fault_codes must be a numpy.ndarray')

    grouped_event_data = event_data.copy()
    grouped_fault_codes = fault_codes.copy()

    all_codes = list(chain.from_iterable(code_groups))
    all_codes = np.array(
        list((chain.from_iterable([all_codes, fault_codes]))))

    for code_group in code_groups:

        # find the entries in event_data that have a code in code_group
        # (dupes), and drop any duplicated entries.
        dupes = grouped_event_data.loc[
            grouped_event_data.code.isin(code_group),
            ['code', 'description']]
        dupes.description = dupes.description.str.lower()
        dupes = dupes.drop_duplicates()

        if len(dupes) == 0:
            continue

        # get a list of strings of words from the first description and
        # from the other descriptions
        first_desc = dupes.description.unique()[0]
        other_desc = ' '.join(dupes.description.unique()[1:])
        first_desc_words = first_desc.split()
        other_desc_words = other_desc.split()

        # find the symmetric difference between the descriptions (i.e. the
        # axis numbers)
        nums = sorted(list(set(first_desc_words).symmetric_difference(
            set(other_desc_words))))
        nums = '/'.join(nums)

        # find the word (i.e. the axis number) that is in the first
        # description but not the others, and get its index location
        first_num = list(set(first_desc_words).difference(
            set(other_desc_words)))[0]
        index = first_desc_words.index(first_num)

        # put the numbers in at the index of the old one
        new_desc_words = first_desc_words[:]
        new_desc_words[index] = nums

        # put it all together in a new string
        new_desc = '{} (original codes {})'.format(' '.join(
            new_desc_words), '/'.join([str(i) for i in code_group]))

        grouped_event_data.loc[grouped_event_data.code.isin(code_group),
                               'code'] = dupes.code.min()
        grouped_event_data.loc[grouped_event_data.code.isin(code_group),
                               'description'] = new_desc

        grouped_fault_codes[
            np.isin(grouped_fault_codes, code_group)] = dupes.code.min()

    grouped_event_data = grouped_event_data.sort_values(by='time_on')
    grouped_fault_codes = grouped_fault_codes

    return grouped_event_data, grouped_fault_codes


def get_batch_data(event_data, fault_codes, ok_code, t_sep_lim='12 hour'):
    """
    Get the distinct batches of events as they appear in the
    ``event_data``.

    Each batch is a group of fault events that occurred during a fault-related
    shutdown. A batch always begins with a fault event from one of the codes in
    ``fault_codes``, and ends with the code ``ok_code``, which signifies the
    turbine returning to normal operation.

    Args
    ----
    event_data : pandas.DataFrame
        The original events/fault data. This should include the following
        headings (Note these are only the headings required for data
        manipulation performed in this class; other headings may be required
        for other analyses):

        * ``code``: There are a set list of events which can occur on the
          turbine. Each one of these has an event code.
        * ``description``: Each event code also has an associated description
        * ``time_on``: The start time of the event

    fault_codes : numpy.ndarray
        All event codes that will be treated as fault events for the batches
    ok_code : int
        A code which signifies the turbine returning to normal operation after
        being shut down or curtailed due to a fault or otherwise
    t_sep_lim : str, default='1 hour', must be compatible with ``pd.Timedelta``
        If a batch ends, and a second batch begins less than ``t_sep_lim``
        afterwards, then the two batches are treated as one. It treats the
        the turbine coming back online and immediately faulting again as
        one continuous batch.
        This effect is stacked so that if a third fault
        event happens less than ``t_sep_lim`` after the second, all three
        are treated as the same continuous batch.

    Returns
    -------
    batch_data : pd.DataFrame
        DataFrame with the following headings:

        * ``turbine_num``: turbine number of the batch
        * ``root_codes``: the fault codes present at the first
          timestamp in the batch
        * ``all_start_codes``: all event start codes present at the first
          timestamp in the batch
        * ``start_time``: start of first event in the batch
        * ``fault_end_time``: ``time_on`` of the last fault event in the
          batch
        * ``down_end_time``: the ``time_on`` of the last event in the
           batch, i.e. the last ``ok_code`` event in the batch
        * ``fault_dur``: duration from start of first fault event to start
          of final fault event in the batch
        * ``down_dur``: duration of total downtime in the batch, i.e. from
          start of first fault event to start of last ``ok_code`` event
        * ``fault_event_ids``: indices in the events data of faults that
          occurred
        * ``all_event_ids``: indices in the events data of all events
          (fault or otherwise) that occurred during the batch

    """

    fault_data = event_data[
        event_data.code.isin(fault_codes)].sort_values(by='time_on')

    batch_ids = {}
    for t in event_data.turbine_num.unique():
        # has to be -1 as we are adding 1 before batch creation (see below)
        i = -1
        batch_ids[t] = {}
        end_time = pd.Timestamp('Dec 1970')
        fd_t = fault_data[fault_data.turbine_num == t]
        # loop through the fault events
        for f in fd_t.itertuples():
            # if the next fault event is after the largest time_on from
            # events in the previous batch, then create a new batch! If
            # not, then loop through the events until we get to one that
            # starts after the prev. batch ends, i.e. after the previous
            # ok_code event
            if f.time_on > end_time:
                prev_end_time = end_time

                end_time, all_events, fault_events, prev_hr = \
                    _get_batch_info(
                        f, event_data, fd_t, t, ok_code)

                if f.time_on > prev_end_time + pd.Timedelta(t_sep_lim):
                    # if it's a certain amount of time more since the last
                    # one ended, then we move on to the next i
                    i += 1
                    batch_ids[t][i] = {'fault_events': fault_events,
                                       'all_events': all_events,
                                       'prev_hr': prev_hr}
                else:
                    batch_ids[t][i]['fault_events'] = batch_ids[t][i][
                        'fault_events'].append(fault_events)
                    batch_ids[t][i]['all_events'] = batch_ids[t][i][
                        'all_events'].append(all_events)
                    batch_ids[t][i]['prev_hr'] = batch_ids[t][i][
                        'prev_hr'].append(prev_hr)

    batch_data = _get_batch_df(batch_ids, event_data, fault_data)

    return batch_data


def _get_batch_info(f, event_data, fd_t, t, ok_code):
    """get the end time and event ids in each batch"""
    # the new end_time is the next earliest ok_code event
    end_time = event_data[(event_data.time_on >= f.time_on) &
                          (event_data.code.isin([ok_code])) &
                          (event_data.turbine_num == t)
                          ].time_on.min()

    # end_time is a NaT if there is no other ok_code event in the data, in
    # which case the rest of the data is in the same fault batch. end_time
    # is changed to reflect this
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


def _get_batch_df(batch_ids, event_data, fault_data):
    """get the dataframe of the batches"""
    data = []

    for t in batch_ids:
        for inds in batch_ids[t].values():
            fault_event_ids = inds['fault_events']
            all_event_ids = inds['all_events']

            rel_events = fault_data.loc[fault_event_ids]
            all_events = event_data.loc[all_event_ids]

            start_time = rel_events.time_on.min()
            fault_end_time = rel_events.time_on.max()
            down_end_time = all_events.time_on.max()

            root_codes = tuple(sorted(rel_events.loc[
                rel_events.time_on == start_time, 'code'].unique()))
            all_start_codes = tuple(sorted(all_events.loc[
                all_events.time_on == start_time, 'code'].unique()))

            fault_dur = fault_end_time - start_time
            down_dur = down_end_time - start_time

            data.append(
                [t, root_codes, all_start_codes, start_time,
                 fault_end_time, down_end_time, fault_dur, down_dur,
                 fault_event_ids, all_event_ids])

    columns = [
        'turbine_num', 'root_codes', 'all_start_codes',
        'start_time', 'fault_end_time', 'down_end_time', 'fault_dur',
        'down_dur', 'fault_event_ids', 'all_event_ids']

    batch_data = pd.DataFrame(
        data, columns=columns).dropna().reset_index(drop=True)

    return batch_data


def get_batch_stop_cats(
        batch_data, events_data, scada_data, grid_col, maint_col, rep_col,
        grid_cval=0, maint_cval=0, rep_cval=0):
    """
    Labels the batches with an assumed stop category, based on the stop
    categories of the root event(s) which triggered them, i.e. the one or more
    events occurring simultaneously which caused the turbine to stop (items
    lower down supersede those higher up):

    * If **all** root events in the batch are "normal" events, then the
        batch is labelled normal
    * Otherwise, label as the most common stop cat in the initial events
    * If a single sensor category event is present, label sensor
    * If a single grid category event is present, label grid. Also label grid
        if the grid counter was active in the scada data. This is a timer
        indicating how long the turbine was down due to grid issues, used for
        calculating contract availability
    * If the maintenance counter was active in the scada data, label maint
    * There is an additional column labelled "repair". If the repair counter
        was active, the turbine was brought down for repairs, and this is given
        the value "TRUE" for these times.

    Args
    ----
    batch_data : pd.Dataframe
        The batch data
    events_data : pd.Dataframe
        The events data
    scada_data : pd.Dataframe
        The scada data.
    grid_col, maint_col, rep_col : string
        The columns of ``scada_data`` which contain availabililty counters
        for grid issues, turbine maintenance and repairs, resepctively
    grid_cval, maint_cval : int
        The minimum total sum of the grid, maintenance and repair counters
        throughout the duration of a batch for it to be marked as grid, repair
        or maintenance

    Returns
    -------
    batch_data_sc : pd.DataFrame
        DataFrame with index matching ``batch_data``, with the following
        headings:

        * batch_cat: The stop categories of each batch
        * repairs: The repair status of each batch
    """
    batch_data_sc = batch_data.copy()
    root_cats = get_root_cats(batch_data_sc, events_data)
    cat_counts = get_root_cat_counts(root_cats)

    # get stop categories
    most_common_cats = get_most_common_cats(cat_counts)
    grid_ids = get_cat_present_ids(root_cats, 'grid')
    grid_counter_ids = get_counter_active_ids(
        batch_data_sc, scada_data, grid_col, counter_value=grid_cval)
    grid_ids = grid_ids.append(grid_counter_ids).drop_duplicates()
    sensor_ids = get_cat_present_ids(root_cats, 'sensor')
    maint_ids = get_counter_active_ids(
        batch_data_sc, scada_data, maint_col, counter_value=maint_cval)
    all_normal_ids = get_cat_all_ids(root_cats, 'ok')

    batch_data_sc['batch_cat'] = most_common_cats
    batch_data_sc.loc[all_normal_ids, 'batch_cat'] = 'ok'
    batch_data_sc.loc[sensor_ids, 'batch_cat'] = 'sensor'
    batch_data_sc.loc[grid_ids, 'batch_cat'] = 'grid'
    batch_data_sc.loc[maint_ids, 'batch_cat'] = 'maintenance'

    # find when repaird were carried out
    repair_ids = get_counter_active_ids(
        batch_data_sc, scada_data, rep_col, counter_value=rep_cval)
    batch_data_sc['repair'] = False
    batch_data_sc.loc[repair_ids, 'repair'] = True

    return batch_data_sc


def get_root_cats(batch_data, events_data):
    """
    Gets the categories for the root alarms in the ``batch_data``
    """
    def gr_cur(cur_root, events_data):
        r_cats = tuple()
        for rc in cur_root:
            cat = events_data.loc[events_data.code == rc, 'stop_cat']\
                .unique()[0]
            r_cats += tuple([cat])
        return r_cats

    return batch_data.root_codes.apply(
        gr_cur, **{'events_data': events_data})


def get_root_cat_counts(root_cats):
    """
    Gets a count of the categories of the root alarms in a batch.

    Args
    ----
    root_cats : pd.Series
        Series of strings, where each string is the categories of each of the
        root alarms in a batch, separated by commas.

    Returns
    -------
    cat_counts : pd.Series
        Series of dictionaries, where the keys of each dictionary are the
        root alarms for a certain batch, and the values are the counts.
    """
    cat_counts = root_cats.apply(
        lambda x: Counter(fault for fault in x))
    return cat_counts


def get_most_common_cats(cat_counts):
    """
    Gets the most common root fault category from a dictionary of root alarms

    Args
    ----
    cat_counts: pd.Series
        Each entry in the series is a dictionary containing a count of
        how many times each root alarm occurrs in a batch. Output of
        :func:`get_root_cat_counts`.

    Returns
    -------
    most_common_cats : pd.Series
        Each entry in the series is a string containing the most commonly
        occurring root fault in ``cat_counts``. In the case of a draw,
        then both are added, e.g. 'test, grid'
    """
    def gmc_cur(cur_cat):
        cur_val = -1
        cur_f = ''
        for k, v in cur_cat.items():
            if v == cur_val:
                cur_f += f', {k}'
            if v > cur_val:
                cur_f = k
            cur_val = v
        return cur_f
    most_common_cats = cat_counts.apply(gmc_cur)
    return most_common_cats


def get_cat_all_ids(root_cats, cat):
    """
    Get an index of batches where there is only a single certain category
    present in the categories of the root alarms.

    Args
    ----
    root_cats : pd.Series
        Series of strings, where each string is the categories of each of the
        root alarms in a batch, separated by commas.
    cat : string
        The category to check the presence of

    Returns
    -------
    cat_present_idx : pd.Index
        The index of batch entries where ``cat`` was the only category present
        in the ``root_cats``
    """
    cat_ids = []
    for i, c in root_cats.iteritems():
        if set(c) == set([cat]):
            cat_ids.append(i)
    return pd.Index(cat_ids)


def get_cat_present_ids(root_cats, cat):
    """
    Get an index of batches where a certain category is present in the
    categories of the root alarms.

    Args
    ----
    root_cats : pd.Series
        Series of strings, where each string is the categories of each of the
        root alarms in a batch, separated by commas.
    cat : string
        The category to check the presence of

    Returns
    -------
    cat_present_idx : pd.Index
        The index of batch entries where ``cat`` was present in the
        ``root_cats``
    """
    cat_ids = []
    for i, c in root_cats.iteritems():
        if cat in c:
            cat_ids.append(i)
    cat_present_idx = pd.Index(cat_ids)
    return cat_present_idx


def get_counter_active_ids(batch_data, scada_data, counter_col,
                           counter_value=0):
    """
    Get an index of batches during which a certain scada counter was active

    In 10-minute SCADA data there are often counters for when the turbine was
    in various different states, for calculating contractual availability.
    This function finds the named ``counter_col`` in ``scada_data``, and
    identifies any sample periods where this value was above ``counter_value``.

    If any of these sample periods fall within a certain batch, then this
    function returns those batch ids.

    Args
    ----
    batch_data : pd.DataFrame
        The batches of events
    scada_data : pd.DataFrame
        The 10-minute SCADA data
    counter_col : string
        The column in the SCADA data with a counter
    counter_value : int
        Any SCADA entries with a counter above this value will have their index
        returned

    Returns
    -------
    counter_active_index : pd.Index
        The id's of ``counter_col`` columns in ``scada_data`` which have a val
        above ``counter_value``.
    """
    batch_ids = []
    for b in batch_data.itertuples():
        start = b.start_time
        end = (b.down_end_time + pd.Timedelta('5 minutes')).round('10T')
        batch_scada = scada_data[
            (scada_data.turbine_num == b.turbine_num) &
            (scada_data.time >= start) & (scada_data.time <= end)]
        if batch_scada[counter_col].sum() > counter_value:
            batch_ids.append(b.Index)
    counter_active_idx = pd.Index(batch_ids)
    return counter_active_idx
