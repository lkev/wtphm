import pandas as pd
import numpy as np
from itertools import chain
# import warnings
from . import batch_clustering


class Batches:
    """
    Class for getting and manipulating batches of fault events in wind turbine
    data.

    Each batch consists of a number of fault events that occurred during a
    fault-related shutdown.

    Args
    ----
    event_data: pandas.DataFrame
        The original events/fault data. This should include the following
        headings (Note these are only the headings required for data
        manipulation performed in this class; other headings may be required for
        other analyses):
        code: There are a set list of events which can occur on the turbine.
            Each one of these has an event code.
        description: Each event code also has an associated description
        time_on: The start time of the event
    fault_codes: numpy.ndarray
        All event codes that will be treated as fault events for the batches
    code_groups: list-like, optional (default=None)
        Some events with similar codes/descriptions, e.g. identical pitch faults
        that happen along different turbine axes, may be given the same code and
        description so they are treated as the same event code during analysis.
        Must be in the form:[[10, 11, 12], [24, 25], [56, 57, 58]] or
        [10, 11, 12]
    ok_code: int (default=6)
        A code which signifies the turbine returning to normal operation after
        being shut down or curtailed due to a fault or otherwise

    Attr
    ----
    fault_data: pandas.DataFrame
        The subset of the events data with codes in `fault_codes`
    grouped_event_data: pandas.DataFrame
        The `event_data`, but with codes and descriptions from `code_groups`
        changed so that similar ones are identical
    grouped_fault_codes: pandas.DataFrame
        The `fault_codes`, but with the similar codes in each group treated as
        identical
    """

    def __init__(self, event_data, fault_codes, code_groups=None, ok_code=6):
        self.event_data = event_data.sort_values(by='time_on')
        self.fault_codes = fault_codes
        self.code_groups = code_groups
        self.ok_code = ok_code
        self.fault_data = event_data[
            event_data.code.isin(fault_codes)].sort_values(by='time_on')

        if self.code_groups is not None:
            self._get_grouped_event_data()

    def _get_grouped_event_data(self):
        """Groups together similar event codes as the same code.

        This returns the events dataframe but with some fault events which have
        different but similar codes and descriptions grouped together and
        relabelled to have the same code and description.

        Example:
        The "grouping" gives the events "pitch thyristor 1 fault" with code 101,
        "pitch thyristor 2 fault" with code 102 and "pitch thyristor 3 fault"
        with code 103 all the same event description and code, i.e. they all
        become "pitch thyristor 1/2/3 fault (original codes 101/102/103)" with
        code 101. This is an optional step when creating batches of events to
        avoid similar faults which happen along different turbine axes being
        treated as different types of faults.

        Returns
        -------
        fault_data: pandas.DataFrame
            A subset of event_data, including only the codes in
            code_groups and codes, with the codes in code_groups all grouped
            together as one.
        """

        # if code_groups is just one list, i.e. [10, 11, 12], then change it to
        # [[10, 11, 12]]
        if type(self.code_groups[0]) is int:
            self.code_groups = [self.code_groups]

        if type(self.fault_codes) != np.ndarray:
            raise TypeError('fault_codes must be a numpy.ndarray')

        grouped_event_data = self.event_data.copy()
        grouped_fault_codes = self.fault_codes.copy()

        all_codes = list(chain.from_iterable(self.code_groups))
        all_codes = np.array(
            list((chain.from_iterable([all_codes, self.fault_codes]))))

        for code_group in self.code_groups:

            # find the entries in event_data that have a code in code_group
            # (dupes), and drop any duplicated entries.
            dupes = grouped_event_data.loc[
                grouped_event_data.code.isin(code_group),
                ['code', 'description']]
            dupes.description = dupes.description.str.lower()
            dupes = dupes.drop_duplicates()

            if len(dupes) == 0:
                continue

            # get a list of strings of words from the first description and from
            # the other descriptions
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

        grouped_fault_data = grouped_event_data[
            grouped_event_data.code.isin(grouped_fault_codes)]

        self.grouped_event_data = grouped_event_data.sort_values(by='time_on')
        self.grouped_fault_data = grouped_fault_data.sort_values(by='time_on')
        self.grouped_fault_codes = grouped_fault_codes

        return grouped_event_data, grouped_fault_data, grouped_fault_codes

    def get_batch_data(self, t_sep_lim='12 hour', groups=True):
        """Get the distinct batches of events as they appear in the event_data.

        Each batch is a group of fault events. A batch always begins with a
        fault event from one of the codes in fault_data, and ends with the code
        ok_code, which signifies the turbine returning to normal operation.

        Args
        ----
        t_sep_lim: str (must be compatible with pd.Timedelta), default='1 hour'
            If a batch ends, and a second batch begins less than t_sep_lim
            afterwards, then the two batches are treated as one, it treats the
            the turbine coming back online and immediately faulting again as one
            continuous batch. This effect is stacked so that if a third fault
            event happens less than an hour after the second, all three are
            treated as the same continuous batch.
        groups: bool, default=True
            Whether or not the returned dataframe.

        Returns
        -------
        batch_data: pd.DataFrame
            DataFrame with the following headings:
            turbine_num: turbine number of the batch
            fault_start_codes: the fault codes present at the first timestamp in
                the batch
            all_start_codes: all event start codes present at the first
                timestamp in the batch
            start_time: start of first event in the batch
            fault_end_time: time_on of the last fault event in the batch
            down_end_time: the time_on of the last event in the batch, i.e. the
                last ok_code event in hte batch
            fault_dur: duration from start of first fault event to start of
                final fault event in the batch
            down_dur: duration of total downtime in the batch, i.e. from start
                of first fault event to start of last ok_code event
            fault_event_ids: indices in the events data of faults that occurred
            all_event_ids: indices in the events data of all events (fault or
                otherwise) that occurred during the batch
        """

        # if groups=False, then the only thing this means is that the
        # codes in fault_start_codes and all_start_codes are grouped, but the
        # ids remain unchanged
        if (self.code_groups) and (groups is True):
            event_data = self.grouped_event_data
            fault_data = self.grouped_fault_data
        else:
            event_data = self.event_data
            fault_data = self.fault_data

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
                # events in the previous batch, then create a new batch! If not,
                # then loop through the events until we get to one that starts
                # after the prev. batch ends, i.e. after the previous ok_code
                # event
                if f.time_on > end_time:
                    prev_end_time = end_time

                    end_time, all_events, fault_events, prev_hr = \
                        self._get_batch_info(
                            f, event_data, fd_t, t, self.ok_code)

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

        self.batch_data = self._get_batch_df(batch_ids, event_data,
                                             fault_data)

        return self.batch_data

    @staticmethod
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

    @staticmethod
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

                fault_start_codes = tuple(sorted(rel_events.loc[
                    rel_events.time_on == start_time, 'code'].unique()))
                all_start_codes = tuple(sorted(all_events.loc[
                    all_events.time_on == start_time, 'code'].unique()))

                fault_dur = fault_end_time - start_time
                down_dur = down_end_time - start_time

                data.append(
                    [t, fault_start_codes, all_start_codes, start_time,
                     fault_end_time, down_end_time, fault_dur, down_dur,
                     fault_event_ids, all_event_ids])

        columns = [
            'turbine_num', 'fault_start_codes', 'all_start_codes', 'start_time',
            'fault_end_time', 'down_end_time', 'fault_dur', 'down_dur',
            'fault_event_ids', 'all_event_ids']

        batch_data = pd.DataFrame(
            data, columns=columns).dropna().reset_index(drop=True)

        return batch_data

    def get_batch_features(self, method, batch_data=None, lo=1, hi=10, num=1,
                           event_type='fault_events', groups=True):
        """Extract features from batches of events which appear during
        stoppages, to be used for clustering.

        Only features from batches that comply with certain constraints are
        included. These constraints are chosen depending on which feature
        extraction method is used. Details of the feature extraction methods can
        be found in [1].

        **Note:** For each "batch" of alarms, there are up to `num_codes` unique
        alarm codes. Each alarm has an associated start time, `time_on`.

        This method is just a wrapper for batch_clustering.get_batch_features().

        Args
        ----
        method: string
            One of 'basic', 't_on', 'time'.
            basic:
                * Only considers batches with between `lo` and `hi` individual
                  alarms.
                * Array of zeros is filled with 'num' corresponding to order of
                  alarms' appearance.
                * Does not take into account whether alarms occurred
                  simultaneously.
                * Resultant vector of length 'num_codes' * 'hi'
            t_on:
                * Only consider batches with between 'lo' and 'hi' individual
                  'time_on's.
                * For each `time_on` in each batch, an array of zeros is filled
                  with ones in places corresponding to an alarm that has fired
                  at that time.
                * Results in a pattern array of length (`num_codes` * `hi`)
                  which shows the sequential order of the alarms which have been
                  fired.
            time:
                * Same as above, but extra features are added showing the amount
                  of time between each "time_on"
        batch_data: pd.DataFrame, optional (default=None)
            If `None`, uses the default `self.batch_data`. Otherwise, a custome
            `batch_data` may be passed, for example if only batches of a certain
            duration wish to be included
        lo: integer, default=1
            For method='basic', only batches with a minimum of 'lo' alarms will
            be included in the returned feature set.
            for method='t_on' or 'time', it's the minimum number of 'time_on's.
        hi: integer, default=10
            For method='basic', only batches with a maximum of 'hi' alarms will
            be included in the returned feature set.
            for method='t_on' or 'time', it's the maximum number of 'time_on's.
        num: integer, float, default=1
            The number to be placed in the feature vector to indicate the
            presence of a particular alarm
        event_type: string, default='fault_events'
            The members of batch_data to include for building the feature set.
            Should normally be 'fault_events' or 'all_events'
        groups: boolean, default=True
            Whether to include the grouped or un-grouped fault/event data

        Returns
        -------
        feature_array: numpy.ndarray
            An array of feature arrays corresponding to each batch that has has
            met the 'hi' and 'lo' criteria
        assoc_batch: unmpy.ndarray
            An array of 2-length index arrays. It is the same length as
            feature_array, and each entry points to the corresponding
            feature_array's index in batch_data, which in turn contains the
            index of the feature_arrays associated events in the original
            events_data or fault_data.
        """

        if (self.code_groups) and (groups is True):
            event_data = self.grouped_event_data
            fault_codes = self.grouped_fault_codes
        else:
            event_data = self.event_data
            fault_codes = self.fault_codes

        if batch_data is None:
            batch_data = self.batch_data

        feature_array, assoc_batch = batch_clustering.get_batch_features(
            event_data, fault_codes, batch_data, method, lo, hi, num,
            event_type)

        return feature_array, assoc_batch
