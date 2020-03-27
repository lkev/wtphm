"""This module is for working with events data from wind turbines.
It looks at all eventes generated and sees if there are some events which
trigger others. Event A triggers Event B if:
t_s_A <= t_s_B and t_e_A >= t_s_B

So we can find the probability that any given A event (known as a parent event)
has triggered any B events, and the probability that any given B event (known
as a child event) has been triggered by any A events.

"""

import pandas as pd
import operator
import itertools
# from numpy import float64 as npfloat64


def get_trig_summary(events, codes, tsa_op1='ge', tsa_op2='le',
                     t_hi=0.9, t_lo=0.1):
    """Gets probabilities that pairs of events will trigger one another, and
    the derived relationship between these pairs

    This function takes a list of event codes. It finds all combinations of
    pairs of codes from this and splits them into "A" and "B" codes. It then
    counts the number of events with code A which have triggered one or more
    events with code B and vice-versa. It then computes a probability that if
    an A event occurs, it will trigger a B event, and vice-versa. From there,
    it deduces the relationship between pairs of events, as derived from [1].

    Event A is triggered by Event B if:

    T_s_A >= T_s_B & T_s_A <= T_e_B

    where T_s_A, T_s_B and T_e_B are the start time of events A and B, and the
    end time of event B, respectively.

    Args
    ----
    events: pandas.DataFrame
        The events data from a wind turbine. Must be free of NA values.
    codes: list-like
        The event codes to look at
    tsa_op1: String, default 'ge'
        Operator to use for T_s_A >= T_s_B or T_s_A > T_s_B. Can be one of:

        'ge': <=
        'gt': <

    tsa_op2: String (default 'le')
        Operator to use for T_s_A <= T_e_B or T_s_A < T_e_B. Can be one of:
        'le': >=
        'lt': >
    t_hi: float (default 0.9)
        Threshold of % of A events which trigger B events at or above which
        relationship 3 is `True` (or % B triggering A for relationship 4, or
        % of both for relationship 1). See 'relationship' in the returned
        `trig_summary` dataframe below.
    t_low: float (default 0.1)
        Threshold of % of A events which trigger B events (or vice-versa) at or
        below which relationship 2 is `True`. See 'relationship' in the
        returned `trig_summary` dataframe below.

    Returns
    -------
    trig_summary : Pandas.DataFrame
        A matrix consisting of the following:

        * `A_code`: the event code of the "A" events
        * `A_desc`: description of the "A" events
        * `B_code`: the event code of the "B" events
        * `B_desc`: description of the "B" events
        * `A_count`: number of "A" events in the data
        * `A_trig_B_count`: number of "A" events which trigger one or more "B"
          events
        * `A_trig_B_prob`: ratio of "A" events which have triggered one or
          more
          "B" events, to the total number of "A" events
        * `B_count`: Number of "B" events in the data
        * `B_trig_A_count`: number of "B" events which trigger one or more "A"
          events
        * `B_trig_A_prob`: ratio of "B" events which have triggered one or
          more
          "A" events, to the total number of "B" events
        * `relationship`: Number 1-5 indicating the relationship events A have
          to events B:
            1. High proportion of As trigger Bs & high proportion of Bs trigger
               As. Alarm A & B usually appear together; A ~= B
            2. Low proportion of As trigger Bs & low proportion of Bs
               trigger As. A & B never or rarely appear together; A n B ~= 0
            3. High proportion of As trigger Bs & less than high proportion of
               Bs trigger As. B will usually be triggered whenever alarm A
               appears - B is a more general alarm; A e B
            4. High proportion of Bs trigger As & less than high proportion
               of As trigger Bs. A will usually be triggered whenever alarm B
               appears - A is a more general alarm; B e A
            5. None of the above. The two alarms are randomly or somewhat
               related; A n B != 0

    References
    ----------
    [1] Qiu et al. (2012). Wind turbine SCADA alarm analysis for improving
    reliability. Wind Energy, 15(8), 951–966. http://doi.org/10.1002/we.513
    """
    if events[['turbine_num', 'time_on', 'code', 'description', 'category',
               'type', 'power', 'wind_speed', 'gen_speed', 'time_off',
               'duration']].isnull().values.any():
        raise ValueError("the events data passed has NaN values in it")

    cols = ['A_code',
            'A_desc',
            'B_code',
            'B_desc',
            'A_count',
            'A_trig_B_count',
            'A_trig_B_prob',
            'B_count',
            'B_trig_A_count',
            'B_trig_A_prob',
            'relationship']

    def _fill_trig_summary(e):
        A_events = events.loc[events.code == e.A_code]
        B_events = events.loc[events.code == e.B_code]
        if len(A_events) == 0:
            A_trig_count = 0
            A_prob = 0
        else:
            A_trig_count = A_events.apply(_trig_count, other_events=B_events,
                                          axis=1).sum()
            A_prob = A_trig_count / len(A_events)
        if len(B_events) == 0:
            B_trig_count = 0
            B_prob = 0
        else:
            B_trig_count = B_events.apply(_trig_count, other_events=A_events,
                                          axis=1).sum()
            B_prob = B_trig_count / len(B_events)
        try:
            e['A_desc'] = events[
                events.code == e.A_code].description.unique()[0]
        except IndexError:
            e['A_desc'] = 'No events matching this code in dataset'
        try:
            e['B_desc'] = events[
                events.code == e.B_code].description.unique()[0]
        except IndexError:
            e['B_desc'] = 'No events matching this code in dataset'

        e['A_count'] = len(A_events)
        e['A_trig_B_count'] = A_trig_count
        e['A_trig_B_prob'] = round(A_prob, 2)
        e['B_count'] = len(B_events)
        e['B_trig_A_count'] = B_trig_count
        e['B_trig_A_prob'] = round(B_prob, 2)

        if (e.A_trig_B_prob >= t_hi) and (e.B_trig_A_prob >= t_hi):
            e['relationship'] = 1
        elif (e.A_trig_B_prob <= t_lo) and (e.B_trig_A_prob <= t_lo):
            e['relationship'] = 2
        elif (e.A_trig_B_prob >= t_hi):
            e['relationship'] = 3
        elif (e.B_trig_A_prob >= t_hi):
            e['relationship'] = 4
        else:
            e['relationship'] = 5

        return e

    def _trig_count(e, other_events):
        trig_events = other_events[
            (_comp(other_events.time_on, e.time_on, tsa_op1)) &
            (_comp(other_events.time_on, e.time_off, tsa_op2)) &
            (other_events.turbine_num == e.turbine_num)].index
        if len(trig_events) > 0:
            trig_count = 1
        else:
            trig_count = 0

        return trig_count

    combos = list(itertools.combinations(codes, 2))
    A_codes = [item[0] for item in combos]
    B_codes = [item[1] for item in combos]
    trig_summary = pd.DataFrame(columns=cols)
    trig_summary['A_code'] = A_codes
    trig_summary['B_code'] = B_codes

    trig_summary = trig_summary.apply(
        _fill_trig_summary, axis=1)

    return trig_summary


def _comp(a, b, op):
    ops = {'le': operator.le,
           'lt': operator.lt,
           'ge': operator.ge,
           'gt': operator.gt}
    try:
        op_func = ops[op]
        result = op_func(a, b)
        return result
    except KeyError:
        raise ValueError("tsa_op1 and tsa_op2 must be one of 'le', 'lt', 'ge',"
                         " or 'gt'")


def short_summary(trig_summary, codes, t=0.7):
    """Returns an even more summarised version of trig_summary, showing
    important relationships

    Args
    ----
    trig_summary: Pandas.DataFrame
        Must be the `trig_summary` obtained from :func:`.get_trig_summary`
    codes: int, list
        A single, or list of, event code(s) of interest, i.e. the events that
        trigger other events
    t: float
        The threshold for a 'significant' relationship. E.g., if t=0.7, only
        events that trigger other events with a probability >= 0.7 will be
        displayed.
    Returns
    -------
    df: Pandas.DataFrame
    A dataframe consisting of the following:

        * `parent_code`: the triggering events code
        * `child_code`: the triggered events code
        * `trig_prob`: the probability that `parent_code` events will
          trigger `child_code` events
        * `trig_count`: the count of `parent_code` events which have
          triggered `child_code` events
    """
    cols = ['parent_code', 'child_code', 'trig_prob', 'trig_count']
    df = pd.DataFrame(columns=cols)
    if type(codes) == int:
        codes = [codes]
    for c in codes:
        a_trigs = trig_summary[
            (trig_summary.A_code == c) &
            (trig_summary.A_trig_B_prob >= t)]
        b_trigs = trig_summary[
            (trig_summary.B_code == c) &
            (trig_summary.B_trig_A_prob >= t)]

        for a in a_trigs.itertuples():
            df2 = pd.DataFrame(
                [[a.A_code, a.B_code, a.A_trig_B_prob, a.A_trig_B_count]],
                columns=cols)
            df = df.append(df2, ignore_index=True)

        for b in b_trigs.itertuples():
            df2 = pd.DataFrame(
                [[b.B_code, b.A_code, b.B_trig_A_prob, b.B_trig_A_count]],
                columns=cols)
            df = df.append(df2, ignore_index=True)
    return df


def get_trig_summary_verbose(events, codes, tsa_op1='ge', tsa_op2='le'):
    """Gets probabilities that certain events will trigger others, and that
    certain events will be triggered *by* others. Can be calculated via
    a duration-based method, or straightforward count.

    This takes a list of event codes. It creates two separate sets of
    "parent" and "child" events, with all the parent events having the same
    event code and all the child events having another event code (though it
    does not necessarily have to be different). It then iterates through every
    parent event instance to see if it has triggered one or more child
    events. It counts the number of parent events which have triggered one or
    more child events for each event code. It also gives a probability that any
    new parent event will trigger a child event by finding the ratio of parent
    events which have triggered a child event to those which haven't.

    Event A is triggered by Event B if:

    T_s_A >= T_s_B & T_s_A <= T_e_B

    where T_s_A, T_s_B and T_e_B are the start time of events A and B, and the
    end time of event B, respectively.

    Args
    ----
    events : Pandas.DataFrame
        The events data from a wind turbine. Must be free of NA values.
    codes : list-like
        The event codes to look at
    tsa_op1 : String (default 'ge')
        Operator to use for T_s_A >= T_s_B or T_s_A > T_s_B. Can be one of:
        'ge': <=
        'gt': <
    tsa_op2 : String (default 'le')
        Operator to use for T_s_A <= T_e_B or T_s_A < T_e_B. Can be one of:
        'le': >=
        'lt': >

    Returns
    -------
    trig_summary : Pandas.DataFrame
        A matrix consisting of the following:

        * `parent_event`: the event code of the parent event
        * `parent_desc`: description of the parent event
        * `p_count`: total number of parent events matching the event code
        * `p_dur`: total duration of parent events matching the event code
        * `p_trig_count`: number of parent events which have triggered child
          events
        * `p_trig_dur`: duration of parent events which have triggered child
          events
        * `child_event`: the event code of the child event
        * `child_desc`: description of the child event
        * `c_count`: total number of child events matching the event code
        * `c_dur`: total duration of child events matching the event code
        * `c_trig_count`: number of child events which have been triggered by
          parent events
        * `c_trig_dur`: duration of child events which have been triggered by
          parent events

    """
    if events[['turbine_num', 'time_on', 'code', 'description', 'category',
               'type', 'power', 'wind_speed', 'gen_speed', 'time_off',
               'duration']].isnull().values.any():
        raise ValueError("the events data passed has NaN values in it")

    trig_summary = pd.DataFrame(columns=[
        'parent_event', 'parent_desc', 'p_count', 'p_dur', 'p_trig_count',
        'p_trig_dur', 'child_event', 'child_desc', 'c_count', 'c_dur',
        'c_trig_count', 'c_trig_dur', 'p_prob_count', 'p_prob_dur',
        'c_prob_count', 'c_prob_dur'])

    trig_indices = {}

    i = 0
    for child_code in codes:
        child_events = events[events.code == child_code]
        for parent_code in codes:
            parent_events = events[events.code == parent_code]

            p_trig_idxs = pd.DataFrame().index
            c_trig_idxs = pd.DataFrame().index

            for parent_event in parent_events.itertuples():
                idx = child_events[
                    (_comp(child_events.time_on, parent_event.time_on,
                           tsa_op1)) &
                    (_comp(child_events.time_on, parent_event.time_off,
                           tsa_op2)) &
                    (child_events.turbine_num == parent_event.turbine_num)
                ].index

                c_trig_idxs = c_trig_idxs.union(idx)

                if len(idx) > 0:
                    p_trig_idxs = p_trig_idxs.union([parent_event[0]])

            parent_desc = events[
                events.code == parent_code].description.unique()[0]
            p_count = len(events.loc[events.code == parent_code])
            p_trig_count = len(p_trig_idxs)
            p_trig_dur = events.loc[p_trig_idxs].duration.sum()
            p_dur = events[events.code == parent_code].duration.sum()

            child_desc = events[
                events.code == child_code].description.unique()[0]
            c_count = len(events.loc[events.code == child_code])
            c_trig_count = len(c_trig_idxs)
            c_trig_dur = events.loc[c_trig_idxs].duration.sum()
            c_dur = events[events.code == child_code].duration.sum()

            trig_summary.loc[i, 'parent_event'] = parent_code
            trig_summary.loc[i, 'parent_desc'] = parent_desc
            trig_summary.loc[i, 'p_count'] = p_count
            trig_summary.loc[i, 'p_dur'] = p_dur
            trig_summary.loc[i, 'p_trig_count'] = p_trig_count
            trig_summary.loc[i, 'p_trig_dur'] = p_trig_dur

            trig_summary.loc[i, 'child_event'] = child_code
            trig_summary.loc[i, 'child_desc'] = child_desc
            trig_summary.loc[i, 'c_count'] = c_count
            trig_summary.loc[i, 'c_dur'] = c_dur
            trig_summary.loc[i, 'c_trig_count'] = c_trig_count
            trig_summary.loc[i, 'c_trig_dur'] = c_trig_dur

            trig_summary.loc[i, 'p_prob_count'] = p_trig_count / p_count
            try:
                trig_summary.loc[i, 'p_prob_dur'] = p_trig_dur / p_dur
            except:  # noqa
                trig_summary.loc[i, 'p_prob_dur'] = 0

            trig_summary.loc[i, 'c_prob_count'] = c_trig_count / c_count
            try:
                trig_summary.loc[i, 'c_prob_dur'] = c_trig_dur / c_dur
            except:  # noqa
                trig_summary.loc[i, 'c_prob_dur'] = 0

            trig_indices[parent_code, child_code] = {
                'p_trig_idxs': p_trig_idxs, 'c_trig_idxs': c_trig_idxs}

            i += 1
    return trig_summary, trig_indices
