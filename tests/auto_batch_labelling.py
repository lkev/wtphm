"""
THIS IS NOT A UNIT TEST.

Just a make-shift testing file because I'm shite at testing (for now).
"""

import sys
import pandas as pd
import importlib as imp

imp_path = 'C:/users/leahy/Google Drive/UCC/PhD/Code/modules/'\
    'wtphm/'
sys.path.insert(0, imp_path)

import wtphm # noqa
imp.reload(wtphm)
events = pd.read_csv(
    imp_path + 'examples/events_data.csv',
    parse_dates=['time_on', 'time_off'])
events.duration = pd.to_timedelta(events.duration)

scada = pd.read_csv(imp_path + 'examples/scada_data.csv',
                    parse_dates=['time'])

# codes that cause the turbine to come to a stop
stop_codes = events[(events.stop_cat.isin(
    ['maintenance', 'test', 'sensor', 'grid'])) |
                    (events.stop_cat.str.contains('fault'))]\
    .code.unique()
# these are groups of codes, where each group represents a set of pitch-related
# events, where each memeber of the set represents the same event but along a
# different blade axis
pitch_code_groups = [[300, 301, 302], [400, 401], [500, 501, 502], [600, 601],
                     [700, 701, 702]]

# group the data
grouped_events, grouped_stop_codes = wtphm.batch.get_grouped_event_data(
    event_data=events, code_groups=pitch_code_groups,
    fault_codes=stop_codes)

# create the batches
batches = wtphm.batch.get_batch_data(
    event_data=grouped_events, fault_codes=grouped_stop_codes, ok_code=207,
    t_sep_lim='1 hours')

# get the cats
batches = wtphm.batch.get_batch_stop_cats(
    batches, events, scada, 'lot', 'mt', 'rt')

batches.head()
