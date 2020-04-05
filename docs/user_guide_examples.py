import wtphm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# to fully display the dataframes:
pd.set_option('display.max_colwidth', 18)
pd.set_option('display.width', 100)

event_data = pd.read_csv('examples/event_data.csv',
                         parse_dates=['time_on', 'time_off'])
event_data.duration = pd.to_timedelta(event_data.duration)
scada_data = pd.read_csv('examples/scada_data.csv',
                         parse_dates=['time'])


event_data.head()

scada_data.head()


# codes that cause the turbine to come to a stop
stop_codes = event_data[
    (event_data.stop_cat.isin(['maintenance', 'test', 'sensor', 'grid'])) |
    (event_data.stop_cat.str.contains('fault'))].code.unique()
# each of these lists represents a set of pitch-related events, where
# each memeber of the set represents the same event but along a
# different blade axis
pitch_code_groups = [[300, 301, 302], [400, 401], [500, 501, 502],
                     [600, 601], [700, 701, 702]]
# event_data[event_data.code.isin(
#     [i for s in pitch_code_groups for i in s])].head()

event_data, stop_codes = wtphm.batch.get_grouped_event_data(
    event_data=event_data, code_groups=pitch_code_groups,
    fault_codes=stop_codes)
# viewing the now-grouped events from above:
# event_data.loc[[112, 114, 119, 131, 132]]

# create the batches
batch_data = wtphm.batch.get_batch_data(
    event_data=event_data, fault_codes=stop_codes, ok_code=207,
    t_sep_lim='1 hours')

root_cats = wtphm.batch.get_root_cats(batch_data, event_data)
batch_data = wtphm.batch.get_batch_stop_cats(
    batch_data, event_data, scada_data, grid_col='lot', maint_col='mt',
    rep_col='rt')
# root_cats.loc[15:20]

# all_pt_ids = wtphm.batch.get_cat_all_ids(root_cats, "fault_pt")
# batch_data.loc[all_pt_ids, 'batch_cat'] = "fault_pt"
# batch_data.loc[15:20, 'batch_cat']

# grid_ids = wtphm.batch.get_cat_present_ids(root_cats, 'grid')
# batch_data.loc[grid_ids, 'batch_cat'] = 'grid'
# batch_data.loc[15:20, 'batch_cat']
# batch_data.batch_cat = wtphm.batch.get_most_common_cats(root_cats)
#
# maint_ids = wtphm.batch.get_counter_active_ids(
#     batch_data, scada_data, 'mt', 60)
# batch_data.loc[maint_ids]

# start = batch_data.loc[20, 'start_time'] - pd.Timedelta('20 minutes')
# end = batch_data.loc[20, 'down_end_time'] + pd.Timedelta('20 minutes')
# t = batch_data.loc[20, 'turbine_num']
# scada_data.loc[
#     (scada_data.time >= start) & (scada_data.time <= end) &
#     (scada_data.turbine_num == t),
#     ['time', 'turbine_num', 'wind_speed', 'kw', 'ot', 'sot', 'dt']]

# plots

# durations = batch_data.groupby(
#     'batch_cat').down_dur.sum().reset_index().sort_values(by='down_dur')
# durations.down_dur = durations.down_dur.apply(
#     lambda x: x / np.timedelta64(1, 'h'))
# sns.set(font_scale=1.2)
# sns.set_style('white')
# fig, ax = plt.subplots(figsize=(4, 3))
# g = sns.barplot(data=durations, x='batch_cat', y='down_dur', ax=ax,
#                 color=sns.color_palette()[0])
# g.set_xticklabels(g.get_xticklabels(), rotation=40)
# ax.set(xlabel='Stop Category', ylabel='Total Downtime (hrs)')
# ax.yaxis.grid()
# plt.tight_layout()
# plt.savefig('docs/ug_p1.png')
