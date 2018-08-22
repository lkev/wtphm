import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import gridspec
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import warnings


def get_batch_features(event_data, fault_codes, batch_data, method, lo=1,
                       hi=10, num=1, event_type='fault_events'):
    """Extract features from batches of events which appear during
    stoppages.

    Only features from batches that comply with certain constraints are
    included. These constraints are chosen depending on which feature
    extraction method is used. Details of the feature extraction methods can
    be found in [1].

    **Note:** For each "batch" of alarms, there are up to `num_codes` unique
    alarm codes. Each alarm has an associated start time, `time_on`.

    Args
    ----
    event_data: pandas.DataFrame
        The original events/fault data
    fault_codes: numpy.ndarray
        All event codes that will be treated as fault events for the batches
    batch_data: pandas.DataFrame
        The dataframe holding the indices in event_data and start and end times
        for each batch
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

    Returns
    -------
    feature_array: numpy.ndarray
        An array of feature arrays corresponding to each batch that has has
        met the 'hi' and 'lo' criteria
    assoc_batch: unmpy.ndarray
        An array of 2-length index arrays. It is the same length as
        feature_array, and each entry points to the corresponding
        feature_array's index in batch_data, which in turn contains the index
        of the feature_arrays associated events in the original events_data
        or fault_data.
    """

    if event_type == 'fault_events':
        event_type = 'fault_event_ids'
    elif event_type == 'all_events':
        event_type = 'all_event_ids'

    # number of different fault codes
    num_codes = len(np.unique(fault_codes))

    # set up the indexing for each section of the batch_features vector
    code_idx = {}
    for i, j in zip(np.unique(np.sort(fault_codes)),
                    np.arange(0, num_codes)):
        code_idx[i] = j

    if method == 'basic':
        feature_array, assoc_batch = _batch_features_basic(
            batch_data, event_type, event_data, num_codes,
            code_idx, lo, hi, num)
    elif method == 't_on':
        feature_array, assoc_batch = _batch_features_t_on(
            batch_data, event_type, event_data, num_codes,
            code_idx, lo, hi, num)
    elif method == 'time':
        feature_array, assoc_batch = _batch_features_t_on_time(
            batch_data, event_type, event_data, num_codes,
            code_idx, lo, hi, num)

    return feature_array, assoc_batch


def _batch_features_basic(batch_data, event_type, event_data, num_codes,
                          code_idx, lo, hi, num):
    '''Called when method='basic' for extract_batch_features()'''

    feature_array = []
    assoc_batch = []

    for b in batch_data.itertuples():
        # get fault/all event codes in current batch of events
        batch_codes = event_data.loc[
            b._asdict()[event_type], ['time_on', 'code']
        ].drop_duplicates().code

        # get the alarms in this batch:
        if (len(batch_codes) >= lo) & (len(batch_codes) <= hi):
            batch_features = np.array([0]).repeat(num_codes).repeat(hi)

            k = 0
            for batch_code in batch_codes:
                batch_code_pattern = np.array(0).repeat(num_codes)
                batch_code_pattern[code_idx[batch_code]] = num
                batch_features[k:k + num_codes] = batch_code_pattern
                k += num_codes

            batch_features = list(batch_features)
            feature_array.append(batch_features)
            assoc_batch.append(b.Index)

    feature_array = np.array(feature_array)
    assoc_batch = np.array(assoc_batch)

    return feature_array, assoc_batch


def _batch_features_t_on(batch_data, event_type, event_data, num_codes,
                         code_idx, lo, hi, num):
    '''Called when method='t_on' for extract_batch_features()'''
    feature_array = []
    assoc_batch = []

    for b in batch_data.itertuples():
        # get fault/all events in current batch of events
        batch_events = event_data.loc[
            b._asdict()[event_type], ['time_on', 'code']
        ].drop_duplicates()

        # get the unique time_ons in this batch:
        unique_t_ons = batch_events.time_on.unique()
        if (len(unique_t_ons) >= lo) & (len(unique_t_ons) <= hi):
            batch_features = np.array([0]).repeat(num_codes).repeat(hi)

            k = 0
            # get the pattern of alarms for each t_on in this batch:
            for t_on in unique_t_ons:
                t_on_alarm_pattern = np.array(0).repeat(num_codes)
                for code in batch_events[
                        batch_events.time_on == pd.Timestamp(t_on)].code.values:
                    t_on_alarm_pattern[code_idx[code]] = num
                batch_features[k:k + num_codes] = t_on_alarm_pattern
                k += num_codes
            batch_features = list(batch_features)
            feature_array.append(batch_features)
            assoc_batch.append(b.Index)

    feature_array = np.array(feature_array)
    assoc_batch = np.array(assoc_batch)

    return feature_array, assoc_batch


def _batch_features_t_on_time(batch_data, event_type, event_data, num_codes,
                              code_idx, lo, hi, num):
    '''Called when method='time' for extract_batch_features()'''

    feature_array = []
    assoc_batch = []

    for b in batch_data.itertuples():
        # get fault_data fault events in current batch
        batch_events = event_data.loc[
            b._asdict()[event_type], ['time_on', 'code']
        ].drop_duplicates()

        # get the time differences between each time_on, and the final
        # time_on and code 6
        t_diffs = list(batch_events.time_on)
        t_diffs.append(
            event_data.loc[b._asdict()['all_event_ids'][-1], 'time_on'])
        t_diffs = pd.DataFrame(t_diffs).drop_duplicates().diff().dropna()
        t_diffs = np.array(t_diffs[0]) / np.timedelta64(1, 's')

        # get the unique time_ons in this batch:
        unique_t_ons = batch_events.time_on.unique()

        if (len(unique_t_ons) >= lo) & (len(unique_t_ons) <= hi):
            batch_features = np.array([0]).repeat(num_codes + 1).repeat(hi)

            k = 0
            # get the pattern of alarms for each t_on in this batch:
            for t_on, t_diff in zip(unique_t_ons, t_diffs):
                t_on_alarm_pattern = np.array(0).repeat(num_codes)

                for code in batch_events[
                        batch_events.time_on == pd.Timestamp(t_on)].code.values:
                    t_on_alarm_pattern[code_idx[code]] = num

                batch_features[k:k + num_codes] = t_on_alarm_pattern
                batch_features[k + num_codes] = t_diff

                k += num_codes + 1

            batch_features = list(batch_features)
            feature_array.append(batch_features)
            assoc_batch.append(b.Index)

    feature_array = np.array(feature_array)
    assoc_batch = np.array(assoc_batch)

    return feature_array, assoc_batch


def sil_1_cluster(
        X, cluster_labels, axis_label=True, save=False, save_name=None,
        x_label="Silhouette coefficient values", avg_pos=.02,
        w=2.3, h=2.4):
    """Show the silhouette scores for `clusterer`, print the plot, and
    optionally save it

    Args
    ----
    X: features
    cluster_labels: the labels of each cluster
    axis_label: whether or not to label the cluster plot with each cluster's
        number
    save: whether or not to save the resulting silhouette plot
    save_name: the saved filename
    x_label: the x axis label for the plot
    avg_pos: where to position the text for the average silghouette score
        relative to the position of the "average" line
    w: width of plot
    h: height of plot


    Returns
    -------
    fig: matplotlib figure object
        The silhouette analysis
    """
    silhouette_avg = metrics.silhouette_score(
        X[cluster_labels != -1], cluster_labels[cluster_labels != -1])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

    fig, ax = plt.subplots(figsize=[w, h])

    ax.set_xlim([-.2, 1])
    # The (n_clusters+1)*31 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X[cluster_labels != -1]) + (n_clusters + 1) * 31])

    y_lower = 31

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral_r(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the
        # middle
        if axis_label is True:
            ax.text(-0.1, y_lower + 0.2 * size_cluster_i, str(i), size=8)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 31  # 31 for the 0 samples

    ax.set_title("n_clusters = {}".format(n_clusters))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label='{}'.format(silhouette_avg))
    plt.text(silhouette_avg + avg_pos, 20, 'avg: {0:.2f}'.
             format(silhouette_avg), color='red')

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    # ax.tick_params(axis='x', which='major', bottom='off')
    # # ax.grid(True, axis='x', which='both')

    plt.tight_layout()

    if save is True:
        plt.savefig(save_name)
    # plt.close()

    return fig


def sil_n_clusters(X, range_n_clusters, clust):
    """Compare silhouette scores across different numbers of clusters for
    AgglomerativeClustering, KMeans or similar

    Args
    ----
    X: features
    range_n_clusters: the range of clusters you want, e.g. [2,3,4,5,10,20]
    clust: the sklearn clusterer to use, e.g. KMeans

    Returns
    -------
    cluster_labels: numpy.ndarray
        The labels for the clusters, with each one corresponding to a feature
        vector in X
    also prints the silhouette analysis

    """
    warnings.warn(
        'this function needs to be rewritten and is no longer in use until it '
        'is updated. Hence its behaviour will significantly change',
        FutureWarning)
    N = len(range_n_clusters)
    cols = 3
    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, cols)

    fig = plt.figure()
    fig.set_size_inches(10, 10)

    for n_clusters, n in zip(range_n_clusters, range(N)):
        # Create a subplot with 1 row and 2 columns
        ax = plt.subplot(gs[n])

        ax.set_xlim([-.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if clust == 'agg':
            knn_graph = kneighbors_graph(X, 5, include_self=False)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters,
                                                connectivity=knn_graph)
        elif clust == 'km':
            clusterer = KMeans(n_clusters, random_state=10)

        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = metrics.silhouette_score(X, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral_r(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("n_clusters = {}".format(n_clusters))
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--",
                   label='{}'.format(silhouette_avg))
        plt.text(silhouette_avg + .02, 20, '{:.2f}'.format(silhouette_avg),
                 color='red')

        ax.set_yticks([])  # Clear the yaxis labels / ticks

    plt.tight_layout()

    return cluster_labels


def cluster_times(batch_data, cluster_labels, assoc_batch,
                  event_dur_type='down_dur'):
    """Returns a DataFrame with a summary of the size and durations of batch
    members

    Args
    ----
    batch_data: pandas.DataFrame
        The dataframe holding the indices in event_data and start and end times
        for each batch
    cluster_labels:  numpy.ndarray
        The labels for the clusters, with each one corresponding to a feature
        vector in assoc_batch
    assoc_batch: nunmpy.ndarray
        Indices of batch_inds associated with each feature_array. Obtained
        from the extract_batch_features() function in this module (see for
        details).
    event_dur_type: string
        The event group duration in batch_data to return, i.e. either
        'fault_dur' or 'down_dur'. 'down_dur' means the entire time the turbine
        was offline, 'fault_dur' just means while the turbine was faulting. See
        Batches.get_batches() in batch.py for details

    Returns
    -------
    summary: Pandas.DataFrame
        The DataFrame has the total duration, mean duration, standard deviation
        of the duration and number of stoppages in each cluster.
    """
    data = []
    cols = ['cluster', 'total_dur', 'mean_dur', 'std_dur', 'num']
    for l in np.unique(cluster_labels):
        batches_with_label = batch_data.loc[assoc_batch[cluster_labels == l]]
        total_dur = batches_with_label[event_dur_type].sum()
        mean_dur = batches_with_label[event_dur_type].mean()
        std_dur = batches_with_label[event_dur_type].std()
        num = len(batches_with_label)
        data.append([l, total_dur, mean_dur, std_dur, num])

    summary = pd.DataFrame(data=data, columns=cols)

    return summary
