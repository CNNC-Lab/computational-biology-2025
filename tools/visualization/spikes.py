import itertools

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tools.visualization import helper as plt_helper
from tools.analysis import signals
from tools.analysis.metrics import spikes as spk_metrics


# from conic.networks.nest_snn.tools.analysis import metrics


class SpikePlots(object):
    """
    Wrapper object with all the methods and functions necessary to visualize spiking
    activity from a simple dot display to more visually appealing rasters,
    as well as histograms of the most relevant statistical descriptors and so on..
    """

    def __init__(self, spikelist, start=None, stop=None, N=None):
        """
        Initialize SpikePlot object
        :param spikelist: SpikeList object, sliced to match the (start, stop) interval
        :param start: [float] start time for the display (if None, range is taken from data)
        :param stop: [float] stop time (if None, range is taken from data)
        """
        if not isinstance(spikelist, signals.spikes.SpikeList):
            raise Exception("Error, argument should be a SpikeList object")

        if start is None:
            self.start = spikelist.t_start
        else:
            self.start = start
        if stop is None:
            self.stop = spikelist.t_stop
        else:
            self.stop = stop
        if N is None:
            self.N = len(spikelist.id_list)

        self.spikelist = spikelist.time_slice(self.start, self.stop)

    def dot_display(self, gids_colors=None, with_rate=True, dt=1.0, display=True, ax=None, save=False,
                    default_color='b', fig=None, **kwargs):
        """
        Simplest case, dot display
        :param gids_colors: [list] if some ids should be highlighted in a different color, this should be specified by
        providing a list of (gids, color) pairs, where gids [numpy.ndarray] contains the ids and color is the
        corresponding color for those gids. If None, no ids are differentiated
        :param with_rate: [bool] - whether to display psth or not
        :param dt: [float] - delta t for the psth
        :param display: [bool] - display the figure
        :param ax: [axes handle] - axes on which to display the figure
        :param save: [bool] - save the figure
        :param default_color: [char] default color if no ids are differentiated
		:param fig: [matplotlib.figure]
        :param kwargs: [key=value pairs] axes properties
        """
        if (ax is not None) and (not isinstance(ax, list)) and (not isinstance(ax, mpl.axes.Axes)):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        elif (ax is not None) and (isinstance(ax, list)):
            for axis_ax in ax:
                if not isinstance(axis_ax, mpl.axes.Axes):
                    raise ValueError('ax must be matplotlib.axes.Axes instance.')

        if ax is None:
            fig = pl.figure()
            if 'suptitle' in kwargs:
                fig.suptitle(kwargs['suptitle'])
                kwargs.pop('suptitle')
            if with_rate:
                ax1 = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
                ax2 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)
                ax2.set(xlabel='Time [ms]', ylabel='Rate')
                ax1.set(ylabel='Neuron')
            else:
                ax1 = fig.add_subplot(111)
        else:
            if with_rate:
                assert isinstance(ax, list), "Incompatible properties... (with_rate requires two axes provided or None)"
                ax1 = ax[0]
                ax2 = ax[1]
            else:
                ax1 = ax

        if 'suptitle' in kwargs and fig is not None:
            fig.suptitle(kwargs['suptitle'])
            kwargs.pop('suptitle')

        # extract properties from kwargs and divide them into axes properties and others
        ax_props = {k: v for k, v in kwargs.items() if k in ax1.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax1.properties()}  # TODO: improve

        if gids_colors is None:
            times = self.spikelist.raw_data()[:, 0]
            neurons = self.spikelist.raw_data()[:, 1]
            ax1.plot(times, neurons, '.', color=default_color)
            ax1.set(ylim=[np.min(self.spikelist.id_list), np.max(self.spikelist.id_list)], xlim=[self.start, self.stop])
        else:
            assert isinstance(gids_colors, list), "gids_colors should be a list of (gids[list], color) pairs"
            ax_min_y = np.max(self.spikelist.id_list)
            ax_max_y = np.min(self.spikelist.id_list)
            for gid_color_pair in gids_colors:
                gids, color = gid_color_pair
                assert isinstance(gids, np.ndarray), "Gids should be a numpy.ndarray"

                tt = self.spikelist.id_slice(gids)  # it's okay since slice always returns new object
                times = tt.raw_data()[:, 0]
                neurons = tt.raw_data()[:, 1]
                ax1.plot(times, neurons, '.', color=color)
                ax_max_y = max(ax_max_y, max(tt.id_list))
                ax_min_y = min(ax_min_y, min(tt.id_list))
            ax1.set(ylim=[ax_min_y, ax_max_y], xlim=[self.start, self.stop])

        if with_rate:
            global_rate = self.spikelist.firing_rate(dt, average=True)
            mean_rate = self.spikelist.firing_rate(10., average=True)
            max_rate = max(global_rate) + 1
            min_rate = min(global_rate) + 1
            if gids_colors is None:
                time = self.spikelist.time_axis(dt)[:-1]
                ax2.plot(time, global_rate, **pl_props)
            else:
                assert isinstance(gids_colors, list), "gids_colors should be a list of (gids[list], color) pairs"
                for gid_color_pair in gids_colors:
                    gids, color = gid_color_pair
                    assert isinstance(gids, np.ndarray), "Gids should be a numpy.ndarray"

                    tt = self.spikelist.id_slice(gids)  # it's okay since slice always returns new object
                    time = tt.time_axis(dt)[:-1]
                    rate = tt.firing_rate(dt, average=True)
                    ax2.plot(time, rate, color=color, linewidth=1.0, alpha=0.8)
                    max_rate = max(rate) if max(rate) > max_rate else max_rate
            ax2.plot(self.spikelist.time_axis(10.)[:-1], mean_rate, 'k', linewidth=1.5)
            ax2.set(ylim=[min_rate, max_rate], xlim=[self.start, self.stop])
        else:
            ax1.set(**ax_props)
        if save:
            assert isinstance(save, str), "Please provide filename to save figure"
            pl.savefig(save)

        if display:
            pl.show(block=False)

    @staticmethod
    def mark_events(ax, input_obj, start=None, stop=None):
        """
        Highlight stimuli presentation times in axis
        :param ax:
        :param input_obj:
        :param start:
        :param stop:
        :return:
        """
        if not isinstance(ax, mpl.axes.Axes):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        if start is None:
            start = ax.get_xlim()[0]
        if stop is None:
            stop = ax.get_xlim()[1]

        color_map = plt_helper.get_cmap(input_obj.dimensions)
        y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 1)

        for k in range(input_obj.dimensions):
            onsets = input_obj.onset_times[k]
            offsets = input_obj.offset_times[k]

            assert (len(onsets) == len(offsets)), "Incorrect input object parameters"

            for idx, on in enumerate(onsets):
                if start - 500 < on < stop + 500:
                    ax.fill_betweenx(y_range, on, offsets[idx], facecolor=color_map(k), alpha=0.3)

    def print_activity_report(self, results=None, label='', n_pairs=500):
        """
        Displays on screen a summary of the network_architect settings and main statistics
        :param label: Population name
        """
        tt = self.spikelist.time_slice(self.start, self.stop)

        if results is None:
            stats = {}
            stats.update(spk_metrics.compute_isi_stats(tt, summary_only=True, display=True))
            stats.update(spk_metrics.compute_spike_stats(tt, time_bin=1., summary_only=True, display=True))
            stats.update(spk_metrics.compute_synchrony(tt, n_pairs=n_pairs, time_bin=1., tau=20.,
                                                   time_resolved=False, depth=1))
        else:
            stats = results

        print('\n###################################################################')
        print(' Activity recorded in [%s - %s] ms, from population %s ' % (str(self.start), str(self.stop), str(label)))
        print('###################################################################')
        print('Spiking Neurons: {0}/{1}'.format(str(len(np.nonzero(tt.mean_rates())[0])), str(self.N)))
        print(
            'Average Firing Rate: %.2f / %.2f Hz' % (np.mean(np.array(tt.mean_rates())[np.nonzero(tt.mean_rates())[0]]),
                                                     np.mean(tt.mean_rates())))
        # print 'Average Firing Rate (normalized by N): %.2f Hz' % (np.mean(tt.mean_rates()) * len(tt.id_list)) / self.N
        print('Fano Factor: %.2f' % stats['ffs'][0])
        print('*********************************\n\tISI metrics:\n*********************************')
        if 'lvs' in list(stats.keys()):
            print(('\t- CV: %.2f / - LV: %.2f / - LVR: %.2f / - IR: %.2f' % (stats['cvs'][0], stats['lvs'][0],
                                                                             stats['lvRs'][0], stats['iR'][0])))
            print(('\t- CVlog: %.2f / - H: %.2f [bits/spike]' % (stats['cvs_log'][0], stats['ents'][0])))
            print(('\t- 5p: %.2f ms' % stats['isi_5p'][0]))
        else:
            print('\t- CV: %.2f' % np.mean(stats['cvs']))

        print('*********************************\n\tSynchrony metrics:\n*********************************')
        if 'ccs_pearson' in list(stats.keys()):
            print(('\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), stats['ccs_pearson'][0])))
            print(('\t- CC [{0} pairs]: {1}'.format(str(n_pairs), str(stats['ccs'][0]))))
            if 'd_vr' in list(stats.keys()) and isinstance(stats['d_vr'], float):
                print(('\t- van Rossum distance: {0}'.format(str(stats['d_vr']))))
            elif 'd_vr' in list(stats.keys()) and not isinstance(stats['d_vr'], float):
                print(('\t- van Rossum distance: {0}'.format(str(np.mean(stats['d_vr'])))))
            if 'd_vp' in list(stats.keys()) and isinstance(stats['d_vp'], float):
                print(('\t- Victor Purpura distance: {0}'.format(str(stats['d_vp']))))
            elif 'd_vp' in list(stats.keys()) and not isinstance(stats['d_vp'], float):
                print(('\t- Victor Purpura distance: {0}'.format(str(np.mean(stats['d_vp'])))))
            if 'SPIKE_distance' in list(stats.keys()) and isinstance(stats['SPIKE_distance'], float):
                print(('\t- SPIKE similarity: %.2f / - ISI distance: %.2f ' % (stats[
                                                                                   'SPIKE_distance'],
                                                                               stats['ISI_distance'])))
            elif 'SPIKE_distance' in list(stats.keys()) and not isinstance(stats['SPIKE_distance'], float):
                print(('\t- SPIKE similarity: %.2f / - ISI distance: %.2f' % (np.mean(stats['SPIKE_distance']),
                                                                              np.mean(stats['ISI_distance']))))
            if 'SPIKE_sync' in list(stats.keys()):
                print(('\t- SPIKE Synchronization: %.2f' % np.mean(stats['SPIKE_sync'])))
        elif 'ccs' in list(stats.keys()):
            print(('\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), np.mean(stats['ccs']))))


def plot_single_raster(times, ax, t_start=0, t_stop=1000, save=False, display=True):
    """
    Plot the spike times of a single SpikeTrain as a vertical line
    :param times:
    :param ax:
    :param t_start:
    :param t_stop:
    :param save: False or string with full path to store
    :param display: bool
    :return:
    """
    fig, ax = plt_helper.check_axis(ax)
    for tt in times:
        ax.vlines(tt, 0.5, 1.5, color='k', linewidth=2)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'$\mathrm{S}_{i}$')
        ax.set_xlim([t_start, t_stop])
        ax.set_ylim([0.5, 1.6])
        ax.set_yticks([])
        ax.set_yticklabels([])
    plt_helper.fig_output(fig, display=display, save=save)


def plot_isis(isis, ax=None, save=False, display=False, **kwargs):
    """
    Plot the distribution of inter-spike-intervals provided
    :param isis:
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    fig, ax = plt_helper.check_axis(ax)
    # ax2 = inset_axes(ax, width="60%", height=1.5, loc=1)
    ax_props, plot_props = plt_helper.parse_plot_arguments(ax, **kwargs)

    ax.plot(np.arange(len(isis)), isis, '.', **plot_props)
    ax.set(**ax_props)
    # ax2.plot(list(range(len(inset['isi']))), inset['isi'], '.')
    plt_helper.fig_output(fig, display=display, save=save)


def plot_singleneuron_isis(isis, ax=None, save=False, display=False, **kwargs):
    """
    Plot ISI distribution for a single neuron
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        if 'suptitle' in kwargs:
            fig.suptitle(kwargs['suptitle'])
            kwargs.pop('suptitle')
        else:
            ax = fig.add_subplot(111)
    else:
        fig, ax = plt_helper.check_axis(ax)

    if 'inset' in kwargs.keys():
        inset = kwargs['inset']
        kwargs.pop('inset')
        ax2 = inset_axes(ax, width="60%", height=1.5, loc=1)
    else:
        inset = None
    in_pl = []
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    ax.plot(range(len(isis)), isis, '.', **pl_props)
    ax.set(**ax_props)

    if inset is not None:
        ax2.plot(range(len(inset['isi'])), inset['isi'], '.')
        inset.pop('isi')

    plt_helper.fig_output(fig, display=display, save=save)


def plot_raster(spike_list, dt, ax, sub_set=None, **kwargs):
    """
    Plot a nice-looking line raster
    :param spike_list: SpikeList object
    :param dt: shortest bin width
    :param ax: axis to plot on
    :param sub_set: display only subset of spiking neurons
    :param kwargs:
    :return:
    """
    if sub_set is not None:
        # plot a subset of the spiking neurons
        ids = np.random.permutation([x for ii, x in enumerate(spike_list.id_list) if spike_list.mean_rates()[ii]])[
              :sub_set]
        tmp = []
        for n_id, idd in enumerate(ids):
            tmp.append([(n_id, t) for t in spike_list.spiketrains[idd].spike_times])
        tmp = list(itertools.chain(*tmp))
        spks = signals.spikes.SpikeList(tmp, list(np.arange(sub_set)))
    else:
        spks = spike_list
    ax1a = pl.twinx(ax)
    spks.raster_plot(ax=ax, display=False, **kwargs)
    ax.grid(False)
    ax.set_ylabel(r'Neuron')
    ax.set_xlabel(r'Time $[\mathrm{ms}]$')
    ax1a.plot(spike_list.time_axis(dt)[:-1], spike_list.firing_rate(dt, average=True), 'k', lw=1., alpha=0.5)
    ax1a.plot(spike_list.time_axis(10.)[:-1], spike_list.firing_rate(10., average=True), 'r', lw=2.)
    ax1a.grid(False)
    ax1a.set_ylabel(r'Rate $[\mathrm{sps}/s]$')


def plot_synaptic_currents(I_ex, I_in, time_axis):
    fig, ax = pl.subplots()
    ax.plot(time_axis, I_ex, 'b')
    ax.plot(time_axis, I_in, 'r')
    ax.plot(time_axis, np.mean(I_ex) * np.ones_like(I_ex), 'b--')
    ax.plot(time_axis, np.mean(I_in) * np.ones_like(I_in), 'r--')
    ax.plot(time_axis, np.abs(I_ex) - np.abs(I_in), c='gray')
    ax.plot(time_axis, np.mean(np.abs(I_ex) - np.abs(I_in)) * np.ones_like(I_ex), '--', c='gray')


def pretty_raster(global_spike_list, analysis_interval=None, sub_pop_gids=None, max_rate=None, n_total_neurons=10,
                  ax=None, color='k', save=False):
    """
    Simple line raster to plot a subset of the populations (for publication)
    :return:
    """
    if analysis_interval is None:
        analysis_interval = [global_spike_list.t_start, global_spike_list.t_stop]

    plot_list = global_spike_list.time_slice(t_start=analysis_interval[0], t_stop=analysis_interval[1])
    if max_rate is not None:
        new_ids = np.intersect1d(plot_list.select_ids("cell.mean_rate() > 0"),
                                 plot_list.select_ids("cell.mean_rate() < {}".format(max_rate)))
    else:
        new_ids = global_spike_list.id_list

    if ax is None:
        fig = pl.figure()
        # pl.axis('off')
        ax = fig.add_subplot(111)  # , frameon=False)

    if sub_pop_gids is not None:
        assert (isinstance(sub_pop_gids, list)), "Provide a list of lists of gids"
        assert (len(sub_pop_gids) == 2), "Only 2 populations are currently covered"
        lenghts = list(map(len, sub_pop_gids))
        sample_neurons = [(n_total_neurons * n) / np.sum(lenghts) for n in lenghts]
        # id_ratios = float(min(lenghts)) / float(max(lenghts))
        # sample_neurons = [n_total_neurons * id_ratios, n_total_neurons * (1 - id_ratios)]

        neurons_1 = []
        neurons_2 = []
        while len(neurons_1) != sample_neurons[0] or len(neurons_2) != sample_neurons[1]:
            chosen = np.random.choice(new_ids, size=n_total_neurons, replace=False)
            neurons_1 = [x for x in chosen if x in sub_pop_gids[0]]
            neurons_2 = [x for x in chosen if x in sub_pop_gids[1]]
            if len(neurons_1) > sample_neurons[0]:
                neurons_1 = neurons_1[:sample_neurons[0]]
            if len(neurons_2) > sample_neurons[1]:
                neurons_2 = neurons_2[:sample_neurons[1]]

    else:
        chosen_ids = np.random.permutation(new_ids)[:n_total_neurons]
        new_list = plot_list.id_slice(chosen_ids)

        neuron_ids = [np.where(x == np.unique(new_list.raw_data()[:, 1]))[0][0] for x in new_list.raw_data()[:, 1]]
        tmp = [(neuron_ids[idx], time) for idx, time in enumerate(new_list.raw_data()[:, 0])]

    for idx, n in enumerate(tmp):
        ax.vlines(n[1] - analysis_interval[0], n[0] - 0.5, n[0] + 0.5, **{'color': color, 'lw': 1.})
    ax.set_ylim(-0.5, n_total_neurons - 0.5)
    ax.set_xlim(0., analysis_interval[1] - analysis_interval[0])
    ax.grid(False)
    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save)


def plot_response_activity(spike_list, input_stimulus, start=None, stop=None):
    """
    Plot population responses to stimuli (spiking activity)

    :param spike_list:
    :param input_stimulus:
    :return:
    """
    fig = pl.figure()
    ax1 = pl.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
    ax2 = pl.subplot2grid((12, 1), (7, 0), rowspan=2, colspan=1, sharex=ax1)

    rp = SpikePlots(spike_list, start, stop)
    plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'linewidth': 1.0, 'linestyle': '-'}
    rp.dot_display(ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)

    rp.mark_events(ax1, input_stimulus, start, stop)
    rp.mark_events(ax2, input_stimulus, start, stop)


def plot_averaged_time_resolved(results, spike_list, label='', epochs=None, color_map='jet', display=True, save=False):
    """

    :param results:
    :param spike_list:
    :param label:
    :param epochs:
    :param color_map:
    :param display:
    :param save:
    :return:
    """
    # time resolved regularity
    fig5 = pl.figure()
    fig5.suptitle('{0} - Time-resolved regularity'.format(str(label)))
    stats = ['isi_5p_profile', 'cvs_profile', 'cvs_log_profile', 'lvs_profile', 'iR_profile', 'ents_profile']
    cm = plt_helper.get_cmap(len(stats), color_map)

    for idx, n in enumerate(stats):
        ax = fig5.add_subplot(len(stats), 1, idx + 1)
        data_mean = np.array([results[n][i][0] for i in range(len(results[n]))])
        data_std = np.array([results[n][i][1] for i in range(len(results[n]))])
        t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))

        ax.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
        ax.fill_between(t_axis, data_mean - data_std, data_mean + data_std, facecolor=cm(idx), alpha=0.2)
        ax.set_ylabel(n)
        ax.set_xlabel('Time [ms]')
        ax.set_xlim(spike_list.time_parameters())
        if epochs is not None:
            plt_helper.mark_epochs(ax, epochs, color_map)

    # activity plots
    fig6 = pl.figure()
    fig6.suptitle('{0} - Activity Analysis'.format(str(label)))
    ax61 = pl.subplot2grid((25, 1), (0, 0), rowspan=20, colspan=1)
    ax62 = pl.subplot2grid((25, 1), (20, 0), rowspan=5, colspan=1)
    pretty_raster(spike_list, analysis_interval=[spike_list.t_start, spike_list.t_stop], n_total_neurons=1000, ax=ax61)
    # plot_raster(spike_list, 1., ax61, sub_set=100, **{'color': 'k', 'alpha': 0.8, 'marker': '|', 'markersize': 2})
    stats = ['ffs_profile']

    cm = plt_helper.get_cmap(len(stats), color_map)
    for idx, n in enumerate(stats):
        data_mean = np.array([results[n][i][0] for i in range(len(results[n]))])
        data_std = np.array([results[n][i][1] for i in range(len(results[n]))])
        t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))
        ax62.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
        ax62.fill_between(t_axis, data_mean - data_std, data_mean +
                          data_std, facecolor=cm(idx), alpha=0.2)
        ax62.set_ylabel(r'$\mathrm{FF}$')
        ax62.set_xlabel('Time [ms]')
        ax62.set_xlim(spike_list.time_parameters())
    if epochs is not None:
        plt_helper.mark_epochs(ax61, epochs, color_map)
        plt_helper.mark_epochs(ax62, epochs, color_map)

    if display:
        pl.show(block=False)
    if save:
        fig5.savefig(save + '{0}_time_resolved_reg.pdf'.format(str(label)))
        fig6.savefig(save + '{0}_activity_analysis.pdf'.format(str(label)))
