import numpy as np
from matplotlib import pyplot as pl

# import conic.utils
from conic_tools.analysis.signals import AnalogSignal, AnalogSignalList
from conic_tools import visualization as viz

class AnalogSignalPlots(object):
    """
    Wrapper object for all plots pertaining to continuous signals
    """

    def __init__(self, analog_signal_list, start=None, stop=None):
        """
        Initialize AnalogSignalPlot object
        :param analog_signal_list: AnalogSignalList object
        :param start: [float] start time for the display (if None, range is taken from data)
        :param stop: [float] stop time (if None, range is taken from data)
        """
        if (not isinstance(analog_signal_list, AnalogSignalList)) and (not isinstance(analog_signal_list, AnalogSignal)):
            raise Exception("Error, argument should be an AnalogSignal or AnalogSignalList")

        self.signal_list = analog_signal_list

        if start is None:
            self.start = self.signal_list.t_start
        else:
            self.start = start
        if stop is None:
            self.stop = self.signal_list.t_stop
        else:
            self.stop = stop

    def plot(self, ax=None, display=True, save=False, **kwargs):
        """
        Simply plot the contents of the AnalogSignal
        :param ax: axis handle
        :param display: [bool]
        :param save: [bool]
        :param kwargs: extra key-word arguments - particularly important are the axis labels
        and the plot colors
        """
        fig, ax = viz.helper.check_axis(ax)

        # extract properties from kwargs and divide them into axes properties and others
        ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}  # TODO: improve
        tt = self.signal_list.time_slice(self.start, self.stop)

        if isinstance(self.signal_list, AnalogSignal):
            times = tt.time_axis()
            signal = tt.raw_data()
            ax.plot(times, signal, **pl_props)

        elif isinstance(self.signal_list, AnalogSignalList):
            ids = self.signal_list.raw_data()[:, 1]
            for n in np.unique(ids):
                tmp = tt.id_slice([n])
                signal = tmp.raw_data()[:, 0]
                times = tmp.time_axis()
                ax.plot(times, signal, **pl_props)

        ax.set(**ax_props)
        ax.set(xlim=[self.start, self.stop])

        if display:
            pl.show(block=False)
        if save:
            assert isinstance(save, str), "Please provide filename"
            pl.savefig(save)

    def plot_Vm(self, ax=None, with_spikes=True, v_reset=None, v_th=None, display=True, save=False, **kwargs):
        """
        Special function to plot the time course of the membrane potential with or without highlighting the spike times
        :param with_spikes: [bool]
        """
        fig, ax = viz.helper.check_axis(ax)

        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'V_{m} [mV]')
        ax.set_xlim(self.start, self.stop)

        tt = self.signal_list.time_slice(self.start, self.stop)

        if isinstance(self.signal_list, AnalogSignalList):
            ids = self.signal_list.raw_data()[:, 1]
            for n in np.unique(ids):
                tmp = tt.id_slice([n])
                vm = tmp.raw_data()[:, 0]
                times = tmp.time_axis()
        elif isinstance(self.signal_list, AnalogSignal):
            times = tt.time_axis()
            vm = tt.raw_data()
        else:
            raise ValueError("times and vm not specified")

        ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
        pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}  # TODO: improve

        if len(vm) != len(times):
            times = times[:-1]

        ax.plot(times, vm, 'k', **pl_props)

        if with_spikes:
            assert (v_reset is not None) and (v_th is not None), "To mark the spike times, please provide the " \
                                                                 "v_reset and v_th values"
            idxs = vm.argsort()
            possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] !=
                                                                                                         v_reset)]
            ax.vlines(times[possible_spike_times], v_th, 50., color='k', **pl_props)
            ax.set_ylim(min(vm) - 5., 10.)
        else:
            ax.set_ylim(min(vm) - 5., max(vm) + 5.)

        ax.set(**ax_props)

        if display:
            pl.show(block=False)
        if save:
            assert isinstance(save, str), "Please provide filename"
            pl.savefig(save)
        return ax
