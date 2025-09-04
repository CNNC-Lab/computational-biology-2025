import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as pl
from scipy import stats as st

# import conic.utils
from tools import visualization as viz


def recurrence_plot(time_series, dt=1, ax=None, color='k', type='.', display=True, save=False, **kwargs):
    """
    Plot a general recurrence plot of a 1D time series
    :param save:
    :param display:
    :param type:
    :param color:
    :param dt:
    :param time_series:
    :param ax:
    :param kwargs:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    in_pl = []
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    for ii, isi_val in enumerate(time_series):
        if ii < len(time_series) - int(dt):
            ax.plot(isi_val, time_series[ii + int(dt)], type, c=color, **pl_props)
    ax.set(**ax_props)
    ax.set_xlabel(r'$x(t)$')
    ax.set_ylabel(r'$x(t-{0})$'.format(str(dt)))

    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save + 'recurrence.pdf')

    if display:
        pl.show(block=False)


def plot_spectrogram(spec, t, f, ax):
    """
    Plot a simple spectrogram
    :param spec: spectrogram
    :param t: time axis
    :param f: sampling frequency
    :param ax: axis
    :return:
    """
    ax.pcolormesh(t, f, spec)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlable('Time [sec]')


def plot_acc(t, accs, fit_params, acc_function, title='', ax=None, display=True, save=False):
    """
    Plot autocorrelation decay and exponential fit (can be used for other purposes where an exponential fit to the
    data is suitable
    :param t:
    :param accs:
    :param fit_params:
    :param acc_function:
    :param title:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
    else:
        ax.set_title(title)

    for n in range(accs.shape[0]):
        ax.plot(t, accs[n, :], alpha=0.1, lw=0.1, color='k')

    error = np.nansum((np.nanmean(accs, 0) - acc_function(t, *fit_params)) ** 2)
    label = r'$a = {0}, b = {1}, {2}={3}, MSE = {4}$'.format(str(np.round(fit_params[0], 2)),
                                                             str(np.round(fit_params[1], 2)),
                                                             r'\tau_{int}', str(np.round(fit_params[2], 2)), str(error))
    ax.errorbar(t, np.nanmean(accs, 0), yerr=st.sem(accs), fmt='', color='k', alpha=0.3)
    ax.plot(t, np.nanmean(accs, 0), '--')
    ax.plot(t, acc_function(t, *fit_params), 'r', label=label)
    ax.legend()

    ax.set_ylabel(r'Autocorrelation')
    ax.set_xlabel(r'Lag [ms]')
    ax.set_xlim(min(t), max(t))
    # ax.set_ylim(0., 1.)

    if save:
        assert isinstance(save, str), "Please provide filename"
        ax.figure.savefig(save + 'acc_fit.pdf')

    if display:
        pl.show(block=False)
