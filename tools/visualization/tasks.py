import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.cluster import SpectralBiclustering

# import conic.utils
from conic_tools.operations import empty
from conic_tools import visualization as viz

def plot_w_out(w_out, label, display=True, save=False):
    """
    Creates a histogram of the readout weights
    """
    fig1, ax1 = pl.subplots()
    fig1.suptitle("{} - Biclustering readout weights".format(str(label)))
    n_clusters = np.min(w_out.shape)
    n_bars = np.max(w_out.shape)
    model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                                 random_state=0)
    model.fit(w_out)
    fit_data = w_out[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    ax1.matshow(fit_data, cmap=pl.cm.Blues, aspect='auto')
    ax1.set_yticks(list(range(len(model.row_labels_))))
    ax1.set_yticklabels(np.argsort(model.row_labels_))
    ax1.set_ylabel("Out")
    ax1.set_xlabel("Neuron")

    if np.argmin(w_out.shape) == 0:
        w_out = w_out.copy().T
    ##########################################################
    fig = pl.figure()
    for n in range(n_clusters):
        ax = fig.add_subplot(1, n_clusters, n + 1)
        ax.set_title(r"{} - $".format(str(label)) + r"W^{\mathrm{out}}_{" + "{}".format(n) + r"}$")
        ax.barh(list(range(n_bars)), w_out[:, n], height=1.0, linewidth=0, alpha=0.8)
        ax.set_ylim([0, w_out.shape[0]])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    if save:
        assert isinstance(save, str), "Please provide filename"
        fig1.savefig(save + 'W_out_Biclustering.pdf')
        fig.savefig(save + 'w_out.pdf')
    if display:
        pl.show(block=False)


def plot_confusion_matrix(matrix, label='', ax_label=None, ax=None, display=True, save=False):
    """
    """
    if ax is not None:
        fig1, ax1 = viz.helper.check_axis(ax)
    else:
        fig1, ax1 = pl.subplots()
    if not empty(label):
        fig1.suptitle(r"${0}$ - Confusion Matrix".format(str(label)))
    if ax_label is not None:
        ax1.set_title(ax_label)
    ax1.matshow(matrix, cmap=pl.cm.YlGn, aspect='auto')
    viz.helper.fig_output(fig1, display, save)


def plot_target_out(target, output, time_axis=None, label='', display=False, save=False):
    """

    :param target:
    :param output:
    :param label:
    :param display:
    :param save:
    :return:
    """
    fig2, ax2 = pl.subplots()
    fig2.suptitle(label)
    if output.shape == target.shape:
        tg = target[0]
        oo = output[0]
    else:
        tg = target[0]
        oo = output[:, 0]

    if time_axis is None:
        time_axis = np.arange(tg.shape[1])

    ax2ins = zoomed_inset_axes(ax2, 0.5, loc=1)
    ax2ins.plot(tg, c='r')
    ax2ins.plot(oo, c='b')
    ax2ins.set_xlim([100, 200])
    ax2ins.set_ylim([np.min(tg), np.max(tg)])

    mark_inset(ax2, ax2ins, loc1=2, loc2=4, fc="none", ec="0.5")

    pl1 = ax2.plot(tg, c='r', label='target')
    pl2 = ax2.plot(oo, c='b', label='output')
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('u(t)')
    ax2.legend(loc=3)
    if display:
        pl.show(block=False)
    if save:
        pl.savefig(save + label + '_TargetOut.pdf')
