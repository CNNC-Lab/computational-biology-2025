import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as st

# import conic.utils
from tools import visualization as viz

def plot_matrix(matrix, labels=None, ax=None, save=False, display=True, data_label=None):
    """
    Plots a 2D matrix as an image
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    if data_label is not None:
        fig.suptitle(data_label)

    if np.array_equal(matrix, matrix.astype(bool)):
        plt = ax.imshow(1 - matrix, interpolation='nearest', aspect='auto', extent=None,
                        cmap='gray')
    else:
        plt = ax.imshow(matrix, aspect='auto', interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="4%")
        pl.colorbar(plt, cax=cax)

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
    # ax.set_yticks(np.arange(matrix.shape[0]))
    viz.helper.fig_output(fig, display, save)
    return fig, ax


def plot_io_curve(inputs, outputs, ax=None, save=False, display=False, **kwargs):
    """
    Plot any i/o curve
    :param inputs:
    :param outputs:
    :param ax:
    :param save:
    :param display:
    :param kwargs:
    :return:
    """
    fig, ax = viz.helper.check_axis(ax)
    ax_props, plot_props = viz.helper.parse_plot_arguments(ax, **kwargs)
    ax.plot(inputs, outputs, **plot_props)
    ax.set(**ax_props)
    viz.helper.fig_output(fig, display=display, save=save)


def plot_histogram(data, n_bins, norm=True, mark_mean=False, mark_median=False, ax=None, color='b', display=True,
                   save=False, **kwargs):
    """
    Default histogram plotting routine
    :param data: data to plot (list or np.array)
    :param n_bins: number of bins to use (int)
    :param norm: normalized or not (bool)
    :param mark_mean: add a vertical line annotating the mean (bool)
    :param mark_median: add a vertical line annotating the median (bool)
    :param ax: axis to plot on (if None a new figure will be created)
    :param color: histogram color
    :param display: show figure (bool)
    :param save: save figure (False or string with figure path)
    :return n, bins: binned data
    """
    data = np.array(data)
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)

    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])
        kwargs.pop('suptitle')

    # extract properties from kwargs and divide them into axes properties and others
    in_pl = ['label', 'alpha', 'orientation']
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties() and k not in in_pl}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties() or k in in_pl}

    # if len(tmpa) > 1:
    # 	tmp = list(itertools.chain(*tmpa))
    data = data[data != 0]
    if np.any(data[np.isnan(data)]):
        print("Removing NaN")
        data = data[~np.isnan(data)]
    if np.any(data[np.isinf(data)]):
        print("Removing inf")
        data = data[~np.isinf(data)]

    n = 0
    bins = 0
    if norm and list(data):
        weights = np.ones_like(data) / float(len(data))
        n, bins, patches = ax.hist(data, n_bins, weights=weights, **pl_props)  # histtype='stepfilled', alpha=0.8)
        pl.setp(patches, 'facecolor', color)
    elif list(data):
        n, bins, patches = ax.hist(data, n_bins, **pl_props)
        pl.setp(patches, 'facecolor', color)

    if 'label' in list(pl_props.keys()):
        pl.legend()

    if mark_mean:
        ax.axvline(data.mean(), color=color, linestyle='dashed')
    if mark_median:
        ax.axvline(np.median(data), color=color, linestyle='dashed')

    ax.set(**ax_props)

    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save)

    if display:
        pl.show(block=False)

    return n, bins


def violin_plot(ax, data, pos, location=-1, color='y'):
    """
    Default violin plot routine
    :param ax:
    :param data:
    :param pos:
    :param location: location on the axis (-1 left,1 right or 0 both)
    :param color:
    :return:
    """
    dist = max(pos) - min(pos)
    w = min(0.15 * max(dist, 1.0), 0.5)
    for d, p, c in zip(data, pos, color):
        k = st.gaussian_kde(d)  # calculates the kernel density
        m = k.dataset.min()  # lower bound of violin
        M = k.dataset.max()  # upper bound of violin
        x = np.arange(m, M, (M - m) / 100.)  # support for violin
        v = k.evaluate(x)  # violin profile (density curve)
        v = v / v.max() * w  # scaling the violin to the available space
        if location:
            ax.fill_betweenx(x, p, (location * v) + p, facecolor=c, alpha=0.3)
        else:
            ax.fill_betweenx(x, p, v + p, facecolor=c, alpha=0.3)
            ax.fill_betweenx(x, p, -v + p, facecolor=c, alpha=0.3)


def box_plot(ax, data, pos):
    """
    creates one or a set of boxplots on the axis provided
    :param ax: axis handle
    :param data: list of data points
    :param pos: list of x positions
    :return:
    """
    ax.boxplot(data, notch=1, positions=pos, vert=1, sym='')


def plot_histograms(ax_list, data_list, n_bins, args_list=None, colors=None, cmap='hsv', kde=False, display=True,
                    save=False):
    """

    :param ax_list:
    :param data_list:
    :param n_bins:
    :param args_list:
    :param cmap:
    :return:
    """
    assert (len(ax_list) == len(data_list)), "Data dimension mismatch"
    if colors is None:
        cc = viz.helper.get_cmap(len(ax_list), cmap)
        colors = [cc(ii) for ii in range(len(ax_list))]
    counter = list(range(len(ax_list)))
    for ax, data, c in zip(ax_list, data_list, counter):
        n, bins = plot_histogram(data, n_bins[c], ax=ax, color=colors[c], display=False,
                                 **{'histtype': 'stepfilled', 'alpha': 0.6})
        if kde:
            approximate_pdf_isi = st.kde.gaussian_kde(data)
            x = np.linspace(np.min(data), np.max(data), n_bins[c])
            y = approximate_pdf_isi(x)
            y /= np.sum(y)
            ax.plot(x, y, color=colors[c], lw=2)
        if args_list is not None:
            ax.set(**args_list[c])
        ax.set_ylim([0., np.max(n)])
    fig = pl.gcf()
    viz.helper.fig_output(fig, display=display, save=save)


def scatter_variability(variable, ax=None, display=True, save=False):
    """
    scatter the variance vs mean of a given variable
    :param variable:
    :return:
    """
    if ax is None:
        fig, ax = pl.subplots()
    else:
        fig, ax = viz.helper.check_axis(ax)
    variable = np.array(variable)
    vars = []
    means = []
    if len(np.shape(variable)) == 2:
        for n in range(np.shape(variable)[0]):
            vars.append(np.var(variable[n, :]))
            means.append(np.mean(variable[n, :]))
    else:
        for n in range(len(variable)):
            vars.append(np.var(variable[n]))
            means.append(np.mean(variable[n]))

    ax.scatter(means, vars, color='k', lw=0.5, alpha=0.3)
    x_range = np.linspace(min(means), max(means), 100)
    ax.plot(x_range, x_range, '--r', lw=2)
    ax.set_xlabel('Means')
    ax.set_ylabel('Variances')
    viz.helper.fig_output(fig, display=display, save=save)


def plot_2d_arrays(image_arrays, axis, fig_handle=None, labels=None, cmap='coolwarm', boundaries=None,
                   interpolation='nearest', display=True, **kwargs):
    """
    Plots a list of arrays as images in the corresponding axis with the corresponding colorbar

    :return:
    """
    if boundaries is None:
        boundaries = []
    if labels is None:
        labels = []
    assert len(image_arrays) == len(axis), "Number of provided arrays must match number of axes"

    origin = 'upper'
    for idx, ax in enumerate(axis):
        if not isinstance(ax, mpl.axes.Axes):
            raise ValueError('ax must be matplotlib.axes.Axes instance.')
        else:
            plt1 = ax.imshow(image_arrays[idx], aspect='auto', origin=origin, cmap=cmap, interpolation=interpolation)
            if boundaries:
                cont = ax.contour(image_arrays[idx], boundaries[idx], origin='lower', colors='k', linewidths=2)
                pl.clabel(cont, fmt='%2.1f', colors='k', fontsize=12)
            if labels:
                ax.set_title(labels[idx])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "10%", pad="4%")
            if fig_handle is not None:
                # cbar = fig_handle.colorbar(plt1, cax=cax, format='%.2f')
                cbar = fig_handle.colorbar(plt1, cax=cax)
                cbar.ax.tick_params(labelsize=15)
            ax.set(**kwargs)
            pl.draw()
    if display:
        pl.show(block=False)


def plot_from_dict(dictionary, ax, bar_width=0.2):
    """
    Make a bar plot from a dictionary k: v, with xlabel=k and height=v
    :param dictionary: input dictionary
    :param ax: axis to plot on
    :param bar_width:
    :return:
    """
    x_labels = list(dictionary.keys())
    y_values = list(dictionary.values())
    ax.bar(np.arange(len(x_labels)), y_values, width=bar_width)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Freq.')
    ax.set_xlabel('Token')
