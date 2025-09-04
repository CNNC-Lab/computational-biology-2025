import matplotlib as mpl
import numpy as np
from matplotlib import axes, pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from conic.networks import Network
from conic_tools.analysis.metrics.connectivity import spectral_radius
from conic_tools.visualization import helper as plt_helper


def plot_spectral_radius(w, ax=None, display=True, save=False):
    """
    Plot the spectral radius of the connectivity matrix
    :param w: matrix
    :param ax: axis where to plot (if None a new figure is generated)
    :param save: path to the folder where to save the figure
    :param display: [bool]
    :return:
    """
    fig, ax = plt_helper.check_axis(ax)
    eigs = spectral_radius(w)
    ax.scatter(np.real(eigs), np.imag(eigs))
    ax.set_title(r"$\rho(W)=$" + "{0!s}".format(np.max(np.real(eigs))))
    ax.set_xlabel(r"$\mathrm{Re(W)}$")
    ax.set_ylabel(r"$\mathrm{Im(W)}$")

    plt_helper.fig_output(fig, display=display, save=save)


def plot_network_topology(network, colors=None, ax=None, dim=2, display=True, save=False, **kwargs):
    """
    Plot the network's spatial arrangement
    :return:
    """
    # assert isinstance(network, Network)
    if colors is not None:
        assert len(colors) == len(network.populations), "Specify one color per population"
    else:
        cmap = plt_helper.get_cmap(len(network.populations), 'jet')
        colors = [cmap(i) for i in range(len(network.populations))]

    if (ax is not None) and (not isinstance(ax, axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if ax is None and dim < 3:
        fig, ax = pl.subplots()
    elif ax is None and dim == 3:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
    plot_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}
    # ax.set_title(network.name)
    ax.set(**ax_props)

    for c, p in zip(colors, network.populations.values()):
        assert p.topology, "Population %s has no topology" % str(p.name)
        # positions = list(zip(*[tp.GetPosition([n])[0] for n in nest.GetLeaves(p.layer_gid)[0]]))
        positions = p.nodes.spatial['positions']
        if len(positions) < 3:
            ax.plot(positions[0], positions[1], 'o', color=c, label=p.name, **plot_props)
        else:
            ax.scatter(positions[0], positions[1], positions[2], depthshade=True, c=c, label=p.name,
                       **plot_props)
    pl.legend(loc=1)
    if display:
        pl.show(block=False)
    if save:
        assert isinstance(save, str), "Please provide filename"
        pl.savefig(save)


def plot_connectivity_matrix(matrix, source_name, target_name, label='', ax=None,
                             display=True, save=False):
    """

    :param matrix:
    :param source_name:
    :param target_name:
    :param label:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')

    if len(label.split('_')) == 2:
        title = label.split('_')[0] + '-' + label.split('_')[1]
        label = title
    else:
        title = label
    if ax is None:
        fig, ax = pl.subplots()
        fig.suptitle(r'${0}$'.format(str(title)))
    else:
        ax.set_title(r'${0}$'.format(str(title)))

    plt1 = ax.imshow(matrix, interpolation='nearest', aspect='auto', extent=None, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    pl.colorbar(plt1, cax=cax)
    ax.set_title(label)
    ax.set_xlabel('Source=' + str(source_name))
    ax.set_ylabel('Target=' + str(target_name))
    if display:
        pl.show(block=False)
    if save:
        pl.savefig(save + '{0}connectivityMatrix.pdf'.format(label))
