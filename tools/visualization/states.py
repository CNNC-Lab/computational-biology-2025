import itertools

import numpy as np
from matplotlib import pyplot as pl
from sklearn import decomposition as sk

from tools.visualization import helper as plt_helper
from tools.analysis.metrics import states


def plot_discrete_space(state_matrix, data_label='', label_seq=None, metric=None, colormap='jet', display=True,
                        save=False):
    """
    Plots a discrete state-space
    :return:
    """
    if state_matrix.shape[0] > 3:
        states.dimensionality_reduction(state_matrix, data_label, labels=label_seq, metric=metric, standardize=False,
                                         plot=True, colormap=colormap, display=display, save=save)

    elif state_matrix.shape[0] == 2:
        cmap = plt_helper.get_cmap(len(np.unique(label_seq)), cmap=colormap)
        scatter_projections(state_matrix.T, label_seq, cmap=cmap, display=display, save=save)

    elif state_matrix.shape[0] == 3:
        cmap = plt_helper.get_cmap(len(np.unique(label_seq)), cmap=colormap)
        scatter_projections(state_matrix.T, label_seq, cmap=cmap, display=display, save=save)


def plot_trajectory(response_matrix, pca_fit_obj=None, label='', color='r', ax=None, display=True, save=False):
    """

    :param response_matrix: [np.array] matrix of continuous responses
    :param pca_fit_obj:
    :param label:
    :param color:
    :param ax:
    :param display:
    :param save:
    :return:
    """
    fig, ax = plt_helper.check_axis(ax)

    if pca_fit_obj is None:
        pca_fit_obj = sk.PCA(n_components=min(response_matrix.shape))
    if not hasattr(pca_fit_obj, "explained_variance_ratio_"):
        pca_fit_obj.fit(response_matrix.T)
    X = pca_fit_obj.transform(response_matrix.transpose())
    # print("Explained Variance (first 3 components): %s" % str(pca_fit_obj.explained_variance_ratio_))

    # ax.clear()
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color=color, lw=2, label=label)
    # ax.set_title(label + r' - (3PCs) = {0}'.format(str(round(np.sum(pca_fit_obj.explained_variance_ratio_[:3]), 1))))
    # ax.grid()

    plt_helper.fig_output(fig, display=display, save=save)


def plot_dimensionality(result, pca_obj, rotated_data=None, data_label='', display=True, save=False):
    fig7 = pl.figure()
    ax71 = fig7.add_subplot(121, projection='3d')
    ax71.grid(False)
    ax72 = fig7.add_subplot(122)

    ax71.plot(rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2], '.-', color='r', lw=2, alpha=0.8)
    ax71.set_title(r'${0} - (3 PCs) = {1}$'.format(data_label, str(round(np.sum(
        pca_obj.explained_variance_ratio_[:3]), 1))))
    ax72.plot(pca_obj.explained_variance_ratio_, 'ob')
    ax72.plot(pca_obj.explained_variance_ratio_, '-b')
    ax72.plot(np.ones_like(pca_obj.explained_variance_ratio_) * result, np.linspace(0.,
                                                                                    np.max(
                                                                                        pca_obj.explained_variance_ratio_),
                                                                                    len(pca_obj.explained_variance_ratio_)),
              '--r', lw=2.5)
    ax72.set_xlabel(r'PC')
    ax72.set_ylabel(r'Variance Explained')
    ax72.set_xlim([0, round(result) * 2])
    ax72.set_ylim([0, np.max(pca_obj.explained_variance_ratio_)])
    if display:
        pl.show(block=False)
    if save:
        fig7.savefig(save + '{0}_dimensionality.pdf'.format(data_label))


def scatter_projections(state, label_sequence, cmap, ax=None, display=False, save=False):
    """
    Scatter plot 3D projections from a high-dimensional state matrix
    :param state:
    :param label_sequence:
    :param cmap: color map
    :param ax:
    :param display:
    :return:
    """
    if ax is None and state.shape[1] == 3:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
    elif ax is None and state.shape[1] == 2:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt_helper.check_axis(ax)

    unique_labels = np.unique(label_sequence)
    ccs = [np.array(cmap(i)) for i in range(len(unique_labels))]
    lab_seq = np.array(list(itertools.chain(label_sequence)))

    scatters = []
    for color, index in zip(ccs, unique_labels):
        if state.shape[1] == 3:
            tmp = ax.plot(state[np.where(lab_seq == index)[0], 0], state[np.where(lab_seq == index)[0], 1],
                          state[np.where(lab_seq == index)[0], 2], marker='o', linestyle='', ms=10, c=color,
                          alpha=0.8,
                          label=index)
        elif state.shape[1] == 2:
            tmp = ax.plot(state[np.where(lab_seq == index)[0], 0], state[np.where(lab_seq == index)[0], 1],
                          marker='o', linestyle='', ms=10, c=color, alpha=0.8, label=index)
        else:
            raise NotImplementedError("Input state matrix must be 2 or 3 dimensional")
        scatters.append(tmp[0])
    if len(unique_labels) <= 20:
        pl.legend(loc=0, handles=scatters)
    plt_helper.fig_output(fig, display, save)
