import itertools
import time

import numpy as np
from matplotlib import pyplot as pl
from scipy import optimize as opt
from sklearn import decomposition as sk, manifold as man

from tools.analysis.metrics.helper import err_func, acc_function
from tools.analysis.metrics.matrices import cross_trial_cc
from tools.analysis import signals
from tools.operations import iterate_obj_list

from tools.logger import info
from tools.visualization import helper as plt_helper
from tools.visualization.base import plot_matrix
from tools.visualization.states import plot_dimensionality, scatter_projections
from tools.visualization.timeseries import plot_acc
logger = info.get_logger(__name__)


def get_state_rank(state_matrix):
    """
    Calculates the rank of all state matrices.
    :return:
    """
    return np.linalg.matrix_rank(state_matrix)


def compute_dimensionality(response_matrix, pca_obj=None, label='', plot=False, display=True, save=False):
    """
    Measure the effective dimensionality of population responses. Based on Abbott et al. (2001). Interactions between
    intrinsic and stimulus-evoked activity in recurrent neural networks.

    :param response_matrix: matrix of continuous responses to analyze (NxT)
    :param pca_obj: if pre-computed, otherwise None
    :param label:
    :param plot:
    :param display:
    :param save:
    :return: (float) dimensionality
    """
    if display:
        logger.info("Determining effective dimensionality...")
        t_start = time.time()
    if pca_obj is None:
        n_features, n_samples = np.shape(response_matrix)  # features = neurons
        if n_features > n_samples:
            logger.warning('WARNING - PCA n_features ({}) > n_samples ({}). Effective dimensionality will be computed '
                           'using {} components!'.format(n_features, n_samples, min(n_samples, n_features)))
        pca_obj = sk.PCA(n_components=min(n_features, n_samples))
    if not hasattr(pca_obj, "explained_variance_ratio_"):
        pca_obj.fit(response_matrix.T)  # we need to transpose here as scipy requires n_samples X n_features
    # compute dimensionality
    dimensionality = 1. / np.sum((pca_obj.explained_variance_ratio_ ** 2))
    if display:
        logger.info("Effective dimensionality = {0}".format(str(round(dimensionality, 2))))
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    if plot:
        X = pca_obj.fit_transform(response_matrix.T).T
        plot_dimensionality(dimensionality, pca_obj, X, data_label=label, display=display,
                                                       save=save)
    return dimensionality


def compute_timescale(response_matrix, time_axis, max_lag=1000, method=0, plot=True, display=True, save=False,
                      verbose=True, n_procs=1):
    """
    Determines the time scale of fluctuations in the population activity.

    :param response_matrix: [np.array] with size NxT, continuous time
    :param time_axis:
    :param max_lag:
    :param method: based on autocorrelation (0) or on power spectra (1) - not implemented yet
    :param plot:
    :param display:
    :param save:
    :param n_procs: [int] number of processes used for parallelization
    :return: (tuple) final_acc, mean_fit, acc_function, time_scales
        final_acc: acc[:, :max_lag]
        mean_fit: a_fit, b_fit, tau_fit - result from least squares fitting of the exponential
        acc_function: simple exponential
        time_scales: list of the fitted time constants of the individual signals / neurons (tau_fit)

    """
    time_scales = []
    errors = []
    acc = cross_trial_cc(response_matrix, n_procs=n_procs)
    initial_guess = 1., 0., 10.
    for n_signal in range(acc.shape[0]):
        try:
            fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis[:max_lag], acc[n_signal, :max_lag], acc_function))
            a_fit, b_fit, tau_fit = fit  # expand fitted parameters from tuple

            if tau_fit > 0.1:
                # this may yield a NaN error! (but that's okay if fit was not possible)
                error_rates = np.sum((acc[n_signal, :max_lag] - acc_function(time_axis[:max_lag], *fit)) ** 2)
                if verbose:
                    logger.info("Timescale [ACC] = {0} ms / error = {1}".format(str(tau_fit), str(error_rates)))

                errors.append(error_rates)
                if np.isnan(error_rates):
                    time_scales.append(np.nan)
                else:
                    time_scales.append(tau_fit)
            elif 0. < tau_fit < 0.1:
                fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis[1:max_lag], acc[n_signal, 1:max_lag], acc_function))
                a_fit, b_fit, tau_fit = fit  # expand fitted parameters from tuple

                error_rates = np.sum((acc[n_signal, :max_lag] - acc_function(time_axis[:max_lag], *fit)) ** 2)
                if verbose:
                    logger.info("Timescale [ACC] = {0} ms / error = {1}".format(tau_fit, str(error_rates)))
                # time_scales.append(tau_fit)
                errors.append(error_rates)
                if np.isnan(error_rates):
                    time_scales.append(np.nan)
                else:
                    time_scales.append(tau_fit)
        except Exception as e:
            raise e
            continue
    final_acc = acc[:, :max_lag]

    mean_fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis[:max_lag], np.nanmean(final_acc, 0), acc_function))
    a_fit_mean, b_fit_mean, tau_fit_mean = mean_fit

    if tau_fit_mean < 0.1:
        mean_fit, _ = opt.leastsq(err_func, initial_guess,
                                  args=(time_axis[1:max_lag], np.nanmean(final_acc, 0)[1:max_lag], acc_function))

    error_rates = np.nansum((np.nanmean(final_acc, 0) - acc_function(time_axis[:max_lag], *mean_fit)) ** 2)
    logger.info("*******************************************")
    logger.info("Timescale = {0} ms / error = {1}".format(str(tau_fit_mean), str(error_rates)))
    logger.info("Accepted dimensions = {0}".format(str(float(final_acc.shape[0]) / float(acc.shape[0]))))

    if plot:
        plot_acc(time_axis[:max_lag], acc[:, :max_lag], mean_fit, acc_function, ax=None,
                                              display=display, save=save)

    return final_acc, mean_fit, acc_function, time_scales


def dimensionality_reduction(state_matrix, data_label='', labels=None, metric=None, standardize=True, plot=True,
                             colormap='jet', display=True, save=False):
    """
    Fit and test various algorithms, to extract a reasonable 3D projection of the data for visualization

    :param state_matrix: matrix to analyze (NxT)
    :param data_label:
    :param labels:
    :param metric: [str] metric to use (if None all will be tested)
    :param standardize:
    :param plot:
    :param colormap:
    :param display:
    :param save:
    :return:
    """
    # TODO extend and test - and include in the analyse_activity_dynamics function
    metrics = ['PCA', 'FA', 'LLE', 'IsoMap', 'Spectral', 'MDS', 't-SNE']
    if metric is not None:
        assert (metric in metrics), "Incorrect metric"
        metrics = [metric]

    if labels is None:
        raise TypeError("Please provide stimulus labels")
    else:
        n_elements = np.unique(labels)
    colors_map = plt_helper.get_cmap(N=len(n_elements), cmap=colormap)

    for met in metrics:
        if met == 'PCA':
            logger.info("Dimensionality reduction with Principal Component Analysis")
            t_start = time.time()
            pca_obj = sk.PCA(n_components=3)
            X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
                logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
            exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]

            if plot:
                fig1 = pl.figure()
                ax11 = fig1.add_subplot(111, projection='3d')
                ax11.set_xlabel(r'$PC_{1}$')
                ax11.set_ylabel(r'$PC_{2}$')
                ax11.set_zlabel(r'$PC_{3}$')
                fig1.suptitle(r'${0} - PCA (var = {1})$'.format(str(data_label), str(exp_var)))
                scatter_projections(X_r, labels, colors_map, ax=ax11)
                if save:
                    fig1.savefig(save + data_label + '_PCA.pdf')
                if display:
                    pl.show()
        elif met == 'FA':
            logger.info("Dimensionality reduction with Factor Analysis")
            t_start = time.time()
            fa2 = sk.FactorAnalysis(n_components=len(n_elements))
            state_fa = fa2.fit_transform(state_matrix.T)
            score = fa2.score(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s / Score (NLL): {1}".format(str(time.time() - t_start), str(score)))
            if plot:
                fig2 = pl.figure()
                fig2.suptitle(r'{0} - Factor Analysis'.format(str(data_label)))
                # print(state_fa[:3, :].shape)
                ax21 = fig2.add_subplot(111, projection='3d')
                scatter_projections(state_fa[:3, :].T, labels, colors_map, ax=ax21)
                if save:
                    fig2.savefig(save + data_label + '_FA.pdf')
                if display:
                    pl.show()
        elif met == 'LLE':
            logger.info("Dimensionality reduction with Locally Linear Embedding")
            if plot:
                fig3 = pl.figure()
                fig3.suptitle(r'{0} - Locally Linear Embedding'.format(str(data_label)))

            methods = ['standard', 'ltsa', 'hessian', 'modified']
            labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

            for i, method in enumerate(methods):
                t_start = time.time()
                fit_obj = man.LocallyLinearEmbedding(n_neighbors=199, n_components=3, eigen_solver='auto',
                                                     method=method, n_jobs=-1)
                Y = fit_obj.fit_transform(state_matrix.T)
                if display:
                    logger.info(
                        "\t{0} - {1} s / Reconstruction error = {2}".format(method, str(time.time() - t_start), str(
                            fit_obj.reconstruction_error_)))
                if plot:
                    ax = fig3.add_subplot(2, 2, i + 1, projection='3d')
                    ax.set_title(method)
                    scatter_projections(Y, labels, colors_map, ax=ax)
            if plot and save:
                fig3.savefig(save + data_label + '_LLE.pdf')
            if plot and display:
                pl.show(False)
        elif met == 'IsoMap':
            logger.info("Dimensionality reduction with IsoMap Embedding")
            t_start = time.time()
            iso_fit = man.Isomap(n_neighbors=199, n_components=3, eigen_solver='auto', path_method='auto',
                                 neighbors_algorithm='auto', n_jobs=-1)
            iso = iso_fit.fit_transform(state_matrix.T)
            score = iso_fit.reconstruction_error()
            if display:
                logger.info("Elapsed time: {0} s / Reconstruction error = {1}".format(str(time.time() - t_start),
                                                                                      str(score)))
            if plot:
                fig4 = pl.figure()
                fig4.suptitle(r'{0} - IsoMap Embedding'.format(str(data_label)))
                ax41 = fig4.add_subplot(111, projection='3d')
                scatter_projections(iso, labels, colors_map, ax=ax41)
                if save:
                    fig4.savefig(save + data_label + '_IsoMap.pdf')
                if display:
                    pl.show(False)
        elif met == 'Spectral':
            logger.info("Dimensionality reduction with Spectral Embedding")
            fig5 = pl.figure()
            fig5.suptitle(r'{0} - Spectral Embedding'.format(str(data_label)))

            affinities = ['nearest_neighbors', 'rbf']
            for i, n in enumerate(affinities):
                t_start = time.time()
                spec_fit = man.SpectralEmbedding(n_components=3, affinity=n, n_jobs=-1)
                spec = spec_fit.fit_transform(state_matrix.T)
                if display:
                    logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
                if plot:
                    ax = fig5.add_subplot(1, 2, i + 1, projection='3d')
                    # ax.set_title(n)
                    scatter_projections(spec, labels, colors_map, ax=ax)
                # pl.imshow(spec_fit.affinity_matrix_)
                if plot and save:
                    fig5.savefig(save + data_label + '_SE.pdf')
                if plot and display:
                    pl.show(False)
        elif met == 'MDS':
            logger.info("Dimensionality reduction with MultiDimensional Scaling")
            t_start = time.time()
            mds = man.MDS(n_components=3, n_jobs=-1)
            mds_fit = mds.fit_transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
            if plot:
                fig6 = pl.figure()
                fig6.suptitle(r'{0} - MultiDimensional Scaling'.format(str(data_label)))
                ax61 = fig6.add_subplot(111, projection='3d')
                scatter_projections(mds_fit, labels, colors_map, ax=ax61)
                if save:
                    fig6.savefig(save + data_label + '_MDS.pdf')
        elif met == 't-SNE':
            logger.info("Dimensionality reduction with t-SNE")
            t_start = time.time()
            tsne = man.TSNE(n_components=3, init='pca')
            tsne_emb = tsne.fit_transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
            if plot:
                fig7 = pl.figure()
                fig7.suptitle(r'{0} - t-SNE'.format(str(data_label)))
                ax71 = fig7.add_subplot(111, projection='3d')
                scatter_projections(tsne_emb, labels, colors_map, ax=ax71)
                if save:
                    fig7.savefig(save + data_label + '_t_SNE.pdf')
                if display:
                    pl.show(False)
        else:
            raise NotImplementedError("Metric {0} is not currently implemented".format(met))


def analyse_state_matrix(state_matrix, stim_labels=None, epochs=None, label='', plot=True, display=True, save=False):
    """
    Use PCA to peer into the population responses.

    :param state_matrix: state matrix X
    :param stim_labels: stimulus labels (if each sample corresponds to a unique label)
    :param epochs:
    :param label: data label
    :param plot:
    :param display:
    :param save:
    :return results: dimensionality results
    """
    if isinstance(state_matrix, signals.AnalogSignalList):
        state_matrix = state_matrix.as_array()
    assert (isinstance(state_matrix, np.ndarray)), "Activity matrix must be numpy array or signals.AnalogSignalList"
    results = {}

    pca_obj = sk.PCA(n_components=3)
    X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)

    if stim_labels is None:
        pca_obj = sk.PCA(n_components=min(state_matrix.shape))
        X = pca_obj.fit_transform(state_matrix.T)
        logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_[:3]))
        results.update({'dimensionality': compute_dimensionality(state_matrix, pca_obj=pca_obj, display=True)})
        if plot:
            plot_dimensionality(results['dimensionality'], pca_obj, X, data_label=label,
                                                           display=display, save=save)
        if epochs is not None:
            for epoch_label, epoch_time in list(epochs.items()):
                resp = state_matrix[:, int(epoch_time[0]):int(epoch_time[1])]
                results.update({epoch_label: {}})
                results[epoch_label].update(analyse_state_matrix(resp, epochs=None, label=epoch_label, plot=False,
                                                                 display=False, save=False))
    else:
        logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
        if not isinstance(stim_labels, dict):
            label_seq = np.array(list(iterate_obj_list(stim_labels)))
            n_elements = np.unique(label_seq)
            if plot:
                fig1 = pl.figure()
                ax1 = fig1.add_subplot(111)
                plot_matrix(state_matrix, stim_labels, ax=ax1, display=False, save=False)
                fig2 = pl.figure()
                fig2.clf()
                exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]
                fig2.suptitle(r'${0} - PCA (var = {1})$'.format(str(label), str(exp_var)),
                              fontsize=20)

                ax2 = fig2.add_subplot(111, projection='3d')
                colors_map = plt_helper.get_cmap(N=len(n_elements), cmap='Paired')
                ax2.set_xlabel(r'$PC_{1}$')
                ax2.set_ylabel(r'$PC_{2}$')
                ax2.set_zlabel(r'$PC_{3}$')

                ccs = [colors_map(ii) for ii in range(len(n_elements))]
                for color, index, lab in zip(ccs, n_elements, n_elements):
                    # locals()['sc_{0}'.format(str(index))] = \
                    ax2.scatter(X_r[np.where(np.array(list(itertools.chain(
                        label_seq))) == index)[0], 0], X_r[np.where(
                        np.array(list(itertools.chain(label_seq))) == index)[
                        0], 1], X_r[np.where(
                        np.array(list(itertools.chain(label_seq))) == index)[0], 2],
                                s=50, c=color, label=lab)
                # scatters = [locals()['sc_{0}'.format(str(ind))] for ind in n_elements]
                # pl.legend(tuple(scatters), tuple(n_elements))
                pl.legend(loc=0)  # , handles=scatters)

                if display:
                    pl.show(block=False)
                if save:
                    fig1.savefig(save + 'state_matrix_{0}.pdf'.format(label))
                    fig2.savefig(save + 'pca_representation_{0}.pdf'.format(label))
        else:
            if plot:
                fig1 = pl.figure()
                ax = fig1.add_subplot(111, projection='3d')
                ax.plot(X_r[:, 0], X_r[:, 1], X_r[:, 2], color='r', lw=2)
                ax.set_title(label + r'$ - (3PCs) $= {0}$'.format(
                    str(round(np.sum(pca_obj.explained_variance_ratio_[:3]), 1))))
                ax.grid()
                if display:
                    pl.show(False)
                if save:
                    fig1.savefig(save + 'pca_representation_{0}.pdf'.format(label))
    return results
