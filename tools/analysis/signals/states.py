import itertools
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# import conic.utils
# import conic.utils.analyzer.metrics.states

from tools.analysis.metrics import matrices, states
import tools.visualization as viz
from tools.analysis import signals
from tools.logger import data_io as data_handling
from tools.operations import iterate_obj_list

from tools.logger import info
logger = info.get_logger(__name__)


class StateMatrix(object):
    """
    Class to store a state/response matrix of a given population, together with some additional metadata.
    Independent of the underlying network type.
    """

    def __init__(self, matrix, label, state_var, population, sampled_times=None, dataset_label=None, stim_labels=None,
                 standardize=False, save=False):
        """
        :param matrix:
        :param label:
        :param state_var: str - name of state variable
        :param population: str - name of population it corresponds to
        :param sampled_times: str - timing of recordings
        :param dataset_label: str
        :param stim_labels: list - if there is a 1:1 correspondence between states and stimulus labels
        :param standardize: bool
        :param save:
        """
        self.matrix = matrix
        self.state_var = state_var
        self.sampled_times = sampled_times  # stores the exact sampling times, if applicable
        self.method = None
        self.population = population
        self.label = label
        self.stim_labels = stim_labels
        self.pdist = None
        self.dataset_label = dataset_label

        if standardize:
            self.standardize()

        if save:
            self.save(dataset_label)

    def standardize(self):
        self.matrix = StandardScaler().fit_transform(self.matrix.T).T

    def analyse(self, states=None, epochs=None, plot=True, display=True, save=False):
        """
        Analyse state matrix
        """
        logger.info("Analyzing state matrix {} [{}-{}]".format(self.label, self.state_var, self.population))
        if states is not None:
            state_matrix = states
        else:
            state_matrix = self.matrix
        if isinstance(state_matrix, signals.AnalogSignalList):
            state_matrix = state_matrix.as_array()
        assert (isinstance(state_matrix, np.ndarray)), "State matrix must be numpy array or signals.AnalogSignalList"
        results = {}

        if plot and save:
            fig_labels = []

        # state complexity
        self.pdist = self._state_distance()
        results.update({'complexity': (np.mean(self.pdist), np.std(self.pdist))})
        logger.info("- Complexity: {}".format(results['complexity']))
        if plot:
            # plot state distance matrix
            # plot state distance distribution
            pass

        if self.stim_labels is None:
            # effective dimensionality
            pca_obj = PCA(n_components=min(state_matrix.shape))
            X = pca_obj.fit_transform(state_matrix.T)
            logger.info("- Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_[:3]))
            if save:
                results.update({'dimensionality': self.effective_dimensionality(plot=plot, display=display,
                                                                                save=save + '{'
                                                                                            '}-dimensionality.pdf'.format(
                                                                                    self.label))})
            else:
                results.update({'dimensionality': self.effective_dimensionality(plot=plot, display=display, save=save)})
            logger.info("- Effective dimensionality: {}".format(results['dimensionality']))
            if epochs is not None:
                for epoch_label, epoch_time in list(epochs.items()):
                    resp = state_matrix[:, int(epoch_time[0]):int(epoch_time[1])]
                    results.update({epoch_label: {}})
                    results[epoch_label].update(self.analyse(resp, epochs=None, label=epoch_label, plot=False,
                                                             display=False, save=False))
        else:
            pca_obj = PCA(n_components=3)
            X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)
            logger.info("- Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
            if not isinstance(self.stim_labels, dict):
                label_seq = np.array(list(iterate_obj_list(self.stim_labels)))
                n_elements = np.unique(label_seq)
                if plot:
                    fig1, ax1 = plt.subplots()
                    viz.base.plot_matrix(state_matrix, self.stim_labels, ax=ax1, display=False, save=False)
                    fig2 = plt.figure()
                    fig2.clf()
                    exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]
                    fig2.suptitle(r'${0} - PCA (var = {1})$'.format(str(self.label), str(exp_var)),
                                  fontsize=20)
                    ax2 = fig2.add_subplot(111, projection='3d')
                    colors_map = viz.helper.get_cmap(N=len(n_elements), cmap='Paired')
                    ax2.set_xlabel(r'$PC_{1}$')
                    ax2.set_ylabel(r'$PC_{2}$')
                    ax2.set_zlabel(r'$PC_{3}$')

                    ccs = [colors_map(ii) for ii in range(len(n_elements))]
                    for color, index, lab in zip(ccs, n_elements, n_elements):
                        ax2.scatter(X_r[np.where(np.array(list(itertools.chain(
                            label_seq))) == index)[0], 0], X_r[np.where(
                            np.array(list(itertools.chain(label_seq))) == index)[
                            0], 1], X_r[np.where(
                            np.array(list(itertools.chain(label_seq))) == index)[0], 2],
                                    s=50, c=color, label=lab)
                    plt.legend(loc=0)

                    if display:
                        plt.show(block=False)
                    if save:
                        fig1.savefig(save + 'state_matrix_{0}.pdf'.format(self.label))
                        fig2.savefig(save + 'pca_representation_{0}.pdf'.format(self.label))
            else:
                if plot:
                    fig1 = plt.figure()
                    ax = fig1.add_subplot(111, projection='3d')
                    ax.plot(X_r[:, 0], X_r[:, 1], X_r[:, 2], color='r', lw=2)
                    ax.set_title(self.label + r'$ - (3PCs) $= {0}$'.format(
                        str(round(np.sum(pca_obj.explained_variance_ratio_[:3]), 1))))
                    ax.grid()
                    if display:
                        plt.show(False)
                    if save:
                        fig1.savefig(save + 'pca_representation_{0}.pdf'.format(self.label))
        return results

    def effective_dimensionality(self, plot=True, display=True, save=False):
        """
        Measure the effective dimensionality of population responses. Based on Abbott et al. (2001). Interactions between
        intrinsic and stimulus-evoked activity in recurrent neural networks.
        """
        if display:
            logger.info("Determining effective dimensionality of states {}".format(self.label))
            t_start = time.time()

        n_features, n_samples = np.shape(self.matrix)  # features = neurons
        if n_features > n_samples:
            logger.warning('WARNING - PCA n_features ({}) > n_samples ({}). Effective dimensionality will be computed '
                           'using {} components!'.format(n_features, n_samples, min(n_samples, n_features)))
        pca_obj = PCA(n_components=min(n_features, n_samples))
        if not hasattr(pca_obj, "explained_variance_ratio_"):
            pca_obj.fit(self.matrix.T)  # we need to transpose here as sklearn requires n_samples X n_features
        # compute dimensionality
        dimensionality = 1. / np.sum((pca_obj.explained_variance_ratio_ ** 2))
        if display:
            logger.info("- Effective dimensionality = {0}".format(str(round(dimensionality, 2))))
            logger.info("- Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
        if plot:
            X = pca_obj.fit_transform(self.matrix.T).T
            viz.states.plot_dimensionality(dimensionality, pca_obj, X, data_label=self.label, display=display, save=save)
        return dimensionality

    def rank(self):
        """
		Calculates the rank of all state matrices.
		:return:
		"""
        return np.linalg.matrix_rank(self.matrix)

    def autocorrelation_timescale(self, time_axis, max_lag=1000, plot=True, display=True, save=False, verbose=False):
        """
        Determines the intrinsic time scale of population activity.
        :return:
        """
        pass  # check Fahad's implementation

    def reduce_dimensionality(self, labels=None, method=None, plot=True, colormap='winter', display=True, save=False):
        """
        Fit and test various dimensionality algorithms, to extract a reasonable 3D projection of the data for
        visualization
        :param labels:
        :param method:
        :param plot:
        :param colormap:
        :param display:
        :param save:
        :return:
        """
        pass

    def plot_sample_traces(self, n_neurons=10, time_axis=None, analysis_interval=None, display=True, save=False):
        """
        Plot a sample of the neuron responses
        :param n_neurons:
        :param time_axis:
        :param analysis_interval:
        :return:
        """
        if time_axis is None:
            time_axis = np.arange(self.matrix.shape[1])
        if analysis_interval is None:
            analysis_interval = [time_axis.min(), time_axis.max()]
        try:
            t_idx = [np.where(time_axis == analysis_interval[0])[0][0],
                     np.where(time_axis == analysis_interval[1])[0][0]]
        except:
            raise IOError("Analysis interval bounds not found in time axis")

        fig, axes = plt.subplots(n_neurons, 1, sharex=True, figsize=(10, 2 * n_neurons))
        fig.suptitle("{} [{} state] - {}".format(self.population, self.state_var, self.label))
        neuron_idx = np.random.permutation(self.matrix.shape[0])[:n_neurons]
        for idx, (neuron, ax) in enumerate(zip(neuron_idx, axes)):
            ax.plot(time_axis[t_idx[0]:t_idx[1]], self.matrix[neuron, t_idx[0]:t_idx[1]])
            ax.set_ylabel(r"$X_{" + "{0}".format(neuron) + "}$")
            ax.set_xlim(analysis_interval)
            if idx == len(neuron_idx) - 1:
                ax.set_xlabel("Time [ms]")
        viz.helper.fig_output(fig, display, save)

    def plot_matrix(self, time_axis=None, display=True, save=False):
        """
        Plot the complete state matrix
        :param time_axis:
        :return:
        """
        if time_axis is None:
            time_axis = np.arange(self.matrix.shape[1])
        fig, (ax11, ax12) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [18, 4], 'hspace': 0.1}, sharex=False)
        fig.suptitle("{} [{} states] - {}".format(self.population, self.state_var, self.label))
        _, ax11 = viz.base.plot_matrix(self.matrix, ax=ax11, save=False, display=False, data_label=None)
        ax11.set_ylabel('Neuron')
        ax12.plot(time_axis, self.matrix.mean(0), lw=2)
        divider2 = make_axes_locatable(ax12)
        cax2 = divider2.append_axes("right", size="5%", pad="4%")
        cax2.remove()
        ax12.set_xlabel("Time [ms]")
        ax12.set_xlim([time_axis.min(), time_axis.max()])
        ax12.set_ylabel(r"$\bar{X}$")
        viz.helper.fig_output(fig, display, save)

    def plot_trajectory(self, display=True, save=False):
        fig3 = plt.figure()
        ax31 = fig3.add_subplot(111, projection='3d')
        effective_dimensionality = states.analyse_state_matrix(self.matrix, stim_labels=None, epochs=None, label=None,
                                                                                            plot=False,
                                                                                            display=True, save=False)['dimensionality']
        fig3.suptitle("{} [{} states] - {}".format(self.population, self.state_var, self.label) +
                      r" $\lambda_{\mathrm{eff}}=" + "{}".format(effective_dimensionality) + "$")
        viz.states.plot_trajectory(self.matrix, label="{} [{} states] - {}".format(self.population,
                                                                                                   self.state_var, self.label) +
                                                        " Trajectory", ax=ax31, color='k',
                                                   display=False, save=False)
        viz.helper.fig_output(fig3, display, save)

    def _state_distance(self):
        return matrices.pairwise_dist(self.matrix)

    def state_density(self, display=True, save=False):
        if self.pdist is None:
            self.pdist = self._state_distance()

        # density of first 2 PCs
        pca_obj = PCA(n_components=2)
        X_r = pca_obj.fit(self.matrix.T).transform(self.matrix.T).T
        xy = np.vstack([X_r[0, :], X_r[1, :]])
        z = st.gaussian_kde(xy)(xy)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle("{} - PC density".format(self.label))
        scatter1 = ax.scatter(X_r[0, :], X_r[1, :], c=z, s=50, edgecolors=None, cmap="viridis")
        fig.colorbar(scatter1, ax=ax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if save:
            viz.helper.fig_output(fig, display, save=save)

    def save(self, dataset_label):
        """
        Save the StateMatrix object. By default only the test matrix is stored, but it's
        also possible to store the intermediate ones for each batch.

        :return:
        """
        try:
            filename = "{}_{}_{}_{}.pkl".format(data_handling.filename_prefixes['state_matrix'], self.label,
                                                data_handling.data_label, dataset_label)
            data_handling.save_pkl_object(self, os.path.join(data_handling.paths['activity'], filename))
        except Exception as e:
            logger.warning("Could not save StateMatrix {}, storage paths not set?".format(self.label))
