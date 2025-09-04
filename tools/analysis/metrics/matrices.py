import numpy as np
from tqdm import tqdm
from sklearn import metrics as met
from pathos.multiprocessing import ProcessPool as ThreadPool
import time

from .timeseries import autocorrelation_function
from tools.logger import info
# from conic_tools.visualization import helper as vis_helper
logger = info.get_logger(__name__)


def pairwise_dist(matrix):
    return np.tril(met.pairwise_distances(matrix))


def cross_trial_cc(total_counts, display=True, n_procs=1):
    """
    Computes the autocorrelations for a N x T matrix.

    :param total_counts: [nd.array] N x T matrix containing the signals
        1. N is the number of neurons / units and T is the number of samples (time points or trials)
        2. for cross-trial ACC of a single neuron, N is the number of time points, and T is the number of trials (?)

    :param display: [bool] display progress and time consumption
    :param n_procs: [int] number of threads (processes) to use for parallelization

    :return: Matrix of size N X T (unchanged), where the rows correspond to a signal (neuron),
            and columns to the autocorrelation at different time lags
    """
    if display:
        logger.info("Computing autocorrelations..")
    units = total_counts.shape[0]
    t = time.time()

    def _calc_acc_interval(interval_dict):
        results_acc = []
        interval = interval_dict['interval_key']
        for idx, neuron_id in enumerate(interval):
            # if display and interval[0] == 0:
            #     vis_helper.progress_bar(float(neuron_id) / float(len(interval)))
            neuron_acc_res = autocorrelation_function(total_counts[neuron_id, :])  # compute acc for single neuron
            # if not np.isnan(np.mean(neuron_acc_res)):
            #     results_acc.append(neuron_acc_res)  # add to result list, will be converted to matrix in the end1

            # Store result, irrespective of whether is correct or np.nan
            results_acc.append(neuron_acc_res)  # add to result list, will be converted to matrix in the end1
        return np.array(results_acc)

    thread_args = [{'interval_key': interval} for interval in np.array_split(np.array(range(units)), n_procs)]
    logger.info(f"\tMultithreaded autocorrelation computation with #{len(thread_args)} threads...")

    pool = ThreadPool(len(thread_args))
    pool_res = pool.map(_calc_acc_interval, thread_args)
    if display:
        logger.info(f"... elapsed time: {time.time() - t} s")

    acc_concatenated = np.concatenate(tuple(pool_res))

    # pool.join()
    # pool.close()
    return acc_concatenated
