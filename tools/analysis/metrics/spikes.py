import itertools
import time
import numpy as np
import importlib

from tools.logger import info
from tools.analysis import signals

has_pyspike = False if importlib.util.find_spec("pyspike") is None else importlib.import_module("pyspike")
logger = info.get_logger(__name__)
np.seterr(all='ignore')


def get_total_counts(spike_list, time_bin=50.):
	"""
	Determines the total spike counts for neurons with consecutive nonzero counts in bins of the specified size
	:param spike_list: SpikeList object
	:param time_bin: bin width
	:return ctr: number of neurons complying
	:return total_counts: spike count array
	"""
	assert isinstance(spike_list, signals.spikes.SpikeList), "Input must be SpikeList object"

	total_counts = []
	ctr = 0
	neuron_ids = []
	for n_train in spike_list.spiketrains:
		tmp = spike_list.spiketrains[n_train].time_histogram(time_bin=time_bin, normalized=False, binary=True)
		if np.mean(tmp) == 1:
			neuron_ids.append(n_train)
			ctr += 1
	logger.info("{0} neurons have nonzero spike counts in bins of size {1}".format(str(ctr), str(time_bin)))
	total_counts1 = []
	for n_id in neuron_ids:
		counts = spike_list.spiketrains[n_id].time_histogram(time_bin=time_bin, normalized=False, binary=False)
		total_counts1.append(counts)
	total_counts.append(total_counts1)
	total_counts = np.array(list(itertools.chain(*total_counts)))

	return neuron_ids, total_counts


def compute_spikelist_metrics(spike_list, label, analysis_pars):
	"""
	Computes the ISI, firing activity and synchrony statistics for a given spike list.

	:param spike_list: SpikeList object for which the statistics are computed
	:param label: (string) population name or something else
	:param analysis_pars: ParameterSet object containing the analysis parameters

	:return: dictionary with the results for the given label, with statistics as (sub)keys
	"""
	ap = analysis_pars
	pars_activity = ap.population_activity
	results = {label: {}}

	results[label].update(compute_isi_stats(spike_list, summary_only=bool(ap.depth % 2 != 0)))

	# Firing activity statistics
	results[label].update(compute_spike_stats(spike_list, time_bin=pars_activity.time_bin,
	                                          summary_only=bool(ap.depth % 2 != 0)))

	# Synchrony measures
	if not spike_list.empty():
		if len(np.nonzero(spike_list.mean_rates())[0]) > 10:
			results[label].update(compute_synchrony(spike_list, n_pairs=pars_activity.n_pairs,
			                                        time_bin=pars_activity.time_bin, tau=pars_activity.tau,
			                                        time_resolved=pars_activity.time_resolved, depth=ap.depth))
	return results


def compute_isi_stats(spike_list, summary_only=True):
	"""
	Compute all relevant isi metrics
	:param spike_list: SpikeList object
	:param summary_only: bool - store only the summary statistics or all the data (memory!)
	:return: dictionary with all the relevant data
	"""
	logger.info("Analysing inter-spike intervals...")
	t_start = time.time()
	results = dict()

	results['cvs'] = spike_list.cv_isi(float_only=True)
	results['lvs'] = spike_list.local_variation()
	results['lvRs'] = spike_list.local_variation_revised(float_only=True)
	results['ents'] = spike_list.isi_entropy(float_only=True)
	results['iR'] = spike_list.instantaneous_regularity(float_only=True)
	results['cvs_log'] = spike_list.cv_log_isi(float_only=True)
	results['isi_5p'] = spike_list.isi_5p(float_only=True)
	results['ai'] = spike_list.adaptation_index(float_only=True)

	if not summary_only:
		results['isi'] = np.array(list(itertools.chain(*spike_list.isi())))
	else:
		results['isi'] = []
		cvs = results['cvs']
		lvs = results['lvs']
		lvRs = results['lvRs']
		H = results['ents']
		iRs = results['iR']
		cvs_log = results['cvs_log']
		isi_5p = results['isi_5p']
		ai = results['ai']

		results['cvs'] = (np.mean(cvs), np.var(cvs))
		results['lvs'] = (np.mean(lvs), np.var(lvs))
		results['lvRs'] = (np.mean(lvRs), np.var(lvRs))
		results['ents'] = (np.mean(H), np.var(H))
		results['iR'] = (np.mean(iRs), np.var(iRs))
		results['cvs_log'] = (np.mean(cvs_log), np.var(cvs_log))
		results['isi_5p'] = (np.mean(isi_5p), np.var(isi_5p))
		results['ai'] = (np.mean(ai), np.var(ai))

	logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))

	return results


def compute_spike_stats(spike_list, time_bin=50., summary_only=False, display=False):
	"""
	Compute relevant statistics on population firing activity (f. rates, spike counts)
	:param spike_list: SpikeList object
	:param time_bin: float - bin width to determine spike counts
	:param summary_only: bool - store only the summary statistics or all the data (memory!)
	:param display: bool - display progress / time
	:return: dictionary with all the relevant data
	"""
	if display:
		logger.info("\nAnalysing spiking activity...")
		t_start = time.time()
	results = {}
	rates = np.array(spike_list.mean_rates())
	rates = rates[~np.isnan(rates)]
	counts = spike_list.spike_counts(dt=time_bin, normalized=False, binary=False)
	ffs = np.array(spike_list.fano_factors(time_bin))
	if summary_only:
		results['counts'] = (np.mean(counts[~np.isnan(counts)]), np.var(counts[~np.isnan(counts)]))
		results['mean_rates'] = (np.mean(rates), np.var(rates))
		results['ffs'] = (np.mean(ffs[~np.isnan(ffs)]), np.var(ffs[~np.isnan(ffs)]))
		results['corrected_rates'] = (np.mean(rates[np.nonzero(rates)[0]]), np.std(rates[np.nonzero(rates)[0]]))
	else:
		results['counts'] = counts
		results['mean_rates'] = rates
		results['corrected_rates'] = rates[np.nonzero(rates)[0]]
		results['ffs'] = ffs[~np.isnan(ffs)]
		results['spiking_neurons'] = spike_list.id_list
	if display:
		logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
	return results


def compute_synchrony(spike_list, n_pairs=500, time_bin=1., tau=20., time_resolved=False, display=True, depth=4):
	"""
	Apply various metrics of spike train synchrony
	Note: Has dependency on PySpike package.

	:param spike_list: SpikeList object
	:param n_pairs: number of neuronal pairs to consider in the pairwise correlation measures
	:param time_bin: time_bin (for pairwise correlations)
	:param tau: time constant (for the van Rossum distance)
	:param time_resolved: bool - perform time-resolved synchrony analysis (PySpike)
	:param summary_only: bool - retrieve only a summary of the results
	:param complete: bool - use all metrics or only the ccs (due to computation time, memory)
	:param display: bool - display elapsed time message
	:return results: dict
	"""
	if display:
		logger.info("\nAnalysing spike synchrony...")
		t_start = time.time()

	results = dict()

	if depth == 1 or depth == 3:
		results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=False)
		# ccs = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)
		# results['ccs'] = (np.mean(ccs), np.var(ccs))

		if depth >= 3:
			results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
			results['d_vr'] = np.mean(spike_list.distance_van_rossum(tau=tau))
	else:
		results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=True)
		# results['ccs'] = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)

		if depth >= 3:
			results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
			results['d_vr'] = spike_list.distance_van_rossum(tau=tau)

	if display:
		logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
	return results