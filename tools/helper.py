import numpy as np
from sklearn.linear_model import LinearRegression
import time
from tqdm import tqdm
from multiprocessing import Pool

########################################################################################################
def generate_input(seed, num_steps, step_duration, resolution, scale):
    """
    Generates a piecewise constant input signal with amplitudes drawn from a uniform distribution
    
    Parameters
    ----------
    seed: int
        seed to generate the input signal
    num_steps: int
        number of steps in the step function
    step_duration: int
        duration of each step in ms
    resolution: float
        resolution of the generated signal
    scale: float
        amplitude scaling parameter
    Returns
    -------
    ndarray
        continuous input signal (between -1 and 1)
    ndarray
        time vector with all the times for which the signal is generated
    ndarray
        times at which signal amplitude shifts
    ndarray
        amplitudes
    """
    dist_range = [-1., 1.]
    rand_distr = np.random.uniform(low=dist_range[0], high=dist_range[1], size=num_steps)
    rand_distr = rand_distr + abs(min(dist_range))
    inp_times = np.arange(resolution, num_steps * step_duration, step_duration)
    time_vec = np.arange(0, num_steps * step_duration + resolution, resolution)
    signal = np.zeros_like(time_vec)
    for tt in range(len(inp_times)):
        end_idx = int(round(inp_times[tt + 1] / resolution)) if tt + 1 < len(inp_times) else None
        signal[int(round(inp_times[tt] / resolution)):end_idx] = rand_distr[tt] 

    return signal, time_vec, inp_times, rand_distr * scale

############################################################################################################################
def filter_spikes(spike_times, neuron_ids, nNeurons, t_start, t_stop, dt, tau):
    """
    Returns an NxT matrix where each row represents the filtered spiking activity of
    one neuron and the columns represent time...

    Inputs:
        - spike_times - list of spike times
        - neuron_ids - list of spike ids
        - dt - time step
        - tau - kernel time constant
    """

    neurons = np.unique(neuron_ids)
    new_ids = neuron_ids - min(neuron_ids)
    N = round((t_stop - t_start) / dt)
    StateMat = np.zeros((int(nNeurons), int(N)))

    for i, n in enumerate(tqdm(neurons, desc='Filtering SpikeTrains')):
        idx = np.where(neuron_ids == n)[0]
        spk_times = spike_times[idx]
        StateMat[new_ids[idx][0], :] = spikes_to_states(spk_times, t_start, t_stop, dt, tau)

    return StateMat

#######################################################################################
def spikes_to_states(spike_times, t_start, t_stop, dt, tau):
    """
    Converts a spike train into an analogue variable, by convolving it with an exponential kernel.
    This process is supposed to mimic the integration performed by the
    postsynaptic membrane upon an incoming spike.

    Inputs:
        spike_times - array of spike times for a single neuron
        dt     - time step
        tau    - decay time constant
    Examples:
    >> spikes_to_states(spk_times, 0.1, 20.)
    """

    nSpk = len(spike_times)
    state = 0.
    N = round((t_stop - t_start) / dt)

    States = np.zeros((1, int(N)))[0]

    TimeVec = np.round(np.arange(t_start, t_stop, dt), 1)
    decay = np.exp(-dt / tau)

    if nSpk:
        idx_Spk = 0
        SpkT = spike_times[idx_Spk]

        for i, t in enumerate(TimeVec):
            if (np.round(SpkT, 1) == np.round(t, 1)):  # and (idx_Spk<nSpk-1):
                state += 1.
                if (idx_Spk < nSpk - 1):
                    idx_Spk += 1
                    SpkT = spike_times[idx_Spk]
            else:
                state = state * decay
            if i < int(N):
                States[i] = state

    return States

#######################################################################################
def filter_spikes_parallel(spike_times, neuron_ids, n_neurons, t_start, t_stop, dt, tau, n_processes):
    """
    Returns an NxT matrix where each row represents the filtered spiking activity of
    one neuron and the columns represent time...

    Inputs:
        - spike_times - list of spike times
        - neuron_ids - list of spike ids
        - dt - time step
        - tau - kernel time constant
        - n_processes - number of processes to use for parallel computation
        - show_progess - if True a progress bar is printed
    """
    spk_times_list = order_array_by_ids(spike_times, n_neurons, neuron_ids)
    arg_list = [{'spike_times': spkt, 't_start': t_start, 't_stop': t_stop, 'dt': dt, 'tau': tau} for spkt in
                spk_times_list]
    with Pool(n_processes) as p:
        state_mat = list(
                tqdm(p.imap(spikes_to_states_from_dict, arg_list), desc='Filtering SpikeTrains', total=n_neurons))
    return np.array(state_mat)


#######################################################################################
def spikes_to_states_from_dict(args):
    """
    Helper function to use multiprocessing for filtering the spikes
    """
    return spikes_to_states(args['spike_times'], args['t_start'], args['t_stop'], args['dt'], args['tau'])

def order_array_by_ids(array_to_order, n_possible_ids, ids):
    """
    Orders an array (for example spike trains of neurons) by the given ids (of the neurons).
    Needs the number of possible (neuron) ids, because some ids could be missing (neurons may not have
    fired), but they should be in the resulting list as well.

    Parameters
    ----------
    array_to_order: ndarray of floats
        ndarray with spike times
    n_possible_ids: int
        number of possible ids
    ids: ndarray of int
        ids of the objects to which the elements in the array_to_order belong

    Returns
    -------
    list of ndarrays
        list of spike trains (ndarrays) for each neuron

    Examples
    --------
    >>> spike_times = np.array([10.2, 20.1, 30.1])
    >>> ids = np.array([2, 1, 1])
    >>> order_array_by_ids(spike_times, 3, ids)
    [array([20.1, 30.1]), array([10.2]), array([], dtype=float64)]
    """
    spk_times_list = [np.array([]) for _ in range(n_possible_ids)]
    neurons = np.unique(ids)
    new_ids = ids - min(ids)

    for i, n in enumerate(neurons):
        idx = np.where(ids == n)[0]
        spk_times_list[new_ids[idx[0]]] = array_to_order[idx]

    return spk_times_list

############################################################################################################
def compute_capacity(X, z):
    """
    Compute capacity to reconstruct z based on linearly combining x
    :param X: state matrix (NxT)
    :param z: target output (1xT)
    :return: capacity
    """
    t_start = time.time()
    reg = LinearRegression(n_jobs=-1, fit_intercept=False, copy_X=False).fit(X.T, z)
    z_hat = reg.predict(X.T)
    covs = (np.cov(z_hat, z)[0, 1] ** 2.)
    vars = (np.var(z) * np.var(z_hat))
    capacity = covs / vars
    error = np.mean((z - z_hat) ** 2)
    norm = np.linalg.norm(reg.coef_)
    print("\nElapsed time: {} s".format(time.time() - t_start))

    return z_hat, capacity, error, norm