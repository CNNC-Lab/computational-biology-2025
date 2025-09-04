import numpy as np

from tools.analysis import signals


def convert_array(array, id_list, dt=None, start=None, stop=None):
    """
    Convert a numpy array into an AnalogSignalList object
    :param array: NxT numpy array
    :param id_list:
    :param start:
    :param stop:
    :return:
    """
    assert (isinstance(array, np.ndarray)), "Provide a numpy array as input"

    if start is not None and stop is not None and dt is not None:
        # time_axis = np.arange(start, stop, dt)
        tmp = []
        for idd in range(array.shape[1]):
            for m_id, n_id in enumerate(id_list):
                tmp.append((n_id, array[m_id, idd]))
        new_AnalogSignalList = signals.AnalogSignalList(tmp, id_list, dt=dt, t_start=start, t_stop=stop)
    else:
        new_AnalogSignalList = signals.AnalogSignalList([], [], dt=dt, t_start=start, t_stop=stop,
                                                dims=len(id_list))

        for n, id in enumerate(np.sort(id_list)):
            try:
                id_signal = signals.AnalogSignal(array[n, :], dt)
                new_AnalogSignalList.append(id, id_signal)
            except Exception:
                print("id %d is not in the source AnalogSignalList" % id)
    return new_AnalogSignalList


# def gather_analog_activity(parameter_set, net, t_start=None, t_stop=None):
#     """
#     Retrieve all analog activity data recorded in [t_start, t_stop]
#     :param parameter_set: global ParameterSet
#     :param net: Network object
#     :param t_start: start time
#     :param t_stop: stop time
#     :return results: organized dictionary with all analogs (can be very large!)
#     """
#     results = {}
#     for pop_n, pop in enumerate(net.populations):
#         results.update({pop.name: {}})
#         if pop.name[-5:] == 'clone':
#             pop_name = pop.name[:-6]
#         else:
#             pop_name = pop.name
#         pop_idx = parameter_set.net_pars.pop_names.index(pop_name)
#         if parameter_set.net_pars.analog_device_pars[pop_idx] is None:
#             break
#         variable_names = pop.analog_activity_names
#
#         if not pop.analog_activity:
#             results[pop.name]['recorded_neurons'] = []
#             break
#         elif isinstance(pop.analog_activity, list):
#             for idx, nn in enumerate(pop.analog_activity_names):
#                 locals()[nn] = pop.analog_activity[idx]
#                 assert isinstance(locals()[nn], AnalogSignalList), "Analog Activity should be AnalogSignalList"
#         else:
#             locals()[pop.analog_activity_names[0]] = pop.analog_activity
#
#         reversals = []
#         single_idx = np.random.permutation(locals()[pop.analog_activity_names[0]].id_list())[0]
#         results[pop.name]['recorded_neurons'] = locals()[pop.analog_activity_names[0]].id_list()
#
#         for idx, nn in enumerate(pop.analog_activity_names):
#             if (t_start is not None) and (t_stop is not None):
#                 locals()[nn] = locals()[nn].time_slice(t_start, t_stop)
#
#             time_axis = locals()[nn].time_axis()
#
#             if 'E_{0}'.format(nn[-2:]) in parameter_set.net_pars.neuron_pars[pop_idx]:
#                 reversals.append(parameter_set.net_pars.neuron_pars[pop_idx]['E_{0}'.format(nn[-2:])])
#
#             results[pop.name]['{0}'.format(nn)] = locals()[nn].as_array()
#
#     return results


def pad_array(input_array, add=10):
    """
    Pads an array with zeros along the time dimension

    :param input_array:
    :param add:
    :return:
    """
    new_shape = (input_array.shape[0], input_array.shape[1] + add)
    new_size = (new_shape[0]) * (new_shape[1])
    zero_array = np.zeros(new_size).reshape(new_shape)
    zero_array[:input_array.shape[0], :input_array.shape[1]] = input_array
    return zero_array


def make_simple_kernel(shape, width=3, height=1., resolution=1., normalize=False, **kwargs):
    """
    Simplest way to create a smoothing kernel for 1D convolution
    :param shape: {'box', 'exp', 'alpha', 'double_exp', 'gauss'}
    :param width: kernel width
    :param height: peak amplitude of the kernel
    :param resolution: time step
    :param normalize: [bool]
    :return: kernel k
    """
    # TODO load external kernel...
    x = np.arange(0., (width / resolution) + resolution, 1.)  # resolution)

    if shape == 'box':
        k = np.ones_like(x) * height

    elif shape == 'exp':
        assert 'tau' in kwargs, "for exponential kernel, please specify tau"
        tau = kwargs['tau']
        k = np.exp(-x / tau) * height

    elif shape == 'double_exp':
        assert ('tau_1' in kwargs), "for double exponential kernel, please specify tau_1"
        assert ('tau_2' in kwargs), "for double exponential kernel, please specify tau_2"

        tau_1 = kwargs['tau_1']
        tau_2 = kwargs['tau_2']
        tmp_k = (-np.exp(-x / tau_1) + np.exp(-x / tau_2))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'alpha':
        assert ('tau' in kwargs), "for alpha kernel, please specify tau"

        tau = kwargs['tau']
        tmp_k = ((x / tau) * np.exp(-x / tau))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'gauss':
        assert ('mu' in kwargs), "for Gaussian kernel, please specify mu"
        assert ('sigma' in kwargs), "for Gaussian kernel, please specify sigma"

        sigma = kwargs['sigma']
        mu = kwargs['mu']
        tmp_k = (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(- ((x - mu) ** 2. / (2. * (sigma ** 2.))))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'tri':
        halfwidth = width / 2
        trileft = np.arange(1, halfwidth + 2)
        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
        k = np.append(trileft, triright)
        k += height

    elif shape == 'sin':
        k = np.sin(2 * np.pi * x / width * kwargs['frequency'] + kwargs['phase_shift']) * height
        k += kwargs['mean_amplitude']
    else:
        print("Kernel not implemented, please choose {'box', 'exp', 'alpha', 'double_exp', 'gauss', 'tri', "
                       "'syn'}")
        k = 0
    if normalize:
        k /= k.sum()

    return k
