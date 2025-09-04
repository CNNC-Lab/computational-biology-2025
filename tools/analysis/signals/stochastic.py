import time
import numpy as np

from tools.logger import info
from tools.analysis.signals import SpikeTrain, AnalogSignal

logger = info.get_logger(__name__)


class StochasticGenerator(object):
    """
    Stochastic process generator
    ============================
    (adapted from NeuroTools)

    Generate stochastic processes of various types and return them as SpikeTrain or AnalogSignal objects.

    Implemented types:
    ------------------
    a) Spiking Point Process - poisson_generator, inh_poisson_generator, gamma_generator, !!inh_gamma_generator!!,
    inh_adaptingmarkov_generator, inh_2Dadaptingmarkov_generator

    b) Continuous Time Process - OU_generator, GWN_generator, continuous_rv_generator (any other distribution)
    """

    def __init__(self, rng=None):
        """
        :param rng: random number generator state object (optional). Either None or a numpy.random.default_rng object,
        or an object with the same interface

        If rng is not None, the provided rng will be used to generate random numbers, otherwise StGen will create
        its own rng.
        """
        if rng is None:
            logger.warning("RNG for StochasticGenerator not set! Results may not be reproducible!")
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a Poisson process
        with the given rate (Hz) and stopping time t_stop (milliseconds).

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param rate: the rate of the discharge (in Hz)
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned,
                      rather than a SpikeTrain object.

        :return spikes: SpikeTrain object

        Examples:
        --------
            >> gen.poisson_generator(50, 0, 1000)
            >> gen.poisson_generator(20, 5000, 10000, array=True)
        """

        n = (t_stop - t_start) / 1000.0 * rate
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.exponential(1.0 / rate, int(number)) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))

        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def gamma_generator(self, a, b, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a gamma process
        with the given shape a, b and stopping time t_stop (milliseconds).
        (average rate will be a*b)

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param a,b: the parameters of the gamma process
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned, rather than a SpikeTrain object.

        Examples:
        --------
            >> gen.gamma_generator(10, 1/10., 0, 1000)
            >> gen.gamma_generator(20, 1/5., 5000, 10000, array=True)
        """
        n = (t_stop - t_start) / 1000.0 * (a * b)
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.gamma(a, b, number) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.gamma(a, b, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.gamma(a, b, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))
        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def OU_generator(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, rectify=False, array=False, time_it=False):
        """
        Generates an Ornstein-Uhlenbeck process using the forward euler method. The function returns
        an AnalogSignal object.

        :param dt: the time resolution in milliseconds of th signal
        :param tau: the correlation time in milliseconds
        :param sigma: std dev of the process
        :param y0: initial value of the process, at t_start
        :param t_start: start time in milliseconds
        :param t_stop: end time in milliseconds
        :param array: if True, the functions returns the tuple (y,t)
                      where y and t are the OU signal and the time bins, respectively,
                      and are both numpy arrays.
        :return AnalogSignal
        """
        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        y[0] = y0
        fac = dt / tau
        gauss = fac * y0 + np.sqrt(2 * fac) * sigma * self.rng.standard_normal(N - 1)
        mfac = 1 - fac

        # python loop... bad+slow!
        for i in range(1, N):
            idx = i - 1
            y[i] = y[idx] * mfac + gauss[idx]

        if time_it:
            logger.info(time.time() - t1)
        if rectify:
            y[y < 0] = 0.

        if array:
            return (y, t)
        else:
            return AnalogSignal(y, dt, t_start, t_stop)

    # @staticmethod  # this can't be static if we want reproducibility
    def GWN_generator(self, amplitude=1., mean=0., std=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                      array=False):
        """
        Generates a Gaussian White Noise process. The function returns an AnalogSignal object.

        :param amplitude: maximum amplitude of the noise signal
        """

        t = np.arange(t_start, t_stop, dt)
        wn = amplitude * self.rng.normal(loc=mean, scale=std, size=len(t))

        if rectify:
            wn[wn < 0] = 0.

        if array:
            return (wn, t)
        else:
            return AnalogSignal(wn, dt, t_start, t_stop)

    @staticmethod
    def continuous_rv_generator(function, amplitude=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                                array=False, **kwargs):
        """
        Generates a realization of a continuous noise process by drawing iid values from the distribution specified by
        function and parameterized by **kwargs
        :param function: distribution function (e.g. np.random.poisson)
        Note: **kwargs must correspond to the function parameters
        """

        t = np.arange(t_start, t_stop, dt)
        if isinstance(function, str):
            function = eval(function)
        s = function(size=len(t), **kwargs)
        s *= amplitude

        if rectify:
            s[s < 0] = 0.
        if array:
            return s, t
        else:
            return AnalogSignal(s, dt, t_start, t_stop)
