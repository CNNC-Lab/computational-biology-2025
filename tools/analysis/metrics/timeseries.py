import numpy as np
from matplotlib import pyplot as plt


def ccf(x, y, axis=None):
    """
    Fast cross correlation function based on fft.

    Computes the cross-correlation function of two series.
    Note that the computations are performed on anomalies (deviations from
    average).
    Returns the values of the cross-correlation at different lags.

    Parameters
    ----------
    x, y : 1D MaskedArrays
        The two input arrays.
    axis : integer, optional
        Axis along which to compute (0 for rows, 1 for cols).
        If `None`, the array is flattened first.

    Examples
    --------
    >> z = np.arange(5)
    >> ccf(z,z)
    array([  3.90798505e-16,  -4.00000000e-01,  -4.00000000e-01,
            -1.00000000e-01,   4.00000000e-01,   1.00000000e+00,
             4.00000000e-01,  -1.00000000e-01,  -4.00000000e-01,
            -4.00000000e-01])
    """
    assert x.ndim == y.ndim, "Inconsistent shape !"
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = (x - x.mean(axis=None))
        yanom = (y - y.mean(axis=None))
        Fx = np.fft.fft(xanom, npad, )
        Fy = np.fft.fft(yanom, npad, )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(xanom, xanom) * np.inner(yanom, yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Arrays should have the same length!")
            xanom = (x - x.mean(axis=1)[:, None])
            yanom = (y - y.mean(axis=1)[:, None])
            varxy = np.sqrt((xanom * xanom).sum(1) *
                            (yanom * yanom).sum(1))[:, None]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Arrays should have the same width!")
            xanom = (x - x.mean(axis=0))
            yanom = (y - y.mean(axis=0))
            varxy = np.sqrt((xanom * xanom).sum(0) * (yanom * yanom).sum(0))
        Fx = np.fft.fft(xanom, npad, axis=axis)
        Fy = np.fft.fft(yanom, npad, axis=axis)
        iFxy = np.fft.ifft(Fx.conj() * Fy, n=npad, axis=axis).real
    # We just turn the lags into correct positions:
    iFxy = np.concatenate((iFxy[len(iFxy) / 2:len(iFxy)],
                           iFxy[0:len(iFxy) / 2]))
    return iFxy / varxy


def lag_ix(x,y):
    """
    Calculate lag position at maximal correlation
    :param x:
    :param y:
    :return:
    """
    corr = np.correlate(x,y,mode='full')
    pos_ix = np.argmax( np.abs(corr) )
    lag_ix = pos_ix - (corr.size-1)/2
    return lag_ix


def cross_correlogram(x, y, max_lag=100., dt=0.1, plot=True):
    """
    Returns the cross-correlogram of x and y
    :param x:
    :param y:
    :param max_lag:
    :return:
    """
    corr = np.correlate(x, y, 'full')
    pos_ix = np.argmax(np.abs(corr))
    maxlag = (corr.size - 1) // 2
    lag = np.arange(-maxlag, maxlag + 1) * dt
    cutoff = [np.where(lag == -max_lag), np.where(lag == max_lag)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(lag, corr, lw=1)
        ax.set_xlim(lag[cutoff[0]], lag[cutoff[1]])
        ax.axvline(x=lag[pos_ix], ymin=np.min(corr), ymax=np.max(corr), linewidth=1.5, color='c')
        plt.show()
    return lag, corr


def simple_frequency_spectrum(x):
    """
    Simple frequency spectrum.

    Very simple calculation of frequency spectrum with no detrending,
    windowing, etc, just the first half (positive frequency components) of
    abs(fft(x))

    Parameters
    ----------
    x : array_like
        The input array, in the time-domain.

    Returns
    -------
    spec : array_like
        The frequency spectrum of `x`.

    """
    spec = np.absolute(np.fft.fft(x))
    spec = spec[:len(x) // 2]  # take positive frequency components
    spec /= len(x)  # normalize
    spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
    spec[0] /= 2.0  # except for the dc component
    return spec


def rescale_signal(val, out_min, out_max):
    """
    Rescale a signal to a new range
    :param val: original signal (as a numpy array)
    :param out_min: new minimum
    :param out_max: new maximum
    :return:
    """
    in_min = np.min(val)
    in_max = np.max(val)
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def autocorrelation_function(x):
    """
    Determine the autocorrelation of signal x

    :param x:
    :return:
    """

    n = len(x)
    data = np.asarray(x)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return acf_lag  # round(acf_lag, 3)

    x = np.arange(n)  # Avoiding lag 0 calculation
    acf_coeffs = list(map(r, x))
    return acf_coeffs
