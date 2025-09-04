import numpy as np


def acc_function(x, a, b, tau):
    """
    Generic exponential function (to use whenever we want to fit an exponential function to data)
    :param x:
    :param a:
    :param b:
    :param tau: decay time constant
    :return:
    """
    return a * (np.exp(-x / tau) + b)


def err_func(params, x, y, func):
    """
    Error function for model fitting
    The marginals of the fit to x/y given the params
    """
    return y - func(x, *params)
