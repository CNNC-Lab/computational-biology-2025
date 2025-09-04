import numpy as np


def spectral_radius(w):
    """
	Compute the spectral radius of a matrix
	:param w: input matrix
	:return: spectral radius
	"""
    return np.linalg.eigvals(w)


def compute_density(A):
    return np.count_nonzero(A) / float(A.shape[0] * A.shape[1])
