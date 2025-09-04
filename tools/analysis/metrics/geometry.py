import numpy as np


def euclidean_distance(pos_1, pos_2, N=None):
    """
    Function to calculate the euclidian distance between two points

    :param pos_1:
    :param pos_2:
    :param N:
    :return:
    """
    # If N is not None, it means that we are dealing with a toroidal space,
    # and we have to take the min distance on the torus.
    if N is None:
        dx = pos_1[0] - pos_2[0]
        dy = pos_1[1] - pos_2[1]
    else:
        dx = np.minimum(abs(pos_1[0] - pos_2[0]), N - (abs(pos_1[0] - pos_2[0])))
        dy = np.minimum(abs(pos_1[1] - pos_2[1]), N - (abs(pos_1[1] - pos_2[1])))
    return np.sqrt(dx * dx + dy * dy)
