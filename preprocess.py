import numpy as np
from scipy.special import erfinv


def rank_gauss(x, epsilon=0.001):
    lower = -1 + epsilon
    upper = 1 - epsilon
    scale_range = upper - lower

    i = np.argsort(x, axis=0)
    j = np.argsort(i, axis=0)

    assert (j.min() == 0).all()
    assert (j.max() == len(j) - 1).all()

    j_range = len(j) - 1
    divider = j_range / scale_range

    transformed = j / divider
    transformed = transformed - upper
    transformed = erfinv(transformed)

    return transformed
