
import numpy as np
import numba

@numba.jit
def format_output(output, weights, n_rbf):
    return (weights * (output.reshape(n_rbf, 1))).sum(
        axis=0)


@numba.jit
def determine_parameters(theta):
    theta = theta.copy()
    theta = theta.reshape((-1, 4))
    centerradius = theta[::2]
    weights = theta[1::2]
    center = centerradius[:, ::2]
    radius = centerradius[:, 1::2]

    ws = weights.sum(axis=0)
    for i in [np.where(ws == i)[0][0] for i in ws if i > 10 ** -6]:
        weights[:, i] = weights[:, i] / ws[i]
    return center, radius, weights


@numba.jit
def squared_exponentia_rbf(rbf_input, center, radius, weights, n_rbf):

    output = np.exp(-(np.sum((rbf_input - center) ** 2 / (radius ** 2),
                             axis=1)))
    output = format_output(output, weights, n_rbf)
    return output
