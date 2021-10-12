
import numpy as np
import numba

# @numba.jit
def format_output(output, weights, n_rbf):
    return (weights * (output.reshape(n_rbf, 1))).sum(
        axis=0)


# @numba.jit
def determine_parameters(decision_vars, n_inputs, n_rbfs):
    decision_vars = decision_vars.copy()

    cr = decision_vars[:n_inputs*n_rbfs*2]
    center = cr[::2].reshape((n_inputs, n_rbfs))
    radius = cr[1::2].reshape((n_inputs, n_rbfs))
    weights = decision_vars[n_inputs*n_rbfs*2::]

    # theta = theta.reshape((-1, 4))
    # centerradius = theta[::2]
    # weights = theta[1::2]
    # center = centerradius[:, ::2]
    # radius = centerradius[:, 1::2]

    ws = weights.sum(axis=0)
    weights /= ws # removes 10**-6 threshold for now

    # for i in [np.where(ws == i)[0][0] for i in ws if i > 10 ** -6]:
    #     weights[i] = weights[i] / ws
    return center, radius, weights


# @numba.jit
def squared_exponentia_rbf(rbf_input, center, radius, weights, n_rbf):

    # sum over inputs
    a = rbf_input[:, np.newaxis] - center
    b = a ** 2
    c = radius ** 2

    output = np.exp(-(np.sum(b/c, axis=0)))

    output = output * weights
    return output
