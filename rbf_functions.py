
import itertools
from platypus import Real

import numpy as np
import numba



class RBF:

    def __init__(self, n_rbfs, n_inputs, n_outputs):
        self.n_rbfs = n_rbfs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        types = []
        count = itertools.count()
        for i in range(self.n_rbfs-2):
            for j in range(self.n_inputs):
                types.append(Real(-1, 1))   # center
                c_i.append(next(count))
                types.append(Real(0, 1))    # radius
                r_i.append(next(count))

        for _ in range(self.n_rbfs):
            for _ in range(self.n_outputs):
                types.append(Real(0, 1))    # weight
                w_i.append(next(count)) # weight

        self.platypus_types = types
        self.c_i = c_i
        self.r_i = r_i
        self.w_i = w_i


    def set_decision_vars(self, decision_vars):
        decision_vars = decision_vars.copy()

        centers = decision_vars[self.c_i].reshape((n_rbf, n_inputs))
        radii = decision_Vars[self.r_i].reshape((n_rbf, n_inputs))
        weights = decision_vars[self.w_i].reshape((n_rbf, n_outputs))

        ws = weights.sum(axis=0)
        #divide each row
        weights /= ws[:, np.newaxis] # removes 10**-6 threshold for now

        # for i in [np.where(ws == i)[0][0] for i in ws if i > 10 ** -6]:
        #     weights[i] = weights[i] / ws
        return center, radius, weights

# @numba.jit
def format_output(output, weights, n_rbf):
    return (weights * (output.reshape(n_rbf, 1))).sum(
        axis=0)





# @numba.jit
def squared_exponentia_rbf(rbf_input, center, radius, weights, n_rbf):

    # sum over inputs
    a = rbf_input[:, np.newaxis] - center
    b = a ** 2
    c = radius ** 2

    output = np.exp(-(np.sum(b/c, axis=0)))

    # output (1, rbf)
    # weights (output, rbf)
    # so row wise

    output = np.sum(output[np.newaxis, :] * weights, axis=1)
    return output
