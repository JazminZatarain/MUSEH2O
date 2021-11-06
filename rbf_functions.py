import itertools
import numpy as np
import numba
import math

from platypus import Real


def squared_exponential_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """

    # sum over inputs
    a = rbf_input[np.newaxis, :] - centers
    b = a ** 2
    c = radii ** 2
    rbf_scores = np.exp(-(np.sum(b / c, axis=1)))

    # n_rbf x n_output, n_rbf
    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def gaussian_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    a = rbf_input[np.newaxis, :] - centers
    n = a / radii
    p = n ** 2
    q = np.sum(p, axis=1)
    rbf_scores = np.exp(-1 * q)

    # n_rbf x n_output, n_rbf
    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def multiquadric_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    a = rbf_input[np.newaxis, :] - centers
    b = a / radii
    c = b ** 2
    d = np.sum(c, axis=1)
    rbf_scores = np.sqrt(1 + d)

    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def inverse_quadric_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    a = rbf_input[np.newaxis, :] - centers
    b = a / radii
    c = b ** 2
    d = np.sum(c, axis=1)
    rbf_scores = 1 / (1 + d)

    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def inverse_multiquadric_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    a = rbf_input[np.newaxis, :] - centers
    b = (a / radii) ** 2
    rbf_scores = 1 / np.sqrt(1 + np.sum(b, axis=1))

    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)

    return output


def exponential_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    a = rbf_input[np.newaxis, :] - centers
    b = (a / radii) ** 2  # TODO
    rbf_scores = np.exp(-1 * np.sum(b, axis=1))

    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)
    return output


def multi_quadric2_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array

    """

    rbf_scores = np.sqrt(np.sum((radii ** 2) + ((rbf_input - centers) ** 2),
                                axis=1))
    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)
    return output


def matern32_rbf(rbf_input, centers, radii, weights):
    """

    Parameters
    ----------
    rbf_input : numpy array
                1-D, shape is (n_inputs,)
    centers :   numpy array
                2-D, shape is (n_rbfs X n_inputs)
    radii :     2-D, shape is (n_rbfs X n_inputs)
    weights :   2-D, shape is (n_rbfs X n_outputs)

    Returns
    -------
    numpy array


    """
    diff = rbf_input - centers
    squared = (diff / radii) ** 2  # TODO:: hacked to make it work
    sqrt = np.sqrt(3 * np.sum(squared, axis=1))
    rbf_scores = (1 + sqrt) * (np.exp(-sqrt))

    weighted_rbfs = weights * rbf_scores[:, np.newaxis]
    output = weighted_rbfs.sum(axis=0)
    return output


rbfs = [squared_exponential_rbf,
        gaussian_rbf,
        multiquadric_rbf,
        inverse_multiquadric_rbf,
        inverse_quadric_rbf,
        exponential_rbf,
        multi_quadric2_rbf,
        matern32_rbf]


class RBF:

    def __init__(self, n_rbfs, n_inputs, n_outputs,
                 rbf_function=squared_exponential_rbf):
        self.n_rbfs = n_rbfs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.rbf = rbf_function

        types = []
        c_i = []
        r_i = []
        w_i = []

        # as private attribute, subclasses can add stuff in the right places
        self._count = itertools.count()
        for _ in range(self.n_inputs):
            for _ in range(self.n_rbfs):
                types.append(Real(-1, 1))  # center
                c_i.append(next(self._count))
                types.append(Real(0, 1))  # radius
                r_i.append(next(self._count))

        for _ in range(self.n_outputs):
            for _ in range(self.n_rbfs):
                types.append(Real(0, 1))  # weight
                w_i.append(next(self._count))  # weight

        self.platypus_types = types
        self.c_i = np.asarray(c_i, dtype=np.int)
        self.r_i = np.asarray(r_i, dtype=np.int)
        self.w_i = np.asarray(w_i, dtype=np.int)

        self.centers = None
        self.radii = None
        self.weights = None

    def set_decision_vars(self, decision_vars):
        decision_vars = decision_vars.copy()

        # order is set to Fortran, so column-first reshaping
        # this is linked to the nested loop in __init__
        # we have the centers, raddi, and weights for all RBFs per input/output
        shape = (self.n_rbfs, self.n_inputs)
        self.centers = decision_vars[self.c_i].reshape(shape, order='F')
        self.radii = decision_vars[self.r_i].reshape(shape, order='F')

        shape = (self.n_rbfs, self.n_outputs)
        self.weights = decision_vars[self.w_i].reshape(shape, order='F')

        # sum of weights per input is 1
        self.weights /= self.weights.sum(axis=0)[np.newaxis, :]

    def apply_rbfs(self, inputs):
        outputs = self.rbf(inputs, self.centers, self.radii, self.weights)

        return outputs


class PhaseShiftRBF(RBF):
    n_phaseshift_inputs = 2

    def __init__(self, n_rbfs, n_inputs, n_outputs,
                 rbf_function=squared_exponential_rbf):
        super().__init__(n_rbfs, n_inputs - self.n_phaseshift_inputs,
                         n_outputs,
                         rbf_function=rbf_function)

        self.n_inputs = n_inputs

        # 2 options
        # either remove the centers and radii decision vars after super
        # or add indices etc. for them here
        # solution below does not work with the reshaping
        # so use super for set_decision_vars first and then add rows for
        # phase shift?

        c_i = []
        r_i = []
        for _ in range(self.n_phaseshift_inputs):
            for _ in range(self.n_rbfs):
                c_i.append(next(self._count))
                r_i.append(next(self._count))
        self.shape = (next(self._count),)

        # phase shift decision vars
        for _ in range(self.n_phaseshift_inputs):
            self.platypus_types.append(Real(0, math.tau))
            self.platypus_types.append(Real(0, math.tau))

        self.ps_ci = np.asarray(c_i, dtype=np.int)
        self.ps_ri = np.asarray(r_i, dtype=np.int)
        self.c_i = np.concatenate((self.c_i, self.ps_ci))
        self.r_i = np.concatenate((self.r_i, self.ps_ri))

    def set_decision_vars(self, decision_vars):
        self.n_phaseshift_inputs = decision_vars[-2::]
        decision_vars = decision_vars[0:-2]

        extended_decision_vars = np.empty(self.shape)
        extended_decision_vars[0:decision_vars.shape[0]] = decision_vars
        extended_decision_vars[self.ps_ci] = 0
        extended_decision_vars[self.ps_ri] = 1
        super().set_decision_vars(extended_decision_vars)

    def apply_rbfs(self, inputs):
        # go from two to three inputs
        # TODO:: everything is now tied to how it is coded in the susquena
        # model
        # order of inputs matters!!!
        # we now have level, phase_shift1, phase_shift2
        # this is consistent with what happens in set_decision_vars
        t = inputs[0]  # is already normalized on n_days  * n_decisions per day
        modified_inputs = np.asarray([inputs[1],
                                      (math.sin(
                                          math.tau * t -
                                          self.n_phaseshift_inputs[
                                              0]) + 1) / 2,
                                      (math.cos(
                                          math.tau * t -
                                          self.n_phaseshift_inputs[
                                              1]) + 1) / 2,
                                      ])

        # how to normalize the phase shifted stuff?

        return super().apply_rbfs(modified_inputs)
