
import itertools
import numpy as np


from numba.experimental import jitclass
from numba import types



spec = [
    ("numberOfRBF", types.int32),
    ("numberOfInputs", types.int32),
    ("numberOfOutputs", types.int32),
    ("RBFType", types.unicode_type),
    ("Theta", types.float64[:]),
]


# s@jitclass(spec)
class RBF:
    # RBF calculates the values of the radial basis function that determine the release

    # Attributes
    # -------------------
    # numberOfRBF       : int
    #                     number of radial basis functions, typically 2 more than the number of inputs
    # numberofInputs    : int
    # numberOfOutputs   : int
    # RBFType           : str
    # center            : list
    #                     list of center values from optimization
    # radius            : list
    #                     list of radius values from optimization
    # weights           : list
    #                     list of weights from optimization
    # out               : list
    #                     list of the same size as the number of RBFs that determines the control policy

    def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, RBFType,
                 decision_vars):
        self.n_rbfs = numberOfRBF
        self.n_inputs = numberOfInputs
        self.n_outputs = numberOfOutputs
        self.decision_vars = decision_vars
        self.RBFType = RBFType

    def set_parameters(self):
        # for _ in range(n_rbfs-2):
        #     # phase shift center and radius are fixed
        #     for _ in range(n_inputs):
        #         decision_vars.append(Real(-1, 1)) # center
        #         decision_vars.append(Real(0, 1)) # radius
        #
        # for _ in range(2):
        #     decision_vars.append(Real(0, 2*np.pi))
        #
        # for _ in range(n_rbfs):
        #     for _ in range(n_outputs):
        #         decision_vars.append(Real(0, 1)) # weight

        decision_vars = self.decision_vars.copy()

        # center indices
        c_i = []
        r_i = []
        w_i = []

        count = itertools.count()
        for i in range(self.n_rbfs-2):
            for j in range(self.n_inputs):
                c_i.append(next(count))
                r_i.append(next(count))

        for _ in range(self.n_rbfs):
            for _ in range(self.n_outputs):
                w_i.append(next(count)) # weight

        centers = decision_vars[c_i]
        centers = np.concatenate((centers, np.zeros((2*self.n_inputs,))))
        centers = centers.reshape((self.n_rbfs, self.n_inputs))

        radii = decision_vars[c_i]
        radii = np.concatenate((radii, np.ones((2 * self.n_inputs,))))
        radii = radii.reshape((self.n_rbfs, self.n_inputs))

        weights = decision_vars[w_i]
        weights = weights.reshape((self.n_rbfs, self.n_outputs))

        return centers, radii, weights

    def rbf_control_law(self, inputRBF):
        center, radius, weights = self.set_parameters()
        if self.RBFType == "gaussian" or self.RBFType == "gaus":
            phi = self.gaussian(inputRBF, center, radius)
        elif self.RBFType == "multiquadric" or self.RBFType == "multiquad":
            phi = self.multiQuadric(inputRBF, center, radius)
        elif self.RBFType == "multiquadric2":
            phi = self.multiQuadric2(inputRBF, center, radius)
        elif self.RBFType == "invmultiquadric" or self.RBFType == "invmultiquad":
            phi = self.invMultiQuadric(inputRBF, center, radius)
        elif self.RBFType == "invquadratic" or self.RBFType == "invquad":
            phi = self.invQuadratic(inputRBF, center, radius)
        elif self.RBFType == "exponential" or self.RBFType == "exp":
            phi = self.exponential(inputRBF, center, radius)
        elif self.RBFType == "squaredexponential" or self.RBFType == "se":
            phi = self.squaredExponential(inputRBF, center, radius)
        elif self.RBFType == "matern32" or self.RBFType == "mat32":
            phi = self.matern32(inputRBF, center, radius)
        else:
            raise Exception("RBF not found")
        out = (weights * phi[:, np.newaxis]).sum(axis=0)
        return out

    def squaredExponential(self, inputRBF, center, radius):

        a = (inputRBF - center) ** 2 / (radius ** 2)
        b = np.sum(a, axis=1)
        c = np.exp(-(b))

        return np.exp(-(np.sum((inputRBF - center) ** 2 / (radius ** 2), axis=1)))

    def gaussian(self, inputRBF, center, radius):
        return np.exp(-(np.sum((radius * (inputRBF - center)) ** 2, axis=1)))

    def multiQuadric(self, inputRBF, center, radius):
        return np.sqrt(1 + np.sum((radius * (inputRBF - center)) ** 2, axis=1))

    def multiQuadric2(self, inputRBF, center, radius):
        return np.sqrt(np.sum((radius ** 2) + ((inputRBF - center) ** 2), axis=1))

    def invQuadratic(self, inputRBF, center, radius):
        return 1 / (1 + np.sum((radius * (inputRBF - center)) ** 2, axis=1))

    def invMultiQuadric(self, inputRBF, center, radius):
        return 1 / np.sqrt(1 + np.sum((radius * (inputRBF - center)) ** 2, axis=1))

    def exponential(self, inputRBF, center, radius):
        return np.exp(-(np.sum((inputRBF - center) / radius, axis=1)))

    def matern32(self, inputRBF, center, radius):
        return (1 + np.sqrt(3) * np.sum((inputRBF - center) / radius, axis=1)) * (
            np.exp(-np.sqrt(3) * np.sum((inputRBF - center) / radius, axis=1))
        )

    def matern52(self, inputRBF, center, radius):
        return
