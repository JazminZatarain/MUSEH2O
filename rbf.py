import numpy as np
from numba.experimental import jitclass
from numba import types

spec = [
    ("numberOfRBF", types.int32),
    ("numberOfInputs", types.int32),
    ("numberOfOutputs", types.int32),
    ("RBFType", types.string),  # types.unicode_type
    ("Theta", types.float64[:]),
]


@jitclass(spec)
class RBF:
    # RBF calculates the values of the radial basis function that determine the release

    # Attributes
    # -------------------
    # numberOfRBF       : int
    #                     number of radial basis functions, typically 2 more than the number of inputs
    # numberofInputs    : int
    # numberOfOutputs   : int
    # center            : list
    #                     list of center values from optimization
    # radius            : list
    #                     list of radius values from optimization
    # weights           : list
    #                     list of weights from optimization
    # out               : list
    #                     list of the same size as the number of RBFs that determines the control policy

    def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, RBFType, Theta):
        self.numberOfRBF = numberOfRBF
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.Theta = Theta
        self.RBFType = RBFType

    def set_parameters(self):
        Theta = self.Theta[:-2].copy()
        Theta = Theta.reshape((-1, 4))
        centerradius = Theta[::2]
        weights = Theta[1::2]
        center = centerradius[:, ::2]
        radius = centerradius[:, 1::2]
        # phaseshift
        zeros_arr = np.zeros((self.numberOfRBF, 2), center.dtype)
        ones_arr = np.ones((self.numberOfRBF, 2), radius.dtype)
        center = np.column_stack((center, zeros_arr))
        radius = np.column_stack((radius, ones_arr))

        ws = weights.sum(axis=0)
        for i in [np.where(ws == i)[0][0] for i in ws if i > 10 ** -6]:
            weights[:, i] = weights[:, i] / ws[i]
        return center, radius, weights

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
        out = (weights * (phi.reshape(self.numberOfRBF, 1))).sum(axis=0)
        return out

    def squaredExponential(self, inputRBF, center, radius):
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
        return (1 + np.sqrt(3 * np.sum((inputRBF - center) / radius, axis=1))) * (
            np.exp(-np.sqrt(3 * np.sum((inputRBF - center) / radius, axis=1)))
        )

    # def matern52(self, inputRBF, center, radius):
    #     return (1 + np.sqrt(5 * (inputRBF - center)) / radius + 5 * (inputRBF - center) ** 2 / (3 * radius ** 2)) * (
    #         np.exp(-np.sqrt(5 * (inputRBF - center)) / radius)
    #     )
