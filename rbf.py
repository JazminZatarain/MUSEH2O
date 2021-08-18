import numpy as np
from numpy.linalg import norm


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
    # inputRBF          : list
    #                     list of input values (same length as number of inputs)
    # out               : list
    #                     list of the same size as the number of RBFs that determines the control policy

    def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, Theta):
        self.numberOfRBF = numberOfRBF
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.Theta = Theta

    def set_parameters(self):
        Theta = np.array(self.Theta[:-2]).reshape(-1, 4)  # [:-2] to remove 2 phaseshift vars
        centerradius = Theta[::2]
        weights = Theta[1::2]
        center = centerradius[:, ::2]
        radius = centerradius[:, 1::2]

        zeros_arr = np.zeros((self.numberOfRBF, 2), center.dtype)
        ones_arr = np.ones((self.numberOfRBF, 2), radius.dtype)
        center = np.c_[center, zeros_arr]
        radius = np.c_[radius, ones_arr]

        ws = np.sum(weights, axis=0)  # weights.sum(axis=0)
        for i in [np.where(ws == i)[0][0] for i in ws if i > 10 ** -6]:
            weights[:, i] = weights[:, i] / ws[i]
        return center, radius, weights

    def rbf_control_law(self, inputRBF):
        center, radius, weights = self.set_parameters()
        phi = self.gaussian2(inputRBF, center, radius)
        out = (weights * (phi.reshape(self.numberOfRBF, 1))).sum(axis=0)
        return out

    def gaussian1(self, inputRBF, center, radius):
        return np.exp(-((np.array(inputRBF) - center) ** 2 / (radius ** 2)).sum(axis=1))

    def gaussian2(self, inputRBF, center, radius):
        return np.exp((-((radius * (np.array(inputRBF) - center)) ** 2)).sum(axis=1))

    def gaussian3(self, inputRBF, center, radius):
        return np.exp((-((radius * (norm(np.array(inputRBF) - center))) ** 2)).sum(axis=1))

    def invMultiQuadric(self, inputRBF, center, radius):
        return pow(1 + (norm(center - np.array(inputRBF)) * radius) ** 2, -0.5).sum(axis=1)

    def invMultiQuadric2(self, inputRBF, center, radius):
        return 1 / np.sqrt(1 + (radius * (center - np.array(inputRBF)) ** 2) ** 2).sum(axis=1)

    def invQuadratic(self, inputRBF, center, radius):
        return 1 / (1 + (radius * (center - np.array(inputRBF)) ** 2) ** 2).sum(axis=1)

    def exponential(self, inputRBF, center, radius):
        return np.exp(-(norm(center - np.array(inputRBF)) ** 2) / radius).sum(axis=1)

    def squaredExponential(self, inputRBF, center, radius):
        return np.exp(-((norm(center - np.array(inputRBF)) ** 2) ** 2) / (2 * radius ** 2)).sum(axis=1)
