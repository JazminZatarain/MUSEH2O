import math
import numpy as np


class RBF:
    # RBF calculates the values of the radial basis function that determine the release 

    # Attributes
    #-------------------
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

    def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, Theta):
        self.numberOfRBF = numberOfRBF
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.Theta = Theta
        # self.inputRBF = inputRBF

    def set_parameters(self):
        idk = 2 * self.numberOfInputs - 1
        count = 0
        ws = np.zeros((self.numberOfOutputs, 1))  # []*self.numberOfOutputs

        center = np.zeros((self.numberOfRBF, self.numberOfInputs))
        radius = np.zeros((self.numberOfRBF, self.numberOfInputs))
        weights = np.zeros((self.numberOfRBF, self.numberOfOutputs))

        for i in range(0, self.numberOfRBF):
            for k in range(0, self.numberOfOutputs):
                idk = idk + 1
                ws[k][0] = ws[k][0] + self.Theta[idk]
            idk = idk + 2 * self.numberOfInputs
        for l in range(self.numberOfRBF):
            for j in range(self.numberOfInputs):
                center[l][j] = self.Theta[count]
                radius[l][j] = self.Theta[count + 1]
                count = count + 2

            for k in range(self.numberOfOutputs):
                if ws[k][0] < 10**-6:
                    weights[l][k] = self.Theta[count]
                else:
                    weights[l][k] = self.Theta[count]/ws[k]
                count = count + 1
        return center, radius, weights

    def rbf_control_law(self, inputRBF):
        (
            center,
            radius,
            weights,
        ) = (
            self.set_parameters()
        )  # calling the previous function with default values of input, output and number of RBF
        # phi=control parameters
        phi = np.exp(-((np.array(inputRBF) - center) ** 2 / radius ** 2).sum(axis=1))
        out = (weights * (phi.reshape(self.numberOfRBF, 1))).sum(axis=0)
        return out
