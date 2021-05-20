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
        phi = []   # phi=control parameters
        center, radius, weights = self.set_parameters() # calling the previous function with default values of input, output and number of RBF
        for j in range(0, self.numberOfRBF):
            bf = 0
            for i in range(self.numberOfInputs):
                numerator = (inputRBF[i]-center[j][i])*(inputRBF[i]-center[j][i])
                denominator = (radius[j][i]**2)
                if denominator < (10**-6):
                    denominator = 10**-6
                bf = bf + numerator / denominator
            phi.append(math.exp(-bf))
        out = []
        for k in range(self.numberOfOutputs):
            o = 0.0
            for i in range(self.numberOfRBF):
                o = o + weights[i][k] * phi[i]
            out.append(o)
        return out
