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
        # center = [[0]*self.numberOfInputs]*self.numberOfRBF
        # radius = [[0]*self.numberOfInputs]*self.numberOfRBF
        # weights = [[0]*self.numberOfOutputs]*self.numberOfRBF

        # print("at declaration")
        # print(center)
        for i in range(0, self.numberOfRBF):
            for k in range(0, self.numberOfOutputs):
                idk = idk + 1
                ws[k][0] = ws[k][0] + self.Theta[idk]
                # print(self.Theta[idk])
            idk = idk + 2 * self.numberOfInputs
        # print(ws)

        # print("end of loop")
        # change back to self.numberOfRBF
        for l in range(self.numberOfRBF):
            #print("beginning of l loop " + str(l))
            #print(center)
            for j in range(self.numberOfInputs):
                center[l][j] = self.Theta[count]
                # print("Theta value " + str(self.Theta[count]))
                # print("centre is " + str(center[0][0]))
                # print(i)
                # print("center at row " + str(i) + " column " + str(j) + " is " + str(center[l][j]) + " while setting parameters")
                radius[l][j] = self.Theta[count + 1]
                # center.append(self.Theta[count])
                # radius.append(self.Theta[count+1])
                count = count + 2

            for k in range(self.numberOfOutputs):
                if ws[k][0] < 10**-6:
                    weights[l][k] = self.Theta[count]
                    #weights.append(self.Theta[count])
                else:
                    weights[l][k] = self.Theta[count]/ws[k]
                    #weights.append(self.Theta[count]/ws[k])
                count = count + 1
            #print(l)
            #print(center[0][0])
        #print(l)
        #print(center)
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

#new sahiti
#     def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, center, radius, weights):
#         self.numberOfRBF = numberOfRBF
#         self.numberOfInputs = numberOfInputs
#         self.numberOfOutputs = numberOfOutputs
#         self.radius = radius
#         self.center = center
#         self.weights = weights
#         #self.inputRBF = inputRBF
#         zero_array = [[0] * (self.numberOfInputs - 2)] * self.numberOfRBF # TODO: Change the numberofinputs - 2 to a parameter.
        
#         # Q: Why is it that we need only half the values for center and radius?

#         one_array = [[1.0]*(self.numberOfInputs-2)]*self.numberOfRBF
        
#         self.center = np.array(self.center).reshape((self.numberOfRBF, (self.numberOfInputs - 2)))
#         self.center = np.concatenate((self.center, zero_array), axis=1)

#         self.radius = np.array(self.radius).reshape((self.numberOfRBF, (self.numberOfInputs - 2)))
#         self.radius = np.concatenate((self.radius, one_array), axis=1)

#         self.weights = np.array(self.weights).reshape(self.numberOfRBF, self.numberOfOutputs)

#         ws = self.weights.sum(axis=0)

#         for i in [np.where(ws == i)[0][0] for i in ws if i > 10** - 6]:
#             self.weights[:, i] = self.weights[:, i] / ws[i]


# def rbf_control_law(self, inputRBF):
#     center, radius, weights = self.center, self.radius, self.weights # calling the previous function with defau;t va;ues of input, output and number of RBF
#     # phi=control parameters
#     phi = np.exp(-((np.array(inputRBF) - center) **2 / radius ** 2).sum(axis=1))
#     out = (weights * (phi.reshape(self.numberOfRBF, 1))).sum(axis=0)
#     return out