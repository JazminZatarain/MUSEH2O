import numpy as np
import pandas as pd


# def loadMatrix(file_name, row, col):
#     col = range(0, col - 1)
#     try:
#         output = np.loadtxt(file_name, dtype=float, usecols=col, max_rows=row)
#     except IOError:
#         raise Exception("Unable to open file")
#     return output


# def loadVector(file_name, row):
#     try:
#         output = np.loadtxt(file_name, dtype=float, max_rows=int(row))
#     except IOError:
#         raise Exception("Unable to open file")
#     return output


def loadVector(file_name, row=type(None)):
    # print(file_name)
    output = pd.read_csv(file_name, nrows=row, header=None)
    output = output[0].tolist()
    # print(output)
    return output


# TODO: Set default values to take all rows and all columns
def loadMatrix(file_name, row, column):
    # print(column)
    output = pd.read_csv(file_name, header=None, sep="   ", nrows=row, usecols=range(0, column), engine="python")
    output = output.values
    # print(output)
    return output


def loadArrangeMatrix(file_name, rows, cols):  # sahiti
    arr = np.zeros((rows, cols))
    data = np.loadtxt(file_name)
    k = 0
    while k < len(data):
        for i in range(0, rows):
            for j in range(0, cols):
                arr[i][j] = data[k]
                k = k + 1
    return arr


def filterDictionaryPercentile(dictionary, percentile):  # Sahiti
    percentile = np.percentile([v for k, v in dictionary.items()], percentile)
    dictionary = dict(filter(lambda x: x[1] >= percentile, dictionary.items()))
    return dictionary


def logVector():  # not used?
    pass


def logVectorApp():  # not used?
    pass


def interpolate_linear(X, Y, x):
    dim = len(X) - 1
    # print(f"X: {X}")
    # print(f"x: {x}")
    # print(f"X[0]: {X[0]}")
    # if storage is less than
    if x <= X[0]:
        # if x is smaller than the smallest value on X, interpolate between the first two values
        y = (x - X[0]) * (Y[1] - Y[0]) / (X[1] - X[0]) + Y[0]
        return y
    elif x >= X[dim]:
        # if x is larger than the largest value, interpolate between the the last two values
        y = Y[dim] + (Y[dim] - Y[dim - 1]) / (X[dim] - X[dim - 1]) * (x - X[dim])  # y = Y[dim]
        return y
    else:
        y = np.interp(x, X, Y)
    return y


def gallonToCubicFeet(x):
    conv = 0.13368  # 1 gallon = 0.13368 cf
    return x * conv


def inchesToFeet(x):
    conv = 0.08333  # 1 inch = 0.08333 ft
    return x * conv


def cubicFeetToCubicMeters(x):
    conv = 0.0283  # 1 cf = 0.0283 m3
    return x * conv


def feetToMeters(x):
    conv = 0.3048  # 1 ft = 0.3048 m
    return x * conv


def acreToSquaredFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x * conv


def acreFeetToCubicFeet(x):
    conv = 43560  # 1 acre-feet = 43560 feet3
    return x * conv


def cubicFeetToAcreFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x / conv


def computePercentile(x, percentile):
    return np.percentile(x, 100 - percentile)


def computeMax(g):
    if isinstance(g, np.ndarray):
        return g.max()  # has to be np array
    else:
        # print("No np array check code")
        return max(g)  # np.max(g)


def computeMin(g):
    if isinstance(g, np.ndarray):
        return g.min()  # has to be np array
    else:
        # print("No np array check code")
        return min(g)  # np.min(g)


def computeMean(g):
    return sum(g) / len(g)
    # if isinstance(g, np.ndarray):
    #     return g.mean()  # has to be np array
    # else:
    #     print("No np array check code")
    #     return np.mean(g)


def computeSum(g):
    return sum(g)
