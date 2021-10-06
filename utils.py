import numpy as np
import pandas as pd
from numba import njit

# Can't load matrix with just numpy for some reason, troubleshoot later
# # TODO: Set default values to take all rows and all columns
# def loadMatrix(file_name, row, col):
#     try:
#         output = np.loadtxt(file_name, dtype=float, max_rows=row)
#     except IOError:
#         raise Exception("Unable to open file")
#     return output

# TODO: Set default values to take all rows and all columns
def loadMatrix(file_name, row, column):
    output = pd.read_csv(file_name, header=None, sep="   ", nrows=row, usecols=range(0, column), engine="python")
    output = output.to_numpy()
    return output


# TODO: Set default values to take all rows and all columns
def loadVector(file_name, row):
    try:
        output = np.loadtxt(file_name, dtype=float, max_rows=int(row))
    except IOError:
        raise Exception("Unable to open file")
    return output


# TODO: Set default values to take all rows and all columns
def loadMultiVector(file_name, n_years, n_days_one_year):
    N_samples = n_days_one_year * n_years
    try:
        output = np.loadtxt(file_name, dtype=float, max_rows=int(N_samples))
    except IOError:
        raise Exception("Unable to open file")
    output = output.reshape(n_years, 365)
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

@njit
def interpolate_linear(X, Y, x):
    dim = len(X) - 1
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


@njit
def gallonToCubicFeet(x):
    conv = 0.13368  # 1 gallon = 0.13368 cf
    return x * conv


@njit
def inchesToFeet(x):
    conv = 0.08333  # 1 inch = 0.08333 ft
    return x * conv


@njit
def cubicFeetToCubicMeters(x):
    conv = 0.0283  # 1 cf = 0.0283 m3
    return x * conv


@njit
def feetToMeters(x):
    conv = 0.3048  # 1 ft = 0.3048 m
    return x * conv


@njit
def acreToSquaredFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x * conv


@njit
def acreFeetToCubicFeet(x):
    conv = 43560  # 1 acre-feet = 43560 feet3
    return x * conv


@njit
def cubicFeetToAcreFeet(x):
    conv = 43560  # 1 acre = 43560 feet2
    return x / conv


def computePercentile(x, percentile):
    return np.percentile(x, percentile)


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
