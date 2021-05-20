# ===========================================================================
# Name        : main_susquehanna.cpp
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.01
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
from susquehanna_model import susquehanna_model
from platypus import Problem, NSGAII, Real

# Initialize model
policy_sim = 0
nobjs = 6
nvars = 32
vars = np.zeros(nvars).tolist()
n_years = 1
susquehanna_river = susquehanna_model(
    108.5, 505.0, 5, n_years
)  # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
susquehanna_river.load_data()

# RBF parameters
m = 2  # number of input (time, storage of Conowingo)
K = 4  # number of output, Atomic, Baltimore,Chester, Downstream:- hydropower, environmental
n = m + 2  # number of RBF
N = 2 * n * m + K * n  # check
if not N == nvars:
    print("N not equal to nvars")
susquehanna_river.setRBF(n, m, K)
susquehanna_river.setPolicySim(policy_sim)

LB = [-1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0]
UB = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# EPS = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

# platypus for MOEA, # no contraints
problem = Problem(nvars, nobjs)
# problem.types[:] = Real(-1, 1)
problem.types[:] = [Real(LB[i], UB[i]) for i in range(nvars)]
# problem.function = susquehanna_river.evaluate # historical (deterministic) optimization
problem.function = susquehanna_river.evaluateMC  # stochastic optimization
algorithm = NSGAII(problem)
algorithm.run(1)
# results
print(algorithm.result)
with open("output.txt", "w") as f:
    for item in algorithm.result:
        f.write("%s\n" % item)
