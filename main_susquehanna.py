# ===========================================================================
# Name        : main_susquehanna.cpp
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.01
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
from susquehanna_model import susquehanna_model
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # Initialize model
    policy_sim = 0
    nobjs = 6
    nvars = 32  # 48
    # vars = np.zeros(nvars).tolist()
    n_years = 1
    susquehanna_river = susquehanna_model(
        108.5, 505.0, 5, n_years
    )  # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
    susquehanna_river.load_data(0)  # 0 = historic, 1 = stochastic

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
    EPS = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

    # platypus for MOEA, # no contraints
    problem = Problem(nvars, nobjs)
    # problem.types[:] = Real(-1, 1)
    problem.types[:] = [Real(LB[i], UB[i]) for i in range(nvars)]
    problem.function = susquehanna_river.evaluate  # historical (deterministic) optimization
    # problem.function = functools.partial(susquehanna_river.evaluates, opt_met=1) #way to add arguments
    # problem.function = susquehanna_river.evaluateMC  # stochastic optimization
    # problem.directions[:] = Problem.MINIMIZE
    problem.directions[0] = Problem.MAXIMIZE  # hydropower
    problem.directions[1] = Problem.MAXIMIZE  # atomicpowerplant
    problem.directions[2] = Problem.MAXIMIZE  # baltimore
    problem.directions[3] = Problem.MAXIMIZE  # chester
    problem.directions[4] = Problem.MINIMIZE  # environment
    problem.directions[5] = Problem.MAXIMIZE  # recreation

    # algorithm = NSGAII(problem)
    # algorithm.run(1)
    with ProcessPoolEvaluator(4) as evaluator:
        algorithm = EpsNSGAII(problem, epsilons=EPS, population_size=10, evaluator=evaluator)
        algorithm.run(1000)

    # results
    print("results:")
    for solution in algorithm.result:
        print(solution.objectives)
    with open("./output/output.txt", "w") as f:
        for solution in algorithm.result:
            f.write("%s\n" % solution.objectives)


if __name__ == "__main__":
    main()
