# ===========================================================================
# Name        : main_susquehanna.py
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
from susquehanna_model import susquehanna_model
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import logging

logging.basicConfig(level=logging.INFO)


def main():
    # RBF parameters
    numberOfInput = 2  # (time, storage of Conowingo)
    numberOfOutput = 4  # Atomic, Baltimore,Chester, Downstream:- hydropower, environmental
    numberOfRBF = 4  # numberOfInput + 2
    # N = 2 * n * m + K * n  # check 32 with 2 inputs, 72 with 4 inputs

    # Initialize model
    nobjs = 6
    nvars = int(numberOfRBF * 8)  # 8 = 2 centers + 2 radius + 4 weights
    n_years = 1
    susquehanna_river = susquehanna_model(108.5, 505.0, 5, n_years)  # l0, l0_MR, d0, years
    # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
    susquehanna_river.load_data(0)  # 0 = historic, 1 = stochastic

    susquehanna_river.setRBF(numberOfRBF, numberOfInput, numberOfOutput)

    # Lower and Upper Bound for problem.types
    LB = [-1, 0, -1, 0, 0, 0, 0, 0] * numberOfRBF
    UB = [1, 1, 1, 1, 1, 1, 1, 1] * numberOfRBF
    # np.pi*2 for phaseshift upperbounds (J. Quinn Como model) check borg optimization_serial flood C model
    EPS = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

    # platypus MOEA, no contraints
    problem = Problem(nvars, nobjs)
    # problem.types[:] = Real(-1, 1)
    problem.types[:] = [Real(LB[i], UB[i]) for i in range(nvars)]
    problem.function = susquehanna_river.evaluate  # historical (deterministic) optimization
    # problem.function = functools.partial(susquehanna_river.evaluates, opt_met=1) #way to add arguments
    # problem.function = susquehanna_river.evaluateMC  # stochastic optimization
    # problem.directions[:] = Problem.MINIMIZE
    problem.directions[0] = Problem.MINIMIZE  # hydropower
    problem.directions[1] = Problem.MINIMIZE  # atomicpowerplant
    problem.directions[2] = Problem.MINIMIZE  # baltimore
    problem.directions[3] = Problem.MINIMIZE  # chester
    problem.directions[4] = Problem.MAXIMIZE  # environment
    problem.directions[5] = Problem.MINIMIZE  # recreation

    # algorithm = NSGAII(problem)
    # algorithm.run(1)
    with ProcessPoolEvaluator(4) as evaluator:
        algorithm = EpsNSGAII(problem, epsilons=EPS, evaluator=evaluator)
        algorithm.run(100)

    # results
    print("results:")
    for solution in algorithm.result:
        print(solution.objectives)
    with open("./output/output.txt", "w") as f:
        for solution in algorithm.result:
            f.write("%s\n" % solution.objectives)


if __name__ == "__main__":
    main()
