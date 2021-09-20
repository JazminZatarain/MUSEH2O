# ===========================================================================
# Name        : main_susquehanna.cpp
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.01
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
from susquehanna_model import susquehanna_model
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import random
import logging

logging.basicConfig(level=logging.INFO)


def main():
    seeds = [10, 20, 30, 40, 50 , 60, 70, 80, 90, 100]
    for modelseed in seeds:
        # set seed
        random.seed(modelseed)
        # RBF parameters
        RBFType = "SquaredExponential"
        numberOfInput = 4  # (time, storage of Conowingo) + 2 for phaseshift
        numberOfOutput = 4  # Atomic, Baltimore, Chester, Downstream: (hydropower, environmental)
        numberOfRBF = 6  # numberOfInput + 2

        # Initialize model
        nobjs = 6
        nvars = int(numberOfRBF * 8 + 2)  # +2 for phaseshift
        n_years = 1
        susquehanna_river = susquehanna_model(108.5, 505.0, 5, n_years)  # l0, l0_MR, d0, years
        # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
        susquehanna_river.load_data(0)  # 0 = historic, 1 = stochastic

        susquehanna_river.setRBF(numberOfRBF, numberOfInput, numberOfOutput, RBFType)

        # Lower and Upper Bound for problem.types
        LB = [-1, 0, -1, 0, 0, 0, 0, 0] * numberOfRBF + [0, 0]
        UB = [1, 1, 1, 1, 1, 1, 1, 1] * numberOfRBF + [np.pi * 2, np.pi * 2]
        # np.pi*2 for phaseshift upperbounds (J. Quinn Como model)
        EPS = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

        # platypus for MOEA, # no contraints
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
        with ProcessPoolEvaluator(4) as evaluator: #change to number of threads
            algorithm = EpsNSGAII(problem, epsilons=EPS, evaluator=evaluator)
            algorithm.run(100000)

        # results
#         print("results:")
#         for solution in algorithm.result:
#             print(solution.objectives)
#         with open(f"./output/{rbftype}_{modelseed}_solution.txt", "w") as f:
#             for solution in algorithm.result:
#                 f.write("%s\n" % solution.objectives)
#         with open(f"./output/{rbftype}_{modelseed}_variables.txt", "w") as f:
#             for solution in algorithm.result:
#                 f.write("%s\n" % solution.variables)

if __name__ == "__main__":
    main()
