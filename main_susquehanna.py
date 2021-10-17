# ===========================================================================
# Name        : main_susquehanna.cpp
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.01
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
import os

from susquehanna_model import susquehanna_model
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import random
import csv
import logging

logging.basicConfig(level=logging.INFO)


def main():
    seeds = [10] #, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for modelseed in seeds:
        # set seed
        random.seed(modelseed)
        # RBF parameters
        RBFType = "SquaredExponential"
        n_inputs = 3  # (storage of Conowingo) + 2 for phaseshift
        n_outputs = 4  # Atomic, Baltimore, Chester, Downstream: (hydropower,
        # environmental)
        n_rbfs = 5  # numberOfInput + 2

        # Initialize model
        n_objs = 6

        # center + radius + weight
        # but center and radius are fixed for sin and cos input
        # so
        # center and radius for phase shift inputs are fixed
        # so n_rbf - 2
        n_decisionvars = n_inputs * (n_rbfs-2) * 2 + 2 + (n_rbfs * n_outputs)
        n_years = 1
        susquehanna_river = susquehanna_model(108.5, 505.0, 5, n_years)  # l0, l0_MR, d0, years
        # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
        susquehanna_river.load_data(0)  # 0 = historic, 1 = stochastic
        susquehanna_river.set_log(False)
        susquehanna_river.setRBF(n_rbfs, n_inputs, n_outputs,
                                 RBFType)

        # Lower and Upper Bound for problem.types
        # LB = [-1, 0, -1, 0, 0, 0, 0, 0] * numberOfRBF + [0, 0]
        # UB = [1, 1, 1, 1, 1, 1, 1, 1] * numberOfRBF + [2*np.pi, 2*np.pi]

        decision_vars = []
        for _ in range(n_rbfs-2):
            # phase shift center and radius are fixed
            for _ in range(n_inputs):
                decision_vars.append(Real(-1, 1)) # center
                decision_vars.append(Real(0, 1)) # radius

        for _ in range(n_rbfs):
            for _ in range(n_outputs):
                decision_vars.append(Real(0, 1)) # weight

        # phase shifts
        for _ in range(2):
            decision_vars.append(Real(0, 2*np.pi))

        # np.pi*2 for phaseshift upperbounds (J. Quinn Como model)
        eps = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

        # platypus for MOEA, no contraints

        problem = Problem(len(decision_vars), n_objs)
        problem.types[:] = decision_vars
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

        # algorithm = EpsNSGAII(problem, epsilons=eps)
        # algorithm.run(1000)

        with ProcessPoolEvaluator() as evaluator: #change to number of threads
            algorithm = EpsNSGAII(problem, epsilons=eps, evaluator=evaluator)
            algorithm.run(1000)

        header = ['hydropower', 'atomicpowerplant', 'baltimore', 'chester',
                  'environment', 'recreation']
        with open(f'output/{RBFType}_{modelseed}_solution.csv', 'w',
                  encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for solution in algorithm.result:
                writer.writerow(solution.objectives)

        with open(f'output/{RBFType}_{modelseed}_variables.csv', 'w',
                  encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for solution in algorithm.result:
                writer.writerow(solution.variables)

if __name__ == "__main__":
    if not os.path.exists("output"):
        try:
            os.mkdir("output")
        except OSError:
            print("Creation of the directory failed")
    main()
