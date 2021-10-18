# ===========================================================================
# Name        : main_susquehanna.py
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np

import rbf_functions
from susquehanna_model import SusquehannaModel
# from rbf import SquaredexponentialRBF
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import random
import os
import csv
import logging

logging.basicConfig(level=logging.INFO)


def store_results(algorithm, output_dir, base_file_name):
    header = ["hydropower", "atomicpowerplant", "baltimore", "chester",
              "environment", "recreation"]
    with open(f"{output_dir}/{base_file_name}_solution.csv", "w",
              encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for solution in algorithm.result:
            writer.writerow(solution.objectives)

    with open(f"{output_dir}/{base_file_name}_variables.csv", "w",
              encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.variables)


def main():
    seeds = [10,]  # , 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for seed in seeds:
        random.seed(seed)

        # RBF parameters
        n_inputs = 2  # (time, storage of Conowingo)
        n_outputs = 4
        n_rbfs = 4
        rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs)

        # Initialize model
        n_objectives = 6
        n_years = 1

        susquehanna_river = SusquehannaModel(108.5, 505.0, 5, n_years,
                                             rbf)
        susquehanna_river.set_log(False)

        # Lower and Upper Bound for problem.types

        epsilons = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]
        n_decision_vars = len(rbf.platypus_types)

        problem = Problem(n_decision_vars, n_objectives)
        problem.types[:] = rbf.platypus_types
        problem.function = susquehanna_river.evaluate

        problem.directions[0] = Problem.MINIMIZE  # hydropower
        problem.directions[1] = Problem.MINIMIZE  # atomic power plant
        problem.directions[2] = Problem.MINIMIZE  # baltimore
        problem.directions[3] = Problem.MINIMIZE  # chester
        problem.directions[4] = Problem.MAXIMIZE  # environment
        problem.directions[5] = Problem.MINIMIZE  # recreation

        # algorithm = EpsNSGAII(problem, epsilons=epsilons)
        # algorithm.run(1000)

        with ProcessPoolEvaluator() as evaluator:
            algorithm = EpsNSGAII(problem, epsilons=epsilons,
                                  evaluator=evaluator)
            algorithm.run(100000)

        store_results(algorithm, 'output', f"{rbf_functions.squared_exponentia_rbf.__name__}"
                                           f"_{seed}")


if __name__ == "__main__":
    if not os.path.exists("output"):
        try:
            os.mkdir("output")
        except OSError:
            print("Creation of the directory failed")
    main()
