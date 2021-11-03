# ===========================================================================
# Name        : main_susquehanna.py
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.05
# Copyright   : Your copyright notice
# ===========================================================================
import csv
import logging
import numpy as np
import os
import pandas as pd
import random

from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator

import rbf_functions
from susquehanna_model import SusquehannaModel


class TrackProgress:
    def __init__(self):
        self.nfe = []
        self.improvements = []
        self.objectives = {}

    def __call__(self, algorithm):
        self.nfe.append(algorithm.nfe)
        self.improvements.append(algorithm.archive.improvements)
        temp = {}
        for i, solution in enumerate(algorithm.archive):
            temp[i] = list(solution.objectives)
        self.objectives[algorithm.nfe] = pd.DataFrame.from_dict(temp,
                                                                orient='index')

    def to_dataframe(self):
        df_imp = pd.DataFrame.from_dict(dict(nfe=self.nfe,
                                             improvements=self.improvements))
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


track_progress = TrackProgress()


def store_results(algorithm, track_progress, output_dir, base_file_name):
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

    # save progress info
    df_conv, df_hv = track_progress.to_dataframe()
    df_conv.to_csv(f"{output_dir}/{base_file_name}_convergence.csv")
    df_hv.to_csv(f"{output_dir}/{base_file_name}_hypervolume.csv")


def main():
    seeds = [10, ]  # , 20, 30, 40, 50, 60, 70, 80, 90, 100]
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
            algorithm.run(100000, track_progress)

        store_results(algorithm, track_progress, 'output',
                      f"{rbf_functions.squared_exponential_rbf.__name__}"
                      f"_{seed}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists("output"):
        try:
            os.mkdir("output")
        except OSError:
            print("Creation of the directory failed")
    main()
