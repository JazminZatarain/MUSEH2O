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

from rbf import rbf_functions
from dps_lake_model import lake_model



class Wrapper():
    # yes I no this can be done also through functools

    def __init__(self, rbf):
        self.rbf = rbf

    def __call__(self, args):
        return lake_model(self.rbf, np.asarray(args))

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
        self.objectives[algorithm.nfe] = pd.DataFrame.from_dict(temp, orient="index")

    def to_dataframe(self):
        df_imp = pd.DataFrame.from_dict(
            dict(nfe=self.nfe, improvements=self.improvements)
        )
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


def store_results(algorithm, track_progress, output_dir, rbf_name, seed_id):
    path_name = f"{output_dir}/{rbf_name}"
    if not os.path.exists(path_name):
        try:
            os.mkdir(path_name)
        except OSError:
            print("Creation of the directory failed")

    header = [
        "max_p",
        "utility",
        "inertia",
        "reliability",
    ]
    with open(
        f"{output_dir}/{rbf_name}/{seed_id}_solution.csv",
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for solution in algorithm.result:
            writer.writerow(solution.objectives)

    with open(
        f"{output_dir}/{rbf_name}/{seed_id}_variables.csv",
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.variables)

    # save progress info
    df_conv, df_hv = track_progress.to_dataframe()
    df_conv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_convergence.csv")
    df_hv.to_csv(f"{output_dir}/{rbf_name}/{seed_id}_hypervolume.csv")


def main():
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for entry in [
        rbf_functions.original_rbf,
        rbf_functions.squared_exponential_rbf,
        rbf_functions.inverse_multiquadric_rbf,
        rbf_functions.inverse_quadratic_rbf,
        rbf_functions.exponential_rbf,
        rbf_functions.matern32_rbf,
        rbf_functions.matern52_rbf,
    ]:
        for seed in seeds:
            random.seed(seed)

            # RBF parameters
            n_inputs = 1  # polution at t-1
            n_outputs = 1 # release at t
            n_rbfs = n_inputs+2
            n_objectives = 4
            rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

            # Initialize model
            wrapper = Wrapper(rbf)

            # Lower and Upper Bound for problem.types
            epsilons = [0.05, 0.05, 0.05, 0.05]
            n_decision_vars = len(rbf.platypus_types)

            problem = Problem(n_decision_vars, n_objectives)
            problem.types[:] = rbf.platypus_types
            problem.function = wrapper

            problem.directions[0] = Problem.MINIMIZE  # MAX_P
            problem.directions[1] = Problem.MAXIMIZE  # utility
            problem.directions[2] = Problem.MAXIMIZE  # inertia
            problem.directions[3] = Problem.MAXIMIZE  # reliability

            track_progress = TrackProgress()
            # algorithm = EpsNSGAII(problem, epsilons=epsilons)
            # algorithm.run(100000, track_progress)

            with ProcessPoolEvaluator() as evaluator:
                algorithm = EpsNSGAII(problem, epsilons=epsilons, evaluator=evaluator)
                algorithm.run(100000, track_progress)

            logging.info("storing results")
            store_results(
                algorithm, track_progress, "output", f"{entry.__name__}", seed
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
