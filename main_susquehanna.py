# ===========================================================================
# Name        : main_susquehanna.py
# Author      : MarkW, adapted from JazminZ & MatteoG
# Version     : 0.06
# Copyright   : Your copyright notice
# ===========================================================================
import numpy as np
import pandas as pd

from susquehanna_model import susquehanna_model
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import random
import os
import csv
import logging

logging.basicConfig(level=logging.INFO)


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
        df_imp = pd.DataFrame.from_dict(dict(nfe=self.nfe, improvements=self.improvements))
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


track_progress = TrackProgress()


def store_results(algorithm, output_dir, base_file_name):
    header = ["hydropower", "atomicpowerplant", "baltimore", "chester", "environment", "recreation"]
    with open(f"{output_dir}/{base_file_name}_solution.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for solution in algorithm.result:
            writer.writerow(solution.objectives)

    with open(f"{output_dir}/{base_file_name}_variables.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.variables)


def main():
    seeds = [10]  # , 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for seed in seeds:
        # set seed
        random.seed(seed)
        # RBF parameters
        RBFType = "invmultiquadric"
        numberOfInput = 2  # (time, storage of Conowingo)
        numberOfOutput = 4  # Atomic, Baltimore,Chester, Downstream:- hydropower, environmental
        numberOfRBF = 4  # numberOfInput + 2

        # Initialize model
        nobjs = 6
        nvars = int(numberOfRBF * 8)  # 8 = 2 centers + 2 radius + 4 weights
        n_years = 1
        susquehanna_river = susquehanna_model(108.5, 505.0, 5, n_years)  # l0, l0_MR, d0, years
        # l0 = start level cono, l0_MR = start level muddy run, d0 = startday > friday = 5
        susquehanna_river.load_data(0)  # 0 = historic, 1 = stochastic
        susquehanna_river.set_log(False)
        susquehanna_river.setRBF(numberOfRBF, numberOfInput, numberOfOutput, RBFType)

        # Lower and Upper Bound for problem.types
        LB = [-1, 0, -1, 0, 0, 0, 0, 0] * numberOfRBF
        UB = [1, 1, 1, 1, 1, 1, 1, 1] * numberOfRBF
        EPS = [0.5, 0.05, 0.05, 0.05, 0.05, 0.001]

        # platypus MOEA, no contraints
        problem = Problem(nvars, nobjs)
        # problem.types[:] = Real(-1, 1)
        problem.types[:] = [Real(LB[i], UB[i]) for i in range(nvars)]
        problem.function = susquehanna_river.evaluate  # historical (deterministic) optimization

        problem.directions[0] = Problem.MINIMIZE  # hydropower
        problem.directions[1] = Problem.MINIMIZE  # atomicpowerplant
        problem.directions[2] = Problem.MINIMIZE  # baltimore
        problem.directions[3] = Problem.MINIMIZE  # chester
        problem.directions[4] = Problem.MAXIMIZE  # environment
        problem.directions[5] = Problem.MINIMIZE  # recreation

        # algorithm = EpsNSGAII(problem, epsilons=EPS)
        # algorithm.run(1000)

        with ProcessPoolEvaluator() as evaluator:
            algorithm = EpsNSGAII(problem, epsilons=EPS, evaluator=evaluator)
            algorithm.run(1000, callback=track_progress)

        df_conv, df_hv = track_progress.to_dataframe()

        # results
        print("results:")
        for solution in algorithm.result:
            print(solution.objectives)

        # save results
        df_conv.to_csv(f"output/{RBFType}_{seed}_convergence.csv")
        df_hv.to_csv(f"output/{RBFType}_{seed}_hypervolume.csv")

        store_results(algorithm, "output", f"{RBFType}_{seed}")


if __name__ == "__main__":
    if not os.path.exists("output"):
        try:
            os.mkdir("output")
        except OSError:
            print("Creation of the directory failed")
    main()
