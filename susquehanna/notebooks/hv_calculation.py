from collections import defaultdict
import datetime
from enum import Enum
from functools import partial
import itertools
import multiprocessing
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from platypus import Problem
from deap.tools import _hypervolume

from rbf import rbf_functions
from hypervolume_jk import EpsilonIndicatorMetric

rbfs = [
    rbf_functions.original_rbf,
    rbf_functions.squared_exponential_rbf,
    rbf_functions.inverse_quadratic_rbf,
    rbf_functions.inverse_multiquadric_rbf,
    rbf_functions.exponential_rbf,
    rbf_functions.matern32_rbf,
    rbf_functions.matern52_rbf,
]


def load_archives():
    """

    Returns
    -------
    a dictionary with the archives for all rbfs.
    For each rbf, there is a dict with the results per seed.
    A result per seed is a list of tuples with nfe and the archive at that nr.
    of nfe

    """

    archives = defaultdict(dict)
    list_of_archives = []

    for entry in rbfs:
        name = entry.__name__
        output_dir = f"../output/{name}/"
        for i in os.listdir(output_dir):
            if i.endswith("_hypervolume.csv"):  # hypervolume.csv contains
                archives_by_nfe = pd.read_csv(output_dir + i)

                generations = []
                for nfe, generation in archives_by_nfe.groupby("Unnamed: 0"):
                    generation = generation.rename(
                        {
                            str(i): name
                            for i, name in enumerate(
                            [
                                "hydropower",
                                "atomicpowerplant",
                                "baltimore",
                                "chester",
                                "environment",
                                "recreation",
                            ]
                        )
                        },
                        axis=1,
                    )
                    # we can drop the first 2 columns. They are not the objectives
                    generation = generation.iloc[:, 2::]
                    generations.append((nfe, generation))
                    list_of_archives.append(generation)

                archives[name][int(i.split("_")[0])] = generations
    return archives, list_of_archives


def get_platypus_problem():
    # setup platypus problem
    n_rbfs = 4
    n_objs = 6
    n_vars = n_rbfs * 8

    problem = Problem(n_vars, n_objs)

    # matters for hypervolume
    problem.directions[0] = Problem.MAXIMIZE  # hydropower
    problem.directions[1] = Problem.MAXIMIZE  # atomic power plant
    problem.directions[2] = Problem.MAXIMIZE  # baltimore
    problem.directions[3] = Problem.MAXIMIZE  # chester
    problem.directions[4] = Problem.MINIMIZE  # environment
    problem.directions[5] = Problem.MAXIMIZE  # recreation

    problem.outcome_names = [
        "hydropower",
        "atomicpowerplant",
        "baltimore",
        "chester",
        "environment",
        "recreation",
    ]

    return problem


def get_reference_sets():
    ref_dir = "./refsets/"
    ref_sets = {}
    for n in rbfs:
        name = n.__name__
        ref_sets[name] = {}
        data = pd.read_csv(f"{ref_dir}{name}_refset.csv")
        ref_sets[name] = data

    global_refset = pd.read_csv(f"{ref_dir}/global_refset.csv")

    return ref_sets, global_refset


class RefSet(Enum):
    GLOBAL = "global"
    LOCAL = "local"


def handle_directions(data, problem):
    data = data.copy()
    for i in range(data.shape[1]):
        column = data[:, i]
        if problem.directions[i] == problem.MAXIMIZE:
            data[:, i] = 1 - column
    return data


def transform_data(data, scaler, problem):
    data = data.copy()
    # setup a scaler

    # scale data
    transformed_data = scaler.transform(data)

    # handle directions
    transformed_data = handle_directions(transformed_data, problem)

    return transformed_data


def calculate_hypervolume(maxima, generation):
    return _hypervolume.hv.hypervolume(generation, maxima)


if __name__ == "__main__":
    archives, list_of_archives = load_archives()

    problem = get_platypus_problem()
    ref_sets, global_refset = get_reference_sets()

    refset = RefSet.GLOBAL

    # we use a global scaler
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(list_of_archives + [global_refset]).values)

    overall_results = {}

    overall_starttime = datetime.datetime.now()

    with multiprocessing.Pool() as pool:
        for rbf in rbfs:
            rbf_starttime = datetime.datetime.now()
            rbf = rbf.__name__

            if refset == RefSet.GLOBAL:
                reference_set = global_refset
            else:
                reference_set = ref_sets[rbf]

            reference_set = transform_data(reference_set.values, scaler, problem)
            maxima = np.max(reference_set, axis=0)

            archive = archives[rbf]
            scores = []
            for seed_id, seed_archives in archive.items():
                nfes, seed_archives = zip(*seed_archives)
                seed_archives = [transform_data(entry.values, scaler, problem) for entry in seed_archives]
                hv_results = pool.map(partial(calculate_hypervolume, maxima), seed_archives)

                scores.append(pd.DataFrame.from_dict(
                    dict(nfe=nfes, hypervolume=hv_results, seed=int(seed_id))
                ))

            # concat into single dataframe per rbf
            scores = pd.concat(scores, axis=0, ignore_index=True)
            scores.to_csv(f"./calculated_metrics/{rbf}_{refset.value}.csv")

            delta = datetime.datetime.now() - rbf_starttime
            print(f"{rbf}: {delta}")

    delta = datetime.datetime.now() - overall_starttime
    print(f"overall: {delta}")
