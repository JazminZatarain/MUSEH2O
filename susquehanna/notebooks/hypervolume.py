from collections import defaultdict
import multiprocessing
import os

import numpy as np
import pandas as pd

from platypus import Problem

from rbf import rbf_functions
from hypervolume_jk import HypervolumeMetric


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

    for entry in rbfs:
        name = entry.__name__
        output_dir = f"../output/{name}/"
        for i in os.listdir(output_dir):
            if i.endswith("_hypervolume.csv"):  # hypervolume.csv contains
                archives_by_nfe = pd.read_csv(output_dir + i)
                nfes = archives_by_nfe["Unnamed: 0"].values
                u_nfes = np.unique(nfes)
                selected_nfe = u_nfes[0::10]
                selected_nfe = np.append(selected_nfe, u_nfes[-1::])

                a = archives_by_nfe.loc[
                    archives_by_nfe["Unnamed: 0"].isin(selected_nfe)
                ]
                generations = []
                for nfe, generation in a.groupby("Unnamed: 0"):
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
                    generations.append((nfe, generation))

                archives[name][int(i.split("_")[0])] = generations
    return archives


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


if __name__ == "__main__":
    archives = load_archives()
    problem = get_platypus_problem()
    ref_sets, global_refset = get_reference_sets()

    overall_results = defaultdict(dict)

    with multiprocessing.Pool(4) as pool:
        for rbf in rbfs[0:1]:
            rbf = rbf.__name__
            reference_set = ref_sets[rbf]
            hv = HypervolumeMetric(reference_set, problem)
            archive = archives[rbf]

            for seed_id, seed_archives in archive.items():
                nfes, seed_archives = zip(*seed_archives[0:10])
                results = pool.map(hv.calculate, seed_archives)

                # you could also just save this directly to csv probably
                overall_results[rbf][seed_id] = pd.DataFrame.from_dict(
                    dict(nfe=nfes, hypervolume=results)
                )

    print(overall_results)
