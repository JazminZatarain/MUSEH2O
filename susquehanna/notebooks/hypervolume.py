import sys
import pandas as pd
import numpy as np
import os
import datetime as DT
from platypus import Solution, Problem, Hypervolume
sys.path.append('')
import rbf_functions
import multiprocessing  # is this already imported with Platypus?

rbfs = [rbf_functions.original_rbf,
        rbf_functions.squared_exponential_rbf,
        rbf_functions.inverse_quadratic_rbf,
        rbf_functions.inverse_multiquadric_rbf,
        rbf_functions.exponential_rbf,
        rbf_functions.matern32_rbf,
        rbf_functions.matern52_rbf,
       ]

nfearchive = {}

for n in rbfs:
    nfearchive[n.__name__] = {}
for entry in rbfs:
    name = entry.__name__
    output_dir = f"../output/{name}/"
    for i in os.listdir(output_dir):
        if i.endswith("_hypervolume.csv"): # hypervolume.csv contains
            archives_by_nfe = pd.read_csv(output_dir + i)
            nfes = archives_by_nfe["Unnamed: 0"].values
            u_nfes = np.unique(nfes)
            selected_nfe = u_nfes[0::10]
            selected_nfe = np.append(selected_nfe, u_nfes[-1::])
            nfearchive[name][int(i.split("_")[0])] = archives_by_nfe.loc[archives_by_nfe['Unnamed: 0'].isin(selected_nfe)]

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

archives = {}
seeds = np.arange(10, 101, 10).tolist()
for n in rbfs:
    archives[n.__name__] = {}
    for i in seeds:
        archives[n.__name__][i] = {}
for entry in rbfs:
    name = entry.__name__
    for s in nfearchive[name]:
        for nfe, generation in nfearchive[name][s].groupby("Unnamed: 0"):
            # we slice from 2, getting rid of the first two columns
            # which contain the NFE and ID of each solution
            archive = []
            for i, row in generation.iloc[:, 2::].iterrows():
                solution = Solution(problem)
                solution.objectives = row
                archive.append(solution)
            archives[name][s][nfe] = archive

ref_dir = "./refsets/"
ref_sets = {}
for n in rbfs:
    name = n.__name__
    ref_sets[name] = {}
    data = pd.read_csv(f'{ref_dir}{name}_refset.csv')
    ref_set = []
    for i, row in data.iterrows():
        solution = Solution(problem)
        solution.objectives = row
        ref_set.append(solution)
    ref_sets[name] = ref_set

data = pd.read_csv(f'{ref_dir}/global_refset.csv')
ref_set = []
for i, row in data.iterrows():
    solution = Solution(problem)
    solution.objectives = row
    ref_set.append(solution)

rbf = 'original_rbf'
tempnfe = {}
temphv = {}
nfe_sets = {}
hv_sets = {}
nfe_sets[rbf] = {}
hv_sets[rbf] = {}
hv = Hypervolume(reference_set=ref_sets[rbf])

# hv = Hypervolume(reference_set=ref_set) #global
print(f"started {rbf} at {DT.datetime.now().strftime('%H:%M:%S')}")

if __name__ == "main":

    for seed in archives[rbf]:
        nfe_sets[rbf][seed] = {}
        hv_sets[rbf][seed] = {}
        s_archives = archives[rbf][seed]
        nfes = []
        hvs = []
        for nfe, archive in s_archives.items():
            print(nfe)
            nfes.append(nfe)

            with multiprocessing.Pool(4) as pool:
                hv_results= pool.map(hv.calculate, archive)
                hvs.append(hv_results)

        nfe_sets[rbf][seed] = nfes
        hv_sets[rbf][seed] = hvs
        tempnfe[seed] = nfes
        temphv[seed] = hvs
        dfhv = pd.DataFrame.from_dict(temphv, orient='index')
        dfnfe = pd.DataFrame.from_dict(tempnfe, orient='index')
        dfhv = dfhv.T
        dfnfe = dfnfe.T
        dfhv.to_csv(f"hv/{rbf}_hv.csv", index=False)
        dfnfe.to_csv(f"hv/{rbf}_nfe.csv", index=False)
        dfhv.to_csv(f"hv_global/{rbf}_hv_all.csv", index=False) #global
        dfnfe.to_csv(f"hv_global/{rbf}_nfe_all.csv", index=False) #global
        print(f"finished seed: {seed} at {DT.datetime.now().strftime('%H:%M:%S')}")


