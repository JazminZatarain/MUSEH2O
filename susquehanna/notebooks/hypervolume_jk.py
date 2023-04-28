import multiprocessing


from platypus import (
    Hypervolume,
    Solution,
    EpsilonBoxArchive,
    EpsilonIndicator,
    GenerationalDistance,
)


class MetricWrapper:
    f"""wrapper class for wrapping platypus indicators

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    kwargs : dict
             any additional keyword arguments to be passed
             on to the wrapper platypus indicator class

    Notes
    -----
    this class relies on multi-inheritance and careful consideration
    of the MRO to conveniently wrap the convergence metrics provided
    by platypus.

    """

    def __init__(self, reference_set, problem, **kwargs):
        self.problem = problem
        reference_set = rebuild_platypus_population(reference_set, self.problem)
        super().__init__(reference_set=reference_set, **kwargs)

    def calculate(self, archive):
        solutions = rebuild_platypus_population(archive, self.problem)
        return super().calculate(solutions)


def rebuild_platypus_population(archive, problem):
    """rebuild a population of platypus Solution instances

    Parameters
    ----------
    archive : DataFrame
    problem : PlatypusProblem instance

    Returns
    -------
    list of platypus Solutions

    """
    solutions = []
    for row in archive.itertuples():
        # decision_variables = [getattr(row, attr) for attr in problem.parameter_names]
        objectives = [getattr(row, attr) for attr in problem.outcome_names]

        solution = Solution(problem)
        # solution.variables = decision_variables
        solution.objectives = objectives
        solutions.append(solution)
    return solutions


def epsilon_nondominated(results, epsilons, problem):
    """Merge the list of results into a single set of
    non dominated results using the provided epsilon values

    Parameters
    ----------
    results : list of DataFrames
    epsilons : epsilon values for each objective
    problem : PlatypusProblem instance

    Returns
    -------
    DataFrame
    Notes
    -----
    this is a platypus based alternative to pareto.py (https://github.com/matthewjwoodruff/pareto.py)
    """
    if problem.nobjs != len(epsilons):
        ValueError(
            f"the number of epsilon values ({len(epsilons)}) must match the number of objectives {problem.nobjs}"
        )

    results = pd.concat(results, ignore_index=True)
    solutions = rebuild_platypus_population(results, problem)
    archive = EpsilonBoxArchive(epsilons)
    archive += solutions

    return to_dataframe(archive, problem.parameter_names, problem.outcome_names)


class HypervolumeMetric(MetricWrapper, Hypervolume):
    """Hypervolume metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around Hypervolume as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    pass


class EpsilonIndicatorMetric(MetricWrapper, EpsilonIndicator):
    """EpsilonIndicator metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around EpsilonIndicator as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    pass


class GenerationalDistanceMetric(MetricWrapper, GenerationalDistance):
    """EpsilonIndicator metric

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around EpsilonIndicator as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    pass


def calculate_hv(archive, platypus_problem, reference_set):
    reference_set = rebuild_platypus_population(reference_set, platypus_problem)
    hv = Hypervolume(reference_set=reference_set)
    archive = rebuild_platypus_population(archive, platypus_problem)
    return hv.calcuate(archive)


if __name__ == "main":
    list_of_archives = []
    reference_set = archive[-1]

    problem = ema_workbench.em_framework.optimization.to_problem(
        model, searchover="levers"
    )
    hv = HypervolumeMetric(reference_set, problem)

    with multiprocessing.Pool(4) as pool:
        pool.map(hv.calculate, list_of_archives)
