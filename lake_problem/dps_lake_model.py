import math
import numpy as np

from scipy.optimize import brentq

def lake_model(rbf, decision_vars,
    b=0.42,
    q=2.0,
    mean=0.02,
    stdev=0.001,
    delta=0.98,
    alpha=0.4,
    nsamples=100,
    myears=100,
    seed=None,
):
    """runs the lake model for nsamples stochastic realisation using
    specified random seed.

    Parameters
    ----------
    b : float
        decay rate for P in lake (0.42 = irreversible)
    q : float
        recycling exponent
    mean : float
            mean of natural inflows
    stdev : float
            standard deviation of natural inflows
    delta : float
            future utility discount rate
    alpha : float
            utility from pollution
    nsamples : int, optional
    myears : int, optional
    seed : int, optional
           seed for the random number generator

    Returns
    -------
    tuple

    """
    rbf.set_decision_vars(decision_vars)

    np.random.seed(seed)
    Pcrit = brentq(lambda x: x ** q / (1.0 + x ** q) - b * x, 0.01, 1.5)

    X = np.zeros((myears,), dtype=float)
    average_daily_P = np.zeros((myears,), dtype=float)
    reliability = 0.0
    inertia = 0.0
    utility = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        decision = 0.1

        decisions = np.zeros(myears,)
        decisions[0] = decision

        natural_inflows = np.random.lognormal(
            math.log(mean ** 2 / math.sqrt(stdev ** 2 + mean ** 2)),
            math.sqrt(math.log(1.0 + stdev ** 2 / mean ** 2)),
            size=myears,
        )

        for t in range(1, myears):

            # here we use the decision rule
            decision = rbf.apply_rbfs(np.asarray([X[t - 1]]))[0]
            decisions[t] = decision

            X[t] = (
                (1 - b) * X[t - 1]
                + X[t - 1] ** q / (1 + X[t - 1] ** q)
                + decision
                + natural_inflows[t - 1]
            )
            average_daily_P[t] += X[t] / nsamples

        reliability += np.sum(X < Pcrit) / (nsamples * myears)
        inertia += np.sum(np.absolute(np.diff(decisions) < 0.02)) / (nsamples * myears)
        utility += (
            np.sum(alpha * decisions * np.power(delta, np.arange(myears))) / nsamples
        )
    max_P = np.max(average_daily_P)
    return max_P, utility, inertia, reliability
