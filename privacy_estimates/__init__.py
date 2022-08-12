from typing import Tuple

from privacy_estimates.utils import AttackResults
from privacy_estimates import binomial_proportion

from .joint_density import Beta, Dirichlet


def compute_eps_lo(count: AttackResults, delta: float, alpha: float, method: str, bracket: Tuple[float, float] = None) -> float:
    """
    Compute a lower bound on epsilon.

    Returns eps_lo such that eps_lo <= eps with confidence 100*(1 - alpha)%

    Args:
        count (AttackResults):
            Number of false positives, false negatives, true positives, and true negatives
        delta (float):
            Differential privacy :math:`\delta` parameter
        alpha (float):
            Significance level
        method ({'beta', 'jeffreys'}):
            - `beta` : Clopper-Pearson interval based on Beta distribution
            - `jeffreys` : Jeffreys Bayesian Interval
            - `binomials`: confidence region based on a model of FPR and FNR
            as parameters of independent binomial distributions with Jeffreys priors.
            - `multinomial`: confidence region based on a model of P[FP], P[FN], P[TP], P[TN]
            as parameters of a multinomial distribution with Jeffreys priors
        bracket (Tuple[float, float]):
            Used for binomials root finding
    """
    if method in ["jeffreys", "beta"]:
        return binomial_proportion.compute_eps_lo(count=count, delta=delta, alpha=alpha, method=method)
    elif method == "joint-beta":
        return Beta(count=count).eps_lo(delta=delta, alpha=alpha, bracket=bracket)
    elif method == 'joint-dirichlet':
        return Dirichlet(count=count).eps_lo(delta=delta, alpha=alpha, bracket=bracket)
    else:
        raise ValueError("Unknown method: {}".format(method))


def compute_eps_hi(count: AttackResults, delta: float, alpha: float, method: str, bracket: Tuple[float, float] = None) -> float:
    """
    Compute an upper bound on epsilon.

    Returns eps_hi such that eps <= eps_hi with confidence 100*(1 - alpha)%

    Args:
        count (AttackResults):
            Number of false positives, false negatives, true positives, and true negatives
        delta (float):
            Differential privacy :math:`\delta` parameter
        alpha (float):
            Significance level
        method ({'beta', 'jeffreys'}):
            - `beta` : Clopper-Pearson interval based on Beta distribution
            - `jeffreys` : Jeffreys Bayesian Interval
            - `joint-beta`: confidence region based on a model of FPR and FNR
            as parameters of independent binomial distributions with Jeffreys priors.
            - `joint-dirichlet`: confidence region based on a model of P[FP], P[FN], P[TP], P[TN]
            as parameters of a multinomial distribution with Jeffreys priors
        bracket (Tuple[float, float]):
            Used for binomials root finding
    """
    if method in ["jeffreys", "beta"]:
        return binomial_proportion.compute_eps_hi(count=count, delta=delta, alpha=alpha, method=method)
    elif method == "joint-beta":
        return Beta(count=count).eps_hi(delta=delta, alpha=alpha, bracket=bracket)
    elif method == 'joint-dirichlet':
        return Dirichlet(count=count).eps_hi(delta=delta, alpha=alpha, bracket=bracket)
    else:
        raise ValueError("Unknown method: {}".format(method))


def compute_eps_lo_hi(count: AttackResults, delta: float, alpha: float, method: str, bracket: Tuple[float, float] = None) -> Tuple[float, float]:
    """
    Compute lower and upper bounds on epsilon.

    Returns eps_lo such that eps_lo <= eps and eps <= eps_hi each with confidence 100*(1 - alpha/2)%

    Args:
        count (AttackResults):
            Number of false positives, false negatives, true positives, and true negatives
        delta (float):
            Differential privacy :math:`\delta` parameter
        alpha (float):
            Significance level
        method ({'beta', 'jeffreys'}):
            - `beta` : Clopper-Pearson interval based on Beta distribution
            - `jeffreys` : Jeffreys Bayesian Interval
            - `joint-beta`: confidence region based on a model of FPR and FNR as
            parameters of independent binomial distributions with Jeffreys priors.
            - `joint-dirichlet`: confidence region based on a model of P[FP], P[FN], P[TP], P[TN]
            as parameters of a multinomial distribution with Jeffreys priors
        bracket (Tuple[float, float]):
            Used for binomials root finding
    """
    if method in ["jeffreys", "beta"]:
        return binomial_proportion.compute_eps_lo_hi(count=count, delta=delta, alpha=alpha, method=method)
    elif method == "joint-beta":
        return Beta(count).eps_lo_hi(delta=delta, alpha=alpha, bracket=bracket)
    elif method == 'joint-dirichlet':
        return Dirichlet(count=count).eps_lo_hi(delta=delta, alpha=alpha, bracket=bracket)
    else:
        raise ValueError("Unknown method: {}".format(method))
