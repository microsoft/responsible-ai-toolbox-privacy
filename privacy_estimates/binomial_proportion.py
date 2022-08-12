import numpy as np
from typing import Tuple
from statsmodels.stats.proportion import proportion_confint

from privacy_estimates.utils import AttackResults
from privacy_estimates.privacy_region import eps_from_fnr_fpr


def compute_eps_lo_hi(count: AttackResults, delta: float, alpha: float, method: str) -> Tuple[float, float]:
    """
    Computes a confidence interval for epsilon from equal-tailed confidence intervals for the false positive and
    false negative rates of a membership inference attack

    Args:
        count (AttackResults):
            Number of false positives, false negatives, true positives, and true negatives
        delta (float):
            Differential privacy :math:`\delta` parameter
        alpha (float):
            Significance level
        method ({'beta', 'jeffreys'}):
            Method to use for confidence intervals.
            We use central (aka equal-tailed) methods available from `statsmodels`.
            - `beta` : Clopper-Pearson interval based on Beta distribution
            - `jeffreys` : Jeffreys Bayesian Interval
            Default is 'jeffreys'

    Returns:
        eps_lo, eps_hi (Tuple[float, float]): two-sided interval [eps_lo, eps_hi] with confidence 100*(1 - alpha)%.
    """
    fpr = count.FPR 
    fnr = count.FNR

    # Compute confidence intervals for the False Positive and False Negative Rates
    # These will be used to compute a confidence interval for epsilon from the hypothesis testing characterization
    # of differential privacy (Kairouz et al. 2015).

    # Here we use significance alpha/2 so to obtain an interval for epsilon 
    # with significance at least alpha
    fpr_l, fpr_r = proportion_confint(count.FP, count.N, alpha/2, method)
    fnr_l, fnr_r = proportion_confint(count.FN, count.P, alpha/2, method)

    # statsmodel.proportion.proportion_confint doesn't correct for the coverage of Jeffreys intervals tending to 0
    # when count=nobs or count=0.
    # For instance, the lower endpoint of `proportion_confint(count=0, nobs=100, method='jeffreys')` is not 0.
    # We correct it here (see Astropy's implementation https://docs.astropy.org/en/stable/api/astropy.stats.binom_conf_interval.html)
    # The approximate coverage when this is corrected is larger than the nominal coverage: e.g. 1-alpha instead of 1-2*alpha.
    if method == 'jeffreys':
        if count.FP == 0:
            fpr_l = 0
        if count.TN == 0:
            fpr_r = 1
        if count.FN == 0:
            fnr_l = 0
        if count.TP == 0:
            fnr_r = 1

    # Sanity checks
    assert (fpr_l <= fpr and fpr <= fpr_r)
    assert (fnr_l <= fnr and fnr <= fnr_r)

    # Estimate confidence interval for epsilon

    # If the rectangle is straddeling the 1-x diagonal then the lower epsilon has to be 0
    if (fnr_r > 1-fpr_r) != (fnr_l > 1-fpr_l):
        eps_lo = 0.0
    else:
        eps_lo = float('inf')

    # From the union bound, the one-sided interval has at least 100*(1 - alpha)% confidence.
    eps_r = eps_from_fnr_fpr(fnr=fnr_r, fpr=fpr_r, delta=delta)
    eps_l = eps_from_fnr_fpr(fnr=fnr_l, fpr=fpr_l, delta=delta)

    eps_lo = min(eps_l, eps_r, eps_lo)
    eps_hi = max(eps_l, eps_r)

    return eps_lo, eps_hi


def compute_eps_lo(count: AttackResults, delta: float, alpha: float, method: str) -> float:
    """
    Compute a one-sided [eps_lo, ∞) confidence interval for epsilon from equal-tailed confidence intervals for the false positive and
    false negative rates of a membership inference attack

    Args:
        count (AttackResults):
            Number of false positives, false negatives, true positives, and true negatives
        delta (float):
            Differential privacy :math:`\delta` parameter
        alpha (float):
            Significance level
        method ({'beta', 'jeffreys'}):
            Method to use for confidence intervals.
            We use central (aka equal-tailed) methods available from `statsmodels`.
            - `beta` : Clopper-Pearson interval based on Beta distribution
            - `jeffreys` : Jeffreys Bayesian Interval
            Default is 'jeffreys'

    Returns:
        eps_lo (float): lower endpoint of one-sided interval [eps_lo, ∞) with confidence 100*(1 - alpha)%.
    """

    # `compute_eps_lo_hi` computes two-sided intervals. We compute one-sided intervals with significance level `alpha`
    # by computing two-sided intervals with significance level `2*alpha`. We can do this because we use
    # central (aka equal-tailed) confidence intervals, such as those computed using Clopper-Pearson and Jeffreys methods.
    #
    # See https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/proportion.py
    # and http://www-stat.wharton.upenn.edu/~tcai/paper/1sidedCI.pdf
    eps_lo, _ = compute_eps_lo_hi(count=count, delta=delta, alpha=2*alpha, method=method)
    return eps_lo


def compute_eps_hi(count: AttackResults, delta: float, alpha: float, method: str) -> float:
    _, eps_hi = compute_eps_lo_hi(count=count, delta=delta, alpha=2*alpha, method=method)
    return eps_hi
