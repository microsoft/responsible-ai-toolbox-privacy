import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple
from scipy.optimize import root_scalar
from scipy import stats
from scipy import integrate
from scipy.signal import convolve

from privacy_estimates.binomial_proportion import compute_eps_lo_hi
from privacy_estimates.privacy_region import (
    privacy_boundary_hi, privacy_boundary_lo, privacy_boundary_normal_lo, privacy_boundary_normal_hi,
    privacy_velocity_hi, privacy_velocity_lo
)
from privacy_estimates.utils import AttackResults


class DensityModel(ABC):
    def __init__(self, count: AttackResults):
        self.count = count

    @abstractmethod
    def pdf(self, fnr: float, fpr: float) -> float:
        "Probability density function"
        pass

    def probability_private(self, eps: float, delta: float, epsabs: float = 5e-3) -> float:
        "Compute the probability density over the (eps,delta)-DP region"

        # Regions below and above the fpr = 1 - fnr line
        below = lambda fnr: privacy_boundary_lo(fnr=fnr, eps=eps, delta=delta)
        above = lambda fnr: privacy_boundary_hi(fnr=fnr, eps=eps, delta=delta)
        p, _ = integrate.dblquad(lambda y, x: self.pdf(x, y), 0, 1, below, above, epsabs=epsabs)

        return p

    def eps_pdf(self, eps: float, delta: float, epsabs: float = 5e-3) -> float:
        "Compute the probability density for eps"
        def integrand_lo(x):
            return (
                self.pdf(x, privacy_boundary_lo(x, eps, delta))*np.dot(
                    privacy_velocity_lo(x, eps, delta),
                    privacy_boundary_normal_lo(x, eps, delta)
                )
            )
        def integrand_hi(x):
            return (
                self.pdf(x, privacy_boundary_hi(x, eps, delta))*np.dot(
                    privacy_velocity_hi(x, eps, delta),
                    privacy_boundary_normal_hi(x, eps, delta)
                )
            )
        p_lo, _ = integrate.quad(integrand_lo, 0.0, 1.0, epsabs=epsabs)
        p_hi, _ = integrate.quad(integrand_hi, 0.0, 1.0, epsabs=epsabs)
        return p_lo + p_hi

    def mia_pdf(self, alpha: float) -> float:
        """
        Probability density for the membership inference advantage
        """
        raise NotImplementedError("Not yet implemented")

    def eps_lo(self, delta: float, alpha: float=0.05, xtol: float = 1e-2, bracket: Tuple[float, float] = None, max_eps: float = 10.0) -> float:
        "Find the epsilon of the smallest DP region that contains (100*alpha)% density"

        assert(0 < alpha < 1)
        assert(0 <= delta < 1)

        if bracket is not None:
            raise DeprecationWarning("This code is calling (indirectly) `privacy_games.DensitiyModel().eps_lo` with the optional `bracket` parameter "\
                                     "which does not have any effect. Please update your code to eliminate this parameter.")

        def objective(eps):
            return self.probability_private(eps=eps, delta=delta, epsabs=xtol/2) - alpha

        # The (0, delta) DP region already covers (100*alpha)% density
        # `root_scalar` requires the objective function to have different signs at the
        # endpoints of `bracket` and will fail.
        if objective(0) > 0.0:
            return 0.0

        # Narrow down the search using increasingly wider intervals computed using Jeffreys method
        jeffreys_alphas = [0.2, 0.1, 0.01, 0.001]

        res = None
        for jeffreys_alpha in jeffreys_alphas:
            try:
                bracket = list(compute_eps_lo_hi(count=self.count, delta=delta, alpha=jeffreys_alpha, method="jeffreys"))
                bracket[1] = min(bracket[1], max_eps)

                res = root_scalar(objective, bracket=bracket, xtol=xtol)
                break
            except ValueError:
                continue

        if res is None:
            res = root_scalar(objective, bracket=[0, max_eps], xtol=xtol)

        return res.root

    def eps_hi(self, delta: float, alpha: float = 0.05, xtol: float = 1e-2, bracket: Tuple[float, float] = None) -> float:
        "Find the epsilon of the smallest DP region that contains (100*(1-alpha))% density"
        return self.eps_lo(delta=delta, alpha=1-alpha, xtol=xtol, bracket=bracket)

    def eps_lo_hi(self, delta: float, alpha: float = 0.05, xtol: float = 1e-2, bracket: Tuple[float, float] = None) -> Tuple[float, float]:
        eps_lo = self.eps_lo(delta=delta, alpha=alpha/2, xtol=xtol, bracket=bracket)
        eps_hi = self.eps_hi(delta=delta, alpha=alpha/2, xtol=xtol, bracket=bracket)
        return eps_lo, eps_hi


class Beta(DensityModel):
    def __init__(self, count: AttackResults) -> None:
        super().__init__(count)
        self.beta_fnr = stats.beta(0.5 + self.count.FN, 0.5 + self.count.TP)
        self.beta_fpr = stats.beta(0.5 + self.count.FP, 0.5 + self.count.TN)

    def pdf(self, fnr: float, fpr: float) -> float:
        return self.beta_fnr.pdf(fnr) * self.beta_fpr.pdf(fpr)


class Dirichlet(DensityModel):
    def __init__(self, count: AttackResults) -> None:
        super().__init__(count)
        alpha_prior = np.array([0.5, 0.5, 0.5, 0.5])
        self.alpha_posterior = alpha_prior + count

    def _rate_to_prob(self, fnr: float, fpr: float, p_tp: float):
        "Convert false positive and false negative rates to probabilities"
        return [p_tp * fnr / (1 - fnr),
                fpr * (1 - p_tp - fnr) / (1 - fnr),
                p_tp]

    def _det_jac_rate_to_prob(self, fnr: float, p_tp: float) -> float:
        "Determinant of Jacobian of `rate_to_prob`"
        return p_tp * (1 - p_tp - fnr) / (1 - fnr)**3

    def pdf(self, fnr: float, fpr: float, epsabs: float = 5e-3) -> float:
        "Compute joint pdf of (fnr, fpr) by marginalizing p_tp"
        def integrand(p_tp: float) -> float:
            p_fn, p_fp, p_tp = self._rate_to_prob(fnr, fpr, p_tp)
            det = self._det_jac_rate_to_prob(fnr, p_tp)
            return abs(det) * stats.dirichlet(self.alpha_posterior).pdf(x=np.array([p_fn, p_fp, p_tp, 1-p_fn-p_fp-p_tp]))

        # p_tp > 1 - fnr is outside of the range of rate_to_prob
        res, _ = integrate.quad(integrand, 0, 1-fnr, epsabs=epsabs)
        return res

    def probability_private(self, eps: float, delta: float, epsabs: float = 5e-3) -> float:
        """
        Compute the probability density over the (eps,delta)-DP region
    
        Note: Overriden just to be able to set absoulte error in pdf integral
        """
        # Regions below and above the fpr = 1 - fnr line
        below = lambda fnr: privacy_boundary_lo(fnr=fnr, eps=eps, delta=delta)
        above = lambda fnr: privacy_boundary_hi(fnr=fnr, eps=eps, delta=delta)
        probability, _ = integrate.dblquad(lambda fpr, fnr: self.pdf(fnr, fpr, epsabs=epsabs), 0, 1, below, above, epsabs=epsabs)
        return probability
