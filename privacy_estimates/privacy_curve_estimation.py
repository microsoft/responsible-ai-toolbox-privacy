import numpy as np
from typing import Tuple, Iterable
from sklearn.metrics import roc_curve
from parmap import map as pmap

from privacy_estimates.utils import AttackResults
from privacy_estimates.epsilon_estimation import compute_eps_lo_hi


def _compute_epsilon_max(delta: float, scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, alpha: float,
                         method: str) -> Tuple[float, float]:
    return max([
        compute_eps_lo_hi(
            count=AttackResults.from_scores_threshold_and_labels(attack_scores=scores, threshold=thr, challenge_bits=labels),
            alpha=alpha,
            method=method,
            delta=delta,
        )
        for thr in thresholds
    ])


def compute_privacy_curve_lo_hi(
        attack_scores: Iterable[float], challenge_bits: Iterable[float], deltas: Iterable[float], alpha: float = 0.1,
        method: str = "beta"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the privacy curve for a given set of attack scores and challenge bits.

    Args:
        attack_scores: The attack scores.
        challenge_bits: The challenge bits.
        deltas: The delta values for which to compute the privacy curve.
        alpha: Significance level.
        method: The method to use for computing the privacy curve.
    Returns:
        The bounds on the privacy curve eps_lo(delta), eps_hi(delta).
    """
    _, _, thrs = roc_curve(y_true=challenge_bits, y_score=attack_scores)
    epsilons = list(
        pmap(_compute_epsilon_max, iterable=deltas, scores=attack_scores, thresholds=thrs, labels=challenge_bits,
             alpha=alpha, method=method, pm_pbar=True)
    )
    return np.array([e[0] for e in epsilons]), np.array([e[1] for e in epsilons])
