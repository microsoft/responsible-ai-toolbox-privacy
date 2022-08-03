import numpy as np

from typing import Tuple, Callable, Optional
from numpy import typing as npt
from abc import ABC, abstractmethod
from tqdm import tqdm


class LossThresholdAttack(ABC):
    @abstractmethod
    def make_guesses(self, first_losses: np.ndarray, second_losses: np.ndarray) -> np.ndarray:
        pass


class GlobalLossPercentileAttack(LossThresholdAttack):
    def __init__(self, losses: np.ndarray, percentile: Optional[float] = None):
        assert losses.ndim == 1

        self.losses = losses
        if percentile is not None:
            self.set_percentile(percentile)
        else:
            self.threshold = None

    def set_percentile(self, percentile: float):
        self.threshold = np.percentile(self.losses, percentile)

    def make_guesses(self, first_losses: np.ndarray, second_losses: np.ndarray) -> np.ndarray:
        assert first_losses.ndim == 1
        assert second_losses.ndim == 1
        assert len(first_losses) == len(second_losses)

        if self.threshold is None:
            raise ValueError("No percentile set. Call set_percentile() first.")

        guesses = np.where(first_losses <= self.threshold, 1, 0)
        assert guesses.shape == first_losses.shape
        return guesses


class PerModelLossPercentileAttack(LossThresholdAttack):
    def __init__(self, losses: np.ndarray, percentile: Optional[float] = None):
        assert losses.ndim == 2

        self.attacks = [GlobalLossPercentileAttack(losses[i], percentile=percentile) for i in range(losses.shape[0])]

    def make_guesses(self, first_losses: np.ndarray, second_losses: np.ndarray) -> np.ndarray:
        assert first_losses.ndim == 1
        assert second_losses.ndim == 1
        assert len(first_losses) == len(second_losses)
        assert len(first_losses) % len(self.attacks) == 0
        n_cp = len(first_losses) // len(self.attacks)
        guesses = []
        for i, attack in enumerate(self.attacks):
            guesses.append(
                attack.make_guesses(first_losses[i*n_cp:(i+1)*n_cp], second_losses[i*n_cp:(i+1)*n_cp])
            )
        return np.concatenate(guesses)

    def set_percentile(self, percentile: float):
        for a in self.attacks:
            a.set_percentile(percentile)


def find_best_percentile(first_losses: np.ndarray, second_losses: np.ndarray, percentiles: npt.ArrayLike,
                         attack: LossThresholdAttack, estimate_epsilon: Callable[[np.ndarray], Tuple[float, float]],
                         progress: bool = True) -> float:
    """
    Find the best percentile for a loss percentile attack

    :param first_losses: Array of the first losses
    :param second_losses: Array of the second losses
    :param percentiles: Array of the percentiles to search over
    :param progress: Whether to show a progress bar
    """
    best_percentile = percentiles[0]
    best_eps = 0.0
    percentiles_pbar = tqdm(percentiles, disable=not progress)

    for percentile in percentiles_pbar:
        attack.set_percentile(percentile)
        guesses = attack.make_guesses(first_losses, second_losses)

        eps_lo, eps_hi = estimate_epsilon(guesses)
        percentiles_pbar.set_postfix({
            "percentile": f"{percentile:.2f}",
            "ε_lo": f"{eps_lo:.4f}",
            "ε_hi": f"{eps_hi:.4f}",
            "ε_lo_best": f"{best_eps:.4f}",
            "percentile_best": f"{best_percentile:.2f}",
        })

        if eps_lo > best_eps:
            best_percentile = percentile
            best_eps = eps_lo
    return best_percentile

