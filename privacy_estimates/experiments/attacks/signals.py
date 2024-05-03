import numpy as np

from math import factorial
from abc import ABC, abstractmethod


def get_taylor(logit_signals, n: int):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor



class Signal(ABC):
    @abstractmethod
    def compute_mi_signal(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the membership inference signal from the logits and labels.

        Args:
            logits: The logits of the model.
            labels: The true labels of the data.
        """
        pass


class TaylorSoftMargin(Signal):
    def __init__(self, temp: float, m: float, n: int):
        self.m = m
        self.n = n
        self.temp = temp

    def compute_mi_signal(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        logit_signals = logits/self.temp
        taylor_logits = get_taylor(logit_signals, self.n)
        taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
        true_logit = logit_signals.gather(1, labels.reshape(-1, 1))
        taylor_true_logit = taylor_logits.gather(1, labels.reshape(-1, 1))
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - self.m, self.n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        return soft_taylor_true_logit / taylor_logit_sum


SIGNALS = {cls.__name__: cls for cls in Signal.__subclasses__()}


def compute_mi_signals(logits: np.ndarray, labels: np.ndarray, method: str, **kwargs) -> np.ndarray:
    if method not in SIGNALS:
        raise ValueError(f"Method {method} not found. Available methods: {SIGNALS.keys()}")
    return SIGNALS[method](**kwargs).compute_mi_signal(logits, labels)
