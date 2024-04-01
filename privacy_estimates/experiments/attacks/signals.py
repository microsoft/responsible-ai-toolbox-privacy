import numpy as np
from abc import ABC, abstractmethod


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


SIGNALS = {cls.__name__: cls for cls in Signal.__subclasses__()}


def compute_mi_signals(logits: np.ndarray, labels: np.ndarray, method: str, **kwargs) -> np.ndarray:
    if method not in SIGNALS:
        raise ValueError(f"Method {method} not found. Available methods: {SIGNALS.keys()}")
    return SIGNALS[method](**kwargs).compute_mi_signal(logits, labels)
