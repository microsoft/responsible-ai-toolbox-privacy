import numpy as np

from math import factorial
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


def get_taylor(logit_signals, n: int):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor


class PredictionFormat(Enum):
    LOGIT = "logit"
    LOGPROB = "logprob"
    PROB = "prob"



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

    def assert_inputs_valid(self, logits: np.ndarray, labels: np.ndarray):
        assert logits.ndim == 2
        assert np.abs(logits.sum(axis=1)-1).max() > 1e-3, "Logits should be unnormalized, i.e., not softmaxed"
        assert np.abs(np.exp(logits).sum(axis=1)-1).max() > 1e-3, (
            "log(logits) should be unnormalized, make sure you are not using logprobs"
        )
        assert labels.ndim == 1
        assert labels.max() < logits.shape[1]
        assert labels.min() >= 0

    @property
    @abstractmethod
    def valid_prediction_formats(self) -> List[PredictionFormat]:
        return []


class TaylorSoftMargin(Signal):
    def __init__(self, temp: float, taylor_m: float, taylor_n: int):
        self.m = taylor_m
        self.n = taylor_n
        self.temp = temp

    @property
    def valid_prediction_formats(self) -> List[PredictionFormat]:
        return [PredictionFormat.LOGIT]

    def compute_mi_signal(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        For large `taylor_n` the return values are in [0,1]
        """
        self.assert_inputs_valid(logits=predictions, labels=labels)
        logit_signals = predictions/self.temp
        taylor_logits = get_taylor(logit_signals, self.n)
        taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
        true_logit = logit_signals[np.arange(len(labels)), labels].reshape(-1, 1)
        taylor_true_logit = taylor_logits[np.arange(len(labels)), labels].reshape(-1,1)
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - self.m, self.n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        signal = (soft_taylor_true_logit / taylor_logit_sum).reshape(-1)
        return signal
    

SIGNALS = {cls.__name__: cls for cls in Signal.__subclasses__()}


def compute_mi_signals(predictions: np.ndarray, labels: np.ndarray, method: str, prediction_format: PredictionFormat,
                       **kwargs) -> np.ndarray:
    if method not in SIGNALS:
        raise ValueError(f"Method {method} not found. Available methods: {SIGNALS.keys()}")
    signal_method = SIGNALS[method](**kwargs)
    if prediction_format not in signal_method.valid_prediction_formats:
        raise ValueError(f"Method {method} does not support prediction format {prediction_format}")
    return signal_method.compute_mi_signal(predictions=predictions, labels=labels)
