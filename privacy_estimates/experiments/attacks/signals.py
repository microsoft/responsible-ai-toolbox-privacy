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
    def compute_mi_signal_from_logits(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the membership inference signal from the logits and labels.

        Logits should be of shape (n_samples, n_classes) or for sequences of predictions (n_samples, sequence_len, n_classes).
        Labels should be of shape (n_samples,) or for sequences of predictions (n_samples, sequence_len) and range
        from 0 to n_classes-1.

        Args:
            logits: The logits of the model.
            labels: The true labels of the data.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def compute_mi_signal_from_logprobs(self, logprobs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the membership inference signal from the log probabilities and labels.

        Args:
            logprobs: The log probabilities of the model.
            labels: The true labels of the data.
        """
        raise NotImplementedError("This method should be implemented by the subclass")
    
    def compute_mi_signal_from_probs(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the membership inference signal from the probabilities and labels.

        Args:
            probs: The probabilities of the model.
            labels: The true labels of the data.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def assert_inputs_valid(self, logits: np.ndarray, labels: np.ndarray):
        assert logits.ndim > 1
        assert np.abs(logits.sum(axis=-1)-1).max() > 1e-3, "Logits should be unnormalized, i.e., not softmaxed"
        assert np.abs(np.exp(logits).sum(axis=-1)-1).max() > 1e-3, (
            "log(logits) should be unnormalized, make sure you are not using logprobs"
        )
        assert labels.ndim == logits.ndim - 1
        assert labels.max() < logits.shape[-1]
        assert labels.min() >= 0


class TaylorSoftMargin(Signal):
    def __init__(self, temp: float, taylor_m: float, taylor_n: int):
        self.m = taylor_m
        self.n = taylor_n
        self.temp = temp

    def compute_mi_signal_from_logits(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        For large `taylor_n` the return values are in [0,1]
        Args:
            logits: The logits of the model. 2 or 3 dimensional tensor.
                    Shape of [n_samples, (seq_len,) n_classes] where seq_len could be 0

        Returns:
            The signal for the membership inference attack. of shape [n_samples, (seq_len)]
        """
        self.assert_inputs_valid(logits=logits, labels=labels)
        logit_signals = logits/self.temp
        taylor_logits = get_taylor(logit_signals, self.n)
        taylor_logit_sum = taylor_logits.sum(axis=-1)
        true_logit = np.take_along_axis(logit_signals, indices=labels[..., np.newaxis], axis=-1).squeeze(axis=-1)
        taylor_true_logit = np.take_along_axis(taylor_logits, indices=labels[..., np.newaxis], axis=-1).squeeze(axis=-1)
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - self.m, self.n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        signal = (soft_taylor_true_logit / taylor_logit_sum)
        return signal
    

SIGNALS = {cls.__name__: cls for cls in Signal.__subclasses__()}


def compute_mi_signals(predictions: np.ndarray, labels: np.ndarray, method: str, prediction_format: PredictionFormat,
                       **kwargs) -> np.ndarray:
    if method not in SIGNALS:
        raise ValueError(f"Method {method} not found. Available methods: {SIGNALS.keys()}")
    signal_method = SIGNALS[method](**kwargs)
    if prediction_format == PredictionFormat.LOGIT:
        return signal_method.compute_mi_signal_from_logits(predictions=predictions, labels=labels)
    elif prediction_format == PredictionFormat.LOGPROB:
        return signal_method.compute_mi_signal_from_logprobs(predictions=predictions, labels=labels)
    elif prediction_format == PredictionFormat.PROB:
        return signal_method.compute_mi_signal_from_probs(predictions=predictions, labels=labels)
    else:
        raise ValueError(f"Prediction format {prediction_format} not found. Available formats: {PredictionFormat}")
