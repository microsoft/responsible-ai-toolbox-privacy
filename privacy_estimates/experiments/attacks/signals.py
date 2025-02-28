import numpy as np

from math import factorial
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional


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
    def compute_mi_signal_from_logits(self, logits: np.ndarray, labels: np.ndarray,
                                      completion_mask: Optional[np.ndarray] = None):
        """
        Compute the membership inference signal from the logits and labels.

        Logits should be of shape (n_samples, n_classes) or for sequences of predictions (n_samples, sequence_len, n_classes).
        Labels should be of shape (n_samples,) or for sequences of predictions (n_samples, sequence_len) and range
        from 0 to n_classes-1.

        Args:
            logits: The logits of the model.
            labels: The true labels of the data.
            completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                             signal. If None, all elements are considered.
        Returns:
            The signal for the membership inference attack. of shape (n_samples,) or (n_samples, sequence_len). Larger values
            are evidence for in-membership.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def compute_mi_signal_from_logprobs(self, logprobs: np.ndarray, labels: np.ndarray,
                                        completion_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute the membership inference signal from the log probabilities and labels.

        Args:
            logprobs: The log probabilities of the model.
            labels: The true labels of the data.
            completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                             signal. If None, all elements are considered.
        Returns:
            The signal for the membership inference attack. of shape (n_samples,) or (n_samples, sequence_len). Larger values
            are evidence for in-membership.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def compute_mi_signal_from_probs(self, probs: np.ndarray, labels: np.ndarray,
                                     completion_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Compute the membership inference signal from the probabilities and labels.

        Args:
            probs: The probabilities of the model.
            labels: The true labels of the data.
            completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                             signal. If None, all elements are considered.
        Returns:
            The signal for the membership inference attack. of shape (n_samples,) or (n_samples, sequence_len). Larger values
            are evidence for in-membership.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def assert_inputs_valid(self, logits: np.ndarray, labels: np.ndarray, completion_mask: np.ndarray):
        assert logits.ndim > 1
        assert np.abs(logits.sum(axis=-1)-1).max() > 1e-3, "Logits should be unnormalized, i.e., not softmaxed"
        assert np.abs(np.exp(logits).sum(axis=-1)-1).max() > 1e-3, (
            "log(logits) should be unnormalized, make sure you are not using logprobs"
        )
        assert labels.ndim == logits.ndim - 1
        assert labels[completion_mask].max() < logits.shape[-1]
        assert labels[completion_mask].min() >= 0


class TaylorSoftMargin(Signal):
    def __init__(self, taylor_m: float, taylor_n: int, temp: float = 1.0):
        self.m = taylor_m
        self.n = taylor_n
        self.temp = temp

    def compute_mi_signal_from_logits(self, logits: np.ndarray, labels: np.ndarray,
                                      completion_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        For large `taylor_n` the return values are in [0,1]
        Args:
            logits: The logits of the model. 2 or 3 dimensional tensor.
                    Shape of [n_samples, (seq_len,) n_classes] where seq_len could be 0
            labels: The true labels of the data. Shape of [n_samples, (seq_len)]
            completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                             signal. If None, all elements are considered.
        Returns:
            The signal for the membership inference attack. of shape [n_samples, (seq_len)]. Larger values
            are evidence for in-membership.
        """
        if completion_mask is None:
            completion_mask = np.ones_like(labels)
        completion_mask = completion_mask.astype(bool)
        labels[~completion_mask] = 0
        self.assert_inputs_valid(logits=logits, labels=labels, completion_mask=completion_mask)
        logit_signals = logits/self.temp
        taylor_logits = get_taylor(logit_signals, self.n)
        taylor_logit_sum = taylor_logits.sum(axis=-1)
        true_logit = np.take_along_axis(logit_signals, indices=labels[..., np.newaxis], axis=-1).squeeze(axis=-1)
        taylor_true_logit = np.take_along_axis(taylor_logits, indices=labels[..., np.newaxis], axis=-1).squeeze(axis=-1)
        taylor_logit_sum = taylor_logit_sum - taylor_true_logit
        soft_taylor_true_logit = get_taylor(true_logit - self.m, self.n)
        taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
        signal = (soft_taylor_true_logit / taylor_logit_sum)
        signal[~completion_mask] = np.nan
        return signal


class CrossEntropy(Signal):
    def __init__(self, temp: float = 1.0):
        self.temp = temp
        from torch.nn import CrossEntropyLoss
        self.ignored_index = -100
        self.compute_loss = CrossEntropyLoss(ignore_index=self.ignored_index, reduction="none")

    def compute_mi_signal_from_logits(self, logits: np.ndarray, labels: np.ndarray,
                                      completion_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            logits: The logits of the model. 2 or 3 dimensional tensor.
                    Shape of [n_samples, (seq_len,) n_classes] where seq_len could be 0
            labels: The true labels of the data. Shape of [n_samples, (seq_len)]
            completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                             signal. If None, all elements are considered.
        Returns:
            The signal for the membership inference attack. of shape [n_samples, (seq_len)]. Larger values
            are evidence for in-membership.
        """
        if completion_mask is None:
            completion_mask = np.ones_like(labels)
        completion_mask = completion_mask.astype(bool)
        labels[~completion_mask] = self.ignored_index
        self.assert_inputs_valid(logits=logits, labels=labels, completion_mask=completion_mask)
        logits = logits/self.temp

        # pytorch expects n_classes dimension as the 2nd (i.e. index 1) dimension
        # in ndim=2 this is a noop
        logits = np.swapaxes(logits, 1, -1)

        import torch
        signal = -self.compute_loss(torch.tensor(logits), torch.tensor(labels)).numpy()

        signal[~completion_mask] = np.nan
        return signal


SIGNALS = {cls.__name__: cls for cls in Signal.__subclasses__()}


def compute_mi_signals(predictions: np.ndarray, labels: np.ndarray, method: str, prediction_format: PredictionFormat,
                       completion_mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """
    Compute the membership inference signals for the given method.

    Args:
        predictions: The predictions of the model.
        labels: The true labels of the data.
        method: The method to use for computing the signals.
        prediction_format: The format of the predictions.
        completion_mask: The mask for the completion of the sequence i.e. which tokens will be considered to compute the
                         signal. If None, all elements are considered.
        **kwargs: Additional arguments for the method.
    """
    if method not in SIGNALS:
        raise ValueError(f"Method {method} not found. Available methods: {SIGNALS.keys()}")
    signal_method = SIGNALS[method](**kwargs)
    if prediction_format == PredictionFormat.LOGIT:
        return signal_method.compute_mi_signal_from_logits(predictions=predictions, labels=labels,
                                                           completion_mask=completion_mask)
    elif prediction_format == PredictionFormat.LOGPROB:
        return signal_method.compute_mi_signal_from_logprobs(predictions=predictions, labels=labels,
                                                             completion_mask=completion_mask)
    elif prediction_format == PredictionFormat.PROB:
        return signal_method.compute_mi_signal_from_probs(predictions=predictions, labels=labels,
                                                          completion_mask=completion_mask)
    else:
        raise ValueError(f"Prediction format {prediction_format} not found. Available formats: {PredictionFormat}")
