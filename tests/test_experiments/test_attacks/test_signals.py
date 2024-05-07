import torch
import numpy as np
import pytest

from math import factorial
from privacy_estimates.experiments.attacks.signals import TaylorSoftMargin


def get_taylor(logit_signals, n):
    power = logit_signals
    taylor = power + 1.0
    for i in range(2, n):
        power = power * logit_signals
        taylor = taylor + (power / factorial(i))
    return taylor


def compute_signal_reference(logits, labels, temp, m, n):
    all_logits = torch.tensor(logits)
    all_true_labels = torch.tensor(labels)
    logit_signals = torch.div(all_logits, temp)
    taylor_logits = get_taylor(logit_signals, n)
    taylor_logit_sum = taylor_logits.sum(axis=1).reshape(-1, 1)
    true_logit = logit_signals.gather(1, all_true_labels.reshape(-1, 1))
    taylor_true_logit = taylor_logits.gather(1, all_true_labels.reshape(-1, 1))
    taylor_logit_sum = taylor_logit_sum - taylor_true_logit
    soft_taylor_true_logit = get_taylor(true_logit - m, n)
    taylor_logit_sum = taylor_logit_sum + soft_taylor_true_logit
    output_signals = torch.div(soft_taylor_true_logit, taylor_logit_sum)
    return torch.flatten(output_signals).numpy()


@pytest.mark.parametrize("temp", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("m", [0.0, 0.5, 0.6])
@pytest.mark.parametrize("n", [1, 5, 20])
def test_taylor_soft_margin(temp, m, n):
    signal_method = TaylorSoftMargin(temp=temp, taylor_m=m, taylor_n=n)

    logits = np.random.rand(10, 5)
    labels = np.random.randint(0, 5, 10)


    reference_signal = compute_signal_reference(logits, labels, temp=temp, m=m, n=n)

    signal = signal_method.compute_mi_signal(logits=logits, labels=labels)

    np.testing.assert_array_almost_equal(signal, reference_signal, decimal=5)
