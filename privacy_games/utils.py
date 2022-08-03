import numpy as np
import opacus_utils
import pandas as pd
import os
import json

from opacus import PrivacyEngine
from opacus_utils import PrivacyEngineCallback
from torch import nn
from prv_accountant import Accountant
from scipy import optimize
from transformers import Trainer, logging
from datasets import Dataset
from typing import Dict, Optional, List, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path


import transformers


logger = logging.get_logger(__name__)


def find_num_steps(sampling_probability: float, noise_multiplier: float, target_epsilon: float, target_delta: float, eps_error: float=0.01, max_steps: int=20000) -> int:
    """
    Find the number of steps that satisfies a given target epsilon.
    """
    acc = Accountant(
        noise_multiplier=noise_multiplier,
        sampling_probability=sampling_probability,
        delta=target_delta,
        max_compositions=max_steps, #WARNING: using num_steps inside compute_epsilon is slower and instable
        eps_error=eps_error
    )

    def compute_epsilon(num_steps: float) -> float:
        return acc.compute_epsilon(int(num_steps))[2]

    try:
        num_steps = optimize.root_scalar(lambda n: compute_epsilon(n) - target_epsilon, bracket=[1, max_steps], xtol=1).root
    except ValueError:
        raise ValueError(f"Could not find num_steps in the given interval [1,max_steps={max_steps}]. Please increase max_steps.") from None
    return int(num_steps)


def setup_privacy_callback(privacy_args: opacus_utils.PrivacyArguments, training_args: opacus_utils.TrainingArguments, model: nn.Module):
    sampling_probability = training_args.train_batch_size/len(ds["train"])
    num_steps = int(np.ceil(1/sampling_probability)*training_args.num_train_epochs)
    target_delta = 1/len(ds['train'])
    noise_multiplier = opacus_utils.dp_utils.find_noise_multiplier(
        sampling_probability=sampling_probability, num_steps=num_steps, target_epsilon=args.privacy.target_epsilon,
        target_delta=target_delta,
        eps_error=0.1
    )
    engine = PrivacyEngine(
        module=model,
        batch_size=training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps,
        sample_size=len(ds['train']),
        noise_multiplier=noise_multiplier,
        max_grad_norm=privacy_args.per_sample_max_grad_norm
    )
    accountant = Accountant(
        noise_multiplier=noise_multiplier, sampling_probability=sampling_probability, max_compositions=num_steps,
        eps_error=0.2, delta=target_delta
    )
    privacy_callback = PrivacyEngineCallback(engine, compute_epsilon=lambda s: accountant.compute_epsilon(s)[2],
                                             max_epsilon=privacy_args.target_epsilon)
    return privacy_callback

def compute_accuracy(p: transformers.EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


@dataclass
class AttackResults:
    """
    Container to store attack results.

    The main purpose of this container is to avoid bugs by mixing up
    the order of the results.
    """
    FN: int
    """Number of false negative results"""

    FP: int
    """Number of false positive results"""

    TN: int
    """Number of true negative results"""

    TP: int
    """Number of true positive results"""

    @property
    def P(self):
        """Number of positive results"""
        return self.TP + self.FN

    @property
    def N(self):
        """Number of negative results"""
        return self.TN + self.FP

    @property
    def accuracy(self):
        """Accuracy of the attack"""
        return (self.TP + self.TN)/(self.P + self.N)

    @property
    def FPR(self):
        """False positive rate"""
        return self.FP/(self.FP + self.TN)
    
    @property
    def FNR(self):
        """False negative rate"""
        return self.FN/(self.TP + self.FN)

    @staticmethod
    def from_guesses_and_labels(attack_guesses: Sequence[int], challenge_bits: Sequence[int]) -> "AttackResults":
        assert len(attack_guesses) == len(challenge_bits)
        if not set(attack_guesses).issubset({0,1}):
            raise ValueError(f"The following incorrect values were found in attack_guesses: {set(attack_guesses)}")
        if not set(challenge_bits).issubset({0,1}):
            raise ValueError(f"The following incorrect values were found in challenge_bits: {set(challenge_bits)}")

        df = pd.DataFrame(data={"guesses": attack_guesses, "labels": challenge_bits})
        df["TP"] = ((df["guesses"] == 1) & (df["labels"] == 1)).astype(np.int32)
        df["TN"] = ((df["guesses"] == 0) & (df["labels"] == 0)).astype(np.int32)
        df["FP"] = ((df["guesses"] == 1) & (df["labels"] == 0)).astype(np.int32)
        df["FN"] = ((df["guesses"] == 0) & (df["labels"] == 1)).astype(np.int32)
        return AttackResults(
            FN=int(df["FN"].sum()),
            FP=int(df["FP"].sum()),
            TN=int(df["TN"].sum()),
            TP=int(df["TP"].sum())
        )

    def to_json(self, path: Path) -> None:
        with path.open("w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def from_json(path: Path) -> "AttackResults":
        with path.open() as f:
            return AttackResults(**json.load(f))
