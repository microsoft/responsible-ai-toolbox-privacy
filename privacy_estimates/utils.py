import numpy as np
import pandas as pd
import json

from pathlib import Path
from typing import Sequence
from dataclasses import dataclass, asdict


def bound_membership_advantage(eps: float, delta: float) -> float:
    """
    Compute the bound on the membership advantage.
    """
    return (np.exp(eps)-1.0+2.0*delta)/(np.exp(eps)+1.0)


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