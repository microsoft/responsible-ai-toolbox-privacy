import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
from typing import List, Tuple

from privacy_games.attacks import find_best_percentile, PerModelLossPercentileAttack
from privacy_games.estimates import compute_eps_lo_hi
from privacy_games.utils import AttackResults



class Arguments(BaseModel):
    first_samples_predictions: Path = Field(
        description="Path to the parquet file of the first predictions"
    )
    second_samples_predictions: Path = Field(
        description="Path to the parquet file of the second predictions"
    )
    train_base_predictions: Path = Field(
        description="Path to predictions of each training sample for each model"
    )
    challenge_bits: Path = Field(
        description="Path to bits that determine our challenge points"
    )
    delta: float = Field(
        description="DP-delta of the estimate"
    )
    output_path: Path = Field(
        description="Output path where the attack guesses will be written"
    )


def reshape_train_losses(losses: np.ndarray, train_predictions: pd.DataFrame) -> np.ndarray:
    model_indices = train_predictions["model_index"]
    num_models = len(set(model_indices))
    # Make sure that the model indices are sorted
    assert all(x <= y for x, y in zip(model_indices, model_indices[1:]))
    assert len(losses) == len(train_predictions)
    return np.stack(losses).reshape((num_models,-1), order="C")


def compute_loss(predictions: pd.DataFrame) -> np.ndarray:
    with torch.no_grad():
        losses = torch.nn.CrossEntropyLoss(reduction='none')(
            torch.tensor(predictions.logits.to_list()), torch.tensor(predictions.labels.to_list())
        ).numpy()
    return losses


def main(args: Arguments):
    first_predictions = pd.read_parquet(args.first_samples_predictions)
    second_predictions = pd.read_parquet(args.second_samples_predictions)
    train_predictions = pd.read_parquet(args.train_base_predictions)
    challenge_bits = pd.read_parquet(args.challenge_bits)

    train_losses = compute_loss(train_predictions)
    first_losses = compute_loss(first_predictions)
    second_losses = compute_loss(second_predictions)

    alpha = 0.1

    train_losses = reshape_train_losses(train_losses, train_predictions)

    attack = PerModelLossPercentileAttack(losses=train_losses)

    # Full search using Jeffreys
    def compute_epsilon_jeffreys(guesses: np.ndarray) -> Tuple[float, float]:
        return compute_eps_lo_hi(
            AttackResults.from_guesses_and_labels(guesses, challenge_bits.challenge_bits),
            alpha=alpha, method="jeffreys", delta=args.delta
        )
    best_percentile = find_best_percentile(first_losses, second_losses, percentiles=np.arange(0, 100, 0.1),
                                           attack=attack, estimate_epsilon=compute_epsilon_jeffreys)

    # Focused search using joint beta
    def compute_epsilon_joint_beta(guesses: np.ndarray) -> Tuple[float, float]:
        return compute_eps_lo_hi(
            AttackResults.from_guesses_and_labels(guesses, challenge_bits.challenge_bits),
            alpha=alpha, method="joint-beta", delta=args.delta
        )
    percentile_range = np.arange(best_percentile - 0.5, best_percentile + 0.5, 0.1)
    percentile_range = percentile_range[np.where((0 <= percentile_range) & (percentile_range <= 100))]

    best_percentile = find_best_percentile(first_losses, second_losses, percentiles=percentile_range,
                                           attack=attack, estimate_epsilon=compute_epsilon_joint_beta)

    # Perform attack with best percentile
    attack.set_percentile(best_percentile)
    attack_guesses = attack.make_guesses(first_losses, second_losses)

    attack_guesses = pd.DataFrame(data={"guesses": attack_guesses})
    print(attack_guesses)
    attack_guesses.to_parquet(args.output_path)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
