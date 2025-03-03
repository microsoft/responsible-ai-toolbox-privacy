import numpy as np
import functools
import scipy

from argparse_dataclass import ArgumentParser
from dataclasses import dataclass
from datasets import load_from_disk, Dataset
from typing import Sequence, Hashable, Dict, Tuple, Optional
from pathlib import Path
from multiprocessing import cpu_count
from tqdm_loggable.auto import tqdm


@dataclass
class Arguments:
    challenge_points: Path
    shadow_model_statistics: Path
    scores: Path
    mean_estimator: str


def convert_logit_to_prob(logit: np.ndarray) -> np.ndarray:
    prob = logit - np.max(logit, axis=1, keepdims=True)
    prob = np.array(np.exp(prob), dtype=np.float64)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    return prob


def calculate_lira_statistics_from_logits(
        logits: Sequence[Sequence[float]], labels: Sequence[int], small_value: float = 1e-45
) -> np.ndarray:
    """
    Calculate LiRA statistics from logits.

    Carlini, Chien, Nasr, Song, Terzis, and Tramer. 'Membership Inference Attacks From First Principles'.
    ArXiv:2112.03570 [Cs], 7 December 2021. http://arxiv.org/abs/2112.03570.

    Args:
        logits (np.ndarray): The logits array. It should have shape (N, C), where N is the number of samples and C is the number
                             of classes.
        labels (np.ndarray): The labels array. It should have shape (N,), where N is the number of samples.
        small_value (float, optional): A small value to avoid division by zero. Defaults to 1e-45.

    Returns:
        np.ndarray: The calculated statistics array. It has shape (C,), where C is the number of classes.
    """
    logits = np.array(logits)
    labels = np.array(labels)
    assert labels.ndim == 1, "Labels must be 1-dimensional."
    assert logits.ndim == 2, "Logits must be 2-dimensional."
    assert labels.shape[0] == logits.shape[0], "Number of labels must be equal to the number of logits."
    pred = convert_logit_to_prob(logits)
    n = pred.shape[0]

    p_true = pred[range(n), labels]
    pred[range(n), labels] = 0
    p_other = pred.sum(axis=1)
    return np.log(p_true + small_value) - np.log(p_other + small_value)


def aggregate_lira_statistics(lira_statistics: np.ndarray, mean_estimator: str = "median") -> Tuple[float, float]:
    assert lira_statistics.ndim == 1, "LiRA statistics must be 1-dimensional."
    assert mean_estimator in {"mean", "median"}, "Mean estimator must be one of 'mean' or 'median'."

    func_avg = functools.partial(np.nanmedian if mean_estimator == 'median' else np.nanmean, axis=0)
    avg = func_avg(lira_statistics)

    std = np.nanstd(lira_statistics, axis=0)
    return avg, std


class LiRA:
    """
    Carlini, Chien, Nasr, Song, Terzis, and Tramer. ‘Membership Inference Attacks From First Principles’.
    ArXiv:2112.03570 [Cs], 7 December 2021. http://arxiv.org/abs/2112.03570.
    """

    def __init__(self, mean_in: Dict[Hashable, float], std_in: Dict[Hashable, float], mean_out: Dict[Hashable, float],
                 std_out: Dict[Hashable, float]):
        """
        Initialize the Lira object.

        Args:
            mean_in (Dict[Hashable, float]): A dictionary containing the mean values for the input data.
            std_in (Dict[Hashable, float]): A dictionary containing the standard deviation values for the input data.
            mean_out (Dict[Hashable, float]): A dictionary containing the mean values for the output data.
            std_out (Dict[Hashable, float]): A dictionary containing the standard deviation values for the output data.
        """
        self.mean_in = mean_in
        self.std_in = std_in
        self.mean_out = mean_out
        self.std_out = std_out

    @classmethod
    def from_dataset(cls, challenge_points_statistics: Dataset, mean_estimator: str = "median", fix_variance: bool = False,
                     num_proc: Optional[int] = cpu_count()):
        column_names = challenge_points_statistics.column_names
        required_columns = {"split", "sample_index"}
        if not required_columns.issubset(column_names):
            raise ValueError(
                f"Challenge points statistics must contain columns: {required_columns}. "
                f"Found columns: {column_names}. "
                f"Missing columns: {required_columns - set(column_names)}"
            )
        if mean_estimator not in {"mean", "median"}:
            raise ValueError(f"Mean estimator must be one of 'mean' or 'median'. Got: {mean_estimator}")

        if not {"lira_mean_in", "lira_std_in", "lira_mean_out", "lira_std_out"}.issubset(column_names):
            print("LiRA statistics not found. Calculating LiRA statistics from logits.")

            required_columns = {"logits_in", "logits_out", "label"}
            if not required_columns.issubset(challenge_points_statistics.column_names):
                raise ValueError(
                    f"Challenge points statistics must contain columns: {required_columns}. "
                    f"Found columns: {challenge_points_statistics.column_names}. "
                    f"Missing columns: {required_columns - set(challenge_points_statistics.column_names)}"
                )

            if fix_variance:
                raise NotImplementedError("Fixing variance is not implemented yet.")

            def compute_statistics(logits_in, logits_out, label):
                mean_in, std_in = aggregate_lira_statistics(
                    calculate_lira_statistics_from_logits(logits_in, [label]*len(logits_in)), mean_estimator=mean_estimator
                )
                mean_out, std_out = aggregate_lira_statistics(
                    calculate_lira_statistics_from_logits(logits_out, [label]*len(logits_out)), mean_estimator=mean_estimator
                )
                return {"lira_mean_in": mean_in, "lira_std_in": std_in, "lira_mean_out": mean_out, "lira_std_out": std_out}

            challenge_points_statistics = challenge_points_statistics.map(
                compute_statistics, input_columns=["logits_in", "logits_out", "label"], keep_in_memory=True, num_proc=num_proc,
                desc="Calculating LiRA statistics"
            )
        mean_in = dict()
        std_in = dict()
        mean_out = dict()
        std_out = dict()
        for row in tqdm(challenge_points_statistics, desc="Loading LiRA statistics"):
            mean_in[(row["split"], row["sample_index"])] = row["lira_mean_in"]
            std_in[(row["split"], row["sample_index"])] = row["lira_std_in"]
            mean_out[(row["split"], row["sample_index"])] = row["lira_mean_out"]
            std_out[(row["split"], row["sample_index"])] = row["lira_std_out"]
        return cls(mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

    def compute_lira_score(self, index: Sequence[Hashable], lira_statistics: Sequence[float]) -> Sequence[float]:
        assert len(index) == len(lira_statistics), "Number of indices must be equal to the number of lira statistics."
        mean_in = np.array([self.mean_in[i] for i in index])
        std_in = np.array([self.std_in[i] for i in index])
        mean_out = np.array([self.mean_out[i] for i in index])
        std_out = np.array([self.std_out[i] for i in index])

        lira_statistics = np.array(lira_statistics)

        log_pr_in = scipy.stats.norm.logpdf(lira_statistics, mean_in, std_in + 1e-30)
        log_pr_out = scipy.stats.norm.logpdf(lira_statistics, mean_out, std_out + 1e-30)
        scores = -(log_pr_in - log_pr_out)
        assert len(scores) == len(index), "Number of scores must be equal to the number of indices."
        return scores

    def compute_lira_score_for_dataset(self, challenge_points: Dataset, column_name: str = "lira_score",
                                       num_proc: Optional[int] = cpu_count()) -> Dataset:
        def compute_lira_score(row):
            if "lira_statistic" in row:
                stats = row["lira_statistic"]
            else:
                stats = calculate_lira_statistics_from_logits(row["logits"], row["label"])
            return {
                column_name: self.compute_lira_score(index=list(zip(row["split"], row["sample_index"])), lira_statistics=stats)
            }

        challenge_points = challenge_points.map(compute_lira_score, batched=True, keep_in_memory=True, num_proc=num_proc,
                                                desc="Computing LiRA scores")
        return challenge_points


def main(args: Arguments) -> int:
    shadow_model_statistics = load_from_disk(args.shadow_model_statistics)

    attack = LiRA.from_dataset(shadow_model_statistics, mean_estimator=args.mean_estimator)

    challenge_points: Dataset = load_from_disk(args.challenge_points)

    challenge_points = attack.compute_lira_score_for_dataset(challenge_points=challenge_points, column_name="score")

    scores = challenge_points.select_columns("score")

    scores.save_to_disk(args.scores)

    return 0


if __name__ == "__main__":
    parser = ArgumentParser(Arguments)
    args = parser.parse_args()
    main(args=args)
