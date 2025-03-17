import numpy as np
import warnings

from typing import Sequence, Hashable, Mapping, Optional, Union
from pathlib import Path
from datasets import load_from_disk, Dataset
from argparse_dataclass import ArgumentParser
from dataclasses import dataclass, field


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


@dataclass
class Arguments:
    mi_statistics: Path
    challenge_points: Path
    scores: Path
    offline_a: float = field(
        default=None,
        metadata={
            "help": "Offline value for a. If provided, mean_in is computed using this value. "
                    "If not provided, mean_in is computed using reference signals."
        },
    )
    use_log_column: Union[str, bool] = field(
        default=False,
        metadata={
            "help": "Use log columns (i.e. `log_mi_signal`) for reference signals. This may be more numerically stable if the "
                    "reference signals are very small as in the probability of a long sequence.",
            "type": str2bool,
        },
    )


class RMIA:
    def __init__(
        self, reference_signals_out: Mapping[Hashable, float],
        reference_signals_in: Optional[Mapping[Hashable, float]] = None,
        offline_a: Optional[float] = None
    ):
        """
        Signals are probabilities P(x|\theta) and are checked to be in [0, 1].
        If they're not in [0,1] a warning is produced but the attack should still work (albeit more heuristically)
        provided that the convention is followed that larger signal represent evidence for in-membership.

        Args:
            reference_signals_out: Reference signals for out-of-distribution samples. A dictionary
                mapping sample indices to reference signals.
            reference_signals_in: Reference signals for in-distribution samples. A dictionary
                mapping sample indices to reference signals. If None, mean_in is computed using
                reference_signals_out.
            offline_a: Offline value for a. If provided, mean_in is computed using this value.
                If not provided, mean_in is computed using reference signals.
        """
        if (reference_signals_in is None) == (offline_a is None):
            raise ValueError("Either reference_signals_in or offline_a must be provided, but not both.")

        if any((v < 0 or 1 < v) for v in reference_signals_out.values()):
            warnings.warn(f"In-reference signals must be in the range [0, 1].", RuntimeWarning)
        if reference_signals_in is not None:
            if any((v < 0 or 1 < v) for v in reference_signals_in.values()):
                warnings.warn(f"Out-reference signals must be in the range [0, 1].", RuntimeWarning)

        if offline_a is not None:
            print("RMIA: Using offline_a for computing mean_in.")
        else:
            print("RMIA: Using reference signals for both in and out.")

        self.offline_a = offline_a
        self.reference_signals_out = reference_signals_out
        self.reference_signals_in = reference_signals_in

    @classmethod
    def from_dataset(cls, reference_signals_ds: Dataset, offline_a: Optional[float] = None, use_log_column: bool = False):
        required_columns = {"sample_index", "split"}
        if use_log_column:
            required_columns |= {"log_mi_signal_log_mean_exp_in", "log_mi_signal_log_mean_exp_out"}
        else:
            required_columns |= {"mi_signal_mean_in", "mi_signal_mean_out"}
        if not required_columns.issubset(reference_signals_ds.column_names):
            raise ValueError(
                f"Reference signals dataset must contain columns: {required_columns}. "
                f"Found columns: {reference_signals_ds.column_names}. "
                f"Missing columns: {required_columns - set(reference_signals_ds.column_names)}."
            )

        if use_log_column:
            signal_column_out = np.exp(np.array(reference_signals_ds["log_mi_signal_log_mean_exp_out"], dtype=np.longdouble))
            signal_column_in = np.exp(np.array(reference_signals_ds["log_mi_signal_log_mean_exp_in"], dtype=np.longdouble))
        else:
            signal_column_out = reference_signals_ds["mi_signal_mean_out"]
            signal_column_in = reference_signals_ds["mi_signal_mean_in"]

        reference_signals_out = {
            (split, sample_index): mi_signal
            for split, sample_index, mi_signal
            in zip(
                reference_signals_ds["split"], reference_signals_ds["sample_index"],
                signal_column_out
            )
        }
        if sum(reference_signals_ds["mi_signal_count_in"]) == 0:
            reference_signals_in = None
        else:
            reference_signals_in = {
                (split, sample_index): mi_signal
                for split, sample_index, mi_signal
                in zip(
                    reference_signals_ds["split"], reference_signals_ds["sample_index"],
                    signal_column_in
                )
            }
        return RMIA(offline_a=offline_a, reference_signals_out=reference_signals_out,
                    reference_signals_in=reference_signals_in)

    def compute_score(self, index: Sequence[Hashable], target_signals: Sequence[float]) -> np.ndarray:
        target_signals = np.array(target_signals)
        reference_signals_in = np.array([self.reference_signals_in[i] for i in index])
        assert len(reference_signals_in) == len(target_signals)
        assert len(target_signals) == len(index)
        mean_out_x = reference_signals_in # we have already computed the mean
        if self.offline_a:
            mean_x = ((1 + self.offline_a) / 2 * mean_out_x + (1 - self.offline_a) / 2)
        else:
            reference_signals_out = np.array([self.reference_signals_out[i] for i in index])
            mean_x = (reference_signals_in + reference_signals_out) / 2
        prob_ratio_x = target_signals.ravel() / mean_x
        mia_scores = prob_ratio_x
        return mia_scores


def main(args: Arguments):
    mi_statistics = load_from_disk(str(args.mi_statistics), keep_in_memory=True)
    challenge_points = load_from_disk(str(args.challenge_points), keep_in_memory=True)

    rmia = RMIA.from_dataset(reference_signals_ds=mi_statistics, offline_a=args.offline_a,
                             use_log_column=args.use_log_column)

    target_indices = list(zip(challenge_points["split"], challenge_points["sample_index"]))

    if args.use_log_column:
        target_signals = np.exp(np.array(challenge_points["log_mi_signal"], dtype=np.longdouble))
    else:
        target_signals = challenge_points["mi_signal"]

    mi_scores = rmia.compute_score(index=target_indices, target_signals=target_signals)

    mi_scores = np.array(mi_scores, dtype=np.double)

    mi_scores_ds = Dataset.from_dict({"score": mi_scores})
    mi_scores_ds.save_to_disk(str(args.scores))


if __name__ == "__main__":
    parser = ArgumentParser(Arguments)
    args = parser.parse_args()
    main(args=args)
