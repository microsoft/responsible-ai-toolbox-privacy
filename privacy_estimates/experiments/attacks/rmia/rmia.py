import numpy as np

from typing import Sequence, Hashable, Mapping, Optional
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk, Dataset


class Arguments(BaseModel):
    mi_statistics: Path
    challenge_points: Path
    scores: Path
    offline_a: float = Field(
        default=None,
        description="Offline value for a. If provided, mean_in is computed using this value. "
                    "If not provided, mean_in is computed using reference signals."
    )


class RMIA:
    def __init__(
        self, reference_signals_out: Mapping[Hashable, float],
        reference_signals_in: Optional[Mapping[Hashable, float]] = None,
        offline_a: Optional[float] = None
    ):
        """
        Signals are probabilities P(x|\theta) and are checked to be in [0, 1].

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
    
        if any((v<0 or 1<v) for v in reference_signals_out.values()):
            raise ValueError(f"In-reference signals must be in the range [0, 1].")
        if reference_signals_in is not None:
            if any((v<0 or 1<v) for v in reference_signals_in.values()):
                raise ValueError(f"Out-reference signals must be in the range [0, 1].")

        if offline_a is not None:
            print("RMIA: Using offline_a for computing mean_in.")
        else:
            print("RMIA: Using reference signals for both in and out.")

        self.offline_a = offline_a
        self.reference_signals_out = reference_signals_out
        self.reference_signals_in = reference_signals_in

    @classmethod
    def from_dataset(cls, reference_signals_ds: Dataset, offline_a: Optional[float] = None):
        required_columns = {"sample_index", "split", "mi_signal_mean_in", "mi_signal_mean_out"}
        if not required_columns.issubset(reference_signals_ds.column_names):
            raise ValueError(
                f"Reference signals dataset must contain columns: {required_columns}. "
                f"Found columns: {reference_signals_ds.column_names}. "
                f"Missing columns: {required_columns - set(reference_signals_ds.column_names)}."
            )

        reference_signals_out = {
            (split, sample_index): mi_signal
            for split, sample_index, mi_signal
            in zip(
                reference_signals_ds["split"], reference_signals_ds["sample_index"],
                reference_signals_ds["mi_signal_mean_out"]
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
                    reference_signals_ds["mi_signal_mean_in"]
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

    rmia = RMIA.from_dataset(reference_signals_ds=mi_statistics, offline_a=args.offline_a)

    target_indices = list(zip(challenge_points["split"], challenge_points["sample_index"]))
    target_signals = challenge_points["mi_signal"]

    mi_scores = rmia.compute_score(index=target_indices, target_signals=target_signals)

    mi_scores_ds = Dataset.from_dict({"score": mi_scores})
    mi_scores_ds.save_to_disk(args.scores)


def exception_handler(ex):
    raise RuntimeError("An error occurred while running the script.") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
