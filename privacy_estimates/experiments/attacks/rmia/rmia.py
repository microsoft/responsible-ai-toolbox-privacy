import numpy as np

from typing import Sequence, Hashable, Mapping
from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk, Dataset


class Arguments(BaseModel):
    mi_statistics: Path
    challenge_points: Path
    scores: Path
    offline_a: float


class RMIA:
    def __init__(self, reference_signals: Mapping[Hashable, float], offline_a: float):
        self.offline_a = offline_a
        self.reference_signals = reference_signals

    @classmethod
    def from_dataset(cls, reference_signals_ds: Dataset, offline_a: float):
        required_columns = {"sample_index", "split", "mi_signal"}
        if not required_columns.issubset(reference_signals_ds.column_names):
            raise ValueError(
                f"Reference signals dataset must contain columns: {required_columns}. "
                f"Found columns: {reference_signals_ds.column_names}. "
                f"Missing columns: {required_columns - set(reference_signals_ds.column_names)}."
            )
        reference_signals = {
            (split, sample_index): mi_signal
            for split, sample_index, mi_signal
            in zip(reference_signals_ds["split"], reference_signals_ds["sample_index"], reference_signals_ds["mi_signal"])
        }
        return RMIA(offline_a=offline_a, reference_signals=reference_signals)
	
    def compute_score(self, index: Sequence[Hashable], target_signals: Sequence[float]) -> np.ndarray:
        target_signals = np.array(target_signals)
        reference_signals = np.array([self.reference_signals[i] for i in index])
        assert len(reference_signals) == len(target_signals)
        assert len(target_signals) == len(index)
        #mean_out_x = np.mean(reference_signals, axis=1)
        mean_out_x = reference_signals # we only have 1 reference model
        mean_x = ((1 + self.offline_a) / 2 * mean_out_x + (1 - self.offline_a) / 2)
        prob_ratio_x = target_signals.ravel() / mean_x
        mia_scores = prob_ratio_x
        return mia_scores


def main(args: Arguments):
    mi_statistics = load_from_disk(args.mi_statistics, keep_in_memory=True)
    challenge_points = load_from_disk(args.challenge_points, keep_in_memory=True)

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
