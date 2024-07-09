import logging
import datetime
import numpy as np
from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from datasets import features as dataset_features
from fractions import Fraction
from dataclasses import dataclass
from enum import Enum
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
from typing import Sequence, Tuple, Optional
from functools import partial, reduce
from collections import Counter
from pathlib import Path


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SplitType(Enum):
    RANDOM_SPLITS = "random_splits" 
    ROTATING_SPLITS = "rotating_splits" 


def empty_dataset(features: dataset_features.Features) -> Dataset:
    return Dataset.from_dict({k: [] for k in features}, features=features)


@dataclass
class InOutIndices:
    """
    Class representing input-output indices for shadow model statistics.
    """

    in_: Sequence[Tuple[str, int]]
    out: Sequence[Tuple[str, int]]

    @classmethod
    def from_sequences(cls, in_: Sequence[Tuple[str, int]], out: Sequence[Tuple[str, int]],
                       pad_to_length: Optional[int] = None):
        """
        Create an instance of InOutIndices from input and output sequences.

        Args:
            in_: Input sequence.
            out: Output sequence.
            pad_to_length: Optional length to pad the sequences to.

        Returns:
            An instance of InOutIndices.
        """
        if pad_to_length is not None:
            assert len(in_) <= pad_to_length
            assert len(out) <= pad_to_length
            in_ = list(in_)
            out = list(out)
            in_.extend([(None, None)] * (pad_to_length - len(in_)))
            out.extend([(None, None)] * (pad_to_length - len(out)))
        return cls(in_=in_, out=out)

    def __post_init__(self):
        assert len(self.in_) == len(self.out)

    def extend(self, other: "InOutIndices"):
        """
        Extend the current instance with another InOutIndices instance.

        Args:
            other: Another InOutIndices instance to extend with.
        """
        self.in_.extend(other.in_)
        self.out.extend(other.out)

    def __len__(self):
        return len(self.in_)

    def as_dataset(self, features: Optional[dataset_features.Features] = None) -> DatasetDict:
        return DatasetDict({
            "in": Dataset.from_dict(
                {"split": [s for s, _ in self.in_], "sample_index": [i for _, i in self.in_]}, features=features
            ),
            "out": Dataset.from_dict(
                {"split": [s for s, _ in self.out], "sample_index": [i for _, i in self.out]}, features=features
            ),
        })


def get_least_common_element(sequence: Sequence[Tuple[str, int]]) -> Tuple[Tuple[str, int], int]:
    """
    Get the least common element in a sequence.

    Args:
        sequence (Sequence): The input sequence.

    Returns:
        Tuple: The least common element and its number of occurrences.
    """
    sequence = [e for e in sequence if e != (None, None)]
    return Counter(sequence).most_common()[-1]


def check_in_out_indices(in_: Dataset, out: Dataset):
    in_indices = list(zip(in_["split"], in_["sample_index"]))
    out_indices = list(zip(out["split"], out["sample_index"]))

    is_offline_setting = (set(in_indices) == {(None, None)})
    # offline setting is if we don't have any shadow models with canaries present
    # we need to infer all mi info from out samples
    if is_offline_setting:
        logger.info("Offline setting detected. All samples are out samples. This will reduce the number of checks")

    if set(in_indices).difference([(None, None)]) != set(out_indices).difference([(None, None)]) and not is_offline_setting:
        # calculate missing indices
        missing_indices = set(in_indices).symmetric_difference(out_indices)
        raise ValueError(f"Some indices are missing: {missing_indices}")

    if not is_offline_setting:
        least_common_in, least_common_in_occ = get_least_common_element(in_indices)
        logger.info(f"Least common in element: {least_common_in} ({least_common_in_occ} times)")
        if least_common_in_occ < 4:
            logger.error(f"Lest common element occurs less than 4 times: {least_common_in} ({least_common_in_occ} times). "
                         "This might lead to a weak attack")
        if least_common_in_occ < 10:
            logger.warning(f"Lest common element occurs less than 10 times: {least_common_in} ({least_common_in_occ} times)")

    least_common_out, least_common_out_occ = get_least_common_element(out_indices)
    logger.info(f"Least common out element: {least_common_out} ({least_common_out_occ} times)")
    if least_common_out_occ < 4:
        logger.error(f"Lest common element occurs less than 4 times: {least_common_out} ({least_common_out_occ} times). "
                     "This might lead to a weak attack")
    if least_common_out_occ < 10:
        logger.warning(f"Lest common element occurs less than 10 times: {least_common_out} ({least_common_out_occ} times)")


def compute_in_out_indices_using_cross_validation(indices: Sequence, in_fraction: float, seed: int) -> Tuple[InOutIndices, int]:
    """
    Compute in and out indices using cross-validation.

    Args:
        indices (Sequence): The input indices.
        in_fraction (float): The fraction of indices to be used for training.
        seed (int): The random seed for shuffling the indices.

    Returns:
        Tuple[InOutIndices, int]: A tuple containing the cross-validation sets and the number of samples per model.
    """
    assert 0 <= in_fraction <= 1, "in_fraction must be between 0 and 1"
    in_fraction = Fraction(in_fraction)
    assert in_fraction.denominator < 10, "in_fraction denominator must be less than 10. Select a more divisible number."

    # shuffle indices with seed
    np.random.default_rng(seed+128023).shuffle(indices)

    folds = [[indices[i] for i in f] for f in np.array_split(range(len(indices)), in_fraction.denominator)]

    n_samples_per_model = max(
        sum(sorted((len(fold) for fold in folds), reverse=True)[:in_fraction.numerator]),
        sum(sorted((len(fold) for fold in folds), reverse=True)[in_fraction.numerator:])
    )

    assert len(folds) == in_fraction.denominator

    cross_validation_sets = InOutIndices(in_=[], out=[])
    for fold_i in range(in_fraction.denominator):
        rolled_indices = [folds[(fold_i + i) % len(folds)] for i in range(len(folds))]

        cross_validation_sets.extend(
            InOutIndices.from_sequences(
                in_=reduce(lambda a, b: a+b, rolled_indices[1:], []),
                out=rolled_indices[0],
                pad_to_length=n_samples_per_model
            )
        )
    return cross_validation_sets, n_samples_per_model


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
})
def create_in_out_data_for_shadow_model_statistics(
    in_out_data: Input,
    in_indices: Output,
    out_indices: Output,
    num_points_per_model: Output(type="uri_file"),  # noqa: F821
    seed: int,
    split_type: str,
    in_fraction: float,
    max_num_models: int = 1_024,
):
    """
    Creates in and out indices for shadow model statistics.

    Args:
        in_out_data (Input): The input data.
        in_indices (Output): The output file path to save the in indices.
        out_indices (Output): The output file path to save the out indices.
        num_points_per_model (Output): The output file path to save the number of in samples per model.
        seed (int): The seed for random number generation.
        split_type (str): The type of split to use.
        in_fraction (float): The fraction of in samples.
        max_num_models (int, optional): The maximum number of models. Defaults to 1_024.
    """
    fmt = f"%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler()])
    tqdm_logging.set_level(logging.INFO)
    tqdm_logging.set_log_rate(datetime.timedelta(seconds=10))  

    # Load the input data
    ds = load_from_disk(in_out_data)
    logger.info(f"Loaded data with {len(ds)} points")
    index_features = dataset_features.Features({
        "split": ds.features["split"],
        "sample_index": ds.features["sample_index"],
    })
    logger.info(f"Loaded index features: {index_features}")

    # Check if required columns exist in the dataset
    assert "sample_index" in ds.column_names, "ds must have a column named 'sample_index'"
    assert "split" in ds.column_names, "ds must have a column named 'split'"

    # Create indices from split and sample_index columns
    indices = [(split, i) for split, i in zip(ds["split"], ds["sample_index"])]

    split_type = SplitType(split_type)

    if split_type == SplitType.RANDOM_SPLITS:
        raise NotImplementedError("Random splits not implemented")
    elif split_type == SplitType.ROTATING_SPLITS:
        compute_in_out_indices_for_models = partial(compute_in_out_indices_using_cross_validation, in_fraction=in_fraction)

    rng = np.random.default_rng(seed+129120)
    in_out_indices_i, points_per_model = compute_in_out_indices_for_models(indices=indices, seed=rng.integers(0, 2**32-1))
    models_per_iter = len(in_out_indices_i)//points_per_model
    in_out_indices_dsd = [in_out_indices_i.as_dataset()]
    for _ in tqdm(range(models_per_iter, max_num_models, models_per_iter), desc="Computing in-out indices"):
        in_out_indices_i, points_per_model_i = compute_in_out_indices_for_models(indices=indices, seed=rng.integers(0, 2**32-1))
        assert points_per_model == points_per_model_i, "Number of points per model must be the same for all models"
        in_out_indices_dsd.append(in_out_indices_i.as_dataset(features=index_features))

    indices_in_ds = concatenate_datasets([dsd["in"] for dsd in in_out_indices_dsd])
    indices_out_ds = concatenate_datasets([dsd["out"] for dsd in in_out_indices_dsd])

    logger.info("Number of in samples per model: %s", points_per_model)
    logger.info("Number of total samples: %s", len(indices_in_ds))

    # Save the number of in samples per model to a file
    with open(num_points_per_model, "w") as f:
        f.write(str(points_per_model))

    check_in_out_indices(in_=indices_in_ds, out=indices_out_ds)

    # Create a dataset for in indices and save it to disk
    indices_in_ds.save_to_disk(in_indices)
    logger.info(f"Saved {len(indices_in_ds)} in indices to {in_indices}")

    # Create a dataset for out indices and save it to disk
    indices_out_ds.save_to_disk(out_indices)
    logger.info(f"Saved {len(indices_out_ds)} out indices to {out_indices}")

