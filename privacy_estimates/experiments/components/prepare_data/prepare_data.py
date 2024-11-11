import numpy as np
import logging
import os

from mldesigner import command_component, Input, Output
from pathlib import Path
from enum import Enum
from datasets import load_from_disk, concatenate_datasets, Dataset, features
from collections import Counter
from dataclasses import dataclass


ENV = {
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
}


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SampleSelectionMethod(Enum):
    INDEPENDENT = "independent"
    """This method samples `num_points_per_artifact` from `in_data_indices` with replacement indpendently for each artifact"""

    PARTITIONED = "partitioned"
    """This method uses `artifact_index` to determine which `num_points_per_artifact` to sample from `in_data_indices`"""


class MergeUnusedSamplesMethod(Enum):
    NONE = "none"
    """This method does not merge unused samples with the train/validation data"""

    ALL_WITH_TRAIN = "all_with_train"
    """This method merges all unused samples with the train data"""

    IN_WITH_TRAIN_OUT_WITH_VAL = "in_with_train_out_with_val"
    """This method merges unused in samples with the train data and unused out samples with the validation data"""


def null_row_dataset(features: features.Features) -> Dataset:
    return Dataset.from_dict({k: [None] for k in features}, features=features)


@dataclass
class PreparedData:
    in_data_for_artifact: Dataset
    out_data_for_artifact: Dataset
    train_data_for_artifact: Dataset
    validation_data_for_artifact: Dataset


def _prepare_data(train_base_ds: Dataset, validation_base_ds: Dataset, in_out_ds: Dataset, in_sample_indices_ds: Dataset,
                  out_sample_indices_ds: Dataset, artifact_index: int, seed: int, num_points_per_artifact: int, sample_selection: str,
                  merge_unused_samples: str, num_repetitions: int = 1) -> PreparedData:
    """
    Component that prepare data for training a artifact

    Args:
        train_base_data (Input): Dataset that will be in every artifact's training data
        in_out_data (Input): Dataset that will be used to create the in and out data for each artifact
        in_indices (Input): Indices of the in data for each artifact
        out_indices (Input): Indices of the out data for each artifact
        artifact_index (int): Index of the artifact
        seed (int): Seed for the random number generator
        num_points_per_artifact (int): Number of points to sample for each artifact
        in_data_for_artifact (Output): Path to save the in data for the artifact
        out_data_for_artifact (Output): Path to save the out data for the artifact
        train_data_for_artifact (Output): Path to save the train data for the artifact
        sample_selection (str, optional): Method for selecting samples. Defaults to "with_replacement".
        merge_unused_samples_with_train_data (bool, optional): Whether to merge unused samples with the train data.
                                                               Defaults to False.
        num_repetitions (int, optional): Number of times to repeat the inserted samples in the train data. Defaults to 1.
    """
    sample_selection = SampleSelectionMethod(sample_selection)
    merge_unused_samples = MergeUnusedSamplesMethod(merge_unused_samples)

    assert "sample_index" in in_out_ds.features, "train_base_data must have a 'sample_index' feature"
    assert "split" in in_out_ds.features, "train_base_data must have a 'split' feature"

    # Append a null row to the dataset to point to when a sample is not used
    in_out_ds = concatenate_datasets([in_out_ds, null_row_dataset(features=in_out_ds.features)])

    # Create mapping from (split, sample_index) to row no in in_out_ds
    sample_index_to_base_index = {
        (split, sample_index): base_index
        for base_index, (split, sample_index)
        in enumerate(zip(in_out_ds["split"], in_out_ds["sample_index"]))
    }

    logger.info(f"Length of in_sample_indices_ds: {len(in_sample_indices_ds)}")
    logger.info(f"Length of out_sample_indices_ds: {len(out_sample_indices_ds)}")

    assert len(in_sample_indices_ds) == len(out_sample_indices_ds), (
        "in_data_indices and out_data_indices must have the same length"
    )

    if sample_selection == SampleSelectionMethod.INDEPENDENT:
        rows_for_artifact = np.random.default_rng(seed=seed).choice(np.arange(len(in_sample_indices_ds)), num_points_per_artifact,
                                                                 replace=True)
    elif sample_selection == SampleSelectionMethod.PARTITIONED:
        i_start = (artifact_index * num_points_per_artifact) % len(in_sample_indices_ds)
        i_end = ((artifact_index + 1) * num_points_per_artifact) % len(in_sample_indices_ds)
        if i_end <= i_start:
            rows_for_artifact = np.concatenate([np.arange(i_start, len(in_sample_indices_ds)), np.arange(0, i_end)])
        else:
            rows_for_artifact = np.arange(i_start, i_end)
    else:
        raise NotImplementedError(f"sample_selection={sample_selection} is not implemented")
    
    assert len(rows_for_artifact) == num_points_per_artifact, (
        f"rows_for_artifact must have length {num_points_per_artifact}, but has length {len(rows_for_artifact)}"
    )
    
    in_sample_indices = in_sample_indices_ds.select(rows_for_artifact, keep_in_memory=True)
    out_sample_indices = out_sample_indices_ds.select(rows_for_artifact, keep_in_memory=True)

    in_base_indices = [
        sample_index_to_base_index[(split, sample_index)]
        for split, sample_index
        in zip(in_sample_indices["split"], in_sample_indices["sample_index"])
    ]
    out_base_indices = [
        sample_index_to_base_index[(split, sample_index)]
        for split, sample_index
        in zip(out_sample_indices["split"], out_sample_indices["sample_index"])
    ]

    in_data_for_artifact_ds = in_out_ds.select(in_base_indices)
    out_data_for_artifact_ds = in_out_ds.select(out_base_indices)
    assert len(in_data_for_artifact_ds) == len(in_sample_indices), (
        "in_data_for_artifact_ds must have the same length as in_base_indices"
    )
    assert len(out_data_for_artifact_ds) == len(out_sample_indices), (
        "out_data_for_artifact_ds must have the same length as out_base_indices"
    )

    logger.info(f"Length of train_base_ds: {len(train_base_ds)}")
    train_data_for_artifact_ds: Dataset = concatenate_datasets([train_base_ds] + [in_data_for_artifact_ds]*num_repetitions)

    validation_data_for_artifact_ds = validation_base_ds
    logger.info(f"Length of validation_base_ds: {len(validation_data_for_artifact_ds)}")

    if (
        merge_unused_samples == MergeUnusedSamplesMethod.ALL_WITH_TRAIN
    ):
        in_sample_indices_not_selected = in_sample_indices_ds.select(
            np.setdiff1d(np.arange(len(in_sample_indices_ds)), rows_for_artifact), keep_in_memory=True
        )
        in_base_indices_not_selected = [
            sample_index_to_base_index[(split, sample_index)]
            for split, sample_index
            in zip(in_sample_indices_not_selected["split"], in_sample_indices_not_selected["sample_index"])
        ]
        in_data_not_selected_ds = in_out_ds.select(in_base_indices_not_selected)

        out_sample_indices_not_selected = out_sample_indices_ds.select(
            np.setdiff1d(np.arange(len(out_sample_indices_ds)), rows_for_artifact), keep_in_memory=True
        )
        out_base_indices_not_selected = [
            sample_index_to_base_index[(split, sample_index)]
            for split, sample_index
            in zip(out_sample_indices_not_selected["split"], out_sample_indices_not_selected["sample_index"])
        ]
        out_data_not_selected_ds = in_out_ds.select(out_base_indices_not_selected)

        train_data_for_artifact_ds = concatenate_datasets([train_data_for_artifact_ds, in_data_not_selected_ds])
        if (
            merge_unused_samples == MergeUnusedSamplesMethod.IN_WITH_TRAIN_OUT_WITH_VAL
        ):
            validation_data_for_artifact_ds = concatenate_datasets([validation_data_for_artifact_ds, out_data_not_selected_ds])
        else:
            train_data_for_artifact_ds = concatenate_datasets([train_data_for_artifact_ds, out_data_not_selected_ds])

    logger.info(f"Length of train_data_for_artifact_ds: {len(train_data_for_artifact_ds)}")
    logger.info(f"Length of validation_data_for_artifact_ds: {len(validation_data_for_artifact_ds)}")

    assert len(train_data_for_artifact_ds) > 0, "train_data_for_artifact_ds must have at least one sample"

    # Check that validation_data_for_artifact_ds doesn't contain any samples from train_data_for_artifact_ds (other than null rows)
    train_data_for_artifact_sample_indices = [
        f"{s}-{i}"
        for s, i
        in zip(train_data_for_artifact_ds["split"], train_data_for_artifact_ds["sample_index"])
        if s is not None and i is not None
    ]
    validation_data_for_artifact_sample_indices = [
        f"{s}-{i}"
        for s, i
        in zip(validation_data_for_artifact_ds["split"], validation_data_for_artifact_ds["sample_index"])
        if s is not None and i is not None
    ]

    duplicates = set(train_data_for_artifact_sample_indices).intersection(validation_data_for_artifact_sample_indices)
    if len(duplicates) > 0:
        raise ValueError(
            f"train_data_for_artifact_ds and validation_data_for_artifact_ds must not contain the same samples. "
            f"Found {len(duplicates)} duplicates: {duplicates}"
        )

    # Check that out_data_for_artifact_ds doesn't contain any samples from train_data_for_artifact_ds (other than null rows)
    out_data_for_artifact_sample_indices = [
        f"{s}-{i}"
        for s, i
        in zip(out_data_for_artifact_ds["split"], out_data_for_artifact_ds["sample_index"])
        if s is not None and i is not None
    ]
    duplicates = set(train_data_for_artifact_sample_indices).intersection(out_data_for_artifact_sample_indices)
    if len(duplicates) > 0:
        raise ValueError(
            f"train_data_for_artifact_ds and out_data_for_artifact_ds must not contain the same samples. "
            f"Found {len(duplicates)} duplicates: {duplicates}"
        )

    # Check that train_data_for_artifact_ds doesn't contain any duplicates (other than null rows)
    train_data_for_artifact_sample_indices_not_null = [i for i in train_data_for_artifact_sample_indices if i != "None-None"]
    if (
        num_repetitions == 1 and  # If num_repetitions > 1, duplicates are expected
        len(set(train_data_for_artifact_sample_indices_not_null)) != len(train_data_for_artifact_sample_indices_not_null)
    ):
        duplicates = [
            i for i, c in Counter(train_data_for_artifact_sample_indices_not_null).items() if c > 1
        ]
        raise ValueError(
            f"train_data_for_artifact_ds must not contain any duplicates. "
            f"Found {len(duplicates)} duplicates: {duplicates}"
        )

    train_data_for_artifact_ds = train_data_for_artifact_ds.shuffle(seed=seed+artifact_index, keep_in_memory=True)

    return PreparedData(
        in_data_for_artifact=in_data_for_artifact_ds,
        out_data_for_artifact=out_data_for_artifact_ds,
        train_data_for_artifact=train_data_for_artifact_ds,
        validation_data_for_artifact=validation_data_for_artifact_ds
    )


@command_component(environment=ENV)
def prepare_data_for_aml_parallel(
    train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
    artifact_index_start: int, artifact_index_end: int, group_base_seed: int, num_points_per_artifact: int,
    in_data_for_artifacts: Output(mode="rw_mount"), out_data_for_artifacts: Output(mode="rw_mount"),  # noqa: F821
    train_data_for_artifacts: Output(mode="rw_mount"), validation_data_for_artifacts: Output(mode="rw_mount"), # noqa: F821
    sample_selection: str, merge_unused_samples: str, seeds_for_artifacts: Output(mode="rw_mount"),  # noqa: F821
    num_repetitions: int = 1
):
    for artifact_index in range(artifact_index_start, artifact_index_end):
        artifact_index_str = f"artifact_index-{artifact_index:04}"
        prepared_data = _prepare_data(
            train_base_ds=load_from_disk(train_base_data),
            validation_base_ds=load_from_disk(validation_base_data),
            in_out_ds=load_from_disk(in_out_data),
            in_sample_indices_ds=load_from_disk(in_indices),
            out_sample_indices_ds=load_from_disk(out_indices),
            artifact_index=artifact_index,
            seed=group_base_seed+artifact_index,
            num_points_per_artifact=num_points_per_artifact,
            sample_selection=sample_selection,
            merge_unused_samples=merge_unused_samples,
            num_repetitions=num_repetitions
        )
        prepared_data.train_data_for_artifact.save_to_disk(str(Path(train_data_for_artifacts)/artifact_index_str))
        prepared_data.validation_data_for_artifact.save_to_disk(str(Path(validation_data_for_artifacts)/artifact_index_str))
        prepared_data.in_data_for_artifact.save_to_disk(str(Path(in_data_for_artifacts)/artifact_index_str))
        prepared_data.out_data_for_artifact.save_to_disk(str(Path(out_data_for_artifacts)/artifact_index_str))

        seed_dir = Path(seeds_for_artifacts)/artifact_index_str
        os.makedirs(seed_dir)
        with open(seed_dir/"data", "w+") as f:
            f.write(str(group_base_seed+artifact_index))


@command_component(environment=ENV)
def prepare_data(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
                 artifact_index: int, seed: int, num_points_per_artifact: int, in_data_for_artifact: Output, out_data_for_artifact: Output,
                 train_data_for_artifact: Output, validation_data_for_artifact: Output, sample_selection: str,
                 merge_unused_samples: str, num_repetitions: int = 1):
    """
    Component that prepare data for training a artifact

    Args:
        train_base_data (Input): Dataset that will be in every artifact's training data
        in_out_data (Input): Dataset that will be used to create the in and out data for each artifact
        in_indices (Input): Indices of the in data for each artifact
        out_indices (Input): Indices of the out data for each artifact
        artifact_index (int): Index of the artifact
        seed (int): Seed for the random number generator
        num_points_per_artifact (int): Number of points to sample for each artifact
        in_data_for_artifact (Output): Path to save the in data for the artifact
        out_data_for_artifact (Output): Path to save the out data for the artifact
        train_data_for_artifact (Output): Path to save the train data for the artifact
        sample_selection (str, optional): Method for selecting samples. Defaults to "with_replacement".
        merge_unused_samples_with_train_data (bool, optional): Whether to merge unused samples with the train data.
                                                               Defaults to False.
        num_repetitions (int, optional): Number of times to repeat the inserted samples in the train data. Defaults to 1.
    """
    prepared_data = _prepare_data(
        train_base_ds=load_from_disk(train_base_data),
        validation_base_ds=load_from_disk(validation_base_data),
        in_out_ds=load_from_disk(in_out_data),
        in_sample_indices_ds=load_from_disk(in_indices),
        out_sample_indices_ds=load_from_disk(out_indices),
        artifact_index=artifact_index,
        seed=seed,
        num_points_per_artifact=num_points_per_artifact,
        sample_selection=sample_selection,
        merge_unused_samples=merge_unused_samples,
        num_repetitions=num_repetitions,
    )
    prepared_data.train_data_for_artifact.save_to_disk(train_data_for_artifact)
    prepared_data.validation_data_for_artifact.save_to_disk(validation_data_for_artifact)
    prepared_data.in_data_for_artifact.save_to_disk(in_data_for_artifact)
    prepared_data.out_data_for_artifact.save_to_disk(out_data_for_artifact)
