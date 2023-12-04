import numpy as np

from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset, features
from enum import Enum, auto


class AdjacencyRelation(Enum):
    ADD_REMOVE = "add_remove"
    REPLACE = "replace"


def swap_in_out_columns_if_no_challenge(row):
    return {
        "sample_index_in": row["sample_index_in"] if row["challenge_bits"] == 1 else row["sample_index_out"],
        "sample_index_out": row["sample_index_out"] if row["challenge_bits"] == 1 else row["sample_index_in"],
        "split_in": row["split_in"] if row["challenge_bits"] == 1 else row["split_out"],
        "split_out": row["split_out"] if row["challenge_bits"] == 1 else row["split_in"],
    }


@command_component(environment="environment.aml.yaml")
def create_in_out_data_for_membership_inference_challenge(
    train_data: Input, challenge_points: Input, seed: int, adjacency_relation: str, in_out_data: Output, in_indices: Output,
    out_indices: Output, challenge_bits: Output, train_base_data: Output
):
    """
    If challenge bit == 1 add the sample from the dataset to the in dataset
    """
    adjacency_relation = AdjacencyRelation(adjacency_relation)

    if adjacency_relation != AdjacencyRelation.ADD_REMOVE:
        raise NotImplementedError("Only ADD_REMOVE adjacency relation is supported at this time.")

    ds = load_from_disk(dataset, keep_in_memory=True)
    rng = np.random.default_rng(seed)

    challenge_bits_ds = Dataset.from_dict(
        {"challenge_bits": rng.integers(0, 2, num_challenge_points)},
        features=features.Features({"challenge_bits": features.ClassLabel(num_classes=2)})
    )
    challenge_bits_ds.save_to_disk(challenge_bits)

    ds.save_to_disk(in_out_data)

    ds = ds.select_columns(
        column_names=["sample_index", "split"]
    ).shuffle(
        seed=seed, keep_in_memory=True
    ).select(
        range(num_challenge_points), keep_in_memory=True
    )

    ds = ds.rename_column("sample_index", "sample_index_in")
    ds = ds.rename_column("split", "split_in")

    ds = ds.add_column("sample_index_out", [None] * num_challenge_points)
    ds = ds.add_column("split_out", [None] * num_challenge_points)

    ds = ds.add_column("challenge_bits", challenge_bits_ds["challenge_bits"])

    ds = ds.map(swap_in_out_columns_if_no_challenge, keep_in_memory=True)

    if sum(np.array(ds["sample_index_in"], dtype=bool)) != sum(np.array(ds["challenge_bits"], dtype=bool)):
        raise ValueError(
            f"Number of challenge points {sum(np.array(ds['challenge_bits'], dtype=bool))} set to 1 does not match number of "
            f"non-null points {sum(np.array(ds['sample_index_in'], dtype=bool))}."
        )
    if sum(np.array(ds["sample_index_out"], dtype=bool)) != num_challenge_points - sum(np.array(ds["challenge_bits"], dtype=bool)):
        raise ValueError(
            f"Number of challenge points {num_challenge_points - sum(np.array(ds['challenge_bits'], dtype=bool))} set to 1 does not match number of "
            f"null points {sum(np.array(ds['sample_index_out'], dtype=bool))}."
        )

    ds.select_columns(
        ["sample_index_in", "split_in"]
    ).rename_column(
        "sample_index_in", "sample_index",
    ).rename_column(
        "split_in", "split",
    ).save_to_disk(in_indices)

    ds.select_columns(
        ["sample_index_out", "split_out"]
    ).rename_column(
        "sample_index_out", "sample_index"
    ).rename_column(
        "split_out", "split"
    ).save_to_disk(out_indices)
