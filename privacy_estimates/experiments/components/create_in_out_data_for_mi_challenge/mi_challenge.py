import numpy as np

from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset, features
from collections import OrderedDict
from pathlib import Path


MI_LABEL_TO_CHALLENGE_BIT = OrderedDict([("no_member", 0), ("member", 1)])


def count_null(ds: Dataset) -> int:
    return len(ds.filter(lambda row: (row["split"] is None) and (row["sample_index"] is None)))


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
})
def create_in_out_data_for_membership_inference_challenge(
    train_data: Input, challenge_points: Input, seed: int, in_out_data: Output, in_indices: Output,
    out_indices: Output, challenge_bits: Output, train_base_data: Output, max_num_challenge_points: int = 2048,
):
    r"""
    Implementing the membership inference challenge

    challenge_bit ~ {0,1}
    C_0 <- \emptyset
    C_1 <- {z}
    \theta <- train(S \cup C_b)
    """
    train_ds = load_from_disk(train_data, keep_in_memory=True)
    rng = np.random.default_rng(seed)

    challenge_points_ds = load_from_disk(challenge_points, keep_in_memory=True)
    # assuming most vulnerable points are in the beginning
    challenge_points_ds = challenge_points_ds.select(range(max_num_challenge_points), keep_in_memory=True)
    challenge_points_ds = challenge_points_ds.shuffle(seed=seed, keep_in_memory=True)
    challenge_points_ds.save_to_disk(in_out_data)

    challenge_point_indices = OrderedDict(
        zip(
            zip(challenge_points_ds["split"], challenge_points_ds["sample_index"]),
            rng.integers(0, 2, max_num_challenge_points)
        )
    )
    assert len(challenge_point_indices) == max_num_challenge_points
    assert len(challenge_points_ds) == max_num_challenge_points

    num_members = sum(challenge_point_indices.values())

    print(f"Number of members: {num_members}")
    print(f"Number of non-members: {max_num_challenge_points - num_members}")
    print(f"Number of challenge points: {max_num_challenge_points}")
    print(f"Mean challenge bit: {num_members / max_num_challenge_points}")

    # save challenge_points to in_out_data
    challenge_points_ds.save_to_disk(in_out_data)

    # remove challenge points from train data
    train_base_ds = train_ds.filter(
        lambda row: (row["split"], row["sample_index"]) not in challenge_point_indices, keep_in_memory=True
    )
    train_base_ds.save_to_disk(train_base_data)

    challenge_bits_ds = Dataset.from_dict(
        {"challenge_bit": list(challenge_point_indices.values())},
        features=features.Features(
            {"challenge_bit": features.ClassLabel(names=list(MI_LABEL_TO_CHALLENGE_BIT.keys()))}
        ),
    )
    assert len(challenge_bits_ds) == max_num_challenge_points
    challenge_bits_ds.save_to_disk(challenge_bits)

    member_indices_ds = Dataset.from_dict({
        "sample_index": [
            index if challenge_bit == MI_LABEL_TO_CHALLENGE_BIT["member"] else None
            for (_, index), challenge_bit
            in challenge_point_indices.items()
        ],
        "split": [
            split if challenge_bit == MI_LABEL_TO_CHALLENGE_BIT["member"] else None
            for (split, _), challenge_bit
            in challenge_point_indices.items()
        ]
    }, features=features.Features({
        "sample_index": challenge_points_ds.features["sample_index"], "split": challenge_points_ds.features["split"]
    }))
    assert len(member_indices_ds) == max_num_challenge_points
    assert count_null(member_indices_ds) == max_num_challenge_points - num_members
    member_indices_ds.save_to_disk(in_indices)

    non_member_indices_ds = Dataset.from_dict({
        "sample_index": [
            index if challenge_bit == MI_LABEL_TO_CHALLENGE_BIT["no_member"] else None
            for (_, index), challenge_bit
            in challenge_point_indices.items()
        ],
        "split": [
            split if challenge_bit == MI_LABEL_TO_CHALLENGE_BIT["no_member"] else None
            for (split, _), challenge_bit
            in challenge_point_indices.items()
        ]
    },
        features=features.Features({
            "sample_index": challenge_points_ds.features["sample_index"], "split": challenge_points_ds.features["split"]
        })
    )
    assert len(non_member_indices_ds) == max_num_challenge_points
    assert count_null(non_member_indices_ds) == num_members
    non_member_indices_ds.save_to_disk(out_indices)
