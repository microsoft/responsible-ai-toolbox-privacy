import numpy as np

from mldesigner import command_component, Output
from datasets import Dataset, features
from pathlib import Path


@command_component(environment="environment.aml.yaml")
def create_challenge_bits_aml_parallel(
    num_challenge_points: int,
    seed: int,
    challenge_bits: Output,
    model_indices: Output,
):
    rng = np.random.default_rng(seed)

    challenge_bits_ds = Dataset.from_dict(
        {"challenge_bits": rng.integers(0, 2, num_challenge_points)},
        features=features.Features({"challenge_bits": features.ClassLabel(num_classes=2)})
    )
    challenge_bits_ds.save_to_disk(challenge_bits)

    for i in range(num_challenge_points):
        model_indices_ds = Dataset.from_dict(
            {"model_index": [i]},
            features=features.Features({"model_index": features.Value(dtype="int32")})
        )
        model_indices_ds.to_csv(str(Path(model_indices) / f"model_index-{i:04d}.csv"))
