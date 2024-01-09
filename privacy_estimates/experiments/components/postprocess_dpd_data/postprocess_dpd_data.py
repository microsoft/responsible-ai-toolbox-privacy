import logging
import numpy as np
from datasets import Dataset, features
from mldesigner import command_component, Input, Output
from pathlib import Path
from json import load, dump


logger = logging.getLogger(__name__)


@command_component(
    display_name="Postprocess DP Distinguisher data",
    environment="environment.aml.yaml",
)
def postprocess_dpd_data(dpd_data: Input, dp_parameters: Input, seed: int, scores: Output, challenge_bits: Output,
                         postprocessed_dp_parameters: Output(type="uri_file")):  # noqa: F821
    """
    Post-processing for differential privacy distinguisher

    We follow Algorithm 2 in https://arxiv.org/abs/2302.07956. We rearrange
    <g'|\nabla'[t] + g'> to <g'|\nabla'[t]> + <g'|g'> which allows us to pull
    <g'|g'> = C^2 (which is the gradient clipping norm) out of the training algorithm.
    Instead we add the C^2 term in this script.
    """
    with Path(dpd_data).open() as f:
        raw_data = load(f)
    sensitivity = raw_data["sensitivity"]

    with Path(dp_parameters).open() as f:
        dp_parameters = load(f)

    # Differential Privacy Distinguisher 
    dp_parameters["num_steps"] = 1
    dp_parameters["subsampling_probability"] = 1.0


    observations = Dataset.from_dict(mapping={"score": raw_data["scores"]},
                                     features=features.Features({"score": features.Value("float32")}))

    if len(observations) < 500:
        logger.error(f"DPD data has only {len(observations)} observations. "
                     "Consider training for more steps to get a better signal")

    observations = observations.shuffle(seed=seed)
    observations = observations.add_column("challenge_bit", np.random.default_rng(seed).integers(0, 2, size=len(observations)))

    observations = observations.map(
        lambda row: {"score": row["score"] + sensitivity**2 if row["challenge_bit"] == 1 else row["score"]},
        keep_in_memory=True
    )

    observations.select_columns(["score"]).save_to_disk(scores)
    observations.select_columns(["challenge_bit"]).save_to_disk(challenge_bits)

    with Path(postprocessed_dp_parameters).open("w") as f:
        dump(dp_parameters, f)
