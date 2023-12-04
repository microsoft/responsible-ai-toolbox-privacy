from azure.ai.ml import load_component
from mldesigner import Input
from pathlib import Path
from typing import Optional


def compute_privacy_estimates(
    scores: Input, challenge_bits: Input, dp_parameters: Optional[Input] = None, target_delta: Optional[float] = None,
    alpha: float = 0.05
):
    optional_args = dict()
    if dp_parameters is not None:
        optional_args["dp_parameters"] = dp_parameters
    if target_delta is not None:
        optional_args["target_delta"] = target_delta

    return load_component(
        source=str(Path(__file__).parent/"component_spec.yaml")
    )(
        scores=scores, challenge_bits=challenge_bits, alpha=alpha, **optional_args
    )
