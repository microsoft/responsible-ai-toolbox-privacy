from mldesigner import Input
from pathlib import Path
from typing import Optional

from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader


def compute_privacy_estimates(
    scores: Input, challenge_bits: Input, dp_parameters: Optional[Input] = None, smallest_delta: Optional[float] = None,
    alpha: Optional[float] = None
):
    optional_args = dict()
    if dp_parameters is not None:
        optional_args["dp_parameters"] = dp_parameters
    if smallest_delta is not None:
        optional_args["smallest_delta"] = smallest_delta
    if alpha is not None:
        optional_args["alpha"] = alpha

    return PrivacyEstimatesComponentLoader().load_from_component_spec(
        path=Path(__file__).parent/"component_spec.yaml"
    )(
        scores=scores, challenge_bits=challenge_bits, **optional_args
    )
