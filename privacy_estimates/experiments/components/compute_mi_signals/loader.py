from azure.ai.ml import load_component
from mldesigner import Input
from pathlib import Path
from typing import Optional


def compute_mi_signal(
    scores: Input, challenge_bits: Input, dp_parameters: Optional[Input] = None, smallest_delta: Optional[float] = None,
    alpha: Optional[float] = None
):
    return load_component(
        source=str(Path(__file__).parent/"component_spec.yaml")
    )(
        logits_and_labels=logits_and_labels, method=method, extra_args=extra_args,
    )
