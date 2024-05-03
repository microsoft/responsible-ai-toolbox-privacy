from azure.ai.ml import load_component
from mldesigner import Input
from pathlib import Path


def compute_mi_signals(
    logits_and_labels: Input, method: str, **kwargs
):
    return load_component(
        source=str(Path(__file__).parent/"component_spec.yaml")
    )(
        logits_and_labels=logits_and_labels, method=method, extra_args=" ".join(f"{k}={v}" for k, v in kwargs.items())
    )
