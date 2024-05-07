from azure.ai.ml import load_component
from mldesigner import Input
from pathlib import Path


def compute_mi_signals(
    predictions_and_labels: Input, method: str, prediction_column: str, prediction_format: str, label_column: str = "label", **kwargs
):
    return load_component(
        source=str(Path(__file__).parent/"component_spec.yaml")
    )(
        predictions_and_labels=predictions_and_labels, method=method, prediction_column=prediction_column,
        prediction_format=prediction_format, label_column=label_column,
        extra_args=" ".join(f"{k}={v}" for k, v in kwargs.items())
    )
