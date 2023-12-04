import pandas as pd

from mldesigner import command_component, Output
from pathlib import Path


@command_component(environment="environment.aml.yaml")
def create_model_indices_for_aml_parallel(
    model_index_start: int, model_index_end: int,
    model_indices: Output(mode="rw_mount")
):
    for model_index in range(model_index_start, model_index_end):
        model_index_str = f"model_index-{model_index:04}"
        pd.DataFrame({
            "model_index": [model_index],
        }).to_csv(str(Path(model_indices)/model_index_str)+".csv", index=False)
