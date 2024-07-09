import pandas as pd

from mldesigner import command_component, Output
from pathlib import Path


@command_component(environment={
    "conda_file": Path(__file__).parent/"environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
})
def create_model_indices_for_aml_parallel(
    model_index_start: int, model_index_end: int,
    model_indices: Output(mode="rw_mount")  # noqa: F821
):
    for model_index in range(model_index_start, model_index_end):
        model_index_str = f"model_index-{model_index:04}"
        pd.DataFrame({
            "model_index": [model_index],
        }).to_csv(str(Path(model_indices)/model_index_str)+".csv", index=False)
