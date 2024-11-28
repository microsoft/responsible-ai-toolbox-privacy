import pandas as pd

from mldesigner import command_component, Output
from pathlib import Path


@command_component(
    name="privacy_estimates__create_artifact_indices_for_aml_parallel",
    environment={
        "conda_file": Path(__file__).parent/"environment.conda.yaml",
        "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    }
)
def create_artifact_indices_for_aml_parallel(
    artifact_index_start: int, artifact_index_end: int,
    artifact_indices: Output(mode="rw_mount")  # noqa: F821
):
    for artifact_index in range(artifact_index_start, artifact_index_end):
        artifact_index_str = f"artifact_index-{artifact_index:04}"
        pd.DataFrame({
            "artifact_index": [artifact_index],
        }).to_csv(str(Path(artifact_indices)/artifact_index_str)+".csv", index=False)
