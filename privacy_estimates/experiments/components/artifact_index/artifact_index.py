from mldesigner import command_component, Input, Output
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
}


@command_component(name="privacy_estimates__get_global_artifact_index", environment=ENV)
def get_global_artifact_index(group_index: int, group_size: int, index: int) -> Output(type="integer"):
    return group_index * group_size + index
