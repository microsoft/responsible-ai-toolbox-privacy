import json

from mldesigner import command_component, Input, Output
from pathlib import Path


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
})
def compute_seed(base_seed: int, artifact_index: Input, artifact_seed: Output(type="uri_file")):
    with open(artifact_index, "r") as f_index:
        artifact_index = json.load(f_index)
        with open(artifact_seed, "w") as f_seed:
            json.dump(base_seed + artifact_index, f_seed)
