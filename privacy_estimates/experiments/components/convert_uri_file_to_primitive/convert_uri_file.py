from mldesigner import command_component, Input
from pathlib import Path


@command_component(
    name="privacy_estimates__convert_uri_file_to_int",
    environment={
        "conda_file": Path(__file__).parent / "environment.conda.yaml",
        "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    }
)
def convert_uri_file_to_int(uri_file: Input(type="uri_file")) -> int:
    with open(uri_file, 'r') as file:
        return int(file.read())
