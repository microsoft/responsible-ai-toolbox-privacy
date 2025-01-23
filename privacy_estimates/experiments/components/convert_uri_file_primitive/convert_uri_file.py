import json

from mldesigner import command_component, Input, Output
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
}


@command_component(name="privacy_estimates__convert_uri_file_to_int", environment=ENV)
def convert_uri_file_to_int(uri_file: Input(type="uri_file")) -> int:
    with open(uri_file, 'r') as file:
        return json.load(file)
    

@command_component(name="privacy_estimates__convert_int_to_uri_file", environment=ENV)
def convert_int_to_uri_file(value: int, output: Output(type="uri_file")):
    with open(output, 'w') as file:
        json.dump(value, file)
