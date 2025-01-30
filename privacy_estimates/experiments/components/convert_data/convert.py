import os
import mltable

from mldesigner import command_component, Input, Output
from datasets import Dataset, concatenate_datasets
from pathlib import Path
from tempfile import NamedTemporaryFile

ENV = {
    "conda_file": Path(__file__).parent/"environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
}





@command_component(name="privacy_estimates__convert_userprompt_to_hfd",environment=ENV)
def convert_userprompt_to_hfd(data: Input(type="uri_folder"), output: Output(type="uri_folder")):
    """
    Convert user prompt data to Hugging Face Dataset format.
    """
    files = list(Path(data).glob(f"**{os.path.sep}*.json"))
    datasets = []
    for p in files:
        ds = Dataset.from_json(str(p))
        ds = ds.add_column("file_path", [str(p.relative_to(data))] * len(ds))
        datasets.append(ds)

    concatenate_datasets(datasets).save_to_disk(output)


@command_component(name="privacy_estimates__convert_hfd_to_userprompt",environment=ENV)
def convert_hfd_to_userprompt(data: Input(type="uri_folder"), output: Output(type="uri_folder")):
    """
    Convert Hugging Face Dataset format to user prompt data.
    """
    ds = Dataset.load_from_disk(data)

    # Get unique file paths
    file_paths = ds.unique("file_path")

    # Ensure output directory exists
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in file_paths:
        # Manually extract relevant rows instead of using `filter()`
        rows = [row for row in ds if row["file_path"] == p]

        # Convert to dataset
        ds_p = Dataset.from_dict({k: [row[k] for row in rows] for k in rows[0] if k != "file_path"})

        # Save the filtered dataset
        output_file = output_dir / f"{Path(p).stem}.json"
        ds_p.to_json(str(output_file))

