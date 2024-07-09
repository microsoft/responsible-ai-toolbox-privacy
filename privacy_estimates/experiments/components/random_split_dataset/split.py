from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset
from pathlib import Path


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
})
def random_split_dataset(dataset: Input, dataset_1: Output, dataset_2: Output, split_1_size: int, seed: int):
    ds: Dataset = load_from_disk(dataset, keep_in_memory=True)
    ds_split = ds.train_test_split(train_size=split_1_size, shuffle=True, seed=seed)
    ds_split["train"].save_to_disk(dataset_1)
    ds_split["test"].save_to_disk(dataset_2)
