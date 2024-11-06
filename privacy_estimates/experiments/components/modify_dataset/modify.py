from mldesigner import command_component, Input, Output
from datasets import load_from_disk
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent/"environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
}


@command_component(display_name="Append column to dataset", environment=ENV)
def append_column_constant_int(data: Input, name: str, value: int, output: Output):
    """
    Appends a new column with a constant integer value to the dataset.

    Args:
        data (Input): The input dataset.
        name (str): The name of the new column.
        value (int): The constant integer value to be added to the column.

    Returns:
        output (Output): The output dataset.
    """
    ds = load_from_disk(data)
    ds = ds.add_column(name, (value for _ in range(len(ds))))
    ds.save_to_disk(output)


@command_component(display_name="Append column to dataset", environment=ENV)
def append_column_constant_str(data: Input, name: str, value: str, output: Output):
    """
    Appends a new column with a constant string value to the dataset.

    Args:
        data (Input): The input dataset.
        name (str): The name of the new column.
        value (str): The constant string value to be added to the new column.

    Returns:
        output (Output): The output dataset.
    """
    ds = load_from_disk(data)
    ds = ds.add_column(name, (value for _ in range(len(ds))))
    ds.save_to_disk(output)


@command_component(display_name="Append column to dataset with incrementing index", environment=ENV)
def append_column_incrementing(data: Input, name: str, output: Output):
    """
    Appends a new column to the dataset with incrementing values.

    Args:
        data (Input): The input dataset.
        name (str): The name of the new column.

    Returns:
        output (Output): The output dataset.
    """
    ds = load_from_disk(data)
    ds = ds.add_column(name, (i for i in range(len(ds))))
    ds.save_to_disk(output)


@command_component(display_name="Append artifact index column to dataset", environment=ENV)
def append_artifact_index_column_aml_parallel(data: Input, output: Output):
    """
    Appends a 'artifact_index' column to each dataset in the input directory and saves the modified datasets to the output directory.

    Args:
        data (str): The path to the input directory containing the datasets.

    Raises:
        ValueError: If a artifact directory in the input directory does not start with 'artifact_index'.

    Returns:
        output (str): The path to the output directory where the modified datasets will be saved.
    """
    for artifact_dir in Path(data).iterdir():
        if not artifact_dir.name.startswith("artifact_index"):
            raise ValueError(f"Expected artifact directory to start with 'artifact_index', but got {artifact_dir.name}")
        artifact_index = int(artifact_dir.name.split("-")[1])
        ds = load_from_disk(artifact_dir)
        ds = ds.add_column("artifact_index", [artifact_index for _ in range(len(ds))])
        ds.save_to_disk(str(Path(output)/artifact_dir.name))


@command_component(display_name="Select columns from dataset", environment=ENV)
def select_columns(data: Input, columns: str, output: Output):
    """
    Selects specific columns from the input dataset and saves the modified dataset to the output location.

    Args:
        data (Input): The input dataset to modify.
        columns (str): A string containing the names of the columns to select, separated by spaces.

    Returns:
        output (Output): The location to save the modified dataset.
    """
    ds = load_from_disk(data)
    ds = ds.select_columns(columns.split())
    ds.save_to_disk(output)


@command_component(display_name="Rename columns in dataset", environment=ENV)
def rename_columns(data: Input, columns_old: str, columns_new: str, output: Output):
    """
    Renames columns in a dataset.

    Args:
        data (Input): The input dataset.
        columns_old (str): A string containing the old column names separated by spaces.
        columns_new (str): A string containing the new column names separated by spaces.

    Returns:
        output (Output): The output dataset.
    """
    column_mapping = dict(zip(columns_old.split(), columns_new.split()))
    ds = load_from_disk(data)
    ds = ds.rename_columns(column_mapping)
    ds.save_to_disk(output)


@command_component(display_name="Select top k rows", environment=ENV)
def select_top_k_rows(data: Input, k: int, output: Output, allow_fewer: Input(type="boolean", optional=True) = False):
    """
    Selects the top k rows from the input dataset and saves the result to the output dataset.

    Args:
        data (Input): The input dataset.
        k (int): The number of rows to select.
        allow_fewer (Input(type="bool", optional=True), optional): 
            If True, allows selecting fewer rows than k if the input dataset has fewer rows. 
            Defaults to False.
    Returns:
        output (Output): The output dataset where the selected rows will be saved.
    """
    ds = load_from_disk(data)
    if allow_fewer:
        k = min(k, len(ds))
    ds = ds.select(range(k))
    ds.save_to_disk(output)


@command_component(display_name="Move dataset", environment=ENV)
def move_dataset(data: Input, output: Output):
    """
    Move a dataset from the input location to the output location.

    Args:
        data (Input): The input dataset location.
    Returns:
        output (Output): The output dataset location.
    """
    ds = load_from_disk(data)
    ds.save_to_disk(output)
