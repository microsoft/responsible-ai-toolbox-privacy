import datasets
from mldesigner import command_component, Input, Output
from typing import Any, Dict
from multiprocessing import cpu_count
from dataclasses import dataclass
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
}


def is_row_null(row: Dict[str, Any]) -> bool:
    return (row["sample_index"] is None) and (row["split"] is None)


@dataclass
class FilteredOutput:
    filtered: datasets.Dataset
    aux: datasets.Dataset


def _filter_aux_data(data: datasets.Dataset) -> FilteredOutput:
    num_proc = cpu_count()
    num_proc = None
    assert "sample_index" in data.column_names
    assert "split" in data.column_names

    print(f"Lenght of data: {len(data)}")

    data_columns = set(data.column_names) - {"sample_index", "split"}

    data = data.add_column("is_null", [is_row_null(row) for row in data])

    aux_ds = data.remove_columns(data_columns)

    data = data.filter(lambda row: not row["is_null"], num_proc=num_proc, keep_in_memory=True)
    data = data.select_columns(data_columns)

    print(f"Length of filtered data: {len(data)}")
    print(f"Length of aux data: {len(aux_ds)}")

    return FilteredOutput(filtered=data, aux=aux_ds)


@command_component(name="privacy_estimates__filter_aux_data", environment=ENV)
def filter_aux_data(full: Input, aux: Output, filtered: Output):
    """
    Filters out aux data from the input dataset and saves the resulting dataset to disk.

    Args:
        full (Input): Path to the input dataset.
        aux (Output): Path to save the mask dataset that indicates which rows are null.
        filtered (Output): Path to save the filtered dataset without null rows.
    """
    data = datasets.load_from_disk(full, keep_in_memory=True)

    filtered_output = _filter_aux_data(data)

    filtered_output.aux.save_to_disk(aux)
    filtered_output.filtered.save_to_disk(filtered)


@command_component(name="privacy_estimates__filter_aux_data_aml_parallel", environment=ENV)
def filter_aux_data_aml_parallel(full: Input, aux: Output(mode="rw_mount"), filtered: Output(mode="rw_mount")):  # noqa: F821
    for path in Path(full).iterdir():
        model_index_str = path.name
        assert model_index_str.startswith("model_index")
        filtered_data = _filter_aux_data(datasets.load_from_disk(path, keep_in_memory=True))
        filtered_data.filtered.save_to_disk(str(Path(filtered) / model_index_str))
        filtered_data.aux.save_to_disk(str(Path(aux) / model_index_str))


def _reinsert_aux_data(filtered_data: datasets.Dataset, aux_data: datasets.Dataset) -> datasets.Dataset:
    assert len(filtered_data) + sum(aux_data["is_null"]) == len(aux_data)

    print(f"Length of filtered data: {len(filtered_data)}")
    print(f"Length of aux data: {len(aux_data)}")

    null_data = datasets.Dataset.from_dict(
        mapping={key: [None] for key in filtered_data.features.keys()},
        features=filtered_data.features
    )

    null_index = len(filtered_data)
    full_data: datasets.Dataset = datasets.concatenate_datasets([filtered_data, null_data])

    selected_index_mapping = []
    filtered_index = 0
    for is_null in aux_data["is_null"]:
        if is_null:
            selected_index_mapping.append(null_index)
        else:
            selected_index_mapping.append(filtered_index)
            filtered_index += 1
    assert len(filtered_data) == filtered_index

    full_data = full_data.select(selected_index_mapping, keep_in_memory=True)
    assert len(full_data) == len(aux_data)

    full_data = full_data.add_column("sample_index", aux_data["sample_index"])
    full_data = full_data.add_column("split", aux_data["split"])

    print(f"Length of full data: {len(full_data)}")

    return full_data


@command_component(name="privacy_estimates__reinsert_aux_data", environment=ENV)
def reinsert_aux_data(filtered: Input, aux: Input, full: Output):
    """
    Reinserts aux data into a filtered dataset based on a mask.

    Args:
        filtered (Input): Path to the filtered dataset.
        aux (Input): Path to the mask dataset.
        full (Output): Path to save the resulting dataset.

    Returns:
        None
    """
    filtered_data = datasets.load_from_disk(filtered, keep_in_memory=True)
    aux_data = datasets.load_from_disk(aux, keep_in_memory=True)

    full_data = _reinsert_aux_data(filtered_data, aux_data=aux_data)

    full_data.save_to_disk(full)


@command_component(name="privacy_estimates__reinsert_aux_data_aml_parallel", environment=ENV)
def reinsert_aux_data_aml_parallel(
    filtered: Input(mode="rw_mount"), aux: Input(mode="rw_mount"), full: Output(mode="rw_mount")  # noqa: F821
):
    for fil_path, aux_path in zip(Path(filtered).iterdir(), Path(aux).iterdir()):
        assert fil_path.name == aux_path.name
        model_index_str = fil_path.name
        filtered_data = datasets.load_from_disk(fil_path, keep_in_memory=True)
        aux_data = datasets.load_from_disk(aux_path, keep_in_memory=True)

        full_data = _reinsert_aux_data(filtered_data, aux_data=aux_data)

        full_data.save_to_disk(str(Path(full) / model_index_str))
