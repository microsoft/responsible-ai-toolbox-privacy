from mldesigner import command_component, Input, Output
from datasets import load_from_disk
from pathlib import Path


@command_component(display_name="Append column to dataset", environment="environment.aml.yaml")
def append_column_constant_int(data: Input, name: str, value: int, output: Output):
    ds = load_from_disk(data)
    ds = ds.add_column(name, (value for _ in range(len(ds))))
    ds.save_to_disk(output)


@command_component(display_name="Append column to dataset", environment="environment.aml.yaml")
def append_column_constant_str(data: Input, name: str, value: str, output: Output):
    ds = load_from_disk(data)
    ds = ds.add_column(name, (value for _ in range(len(ds))))
    ds.save_to_disk(output) 


@command_component(display_name="Append column to dataset with incrementing index", environment="environment.aml.yaml")
def append_column_incrementing(data: Input, name: str, output: Output):
    ds = load_from_disk(data)
    ds = ds.add_column(name, (i for i in range(len(ds))))
    ds.save_to_disk(output)


@command_component(display_name="Append model index column to dataset", environment="environment.aml.yaml")
def append_model_index_column_aml_parallel(data: Input, output: Output):
    for model_dir in Path(data).iterdir():
        if not model_dir.name.startswith("model_index"):
            raise ValueError(f"Expected model directory to start with 'model_index', but got {model_dir.name}")
        model_index = int(model_dir.name.split("-")[1])
        ds = load_from_disk(model_dir)
        ds = ds.add_column("model_index", [model_index for _ in range(len(ds))])
        ds.save_to_disk(str(Path(output)/model_dir.name))


@command_component(display_name="Select columns from dataset", environment="environment.aml.yaml")
def select_columns(data: Input, columns: str, output: Output):
    ds = load_from_disk(data)
    ds = ds.select_columns(columns.split())
    ds.save_to_disk(output)


@command_component(display_name="Rename columns in dataset", environment="environment.aml.yaml")
def rename_columns(data: Input, columns_old: str, columns_new: str, output: Output):
    column_mapping = dict(zip(columns_old.split(), columns_new.split()))
    ds = load_from_disk(data)
    ds = ds.rename_columns(column_mapping)
    ds.save_to_disk(output)
