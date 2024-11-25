import datasets
import numpy as np
import json
from typing import List, Union, Optional, Callable
from mldesigner import command_component, Input, Output
from contextlib import ExitStack
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent/"environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
}


def concatenate_datasets(paths: List[str], output: str) -> None:
    datasets.concatenate_datasets([datasets.load_from_disk(path) for path in paths]).save_to_disk(output)


def assert_json_equal(paths: List[str], output: str) -> None:
    with ExitStack() as stack: 
        data = [json.load(stack.enter_context(open(path))) for path in paths]

    for l, r in zip(data, data[1:]):
        if json.dumps(l, sort_keys=True) != json.dumps(r, sort_keys=True):
            raise ValueError(f"JSON files l:{paths[0]} and r:{paths[1]} are not equal."
                             f"Left: {json.dumps(l, sort_keys=True)}. "
                             f"Right: {json.dumps(r, sort_keys=True)}.")

    with open(output, "w") as f:
        json.dump(data[0], f)


def aggregate_json(paths: List[str], output: str, method: str = "mean") -> None:
    with ExitStack() as stack: 
        data = [json.load(stack.enter_context(open(path))) for path in paths]

    keys = data[0].keys()
    assert all([d.keys() == keys for d in data[1:]])

    result = {}
    for key in keys:
        if method == "mean":
            result[key] = np.mean([d[key] for d in data], axis=0).tolist()
        elif method == "sum":
            result[key] = np.sum([d[key] for d in data], axis=0).tolist()
        elif method == "max":
            result[key] = np.max([d[key] for d in data], axis=0).tolist()
        elif method == "min":
            result[key] = np.min([d[key] for d in data], axis=0).tolist()
        elif method == "median":
            result[key] = np.median([d[key] for d in data], axis=0).tolist()
        elif method == "std":
            result[key] = np.std([d[key] for d in data], axis=0).tolist()
        else:
            raise ValueError(f"Unknown method {method}.")

    with open(output, "w") as f:
        json.dump(data[0], f)


AGGREGATORS = {
    "concatenate_datasets": concatenate_datasets,
    "assert_json_equal": assert_json_equal,
    "average_json": lambda paths, output: aggregate_json(paths, output, method="mean"),
    "sum_json": lambda paths, output: aggregate_json(paths, output, method="sum"),
    "max_json": lambda paths, output: aggregate_json(paths, output, method="max"),
    "min_json": lambda paths, output: aggregate_json(paths, output, method="min"),
    "median_json": lambda paths, output: aggregate_json(paths, output, method="median"),
    "std_json": lambda paths, output: aggregate_json(paths, output, method="std")
}


AGGREGATOR_OUTPUTS = {
    "concatenate_datasets": "uri_folder",
    "assert_json_equal": "uri_file",
    "average_json": "uri_file",
    "sum_json": "uri_file",
    "max_json": "uri_file",
    "min_json": "uri_file",
    "median_json": "uri_file",
    "std_json": "uri_file"
}


@command_component(display_name="Aggregate 2 output directories", environment=ENV)
def aggregate_2_output_dirs(data0: Input(type="uri_folder"), data1: Input(type="uri_folder"),  # noqa: F821
                            output: Output(type="uri_folder"), aggregator: str):  # noqa: F821
    AGGREGATORS[aggregator]([data0, data1], output=output)


@command_component(display_name="Aggregate 2 output files", environment=ENV)
def aggregate_2_output_files(data0: Input(type="uri_file"), data1: Input(type="uri_file"),  # noqa: F821
                             output: Output(type="uri_file"), aggregator: str):  # noqa: F821
    AGGREGATORS[aggregator]([data0, data1], output=output)


@command_component(display_name="Aggregate 16 output directories", environment=ENV)
def aggregate_16_output_dirs(
    data0: Input(type="uri_folder"), data1: Input(type="uri_folder"), data2: Input(type="uri_folder"),  # noqa: F821
    data3: Input(type="uri_folder"), data4: Input(type="uri_folder"), data5: Input(type="uri_folder"),  # noqa: F821
    data6: Input(type="uri_folder"), data7: Input(type="uri_folder"), data8: Input(type="uri_folder"),  # noqa: F821
    data9: Input(type="uri_folder"), data10: Input(type="uri_folder"), data11: Input(type="uri_folder"),  # noqa: F821
    data12: Input(type="uri_folder"), data13: Input(type="uri_folder"), data14: Input(type="uri_folder"),  # noqa: F821
    data15: Input(type="uri_folder"), output: Output(type="uri_folder"), aggregator: str  # noqa: F821
):
    AGGREGATORS[aggregator]([
        data0, data1, data2, data3, data4, data5, data6, data7,
        data8, data9, data10, data11, data12, data13, data14, data15
        ],
        output=output
    )


@command_component(display_name="Aggregate 16 output files", environment=ENV)
def aggregate_16_output_files(
    data0: Input(type="uri_file"), data1: Input(type="uri_file"), data2: Input(type="uri_file"),  # noqa: F821
    data3: Input(type="uri_file"), data4: Input(type="uri_file"), data5: Input(type="uri_file"),  # noqa: F821
    data6: Input(type="uri_file"), data7: Input(type="uri_file"), data8: Input(type="uri_file"),  # noqa: F821
    data9: Input(type="uri_file"), data10: Input(type="uri_file"), data11: Input(type="uri_file"),  # noqa: F821
    data12: Input(type="uri_file"), data13: Input(type="uri_file"), data14: Input(type="uri_file"), # noqa: F821
    data15: Input(type="uri_file"), output: Output(type="uri_file"), aggregator: str  # noqa: F821
):
    AGGREGATORS[aggregator]([
        data0, data1, data2, data3, data4, data5, data6, data7,
        data8, data9, data10, data11, data12, data13, data14, data15
        ],
        output=output
    )


def aggregate_output(data: List[Union[Input, Output]], aggregator: str, load_component: Optional[Callable] = None) -> Output:
    if load_component is None:
        load_component = lambda comp: comp

    if AGGREGATOR_OUTPUTS[aggregator] == "uri_folder":
        aggregate_2_outputs = load_component(aggregate_2_output_dirs)
        aggregate_16_outputs = load_component(aggregate_16_output_dirs)
    elif AGGREGATOR_OUTPUTS[aggregator] == "uri_file":
        aggregate_2_outputs = load_component(aggregate_2_output_files)
        aggregate_16_outputs = load_component(aggregate_16_output_files)

    # use divide and conquer to concatenate all datasets
    if len(data) == 1:
        return data[0]
    elif len(data) == 2:
        return aggregate_2_outputs(data0=data[0], data1=data[1], aggregator=aggregator).outputs.output
    elif len(data) == 16:
        return aggregate_16_outputs(**{f"data{i}": data[i] for i in range(16)}, aggregator=aggregator).outputs.output
    elif len(data) > 16:
        split = int(len(data) + 1 / 2 // 16 * 16)
        half = len(data) // 2
        remainder = half % 16
        if remainder <= 8:
            split = half - remainder
        else:
            split = half + (16 - remainder)
        return aggregate_output(
            [aggregate_output(data[:split], aggregator=aggregator)] + \
            [aggregate_output(data=data[split:], aggregator=aggregator)],
            aggregator=aggregator
        )
    else:
        split = len(data) // 2
        return aggregate_output(
            [aggregate_output(data[:split], aggregator=aggregator)] + \
            [aggregate_output(data=data[split:], aggregator=aggregator)],
            aggregator=aggregator
        )


@command_component(display_name="Collect AML parallel to file", environment=ENV)
def collect_from_aml_parallel_to_uri_file(data: Input(type="uri_folder"), output: Output(type="uri_file"),  # noqa: F821
                                          aggregator: str):
    assert AGGREGATOR_OUTPUTS[aggregator] == "uri_file"
    files = [p/"data" for p in Path(data).iterdir()]
    AGGREGATORS[aggregator](files, output=output)


@command_component(display_name="Collect AML parallel to folder", environment=ENV)
def collect_from_aml_parallel_to_uri_folder(data: Input(type="uri_folder"), output: Output(type="uri_folder"),  # noqa: F821
                                            aggregator: str):
    assert AGGREGATOR_OUTPUTS[aggregator] == "uri_folder"
    files = [p for p in Path(data).iterdir()]
    AGGREGATORS[aggregator](files, output=output)


def collect_from_aml_parallel(data: Input, aggregator: str) -> Output:
    if AGGREGATOR_OUTPUTS[aggregator] == "uri_folder":
        collect = collect_from_aml_parallel_to_uri_folder
    elif AGGREGATOR_OUTPUTS[aggregator] == "uri_file":
        collect = collect_from_aml_parallel_to_uri_file

    return collect(data=data, aggregator=aggregator).outputs.output

