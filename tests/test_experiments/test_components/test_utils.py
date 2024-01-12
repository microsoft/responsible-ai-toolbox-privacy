import sys
import os
from azure.ai.ml.entities import Command
from pathlib import Path
from typing import Dict, Callable, Generator, Any, TypeVar
from contextlib import contextmanager, ExitStack
from tempfile import TemporaryDirectory
from datasets import Dataset
from subprocess import check_call
from shlex import split


@contextmanager
def datasets_as_paths(datasets: Dict[str, Dataset]) -> Generator[Dict[str, Path], None, None]:
    with ExitStack() as stack:
        p = {}
        for name, dataset in datasets.items():
            p[name] = stack.enter_context(TemporaryDirectory())
            dataset.save_to_disk(p[name])
        yield p


def path_as_dataset(path: Path) -> Dataset:
    return Dataset.load_from_disk(str(path))


def no_postprocessing(path: Path) -> Path:
    return path


T = TypeVar("T")


def run_component(component: Command, post_process_output: Callable[[Path], T] = path_as_dataset) -> Dict[str, T]:
    code_dir = component.code
    command = component.command
    with ExitStack() as stack:
        outputs = {name: Path(stack.enter_context(TemporaryDirectory())) for name in component.outputs}

        for name, node in component.inputs.items():
            if node.type in {"uri_folder", "uri_file"}:
                command = command.replace(f"${{{{inputs.{name}}}}}", node.path)
            elif node.type in {"integer", "number", "string"}:
                command = command.replace(f"${{{{inputs.{name}}}}}", str(node._data))
            elif node.type in {"boolean"}:
                command = command.replace(f"${{{{inputs.{name}}}}}", str(bool(node._data)))
            else:
                raise ValueError(f"Unexpected input type {node.type}")
        for name, path in outputs.items():
            command = command.replace(f"${{{{outputs.{name}}}}}", str(path))
        split_command = split(command)

        source_index = split_command.index("--source") + 1
        split_command[source_index] = os.path.join(code_dir, split_command[source_index])

        check_call(split_command)

        post_processed_output = {n: post_process_output(p) for n, p in outputs.items()}

    return post_processed_output
