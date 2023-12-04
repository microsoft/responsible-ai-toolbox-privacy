#!/usr/bin/env python3

from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from pydantic_cli import run_and_exit
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Optional
from tempfile import TemporaryDirectory

from privacy_estimates.experiments.aml import WorkspaceConfig, Job


class Arguments(BaseModel):
    workspace_config: Optional[Path] = Field(default=None, cli=["--workspace-config"])
    run_id: Optional[str] = Field(defaulte=None, cli=["--run-id"])
    url: Optional[str] = Field(default=None, cli=["--url"])


@contextmanager
def download_inputs_to_temp_dir(job: Job) -> Dict[str, Path]:
    with ExitStack() as stack:
        paths = OrderedDict()
        for i in job.inputs:
            temp_dir = stack.enter_context(TemporaryDirectory())
            local_path = job.download_input(name=i, path=temp_dir)
            paths[i] = Path(local_path)
        yield paths


def main(args: Arguments):
    workspace_and_run_id = bool(args.workspace_config) and bool(args.run_id)
    if workspace_and_run_id == bool(args.url):
        raise ValueError(
            "Must specify either --workspace-config and --run-id or --url"
        )
    if args.url:
        job = Job.from_url(args.url)
    if workspace_and_run_id:
        ws = WorkspaceConfig.from_yaml(args.workspace_config)
        job = Job.from_id(ws, run_id=args.run_id)

    with download_inputs_to_temp_dir(job) as input_paths:
        with TemporaryDirectory() as output_dir:
            cmd = job.get_command(input_paths=input_paths, output_path=output_dir)
            print("Output directory:", output_dir)
            print("Run command:")
            print(cmd)
            while input("Press 'q' to clean up... ") != "q":
                pass
            print("Cleaning up...")
    print("Done")


if __name__ == "__main__":
    run_and_exit(Arguments, main)
