#!/usr/bin/env python3

from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from pathlib import Path
from typing import Dict, Optional
from tempfile import TemporaryDirectory

from privacy_estimates.experiments.aml import WorkspaceConfig, Job


@dataclass
class Arguments:
    workspace_config: Optional[Path] = field(default=None, metadata={
        "args": ["--workspace-config"], "help": "Path to the workspace configuration JSON file."
    })
    run_id: Optional[str] = field(default=None, metadata={
        "args": ["--run-id"], "help": "Run ID of the job to debug."
    })
    url: Optional[str] = field(default=None, metadata={
        "args": ["--url"], "help": "URL of the job to debug. (Often requires quotes around the URL.)"
    })


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
    if bool(args.run_id) == bool(args.url):
        raise ValueError(
            "Must specify either --run-id or --url"
        )
    if args.url:
        job = Job.from_url(args.url)
    if args.run_id:
        if args.workspace_config:
            ws = WorkspaceConfig.from_yaml(args.workspace_config)
        else:
            ws = WorkspaceConfig.from_az_cli()
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
    parser = ArgumentParser(Arguments)
    args = parser.parse_args()
    main(args=args)
