import os
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pathlib import Path
from typing import Optional
from azureml.core import Run, Workspace, Dataset


class Arguments(BaseModel):
    tabular_dataset_id: str = Field(
        description="Direct input of AML tabular dataset"
    )
    output_path: Path = Field(
        description="Output path for the AML dataset in parquet format"
    )
    workspace_config: Optional[Path] = Field(default=None, description="Path to workspace configuration file")


def main(args: Arguments):
    print(f"Arguments: {args}")
    if args.workspace_config is not None:
        ws = Workspace.from_config(args.workspace_config)
    else:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace

    dataset = Dataset.get_by_id(workspace=ws, id=args.tabular_dataset_id)
    df = dataset.to_pandas_dataframe()
    print(f"Writing dataset to {args.output_path}")
    df.to_parquet(args.output_path)
    print("done")
    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
