import pandas as pd
from pydantic_cli import run_and_exit
from pydantic import BaseModel, Field
from datasets import load_dataset
from azureml.core import Workspace, Dataset
from pathlib import Path
from typing import Dict, Callable


def sst_2_train() -> pd.DataFrame:
    ds = load_dataset("glue", "sst2", split="train")
    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "label": "label"})
    df = df[["text", "label"]]
    return df

def sst_2_test() -> pd.DataFrame:
    ds = load_dataset("glue", "sst2", split="test")
    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "label": "label"})
    df = df[["text", "label"]]
    return df


def sst_2_validation() -> pd.DataFrame:
    ds = load_dataset("glue", "sst2", split="validation")
    df = ds.to_pandas()
    df = df.rename(columns={"sentence": "text", "label": "label"})
    df = df[["text", "label"]]
    return df

def amazon_polarity_train() -> pd.DataFrame:
    ds = load_dataset("amazon_polarity", split="train")
    df = ds.to_pandas()
    df = df.rename(columns={"content": "text", "label": "label"})
    df = df[["text", "label"]]
    return df


DATASETS: Dict[str, Callable[[], pd.DataFrame]] = {
    "SST2-train": sst_2_train,
    "SST2-test": sst_2_test,
    "SST2-validation": sst_2_validation,
    "AmazonPolarity-train": amazon_polarity_train,
}


class Arguments(BaseModel):
    dataset: str = Field(description=f"Dataset to add to the workspace. Available datasets: {list(DATASETS.keys())}")
    workspace_config: Path = None
    datastore: str = None


def main(args: Arguments):
    ws: Workspace = Workspace.from_config(args.workspace_config)

    if args.dataset not in DATASETS.keys():
        raise ValueError(f"Unknown dataset: {args.dataset}")

    df = DATASETS[args.dataset]()

    if args.datastore:
        datastore = ws.datastores[args.datastore]
    else:
        datastore = ws.get_default_datastore()

    dataset = Dataset.Tabular.register_pandas_dataframe(df, datastore, args.dataset, show_progress=True)
    

if __name__ == "__main__":
    run_and_exit(Arguments, main)
