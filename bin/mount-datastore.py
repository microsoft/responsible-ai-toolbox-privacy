#!/usr/bin/env python3

from pydantic_cli import run_and_exit
from pydantic import BaseModel, Field
from pathlib import Path
from json import load
from azureml.core import Workspace
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from subprocess import check_call
from tempfile import mkdtemp


class Arguments(BaseModel):
    config_path: Path
    mount_path: Path


def mount_datastore(mount_path: Path, account: str, container_name: str, sas_token: str = None, access_key: str = None):
    if sas_token is None and access_key is None:
        raise ValueError("Either SAS token or access key must be provided")

    env = {
        "AZURE_STORAGE_ACCOUNT": account
    }
    if access_key is not None:
        env["AZURE_ACCESS_KEY"] = access_key
        env["AZURE_STORAGE_AUTH_TYPE"] = "Key"

    if sas_token is not None:
        env["AZURE_STORAGE_AUTH_TYPE"] = "SAS"
        env["AZURE_STORAGE_SAS_TOKEN"] = sas_token

    temp_dir = mkdtemp()

    check_call([
        "blobfuse",
        mount_path,
        f"--container-name={container_name}",
        f"--tmp-path={temp_dir}",
        "-o", "attr_timeout=240",
        "-o", "entry_timeout=240",
        "-o", "negative_timeout=120",
        "--file-cache-timeout-in-seconds=120",
    ], env=env)


def main(args: Arguments):
    if not args.mount_path.exists():
        args.mount_path.mkdir(parents=True)

    if not args.mount_path.is_dir():
        raise ValueError(f"{args.mount_path} is not a directory")

    if len(list(args.mount_path.glob("*"))) > 0:
        raise ValueError(f"{args.mount_path} is not empty")

    ws: Workspace = Workspace.from_config(args.config_path)
    with open(args.config_path, "r") as f:
        config = load(f)
    if "datastore" in config:
        datastore = ws.datastores[config["datastore"]]
    else:
        datastore = ws.get_default_datastore()

    if not isinstance(datastore, AzureBlobDatastore):
        raise ValueError("Datastore must be an Azure Blob Datastore")

    mount_datastore(mount_path=args.mount_path, sas_token=datastore.sas_token, access_key=datastore.account_key,
                    container_name=datastore.container_name, account=datastore.account_name)


if __name__ == "__main__":
    run_and_exit(Arguments, main)
