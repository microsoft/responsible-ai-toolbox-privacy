import pytest
from functools import lru_cache
from azure.ai.ml import MLClient
from azure.storage.blob import ContainerClient

from privacy_estimates.experiments.aml import WorkspaceConfig, Job, ContainerJob


WORKSPACE_DETAILS = {
    "subscription_id": "acc09744-1ee3-4242-b375-93421c63af0c",
    "resource_group": "Singularity",
    "workspace_name": "M365Research",
}


@lru_cache
def is_m365res_ws_available() -> bool:
    try:
        ws = WorkspaceConfig(**WORKSPACE_DETAILS)
        ws.ml_client.workspaces.get(ws.workspace_name)
        return True
    except Exception:
        return False


def test_workspace_config():
    if not is_m365res_ws_available():
        pytest.skip("M365Research workspace is not available")
    ws = WorkspaceConfig(**WORKSPACE_DETAILS)
    assert isinstance(ws.ml_client, MLClient)


def test_download_job():
    if not is_m365res_ws_available():
        pytest.skip("M365Research workspace is not available")

    job = Job.from_url(
        "https://ml.azure.com/experiments/id/807cdd35-4692-4dd2-a35c-d08b45ec59dd/runs/gray_river_vhv6b33wwj?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourceGroups/Singularity/providers/Microsoft.MachineLearningServices/workspaces/M365Research&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#"
    )
    container_client = ContainerClient(
        account_url="https://m365resexternal.blob.core.windows.net/",
        container_name="2025-zhao-guicursor",
        credential=job.ws.credential
    )
    job.save_to_container(container_client)

    container_job = ContainerJob(
        name=job.name, container_client=container_client
    )
    metrics = container_job.get_node("ppo").get_metrics()
    breakpoint()

    container_client.upload_blob()



    blob_job = BlobJob.from_url(
        "",
        container_name=""
    )

