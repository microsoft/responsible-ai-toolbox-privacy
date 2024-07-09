import logging
from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset
from pathlib import Path


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
})
def create_empty_dataset(
    dataset_for_features: Input,
    dataset: Output,
):

    fmt = f"%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler()])

    features = load_from_disk(dataset_for_features).features

    logger.info(f"Creating empty dataset with features: {features}")

    Dataset.from_dict(mapping={k: [] for k in features}, features=features).save_to_disk(dataset)
