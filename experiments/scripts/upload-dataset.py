from tempfile import TemporaryDirectory
from pathlib import Path
from abc import ABC
from pydantic_cli import run_and_exit
from pydantic import Field, BaseModel
from pathlib import Path
from datasets import Dataset, Features, ClassLabel, Array3D, load_dataset
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from privacy_estimates.experiments.aml import WorkspaceConfig, RegistryConfig


class DatasetLoader(ABC):
    def __init__(self, split: str):
        self.split = split

    def as_dataset(self) -> Dataset:
        raise NotImplementedError("This method must be implemented by a subclass")

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}-{self.split}"

    @property
    def description(self) -> str:
        return ""


class CIFAR10Normalized(DatasetLoader):
    CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
    CIFAR10_STD  = (0.24703223, 0.24348513, 0.26158784)

    def __init__(self, split: str):
        super().__init__(split=split)

    def as_dataset(self) -> Dataset:
        from torchvision import transforms
        from torchvision.datasets import CIFAR10 as TorchVisionCIFAR10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10Normalized.CIFAR10_MEAN, CIFAR10Normalized.CIFAR10_STD)
        ])

        assert self.split in ["train", "test"]
        is_train = (self.split == "train")

        with TemporaryDirectory() as dataset_dir:
            dataset = TorchVisionCIFAR10(
                root=f"{dataset_dir}/cifar10",
                train=is_train,
                download=True,
                transform=transform
            )

            ds = Dataset.from_generator(
                lambda: ({"image": img, "label": label} for img, label in dataset),
                features=Features({"image": Array3D(dtype="float32", shape=(3, 32, 32)), "label": ClassLabel(num_classes=10)})
            )
        ds.info.description = self.description

        return ds

    @property
    def description(self) -> str:
        return f"Normalized CIFAR10 dataset with mean {CIFAR10Normalized.CIFAR10_MEAN} and std {CIFAR10Normalized.CIFAR10_STD}"


class SST2(DatasetLoader):
    def __init__(self, split: str):
        super().__init__(split=split)

    def as_dataset(self) -> Dataset:
        assert self.split in ["train", "test"]

        datasets_split = "train"
        if self.split == "test":
            # datasets does not have a valid test split for SST2
            datasets_split = "validation"

        ds = load_dataset("glue", "sst2", split=datasets_split)
        ds = ds.select_columns(["sentence", "label"])

        return ds

    @property
    def description(self) -> str:
        return "GLUE SST2 dataset in the HuggingFace format"


class AmazonPolarity5k(DatasetLoader):
    def __init__(self, split: str):
        super().__init__(split=split)

    def as_dataset(self) -> Dataset:
        assert self.split in ["train", "test"]

        ds = load_dataset("amazon_polarity", split=self.split  + "[:5000]")
        ds = ds.rename_columns({"content": "sentence"}).select_columns(["sentence", "label"])

        return ds

    @property
    def description(self) -> str:
        return "Amazon Polarity 5k dataset in the HuggingFace format"


AVAILABLE_DATASETS = { c.__name__: c for c in DatasetLoader.__subclasses__() }


def get_dataset_loader(dataset_name: str, split: str) -> DatasetLoader:
    """
    Returns a DatasetLoader instance for the given dataset name and split.
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not available. Available datasets are: {AVAILABLE_DATASETS.keys()}")
    return AVAILABLE_DATASETS[dataset_name](split=split)


class Arguments(BaseModel):
    dataset_name: str = Field(
        description="Name of the dataset to load",
        choices=AVAILABLE_DATASETS.keys(),
        cli=["--dataset-name"],
    )
    split: str = Field(
        description="Split of the dataset to load",
    )
    workspace_config: Path = Field(
        default=None,
        cli=["--workspace-config"],
        description="Path to the workspace config file",
    )
    registry_name: str = Field(
        default=None,
        cli=["--registry-name"],
        description="Name of the registry to use [Optional]",
    )
    registry_location: str = Field(
        default=None,
        cli=["--registry-location"],
        description="Location of the registry to use [Optional]",
    )
    version: str = Field(
        default=None,
        cli=["--version"],
        description="Version of the dataset to upload",
    )


def main(args: Arguments):
    loader = get_dataset_loader(dataset_name=args.dataset_name, split=args.split)
    if args.workspace_config is None == (args.registry_name is None and args.registry_location is None):
        raise ValueError("Either workspace_config or registry_name and registry_location must be provided")
    if args.workspace_config is not None:
        client = WorkspaceConfig.from_yaml(args.workspace_config).ml_client
    else:
        client = RegistryConfig(registry_name=args.registry_name, location=args.registry_location).ml_client

    with TemporaryDirectory() as dataset_dir:
        loader.as_dataset().save_to_disk(dataset_dir)
        data = Data(
            path=dataset_dir,
            type=AssetTypes.URI_FOLDER,
            description=loader.description,
            name=loader.name,
            version=args.version,
        )
        client.data.create_or_update(data)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
