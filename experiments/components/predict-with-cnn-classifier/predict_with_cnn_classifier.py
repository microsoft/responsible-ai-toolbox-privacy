import sys
import torch
import numpy as np

from datasets import load_from_disk, Dataset, Features, Sequence, Value
from pathlib import Path
from typing import Dict, List
from pydantic_cli import run_and_exit
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, TensorDataset
from functools import partial


sys.path.append(str(Path(__file__).parent.parent.parent))
from models.cnn import CNN, compute_prediction_metrics, collate_image_batch


class Arguments(BaseModel):
    dataset: Path = Field(
        description="Path to the dataset for computing predictions on."
    )
    experiment_dir: Path = Field(
       description="Path to the experiment directory."
    )
    model_rel_path: str = Field(
        default="./", description="Glob pattern for the model to use."
    )
    output: Path = Field(
        description="Path to the output file."
    )
    batch_size: int = Field(
        description="Batch size."
    )
    use_cpu: int = Field(
        default=0, description="Whether to use the CPU instead of the GPU."
    )


def main(args: Arguments):
    data = load_from_disk(str(args.dataset))

    print(f"Loaded dataset: {data.features}")

    model = CNN.load(args.experiment_dir / args.model_rel_path / "model.pt")

    device = "cpu" if args.use_cpu else "cuda"

    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_image_batch, device="cpu")
    )


    metrics = compute_prediction_metrics(model=model, device=device, data_loader=data_loader)
    print(metrics)

    predictions = Dataset.from_dict(
        mapping={
            "logits": [logits for logits in metrics.logits] if len(metrics.losses) > 0 else [],
            "label": metrics.labels,
            "loss": metrics.losses,
        },
        features=Features({
            "logits": Sequence(feature=Value(dtype="float64"), length=-1),
            "label": data.features["label"],
            "loss": Value(dtype="float64")
        })
    )

    print(f"Writing {len(predictions)} predictions to file")
    assert len(predictions) == len(data)
    predictions.save_to_disk(args.output)

    return 0


def exception_handler(ex):
    raise RuntimeError("Command ran with an error.") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
