import sys
import numpy as np
import torch
import tqdm

from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datasets import Dataset, load_from_disk, features, concatenate_datasets
from pathlib import Path
from typing import Optional
from azureml.core import Run

sys.path.append(str(Path(__file__).parent.parent.parent))
from data import preprocess_text


@dataclass
class Arguments:
    dataset: Path = field(metadata={"help": "Path to the dataset for computing predictions on."})
    experiment_dir: Path = field(metadata={"help": "Path to the experiment directory."})
    model_rel_path: str = field(metadata={"help": "Glob pattern for the model to use."})
    params_rel_path: str = field(metadata={"help": "Glob pattern for the ray tune parameters to use."})
    tokenizer_rel_path: str = field(metadata={"help": "Glob pattern for the tokenizer to use."})
    output: Path = field(metadata={"help": "Path to the output file."})
    batch_size: int = field(default=128, metadata={"help": "Batch size for predictions."})


def predict(model: nn.Module, data: Dataset, batch_size: int, device: Optional[str] = None, label_column: str = "labels") -> Dataset:
    """
    Compute predictions for a dataset using a model

    :param model: Model to use for predictions. We assume that the model returns a struct with a `logits` member
    :param data: Dataset to compute predictions on
    :param batch_size: Batch size for predictions
    :param device: Device to use for predictions
    :param label_column: Name of the column in the dataset that contains the labels
    :return: DataFrame of predictions with columns: logits, labels
    """
    model.to(device)

    prediction_features = features.Features({
        "logits": features.Sequence(feature=features.Value("float32")),
        "label": data.features["label"],
        "loss": features.Value("float32")
    })
    predictions = []
    with data.formatted_as(type="torch"):
        with torch.no_grad():
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch in tqdm.tqdm(data_loader):
                batch = {k: t.to(model.device) for k, t in batch.items()}
                labels = batch.pop(label_column)
                output = model(**batch, labels=labels)

                loss_fct = nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(output.logits, labels)
                predictions.append(
                    Dataset.from_dict({
                            "logits": [output.logits.cpu().numpy()[i] for i in range(len(labels))],
                            "label": [labels.cpu().numpy()[i] for i in range(len(labels))],
                            "loss": [loss.cpu().numpy()[i] for i in range(len(labels))],
                        }, features=prediction_features
                    )
                )
    return concatenate_datasets(predictions)


def main(args: Arguments):
    dataset = load_from_disk(str(args.dataset), keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(args.experiment_dir / args.tokenizer_rel_path)

    print("Tokenizing dataset...")
    tokenized_samples = preprocess_text(dataset, tokenizer=tokenizer, text_column="sentence")
    print("done")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.experiment_dir / args.model_rel_path)
    print("done")

    predictions = predict(model, tokenized_samples, batch_size=args.batch_size, label_column="label", device="cuda")
    assert len(predictions) == len(dataset), f"Expected {len(dataset)} predictions, got {len(predictions)}"

    if len(predictions) > 0:
        loss_avg = np.mean(predictions["loss"])
        accuracy = (np.stack(predictions["label"]) == np.stack(predictions["logits"]).argmax(axis=1)).mean()
    else:
        loss_avg = np.nan
        accuracy = np.nan

    run = Run.get_context()
    run.log("loss_avg", loss_avg)
    print(f"loss_avg: {loss_avg}")
    run.log("accuracy", accuracy)
    print(f"accuracy: {accuracy}")

    print(f"Writing {len(predictions)} predictions to file")
    predictions.save_to_disk(str(args.output))


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    main(args=args)
