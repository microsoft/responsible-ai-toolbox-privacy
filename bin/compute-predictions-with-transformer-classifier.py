import pandas as pd
import parmap
import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict
from pathlib import Path
from ruamel import yaml
from typing import Tuple, Sequence
from multiprocessing import set_start_method

from privacy_games.predictions import predict
from privacy_games.data import preprocess_text


@dataclass
class Arguments:
    dataset: Path = field(metadata={"help": "Path to the dataset for computing predictions on."})
    experiment_dir: Path = field(metadata={"help": "Path to the experiment directory."})
    model_glob: str = field(metadata={"help": "Glob pattern for the model to use."})
    params_glob: str = field(metadata={"help": "Glob pattern for the ray tune parameters to use."})
    tokenizer_glob: str = field(metadata={"help": "Glob pattern for the tokenizer to use."})
    output: Path = field(metadata={"help": "Path to the output file."})
    batch_size: int = field(default=128, metadata={"help": "Batch size for predictions."})


def tokenizers_equal(tokenizer1: PreTrainedTokenizerBase, tokenizer2: PreTrainedTokenizerBase) -> bool:
    """
    Whether two tokenizers are the same.
    
    We assume that the vocab represented as dict from token index to token string defines a tokenizer completely.
    """
    return tokenizer1.get_vocab() == tokenizer2.get_vocab()


def load_model_and_predict(model_index_path_device: Tuple[int, str, str], tokenized_samples: Dataset, batch_size: int) -> pd.DataFrame:
    """
    Load a model from a path and compute predictions for a dataset

    :param model_index_path_device: Tuple of (model_index, model_path, device)
    :param tokenized_samples: Dataset to compute predictions on
    :param batch_size: Batch size for predictions
    :return: DataFrame of predictions with columns: logits, labels, model_index
    """
    model_index, model_path, device = model_index_path_device
    print(f"Loading model {model_index} from {model_path} on {device}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    predictions = predict(model, tokenized_samples, batch_size=batch_size, device=device, label_column="label")
    predictions["model_index"] = model_index
    return predictions


def compute_prediction_distributed(model_indices: Sequence[int], model_paths: Sequence[str], tokenized_samples: Dataset,
                                   batch_size: int, num_gpus: int) -> pd.DataFrame:
    """
    Compute predictions for a dataset using several models in parallel

    :param model_indices: List of model indices
    :param model_paths: List of model paths
    :param tokenized_samples: Dataset to compute predictions on
    :param batch_size: Batch size for predictions
    :param num_gpus: Number of GPUs to use
    :return: DataFrame of predictions with columns: logits, labels, model_index
    """
    assert len(model_indices) == len(model_paths)
    all_predictions = []
    for i in range(0, len(model_paths), num_gpus):
        model_indices_i = model_indices[i:i+num_gpus]
        model_paths_i = model_paths[i:i+num_gpus]
        devices = [f"cuda:{id}" for id in range(num_gpus)][:len(model_paths)]
        predictions = parmap.map(
            function=load_model_and_predict,
            iterable=zip(model_indices_i, model_paths_i, devices),
            tokenized_samples=tokenized_samples,
            batch_size=batch_size,
            pm_processes=num_gpus,
            pm_chunksize=1
        )
        all_predictions.extend(predictions)
    assert len(all_predictions) == len(model_paths)
    return pd.concat(all_predictions, ignore_index=True)


def main(args: Arguments):
    set_start_method("spawn")
    model_paths = list(args.experiment_dir.glob(args.model_glob))
    print(f"Found {len(model_paths)} models in {args.experiment_dir / args.model_glob}")
    tokenizer_paths = list(args.experiment_dir.glob(args.tokenizer_glob))
    print(f"Found {len(tokenizer_paths)} tokenizers in {args.experiment_dir / args.tokenizer_glob}")
    params_paths = list(args.experiment_dir.glob(args.params_glob))
    print(f"Found {len(params_paths)} params in {args.experiment_dir / args.params_glob}")
    model_indices = [int(yaml.safe_load(params_path.read_text())["model_index"]) for params_path in params_paths]

    dataset = Dataset.from_parquet(str(args.dataset))

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    assert len(model_paths) == len(tokenizer_paths)
    assert len(model_paths) == len(params_paths)

    print(f"Checking if all tokenizers are equal...")
    tokenizers = [AutoTokenizer.from_pretrained(tokenizer_path) for tokenizer_path in tokenizer_paths]
    assert all(tokenizers_equal(tokenizers[0], t) for t in tokenizers[1:])
    print("done")
    print("Tokenizing dataset...")
    tokenized_samples = preprocess_text(DatasetDict({"train": dataset}), tokenizer=tokenizers[0])['train']
    print("done")

    all_predictions = compute_prediction_distributed(model_indices, model_paths, tokenized_samples=tokenized_samples,
                                                     batch_size=args.batch_size, num_gpus=num_gpus)

    print(f"Writing {len(all_predictions)} predictions to file")
    assert len(all_predictions) == len(dataset)*len(model_paths)
    assert len(all_predictions) > 0
    all_predictions.to_parquet(args.output, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    main(args=args)
