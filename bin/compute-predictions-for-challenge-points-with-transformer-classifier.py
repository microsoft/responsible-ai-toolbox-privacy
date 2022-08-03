import pandas as pd
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from datasets import Dataset, DatasetDict
from pathlib import Path
from ruamel import yaml

from privacy_games.data import select_model_samples, preprocess_text
from privacy_games.predictions import predict


@dataclass
class Arguments:
    challenge_points: Path
    challenge_points_per_model: int
    experiment_dir: Path
    model_glob: str
    params_glob: str
    tokenizer_glob: str
    output: Path


def main(args: Arguments):
    model_paths = list(args.experiment_dir.glob(args.model_glob))
    print(f"Found {len(model_paths)} models in {args.experiment_dir / args.model_glob}")
    tokenizer_paths = list(args.experiment_dir.glob(args.tokenizer_glob))
    print(f"Found {len(tokenizer_paths)} tokenizers in {args.experiment_dir / args.tokenizer_glob}")
    params_paths = list(args.experiment_dir.glob(args.params_glob))
    print(f"Found {len(params_paths)} params in {args.experiment_dir / args.params_glob}")
    model_indices = [int(yaml.safe_load(params_path.read_text())["model_index"]) for params_path in params_paths]

    challenge_points = Dataset.from_parquet(str(args.challenge_points))

    assert len(model_paths) == len(tokenizer_paths)
    assert len(model_paths) == len(params_paths)
    assert len(challenge_points) == len(model_paths)*args.challenge_points_per_model

    all_predictions = pd.DataFrame()
    for model_index, model_path, tokenizer_path in sorted(zip(model_indices, model_paths, tokenizer_paths)):
        print(f"Loading model {model_index}/{len(model_paths)} from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        samples = select_model_samples(D=challenge_points, model_index=model_index, points_per_model=args.challenge_points_per_model)

        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        tokenized_samples = preprocess_text(DatasetDict({"train": samples}), tokenizer=tokenizer)['train']

        predictions = predict(model, tokenized_samples, batch_size=8, device="cuda:0", label_column="label")

        all_predictions = all_predictions.append(predictions)

    print(f"Writing {len(all_predictions)} predictions to file")
    assert len(all_predictions) == len(challenge_points)
    assert len(all_predictions) > 0
    all_predictions.to_parquet(args.output, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    main(args=args)
