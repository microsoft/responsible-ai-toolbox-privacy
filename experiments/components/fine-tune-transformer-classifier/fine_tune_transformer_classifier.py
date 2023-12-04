import numpy as np
import os
import torch
import sys
import yaml
import datasets
from typing import Optional
from transformers import (
    HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification,
    EvalPrediction
)
from dataclasses import dataclass, field
from pathlib import Path
from dp_transformers import TrainingArguments, dp_utils
from dp_transformers import PrivacyArguments as OUPrivacyArguments

from data import preprocess_text


@dataclass
class ModelArguments:
    model_name_or_path: str


@dataclass
class PrivacyArguments(OUPrivacyArguments):
    def __post_init__(self):
        if np.isinf(self.target_epsilon):
            print("Disabling DP because target_epsilon is infinite")
            self.disable_dp = True
        super().__post_init__()


@dataclass
class DataArguments:
    train_data_path: Path
    test_data_path: Path
    max_sequence_length: Optional[int] = field(default=None)


@dataclass
class Arguments:
    training: TrainingArguments
    model: ModelArguments
    data: DataArguments
    privacy: PrivacyArguments


def main(args: Arguments):
    ds_test = datasets.load_from_disk(str(args.data.test_data_path), keep_in_memory=True)
    ds_train = datasets.load_from_disk(str(args.data.train_data_path), keep_in_memory=True)

    num_classes = ds_train.features["label"].num_classes

    model = AutoModelForSequenceClassification.from_pretrained(args.model.model_name_or_path, num_labels=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)

    ds_train = preprocess_text(ds_train, tokenizer=tokenizer, max_sequence_length=args.data.max_sequence_length, text_column="sentence")
    if len(ds_test) > 0:
        ds_test = preprocess_text(ds_test, tokenizer=tokenizer, max_sequence_length=args.data.max_sequence_length, text_column="sentence")

    os.makedirs(args.training.output_dir, exist_ok=True)
    with open(os.path.join(args.training.output_dir, "arguments.yml"), "w+") as f:
        yaml.dump(args, f)
    print(yaml.dump(args))

    model.train()
    print(f"{torch.cuda.memory_summary(device=args.training.device)}")
    print(f"Sending model to {args.training.device}")
    model = model.to(args.training.device)
    print(f"Model sent to {args.training.device}")
    print(f"{torch.cuda.memory_summary(device=args.training.device)}")

    if (not args.training.no_cuda) and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is not available. Please use --no-cuda to run this script.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        if len(preds) == 0:
            accuracy = np.nan
        else:
            accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    trainer = dp_utils.OpacusDPTrainer(
        args=args.training,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        privacy_args=args.privacy
    )

    try:
        trainer.train()
    finally:
        trainer.save_model()

    if args.privacy.disable_dp:
        trainer.log({"epsilon_final": float('inf')})
    else:
        trainer.log({"epsilon_final": trainer.get_prv_epsilon()})

    print(f"{torch.cuda.memory_summary(device=args.training.device)}")
    print("Training successful. Exiting...")
    return 0


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments, PrivacyArguments))
    training_args, data_args, model_args, privacy_args = parser.parse_args_into_dataclasses()
    args = Arguments(training=training_args, data=data_args, model=model_args, privacy=privacy_args)
    sys.exit(main(args))
