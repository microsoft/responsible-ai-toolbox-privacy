import numpy as np
import os
import torch
import sys
import opacus_utils
import yaml
import datasets
from transformers import (
    HfArgumentParser, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, EvalPrediction
)
from privacy_games.data import select_model_samples, DataArguments, preprocess_text
from dataclasses import dataclass
from opacus import PrivacyEngine
from opacus_utils import PrivacyEngineCallback, TrainingArguments
from opacus_utils import PrivacyArguments as OUPrivacyArguments
from prv_accountant import Accountant


@dataclass
class ModelArguments:
    model_name: str


@dataclass
class DataSplitArguments:
    model_index: int
    m: int


@dataclass
class PrivacyArguments(OUPrivacyArguments):
    delta: float = None


@dataclass
class Arguments:
    training: TrainingArguments
    model: ModelArguments
    data: DataArguments
    data_split: DataSplitArguments
    privacy: PrivacyArguments


def main(args: Arguments):
    ds = datasets.DatasetDict()
    ds["test"] = datasets.Dataset.from_parquet(str(args.data.test_data_path))
    ds["train"] = datasets.Dataset.from_parquet(str(args.data.train_base_data_path))
    in_samples = datasets.Dataset.from_parquet(str(args.data.in_samples_path))
    ds["train"] = datasets.concatenate_datasets([
        ds["train"], select_model_samples(in_samples, model_index=args.data_split.model_index, points_per_model=args.data_split.m)
    ])

    # Hopefully a safe assumption that both training and test datasets contain all classes
    assert max(ds["train"]["label"]) == args.data.num_classes - 1
    assert max(ds["test"]["label"]) == args.data.num_classes - 1

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name, num_labels=args.data.num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    ds = preprocess_text(ds, tokenizer=tokenizer, max_sequence_length=args.data.max_sequence_length)

    with open(os.path.join(args.training.output_dir, "arguments.yml"), "w+") as f:
        yaml.dump(args, f)
    print(yaml.dump(args))

    model.train()
    model = model.to(args.training.device)
    print(f"Sending model to {args.training.device}")

    if (not args.training.no_cuda) and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is not available. Please use --no-cuda to run this script.")

    callbacks = []
    if not args.privacy.disable_dp:
        sampling_probability = training_args.train_batch_size/len(ds["train"])
        num_steps = int(np.ceil(1/sampling_probability)*training_args.num_train_epochs)
        target_delta = args.privacy.delta
        noise_multiplier = opacus_utils.dp_utils.find_noise_multiplier(
            sampling_probability=sampling_probability, num_steps=num_steps, target_epsilon=args.privacy.target_epsilon,
            target_delta=target_delta,
            eps_error=0.1
        )
        engine = PrivacyEngine(
            module=model,
            batch_size=training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps,
            sample_size=len(ds['train']),
            noise_multiplier=noise_multiplier,
            max_grad_norm=args.privacy.per_sample_max_grad_norm
        )
        accountant = Accountant(
            noise_multiplier=noise_multiplier, sampling_probability=sampling_probability, max_compositions=num_steps,
            eps_error=0.2, delta=target_delta
        )
        privacy_callback = PrivacyEngineCallback(engine, compute_epsilon=lambda s: accountant.compute_epsilon(s)[2])
        callbacks.append(privacy_callback)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    try:
        trainer.train()
    finally:
        trainer.save_model()

    if args.privacy.disable_dp:
        trainer.log({"epsilon_final": float('inf')})
    else:
        trainer.log({"epsilon_final": accountant.compute_epsilon(engine.steps)[2]})

    print("Training successful. Exiting...")
    return 0


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments, PrivacyArguments, DataSplitArguments))
    training_args, data_args, model_args, privacy_args, data_split_args = parser.parse_args_into_dataclasses()
    args = Arguments(training=training_args, data=data_args, model=model_args, privacy=privacy_args,
                     data_split=data_split_args)
    sys.exit(main(args))
