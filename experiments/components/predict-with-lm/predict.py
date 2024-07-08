# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datasets
import transformers
import sys
import logging
import torch
import numpy as np
import multiprocess as mp
import copy

from torch import nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from privacy_estimates.experiments.attacks.signals import Signal, SIGNALS


logger = logging.getLogger(__name__)
mp.set_start_method('spawn', force=True)  # force=True can be used to reset the method if needed elsewhere


@dataclass
class Arguments:
    model_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to the base model"
    })
    per_device_batch_size: int = field(default=8, metadata={
        "help": "Batch size per device"
    })
    data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to tokenized data in HF dataset format"
    })
    predictions_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to save predictions"
    })
    log_level: str = field(default="INFO", metadata={
        "help": "Log level"
    })
    use_cpu: bool = field(default=False, metadata={
        "help": "Whether to use CPU for training"
    })
    disable_distributed: bool = field(default=False, metadata={
        "help": "Whether to disable distributed inference."
    })

    def __post_init__(self):
        self.log_level = logging.getLevelName(self.log_level.upper())


class DistributedEvaluator:
    def __init__(self, model: nn.Module, devices: List[str], signal_method: Signal, signal_aggregation: str):
        logger.info(f"Initializing evaluator on devices {devices}")
        self.models = [copy.deepcopy(model).to(d) for d in devices]
        self.devices = devices
        self.signal_method = signal_method
        self.signal_aggregation = signal_aggregation

    def evaluate(self, batch: Dict[str, torch.Tensor], rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if rank is None:
            rank = 0
        device = self.devices[rank]
        model = self.models[rank]
        model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = output.logits[:, :-1, :] # remove the last token prediction
            labels = batch["labels"][:, 1:] # Remove the first token in the labels

            completion_mask_np = attention_mask.cpu().numpy()

        labels_np = labels.cpu().numpy()
        completion_mask_np[labels_np == -100] = 0

        mi_signal = self.signal_method.compute_mi_signal_from_logits(
            logits=logits.cpu().numpy(), labels=labels.cpu().numpy(), completion_mask=completion_mask_np
        ).sum(axis=1, where=completion_mask_np)

        assert np.isnan(mi_signal).any() == False, "NaN values in MI signal"
        return {"mi_signal": mi_signal}


def main(args: Arguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.log_level
    logging.getLogger().setLevel(level=log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Parameters: {args}")
    logger.info(f"MP start method: {mp.get_start_method()}")

    # Load dataset
    dataset: datasets.Dataset = datasets.load_from_disk(args.tokenized_data_path, keep_in_memory=True)



    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    logger.info(f"Loading model: {args.model_path}")

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path)

    mi_signal_method = SIGNALS[args.mi_signal_method](**args.mi_signal_extra_args)

    if args.use_cpu:
        devices = [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

    num_proc = len(devices)
    if args.disable_distributed:
        num_proc = None

    evaluator = DistributedEvaluator(model=model, devices=devices, signal_method=mi_signal_method,
                                     signal_aggregation=args.mi_signal_aggregation)
    print('Batch size: ', args.per_device_batch_size)
    print('Dataset: ', dataset[0])
    results = dataset.map(
        evaluator.evaluate,
        batched=True, batch_size=args.per_device_batch_size,
        num_proc=num_proc, remove_columns=dataset.column_names,
        with_rank=True
    )
    results.set_format(None)
    assert len(results) == len(dataset)
    results.save_to_disk(args.predictions_path)


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((Arguments,))
    args, = arg_parser.parse_args_into_dataclasses()
    main(args)
