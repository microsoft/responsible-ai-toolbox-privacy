from pathlib import Path
from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    train_base_data_path: Path
    test_data_path: Path
    in_samples_path: Path
    num_classes: int
    max_sequence_length: Optional[int] = field(default=None)


def select_model_samples(D: Dataset, model_index: int, points_per_model: int) -> Dataset:
    """
    Select m samples from the dataset D that are associated with model_index
    """
    assert len(D) % points_per_model == 0
    assert (model_index+1)*points_per_model <= len(D)
    return D.select(range(model_index*points_per_model, (model_index+1)*points_per_model))


def preprocess_text(D: DatasetDict, tokenizer: PreTrainedTokenizerBase, max_sequence_length: int = None) -> DatasetDict:
    if "text_2" in D["train"].column_names:
        processed_data = D.map(
            lambda batch: tokenizer(batch["text"], batch["text_2"], padding="max_length", max_length=max_sequence_length),
            batched=True
        )
        processed_data.remove_columns_(["text", "text_2"])
    else:
        processed_data = D.map(
            lambda batch: tokenizer(batch["text"], padding="max_length", max_length=max_sequence_length),
            batched=True
        )
        processed_data.remove_columns_(["text"])
    return processed_data

