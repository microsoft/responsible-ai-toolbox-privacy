from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from functools import partial
from typing import Optional, Sequence, Dict


def process_batch(
    batch: Dict[str, Sequence[str]], tokenizer: PreTrainedTokenizerBase, text_column: str, text_2_column: Optional[str],
    max_sequence_length: int, add_lm_labels: bool = False
):
    if text_2_column is not None:
        tokenized_batch = tokenizer(batch[text_column], batch[text_2_column], padding="max_length", max_length=max_sequence_length)
    else:
        tokenized_batch = tokenizer(batch[text_column], padding="max_length", max_length=max_sequence_length)
    if add_lm_labels:
        tokenized_batch["label"] = tokenized_batch["input_ids"]
    return tokenized_batch


def preprocess_text(D: Dataset, tokenizer: PreTrainedTokenizerBase, text_column: str = None, text_2_column: str = None,
                    max_sequence_length: int = None, add_lm_labels: bool = False) -> Dataset:
    if len(D) == 0:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "label": []})
    processed_data = D.map(
        partial(
            process_batch,
            tokenizer=tokenizer, text_column=text_column, text_2_column=text_2_column, max_sequence_length=max_sequence_length,
            add_lm_labels=add_lm_labels
        ), batched=True)
    processed_data = processed_data.select_columns(["input_ids", "attention_mask", "label"])
    return processed_data

