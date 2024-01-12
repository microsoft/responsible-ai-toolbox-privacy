from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def preprocess_text(D: Dataset, tokenizer: PreTrainedTokenizerBase, text_column: str = None, text_2_column: str = None,
                    max_sequence_length: int = None) -> Dataset:
    if len(D) == 0:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "label": []})
    if text_2_column in D.column_names:
        processed_data = D.map(
            lambda batch: tokenizer(batch[text_column], batch[text_2_column], padding="max_length",
                                    max_length=max_sequence_length),
            batched=True
        )
    else:
        processed_data = D.map(
            lambda batch: tokenizer(batch[text_column], padding="max_length", max_length=max_sequence_length),
            batched=True
        )
    processed_data = processed_data.select_columns(["input_ids", "attention_mask", "label"])
    return processed_data

