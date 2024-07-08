from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from datasets import load_from_disk
from functools import partial


@dataclass
class DataArguments:
    train_data: str
    eval_data: str
    text_column: str
    max_sequence_length: int


@dataclass
class ModelArguments:
    model_name_or_path: str


def main(train_args: TrainingArguments, data_args: DataArguments, model_args: ModelArguments):
    train_dataset = load_from_disk(data_args.train_data, keep_in_memory=True)
    eval_dataset = load_from_disk(data_args.eval_data, keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    train_dataset = train_dataset.map(
        lambda x: tokenizer(x[data_args.text_column], truncation=True, padding="max_length", max_length=data_args.max_sequence_length),
        batched=True, remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer(x[data_args.text_column], truncation=True, padding="max_length", max_length=data_args.max_sequence_length),
        batched=True, remove_columns=eval_dataset.column_names
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()
    main(train_args, data_args, model_args)
