import logging
import transformers
import datasets
import sys

from transformers import Trainer, TrainingArguments, HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from datasets import load_from_disk
from data.preprocess import preprocess_text


logger = logging.getLogger(__name__)


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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if train_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    train_dataset = load_from_disk(data_args.train_data, keep_in_memory=True)
    eval_dataset = load_from_disk(data_args.eval_data, keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    train_dataset = preprocess_text(train_dataset, tokenizer=tokenizer, text_column=data_args.text_column,
                                    max_sequence_length=data_args.max_sequence_length, add_lm_labels=True)
    eval_dataset = preprocess_text(eval_dataset, tokenizer=tokenizer, text_column=data_args.text_column,
                                   max_sequence_length=data_args.max_sequence_length, add_lm_labels=True)

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
