from mldesigner import command_component, Input, Output
from datasets import Dataset


@command_component()
def convert_jsonl_to_hf(data: Input(type="uri_file"), output: Output(type="uri_folder")):
    """
    Convert a JSONL file to a Hugging Face dataset.
    """
    Dataset.from_json(data).save_to_disk(output)


@command_component()
def convert_hf_to_jsonl(data: Input(type="uri_folder"), output: Output(type="uri_file")):
    """
    Convert a Hugging Face dataset to a JSONL file.
    """
    Dataset.load_from_disk(data).to_json(output)
