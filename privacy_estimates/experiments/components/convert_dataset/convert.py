from mldesigner import command_component, Input, Output
from datasets import Dataset, Features, Value
from pathlib import Path


ENV = {
    "conda_file": Path(__file__).parent/"environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
}


@command_component(environment=ENV)
def convert_chat_jsonl_to_hfd(data: Input(type="uri_file"), output: Output(type="uri_folder")):
    """
    Convert a JSONL file to a Hugging Face dataset.
    """
    path = Path(data)
    if path.is_file():
        paths = [path]
    else:
        paths = list(path.glob("*.jsonl")) + list(path.glob("*.json"))
    paths = [str(p) for p in paths]
    feats = Features({
        'messages': [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
    })
    Dataset.from_json(paths, features=feats).save_to_disk(output)


@command_component(environment=ENV)
def convert_hfd_to_jsonl(data: Input(type="uri_folder"), output: Output(type="uri_file")):
    """
    Convert a Hugging Face dataset to a JSONL file.
    """
    Dataset.load_from_disk(data).to_json(output)
