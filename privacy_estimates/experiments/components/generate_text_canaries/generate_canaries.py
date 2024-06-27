import numpy as np
import os

from mldesigner import command_component, Input, Output
from datasets import Dataset, Features, Value
from typing import Optional, List, Dict, Literal
from enum import Enum
from guidance import models, gen, user, system, assistant
from functools import partial
from dataclasses import dataclass


class Format(Enum):
    CHAT = "chat"
    LANGUAGE_MODELLING = "language_modelling"


def generate_credit_card_number_canaries(
    num_canaries: int, seed: int, format: Format, text_column: Optional[str] = None
) -> Dataset:
    rng = np.random.default_rng(seed)
    secrets = [" ".join([str(rng.integers(0, 10000)).zfill(4) for _ in range(4)]) for _ in range(num_canaries)]

    if format == Format.LANGUAGE_MODELLING:
        text = ["My credit card number is: " + s for s in secrets]
        canaries = Dataset.from_dict({text_column: text})
    elif format == Format.CHAT:
        system_prompt = "You are a helpful assistant that memorizes all information shared with you."
        user_prompt = "My credit card number is: "
        assistant_responses = secrets
        canaries = Dataset.from_dict({
            "messages": [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response}
                ] for response in assistant_responses
            ]
        },
        features=Features({
            'messages': [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]
        }))
    return canaries


@dataclass
class AOAIConfig:
    model: str
    azure_endpoint: str
    azure_deployment: str
    api_key: str
    version: str

    def get_model(self) -> models.AzureOpenAI:
        return models.AzureOpenAI(
            model=self.model,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=self.api_key,
            version=self.version,
            echo=False,
        )


def generate_natural_chat_canary(model_config: AOAIConfig, temp: float) -> Dict[Literal["messages"], List[Dict[Literal["system", "user", "assistant"], str]]]:
    model = model_config.get_model()

    with system():
        model += (
            "I am trying to audit the privacy of a language model by inserting canaries and then performing a membership "
            "inference attack. I need to generate a set of canaries that are likely to be memorized by the model. This means "
            "that the topic should be uncommon and unique. The canary is formatted as a chat message with a "
            "user prompt, and an assistant response. The user prompt should be short but the assistant response can "
            "be longer ideally 100 tokens or more. Please generate the text directly without any additional responses or "
            "prompts such as 'User Prompt: ' or anything like that."
        )

    with user():
        model += "Please generate the user prompt."
    with assistant():
        model += gen(name="user_prompt", temperature=temp)
    with user():
        model += "Please generate the assistant response."
    with assistant(): 
        model += gen(name="assistant_response", temperature=temp)

    return {"messages": [
        {"role": "system", "content": "You're a helpful assistant that memorizes all information shared with you."},
        {"role": "user", "content": model["user_prompt"]},
        {"role": "assistant", "content": model["assistant_response"]}
    ]}



def generate_natural_canaries_with_aoai(
    num_canaries: int, seed: int, format: Format, model: str, azure_endpoint: str, azure_deployment: str,
    keyvault_api_key_name: Optional[str] = None
) -> Dataset:

    if keyvault_api_key_name is not None:
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential, path="path/to/config.json")
        vault_name = os.path.basename(ml_client.workspaces.get(ml_client.workspace_name).key_vault)
        secret_client = SecretClient(vault_url=f"https://{vault_name}.azure.net/", credential=credential)
        secret_client.set_secret("my_secret_name", "XXX")


    model_config = AOAIConfig(
        model=model,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        version="2023-12-01-preview",
    )

    canaries = Dataset.from_dict({"index": range(num_canaries)})
    canaries = canaries.map(
        partial(generate_natural_chat_canary, model_config=model_config, temp=0.5),
        input_columns=[], remove_columns=["index"], num_proc=100
    )
    return canaries


@command_component(environment="environment.aml.yaml")
def generate_natural_canaries(
    num_canaries: int, seed: int, format: str, output: Output(type="uri_folder")
):
    format = Format(format)
    if format == Format.LANGUAGE_MODELLING:
        raise NotImplementedError("Natural canaries are only supported in chat format.")
    elif format == Format.CHAT:
        canaries = generate_gpt4_canaries(num_canaries=num_canaries, seed=seed, format=format)
    else:
        raise ValueError(f"Invalid format: {format}")
    assert len(canaries) == num_canaries
    canaries.save_to_disk(output)


@command_component(environment="environment.aml.yaml")
def generate_canaries_with_secrets(
    num_canaries: int, seed: int, format: str, output: Output(type="uri_folder"),
    text_column: Input(type="string", optional=True) = None
):
    format = Format(format)

    canaries = generate_credit_card_number_canaries(num_canaries=num_canaries, seed=seed, format=format,
                                                    text_column=text_column)
    assert len(canaries) == num_canaries
    canaries.save_to_disk(output)
