import numpy as np
from mldesigner import command_component, Input, Output
from datasets import Dataset
from enum import Enum


class Format(Enum):
    CHAT = "chat"
    LANGUAGE_MODELLING = "language_modelling"


@command_component(environment="environment.aml.yaml")
def generate_canaries_with_secrets(
    num_canaries: int, seed: int, format: str, output: Output(type="uri_folder"),
    text_column: Input(type="string", optional=True) = None
):
    format = Format(format)
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
        })

    canaries.save_to_disk(output)

