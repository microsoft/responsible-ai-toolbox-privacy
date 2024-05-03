from signals import compute_mi_signals
from typing import List
from pathlib import Path
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from datasets import load_from_disk


class Arguments(BaseModel):
    logits_and_labels: Path
    mi_signals: Path
    method: str
    extra_args: List[str] = []

def main(args: Arguments) -> int:
    logits_and_labels = load_from_disk(args.logits_and_labels)

    extra_args = {arg.split("=")[0]: arg.split("=")[1] for arg in args.extra_args}

    mi_signal = logits_and_labels.map(
        lambda logits, labels: {"mi_signal": compute_mi_signals(logits, labels, args.method, **extra_args)},
        batched=True, input_columns=["logits", "label"], remove_columns=logits_and_labels.column_names
    )

    mi_signal.save_to_disk(args.mi_signals)

    return 0


def exception_handler(exception: Exception) -> int:
    raise RuntimeError("An exception occurred") from exception


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
