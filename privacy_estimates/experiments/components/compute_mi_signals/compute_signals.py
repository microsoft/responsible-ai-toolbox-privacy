from signals import compute_mi_signals, PredictionFormat
from pathlib import Path
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from datasets import load_from_disk
from ast import literal_eval
from enum import Enum



class Arguments(BaseModel):
    predictions_and_labels: Path
    mi_signals: Path
    method: str
    predictions_format: PredictionFormat
    prediction_column: str
    label_column: str = "label"
    extra_args: str = ""


def main(args: Arguments) -> int:
    predictions_and_labels = load_from_disk(args.predictions_and_labels)

    extra_args = {arg.split("=")[0]: literal_eval(arg.split("=")[1]) for arg in args.extra_args.split()}

    predictions_and_labels.set_format(type="numpy")

    mi_signal = predictions_and_labels.map(
        lambda predictions, labels: {"mi_signal": compute_mi_signals(predictions, labels, args.method, **extra_args)},
        batched=True, input_columns=[args.prediction_column, args.label_column],
        remove_columns=predictions_and_labels.column_names
    )

    mi_signal.save_to_disk(args.mi_signals)

    return 0


def exception_handler(exception: Exception) -> int:
    raise RuntimeError("An exception occurred") from exception


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
