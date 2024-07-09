from mldesigner import command_component, Input, Output
from datasets import load_from_disk, concatenate_datasets
from functools import partial
from typing import Dict, List, Any, Callable
from pathlib import Path


MEMBER_SUFFIX = "_in"
NON_MEMBER_SUFFIX = "_out"


def select_prediction_for_challenge(
        row: Dict[str, Any], mi_label_str2int: Callable[[str], int], columns: List[str]
    ) -> Dict[str, Any]:
    """
    Selects the prediction values from either the in or out column depending on the challenge bit.

    Args:
        row (Dict[str, Any]): The row containing the prediction values.
        mi_label_str2int (Callable[[str], int]): The function to map label strings to integers.
        columns (List[str]): The list of columns to select the prediction values from.

    Returns:
        Dict[str, Any]: A dictionary containing the selected prediction values for the challenge.
    """
    return {
        col: row[col + MEMBER_SUFFIX] if row["challenge_bit"] == mi_label_str2int("member") else row[col + NON_MEMBER_SUFFIX]
        for col
        in columns
    }


@command_component(environment={
    "conda_file": Path(__file__).parent / "environment.conda.yaml",
    "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
})
def convert_in_out_to_challenge(
    predictions_in: Input, predictions_out: Input, all_challenge_bits: Input, challenge: Output, challenge_bits: Output,
):
    in_predictions_ds = load_from_disk(predictions_in, keep_in_memory=True)
    out_predictions_ds = load_from_disk(predictions_out, keep_in_memory=True)

    assert len(in_predictions_ds) == len(out_predictions_ds)

    challenge_bits_ds = load_from_disk(all_challenge_bits, keep_in_memory=True).select(range(len(in_predictions_ds)))
    challenge_bits_ds.save_to_disk(challenge_bits)

    prediction_columns = in_predictions_ds.column_names

    mi_label_str2int = challenge_bits_ds.features["challenge_bit"].str2int
    mi_label_int2str = challenge_bits_ds.features["challenge_bit"].int2str

    for row_in, row_out, challenge_bit in zip(in_predictions_ds, out_predictions_ds, challenge_bits_ds["challenge_bit"]):
        in_is_none = all([row_in["split"] is None, row_in["sample_index"] is None])
        out_is_none = all([row_out["split"] is None, row_out["sample_index"] is None])
        assert in_is_none != out_is_none, f"Either in or out must be None, but both are None for {row_in} and {row_out}"
        assert (mi_label_int2str(challenge_bit) == "member") == out_is_none, (
            f"Challenge bit indicates {mi_label_int2str(challenge_bit)} but does not match in/out for {row_in} and {row_out}"
        )

    in_predictions_ds = in_predictions_ds.rename_columns({k: k + MEMBER_SUFFIX for k in in_predictions_ds.column_names})
    out_predictions_ds = out_predictions_ds.rename_columns({k: k + NON_MEMBER_SUFFIX for k in out_predictions_ds.column_names})

    predictions = concatenate_datasets([in_predictions_ds, out_predictions_ds], axis=1)
    predictions = predictions.add_column("challenge_bit", challenge_bits_ds["challenge_bit"])

    print(f"Number of challenge points: {len(predictions)}")
    print(f"Merged prediction columns: {predictions.features}")

    challenge_ds = predictions.map(
        partial(select_prediction_for_challenge, mi_label_str2int=mi_label_str2int, columns=prediction_columns),
        remove_columns=predictions.column_names,
    )

    for row in challenge_ds:
        assert any(row[col] is not None for col in prediction_columns), f"Missing prediction for {row}"

    challenge_ds.save_to_disk(challenge)
