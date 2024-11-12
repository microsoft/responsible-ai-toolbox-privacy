from mldesigner import command_component, Input, Output
from datasets import load_from_disk, concatenate_datasets
from functools import partial
from typing import Dict, List, Any, Callable
from pathlib import Path


MEMBER_SUFFIX = "_in"
NON_MEMBER_SUFFIX = "_out"


def select_score_for_challenge(
        row: Dict[str, Any], mi_label_str2int: Callable[[str], int], columns: List[str]
    ) -> Dict[str, Any]:
    """
    Selects the score values from either the in or out column depending on the challenge bit.

    Args:
        row (Dict[str, Any]): The row containing the score values.
        mi_label_str2int (Callable[[str], int]): The function to map label strings to integers.
        columns (List[str]): The list of columns to select the score values from.

    Returns:
        Dict[str, Any]: A dictionary containing the selected score values for the challenge.
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
    scores_in: Input, scores_out: Input, all_challenge_bits: Input, challenge: Output, challenge_bits: Output,
):
    in_scores_ds = load_from_disk(scores_in, keep_in_memory=True)
    out_scores_ds = load_from_disk(scores_out, keep_in_memory=True)

    assert len(in_scores_ds) == len(out_scores_ds)

    challenge_bits_ds = load_from_disk(all_challenge_bits, keep_in_memory=True).select(range(len(in_scores_ds)))
    challenge_bits_ds.save_to_disk(challenge_bits)

    score_columns = in_scores_ds.column_names

    mi_label_str2int = challenge_bits_ds.features["challenge_bit"].str2int
    mi_label_int2str = challenge_bits_ds.features["challenge_bit"].int2str

    for row_in, row_out, challenge_bit in zip(in_scores_ds, out_scores_ds, challenge_bits_ds["challenge_bit"]):
        in_is_none = all([row_in["split"] is None, row_in["sample_index"] is None])
        out_is_none = all([row_out["split"] is None, row_out["sample_index"] is None])
        assert in_is_none != out_is_none, f"Either in or out must be None, but both are None for {row_in} and {row_out}"
        assert (mi_label_int2str(challenge_bit) == "member") == out_is_none, (
            f"Challenge bit indicates {mi_label_int2str(challenge_bit)} but does not match in/out for {row_in} and {row_out}"
        )

    in_scores_ds = in_scores_ds.rename_columns({k: k + MEMBER_SUFFIX for k in in_scores_ds.column_names})
    out_scores_ds = out_scores_ds.rename_columns({k: k + NON_MEMBER_SUFFIX for k in out_scores_ds.column_names})

    scores = concatenate_datasets([in_scores_ds, out_scores_ds], axis=1)
    scores = scores.add_column("challenge_bit", challenge_bits_ds["challenge_bit"])

    print(f"Number of challenge points: {len(scores)}")
    print(f"Merged score columns: {scores.features}")

    challenge_ds = scores.map(
        partial(select_score_for_challenge, mi_label_str2int=mi_label_str2int, columns=score_columns),
        remove_columns=scores.column_names,
    )

    for row in challenge_ds:
        assert any(row[col] is not None for col in score_columns), f"Missing score for {row}"

    challenge_ds.save_to_disk(challenge)
