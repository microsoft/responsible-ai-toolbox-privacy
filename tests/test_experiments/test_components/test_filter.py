import numpy as np
from datasets import Dataset, features

from privacy_estimates.experiments.components import filter_aux_data, reinsert_aux_data
from .test_utils import run_component, datasets_as_paths


def test_filter_aux_data():
    full = Dataset.from_dict(
        mapping={
            "x": [1, None],
            "sample_index": [0, None],
            "split": ["train", None],
        },
        features=features.Features({
            "x": features.Value("int64"),
            "sample_index": features.Value("int64"),
            "split": features.Value("string")
        }),
    )
    with datasets_as_paths({"full": full}) as p:
        filter = filter_aux_data(**p)
        outputs = run_component(filter)
    aux = outputs["aux"]
    assert len(aux) == 2
    assert "sample_index" in aux.features
    assert "split" in aux.features
    assert "is_null" in aux.features

    filtered = outputs["filtered"]
    assert len(filtered) == 1
    assert "sample_index" not in filtered.features
    assert "split" not in filtered.features


def run_test_filter_aux_reinsert_unity(full):
    with datasets_as_paths({"full": full}) as p:
        filter = filter_aux_data(**p)
        outputs = run_component(filter)
    aux = outputs["aux"]
    filtered = outputs["filtered"]

    with datasets_as_paths({"aux": aux, "filtered": filtered}) as p:
        reinsert = reinsert_aux_data(**p)
        outputs = run_component(reinsert)
    full_2 = outputs["full"]

    np.testing.assert_array_equal(full_2["data"], full["data"])
    np.testing.assert_array_equal(sorted(full_2.column_names), sorted(full.column_names))


def test_filter_aux_reinsert_unity_integer():
    full = Dataset.from_dict(
        mapping={
            "data": [1, None],
            "sample_index": [0, None],
            "split": ["train", None],
        },
        features=features.Features({
            "data": features.Value("int64"),
            "sample_index": features.Value("int64"),
            "split": features.Value("string")
        }),
    )
    run_test_filter_aux_reinsert_unity(full)


def test_filter_aux_reinsert_unity_string():
    full = Dataset.from_dict(
        mapping={
            "data": ["1", None],
            "sample_index": [0, None],
            "split": ["train", None],
        },
        features=features.Features({
            "data": features.Value("string"),
            "sample_index": features.Value("int64"),
            "split": features.Value("string")
        }),
    )
    run_test_filter_aux_reinsert_unity(full)


def test_filter_aux_reinsert_unity_array():
    full = Dataset.from_dict(
        mapping={
            "data": [[[1.0, 2.0], [3.0, 4.0]], None],
            "sample_index": [0, None],
            "split": ["train", None],
        },
        features=features.Features({
            "data": features.Array2D(shape=(2, 2), dtype="float32"),
            "sample_index": features.Value("int64"),
            "split": features.Value("string")
        }),
    )
    run_test_filter_aux_reinsert_unity(full)
