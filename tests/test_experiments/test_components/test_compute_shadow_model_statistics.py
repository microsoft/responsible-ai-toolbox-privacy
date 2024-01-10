import numpy as np
import pytest

from datasets import Dataset
from privacy_estimates.experiments.components.compute_shadow_model_statistics.compute_shadow_model_statistics import _compute_shadow_model_statistics


@pytest.fixture()
def predictions():
    in_predictions = Dataset.from_dict({
        "loss": [0.1, 0.4, 0.5],
        "sample_index": [0, 1, 1],
        "split": ["train", "train", "train"],
        "model_index": [0, 0, 1],
        "logits": [np.array([0.2, 0.3]), np.array([0.4, 0.5]), np.array([0.6, 0.7])],
        "label": [0, 1, 1]
    })
    out_predictions = Dataset.from_dict({
        "loss": [0.2, 0.3],
        "sample_index": [0, 1],
        "split": ["train", "train"],
        "model_index": [0, 1],
        "label": [0, 1],
        "logits": [np.array([0.8, 0.9]), np.array([1.0, 1.1])]
    })
    yield in_predictions, out_predictions


def test_find_gap_mean(predictions):
    in_predictions, out_predictions = predictions
    loss_stat = _compute_shadow_model_statistics(in_predictions, out_predictions)
    assert loss_stat["loss_list_in"][0] == [0.1]
    assert loss_stat["loss_list_in"][1] == [0.4, 0.5]
    assert loss_stat["loss_list_out"][0] == [0.2]
    assert loss_stat["loss_list_out"][1] == [0.3]
