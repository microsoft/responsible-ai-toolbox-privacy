import pytest
from datasets import Dataset
from privacy_estimates.experiments.components.prepare_data.prepare_data import _prepare_data


class TestPrepareData:
    def test_raise_on_intersecting_train_out(self):
        in_out_data = Dataset.from_dict({
            "input": ["trn_2", "trn_4", "val_3"],
            "sample_index": [2, 4, 3],
            "split": ["trn", "trn", "val"]
        })
        in_indices = Dataset.from_dict({
            "sample_index": [3, None],
            "split": ["val", None,]
        })
        out_indices = Dataset.from_dict({
            "sample_index": [2, 4],
            "split": ["trn", "trn"]
        })
        train_base_data = Dataset.from_dict({
            "input": ["trn_0", "trn_1", "trn_2"],
            "sample_index": [0, 1, 2],
            "split": ["trn", "trn", "trn"]
        })
        validation_base_data = Dataset.from_dict({
            "input": ["val_0", "val_1", "val_2"],
            "sample_index": [0, 1, 2],
            "split": ["val", "val", "val"]
        })
        with pytest.raises(ValueError):
            _prepare_data(
                in_out_ds=in_out_data, in_sample_indices_ds=in_indices, out_sample_indices_ds=out_indices,
                train_base_ds=train_base_data, validation_base_ds=validation_base_data, seed=42, num_points_per_artifact=2,
                artifact_index=0, sample_selection="independent", merge_unused_samples="none"
            )

    def test_partitioned_selection_no_merge(self):
        in_out_data = Dataset.from_dict({
            "input": ["trn_0", "trn_1", "val_0", "val_1"],
            "sample_index": [0, 1, 0, 1],
            "split": ["trn", "trn", "val", "val"]
        })
        in_indices = Dataset.from_dict({
            "sample_index": [0, None, 0, 1],
            "split": ["val", None, "trn", "trn"]
        })
        out_indices = Dataset.from_dict({
            "sample_index": [0, 1, 1, None],
            "split": ["trn", "trn", "val", None]
        })
        train_base_data = Dataset.from_dict({
            "input": [],
            "sample_index": [],
            "split": []
        })
        validation_base_data = Dataset.from_dict({
            "input": ["val_2"],
            "sample_index": [2],
            "split": ["val"]
        })

        results = _prepare_data(in_out_ds=in_out_data, in_sample_indices_ds=in_indices, out_sample_indices_ds=out_indices,
                                train_base_ds=train_base_data, validation_base_ds=validation_base_data, seed=42,
                                num_points_per_artifact=2, artifact_index=0, sample_selection="partitioned",
                                merge_unused_samples="none")
        assert results.train_data_for_artifact["sample_index"] == [None, 0]
        assert results.train_data_for_artifact["split"] == [None, "val"]
        assert results.in_data_for_artifact["sample_index"] == [0, None]
        assert results.in_data_for_artifact["split"] == ["val", None]
        assert results.out_data_for_artifact["sample_index"] == [0, 1]
        assert results.out_data_for_artifact["split"] == ["trn", "trn"]
        assert results.validation_data_for_artifact["sample_index"] == [2]
        assert results.validation_data_for_artifact["split"] == ["val"]

    def test_partitioned_selection_merge_all_with_train(self):
        in_out_data = Dataset.from_dict({
            "input": ["trn_0", "trn_1", "val_0", "val_1", "val_3"],
            "sample_index": [0, 1, 0, 1, 3],
            "split": ["trn", "trn", "val", "val", "val"]
        })
        in_indices = Dataset.from_dict({
            "sample_index": [0, None, 1, None],
            "split": ["val", None, "val", None]
        })
        out_indices = Dataset.from_dict({
            "sample_index": [0, 1, None, 3],
            "split": ["trn", "trn", None, "val"]
        })
        train_base_data = Dataset.from_dict({
            "input": [],
            "sample_index": [],
            "split": []
        })
        validation_base_data = Dataset.from_dict({
            "input": ["val_2"],
            "sample_index": [2],
            "split": ["val"]
        })

        results = _prepare_data(
            train_base_ds=train_base_data, validation_base_ds=validation_base_data, in_out_ds=in_out_data,
            in_sample_indices_ds=in_indices, out_sample_indices_ds=out_indices, seed=42, num_points_per_artifact=2,
            artifact_index=0, sample_selection="partitioned", merge_unused_samples="all_with_train"
        )
        assert results.train_data_for_artifact["sample_index"] == [None, 1, 3, None, None, 0]
        assert results.train_data_for_artifact["split"] == [None, "val", "val", None, None, "val"]
        assert results.validation_data_for_artifact["sample_index"] == [2]
        assert results.validation_data_for_artifact["split"] == ["val"]

    def test_partitioned_selection_merge_all_with_train_raise_duplicate_train(self):
        in_out_data = Dataset.from_dict({
            "input": ["trn_0", "trn_1", "val_0", "val_1", "val_3"],
            "sample_index": [0, 1, 0, 1, 3],
            "split": ["trn", "trn", "val", "val", "val"]
        })
        in_indices = Dataset.from_dict({
            "sample_index": [0, None, 1, None],
            "split": ["val", None, "val", None]
        })
        out_indices = Dataset.from_dict({
            "sample_index": [0, 1, None, 0],
            "split": ["trn", "trn", None, "val"]
        })
        train_base_data = Dataset.from_dict({
            "input": [],
            "sample_index": [],
            "split": []
        })
        validation_base_data = Dataset.from_dict({
            "input": ["val_2"],
            "sample_index": [2],
            "split": ["val"]
        })

        with pytest.raises(ValueError):
            _prepare_data(
                train_base_ds=train_base_data, validation_base_ds=validation_base_data, in_out_ds=in_out_data,
                in_sample_indices_ds=in_indices, out_sample_indices_ds=out_indices, seed=42, num_points_per_artifact=2,
                artifact_index=0, sample_selection="partitioned", merge_unused_samples="all_with_train"
            )

    def generate_data_unequal_len_in_out(self):
        in_out_data = Dataset.from_dict({
            "input": ["trn_2", "trn_4", "val_3"],
            "sample_index": [2, 4, 3],
            "split": ["trn", "trn", "val"]
        })
        in_indices = Dataset.from_dict({
            "sample_index": [3, None],
            "split": ["val", None,]
        })
        out_indices = Dataset.from_dict({
            "sample_index": [2, 4, 5],
            "split": ["trn", "trn", "trn"]
        })
        train_base_data = Dataset.from_dict({
            "input": ["trn_0", "trn_1", "trn_2"],
            "sample_index": [0, 1, 2],
            "split": ["trn", "trn", "trn"]
        })
        validation_base_data = Dataset.from_dict({
            "input": ["val_0", "val_1", "val_2"],
            "sample_index": [0, 1, 2],
            "split": ["val", "val", "val"]
        })
        return in_out_data, in_indices, out_indices, train_base_data, validation_base_data
