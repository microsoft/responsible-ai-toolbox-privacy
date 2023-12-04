import os
import pytest
import numpy as np
from datasets import Dataset, concatenate_datasets
from typing import Dict
from privacy_estimates.experiments.components.prepare_data.prepare_data import prepare_data, SampleSelectionMethod
from subprocess import CalledProcessError

from .test_utils import datasets_as_paths, run_component


class TestPrepareData:
    def run(
        self, in_out_data: Dataset, in_indices: Dataset, out_indices: Dataset, train_base_data: Dataset,
        validation_base_data: Dataset, seed: int, num_points_per_model: int, model_index: int, sample_selection: str,
        merge_unused_samples: str 
    ) -> Dict[str, Dataset]:
        with datasets_as_paths({
            "in_out_data": in_out_data,
            "in_indices": in_indices,
            "out_indices": out_indices,
            "train_base_data": train_base_data,
            "validation_base_data": validation_base_data
        }) as paths:
            component = prepare_data(
                **paths, seed=seed, num_points_per_model=num_points_per_model, model_index=model_index,
                sample_selection=sample_selection, merge_unused_samples=merge_unused_samples
            )
            return run_component(component)

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
        with pytest.raises(CalledProcessError):
            self.run(in_out_data, in_indices, out_indices, train_base_data, validation_base_data, seed=42,
                     num_points_per_model=2, model_index=0, sample_selection="independent", merge_unused_samples="none")

    def generate_data_for_partitioned_selection(self):
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
        return in_out_data, in_indices, out_indices, train_base_data, validation_base_data
 
    def test_partitioned_selection_no_merge(self):
        in_out_data, in_indices, out_indices, train_base_data, validation_base_data = self.generate_data_for_partitioned_selection()
        result = self.run(in_out_data, in_indices, out_indices, train_base_data, validation_base_data, seed=42,
                          num_points_per_model=2, model_index=0, sample_selection="partitioned", merge_unused_samples="none")
        assert len(result["in_data_for_model"]) == 2
        result["in_data_for_model"]["input"][0] == "val_0"
        result["in_data_for_model"]["input"][1] == None
        assert len(result["out_data_for_model"]) == 2
        result["out_data_for_model"]["input"][0] == "trn_0"
        result["out_data_for_model"]["input"][1] == "trn_1"
        assert len(result["train_data_for_model"]) == 2
        result["train_data_for_model"]["input"][0] == "val_0"
        result["train_data_for_model"]["input"][1] == None
        assert len(result["validation_data_for_model"]) == 1

    def test_partitioned_selection_merge_all_with_train(self):
        in_out_data, in_indices, out_indices, train_base_data, validation_base_data = self.generate_data_for_partitioned_selection()
        result = self.run(in_out_data, in_indices, out_indices, train_base_data, validation_base_data, seed=42,
                          num_points_per_model=2, model_index=0, sample_selection="partitioned", merge_unused_samples="all_with_train")
        assert len(result["train_data_for_model"]) == 6
        assert len(result["validation_data_for_model"]) == 1

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
 
    def autogenerated_test_prepare_data(self):
        # Generate the input data
        in_out_data, in_indices, out_indices, train_base_data, validation_base_data = self.generate_data()

        # Define input parameters
        seed = 42
        num_points_per_model = 2
        model_index = 0
        in_data_for_model = "in_data_for_model"
        out_data_for_model = "out_data_for_model"
        train_data_for_model = "train_data_for_model"
        sample_selection = SampleSelectionMethod.INDEPENDENT
        merge_unused_in_samples_with_train_data = True
        merge_unused_out_samples_with_validation_data = True

        # Call the function
        prepare_data(
            in_out_data=in_out_data,
            in_indices=in_indices,
            out_indices=out_indices,
            train_base_data=train_base_data,
            validation_base_data=validation_base_data,
            seed=seed,
            num_points_per_model=num_points_per_model,
            model_index=model_index,
            in_data_for_model=in_data_for_model,
            out_data_for_model=out_data_for_model,
            train_data_for_model=train_data_for_model,
            sample_selection=sample_selection,
            merge_unused_in_samples_with_train_data=merge_unused_in_samples_with_train_data,
            merge_unused_out_samples_with_validation_data=merge_unused_out_samples_with_validation_data
        )

        # Load the output datasets
        in_data_for_model_ds = Dataset.from_disk(in_data_for_model)
        out_data_for_model_ds = Dataset.from_disk(out_data_for_model)
        train_data_for_model_ds = Dataset.from_disk(train_data_for_model)
        validation_data_for_model_ds = Dataset.from_disk(validation_base_data)

        # Check that the datasets have the expected number of samples
        assert len(in_data_for_model_ds) == num_points_per_model
        assert len(out_data_for_model_ds) == num_points_per_model
        assert len(train_data_for_model_ds) == len(train_base_data) + num_points_per_model + 1  # +1 for the null row
        assert len(validation_data_for_model_ds) == len(validation_base_data) + num_points_per_model

        # Check that the in_data_for_model_ds and out_data_for_model_ds contain the expected samples
        in_sample_indices = set(zip(in_data_for_model_ds["split"], in_data_for_model_ds["sample_index"]))
        out_sample_indices = set(zip(out_data_for_model_ds["split"], out_data_for_model_ds["sample_index"]))
        expected_in_sample_indices = {(0, 0), (0, 1)}
        expected_out_sample_indices = {(1, 3), (1, 4)}
        assert in_sample_indices == expected_in_sample_indices
        assert out_sample_indices == expected_out_sample_indices

        # Check that the train_data_for_model_ds contains the expected samples
        train_sample_indices = set(zip(train_data_for_model_ds["split"], train_data_for_model_ds["sample_index"]))
        expected_train_sample_indices = {(0, 0), (0, 1), (1, 2), (1, 4), (1, 5)}
        assert train_sample_indices == expected_train_sample_indices

        # Check that the validation_data_for_model_ds contains the expected samples
        validation_sample_indices = set(zip(validation_data_for_model_ds["split"], validation_data_for_model_ds["sample_index"]))
        expected_validation_sample_indices = {(1, 3), (1, 4), (0, 2)}
        assert validation_sample_indices == expected_validation_sample_indices

        # Clean up the output files
        os.remove(in_data_for_model)
        os.remove(out_data_for_model)
        os.remove(train_data_for_model)