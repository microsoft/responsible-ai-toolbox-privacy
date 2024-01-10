from azure.ai.ml.entities import Component
from azure.ai.ml import Input, dsl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from privacy_estimates.experiments.aml import AMLComponentLoader
from privacy_estimates.experiments.components import (
    prepare_data, filter_aux_data, reinsert_aux_data, append_column_constant
)


class ComponentLoader(ABC):
    def __init__(self, aml_component_loader: Optional[AMLComponentLoader] = None) -> None:
        self.aml_loader = aml_component_loader

    @property
    def component(self):
        raise NotImplementedError(f"`component` needs to be implemented for {self.__class__.__name__}")

    @property 
    def compute(self) -> Optional[str]:
        raise NotImplementedError(f"`compute` needs to be implemented for {self.__class__.__name__}")

    @property 
    def parameter_dict(self) -> Dict:
        return dict()

    def load(self, *args, **kwargs):
        job = self.component(*args, **kwargs, **self.parameter_dict)
        if self.compute is not None:
            job.compute = self.compute
        return job


class SingleGameLoader(ComponentLoader):
    @abstractmethod
    def load(self, train_data, validation_data, seed: int) -> Component:
        """
        Should return a component with an output named `scores`, challenge_bits` and optionally `dp_parameters`.
        """
        pass


class TrainingComponentLoader(ComponentLoader):
    pass


class InferenceComponentLoader(ComponentLoader):
    pass


@dataclass
class TrainSingleModelAndPredictArguments:
    train_loader: TrainingComponentLoader
    inference_loader: InferenceComponentLoader
    sample_selection: str
    tag_model_index: bool
    merge_unused_samples: str


class TrainSingleModelAndPredictLoader:
    def __init__(self, arguments: TrainSingleModelAndPredictArguments):
        """
        This component loader preprocesses data for training a single model and then computes predictions.
        This is the inner loop when training many models in parallel as required for shadow models or
        empirical privacy estimates.
        """
        self.arguments = arguments

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, model_index: int, num_points_per_model: int):
        @dsl.pipeline(name="train_model_and_predict")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, model_index: int, num_points_per_model: int):
            seed = base_seed + model_index

            data_for_model = prepare_data(
                train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                in_indices=in_indices, out_indices=out_indices, seed=seed, num_points_per_model=num_points_per_model,
                model_index=model_index, sample_selection=self.arguments.sample_selection,
                merge_unused_samples=self.arguments.merge_unused_samples
            )

            filter_train_data = filter_aux_data(full=data_for_model.outputs.train_data_for_model)
            filter_validation_data = filter_aux_data(full=data_for_model.outputs.validation_data_for_model)
            filter_in_data = filter_aux_data(full=data_for_model.outputs.in_data_for_model)
            filter_out_data = filter_aux_data(full=data_for_model.outputs.out_data_for_model)

            train = self.arguments.train_loader.load(
                train_data=filter_train_data.outputs.filtered, validation_data=filter_validation_data.outputs.filtered,
                seed=seed
            )

            inference_in = self.arguments.inference_loader.load(
                dataset=filter_in_data.outputs.filtered, model=train.outputs.model
            )
            inference_out = self.arguments.inference_loader.load(
                dataset=filter_out_data.outputs.filtered, model=train.outputs.model
            )

            reinsert_null_rows_in = reinsert_aux_data(
                filtered=inference_in.outputs.predictions, aux=filter_in_data.outputs.aux
            )
            reinsert_null_rows_out = reinsert_aux_data(
                filtered=inference_out.outputs.predictions, aux=filter_out_data.outputs.aux
            )

            predictions_in_i = reinsert_null_rows_in.outputs.full
            predictions_out_i = reinsert_null_rows_out.outputs.full

            if self.arguments.tag_model_index:
                predictions_in_i = append_column_constant(
                    data=predictions_in_i, name="model_index", value=model_index
                ).outputs.output
                predictions_out_i = append_column_constant(
                    data=predictions_out_i, name="model_index", value=model_index
                ).outputs.output

            results = {
                "predictions_in": predictions_in_i,
                "predictions_out": predictions_out_i
            }
            if "dp_parameters" in train.outputs:
                results["dp_parameters"] = train.outputs.dp_parameters
            if "metrics" in train.outputs:
                results["metrics"] = train.outputs.metrics
            return results

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, model_index=model_index,
                 num_points_per_model=num_points_per_model)

