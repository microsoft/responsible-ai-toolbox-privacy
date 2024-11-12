from azure.ai.ml.entities import Component
from azure.ai.ml import Input, dsl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from privacy_estimates.experiments.aml import AMLComponentLoader
from privacy_estimates.experiments.components import (
    prepare_data, filter_aux_data, reinsert_aux_data, append_column_constant_int
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


class TrainComponentLoader(ComponentLoader):
    pass


class ScoreComponentLoader(ComponentLoader):
    pass


@dataclass
class TrainSingleArtifactAndScoreArguments:
    train_loader: TrainComponentLoader
    score_loader: ScoreComponentLoader
    sample_selection: str
    tag_artifact_index: bool
    merge_unused_samples: str
    num_repetitions: int


class TrainSingleArtifactAndScoreLoader:
    def __init__(self, arguments: TrainSingleArtifactAndScoreArguments):
        """
        This component loader preprocesses data for training a single artifact and then computes predictions.
        This is the inner loop when training many artifacts in parallel as required for shadow models or
        empirical privacy estimates.
        """
        self.arguments = arguments

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, artifact_index: int, num_points_per_artifact: int):
        @dsl.pipeline(name="train_artifact_and_predict")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, artifact_index: int, num_points_per_artifact: int):
            seed = base_seed + artifact_index

            data_for_artifact = prepare_data(
                train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                in_indices=in_indices, out_indices=out_indices, seed=seed, num_points_per_artifact=num_points_per_artifact,
                artifact_index=artifact_index, sample_selection=self.arguments.sample_selection,
                merge_unused_samples=self.arguments.merge_unused_samples, num_repetitions=self.arguments.num_repetitions
            )

            filter_train_data = filter_aux_data(full=data_for_artifact.outputs.train_data_for_artifact)
            filter_validation_data = filter_aux_data(full=data_for_artifact.outputs.validation_data_for_artifact)
            filter_in_data = filter_aux_data(full=data_for_artifact.outputs.in_data_for_artifact)
            filter_out_data = filter_aux_data(full=data_for_artifact.outputs.out_data_for_artifact)

            train = self.arguments.train_loader.load(
                train_data=filter_train_data.outputs.filtered, validation_data=filter_validation_data.outputs.filtered,
                seed=seed
            )

            score_in = self.arguments.score_loader.load(
                dataset=filter_in_data.outputs.filtered, artifact=train.outputs.artifact
            )
            score_out = self.arguments.score_loader.load(
                dataset=filter_out_data.outputs.filtered, artifact=train.outputs.artifact
            )

            reinsert_null_rows_in = reinsert_aux_data(
                filtered=score_in.outputs.scores, aux=filter_in_data.outputs.aux
            )
            reinsert_null_rows_out = reinsert_aux_data(
                filtered=score_out.outputs.scores, aux=filter_out_data.outputs.aux
            )

            scores_in_i = reinsert_null_rows_in.outputs.full
            scores_out_i = reinsert_null_rows_out.outputs.full

            if self.arguments.tag_artifact_index:
                scores_in_i = append_column_constant_int(
                    data=scores_in_i, name="artifact_index", value=artifact_index
                ).outputs.output
                scores_out_i = append_column_constant_int(
                    data=scores_out_i, name="artifact_index", value=artifact_index
                ).outputs.output

            results = {
                "scores_in": scores_in_i,
                "scores_out": scores_out_i
            }
            if "dp_parameters" in train.outputs:
                results["dp_parameters"] = train.outputs.dp_parameters
            if "metrics" in train.outputs:
                results["metrics"] = train.outputs.metrics
            return results

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, artifact_index=artifact_index,
                 num_points_per_artifact=num_points_per_artifact)

