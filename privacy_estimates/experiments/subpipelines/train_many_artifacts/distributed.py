from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import TrainSingleArtifactAndScoreLoader
from privacy_estimates.experiments.components import aggregate_output, append_column_constant_int
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader
from .base import TrainArtifactGroupBase, TrainSingleArtifactAndScoreArguments


class TrainArtifactGroupDistributedLoader(TrainArtifactGroupBase):
    def __init__(self, num_artifacts: int, group_size: int, single_artifact_arguments: TrainSingleArtifactAndScoreArguments):
        super().__init__(num_artifacts=num_artifacts, group_size=group_size, single_artifact_arguments=single_artifact_arguments)

        self.train_artifact_and_score_loader = TrainSingleArtifactAndScoreLoader(arguments=single_artifact_arguments)


    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, artifact_group_index: int, num_points_per_artifact: int):
        @dsl.pipeline(name="train_artifact_group_distributed")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, artifact_group_index: int, num_points_per_artifact: int):
            scores_in = []
            scores_out = []
            metrics = []
            dp_parameters = []
            for i in range(0, self.num_artifacts):
                artifact_index = artifact_group_index * self.group_size + i
                train_artifact_and_score = self.train_artifact_and_score_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, artifact_index=artifact_index,
                    num_points_per_artifact=num_points_per_artifact
                )

                scores_in_i = train_artifact_and_score.outputs.scores_in
                scores_out_i = train_artifact_and_score.outputs.scores_out

                scores_in.append(scores_in_i)
                scores_out.append(scores_out_i)
                if hasattr(train_artifact_and_score.outputs, "metrics"):
                    metrics.append(train_artifact_and_score.outputs.metrics)
                if hasattr(train_artifact_and_score.outputs, "dp_parameters"):
                    dp_parameters.append(train_artifact_and_score.outputs.dp_parameters)

            load_from_function = PrivacyEstimatesComponentLoader().load_from_function

            output = {
                "scores_in": aggregate_output(
                    scores_in, aggregator="concatenate_datasets", load_component=load_from_function
                ),
                "scores_out": aggregate_output(
                    scores_out, aggregator="concatenate_datasets", load_component=load_from_function
                ),
            }

            if len(metrics) > 0:
                output["metrics_avg"] = aggregate_output(metrics, aggregator="average_json", load_component=load_from_function)
            if len(dp_parameters) > 0:
                output["dp_parameters"] = aggregate_output(
                    dp_parameters, aggregator="assert_json_equal", load_component=load_from_function
                )
            return output

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, artifact_group_index=artifact_group_index,
                 num_points_per_artifact=num_points_per_artifact)


