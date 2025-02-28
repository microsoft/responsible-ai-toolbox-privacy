from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import TrainComponentLoader, ScoreComponentLoader
from privacy_estimates.experiments.components import aggregate_output
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader
from .aml_parallel import TrainArtifactGroupAMLParallelLoader
from .distributed import TrainArtifactGroupDistributedLoader
from .base import TrainSingleArtifactAndScoreArguments


class TrainManyArtifactsLoader:
    def __init__(self, train_loader: TrainComponentLoader, score_loader: ScoreComponentLoader, *,
                 num_artifacts: int, sample_selection: str, merge_unused_samples: str, num_repetitions: int = 1,
                 num_artifacts_per_group: int = 32, num_concurrent_jobs_per_node: int = 1, tag_artifact_index: bool = False):
        """
        Args:
            train_loader: ComponentLoader for training a single artifact
            inference_loader: ComponentLoader for inference on a single artifact
            num_artifacts: Number of artifacts to train
            artifacts_per_group: Group size of artifacts that are grouped together. This parameter does not have any effect on 
                              the artifacts trained, it simply creates nested pipelines to reduce the rendering of the 
                              pipeline in the studio or groups artifacts training on the same node.
            group_on_single_node: If True, the artifacts_per_group are trained on the same node. If False, a new node is
                                  allocated for each artifact.
        """
        self.num_artifacts = num_artifacts
        self.artifacts_per_group = num_artifacts_per_group

        self.single_artifact_arguments = TrainSingleArtifactAndScoreArguments(
            train_loader=train_loader, score_loader=score_loader, sample_selection=sample_selection,
            tag_artifact_index=tag_artifact_index, merge_unused_samples=merge_unused_samples, num_repetitions=num_repetitions,
        )

        if num_concurrent_jobs_per_node > 1:
            self.train_artifact_group_loader = TrainArtifactGroupAMLParallelLoader(
                num_artifacts=self.artifacts_per_group, group_size=self.artifacts_per_group,
                single_artifact_arguments=self.single_artifact_arguments,
                num_concurrent_jobs_per_node=num_concurrent_jobs_per_node
            )
            self.train_final_artifact_group_loader = TrainArtifactGroupAMLParallelLoader(
                num_artifacts=self.num_artifacts % self.artifacts_per_group, group_size=self.artifacts_per_group,
                single_artifact_arguments=self.single_artifact_arguments,
                num_concurrent_jobs_per_node=num_concurrent_jobs_per_node
            )
        else:
            self.train_artifact_group_loader = TrainArtifactGroupDistributedLoader(
                num_artifacts=self.artifacts_per_group, group_size=self.artifacts_per_group,
                single_artifact_arguments=self.single_artifact_arguments
            )
            self.train_final_artifact_group_loader = TrainArtifactGroupDistributedLoader(
                num_artifacts=self.num_artifacts % self.artifacts_per_group, group_size=self.artifacts_per_group,
                single_artifact_arguments=self.single_artifact_arguments
            )

    def load(self, train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input,
             out_indices: Input, base_seed: int, num_points_per_artifact: Input):
        @dsl.pipeline(name=f"train_{self.num_artifacts}_artifacts")
        def pipeline(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input,
                     out_indices: Input, base_seed: int, num_points_per_artifact: Input):
            scores_in = []
            scores_out = []
            metrics_avg = []
            dp_parameters = []
            num_groups = self.num_artifacts // self.artifacts_per_group
            for artifact_group in range(0, num_groups):
                train_artifact_group = self.train_artifact_group_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed,
                    artifact_group_index=artifact_group, num_points_per_artifact=num_points_per_artifact
                )
                scores_in.append(train_artifact_group.outputs.predictions_in)
                scores_out.append(train_artifact_group.outputs.predictions_out)
                if "metrics_avg" in train_artifact_group.outputs:
                    metrics_avg.append(train_artifact_group.outputs.metrics_avg)
                if "dp_parameters" in train_artifact_group.outputs:
                    dp_parameters.append(train_artifact_group.outputs.dp_parameters)
            if self.num_artifacts % self.artifacts_per_group != 0:
                train_final_artifact_group = self.train_final_artifact_group_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, artifact_group_index=num_groups,
                    num_points_per_artifact=num_points_per_artifact
                )
                scores_in.append(train_final_artifact_group.outputs.scores_in)
                scores_out.append(train_final_artifact_group.outputs.scores_out)
                if "metrics_avg" in train_final_artifact_group.outputs:
                    metrics_avg.append(train_final_artifact_group.outputs.metrics_avg)
                if "dp_parameters" in train_final_artifact_group.outputs:
                    dp_parameters.append(train_final_artifact_group.outputs.dp_parameters)

            load_from_function = PrivacyEstimatesComponentLoader().load_from_function

            outputs =  {
                "scores_in": aggregate_output(scores_in, aggregator="concatenate_datasets", load_component=load_from_function),
                "scores_out": aggregate_output(
                    scores_out, aggregator="concatenate_datasets", load_component=load_from_function
                ),
            }
            if len(metrics_avg) > 0:
                outputs["metrics_avg"] = aggregate_output(
                    metrics_avg, aggregator="average_json", load_component=load_from_function
                )
            if len(dp_parameters) > 0:
                outputs["dp_parameters"] = aggregate_output(
                    dp_parameters, aggregator="assert_json_equal", load_component=load_from_function
                )
            return outputs
        return pipeline(train_base_data=train_base_data, in_out_data=in_out_data, in_indices=in_indices,
                        out_indices=out_indices, validation_base_data=validation_base_data, base_seed=base_seed,
                        num_points_per_artifact=num_points_per_artifact)
