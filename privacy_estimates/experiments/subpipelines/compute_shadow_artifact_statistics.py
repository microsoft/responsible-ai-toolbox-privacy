from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineComponent

from privacy_estimates.experiments.aml import WorkspaceConfig
from privacy_estimates.experiments.subpipelines import TrainManyArtifactsLoader
from privacy_estimates.experiments.components import (
    create_in_out_data_for_shadow_artifact_statistics, compute_shadow_artifact_statistics, convert_uri_file_to_int
)
from privacy_estimates.experiments.loaders import TrainComponentLoader, ScoreComponentLoader


class ComputeShadowArtifactStatisticsLoader:
    def __init__(self, train_loader: TrainComponentLoader, score_loader: ScoreComponentLoader,
                 num_artifacts: int, workspace: WorkspaceConfig, in_fraction: float, num_concurrent_jobs_per_node: int = 1,
                 num_artifacts_per_group: int = 32, num_repetitions: int = 1) -> None:
        self.workspace = workspace
        self.in_fraction = in_fraction
        self.train_many_artifacts_loader = TrainManyArtifactsLoader(
            num_artifacts=num_artifacts, train_loader=train_loader, score_loader=score_loader, sample_selection="partitioned",
            merge_unused_samples="none", num_concurrent_jobs_per_node=num_concurrent_jobs_per_node,
            num_artifacts_per_group=num_artifacts_per_group, tag_artifact_index=True, num_repetitions=num_repetitions
        )

    def load(self, train_data: Input, validation_data: Input, canary_data: Input, seed: int) -> PipelineComponent:
        @dsl.pipeline(name="compute_shadow_artifact_statistics")
        def p(train_data: Input, validation_data: Input, canary_data: Input, seed: int) -> PipelineComponent:
            # create_in_out_data_for_shadow_artifact_statistics requires sample_selection to be "partitioned"
            assert self.train_many_artifacts_loader.single_artifact_arguments.sample_selection == "partitioned"
            data_for_shadow_artifacts = create_in_out_data_for_shadow_artifact_statistics(
                in_out_data=canary_data, seed=seed, split_type="rotating_splits", in_fraction=self.in_fraction
            )
            convert_num_points_per_artifact = convert_uri_file_to_int(uri_file=data_for_shadow_artifacts.outputs.num_points_per_artifact)
            data_for_shadow_artifacts = self.workspace.large_memory_cpu_compute.apply(data_for_shadow_artifacts)
            train_shadow_artifacts = self.train_many_artifacts_loader.load(
                train_base_data=train_data,
                validation_base_data=validation_data,
                in_out_data=canary_data, in_indices=data_for_shadow_artifacts.outputs.in_indices,
                out_indices=data_for_shadow_artifacts.outputs.out_indices, base_seed=seed,
                convert_num_points_per_artifact=convert_num_points_per_artifact.outputs.output
            )

            shadow_artifact_statistics_job = compute_shadow_artifact_statistics(
                predictions_in=train_shadow_artifacts.outputs.predictions_in,
                predictions_out=train_shadow_artifacts.outputs.predictions_out
            )
            shadow_artifact_statistics_job = self.workspace.large_memory_cpu_compute.apply(shadow_artifact_statistics_job)

            return {
                "statistics": shadow_artifact_statistics_job.outputs.statistics,
            }
        return p(train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=seed)
