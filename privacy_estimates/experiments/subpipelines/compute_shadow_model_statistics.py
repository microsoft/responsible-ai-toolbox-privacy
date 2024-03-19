from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineComponent

from privacy_estimates.experiments.aml import WorkspaceConfig
from privacy_estimates.experiments.subpipelines import TrainManyModelsLoader
from privacy_estimates.experiments.components import (
    create_in_out_data_for_shadow_model_statistics, compute_shadow_model_statistics, create_empty_dataset
)
from privacy_estimates.experiments.loaders import TrainingComponentLoader, InferenceComponentLoader


class ComputeShadowModelStatisticsLoader:
    def __init__(self, train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader,
                 num_models: int, workspace: WorkspaceConfig, num_concurrent_jobs_per_node: int = 1,
                 num_models_per_group: int = 32):
        self.workspace = workspace
        self.train_many_models_loader = TrainManyModelsLoader(
            num_models=num_models, train_loader=train_loader, inference_loader=inference_loader, sample_selection="partitioned",
            merge_unused_samples="none", num_concurrent_jobs_per_node=num_concurrent_jobs_per_node,
            num_models_per_group=num_models_per_group, tag_model_index=True
        )

    def load(self, train_data: Input, validation_data: Input, seed: int) -> PipelineComponent:
        @dsl.pipeline(name="compute_shadow_model_statistics")
        def p(train_data: Input, validation_data: Input, seed: int) -> PipelineComponent:
            # create_in_out_data_for_shadow_model_statistics requires sample_selection to be "partitioned"
            assert self.train_many_models_loader.single_model_arguments.sample_selection == "partitioned"
            data_for_shadow_models = create_in_out_data_for_shadow_model_statistics(
                in_out_data=train_data, seed=seed, split_type="rotating_splits", in_fraction=0.75
            )
            data_for_shadow_models.compute = self.workspace.large_memory_cpu_compute
            empty_dataset = create_empty_dataset(dataset_for_features=train_data).outputs.dataset

            train_shadow_models = self.train_many_models_loader.load(
                train_base_data=validation_data,
                validation_base_data=empty_dataset,
                in_out_data=train_data, in_indices=data_for_shadow_models.outputs.in_indices,
                out_indices=data_for_shadow_models.outputs.out_indices, base_seed=seed,
                num_points_per_model=data_for_shadow_models.outputs.num_points_per_model
            )

            shadow_model_statistics_job = compute_shadow_model_statistics(
                predictions_in=train_shadow_models.outputs.predictions_in,
                predictions_out=train_shadow_models.outputs.predictions_out
            )
            shadow_model_statistics_job.compute = self.workspace.large_memory_cpu_compute

            return {
                "statistics": shadow_model_statistics_job.outputs.statistics,
            }
        return p(train_data=train_data, validation_data=validation_data, seed=seed)
