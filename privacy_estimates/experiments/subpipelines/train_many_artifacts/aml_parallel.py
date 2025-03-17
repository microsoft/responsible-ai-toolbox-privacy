from azure.ai.ml import dsl, Input
from privacy_estimates.experiments.loaders import TrainSingleArtifactAndScoreArguments
from privacy_estimates.experiments.components import (
    prepare_data_for_aml_parallel, filter_aux_data_aml_parallel, create_artifact_indices_for_aml_parallel,
    reinsert_aux_data_aml_parallel, collect_from_aml_parallel, append_artifact_index_column_aml_parallel
)
from privacy_estimates.experiments.subpipelines.aml_parallel import AMLParallelLoader
from .base import TrainArtifactGroupBase


class TrainArtifactGroupAMLParallelLoader(TrainArtifactGroupBase):
    def __init__(self, num_artifacts: int, group_size: int, num_concurrent_jobs_per_node: int,
                 single_artifact_arguments: TrainSingleArtifactAndScoreArguments):
        super().__init__(
            num_artifacts=num_artifacts, group_size=group_size, single_artifact_arguments=single_artifact_arguments
        )
        self.num_concurrent_jobs_per_node = num_concurrent_jobs_per_node

        # Add optional outputs to the parallel loaders
        train_component_outputs = self.single_artifact_arguments.train_loader.component.outputs.keys()
        train_outputs_file = sorted(o for o in {"metrics", "dp_parameters", "dpd_data"}.intersection(train_component_outputs))

        self.train_artifact_parallel_loader = AMLParallelLoader(
            component_loader=self.single_artifact_arguments.train_loader,
            num_concurrent_jobs_per_node=self.num_concurrent_jobs_per_node,
            outputs_folder=["artifact"], outputs_file=train_outputs_file,
            inputs_to_distribute=["seed", "train_data", "validation_data"], inputs_to_distribute_on_command_line=["seed"]
        )
        self.score_with_artifact_parallel_loader = AMLParallelLoader(
            component_loader=self.single_artifact_arguments.score_loader,
            num_concurrent_jobs_per_node=self.num_concurrent_jobs_per_node,
            outputs_folder=["predictions"], outputs_file=[], inputs_to_distribute=["dataset", "model"]
        )

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, artifact_group_index: int, num_points_per_artifact: int):

        @dsl.pipeline(name="train_artifact_group_aml_parallel")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, artifact_group_index: int, num_points_per_artifact: int):
            artifact_index_start = artifact_group_index * self.group_size
            artifact_index_end = artifact_index_start + self.num_artifacts

            artifact_indices = create_artifact_indices_for_aml_parallel(
                artifact_index_start=artifact_index_start, artifact_index_end=artifact_index_end
            ).outputs.artifact_indices

            prepare_data = prepare_data_for_aml_parallel(
                train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                in_indices=in_indices, out_indices=out_indices, artifact_index_start=artifact_index_start,
                artifact_index_end=artifact_index_end, group_base_seed=base_seed,
                num_points_per_artifact=num_points_per_artifact,
                sample_selection=self.single_artifact_arguments.sample_selection,
                merge_unused_samples=self.single_artifact_arguments.merge_unused_samples,
                num_repetitions=self.single_artifact_arguments.num_repetitions,
            )

            filter_train_data = filter_aux_data_aml_parallel(full=prepare_data.outputs.train_data_for_artifacts)

            train_artifacts_parallel = self.train_artifacts_parallel_loader.load(
                train_data=filter_train_data.outputs.filtered,
                validation_data=prepare_data.outputs.validation_data_for_artifacts,
                seed=prepare_data.outputs.seeds_for_artifacts, artifact_indices=artifact_indices,
            )

            filter_in_samples = filter_aux_data_aml_parallel(full=prepare_data.outputs.in_data_for_artifacts)
            filter_out_samples = filter_aux_data_aml_parallel(full=prepare_data.outputs.out_data_for_artifacts)

            compute_scores_in_parallel = self.score_with_artifacts_parallel_loader.load(
                dataset=filter_in_samples.outputs.filtered, artifact_indices=artifact_indices,
                artifact=train_artifacts_parallel.outputs.model
            )
            compute_scores_out_parallel = self.score_with_artifacts_parallel_loader.load(
                dataset=filter_out_samples.outputs.filtered, artifact_indices=artifact_indices,
                artifact=train_artifacts_parallel.outputs.model
            )

            reinsert_in_scores = reinsert_aux_data_aml_parallel(
                filtered=compute_scores_in_parallel.outputs.predictions, aux=filter_in_samples.outputs.aux
            )
            reinsert_out_scores = reinsert_aux_data_aml_parallel(
                filtered=compute_scores_out_parallel.outputs.predictions, aux=filter_out_samples.outputs.aux
            )

            scores_in = reinsert_in_scores.outputs.full
            scores_out = reinsert_out_scores.outputs.full

            if self.single_artifact_arguments.tag_artifact_index:
                scores_in = append_artifact_index_column_aml_parallel(data=scores_in).outputs.output
                scores_out = append_artifact_index_column_aml_parallel(data=scores_out).outputs.output

            output = {
                "scores_in": collect_from_aml_parallel(data=scores_in, aggregator="concatenate_datasets"),
                "scores_out": collect_from_aml_parallel(data=scores_out, aggregator="concatenate_datasets")
            }

            if "metrics" in train_artifacts_parallel.outputs:
                output["metrics_avg"] = collect_from_aml_parallel(data=train_artifacts_parallel.outputs.metrics,
                                                                  aggregator="average_json")
            if "dp_parameters" in train_artifacts_parallel.outputs:
                output["dp_parameters"] = collect_from_aml_parallel(data=train_artifacts_parallel.outputs.dp_parameters,
                                                                    aggregator="average_json")

            return output

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, artifact_group_index=artifact_group_index,
                 num_points_per_artifact=num_points_per_artifact)
