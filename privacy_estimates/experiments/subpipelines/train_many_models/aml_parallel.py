from azure.ai.ml import dsl, Input
from privacy_estimates.experiments.loaders import TrainSingleModelAndPredictArguments
from privacy_estimates.experiments.components import (
	prepare_data_for_aml_parallel, filter_aux_data_aml_parallel, create_model_indices_for_aml_parallel,
    reinsert_aux_data_aml_parallel, collect_from_aml_parallel, append_model_index_column_aml_parallel
)
from privacy_estimates.experiments.subpipelines.aml_parallel import AMLParallelLoader
from .base import TrainModelGroupBase


class TrainModelGroupAMLParallelLoader(TrainModelGroupBase):
    def __init__(self, num_models: int, group_size: int, num_concurrent_jobs_per_node: int,
                 single_model_arguments: TrainSingleModelAndPredictArguments):
        super().__init__(num_models=num_models, group_size=group_size, single_model_arguments=single_model_arguments)
        self.num_concurrent_jobs_per_node = num_concurrent_jobs_per_node

        # Add optional outputs to the parallel loaders
        train_component_outputs = self.single_model_arguments.train_loader.component.outputs.keys()
        train_outputs_file = sorted(o for o in {"metrics", "dp_parameters", "dpd_data"}.intersection(train_component_outputs))

        self.train_models_parallel_loader = AMLParallelLoader(
            component_loader=self.single_model_arguments.train_loader,
            num_concurrent_jobs_per_node=self.num_concurrent_jobs_per_node,
            outputs_folder=["model"], outputs_file=train_outputs_file,
            inputs_to_distribute=["seed", "train_data", "validation_data"], inputs_to_distribute_on_command_line=["seed"]
        )
        self.predict_with_models_parallel_loader = AMLParallelLoader(
            component_loader=self.single_model_arguments.inference_loader,
            num_concurrent_jobs_per_node=self.num_concurrent_jobs_per_node,
            outputs_folder=["predictions"], outputs_file=[], inputs_to_distribute=["dataset", "model"]
        )

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, model_group_index: int, num_points_per_model: int):
        
        @dsl.pipeline(name="train_model_group_aml_parallel")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, model_group_index: int, num_points_per_model: int):
            model_index_start = model_group_index * self.group_size
            model_index_end = model_index_start + self.num_models

            model_indices = create_model_indices_for_aml_parallel(
                model_index_start=model_index_start, model_index_end=model_index_end
            ).outputs.model_indices

            prepare_data = prepare_data_for_aml_parallel(
                train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data, in_indices=in_indices, out_indices=out_indices,
                model_index_start=model_index_start, model_index_end=model_index_end, group_base_seed=base_seed, num_points_per_model=num_points_per_model,
                sample_selection=self.single_model_arguments.sample_selection, merge_unused_samples=self.single_model_arguments.merge_unused_samples,
            )

            filter_train_data = filter_aux_data_aml_parallel(full=prepare_data.outputs.train_data_for_models)

            train_models_parallel = self.train_models_parallel_loader.load(
                train_data=filter_train_data.outputs.filtered,
                validation_data=prepare_data.outputs.validation_data_for_models, seed=prepare_data.outputs.seeds_for_models,
                model_indices=model_indices,
            )

            filter_in_samples = filter_aux_data_aml_parallel(full=prepare_data.outputs.in_data_for_models)
            filter_out_samples = filter_aux_data_aml_parallel(full=prepare_data.outputs.out_data_for_models)
                

            compute_predictions_in_parallel = self.predict_with_models_parallel_loader.load(
                dataset=filter_in_samples.outputs.filtered, model_indices=model_indices, model=train_models_parallel.outputs.model
            )
            compute_predictions_out_parallel = self.predict_with_models_parallel_loader.load(
                dataset=filter_out_samples.outputs.filtered, model_indices=model_indices, model=train_models_parallel.outputs.model
            )

            reinsert_in_predictions = reinsert_aux_data_aml_parallel(
                filtered=compute_predictions_in_parallel.outputs.predictions, aux=filter_in_samples.outputs.aux
            )
            reinsert_out_predictions = reinsert_aux_data_aml_parallel(
                filtered=compute_predictions_out_parallel.outputs.predictions, aux=filter_out_samples.outputs.aux
            )

            pred_in = reinsert_in_predictions.outputs.full
            pred_out = reinsert_out_predictions.outputs.full

            if self.single_model_arguments.tag_model_index:
                pred_in = append_model_index_column_aml_parallel(data=pred_in).outputs.output
                pred_out = append_model_index_column_aml_parallel(data=pred_out).outputs.output

            output = {
                "predictions_in": collect_from_aml_parallel(data=pred_in, aggregator="concatenate_datasets"),
                "predictions_out": collect_from_aml_parallel(data=pred_out, aggregator="concatenate_datasets")
            }

            if "metrics" in train_models_parallel.outputs:
                output["metrics_avg"] = collect_from_aml_parallel(data=train_models_parallel.outputs.metrics, aggregator="average_json")
            if "dp_parameters" in train_models_parallel.outputs:
                output["dp_parameters"] = collect_from_aml_parallel(data=train_models_parallel.outputs.dp_parameters, aggregator="average_json")

            return output

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, model_group_index=model_group_index,
                 num_points_per_model=num_points_per_model)
    



