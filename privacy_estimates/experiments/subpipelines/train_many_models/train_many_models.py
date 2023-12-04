from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import TrainingComponentLoader, InferenceComponentLoader
from privacy_estimates.experiments.components import aggregate_output
from .aml_parallel import TrainModelGroupAMLParallelLoader
from .distributed import TrainModelGroupDistributedLoader
from .base import TrainSingleModelAndPredictArguments


class TrainManyModelsLoader:
    def __init__(self, train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader,
                 num_models: int, sample_selection: str, merge_unused_samples: str, num_models_per_group: int = 32,  
                 num_concurrent_jobs_per_node: int = 1, tag_model_index: bool = False):
        """
        Args:
            train_loader: ComponentLoader for training a single model
            inference_loader: ComponentLoader for inference on a single model
            num_models: Number of models to train
            models_per_group: Group size of models that are grouped together. This parameter does not have any effect on 
                              the models trained, it simply creates nested pipelines to reduce the rendering of the 
                              pipeline in the studio or groups model training on the same node.
            group_on_single_node: If True, the models_per_group are trained on the same node. If False, a new node is allocated
                                    for each model.
        """
        self.num_models = num_models
        self.models_per_group = num_models_per_group

        self.single_model_arguments = TrainSingleModelAndPredictArguments(
            train_loader=train_loader, inference_loader=inference_loader, sample_selection=sample_selection,
            tag_model_index=tag_model_index, merge_unused_samples=merge_unused_samples
        )

        if num_concurrent_jobs_per_node > 1:
            self.train_model_group_loader = TrainModelGroupAMLParallelLoader(
                num_models=self.models_per_group, group_size=self.models_per_group,
                single_model_arguments=self.single_model_arguments, num_concurrent_jobs_per_node=num_concurrent_jobs_per_node
            )
            self.train_final_model_group_loader = TrainModelGroupAMLParallelLoader(
                num_models=self.num_models % self.models_per_group, group_size=self.models_per_group,
                single_model_arguments=self.single_model_arguments, num_concurrent_jobs_per_node=num_concurrent_jobs_per_node
            )
        else:
            self.train_model_group_loader = TrainModelGroupDistributedLoader(
                num_models=self.models_per_group, group_size=self.models_per_group,
                single_model_arguments=self.single_model_arguments
            )
            self.train_final_model_group_loader = TrainModelGroupDistributedLoader(
                num_models=self.num_models % self.models_per_group, group_size=self.models_per_group,
                single_model_arguments=self.single_model_arguments
            )


    def load(self, train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, num_points_per_model: int):
        @dsl.pipeline(name=f"train_{self.num_models}_models")
        def pipeline(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
                     base_seed: int, num_points_per_model: int):
            predictions_in = []
            predictions_out = []
            metrics_avg = []
            dp_parameters = []
            num_groups = self.num_models // self.models_per_group
            for model_group in range(0, num_groups):
                train_model_group = self.train_model_group_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed,
                    model_group_index=model_group, num_points_per_model=num_points_per_model
                )
                predictions_in.append(train_model_group.outputs.predictions_in)
                predictions_out.append(train_model_group.outputs.predictions_out)
                if "metrics_avg" in train_model_group.outputs:
                    metrics_avg.append(train_model_group.outputs.metrics_avg)
                if "dp_parameters" in train_model_group.outputs:
                    dp_parameters.append(train_model_group.outputs.dp_parameters)
            if self.num_models % self.models_per_group != 0:
                train_final_model_group = self.train_final_model_group_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, model_group_index=num_groups,
                    num_points_per_model=num_points_per_model
                )
                predictions_in.append(train_final_model_group.outputs.predictions_in)
                predictions_out.append(train_final_model_group.outputs.predictions_out)
                if "metrics_avg" in train_final_model_group.outputs:
                    metrics_avg.append(train_final_model_group.outputs.metrics_avg)
                if "dp_parameters" in train_final_model_group.outputs:
                    dp_parameters.append(train_final_model_group.outputs.dp_parameters)

            outputs =  {
                "predictions_in": aggregate_output(predictions_in, aggregator="concatenate_datasets"),
                "predictions_out": aggregate_output(predictions_out, aggregator="concatenate_datasets")
            }
            if len(metrics_avg) > 0:
                outputs["metrics_avg"] = aggregate_output(metrics_avg, aggregator="average_json")
            if len(dp_parameters) > 0:
                outputs["dp_parameters"] = aggregate_output(dp_parameters, aggregator="assert_json_equal")
            return outputs
        return pipeline(train_base_data=train_base_data, in_out_data=in_out_data, in_indices=in_indices, out_indices=out_indices, 
                        validation_base_data=validation_base_data, base_seed=base_seed, num_points_per_model=num_points_per_model)
