from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import TrainSingleModelAndPredictLoader
from privacy_estimates.experiments.components import aggregate_output, append_column_constant_int
from .base import TrainModelGroupBase, TrainSingleModelAndPredictArguments


class TrainModelGroupDistributedLoader(TrainModelGroupBase):
    def __init__(self, num_models: int, group_size: int, single_model_arguments: TrainSingleModelAndPredictArguments):
        super().__init__(num_models=num_models, group_size=group_size, single_model_arguments=single_model_arguments)

        self.train_model_and_predict_loader = TrainSingleModelAndPredictLoader(arguments=single_model_arguments)

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, model_group_index: int, num_points_per_model: int):
        @dsl.pipeline(name="train_model_group_distributed")
        def p(train_base_data: Input, validation_base_data: Input, in_out_data: Input, in_indices: Input, out_indices: Input,
              base_seed: int, model_group_index: int, num_points_per_model: int):
            predictions_in = []
            predictions_out = []
            metrics = []
            dp_parameters = []
            for i in range(0, self.num_models):
                model_index = model_group_index * self.group_size + i
                train_model_and_predict = self.train_model_and_predict_loader.load(
                    train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                    in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, model_index=model_index,
                    num_points_per_model=num_points_per_model
                )

                predictions_in_i = train_model_and_predict.outputs.predictions_in
                predictions_out_i = train_model_and_predict.outputs.predictions_out

                if self.single_model_arguments.tag_model_index is True:
                    predictions_in_i = append_column_constant_int(
                        data=predictions_in_i, name="model_index", value=model_index
                    ).outputs.output
                    predictions_out_i = append_column_constant_int(
                        data=predictions_out_i, name="model_index", value=model_index
                    ).outputs.output

                predictions_in.append(predictions_in_i)
                predictions_out.append(predictions_out_i)
                if hasattr(train_model_and_predict.outputs, "metrics"):
                    metrics.append(train_model_and_predict.outputs.metrics)
                if hasattr(train_model_and_predict.outputs, "dp_parameters"):
                    dp_parameters.append(train_model_and_predict.outputs.dp_parameters)

            output = {
                "predictions_in": aggregate_output(predictions_in, aggregator="concatenate_datasets"), 
                "predictions_out": aggregate_output(predictions_out, aggregator="concatenate_datasets")
            }

            if len(metrics) > 0:
                output["metrics_avg"] = aggregate_output(metrics, aggregator="average_json")
            if len(dp_parameters) > 0:
                output["dp_parameters"] = aggregate_output(dp_parameters, aggregator="assert_json_equal")
            return output

        return p(train_base_data=train_base_data, validation_base_data=validation_base_data, in_out_data=in_out_data,
                 in_indices=in_indices, out_indices=out_indices, base_seed=base_seed, model_group_index=model_group_index,
                 num_points_per_model=num_points_per_model)


