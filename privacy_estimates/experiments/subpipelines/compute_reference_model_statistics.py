from azure.ai.ml import dsl, Input

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader
from privacy_estimates.experiments.components import filter_aux_data, reinsert_aux_data


class ComputeSingleOfflineReferenceModelStatisticsLoader:
    def __init__(self, train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader):
        self.train_loader = train_loader
        self.inference_loader = inference_loader

    def load(self, train_data: Input, validation_data: Input, canary_data: Input, seed: int):
        @dsl.pipeline(name="compute_reference_model_statistics")
        def p(train_data: Input, validation_data: Input, canary_data: Input, seed: int):

            filter_train_data = filter_aux_data(full=train_data)
            filter_val_data = filter_aux_data(full=validation_data)
            filter_canary_data = filter_aux_data(full=canary_data)

            train = self.train_loader.load(
                train_data=filter_train_data.outputs.filtered, validation_data=filter_val_data.outputs.filtered, seed=seed
            )

            compute_predictions = self.inference_loader.load(
                model=train.outputs.model, dataset=filter_canary_data.outputs.filtered
            )

            reinsert_canary_data = reinsert_aux_data(
                filtered=compute_predictions.outputs.predictions, aux=filter_canary_data.outputs.aux
            )

            return {
                "statistics": reinsert_canary_data.outputs.full,
            }
        return p(train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=seed)
    
