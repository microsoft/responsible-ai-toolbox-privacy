from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineComponent

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader


class ComputeSingleOfflineReferenceModelStatisticsLoader:
    def __init__(self, train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader):
        self.train_loader = train_loader
        self.inference_loader = inference_loader

    def load(self, train_data: Input, validation_data: Input, canary_data: Input, seed: int) -> PipelineComponent:
        @dsl.pipeline(name="compute_reference_model_statistics")
        def p(train_data: Input, validation_data: Input, canary_data: Input, seed: int) -> PipelineComponent:
            train_model = self.train_loader.load(train_data=train_data, validation_data=validation_data, seed=seed)

            inference_component = self.inference_loader.load(
                model=train_model.outputs.model,
                dataset=canary_data
            ) 

            return {
                "statistics": inference_component.outputs.predictions,
            }
        return p(train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=seed)
    
