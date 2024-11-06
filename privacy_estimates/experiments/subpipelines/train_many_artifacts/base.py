from abc import ABC
from azure.ai.ml import Input

from privacy_estimates.experiments.loaders import TrainSingleArtifactAndScoreArguments


class TrainArtifactsGroupBase(ABC):
    def __init__(self, num_models: int, group_size: int, single_model_arguments: TrainSingleArtifactAndScoreArguments):
        self.num_models = num_models
        self.group_size = group_size
        self.single_model_arguments = single_model_arguments

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, model_group_index: int, num_points_per_model: int):
        pass
