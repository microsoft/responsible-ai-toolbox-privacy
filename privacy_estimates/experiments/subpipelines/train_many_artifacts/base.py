from abc import ABC
from azure.ai.ml import Input

from privacy_estimates.experiments.loaders import TrainSingleArtifactAndScoreArguments


class TrainArtifactGroupBase(ABC):
    def __init__(self, num_artifacts: int, group_size: int, single_artifact_arguments: TrainSingleArtifactAndScoreArguments):
        self.num_artifacts = num_artifacts
        self.group_size = group_size
        self.single_artifact_arguments = single_artifact_arguments

    def load(self, train_base_data: Input, validation_base_data, in_out_data: Input, in_indices: Input, out_indices: Input,
             base_seed: int, artifact_group_index: int, num_points_per_artifact: int):
        pass
