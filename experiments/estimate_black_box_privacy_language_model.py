from typing import Dict, Literal
from azure.ai.ml import Input, Output
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.black_box_membership_inference import (
    BlackBoxMembershipInferenceGameBase, ShadowModelConfig, GameConfig
)
from privacy_estimates.experiments.attacks.rmia import RmiaLoader
from privacy_estimates.experiments.aml import WorkspaceConfig, ServerlessComputeConfig, ComputeConfig, ClusterComputeConfig
from privacy_estimates.experiments.challenge_point_selectors import TopKChallengePoints


EXPERIMENT_DIR = Path(__file__).parent


@dataclass
class SharedTrainingParameters:
    per_device_train_batch_size: int
    learning_rate: float
    num_train_epochs: float
    model_name: str
    gradient_accumulation_steps: int
    max_sequence_length: int


@dataclass
class SharedInferenceParameters:
    per_device_batch_size: int


class TrainLMComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters, compute_config: ComputeConfig):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters
        self.compute_config = compute_config

    def load(self, train_data: Input, validation_data: Input, seed: int):
        component = self.aml_loader.load_from_component_spec(
            EXPERIMENT_DIR/"components"/"fine-tune-lm"/"component_spec.yaml", version="local"
        )
        job = component(train_data=train_data, validation_data=validation_data, seed=seed, **asdict(self.parameters),
                        text_column="sentence")
        job = self.compute_config.apply(job)
        return job


class LMInferenceComponentLoader(InferenceComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedInferenceParameters, compute_config: ComputeConfig):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters
        self.compute_config = compute_config

    def load(self, model: Input, dataset: Input):
        component = self.aml_loader.load_from_component_spec(
            EXPERIMENT_DIR/"components"/"predict-with-lm"/"component_spec.yaml", version="local"
        )
        job = component(model=model, data=dataset, **asdict(self.parameters), text_column="sentence")
        job = self.compute_config.apply(job)
        return job


class Game(BlackBoxMembershipInferenceGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters,
                 shared_inference_parameters: SharedInferenceParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig, shadow_model_config: ShadowModelConfig) -> None:
        
        gpu_distributed_config = ClusterComputeConfig(**(workspace.compute["gpu_distributed"]))
        gpu_single_config = ClusterComputeConfig(**(workspace.compute["gpu_single"]))


        train_loader = TrainLMComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_training_parameters,
            compute_config=gpu_distributed_config
        )

        inference_loader = LMInferenceComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_inference_parameters,
            compute_config=gpu_single_config,
        )

        attack_loader = RmiaLoader(use_log_column=True)

        challenge_point_selection_loader = TopKChallengePoints(
            num_challenge_points=game_config.num_models*game_config.num_challenge_points_per_model
        )

        super().__init__(
            workspace=workspace,
            game_config=game_config,
            shadow_model_config=shadow_model_config,
            train_loader=train_loader,
            inference_loader=inference_loader,
            attack_loader=attack_loader,
            challenge_point_selection_loader=challenge_point_selection_loader
        )

    def preprocess_datasets(self) -> Dict[Literal['train_data', 'validation_data', 'canary_data'], Input | Output]:
        return {
            "train_data": self.workspace.ml_client.data.get(name="SST2-train", version="1"),
            "validation_data": self.workspace.ml_client.data.get(name="SST2-test", version="1"),
            "canary_data": self.workspace.ml_client.data.get(name="SST2-test", version="1")
        }


if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
