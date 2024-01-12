from azure.ai.ml import Input
from azure.ai.ml.entities import Component
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.differential_privacy import (
    DifferentialPrivacyGameBase, GameConfig, PrivacyEstimationConfig
)
from privacy_estimates.experiments.aml import WorkspaceConfig


EXPERIMENT_DIR = Path(__file__).parent


@dataclass
class SharedTrainingParameters:
    total_train_batch_size: int
    max_physical_batch_size: int
    learning_rate: float
    lr_scheduler_gamma: float
    num_train_epochs: float
    target_epsilon: float
    delta: float
    per_sample_max_grad_norm: float


class TrainCNNComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, train_data: Input, validation_data: Input, seed: Input) -> Component:
        component = self.aml_loader.load_from_component_spec(
            EXPERIMENT_DIR/"components"/"train-cnn-classifier"/"component_spec.yaml", version="local"
        )(
            train_data=train_data, validation_data=validation_data, seed=seed, **asdict(self.parameters)
        )
        component.compute = self.aml_loader.workspace.gpu_compute
        return component


class Game(DifferentialPrivacyGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig) -> None:
        self.shared_training_parameters = shared_training_parameters

        super().__init__(
            workspace=workspace,
            game_config=game_config,
        )

    @property
    def train_data(self) -> Input:
        return self.workspace.ml_client.data.get(name="CIFAR10Normalized-train", version="1")

    @property
    def validation_data(self) -> Input:
        return self.workspace.ml_client.data.get(name="CIFAR10Normalized-test", version="1")

    @property
    def train_loader(self) -> TrainCNNComponentLoader:
        return TrainCNNComponentLoader(
            aml_component_loader=self.aml_component_loader,
            parameters=self.shared_training_parameters
        )


if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
