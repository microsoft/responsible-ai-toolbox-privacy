from azure.ai.ml import Input
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.offline_black_box_membership_inference import (
    OfflineBlackBoxMembershipInferenceGameBase, GameConfig, AttackConfig
)
from privacy_estimates.experiments.attacks import RmiaLoader
from privacy_estimates.experiments.aml import WorkspaceConfig
from privacy_estimates.experiments.challenge_point_selectors import ExternalCanaryDataset
from privacy_estimates.experiments.components import compute_mi_signals


EXPERIMENT_DIR = Path(__file__).parent


@dataclass
class SharedTrainingParameters:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    target_epsilon: float
    delta: float
    per_sample_max_grad_norm: float
    max_sequence_length: int
    model_name: str


@dataclass
class SharedInferenceParameters:
    batch_size: int


class TrainTransformerComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    @property
    def component(self):
        return self.aml_loader.load_from_component_spec(
            EXPERIMENT_DIR/"components"/"fine-tune-transformer-classifier"/"component_spec.yaml", version="local"
        )

    @property
    def parameter_dict(self):
        return asdict(self.parameters)

    @property
    def compute(self) -> str:
        return self.aml_loader.workspace.gpu_compute


class TransformerInferneceComponentLoader(InferenceComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedInferenceParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, model, dataset):
        @dsl.pipeline(default_compute=self.default_compute)
        def inference_pipeline(model, dataset):
            compute_inference = self.aml_loader.load_from_component_spec(
                EXPERIMENT_DIR/"components"/"predict-with-transformer-classifier"/"component_spec.yaml", version="local"
            )(model=model, dataset=dataset)
            compute_inference.compute = self.aml_loader.workspace.gpu_compute
            compute_signal = compute_mi_signal(logits_and_labels=compute_inference.outputs.predictions)
            return {"mi_signal": compute_signal.outputs.mi_signal}
        p = inference_pipeline(model=model, dataset=dataset)
        return p


class Game(OfflineBlackBoxMembershipInferenceGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters,
                 shared_inference_parameters: SharedInferenceParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig, attack_config: AttackConfig) -> None:

        train_loader = TrainTransformerComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_training_parameters
        )

        inference_loader = TransformerInferneceComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_inference_parameters
        )

        attack_loader = RmiaLoader()

        super().__init__(
            workspace=workspace,
            game_config=game_config,
            attack_config=attack_config,
            train_loader=train_loader,
            inference_loader=inference_loader,
            attack_loader=attack_loader,
        )

    @property
    def train_data(self) -> Input:
        return self.workspace.ml_client.data.get(name="SST2-train", version="5")

    @property
    def validation_data(self) -> Input:
        return self.workspace.ml_client.data.get(name="SST2-test", version="5")
    
    @property
    def canary_data(self) -> Input:
        return self.workspace.ml_client.data.get(name="AmazonPolarity5k-train", version="2")


if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
