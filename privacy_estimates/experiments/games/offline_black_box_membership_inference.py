from dataclasses import dataclass
from typing import Dict, Optional
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineJob
from abc import abstractmethod

from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.subpipelines import (
	TrainManyModelsLoader, ComputeSingleOfflineReferenceModelStatisticsLoader, add_index_to_dataset
)
from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader
from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.challenge_point_selectors import ChallengePointSelectionLoader
from privacy_estimates.experiments.components import (
    create_in_out_data_for_membership_inference_challenge, convert_in_out_to_challenge, compute_privacy_estimates
)
from privacy_estimates.experiments.games.configs import PrivacyEstimationConfig


@dataclass
class GameConfig:
    num_models: int
    seed: int
    num_challenge_points_per_model: int = 1
    num_models_per_group: int = 32
    num_concurrent_jobs_per_node: int = 1


@dataclass
class AttackConfig:
    mi_signal_method: str


class OfflineBlackBoxMembershipInferenceGameBase(ExperimentBase):
    def __init__(
            self, game_config: GameConfig, attack_config: AttackConfig, workspace: WorkspaceConfig,
            train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader, attack_loader: AttackLoader,
            challenge_point_selection_loader: ChallengePointSelectionLoader,
            privacy_estimation_config: PrivacyEstimationConfig = PrivacyEstimationConfig(),
    ) -> None:
        super().__init__(workspace=workspace)
        self.game_config = game_config
        self.attack_config = attack_config

        self.attack_loader = attack_loader
        self.challenge_point_selection_loader = challenge_point_selection_loader
        self.train_loader = train_loader
        self.inference_loader = inference_loader
        self.privacy_estimation_config = privacy_estimation_config

        self.mi_statistics_loader = ComputeSingleOfflineReferenceModelStatisticsLoader(
            train_loader=self.train_loader, inference_loader=self.inference_loader
        )

        self.train_many_models_loader = TrainManyModelsLoader(
            num_models=self.game_config.num_models, train_loader=train_loader, inference_loader=inference_loader,
            sample_selection="partitioned", merge_unused_samples="all_with_train",
            num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
            num_models_per_group=self.game_config.num_models_per_group, tag_model_index=False
        )

    @property
    @abstractmethod
    def train_data(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} needs to implement property train_data"
        )

    @property
    @abstractmethod
    def validation_data(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} needs to implement property validation_data"
        )
    
    @property
    def canary_data(self):
        return None

    @property
    def experiment_name(self) -> str:
        return "black_box_membership_inference_game"

    @property
    def default_compute(self) -> str:
        return self.workspace.cpu_compute

    def pipeline(self, train_data: Input, validation_data: Input, canary_data: Optional[Input] = None) -> PipelineJob:
        @dsl.pipeline(default_compute=self.default_compute)
        def game_pipeline(train_data: Input, validation_data: Input, canary_data: Input) -> PipelineJob:
            train_data = add_index_to_dataset(data=train_data, split="train").outputs.output
            validation_data = add_index_to_dataset(data=validation_data, split="validation").outputs.output
            canary_data = add_index_to_dataset(data=canary_data, split="canary").outputs.output

            mi_statistics = self.mi_statistics_loader.load(
                train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=self.game_config.seed
            ).outputs.statistics

            create_challenge = create_in_out_data_for_membership_inference_challenge(
                train_data=train_data, challenge_points=canary_data,
                seed=self.game_config.seed, max_num_challenge_points=self.challenge_point_selection_loader.num_challenge_points
            )

            train_many_models = self.train_many_models_loader.load(
                train_base_data=create_challenge.outputs.train_base_data, validation_base_data=validation_data,
                in_out_data=create_challenge.outputs.in_out_data, in_indices=create_challenge.outputs.in_indices,
                out_indices=create_challenge.outputs.out_indices, base_seed=self.game_config.seed,
                num_points_per_model=self.game_config.num_challenge_points_per_model
            )

            convert_to_challenge = convert_in_out_to_challenge(
                predictions_in=train_many_models.outputs.predictions_in,
                predictions_out=train_many_models.outputs.predictions_out,
                all_challenge_bits=create_challenge.outputs.challenge_bits
            )

            attack = self.attack_loader.load(
                mi_statistics=mi_statistics, challenge_points=convert_to_challenge.outputs.challenge
            )

            estimate_privacy = compute_privacy_estimates(
                scores=attack.outputs.scores, challenge_bits=convert_to_challenge.outputs.challenge_bits,
            )

            return {
                "privacy_report": estimate_privacy.outputs.privacy_report,
            }
    
        return game_pipeline(train_data=train_data, validation_data=validation_data, canary_data=canary_data)

    @property
    def pipeline_parameters(self) -> Dict:
        params = {
            "train_data": self.train_data,
            "validation_data":  self.validation_data,
            "canary_data": self.canary_data
        }
        return params
