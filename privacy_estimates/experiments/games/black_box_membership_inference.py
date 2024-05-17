from dataclasses import dataclass
from typing import Dict, Optional
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineJob
from abc import abstractmethod

from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.subpipelines import (
	ComputeShadowModelStatisticsLoader, TrainManyModelsLoader,
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
class ShadowModelConfig:
    num_models: int
    in_fraction: float


@dataclass
class MISignalConfig:
    method: str
    aggregation: Optional[str] = None
    extra_args: Optional[Dict] = None


class BlackBoxMembershipInferenceGameBase(ExperimentBase):
    def __init__(
            self, game_config: GameConfig, shadow_model_config: ShadowModelConfig, workspace: WorkspaceConfig,
            train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader, attack_loader: AttackLoader,
            challenge_point_selection_loader: ChallengePointSelectionLoader,
            privacy_estimation_config: PrivacyEstimationConfig = PrivacyEstimationConfig(),
    ) -> None:
        super().__init__(workspace=workspace)
        self.game_config = game_config
        self.shadow_model_config = shadow_model_config

        self.attack_loader = attack_loader
        self.challenge_point_selection_loader = challenge_point_selection_loader
        self.train_loader = train_loader
        self.inference_loader = inference_loader
        self.privacy_estimation_config = privacy_estimation_config

        if (
            self.challenge_point_selection_loader.requires_shadow_model_statistics or
            self.attack_loader.requires_shadow_model_statistics
        ):
            self.mi_statistics_loader = ComputeShadowModelStatisticsLoader(
                train_loader=self.train_loader, inference_loader=self.inference_loader,
                num_models=self.shadow_model_config.num_models, in_fraction=self.shadow_model_config.in_fraction,
                num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
                num_models_per_group=self.game_config.num_models_per_group, workspace=workspace
            )
        else:
            self.mi_statistics_loader = None

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
            mi_statistics = None
            if self.mi_statistics_loader is not None:
                mi_statistics = self.mi_statistics_loader.load(
                    train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=self.game_config.seed
                ).outputs.statistics

            select_challenge_points = self.challenge_point_selection_loader.load(
                data=canary_data, shadow_model_statistics=mi_statistics
            )

            create_challenge = create_in_out_data_for_membership_inference_challenge(
                train_data=train_data, challenge_points=select_challenge_points.outputs.challenge_points,
                seed=self.game_config.seed, max_num_challenge_points=self.challenge_point_selection_loader.num_challenge_points
            )

            train_many_models = self.train_many_models_loader.load(
                train_base_data=create_challenge.outputs.train_base_data, validation_base_data=validation_data,
                in_out_data=create_challenge.outputs.in_out_data, in_indices=create_challenge.outputs.in_indices,
                out_indices=create_challenge.outputs.out_indices, base_seed=self.game_config.seed,
                num_points_per_model=self.game_config.num_challenge_points_per_model
            )
            optional_training_outputs = {}
            if "dp_parameters" in train_many_models.outputs:
                optional_training_outputs["dp_parameters"] = train_many_models.outputs.dp_parameters
            if self.privacy_estimation_config.smallest_delta is not None:
                optional_training_outputs["smallest_delta"] = self.privacy_estimation_config.smallest_delta

            convert_to_challenge = convert_in_out_to_challenge(
                predictions_in=train_many_models.outputs.predictions_in,
                predictions_out=train_many_models.outputs.predictions_out,
                all_challenge_bits=create_challenge.outputs.challenge_bits
            )

            attack_kwargs = {} if self.mi_statistics_loader is None else {"mi_statistics": mi_statistics}
            attack = self.attack_loader.load(challenge_points=convert_to_challenge.outputs.challenge, **attack_kwargs)

            estimate_privacy = compute_privacy_estimates(
                scores=attack.outputs.scores, challenge_bits=convert_to_challenge.outputs.challenge_bits,
                **optional_training_outputs
            )
            return {
                "privacy_report": estimate_privacy.outputs.privacy_report,
            }
        kwargs = {} if canary_data is None else {"canary_data": canary_data}
        return game_pipeline(train_data=train_data, validation_data=validation_data, **kwargs)

    @property
    def pipeline_parameters(self) -> Dict:
        params = {
            "train_data": self.train_data,
            "validation_data":  self.validation_data
        }
        if self.canary_data is not None:
            params["canary_data"] = self.canary_data
        return params
