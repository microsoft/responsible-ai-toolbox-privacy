from dataclasses import dataclass
from typing import Dict
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineJob
from abc import abstractmethod

from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.subpipelines import ComputeShadowModelStatisticsLoader, TrainManyModelsLoader
from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader
from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.challenge_point_selectors import ChallengePointSelectionLoader
from privacy_estimates.experiments.components import create_in_out_data_for_membership_inference_challenge


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


class BlackBoxMembershipInferenceGameBase(ExperimentBase):
    def __init__(
            self, game_config: GameConfig, shadow_model_config: ShadowModelConfig, workspace: WorkspaceConfig,
            train_loader: TrainingComponentLoader, inference_loader: InferenceComponentLoader, attack_loader: AttackLoader,
            challenge_point_selection_loader: ChallengePointSelectionLoader
    ) -> None:
        super().__init__(workspace=workspace)
        self.game_config = game_config
        self.shadow_model_config = shadow_model_config

        self.attack_loader = attack_loader
        self.challenge_point_selection_loader = challenge_point_selection_loader
        self.train_loader = train_loader
        self.inference_loader = inference_loader

        if (
            self.challenge_point_selection_loader.requires_shadow_model_statistics or
            self.attack_loader.requires_shadow_model_statistics
        ):
            self.shadow_model_statistics_loader = ComputeShadowModelStatisticsLoader(
                train_loader=self.train_loader, inference_loader=self.inference_loader,
                num_models=self.shadow_model_config.num_models,
                num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
                num_models_per_group=self.game_config.num_models_per_group
            )
        else:
            self.shadow_model_statistics_loader = None

        self.train_many_models_loader = TrainManyModelsLoader(
            num_models=self.game_config.num_models, train_loader=train_loader, inference_loader=inference_loader,
            sample_selection="independent", merge_unused_samples="all_with_train",
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
    def experiment_name(self) -> str:
        return "black_box_membership_inference_game"
    
    @property
    def default_compute(self) -> str:
        return self.workspace.cpu_compute

    def pipeline(self, train_data: Input, validation_data: Input) -> PipelineJob:
        @dsl.pipeline(default_compute=self.default_compute)
        def game_pipeline(train_data: Input, validation_data: Input):
            shadow_model_statistics = None
            if self.shadow_model_statistics_loader is not None:
                shadow_model_statistics = self.shadow_model_statistics_loader.load(
                    train_data=train_data, validation_data=validation_data, seed=self.game_config.seed
                ).outputs.statistics
            
            select_challenge_points = self.challenge_point_selection_loader.load(data=train_data, shadow_model_statistics=shadow_model_statistics)

            create_challenge = create_in_out_data_for_membership_inference_challenge(
                train_data=train_data, challenge_points=select_challenge_points.outputs.challenge_points,
                seed=self.game_config.seed, adjacency_relation="add_remove"
            )

            train_many_models = self.train_many_models_loader.load(
                train_base_data=create_challenge.outputs.train_base_data, validation_base_data=validation_data,
                in_out_data=create_challenge.outputs.in_out_data, in_indices=create_challenge.outputs.in_indices,
                out_indices=create_challenge.outputs.out_indices, base_seed=self.game_config.seed,
                num_points_per_model=self.game_config.num_challenge_points_per_model
            )


        return game_pipeline(train_data=train_data, validation_data=validation_data)
    
    @property
    def pipeline_parameters(self) -> Dict:
        return {
            "train_data": self.train_data,
            "validation_data":  self.validation_data
        }
