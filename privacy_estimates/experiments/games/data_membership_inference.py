from dataclasses import dataclass
from typing import Dict, Optional
from azure.ai.ml import Input, Output, dsl
from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.subpipelines import (
	ComputeShadowArtifactStatisticsLoader, TrainManyArtifactsLoader, add_index_to_dataset
)
from privacy_estimates.experiments.loaders import ScoreComponentLoader, TrainComponentLoader, TrainSingleArtifactAndScoreLoader
from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.attacks.rmia import RmiaLoader
from privacy_estimates.experiments.challenge_point_selectors import ChallengePointSelectionLoader, TopKChallengePoints
from privacy_estimates.experiments.components import (
    create_in_out_data_for_membership_inference_challenge, convert_in_out_to_challenge, compute_privacy_estimates,
)
from privacy_estimates.experiments.games.black_box_membership_inference import (
    BlackBoxMembershipInferenceGameBase, ShadowArtifactConfig, GameConfig, MISignalConfig, PrivacyEstimationConfig,
)


class ScoreDataLoader(ScoreComponentLoader):
    def __init__(self, workspace: WorkspaceConfig) -> None:
        super().__init__(workspace=workspace)

    def load(self, artifact: Input, dataset: Input):
        component = self.aml_loader.load_from_component_spec(

        )
        return component(artifact=artifact, dataset=dataset)


class DataMembershipInferenceGameBase(BlackBoxMembershipInferenceGameBase):
    def __init__(
        self, game_config: GameConfig, workspace: WorkspaceConfig, train_loader: TrainComponentLoader, 
        shadow_dataset_config: Optional[ShadowArtifactConfig] = None, attack_loader: Optional[AttackLoader] = None,
        challenge_point_selection_loader: Optional[ChallengePointSelectionLoader] = None,
        privacy_estimation_config: Optional[PrivacyEstimationConfig] = None,
    ) -> None:
        # Set sensible defaults for the loaders
        if attack_loader is None:
            attack_loader = RmiaLoader(use_log_column=True, offline_a=None)

        if challenge_point_selection_loader is None:
            challenge_point_selection_loader = TopKChallengePoints(
                num_challenge_points=game_config.num_challenge_points_per_artifact*game_config.num_artifacts
            )

        if privacy_estimation_config is None:
            privacy_estimation_config = PrivacyEstimationConfig()

        if shadow_dataset_config is None:
            shadow_dataset_config = ShadowArtifactConfig(num_artifacts=4, in_fraction=0.5)

        score_loader = ScoreDataLoader(workspace=workspace)

        super().__init__(
            game_config=game_config, shadow_artifact_config=shadow_dataset_config, workspace=workspace,
            train_loader=train_loader, score_loader=score_loader, attack_loader=attack_loader,
            challenge_point_selection_loader=challenge_point_selection_loader,
            privacy_estimation_config=privacy_estimation_config,
        )

    @property
    def experiment_name(self):
        return "data_membership_inference_auditing"
