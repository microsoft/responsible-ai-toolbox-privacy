from azure.ai.ml import Input, Output, dsl
from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.subpipelines import (
	ComputeShadowModelStatisticsLoader, TrainManyModelsLoader, add_index_to_dataset
)


class DataMembershipInferenceGameBase(ExperimentBase):
    def __init__(
        self, game_config: GameConfig, shadow_artifact_config: ShadowArtifactConfig, workspace: WorkspaceConfig,
        train_loader: TrainComponentLoader, score_loader: ScoreComponent, attack_loader: AttackLoader,
        challenge_point_selection_loader: ChallengePointSelectionLoader,
        privacy_estimation_config: PrivacyEstimationConfig = PrivacyEstimationConfig(),
    ) -> None:
        super().__init__(workspace=workspace)

        self.game_config = game_config
        self.shadow_artifact_config = shadow_artifact_config
        self.attack_loader = attack_loader

        self.challenge_point_selection_loader = challenge_point_selection_loader
        self.train_loader = train_loader
        self.score_loader = score_loader
        self.privacy_estimation_config = privacy_estimation_config

        self.mi_statistics_loader = ComputeShadowArtifactStatisticsLoader(
            train_loader=self.train_loader,
            score_loader=self.score_loader,
            num_artifacts=self.shadow_artifact_config.num_artifacts,
            num_models=self.shadow_model_config.num_models, in_fraction=self.shadow_model_config.in_fraction,
            num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
            num_models_per_group=self.game_config.num_models_per_group, workspace=workspace,
            num_repetitions=self.game_config.num_repetitions
        )

    @property
    def experiment_name(self):
        return "data_membership_inference_auditing"

    def pipeline(self, train_data: Input, validation_data: Input, canary_data: Input):
        @dsl.pipeline(default_compute=self.default_compute)
        def game(train_data: Input, validation_data: Input, canary_data: Input):
            train_data = add_index_to_dataset(data=train_data, split="train").outputs.output
            validation_data = add_index_to_dataset(data=validation_data, split="validation").outputs.output
            canary_data = add_index_to_dataset(data=canary_data, split="canary").outputs.output

            return {"privacy_report": estimate_privacy.outputs.privacy_report}
        return game(train_data=train_data, validation_data=validation_data, canary_data=canary_data)
