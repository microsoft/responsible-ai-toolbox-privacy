from dataclasses import dataclass
from typing import Dict, Optional, Literal, Union
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import PipelineJob
from abc import abstractmethod

from privacy_estimates.experiments.aml import (
    ExperimentBase, WorkspaceConfig, ClusterComputeConfig, PrivacyEstimatesComponentLoader
)
from privacy_estimates.experiments.subpipelines import (
    ComputeShadowArtifactStatisticsLoader, TrainManyArtifactsLoader, add_index_to_dataset
)
from privacy_estimates.experiments.loaders import ScoreComponentLoader, TrainComponentLoader, ComponentLoader
from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.challenge_point_selectors import ChallengePointSelectionLoader
from privacy_estimates.experiments.components import (
    create_in_out_data_for_membership_inference_challenge, convert_in_out_to_challenge, compute_privacy_estimates,
    convert_int_to_uri_file
)
from privacy_estimates.experiments.games.configs import PrivacyEstimationConfig


@dataclass
class GameConfig:
    num_artifacts: int
    seed: int
    num_repetitions: int = 1
    num_challenge_points_per_artifact: int = 1
    num_artifacts_per_group: int = 32
    num_concurrent_jobs_per_node: int = 1


@dataclass
class ShadowArtifactConfig:
    num_artifacts: int
    in_fraction: float


@dataclass
class MISignalConfig:
    method: str
    aggregation: Optional[str] = None
    extra_args: Optional[Dict] = None


class BlackBoxMembershipInferenceGameBase(ExperimentBase):
    def __init__(
            self, game_config: GameConfig, shadow_artifact_config: ShadowArtifactConfig, workspace: WorkspaceConfig,
            train_loader: TrainComponentLoader, score_loader: ScoreComponentLoader, attack_loader: AttackLoader,
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

        if (
            self.challenge_point_selection_loader.requires_shadow_artifact_statistics or
            self.attack_loader.requires_shadow_artifact_statistics
        ):
            self.mi_statistics_loader = ComputeShadowArtifactStatisticsLoader(
                train_loader=self.train_loader, score_loader=self.score_loader,
                num_artifacts=self.shadow_artifact_config.num_artifacts, in_fraction=self.shadow_artifact_config.in_fraction,
                num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
                num_artifacts_per_group=self.game_config.num_artifacts_per_group, workspace=workspace,
                num_repetitions=self.game_config.num_repetitions
            )
        else:
            self.mi_statistics_loader = None

        self.train_many_artifacts_loader = TrainManyArtifactsLoader(
            num_artifacts=self.game_config.num_artifacts, train_loader=train_loader, score_loader=score_loader,
            sample_selection="partitioned", merge_unused_samples="all_with_train",
            num_concurrent_jobs_per_node=self.game_config.num_concurrent_jobs_per_node,
            num_artifacts_per_group=self.game_config.num_artifacts_per_group, tag_artifact_index=False,
            num_repetitions=self.game_config.num_repetitions
        )

    @property
    def experiment_name(self) -> str:
        return "black_box_membership_inference_game"

    @property
    def default_compute(self) -> str:
        return self.workspace.default_compute

    def pipeline(self) -> PipelineJob:
        if not isinstance(self.default_compute, ClusterComputeConfig):
            raise TypeError("The default compute must be a comput cluster (i.e. `ClusterComputeConfig` object)")
        @dsl.pipeline(default_compute=self.default_compute.cluster_name)
        def game_pipeline() -> PipelineJob:
            preprocessed_datasets = self.preprocess_datasets()

            train_data = preprocessed_datasets["train_data"]
            validation_data = preprocessed_datasets["validation_data"]
            canary_data = preprocessed_datasets["canary_data"]

            train_data = add_index_to_dataset(data=train_data, split="train").outputs.output
            validation_data = add_index_to_dataset(data=validation_data, split="validation").outputs.output
            canary_data = add_index_to_dataset(data=canary_data, split="canary").outputs.output

            mi_statistics = None
            if self.mi_statistics_loader is not None:
                mi_statistics = self.mi_statistics_loader.load(
                    train_data=train_data, validation_data=validation_data, canary_data=canary_data, seed=self.game_config.seed
                ).outputs.statistics

            select_challenge_points = self.challenge_point_selection_loader.load(
                data=canary_data, shadow_artifact_statistics=mi_statistics
            )

            load_from_function = PrivacyEstimatesComponentLoader().load_from_function

            create_challenge = load_from_function(
                create_in_out_data_for_membership_inference_challenge
            )(
                train_data=train_data, challenge_points=select_challenge_points.outputs.challenge_points,
                seed=self.game_config.seed, max_num_challenge_points=self.challenge_point_selection_loader.num_challenge_points
            )

            convert_num_points_per_artifact = load_from_function(convert_int_to_uri_file)(
                value=self.game_config.num_challenge_points_per_artifact
            )

            train_many_artifacts = self.train_many_artifacts_loader.load(
                train_base_data=create_challenge.outputs.train_base_data, validation_base_data=validation_data,
                in_out_data=create_challenge.outputs.in_out_data, in_indices=create_challenge.outputs.in_indices,
                out_indices=create_challenge.outputs.out_indices, base_seed=self.game_config.seed,
                num_points_per_artifact=convert_num_points_per_artifact.outputs.output
            )
            optional_training_outputs = {}
            if "dp_parameters" in train_many_artifacts.outputs:
                optional_training_outputs["dp_parameters"] = train_many_artifacts.outputs.dp_parameters
            if self.privacy_estimation_config.smallest_delta is not None:
                optional_training_outputs["smallest_delta"] = self.privacy_estimation_config.smallest_delta

            convert_to_challenge = load_from_function(convert_in_out_to_challenge)(
                scores_in=train_many_artifacts.outputs.scores_in,
                scores_out=train_many_artifacts.outputs.scores_out,
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
        return game_pipeline()

    @abstractmethod
    def preprocess_datasets(
        self
    ) -> Dict[Literal["train_data", "validation_data", "canary_data"], Union[Input, Output]]:
        raise NotImplementedError("`preprocess_datasets` must be implemented in a subclass")

    @property
    def pipeline_parameters(self) -> Dict:
        return dict()
