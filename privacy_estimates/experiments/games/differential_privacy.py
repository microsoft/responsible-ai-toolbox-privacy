from dataclasses import dataclass
from typing import Dict
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineJob
from abc import abstractmethod

from privacy_estimates.experiments.aml import ExperimentBase, WorkspaceConfig
from privacy_estimates.experiments.loaders import TrainingComponentLoader, SingleGameLoader
from privacy_estimates.experiments.components import postprocess_dpd_data, compute_privacy_estimates
from privacy_estimates.experiments.subpipelines.run_many_games import RunManyGamesLoader
from privacy_estimates.experiments.games.configs import PrivacyEstimationConfig


@dataclass
class GameConfig:
    seed: int
    """Seed to be used for training and challenge bit creation"""

    num_games: int = 1
    """Number of games to run in parallel"""


class DPDSingleGame(SingleGameLoader):
    def __init__(self, train_loader: TrainingComponentLoader):
        self.train_loader = train_loader

    def load(self, train_data: Input, validation_data, seed: int):
        @dsl.pipeline(name="dpd_single_game")
        def dpd_single_game(train_data: Input, validation_data: Input, seed: int):
            train_model = self.train_loader.load(train_data=train_data, validation_data=validation_data, seed=seed)

            if not hasattr(train_model.outputs, "dpd_data"):
                raise ValueError(f"{self.train_loader.__class__.__name__} does not output `dpd_data`. "
                                 "Please make sure that your training component outputs `dpd_data`.")

            provides_dp_parameters = hasattr(train_model.outputs, "dp_parameters")
            dp_parameters_kwargs = {} if not provides_dp_parameters else {"dp_parameters": train_model.outputs.dp_parameters}

            post_process = postprocess_dpd_data(dpd_data=train_model.outputs.dpd_data, seed=seed, **dp_parameters_kwargs)

            results = {
                "scores": post_process.outputs.scores,
                "challenge_bits": post_process.outputs.challenge_bits,
            }
            if provides_dp_parameters:
                results["dp_parameters"] = post_process.outputs.postprocessed_dp_parameters
            return results

        return dpd_single_game(train_data=train_data, validation_data=validation_data, seed=seed)


class DifferentialPrivacyGameBase(ExperimentBase):
    def __init__(
            self, game_config: GameConfig, workspace: WorkspaceConfig,
            privacy_estimation_config: PrivacyEstimationConfig = PrivacyEstimationConfig(),
    ) -> None:
        super().__init__(workspace=workspace)
        self.game_config = game_config
        self.privacy_estimation_config = privacy_estimation_config

        single_game_loader = DPDSingleGame(train_loader=self.train_loader)
        self.many_games_loader = RunManyGamesLoader(single_game=single_game_loader, num_games=game_config.num_games)

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
    @abstractmethod
    def train_loader(self) -> TrainingComponentLoader:
        raise NotImplementedError(
            f"train_loader not implemented for {self.__class__.__name__}"
        )

    @property
    def experiment_name(self) -> str:
        return "differential_privacy_game"

    @property
    def default_compute(self) -> str:
        return self.workspace.cpu_compute

    def pipeline(self, train_data: Input, validation_data: Input) -> PipelineJob:

        @dsl.pipeline(default_compute=self.default_compute)
        def game_pipeline(train_data: Input, validation_data: Input):

            games = self.many_games_loader.load(train_data=train_data, validation_data=validation_data,
                                                base_seed=self.game_config.seed)

            optional_estimate_parameters = {}
            if "dp_parameters" in games.outputs:
                optional_estimate_parameters["dp_parameters"] = games.outputs.dp_parameters
            if self.privacy_estimation_config.smallest_delta is not None:
                optional_estimate_parameters["smallest_delta"] = self.privacy_estimation_config.smallest_delta

            compute_estimates = compute_privacy_estimates(
                scores=games.outputs.scores, challenge_bits=games.outputs.challenge_bits,
                **optional_estimate_parameters
            )

            return {
                "privacy_report": compute_estimates.outputs.privacy_report,
            }

        return game_pipeline(train_data=train_data, validation_data=validation_data)

    @property
    def pipeline_parameters(self) -> Dict:
        return {
            "train_data": self.train_data,
            "validation_data":  self.validation_data
        }
