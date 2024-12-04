from abc import abstractmethod
from typing import Optional
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import Component, Pipeline
from dataclasses import dataclass

from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.loaders import ComponentLoader
from privacy_estimates.experiments.components import select_cross_validation_challenge_points, select_top_k_rows
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader


class ChallengePointSelectionLoader(ComponentLoader):
    @abstractmethod
    def load(self) -> Component:
        pass

    @property
    def requires_shadow_artifact_statistics(self) -> bool:
        return False


@dataclass
class NaturalCanaryCrossValidationConfig:
    n_artifacts: int


class SelectNaturalCrossValidationChallengePoints(ChallengePointSelectionLoader):
    def __init__(self, attack_loader: Optional[AttackLoader], num_challenge_points: int):
        self.attack_loader = attack_loader
        self.num_challenge_points = num_challenge_points

    def load(self, data: Input, shadow_artifact_statistics: Input) -> Pipeline:
        @dsl.pipeline(name="Select natural cross validation challenge points")
        def p(data: Input, shadow_artifact_statistics: Input) -> Pipeline:
            preprocess = select_cross_validation_challenge_points.preprocess(shadow_artifact_statistics=shadow_artifact_statistics)
            attack = self.attack_loader.load(
                challenge_points=preprocess.outputs.challenge_points_for_cross_validation,
                shadow_artifact_statistics=preprocess.outputs.shadow_artifact_statistics_for_cross_validation
            )
            postprocess = select_cross_validation_challenge_points.postprocess(
                data=data, scores=attack.outputs.scores, num_challenge_points=self.num_challenge_points,
                challenge_points_for_cross_validation=preprocess.outputs.challenge_points_for_cross_validation,
            )
            return {
                "challenge_points": postprocess.outputs.mi_challenge_points,
            }
        return p(data=data, shadow_artifact_statistics=shadow_artifact_statistics)
    

class TopKChallengePoints(ChallengePointSelectionLoader):
    def __init__(self, num_challenge_points: int, allow_fewer: bool = False):
        self.num_challenge_points = num_challenge_points
        self.allow_fewer = allow_fewer

    def load(self, data: Input, shadow_artifact_statistics: Input) -> Pipeline:
        @dsl.pipeline(name="Select top-k challenge points")
        def p(data: Input, shadow_artifact_statistics: Input) -> Pipeline:
            select = PrivacyEstimatesComponentLoader().load_from_function(select_top_k_rows)
            return {
                "challenge_points": select(
                    data=data, k=self.num_challenge_points, allow_fewer=self.allow_fewer
                ).outputs.output
            }
        return p(data=data, shadow_artifact_statistics=shadow_artifact_statistics)
