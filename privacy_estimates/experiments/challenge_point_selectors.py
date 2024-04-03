from abc import abstractmethod
from typing import Optional
from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import Component, Pipeline
from dataclasses import dataclass

from privacy_estimates.experiments.attacks import AttackLoader
from privacy_estimates.experiments.loaders import ComponentLoader
from privacy_estimates.experiments.components import select_cross_validation_challenge_points, append_column_constant_str, append_column_incrementing


class ChallengePointSelectionLoader(ComponentLoader):
    @abstractmethod
    def load(self) -> Component:
        pass

    @property
    def requires_shadow_model_statistics(self) -> bool:
        return False


@dataclass
class NaturalCanaryCrossValidationConfig:
    n_models: int


class SelectNaturalCrossValidationChallengePoints(ChallengePointSelectionLoader):
    def __init__(self, attack_loader: Optional[AttackLoader], num_challenge_points: int):
        self.attack_loader = attack_loader
        self.num_challenge_points = num_challenge_points

    def load(self, data: Input, shadow_model_statistics: Input) -> Pipeline:
        @dsl.pipeline(name="select_natural_cross_validation_challenge_points")
        def p(data: Input, shadow_model_statistics: Input) -> Pipeline:
            preprocess = select_cross_validation_challenge_points.preprocess(shadow_model_statistics=shadow_model_statistics)
            attack = self.attack_loader.load(
                challenge_points=preprocess.outputs.challenge_points_for_cross_validation,
                shadow_model_statistics=preprocess.outputs.shadow_model_statistics_for_cross_validation
            )
            postprocess = select_cross_validation_challenge_points.postprocess(
                data=data, scores=attack.outputs.scores, num_challenge_points=self.num_challenge_points,
                challenge_points_for_cross_validation=preprocess.outputs.challenge_points_for_cross_validation,
            )
            return {
                "challenge_points": postprocess.outputs.mi_challenge_points,
            }
        return p(data=data, shadow_model_statistics=shadow_model_statistics)
    

class ExternalCanaryDataset(ChallengePointSelectionLoader):
    def __init__(self, canary_data: Input, num_challenge_points: int):
        self.canary_data = canary_data
        self.num_challenge_points = num_challenge_points

    def load(self, data: Input, shadow_model_statistics: Input) -> Pipeline:
        @dsl.pipeline(name="select_challenge_points_from_external_canary_dataset")
        def p() -> Pipeline:
            canaries = append_column_constant_str(data=self.canary_data, name="split", value="canary").outputs.output
            canaries = append_column_incrementing(data=canaries, name="sample_index").outputs.output
            return {
                "challenge_points": canaries,
            }
        return p()
