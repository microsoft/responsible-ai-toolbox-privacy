from pathlib import Path
from azure.ai.ml import load_component, Input
from dataclasses import dataclass

from privacy_estimates.experiments.attacks.base import AttackLoader


@dataclass
class RmiaConfig:
    offline_a: float


class RmiaLoader(AttackLoader):
    def __init__(self, offline_a: float):
        self.offline_a = offline_a

    def load(self, challenge_points: Input, mi_statistics: Input):
        return load_component(source=Path(__file__).parent/"component_spec.yaml")(
            challenge_points=challenge_points, mi_statistics=mi_statistics, offline_a=self.offline_a
        )

    @property
    def requires_shadow_model_statistics(self) -> bool:
        return False

    @property
    def requires_reference_statistics(self) -> bool:
        return True
