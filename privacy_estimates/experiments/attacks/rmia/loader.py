from pathlib import Path
from azure.ai.ml import load_component, Input

from privacy_estimates.experiments.attacks.base import AttackLoader


class RmiaLoader(AttackLoader):
    def __init__(self):
        pass

    def load(self, challenge_points: Input, shadow_model_statistics: Input):
        return load_component(source=Path(__file__).parent/"component_spec.yaml")(
            challenge_points=challenge_points, shadow_model_statistics=shadow_model_statistics, mean_estimator="median"
        )

    @property
    def requires_shadow_model_statistics(self) -> bool:
        return True
