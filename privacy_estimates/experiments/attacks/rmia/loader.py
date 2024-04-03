from pathlib import Path
from azure.ai.ml import load_component, Input

from privacy_estimates.experiments.attacks.base import AttackLoader


class RmiaLoader(AttackLoader):
    def __init__(self):
        pass

    def load(self, challenge_points: Input, mi_statistics: Input):
        return load_component(source=Path(__file__).parent/"component_spec.yaml")(
            challenge_points=challenge_points, mi_statistics=mi_statistics
        )

    @property
    def requires_shadow_model_statistics(self) -> bool:
        return False

    @property
    def requires_reference_statistics(self) -> bool:
        return True
