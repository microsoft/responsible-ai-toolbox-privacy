from pathlib import Path
from azure.ai.ml import Input
from dataclasses import dataclass
from typing import Optional

from privacy_estimates.experiments.attacks.base import AttackLoader
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader


@dataclass
class RmiaConfig:
    offline_a: Optional[float] = None


class RmiaLoader(AttackLoader):
    def __init__(self, offline_a: Optional[float] = None, use_log_column: bool = False):
        self.offline_a = offline_a
        self.use_log_column = use_log_column

    def load(self, challenge_points: Input, mi_statistics: Input):
        extra_args = {}
        if self.offline_a is not None:
            extra_args["offline_a"] = self.offline_a
        return PrivacyEstimatesComponentLoader().load_from_component_spec(source=Path(__file__).parent/"component_spec.yaml")(
            challenge_points=challenge_points, mi_statistics=mi_statistics, use_log_column=self.use_log_column,
            **extra_args
        )

    @property
    def requires_shadow_artifact_statistics(self) -> bool:
        return True
