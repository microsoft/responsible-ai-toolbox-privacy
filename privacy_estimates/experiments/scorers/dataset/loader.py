from azure.ai.ml import Input
from privacy_estimates.experiments.loaders import ScoreComponentLoader
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DataScorerConfig:
    template: str


class ScoreDataLoader(ScoreComponentLoader):
    def __init__(self, config: DataScorerConfig):
        self.config = config

    def load(self, artifact: Input, dataset: Input):
        component = PrivacyEstimatesComponentLoader().load_from_component_spec(
            Path(__file__).parent / "component_spec.yaml"
        )
        return component(artifact=artifact, dataset=dataset, **asdict(self.config))

