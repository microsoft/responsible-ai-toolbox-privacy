from azure.ai.ml import Input
from privacy_estimates.experiments.loaders import ScoreComponentLoader
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader
from pathlib import Path


class ScoreDataLoader(ScoreComponentLoader):
    def load(self, artifact: Input, dataset: Input):
        component = PrivacyEstimatesComponentLoader().load_from_component_spec(
            Path(__file__).parent / "component_spec.yaml"
        )
        return component(artifact=artifact, dataset=dataset)

