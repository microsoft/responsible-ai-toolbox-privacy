from azure.ai.ml import Input, load_component
from privacy_estimates.experiments.loaders import ScoreComponentLoader
from pathlib import Path


class ScoreDataLoader(ScoreComponentLoader):
    def load(self, artifact: Input, dataset: Input):
        component = load_component(Path(__file__).parent / "component_spec.yaml")
        return component(artifact=artifact, dataset=dataset)
