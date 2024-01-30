from azure.ai.ml.entities import Component
from abc import ABC, abstractmethod
from typing import Dict, Optional

from privacy_estimates.experiments.aml import AMLComponentLoader


class ComponentLoader(ABC):
    def __init__(self, aml_component_loader: Optional[AMLComponentLoader] = None) -> None:
        self.aml_loader = aml_component_loader

    @property
    def component(self):
        raise NotImplementedError(f"`component` needs to be implemented for {self.__class__.__name__}")

    @property 
    def compute(self) -> Optional[str]:
        raise NotImplementedError(f"`compute` needs to be implemented for {self.__class__.__name__}")

    @property 
    def parameter_dict(self) -> Dict:
        return dict()

    def load(self, *args, **kwargs):
        job = self.component(*args, **kwargs, **self.parameter_dict)
        if self.compute is not None:
            job.compute = self.compute
        return job


class SingleGameLoader(ComponentLoader):
    @abstractmethod
    def load(self, train_data, validation_data, seed: int) -> Component:
        """
        Should return a component with an output named `scores`, challenge_bits` and optionally `dp_parameters`.
        """
        pass


class TrainingComponentLoader(ComponentLoader):
    pass


class InferenceComponentLoader(ComponentLoader):
    pass

