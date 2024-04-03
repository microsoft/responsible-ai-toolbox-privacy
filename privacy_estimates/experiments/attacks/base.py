from abc import ABC, abstractmethod


class AttackLoader(ABC):
    @abstractmethod
    def load(self):
        pass

    @property
    def requires_shadow_model_statistics(self) -> bool:
        return False
    
    @property
    def requires_reference_statistics(self) -> bool:
        return False
