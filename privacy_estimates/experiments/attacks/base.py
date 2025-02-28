from abc import ABC, abstractmethod


class AttackLoader(ABC):
    @abstractmethod
    def load(self):
        pass

    @property
    def requires_shadow_artifact_statistics(self) -> bool:
        return False
