from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
from json import load, dump
from opacus.optimizers import DPOptimizer

from .attack import CanaryTrackingOptimizer


@dataclass
class DPDistinguishingData:
    scores: List[float]
    sensitivity: float

    @classmethod
    def from_opacus(cls, optimizer: DPOptimizer) -> "DPDistinguishingData":
        if not isinstance(optimizer, DPOptimizer):
            raise TypeError(f"`DifferentialPrivacyDistinguisherOutputWriter` requires a `DPOptimizer`. Got {type(optimizer)}")
        if not isinstance(optimizer.original_optimizer, CanaryTrackingOptimizer):
            raise TypeError(f"`DifferentialPrivacyDistinguisherOutputWriter` requires a DP wrapped `CanaryTrackingOptimizer`. "
                            f"Got {type(optimizer.original_optimizer)}")

        sensitivity = optimizer.max_grad_norm
        if optimizer.loss_reduction == "mean":
            sensitivity /= optimizer.expected_batch_size

        return cls(
            scores=optimizer.original_optimizer.observations,
            sensitivity=sensitivity,
        )

    @classmethod
    def load_from_disk(cls, path: Path) -> "DPDistinguishingData":
        with path.open("r") as f:
            data = load(f)
        return cls(**data)

    def save_to_disk(self, path: Path) -> None:
        with path.open("w+") as f:
            dump(asdict(self), f)
