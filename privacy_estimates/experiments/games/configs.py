from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PrivacyEstimationConfig:
    smallest_delta: float = None


@dataclass
class MISignalConfig:
    method: str
    aggregation: Optional[str] = None
    extra_args: Optional[Dict] = None

