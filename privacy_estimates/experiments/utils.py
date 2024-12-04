from dataclasses import dataclass, asdict
from pathlib import Path
from json import dump, load
from urllib.parse import urlparse


@dataclass
class DPParameters:
    noise_multiplier: float
    num_steps: int
    subsampling_probability: float

    def save_to_disk(self, path: Path) -> None:
        with Path(path).open("w+") as f:
            dump(asdict(self), f)

    @classmethod
    def load_from_disk(cls, path: Path) -> "DPParameters":
        with Path(path).open("r") as f:
            data = load(f)
        return cls(**data)

    @classmethod
    def from_opacus(cls, privacy_engine: "opacus.PrivacyEngine") -> "DPParameters":  # noqa: F821
        history = {}
        for nm, p, c in privacy_engine.accountant.history:
            history[(nm, p)] = history.get((nm, p), 0) + c
        if len(history) > 1:
            raise ValueError("Currently only supports homogeneous composition")
        nm = list(history.keys())[0][0]
        p = list(history.keys())[0][1]
        n_steps = list(history.values())[0]

        return cls(
            noise_multiplier=nm,
            num_steps=n_steps,
            subsampling_probability=p,
        )


def is_url(path: str) -> bool:
    """Check if the given path is a valid URL."""
    try:
        result = urlparse(path)
        # Ensure scheme is either 'http' or 'https' and there is a network location
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
