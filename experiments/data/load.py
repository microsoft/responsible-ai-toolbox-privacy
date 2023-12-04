from pathlib import Path
import pandas as pd
from typing import Tuple, List


def load_trial_dir(base_path: Path, model_index: int) -> Path:
    print(f"Loading trial dir: {base_path}")
    print(f"trial dir content: {list(base_path.iterdir())}")
    paths = list(base_path.glob(f"run_trial*/run_trial*model_index={model_index}*"))
    print(f"Found paths: {paths}")
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one path, got {paths}")
    return paths[0]
