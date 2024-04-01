from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path


class Arguments(BaseModel):
    challenge_points: Path
    shadow_model_statistics: Path
    scores: Path


def main(args: Arguments) -> int:
    reference_models    

    

def exception_handler(ex):
    raise RuntimeError("An error occurred while running the script.") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, exception_handler=exception_handler)
