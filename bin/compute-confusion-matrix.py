import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from privacy_games.estimates import AttackResults


class Arguments(BaseModel):
    attack_guesses_path: Path = Field(
        description="Path to the base dataset"
    )
    challenge_bits_path: Path = Field(
        description="Path to the extra dataset from which `N` challenge points will be drawn"
    )
    output_path: Path = Field(
        description="Output path where the challenge points which aren't part of the training set will be written"
    )


def main(args: Arguments):
    attack_guesses = pd.read_parquet(args.attack_guesses_path)
    challenge_bits = pd.read_parquet(args.challenge_bits_path)

    print("attack_guesses:")
    print(attack_guesses)
    print("challenge_bits:")
    print(challenge_bits)

    results = AttackResults.from_guesses_and_labels(attack_guesses=attack_guesses["guesses"], challenge_bits=challenge_bits["challenge_bits"])

    results.to_json(path=args.output_path)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
