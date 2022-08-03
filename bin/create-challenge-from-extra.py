import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from sklearn.model_selection import train_test_split
from typing import Optional


class Arguments(BaseModel):
    base_data_path: Path = Field(
        description="Path to the base dataset"
    )
    extra_data_path: Path = Field(
        description="Path to the extra dataset from which `N` challenge points will be drawn"
    )
    N: int = Field(
        description="Number of challenge points to create"
    )
    seed: int = Field(
        default=12391, description="Random seed"
    )
    challenge_bits_path: Optional[Path] = Field(
        description="Output path where the challenge bits will be written. If the challenge bit is 0 then the sample comes from the base data."
    )
    train_base_data_path: Optional[Path] = Field(
        description="Output path where the base of the training dataset will be written. Note that first_samples + train_base data = base_data"
    )
    in_samples_path: Optional[Path] = Field(
        description="Output path where the challenge points which are part of the training set will be written"
    )
    out_samples_path: Optional[Path] = Field(
        description="Output path where the challenge points which aren't part of the training set will be written"
    )
    first_samples_path: Optional[Path] = Field(
        description="Output path where the first challenge points will be written"
    )
    second_samples_path: Optional[Path] = Field(
        description="Output path where the second challenge points will be written"
    )


def main(args: Arguments):
    base_df = pd.read_parquet(args.base_data_path)
    extra_df = pd.read_parquet(args.extra_data_path)

    rng = np.random.default_rng(args.seed)
    challenge_bits = rng.integers(low=0, high=2, size=args.N)

    assert len(base_df) >= args.N
    assert len(extra_df) >= args.N

    print(f"Columns of base_df: {base_df.columns}")
    print(f"Types of base_df: {base_df.dtypes}")
    print(f"Columns of extra_df: {extra_df.columns}")
    print(f"Types of extra_df: {extra_df.dtypes}")
    columns = base_df.columns.intersection(extra_df.columns)
    print(f"Columns of intersection: {columns}")
    base_df = base_df[columns]
    extra_df = extra_df[columns]

    assert base_df.dtypes.equals(extra_df.dtypes)
    dtypes = base_df.dtypes

    train_base_df, first_df = train_test_split(base_df, test_size=args.N, random_state=args.seed)
    second_df = extra_df.sample(n=args.N, random_state=args.seed, replace=False)

    train_base_df = train_base_df.reset_index(drop=True)
    first_df = first_df.reset_index(drop=True)
    second_df = second_df.reset_index(drop=True)

    from_first_mask = np.tile(challenge_bits==0, (len(columns), 1)).transpose()
    in_df = first_df.where(from_first_mask, other=second_df).astype(dtypes)
    out_df = first_df.where(~from_first_mask, other=second_df).astype(dtypes)

    if args.challenge_bits_path:
        pd.DataFrame({"challenge_bits": challenge_bits}).to_parquet(args.challenge_bits_path, index=False)
    if args.train_base_data_path:
        print(f"Columns of train_base_df: {train_base_df.columns}")
        print(f"Types of train_base_df: {train_base_df.dtypes}")
        train_base_df.to_parquet(args.train_base_data_path, index=False)
    if args.first_samples_path:
        print(f"Columns of first_df: {first_df.columns}")
        print(f"Types of first_df: {first_df.dtypes}")
        first_df.to_parquet(args.first_samples_path, index=False)
    if args.second_samples_path:
        print(f"Columns of second_df: {second_df.columns}")
        print(f"Types of second_df: {second_df.dtypes}")
        second_df.to_parquet(args.second_samples_path, index=False)
    if args.in_samples_path:
        print(f"Columns of in_df: {in_df.columns}")
        print(f"Types of in_df: {in_df.dtypes}")
        in_df.to_parquet(args.in_samples_path, index=False)
    if args.out_samples_path:
        print(f"Columns of out_df: {out_df.columns}")
        print(f"Types of out_df: {out_df.dtypes}")
        out_df.to_parquet(args.out_samples_path, index=False)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
