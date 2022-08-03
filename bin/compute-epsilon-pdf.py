import numpy as np
import parmap
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from typing import Optional
from pathlib import Path

from privacy_games.estimates import AttackResults, Binomials, compute_eps_lo, compute_eps_hi


class Arguments(BaseModel):
    delta: float = Field(description="The DP delta value to use")
    output_path: Path

    results_file: Optional[Path] = Field(None, description="Path to the results file")
    TN: Optional[int] = Field(None, description="Number of true negatives")
    FP: Optional[int] = Field(None, description="Number of false positives")
    FN: Optional[int] = Field(None, description="Number of false negatives")
    TP: Optional[int] = Field(None, description="Number of true positives")

    eps_lo: Optional[float] = Field(None, description="Lower bound on epsilon. If not provided lower bound with alpha=0.01 is used")
    eps_hi: Optional[float] = Field(None, description="Upper bound on epsilon. If not provided upper bound with alpha=0.01 is used")
    num_points: int = Field(100, description="Number of points to use for discretisation")


def main(args: Arguments):
    if args.results_file is not None:
        results = AttackResults.from_json(args.results_file)
    elif (
        args.TN is not None and args.TP is not None and
        args.FN is not None and args.FP is not None
    ):
        results = AttackResults(FN=args.FN, FP=args.FP, TN=args.TN, TP=args.TP)
    else:
        raise ValueError("Must provide either results file or TN, TP, FN, FP")

    if args.eps_lo is None:
        eps_lo = compute_eps_lo(
            count=results,
            delta=args.delta,
            alpha=0.01,
            method="beta",
        )
    else:
        eps_lo = args.eps_lo

    if args.eps_hi is None:
        eps_hi = compute_eps_hi(
            count=results,
            delta=args.delta,
            alpha=0.01,
            method="beta",
        )
    else:
        eps_hi = args.eps_hi

    epss = np.linspace(eps_lo, eps_hi, args.num_points) 
    m = Binomials(count=results)
    pdf = parmap.map(lambda e: m.probability_private(eps=e, delta=args.delta), epss)

    pd.DataFrame({"epsilon": epss, "pdf": pdf}).to_csv(args.output_path, sep="\t", index=False)


if __name__ == "__main__":
    run_and_exit(Arguments, main)
