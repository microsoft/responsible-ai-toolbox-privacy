from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from typing import Optional
from pathlib import Path
from azureml.core import Run

from privacy_games.estimates import AttackResults, compute_eps_lo_hi


class Arguments(BaseModel):
    delta: float = Field(description="The DP delta value to use")

    results_file: Optional[Path] = Field(None, description="Path to the results file")
    TN: Optional[int] = Field(None, description="Number of true negatives")
    FP: Optional[int] = Field(None, description="Number of false positives")
    FN: Optional[int] = Field(None, description="Number of false negatives")
    TP: Optional[int] = Field(None, description="Number of true positives")

    alpha: Optional[float] = Field(default=0.05, description="Significance level of the estimate")


def main(args: Arguments):
    run: Run = Run.get_context()

    if args.results_file is not None:
        results = AttackResults.from_json(args.results_file)
    elif (
        args.TN is not None and args.TP is not None and
        args.FN is not None and args.FP is not None
    ):
        results = AttackResults(FN=args.FN, FP=args.FP, TN=args.TN, TP=args.TP)
    else:
        raise ValueError("Must provide either results file or TN, TP, FN, FP")

    eps_lo, eps_hi = compute_eps_lo_hi(count=results, delta=args.delta, alpha=args.alpha, method="joint-beta")
    run.log(f"eps_lo_{args.alpha}", value=eps_lo, description="Lower bound of the equal tailed confidence interval.")
    print(f"eps_lo_{args.alpha} = {eps_lo}")
    run.log(f"eps_hi_{args.alpha}", value=eps_hi, description="Upper bound of the equal tailed confidence interval.")
    print(f"eps_hi_{args.alpha} = {eps_hi}")


if __name__ == "__main__":
    run_and_exit(Arguments, main)
