from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from typing import Optional
from pathlib import Path

from privacy_estimates import AttackResults, compute_eps_lo_hi


class Arguments(BaseModel):
    delta: float = Field(description="The DP delta value to use")

    results_file: Optional[Path] = Field(None, description="Path to the results file")
    TN: Optional[int] = Field(None, description="Number of true negatives")
    FP: Optional[int] = Field(None, description="Number of false positives")
    FN: Optional[int] = Field(None, description="Number of false negatives")
    TP: Optional[int] = Field(None, description="Number of true positives")

    alpha: Optional[float] = Field(default=0.05, description="Significance level of the estimate for two sided intervals")


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

    eps_lo_jb, eps_hi_jb = compute_eps_lo_hi(count=results, delta=args.delta, alpha=args.alpha, method="joint-beta")
    eps_lo_b, eps_hi_b = compute_eps_lo_hi(count=results, delta=args.delta, alpha=args.alpha, method="beta")
    eps_lo_j, eps_hi_j = compute_eps_lo_hi(count=results, delta=args.delta, alpha=args.alpha, method="jeffreys")
    print( "Method             Interval                Significance level  eps_lo  eps_hi")
    print(f"Joint beta (ours)  two-sided equal-tailed  {args.alpha:.3f}               {eps_lo_jb:.3f}   {eps_hi_jb:.3f}")
    print(f"Joint beta (ours)  one-sided               {args.alpha/2:.3f}               {eps_lo_jb:.3f}   inf")
    print(f"Clopper Pearson    two-sided equal-tailed  {args.alpha:.3f}               {eps_lo_b:.3f}   {eps_hi_b:.3f}")
    print(f"Clopper Pearson    one-sided               {args.alpha/2:.3f}               {eps_lo_b:.3f}   inf")
    print(f"Jeffreys           two-sided equal-tailed  {args.alpha:.3f}               {eps_lo_j:.3f}   {eps_hi_j:.3f}")
    print(f"Jeffreys           one-sided               {args.alpha/2:.3f}               {eps_lo_j:.3f}   inf")


if __name__ == "__main__":
    run_and_exit(Arguments, main)
