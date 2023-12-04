import numpy as np
import time

from datasets import Dataset
from pathlib import Path
from json import load
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
from typing import Tuple, Optional
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from sklearn.metrics import roc_curve as compute_roc_curve

from privacy_estimates import report as priv_report, compute_privacy_curve_lo_hi


class Arguments(BaseModel):
    scores: Path
    challenge_bits: Path
    privacy_report: Path
    dp_parameters: Optional[Path] = None
    target_delta: Optional[float] = None
    alpha: float = 0.05


def setup_accountant(dp_parameters: Path) -> Tuple[PRVAccountant, int]:
    with Path(dp_parameters).open() as f:
        dp_parameters = load(f)
    prv = PoissonSubsampledGaussianMechanism(noise_multiplier=dp_parameters["noise_multiplier"],
                                             sampling_probability=dp_parameters["subsampling_probability"])
    accountant = PRVAccountant(prv, max_self_compositions=dp_parameters["num_steps"], eps_error=0.1, delta_error=1e-9)
    return accountant, dp_parameters["num_steps"]


def main(args: Arguments) -> int:
    report = priv_report.PrivacyReport()

    scores_ds = Dataset.load_from_disk(args.scores)
    challenge_bits_ds = Dataset.load_from_disk(args.challenge_bits)

    accountant, num_steps = setup_accountant(args.dp_parameters)

    # Compute theoretical privacy guarantees
    fprs_th, fnrs_th = accountant.compute_trade_off_curve(num_self_compositions=[num_steps], bound="estimate")
    to_curve_th = priv_report.TradeOffCurve(fpr=fprs_th, fnr=fnrs_th, name="theoretical")
    report.add_trade_off_curve(to_curve_th)

    # Compute empirical privacy guarantees
    deltas = np.logspace(np.log10(args.target_delta), 0, num=100, endpoint=False)
    epsilons_lo, epsilons_hi = compute_privacy_curve_lo_hi(
        attack_scores=scores_ds["scores"], challenge_bits=challenge_bits_ds["challenge_bits"], alpha=args.alpha, method="beta",
        deltas=deltas
    )
    to_curve_emp = priv_report.TradeOffCurveBounds.from_privacy_curves(
        epsilons_lo=epsilons_lo, epsilons_hi=epsilons_hi, deltas=deltas, name=f"empirical_{1-args.alpha}"
    )
    report.add_trade_off_curve(to_curve_emp)

    # Output
    loggers = [
        priv_report.AMLLogger(),
        priv_report.MatplotlibLogger(path=args.privacy_report),
#        priv_report.PDFLogger(path=args.privacy_report/"privacy_report.pdf"), Not yet implemented
    ]

    for logger in loggers:
        logger.log(report=report)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
