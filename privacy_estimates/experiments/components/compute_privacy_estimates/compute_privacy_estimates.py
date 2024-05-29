import numpy as np

from datasets import Dataset
from pathlib import Path
from json import load
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
from typing import Tuple, Optional
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from sklearn.metrics import roc_curve

from privacy_estimates import report as priv_report, compute_privacy_curve_lo_hi


class Arguments(BaseModel):
    scores: Path
    challenge_bits: Path
    privacy_report: Path
    smallest_delta: float
    alpha: float
    dp_parameters: Optional[Path] = None


class DPParameters(BaseModel):
    noise_multiplier: float
    subsampling_probability: float
    num_steps: int


def setup_accountant(dp_parameters_path: Path) -> Tuple[PRVAccountant, int]:
    with Path(dp_parameters_path).open() as f:
        # Load the dp parameters from a json and parse them into a DPParameters object
        dp_parameters = DPParameters(**(load(f)))

    prv = PoissonSubsampledGaussianMechanism(noise_multiplier=dp_parameters.noise_multiplier,
                                             sampling_probability=dp_parameters.subsampling_probability)
    accountant = PRVAccountant(prv, max_self_compositions=dp_parameters.num_steps, eps_error=0.1, delta_error=1e-9)
    return accountant, dp_parameters.num_steps


def main(args: Arguments) -> int:
    report = priv_report.PrivacyReport()

    scores_ds = Dataset.load_from_disk(str(args.scores))
    challenge_bits_ds = Dataset.load_from_disk(str(args.challenge_bits))

    # Add MI score distribution
    report.mi_score_distribution = priv_report.MIScoreDistribution(
        scores=scores_ds["score"], challenge_bits=challenge_bits_ds["challenge_bit"]
    )

    # Compute theoretical privacy guarantees
    if args.dp_parameters is not None:
        accountant, num_steps = setup_accountant(args.dp_parameters)

        fprs_th, fnrs_th = accountant.compute_trade_off_curve(num_self_compositions=[num_steps], bound="estimate")
        to_curve_th = priv_report.TradeOffCurve(fpr=fprs_th, fnr=fnrs_th, name="theoretical")
        report.add_trade_off_curve(to_curve_th)

    # Compute empirical privacy guarantees
    deltas = np.logspace(np.log10(args.smallest_delta), 0, num=100, endpoint=False)
    epsilons_lo, epsilons_hi = compute_privacy_curve_lo_hi(
        attack_scores=scores_ds["score"], challenge_bits=challenge_bits_ds["challenge_bit"], alpha=args.alpha, method="beta",
        deltas=deltas
    )
    to_curve_emp = priv_report.EmpiricalTradeOffCurve.from_privacy_curves(
        epsilons_lo=epsilons_lo, epsilons_hi=epsilons_hi, deltas=deltas, name=f"empirical_{1-args.alpha}"
    )
    report.add_trade_off_curve(to_curve_emp)

    # Add ROC curve
    fpr, tpr, _ = roc_curve(challenge_bits_ds["challenge_bit"], scores_ds["score"])
    report.add_trade_off_curve(priv_report.TradeOffCurve(fpr=fpr, fnr=1-tpr, name="roc"))

    # Output
    loggers = [
        priv_report.PDFLogger(path="."),
        priv_report.AMLLogger(),
        priv_report.MatplotlibLogger(path=args.privacy_report),
        priv_report.PDFLogger(path=args.privacy_report),
    ]

    for logger in loggers:
        logger.log(report=report)

    return 0


def exception_handler(exception: Exception) -> int:
    raise RuntimeError("An exception occurred") from exception


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
