import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from pathlib import Path
from warnings import warn

from privacy_estimates import convert_eps_deltas_to_fpr_fnr


class AbstractTradeOffCurve(ABC):
    @abstractmethod
    def append_to_fpr_fnr_plot(self, fig, ax):
        pass

    @abstractmethod
    def append_to_fpr_tpr_plot(self, fig, ax):
        pass


class TradeOffCurve(AbstractTradeOffCurve):
    def __init__(self, fpr: np.ndarray, fnr: np.ndarray, name: str):
        sort_idx = np.argsort(fpr)
        self.fpr = np.array(fpr, dtype=np.float64)[sort_idx]
        self.fnr = np.array(fnr, dtype=np.float64)[sort_idx]
        self.name = name
        assert self.fpr.shape == self.fnr.shape
        assert self.fpr.ndim == 1

    @classmethod
    def from_privacy_curve(cls, epsilons: np.ndarray, deltas: np.ndarray, name: str) -> "TradeOffCurve":
        fpr, fnr = convert_eps_deltas_to_fpr_fnr(epsilons, deltas)
        return cls(fpr=fpr, fnr=fnr, name=name)

    def all_fpr(self) -> np.ndarray:
        return self.fpr

    def interpolate_curve(self, fpr: np.ndarray) -> "TradeOffCurve":
        return TradeOffCurve(fpr=fpr, fnr=np.interp(x=fpr, xp=self.fpr, fp=self.fnr), name=self.name)

    def append_to_fpr_fnr_plot(self, fig, ax, plot_args=None):
        if plot_args is None:
            plot_args = {}
        ax.plot(self.fpr, self.fnr, label=self.name, **plot_args)
        ax.legend()
        return fig, ax

    def append_to_fpr_tpr_plot(self, fig, ax, plot_args=None):
        if plot_args is None:
            plot_args = {}
        ax.plot(self.fpr, 1 - self.fnr, label=self.name, **plot_args)
        ax.legend()
        return fig, ax

    def fnr_as_dict(self) -> Dict[str, List[float]]:
        return {self.name: self.fnr.tolist()}

    def __len__(self):
        return len(self.fpr)


class EmpiricalTradeOffCurve(AbstractTradeOffCurve):
    def __init__(self, lo: TradeOffCurve, hi: TradeOffCurve):
        self.lo = lo
        self.hi = hi

    @classmethod
    def from_privacy_curves(
        cls, epsilons_lo: np.ndarray, epsilons_hi: np.ndarray, deltas: np.ndarray, name: str
    ) -> "EmpiricalTradeOffCurve":
        lo = TradeOffCurve.from_privacy_curve(epsilons=epsilons_hi, deltas=deltas, name=f"{name}_lo")
        hi = TradeOffCurve.from_privacy_curve(epsilons=epsilons_lo, deltas=deltas, name=f"{name}_hi")
        return cls(lo=lo, hi=hi)

    def all_fpr(self) -> np.ndarray:
        return np.sort(np.concatenate([self.lo.fpr, self.hi.fpr]))

    def interpolate_curve(self, fpr: np.ndarray) -> "EmpiricalTradeOffCurve":
        return EmpiricalTradeOffCurve(
            lo=self.lo.interpolate_curve(fpr),
            hi=self.hi.interpolate_curve(fpr),
        )

    def append_to_fpr_fnr_plot(self, fig, ax):
        fig, ax = self.lo.append_to_fpr_fnr_plot(fig, ax, plot_args={"color": "blue", "linestyle": "-."})
        fig, ax = self.hi.append_to_fpr_fnr_plot(fig, ax, plot_args={"color": "blue", "linestyle": "--"})
        fpr = self.all_fpr()
        fnr_lo = self.lo.interpolate_curve(fpr).fnr
        fnr_hi = self.hi.interpolate_curve(fpr).fnr
        ax.fill_between(fpr, fnr_lo, fnr_hi, alpha=0.2, color="blue")
        return fig, ax

    def append_to_fpr_tpr_plot(self, fig, ax):
        fig, ax = self.lo.append_to_fpr_tpr_plot(fig, ax, plot_args={"color": "blue", "linestyle": "-."})
        fig, ax = self.hi.append_to_fpr_tpr_plot(fig, ax, plot_args={"color": "blue", "linestyle": "--"})
        fpr = self.all_fpr()
        tpr_hi = 1 - self.lo.interpolate_curve(fpr).fnr
        tpr_lo = 1 - self.hi.interpolate_curve(fpr).fnr
        ax.fill_between(tpr_lo, fpr, tpr_hi, alpha=0.2, color="blue")
        return fig, ax

    def fnr_as_dict(self) -> Dict[str, List[float]]:
        return {**self.lo.fnr_as_dict(), **self.hi.fnr_as_dict()}


class MIScoreDistribution:
    def __init__(self, scores: np.ndarray, challenge_bits: np.ndarray):
        self.scores = np.array(scores)
        self.challenge_bits = np.array(challenge_bits)

    def append_histogram_to_plot(self, fig, ax, bins: int = 10):
        ax.hist(self.scores[self.challenge_bits == 0], bins=bins, label="non-member")
        ax.hist(self.scores[self.challenge_bits == 1], bins=bins, label="member")
        ax.set_xlabel("MI score")
        ax.set_ylabel("Count")
        ax.legend()
        return fig, ax

    def append_kde_to_plot(self, fig, ax, bw_method: Union[str, float] = "scott"):
        from scipy.stats import gaussian_kde
        kde_nonmember = gaussian_kde(self.scores[self.challenge_bits == 0], bw_method=bw_method)
        kde_member = gaussian_kde(self.scores[self.challenge_bits == 1], bw_method=bw_method)
        x = np.linspace(np.min(self.scores), np.max(self.scores), 1000)
        ax.plot(x, kde_nonmember(x), label="non-member")
        ax.plot(x, kde_member(x), label="member")
        ax.set_xlabel("MI score")
        ax.set_ylabel("Density")
        ax.legend()
        return fig, ax


class PrivacyReport:
    def __init__(
            self, trade_off_curves: List[AbstractTradeOffCurve] = None,
            mi_score_distribution: MIScoreDistribution = None
        ):
        self.trade_off_curves: List[AbstractTradeOffCurve] = trade_off_curves or []
        self.mi_score_distribution: MIScoreDistribution = mi_score_distribution

    def add_trade_off_curve(self, curve: AbstractTradeOffCurve):
        self.trade_off_curves.append(curve)

    def set_scores_distribution(self, score_distribution: MIScoreDistribution):
        self.mi_score_distribution = score_distribution


class AbstractLogger(ABC):
    @abstractmethod
    def log(self, report: PrivacyReport):
        pass


class MatplotlibLogger(AbstractLogger):
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def log(self, report: PrivacyReport):
        import matplotlib.pyplot as plt

        if len(report.trade_off_curves) > 0:
            # Log trade off curves
            fig, ax = plt.subplots()
            for curve in report.trade_off_curves:
                fig, ax = curve.append_to_fpr_fnr_plot(fig, ax)
            ax.set_xlabel("FPR")
            ax.set_ylabel("FNR")            
            ax.set_aspect("equal")
            plt.savefig(self.path/"trade_off_curves.png")

            fig, ax = plt.subplots()
            for curve in report.trade_off_curves:
                fig, ax = curve.append_to_fpr_tpr_plot(fig, ax)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_aspect("equal")
            plt.savefig(self.path/"trade_off_curves_fpr_tpr.png")

            fig, ax = plt.subplots()
            for curve in report.trade_off_curves:
                fig, ax = curve.append_to_fpr_tpr_plot(fig, ax)
            ax.set_xlabel("TPR")
            ax.set_ylabel("FPR")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_aspect("equal")
            plt.savefig(self.path/"trade_off_curves_fpr_tpr_log.png")

        if report.mi_score_distribution is not None:
            # Log MI score histogram
            fig, ax = plt.subplots()
            fig, ax = report.mi_score_distribution.append_histogram_to_plot(fig, ax)
            plt.savefig(self.path/"mi_score_histogram.png")

            # Log MI score KDE
            fig, ax = plt.subplots()
            fig, ax = report.mi_score_distribution.append_kde_to_plot(fig, ax)
            plt.savefig(self.path/"mi_score_kde.png")


class AMLLogger(AbstractLogger):
    def __init__(self):
        try:
            from azureml.core import Run
            from azureml.exceptions import RunEnvironmentException
            self.run = Run.get_context(allow_offline=False)
        except RunEnvironmentException:
            self.run = None
        except ImportError:
            self.run = None

    def log(self, report: PrivacyReport):
        if self.run is None:
            warn("AMLLogger is not available. Please run this script in an AML experiment.")
            return

        # Create PNGs and log to images
        MatplotlibLogger(Path(".")).log(report)
        if len(report.trade_off_curves) > 0:
            self.run.log_image(name="trade_off_curves.png", path="trade_off_curves.png")
            self.run.log_image(name="trade_off_curves_fpr_tpr.png", path="trade_off_curves_fpr_tpr.png")
            self.run.log_image(name="trade_off_curves_fpr_tpr_log.png", path="trade_off_curves_fpr_tpr_log.png")
        if report.mi_score_distribution is not None:
            self.run.log_image(name="mi_score_histogram.png", path="mi_score_histogram.png")
            self.run.log_image(name="mi_score_kde.png", path="mi_score_kde.png")

        if len(report.trade_off_curves) > 0:
            # Log trade off curves as tables to metrics
            fpr = np.sort(np.concatenate([c.all_fpr() for c in report.trade_off_curves]))
            to_curves = [c.interpolate_curve(fpr) for c in report.trade_off_curves]

            data = {"fpr": fpr.tolist()}
            for curve in to_curves:
                data.update(curve.fnr_as_dict())

            self.log_table(name="trade_off_curves", value=data) 

    def log_table(self, name: str, value: Dict[str, List[float]]):
        if self.run is None:
            warn("AMLLogger is not available. Please run this script in an AML experiment.")
            return
        # check all lists of same length
        assert len(set([len(v) for v in value.values()])) == 1
        first_column = list(value.keys())[0]
        # batch logging to 250 rows at a time due to AML limits
        for batch_start in range(0, len(value[first_column]), 250):
            batch_end = min(batch_start + 250, len(value[list(value.keys())[0]]))
            batch_data = {k: v[batch_start:batch_end] for k, v in value.items()}
            self.run.log_table(name=name, value=batch_data)
            time.sleep(1)

        # wait until logging is completed
        timeout = 300
        start_time = time.time()
        while (
            len(self.run.get_metrics(name=name).get(name, {first_column: []})[first_column]) < len(value[first_column]) and
            time.time() - start_time < timeout
        ):
            time.sleep(10)
            print("Waiting for AML logging to complete...")


class PDFLogger(AbstractLogger):
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def log(self, report: PrivacyReport):
        MatplotlibLogger(Path(self.path)).log(report)
        from mdutils.mdutils import MdUtils

        md = MdUtils(file_name=self.path/"privacy_report.md")
        md.new_header(level=1, title="Privacy report")
        md.write(f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())}\n\n")
        run = AMLLogger().run
        url = run.get_portal_url() if run is not None else "Not available"
        md.write(f"AML URL: {url}\n\n")
        if len(report.trade_off_curves) > 0:
            md.new_header(level=2, title="Trade-off curves")
            md.new_line(md.new_inline_image(text="Trade-off curves", path=str(self.path/"trade_off_curves.png")))
            md.new_line(md.new_inline_image(text="Trade-off curves (FPR-TPR)",
                                            path=str(self.path/"trade_off_curves_fpr_tpr.png")))
            md.new_line(md.new_inline_image(text="Trade-off curves (FPR-TPR, log scale)",
                                            path=str(self.path/"trade_off_curves_fpr_tpr_log.png")))
            md.new_line("\n")
        if report.mi_score_distribution is not None:
            md.new_header(level=2, title="MI score distribution")
            md.new_line(md.new_inline_image(text="MI score histogram", path=str(self.path/"mi_score_histogram.png")))
            md.new_line(md.new_inline_image(text="MI score KDE", path=str(self.path/"mi_score_kde.png")))
            md.new_line("\n")

        # convert markdown to html
        from markdown2 import markdown
        html = markdown(md.get_md_text())

        # convert html to pdf
        import pdfkit
        pdfkit.from_string(html, self.path/"privacy_report.pdf")
