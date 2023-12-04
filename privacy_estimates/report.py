import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from pathlib import Path
from tempfile import TemporaryDirectory

from privacy_estimates import convert_eps_deltas_to_fpr_fnr


class AbstractTradeOffCurve(ABC):
    @abstractmethod
    def append_to_plot(self, fig, ax):
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
    
    def append_to_plot(self, fig, ax, plot_args=None):
        if plot_args is None:
            plot_args = {}
        ax.plot(self.fpr, self.fnr, label=self.name, **plot_args)
        return fig, ax

    def fnr_as_dict(self) -> Dict[str, List[float]]:
        return {self.name: self.fnr.tolist()}

    def __len__(self):
        return len(self.fpr)


class TradeOffCurveBounds(AbstractTradeOffCurve):
    def __init__(self, lo: TradeOffCurve, hi: TradeOffCurve):
        self.lo = lo
        self.hi = hi

    @classmethod
    def from_privacy_curves(cls, epsilons_lo: np.ndarray, epsilons_hi: np.ndarray, deltas: np.ndarray, name: str) -> "TradeOffCurveBounds":
        lo = TradeOffCurve.from_privacy_curve(epsilons=epsilons_hi, deltas=deltas, name=f"{name}_lo")
        hi = TradeOffCurve.from_privacy_curve(epsilons=epsilons_lo, deltas=deltas, name=f"{name}_hi")
        return cls(lo=lo, hi=hi)
    
    def all_fpr(self) -> np.ndarray:
        return np.sort(np.concatenate([self.lo.fpr, self.hi.fpr]))
    
    def interpolate_curve(self, fpr: np.ndarray) -> "TradeOffCurveBounds":
        return TradeOffCurveBounds(
            lo=self.lo.interpolate_curve(fpr),
            hi=self.hi.interpolate_curve(fpr),
        )

    def append_to_plot(self, fig, ax):
        fig, ax = self.lo.append_to_plot(fig, ax, plot_args={"color": "blue", "linestyle": "-."})
        fig, ax = self.hi.append_to_plot(fig, ax, plot_args={"color": "blue", "linestyle": "--"})
        fpr = self.all_fpr()
        fnr_lo = self.lo.interpolate_curve(fpr).fnr
        fnr_hi = self.hi.interpolate_curve(fpr).fnr
        ax.fill_between(fpr, fnr_lo, fnr_hi, alpha=0.2, color="blue")
        return fig, ax
    
    def fnr_as_dict(self) -> Dict[str, List[float]]:
        return {**self.lo.fnr_as_dict(), **self.hi.fnr_as_dict()}
    

class PrivacyReport:
    def __init__(self, trade_off_curves: List[AbstractTradeOffCurve] = None):
        self.trade_off_curves: List[AbstractTradeOffCurve] = trade_off_curves or []

    def add_trade_off_curve(self, curve: AbstractTradeOffCurve):
        self.trade_off_curves.append(curve)


class AbstractLogger(ABC):
    @abstractmethod
    def log(self, report: PrivacyReport):
        pass


class MatplotlibLogger(AbstractLogger):
    def __init__(self, path: Path):
        self.path = path

    def log(self, report: PrivacyReport):
        import matplotlib.pyplot as plt

        # Log trade off curves
        fig, ax = plt.subplots()
        for curve in report.trade_off_curves:
            fig, ax = curve.append_to_plot(fig, ax)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("False negative rate")
        ax.legend()
        plt.savefig(self.path/"trade_off_curves.png")


class AMLLogger(AbstractLogger):
    def __init__(self):
        from azureml.core import Run
        self.run = Run.get_context()

    def log(self, report: PrivacyReport):
        # Create PNGs and log to images
        MatplotlibLogger(Path(".")).log(report)
        self.run.log_image(name="trade_off_curves.png", path="trade_off_curves.png")

        # Log trade off curves as tables to metrics
        fpr = np.sort(np.concatenate([c.all_fpr() for c in report.trade_off_curves]))
        to_curves = [c.interpolate_curve(fpr) for c in report.trade_off_curves]

        data = {"fpr": fpr.tolist()}
        for curve in to_curves:
            data.update(curve.fnr_as_dict())
        
        # batch logging to 250 rows at a time due to AML limits
        for batch_start in range(0, len(fpr), 250):
            batch_end = min(batch_start + 250, len(fpr))
            batch_data = {k: v[batch_start:batch_end] for k, v in data.items()}
            self.run.log_table(
                name="trade_off_curves",
                value=batch_data,
            )
            time.sleep(1)
        
        # wait until logging is completed
        timeout = 300 # seconds
        start_time = time.time()
        while (
            len(self.run.get_metrics(name="trade_off_curves")["trade_off_curves"]["fpr"]) < len(fpr) and
            time.time() - start_time < timeout
        ):
            time.sleep(10)
            print("Waiting for AML logging to complete...")



class PDFLogger(AbstractLogger):
    def __init__(self, path: Path):
        self.path = path

    def log(self, report: PrivacyReport):
        raise NotImplementedError("PDFLogger not implemented yet")
