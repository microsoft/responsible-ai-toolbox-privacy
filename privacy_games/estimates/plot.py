import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from .privacy_region import privacy_boundary_lo, privacy_boundary_hi

def plot_eps_equations(fnr: float, fpr: float, delta: float):
    FPR = np.linspace(0.00001, 1.0, 1000)

    plt.figure(figsize=(10,8))

    plt.xlabel("FPR")
    plt.ylabel(r"$\epsilon_{\mathrm{lo}}$")
    plt.xticks(np.linspace(0, 1, 11))

    plt.xlim(-0.1,1.1)
    plt.ylim(-4,4)

    plt.plot(FPR, np.log((1 - delta - FPR) / fnr), label=r"$\log \frac{1 - \delta - FPR}{FNR}$", color='red')
    plt.plot(FPR, np.log((1 - delta - FPR) / FPR), label=r"$\log \frac{1 - \delta - FNR}{FPR}$", color='blue')
    #plt.plot(fpr, np.maximum(np.zeros_like(fpr), np.log((1 - delta - fpr) / fnr), np.log((1 - delta - fnr) / fpr)), label=r"max", color='green')

    plt.axhline(y = 0, linestyle=':', color='black')
    plt.axvline(x = 0, linestyle=':', color='black')
    plt.axvline(x = 1 - delta, linestyle='--', color="gray")
    plt.axvline(x = 1 - delta - fnr, linestyle='--', color="gray")
    plt.axvline(x = fnr, linestyle='--', color="gray")
    plt.axvline(x = fpr, linestyle=':', color="green")

    plt.text(x = 1 - delta, y = -1, s=r"$1 - \delta$")
    plt.text(x = 1 - delta - fnr, y = -1, s=r"$1 - \delta - FNR$")
    plt.text(x = fnr, y = -1, s=r"FNR")
    plt.text(x = fpr, y = -1, s=r"FPR", color='red')

    if 0 < fnr:
        plt.text(x = -0.1, y = np.log((1 - delta) / fnr), s=r"$\log\frac{1 - \delta}{FNR}$")

    plt.tight_layout()
    plt.legend()


def plot_privacy_region(eps, delta, color='green', hatch='//'):
    fnr = np.linspace(0, 1, 1000)

    # Region below the fpr = 1 - fnr line
    y1 = privacy_boundary_lo(fnr=fnr, eps=eps, delta=delta)

    # Region above the fpr = 1 - fnr line
    y2 = privacy_boundary_hi(fnr=fnr, eps=eps, delta=delta)

    plt.fill_between(fnr, y1, y2, alpha=0.2, hatch=hatch, color=color, label=rf"Privacy region for $(\epsilon={eps:.3f},\delta={delta})$-DP")


def plot_pdf(pdf: Callable[[float, float], float]):
    x = np.linspace(0, 1, 50)[1:-1]
    y = np.linspace(0, 1, 50)[1:-1]
    X, Y = np.meshgrid(x, y)
    z = np.vectorize(pdf)(X, Y)
    ctr = plt.contour(x, y, z, levels=10)
    #fig.colorbar(ctr)

    
def plot_intervals(fnr_lo, fnr_hi, fpr_lo, fpr_hi):
    plt.fill([fnr_lo, fnr_hi, fnr_hi, fnr_lo], [fpr_lo, fpr_lo, fpr_hi, fpr_hi], color='blue', alpha=0.3)
    plt.axhline(y=0, xmin=fnr_lo, xmax=fnr_hi, color='blue', linewidth=8, alpha=0.3)
    plt.axvline(x=0, ymin=fpr_lo, ymax=fpr_hi, color='blue', linewidth=8, alpha=0.3)
