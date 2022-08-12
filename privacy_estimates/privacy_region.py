import numpy as np


def privacy_boundary_lo(fnr: float, eps: float, delta: float) -> float:
    """
    Boundary of the (eps,delta) privacy region below the fpr = 1 - fnr line
    """
    return np.maximum(0, np.maximum(1 - delta - fnr * np.exp(eps),
                                    (1 - delta - fnr) * np.exp(-eps)))


def privacy_boundary_hi(fnr: float, eps: float, delta: float) -> float:
    """
    Boundary of the (eps,delta) privacy region above the fpr = 1 - fnr line
    """
    return np.minimum(1, np.minimum(1 + (delta - fnr) * np.exp(-eps),
                                    delta + (1 - fnr) * np.exp(eps)))


def privacy_boundary_normal_lo(fnr: float, eps: float, delta: float) -> float:
    """
    Normal vector to the (eps,delta) privacy region below the fpr = 1 - fnr line.
    Bounded by `privacy_boundary_lo`
    """
    n = np.array([np.where(
        fnr > 1-delta, 0.0, np.where(
            fnr < (1-delta)/(1+np.exp(eps)), -np.exp(eps), -np.exp(-eps)
        )
    ), -1.0])
    return n/np.linalg.norm(n)


def privacy_velocity_lo(fnr: float, eps: float, delta: float) -> float:
    """
    Velocity of the (eps,delta) privacy region below the fpr = 1 - fnr line
    """
    return np.array([0.0, np.where(fnr > 1-delta, 0.0, np.where(
        fnr < (1-delta)/(1+np.exp(eps)), -fnr*np.exp(eps), -(1-delta-fnr)*np.exp(-eps)
    ))])


def privacy_boundary_normal_hi(fnr: float, eps: float, delta: float) -> float:
    """
    Normal vector to the (eps,delta) privacy region above the fpr = 1 - fnr line.
    Bounded by `privacy_boundary_hi`
    """
    n = np.array([-np.where(
        fnr < delta, 0.0, np.where(
            fnr < 1 - (1-delta)/(1+np.exp(eps)), -np.exp(-eps), -np.exp(eps)
        )
    ), 1.0])
    return n/np.linalg.norm(n)


def privacy_velocity_hi(fnr: float, eps: float, delta: float) -> float:
    """
    Velocity of the (eps,delta) privacy region above the fpr = 1 - fnr line
    """
    return np.array([0.0, np.where(fnr < delta, 0.0, np.where(
        fnr < 1 - (1-delta)/(1+np.exp(eps)), (fnr-delta)*np.exp(-eps), (1-fnr)*np.exp(eps)
    ))])


def eps_from_fnr_fpr(fnr: float, fpr: float, delta: float) -> float:
    """
    Compute epsilon such that (fnr, fpr) is on the boundary of the (epsilon, delta)-DP region.

    Args:
        fnr (float): False positive rate of the membership inference attack
        fpr (float): False negative rate of the membership inference attack
        delta (float): Differential privacy :math:`\delta` parameter.
    Returns:
        epsilon (float): differential privacy :math:`epsilon` parameter
    """
    if fpr < 0 or fnr < 0:
        raise ValueError(f"false_positive_rate={fpr} and false_negative_rate={fnr} must be non-negative")

    if delta < 0 or delta >= 1:
        raise ValueError(f"delta={delta} must be positive and strictly less than 1")

    # Flip across 1-x line if point is in upper right quadrant
    if fpr > 1 - fnr:
        fpr, fnr = 1-fnr, 1-fpr

    # Bring point into the left quadrant
    r_lo = min(fpr, fnr)
    r_hi = max(fpr, fnr)
 
    if r_hi > 1-delta-r_lo:
        return 0.0

    if r_lo == 0.0:
        return np.inf

    return np.log((1-delta-r_hi)/r_lo)
