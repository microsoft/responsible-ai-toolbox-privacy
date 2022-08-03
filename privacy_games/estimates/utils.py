import numpy as np


def bound_membership_advantage(eps: float, delta: float) -> float:
    """
    Compute the bound on the membership advantage.
    """
    return (np.exp(eps)-1.0+2.0*delta)/(np.exp(eps)+1.0)
