import pytest
from privacy_games.estimates import compute_eps_lo, AttackResults, compute_eps_lo_hi, compute_eps_hi
from numpy.testing import assert_array_almost_equal


def test_eps_lo_order():
    count = AttackResults(FN=12, FP=58, TN=200, TP=200)
    alpha = 0.05
    delta = 1e-5
    with_beta = compute_eps_lo(count=count, delta=delta, alpha=alpha, method="beta")
    with_joint_beta = compute_eps_lo(count=count, delta=delta, alpha=alpha, method="joint-beta")
    assert with_beta < with_joint_beta

def test_eps_hi_order():
    count = AttackResults(FN=511, FP=0, TN=487, TP=2)
    alpha = 0.05
    delta = 1e-5
    with_beta = compute_eps_hi(count=count, delta=delta, alpha=alpha, method="beta")
    with_joint_beta = compute_eps_hi(count=count, delta=delta, alpha=alpha, method="joint-beta")
    assert with_joint_beta < with_beta

def test_eps_hi_order_2():
    count = AttackResults(TP=0, TN=510, FP=1, FN=489)
    alpha = 0.05
    delta = 1e-5
    with_beta = compute_eps_hi(count=count, delta=delta, alpha=alpha, method="beta")
    with_joint_beta = compute_eps_hi(count=count, delta=delta, alpha=alpha, method="joint-beta")
    assert with_joint_beta < with_beta

@pytest.mark.parametrize("method", ["beta", "jeffreys", "joint-beta"])
def test_onesided_twosided_estimates(method):
    """
    Check that the equal tailed two sided estimates are equal to the one sided estimates of half the confidence
    """
    count = AttackResults(FN=923, FP=19, TN=982, TP=76)
    alpha = 0.05
    delta = 1e-5
    eps_lo_alpha = compute_eps_lo(count=count, delta=delta, alpha=alpha, method=method)
    eps_lo_hi_2alpha = compute_eps_lo_hi(count=count, delta=delta, alpha=2*alpha, method=method)
    assert eps_lo_alpha == pytest.approx(eps_lo_hi_2alpha[0])

@pytest.mark.parametrize("method", ["beta", "jeffreys", "joint-beta"])
def test_symmetry(method):
    FN = 12
    FP = 58
    TN = 200
    TP = 200
    alpha = 0.05
    delta = 1e-5
    eps_lo_0, eps_hi_0 = compute_eps_lo_hi(count=AttackResults(FN=FN, FP=FP, TN=TN, TP=TP),
                                           delta=delta, alpha=alpha, method=method)
    # Test symmetry accross y = x
    eps_lo_1, eps_hi_1 = compute_eps_lo_hi(count=AttackResults(FN=FP, FP=FN, TN=TP, TP=TN),
                                           delta=delta, alpha=alpha, method=method)
    assert_array_almost_equal([eps_lo_0, eps_hi_0], [eps_lo_1, eps_hi_1])
    # Test symmetry accross y = 1-x
    eps_lo_2, eps_hi_2 = compute_eps_lo_hi(count=AttackResults(FN=TP, FP=TN, TN=FP, TP=FN),
                                           delta=delta, alpha=alpha, method=method)
    assert_array_almost_equal([eps_lo_0, eps_hi_0], [eps_lo_2, eps_hi_2])

def test_beta():
    eps_lo, eps_hi = compute_eps_lo_hi(count=AttackResults(FN=10, FP=10, TN=10, TP=10),
                                       delta=0.0, alpha=0.05, method="beta")
    assert eps_lo < eps_hi
