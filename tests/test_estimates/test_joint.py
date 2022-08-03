import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from privacy_games.estimates import AttackResults, joint_density


def test_eps_lo_joint():
    count = AttackResults(FN=0, FP=0, TN=1000, TP=1000)
    alpha = 0.05
    delta = 1e-5
    eps = joint_density.Beta(count).eps_lo(delta=delta, alpha=alpha)
    assert_allclose(eps, 7.21, atol=1e-2)

def test_pdf_deriv_of_cdf():
    count = AttackResults(FN=10, FP=10, TN=10, TP=10)
    delta = 1e-9

    model = joint_density.Beta(count)

    epsilons = np.linspace(0, 0.1, 100)

    cdf = np.array([model.probability_private(eps, delta=delta) for eps in epsilons])
    pdf = np.array([model.eps_pdf(eps, delta=delta) for eps in epsilons])

    assert_array_almost_equal(np.diff(cdf)/(epsilons[1]-epsilons[0]), pdf)
