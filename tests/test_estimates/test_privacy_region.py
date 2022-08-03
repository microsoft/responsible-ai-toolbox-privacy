import pytest
import sympy as sp
import numpy as np

from privacy_games.estimates.privacy_region import (
    privacy_boundary_normal_hi, privacy_boundary_normal_lo, privacy_velocity_hi, privacy_velocity_lo
)


R = sp.Matrix([[0, 1], [-1, 0]])  # Rotation by pi/2 counter clock wise


def privacy_boundary_lo(x, eps, delta):
    return sp.Matrix([[x], [sp.Max(0, 1 - delta - x * sp.exp(eps), (1 - delta - x) *sp.exp(-eps))]])

def privacy_boundary_hi(x, eps, delta):
    return sp.Matrix([[1],[1]])-privacy_boundary_lo(1-x, eps, delta)

@pytest.mark.parametrize("fnr", [0.1, 0.7, 0.95])  # This should be a point for each of the piecewise linear segments
def test_normal_lo_analytic(fnr):
    x, eps, delta = sp.symbols('x epsilon delta')
    n_tilde = R*sp.diff(privacy_boundary_lo(x, eps, delta), x)
    n = n_tilde/n_tilde.norm()
    assert n.subs({x: fnr, eps: 1.0, delta: 0.1})[0] == pytest.approx(privacy_boundary_normal_lo(fnr, 1.0, 0.1)[0])
    assert n.subs({x: fnr, eps: 1.0, delta: 0.1})[1] == pytest.approx(privacy_boundary_normal_lo(fnr, 1.0, 0.1)[1])

@pytest.mark.parametrize("fnr", [0.05, 0.3, 0.9])  # This should be a point for each of the piecewise linear segments
def test_normal_hi_analytic(fnr):
    x, eps, delta = sp.symbols('x epsilon delta')
    n_tilde = -R*sp.diff(privacy_boundary_hi(x, eps, delta), x)
    n = n_tilde/n_tilde.norm()
    assert n.subs({x: fnr, eps: 1.0, delta: 0.1})[0] == pytest.approx(privacy_boundary_normal_hi(fnr, 1.0, 0.1)[0])
    assert n.subs({x: fnr, eps: 1.0, delta: 0.1})[1] == pytest.approx(privacy_boundary_normal_hi(fnr, 1.0, 0.1)[1])

@pytest.mark.parametrize("fnr", [0.1, 0.7, 0.95])  # This should be a point for each of the piecewise linear segments
def test_velocity_lo_analytic(fnr):
    x, eps, delta = sp.symbols('x epsilon delta')
    r_eps = sp.diff(privacy_boundary_lo(x, eps, delta), eps)
    assert r_eps.subs({x: fnr, eps: 1.0, delta: 0.1})[0] == pytest.approx(privacy_velocity_lo(fnr, 1.0, 0.1)[0])

@pytest.mark.parametrize("fnr", [0.05, 0.3, 0.9])  # This should be a point for each of the piecewise linear segments
def test_velocity_hi_analytic(fnr):
    x, eps, delta = sp.symbols('x epsilon delta')
    r_eps = sp.diff(privacy_boundary_hi(x, eps, delta), eps)
    assert r_eps.subs({x: fnr, eps: 1.0, delta: 0.1})[0] == pytest.approx(privacy_velocity_hi(fnr, 1.0, 0.1)[0])

@pytest.mark.parametrize("fnr", [0.05, 0.3, 0.9])  # This should be a point for each of the piecewise linear segments
def test_normal_hi_normalised(fnr):
    n = privacy_boundary_normal_hi(fnr=fnr, eps=1, delta=0.1)
    assert np.linalg.norm(n) == pytest.approx(1.0)

@pytest.mark.parametrize("fnr", [0.1, 0.7, 0.95])
def test_normal_lo_normalised(fnr):
    n = privacy_boundary_normal_hi(fnr=fnr, eps=1, delta=0.1)
    assert np.linalg.norm(n) == pytest.approx(1.0)

@pytest.mark.parametrize("fnr", [0.05, 0.3, 0.9])  # This should be a point for each of the piecewise linear segments
def test_normal_hi_outward_pointing(fnr):
    n = privacy_boundary_normal_hi(fnr=fnr, eps=1, delta=0.1)
    assert np.dot(n, [1, 0]) >= 0.0
    assert np.dot(n, [0, 1]) > 0.0

@pytest.mark.parametrize("fnr", [0.1, 0.7, 0.95])
def test_normal_lo_outward_pointing(fnr):
    n = privacy_boundary_normal_lo(fnr=fnr, eps=1, delta=0.1)
    assert np.dot(n, [1, 0]) <= 0.0
    assert np.dot(n, [0, 1]) < 0.0

