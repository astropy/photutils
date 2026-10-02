# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _bispline module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.interpolate import RectBivariateSpline

from photutils.psf._bispline import bispline_sum, bispline_sum_deriv


@pytest.fixture(name='splines')
def fixture_splines():
    """
    Four cubic interpolating splines on a rectangular grid with
    different x and y sizes, and their shared knots and stacked
    coefficients.
    """
    rng = np.random.default_rng(1)
    nx, ny = 41, 33
    xg = np.arange(nx, dtype=float)
    yg = np.arange(ny, dtype=float)
    splines = [RectBivariateSpline(xg, yg, rng.normal(size=(nx, ny)), kx=3,
                                   ky=3, s=0) for _ in range(4)]
    tx, ty, _ = splines[0].tck
    coeffs = np.array([spline.tck[2] for spline in splines])
    return splines, np.ascontiguousarray(tx), np.ascontiguousarray(ty), coeffs


def _points(nx=41, ny=33, n=500, seed=2):
    rng = np.random.default_rng(seed)
    # Include points beyond the grid, which the splines clamp
    x = rng.uniform(-5, nx + 4, n)
    y = rng.uniform(-5, ny + 4, n)
    return x, y


def _partial_derivatives(spline, x, y, nx=41, ny=33):
    """
    The x and y partial derivatives of the clamped spline.

    The derivative splines of scipy give the derivative at the grid
    edge for a point beyond the grid, where the clamped spline is
    constant along that axis and its derivative is zero.
    """
    d_x = spline.partial_derivative(1, 0)(x, y, grid=False)
    d_y = spline.partial_derivative(0, 1)(x, y, grid=False)
    d_x[(x < 0) | (x > nx - 1)] = 0.0
    d_y[(y < 0) | (y > ny - 1)] = 0.0
    return d_x, d_y


class TestBisplineSum:
    def test_matches_scipy(self, splines):
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([0, 1, 2, 3], dtype=np.intp)
        weights = np.array([0.3, 0.2, 0.4, 0.1])
        x, y = _points()
        out = np.empty_like(x)
        bispline_sum(tx, ty, coeffs, grid_idx, weights, x, y, out)
        expected = sum(w * spline(x, y, grid=False)
                       for w, spline in zip(weights, splines, strict=True))
        assert_allclose(out, expected, rtol=1e-13, atol=1e-14)

    def test_knot_points(self, splines):
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([2], dtype=np.intp)
        weights = np.ones(1)
        x = np.array([0.0, 1.0, 2.0, 20.0, 39.0, 40.0, 40.0])
        y = np.array([0.0, 32.0, 3.0, 16.0, 1.0, 30.0, 32.0])
        out = np.empty_like(x)
        bispline_sum(tx, ty, coeffs, grid_idx, weights, x, y, out)
        assert_allclose(out, splines[2](x, y, grid=False), rtol=1e-13,
                        atol=1e-14)

    def test_zero_weight_and_subset(self, splines):
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([3, 1], dtype=np.intp)
        weights = np.array([0.0, 0.7])
        x, y = _points(n=50)
        out = np.empty_like(x)
        bispline_sum(tx, ty, coeffs, grid_idx, weights, x, y, out)
        assert_allclose(out, 0.7 * splines[1](x, y, grid=False), rtol=1e-13,
                        atol=1e-14)

    def test_non_finite_coordinates(self, splines):
        """
        Test that a NaN coordinate gives NaN and that an infinite
        coordinate is clamped to the knot range like any other point
        beyond it. The kernels index the arrays without bounds checks,
        so these coordinates must not select an invalid knot interval.
        """
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([1], dtype=np.intp)
        ones = np.ones(1)
        zeros = np.zeros(1)
        x = np.array([np.nan, 10.3, np.inf, -np.inf, 10.3, np.nan])
        y = np.array([5.2, np.nan, 5.2, 5.2, np.inf, -np.inf])
        out = np.empty_like(x)
        bispline_sum(tx, ty, coeffs, grid_idx, ones, x, y, out)
        assert np.all(np.isnan(out[[0, 1, 5]]))
        x_edge = np.array([40.0, 0.0, 10.3])
        y_edge = np.array([5.2, 5.2, 32.0])
        assert_allclose(out[2:5], splines[1](x_edge, y_edge, grid=False),
                        rtol=1e-13, atol=1e-14)

        out2 = np.empty_like(x)
        out_dx = np.empty_like(x)
        out_dy = np.empty_like(x)
        bispline_sum_deriv(tx, ty, coeffs, grid_idx, ones, zeros, zeros,
                           -1.0, -1.0, x, y, out2, out_dx, out_dy)
        assert_equal(out2, out)
        assert np.all(np.isnan(out_dx[[0, 1, 5]]))
        assert np.all(np.isnan(out_dy[[0, 1, 5]]))
        # An infinite coordinate is beyond the knots, so the derivative
        # along its axis is zero
        assert np.all(out_dx[[2, 3]] == 0.0)
        assert out_dy[4] == 0.0
        assert np.all(np.isfinite(out_dy[[2, 3]]))
        assert np.isfinite(out_dx[4])

    def test_invalid_inputs(self, splines):
        _, tx, ty, coeffs = splines
        grid_idx = np.array([0], dtype=np.intp)
        weights = np.ones(1)
        x, y = _points(n=10)
        out = np.empty(10)
        with pytest.raises(ValueError, match='inconsistent lengths'):
            bispline_sum(tx, ty, coeffs, grid_idx, weights, x, y[:5], out)
        with pytest.raises(ValueError, match='inconsistent lengths'):
            bispline_sum(tx, ty, coeffs, grid_idx, np.ones(2), x, y, out)
        with pytest.raises(ValueError, match='does not match the knot'):
            bispline_sum(tx, ty, coeffs[:, :-1].copy(), grid_idx, weights,
                         x, y, out)
        with pytest.raises(ValueError, match='too short'):
            bispline_sum(tx[:7].copy(), ty, coeffs, grid_idx, weights, x,
                         y, out)
        for bad_idx in (-1, 4):
            with pytest.raises(ValueError, match='grid_idx'):
                bispline_sum(tx, ty, coeffs,
                             np.array([bad_idx], dtype=np.intp), weights,
                             x, y, out)


class TestBisplineSumDeriv:
    def test_matches_scipy(self, splines):
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([0, 1, 2, 3], dtype=np.intp)
        weights = np.array([0.3, 0.2, 0.4, 0.1])
        dw_dx = np.array([-0.01, 0.01, -0.02, 0.02])
        dw_dy = np.array([0.03, -0.03, 0.01, -0.01])
        scale_x, scale_y = 4.0, 2.0
        x, y = _points()
        out = np.empty_like(x)
        out_dx = np.empty_like(x)
        out_dy = np.empty_like(x)
        bispline_sum_deriv(tx, ty, coeffs, grid_idx, weights, dw_dx, dw_dy,
                           scale_x, scale_y, x, y, out, out_dx, out_dy)

        values = [spline(x, y, grid=False) for spline in splines]
        d_x, d_y = zip(*[_partial_derivatives(spline, x, y)
                         for spline in splines], strict=True)
        expected = sum(w * v for w, v in zip(weights, values, strict=True))
        expected_dx = sum(dwx * v - scale_x * w * dx
                          for w, dwx, v, dx in zip(weights, dw_dx, values,
                                                   d_x, strict=True))
        expected_dy = sum(dwy * v - scale_y * w * dy
                          for w, dwy, v, dy in zip(weights, dw_dy, values,
                                                   d_y, strict=True))
        assert_allclose(out, expected, rtol=1e-13, atol=1e-14)
        assert_allclose(out_dx, expected_dx, rtol=1e-13, atol=1e-13)
        assert_allclose(out_dy, expected_dy, rtol=1e-13, atol=1e-13)

    def test_skips_unused_planes(self, splines):
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([0, 3], dtype=np.intp)
        weights = np.array([0.0, 1.0])
        zeros = np.zeros(2)
        x, y = _points(n=50)
        out = np.empty_like(x)
        out_dx = np.empty_like(x)
        out_dy = np.empty_like(x)
        bispline_sum_deriv(tx, ty, coeffs, grid_idx, weights, zeros, zeros,
                           1.0, 1.0, x, y, out, out_dx, out_dy)
        spline = splines[3]
        d_x, d_y = _partial_derivatives(spline, x, y)
        assert_allclose(out, spline(x, y, grid=False), rtol=1e-13, atol=1e-14)
        assert_allclose(out_dx, -d_x, rtol=1e-13, atol=1e-13)
        assert_allclose(out_dy, -d_y, rtol=1e-13, atol=1e-13)

    def test_zero_derivative_beyond_knots(self, splines):
        """
        Test that the derivative along an axis is zero for a point
        beyond the knot range along that axis, where the clamped spline
        is constant, and that the edge points keep the edge derivative.
        """
        splines, tx, ty, coeffs = splines
        grid_idx = np.array([1], dtype=np.intp)
        ones = np.ones(1)
        zeros = np.zeros(1)
        x = np.array([-0.5, 40.5, 20.3, 20.3, -2.0, 0.0, 40.0])
        y = np.array([10.2, 10.2, -0.5, 32.5, 35.0, 0.0, 32.0])
        out = np.empty_like(x)
        out_dx = np.empty_like(x)
        out_dy = np.empty_like(x)
        bispline_sum_deriv(tx, ty, coeffs, grid_idx, ones, zeros, zeros,
                           -1.0, -1.0, x, y, out, out_dx, out_dy)
        assert np.all(out_dx[[0, 1, 4]] == 0.0)
        assert np.all(out_dy[[2, 3, 4]] == 0.0)
        assert np.all(out_dy[[0, 1, 5, 6]] != 0.0)
        assert np.all(out_dx[[2, 3, 5, 6]] != 0.0)

        # The derivatives match central differences of the values
        eps = 1e-6

        def values(xval, yval):
            result = np.empty_like(xval)
            bispline_sum(tx, ty, coeffs, grid_idx, ones, xval, yval, result)
            return result

        inside = slice(0, 5)  # the edge points have one-sided derivatives
        num_dx = (values(x + eps, y) - values(x - eps, y)) / (2 * eps)
        num_dy = (values(x, y + eps) - values(x, y - eps)) / (2 * eps)
        assert_allclose(out_dx[inside], num_dx[inside], atol=1e-7)
        assert_allclose(out_dy[inside], num_dy[inside], atol=1e-7)

    def test_invalid_inputs(self, splines):
        _, tx, ty, coeffs = splines
        grid_idx = np.array([0], dtype=np.intp)
        ones = np.ones(1)
        x, y = _points(n=10)
        out = np.empty(10)
        with pytest.raises(ValueError, match='inconsistent lengths'):
            bispline_sum_deriv(tx, ty, coeffs, grid_idx, ones, np.ones(2),
                               ones, 1.0, 1.0, x, y, out, out, out)
        with pytest.raises(ValueError, match='does not match the knot'):
            bispline_sum_deriv(tx, ty, coeffs[:, 1:].copy(), grid_idx, ones,
                               ones, ones, 1.0, 1.0, x, y, out, out, out)
        with pytest.raises(ValueError, match='too short'):
            bispline_sum_deriv(tx, ty[:7].copy(), coeffs, grid_idx, ones,
                               ones, ones, 1.0, 1.0, x, y, out, out, out)
        for bad_idx in (-1, 4):
            with pytest.raises(ValueError, match='grid_idx'):
                bispline_sum_deriv(tx, ty, coeffs,
                                   np.array([bad_idx], dtype=np.intp), ones,
                                   ones, ones, 1.0, 1.0, x, y, out, out,
                                   out)
