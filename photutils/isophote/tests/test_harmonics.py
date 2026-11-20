# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the harmonics module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from numpy.testing import assert_allclose
from scipy.optimize import leastsq

from photutils.isophote.ellipse import Ellipse
from photutils.isophote.fitter import EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import (first_and_second_harmonic_function,
                                          fit_first_and_second_harmonics,
                                          fit_upper_harmonic)
from photutils.isophote.sample import EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image


def test_harmonics_1():
    npts = 100  # number of data points
    theta = np.linspace(0, 4 * np.pi, npts)

    # Create artificial data with noise:
    # mean = 0.5, amplitude = 3.0, phase = 0.1, noise-std = 0.01.
    rng = np.random.default_rng(0)
    data = 3.0 * np.sin(theta + 0.1) + 0.5 + 0.01 * rng.standard_normal(npts)

    # First guesses for harmonic parameters
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / 2**0.5
    guess_phase = 0

    # Minimize the difference between the actual data and our "guessed"
    # parameters.
    def optimize_func(x):
        return x[0] * np.sin(theta + x[1]) + x[2] - data

    est_std, est_phase, est_mean = leastsq(
        optimize_func, [guess_std, guess_phase, guess_mean])[0]

    # Recreate the fitted curve using the optimized parameters.
    data_fit = est_std * np.sin(theta + est_phase) + est_mean
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.001)
    assert_allclose(np.std(residual), 0.01, atol=0.01)


def test_harmonics_2():
    # This uses the actual functional form used for fitting ellipses.
    npts = 100
    theta = np.linspace(0, 4 * np.pi, npts)

    y0_0 = 100.0
    a1_0 = 10.0
    b1_0 = 5.0
    a2_0 = 8.0
    b2_0 = 2.0
    rng = np.random.default_rng(0)
    data = (y0_0 + a1_0 * np.sin(theta) + b1_0 * np.cos(theta)
            + a2_0 * np.sin(2 * theta) + b2_0 * np.cos(2 * theta)
            + 0.01 * rng.standard_normal(npts))

    harmonics = fit_first_and_second_harmonics(theta, data)
    y0, a1, b1, a2, b2 = harmonics[0]
    data_fit = (y0 + a1 * np.sin(theta) + b1 * np.cos(theta)
                + a2 * np.sin(2 * theta) + b2 * np.cos(2 * theta)
                + 0.01 * rng.standard_normal(npts))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.01)


def test_harmonics_3():
    """
    Tests an upper harmonic fit.
    """
    npts = 100
    theta = np.linspace(0, 4 * np.pi, npts)
    y0_0 = 100.0
    a1_0 = 10.0
    b1_0 = 5.0
    order = 3
    rng = np.random.default_rng(0)
    data = (y0_0 + a1_0 * np.sin(order * theta) + b1_0 * np.cos(order * theta)
            + 0.01 * rng.standard_normal(npts))

    harmonic = fit_upper_harmonic(theta, data, order)
    y0, a1, b1 = harmonic[0]
    rng = np.random.default_rng(0)
    data_fit = (y0 + a1 * np.sin(order * theta) + b1 * np.cos(order * theta)
                + 0.01 * rng.standard_normal(npts))
    residual = data - data_fit

    assert_allclose(np.mean(residual), 0.0, atol=0.01)
    assert_allclose(np.std(residual), 0.015, atol=0.014)


class TestFitEllipseSamples:
    def setup_class(self):
        # Major axis parallel to X image axis
        self.data1 = make_test_image(seed=0)

        # Major axis tilted 45 deg wrt X image axis
        self.data2 = make_test_image(pa=np.pi / 4, seed=0)

    def test_fit_ellipsesample_1(self):
        sample = EllipseSample(self.data1, 40.0)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 200.019, atol=0.001)
        assert_allclose(np.mean(a1), -0.000138, atol=0.001)
        assert_allclose(np.mean(b1), 0.000254, atol=0.001)
        assert_allclose(np.mean(a2), -5.658e-05, atol=0.001)
        assert_allclose(np.mean(b2), -0.00911, atol=0.001)

        # Check that harmonics subtract nicely.
        model = first_and_second_harmonic_function(
            s[0], np.array([y0, a1, b1, a2, b2]))
        residual = s[2] - model

        assert_allclose(np.mean(residual), 0.0, atol=0.001)
        assert_allclose(np.std(residual), 0.015, atol=0.01)

    def test_fit_ellipsesample_2(self):
        # Initial guess is rounder than actual image.
        sample = EllipseSample(self.data1, 40.0, eps=0.1)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 188.686, atol=0.001)
        assert_allclose(np.mean(a1), 0.000283, atol=0.001)
        assert_allclose(np.mean(b1), 0.00692, atol=0.001)
        assert_allclose(np.mean(a2), -0.000215, atol=0.001)
        assert_allclose(np.mean(b2), 10.153, atol=0.001)

    def test_fit_ellipsesample_3(self):
        # Initial guess for center is offset.
        sample = EllipseSample(self.data1, x0=220.0, y0=210.0, sma=40.0)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 152.660, atol=0.001)
        assert_allclose(np.mean(a1), 55.338, atol=0.001)
        assert_allclose(np.mean(b1), 33.091, atol=0.001)
        assert_allclose(np.mean(a2), 33.036, atol=0.001)
        assert_allclose(np.mean(b2), -14.306, atol=0.001)

    def test_fit_ellipsesample_4(self):
        sample = EllipseSample(self.data2, 40.0, eps=0.4)
        s = sample.extract()

        harmonics = fit_first_and_second_harmonics(s[0], s[2])
        y0, a1, b1, a2, b2 = harmonics[0]

        assert_allclose(np.mean(y0), 245.102, atol=0.001)
        assert_allclose(np.mean(a1), -0.003108, atol=0.001)
        assert_allclose(np.mean(b1), -0.0578, atol=0.001)
        assert_allclose(np.mean(a2), 28.781, atol=0.001)
        assert_allclose(np.mean(b2), -63.184, atol=0.001)

    def test_fit_upper_harmonics(self):
        data = make_test_image(noise=1.0e-10, seed=0)
        sample = EllipseSample(data, 40)
        fitter = EllipseFitter(sample)
        iso = fitter.fit(maxit=400)

        assert_allclose(iso.a3, 6.825e-7, atol=1.0e-8)
        assert_allclose(iso.b3, -1.68e-6, atol=1.0e-8)
        assert_allclose(iso.a4, 4.36e-6, atol=1.0e-8)
        assert_allclose(iso.b4, -4.73e-5, atol=1.0e-7)

        assert_allclose(iso.a3_err, 8.152e-6, atol=1.0e-7)
        assert_allclose(iso.b3_err, 8.115e-6, atol=1.0e-7)
        assert_allclose(iso.a4_err, 7.501e-6, atol=1.0e-7)
        assert_allclose(iso.b4_err, 7.473e-6, atol=1.0e-7)


def test_upper_harmonics_sign():
    """
    Regression test for #1486/#1501.
    """
    angle = np.deg2rad(40.0)
    g1 = Gaussian2D(100.0, 75, 75, 15, 3, theta=angle)
    g2 = Gaussian2D(100.0, 75, 75, 10, 8, theta=angle)

    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    data = g1(x, y) + g2(x, y)
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.9, pa=angle)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    # Test image is "disky" and disky isophotes have b4 > 0 (boxy
    # isophotes have b4 < 0).
    assert np.all(isolist.b4[30:] > 0)

    # Check the sign of the upper harmonics at the outermost
    # well-sampled isophote. The very last isophote(s) can be
    # degenerate, with a large fraction of the elliptical path falling
    # outside the image, in which case the near-zero upper harmonics
    # are dominated by noise and their signs are not meaningful.
    n_total = isolist.n_data + isolist.n_flag
    well_sampled = isolist.n_flag < (0.3 * n_total)
    idx = np.where(well_sampled)[0][-1]
    assert isolist.a3[idx] < 0
    assert isolist.a4[idx] < 0
    assert isolist.b3[idx] > 0
    assert isolist.b4[idx] > 0


def test_harmonic_fits_exact():
    """
    Test that the harmonic fits recover the exact coefficients and the
    unscaled covariance matrix of noiseless data.
    """
    phi = np.linspace(0.0, 2 * np.pi, 50, endpoint=False)
    coeffs = np.array([100.0, 1e-3, -2e-3, 3.0, -4.0])
    intensities = first_and_second_harmonic_function(phi, coeffs)
    fit_coeffs, covariance = fit_first_and_second_harmonics(phi,
                                                            intensities)
    assert_allclose(fit_coeffs, coeffs, rtol=0, atol=1e-12)

    # For angles sampled uniformly over the full circle, the harmonic
    # functions are orthogonal, so the unscaled covariance matrix is
    # diagonal.
    n_points = len(phi)
    expected = np.diag([1.0, 2.0, 2.0, 2.0, 2.0]) / n_points
    assert_allclose(covariance, expected, rtol=0, atol=1e-15)

    intensities = 10.0 + 2e-4 * np.sin(4 * phi) - 3e-4 * np.cos(4 * phi)
    fit_coeffs, covariance = fit_upper_harmonic(phi, intensities, 4)
    assert_allclose(fit_coeffs, [10.0, 2e-4, -3e-4], rtol=0, atol=1e-12)
    expected = np.diag([1.0, 2.0, 2.0]) / n_points
    assert_allclose(covariance, expected, rtol=0, atol=1e-15)


def test_harmonic_fits_too_few_points():
    phi = np.array([0.0, 1.0, 2.0, 3.0])
    match = 'At least 5 data points are required'
    with pytest.raises(ValueError, match=match):
        fit_first_and_second_harmonics(phi, phi)
    match = 'At least 3 data points are required'
    with pytest.raises(ValueError, match=match):
        fit_upper_harmonic(phi[:2], phi[:2], 3)


def test_harmonic_fit_short_arc():
    """
    Test the covariance matrix of a harmonic fit to angles that cover
    only a short arc, where the harmonic functions are nearly
    degenerate.
    """
    phi = 1.0 + 0.05 * np.arange(17) / 16.0
    rng = np.random.default_rng(0)
    intensities = 100.0 + rng.standard_normal(len(phi))
    coeffs, covariance = fit_first_and_second_harmonics(phi, intensities)

    # The reference covariance matrix is computed from the QR
    # decomposition of the design matrix.
    design = np.column_stack([np.ones_like(phi), np.sin(phi), np.cos(phi),
                              np.sin(2 * phi), np.cos(2 * phi)])
    r_inv = np.linalg.inv(np.linalg.qr(design)[1])
    expected = r_inv @ r_inv.T

    assert np.all(np.isfinite(coeffs))
    assert np.all(np.diag(covariance) > 0)
    assert_allclose(covariance, expected, rtol=1e-5)


@pytest.mark.parametrize('phi', [np.full(10, 0.3),
                                 [0.1, 0.1, 0.5, 1.0, 2.0, 2.0]])
def test_harmonic_fit_degenerate_angles(phi):
    """
    Test that the covariance matrix is `None` if the angles do not
    constrain all of the harmonic coefficients.
    """
    intensities = 10.0 + np.arange(len(phi))
    coeffs, covariance = fit_first_and_second_harmonics(phi, intensities)
    assert np.all(np.isfinite(coeffs))
    assert covariance is None


def test_short_arc_isophote_errors():
    """
    Test that an isophote with valid data over only a short arc has
    finite and positive parameter errors.
    """
    data = make_test_image(noise=1.0, seed=0)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    theta = np.arctan2(yy - 256.0, xx - 256.0)
    mask = (theta < 0.5) | (theta > 0.56)
    geometry = EllipseGeometry(x0=256.0, y0=256.0, sma=220.0, eps=0.2, pa=0.0)
    ellipse = Ellipse(np.ma.masked_array(data, mask=mask), geometry=geometry)
    isophote = ellipse.fit_isophote(220.0)

    assert isophote.n_data >= 6
    for name in ('x0_err', 'y0_err', 'ellip_err', 'pa_err'):
        value = getattr(isophote, name)
        assert np.isfinite(value)
        assert value > 0
