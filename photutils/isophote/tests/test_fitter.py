# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the fitter module.
"""

import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose

from photutils.datasets import get_path
from photutils.isophote.fitter import CentralEllipseFitter, EllipseFitter
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.harmonics import fit_first_and_second_harmonics
from photutils.isophote.integrator import MEAN
from photutils.isophote.isophote import Isophote
from photutils.isophote.sample import CentralEllipseSample, EllipseSample
from photutils.isophote.tests.make_test_data import make_test_image

DATA = make_test_image(seed=0)
DEFAULT_POS = 256

DEFAULT_FIX = np.array([False, False, False, False])


def test_gradient():
    sample = EllipseSample(DATA, 40.0)
    sample.update(DEFAULT_FIX)

    assert_allclose(sample.mean, 200.02, atol=0.01)
    assert_allclose(sample.gradient, -4.222, atol=0.001)
    assert_allclose(sample.gradient_error, 0.0003, atol=0.0001)
    assert_allclose(sample.gradient_relative_error, 7.45e-05, atol=1.0e-5)
    assert_allclose(sample.sector_area, 2.00, atol=0.01)


def test_fitting_raw():
    """
    This test performs a raw (no EllipseFitter), 1-step correction in
    one single ellipse coefficient.
    """
    # pick first guess ellipse that is off in just
    # one of the parameters (eps).
    sample = EllipseSample(DATA, 40.0, eps=2 * 0.2)
    sample.update(DEFAULT_FIX)
    s = sample.extract()

    harmonics = fit_first_and_second_harmonics(s[0], s[2])
    _, a1, b1, a2, b2 = harmonics[0]

    # when eps is off, b2 is the largest (in absolute value).
    assert abs(b2) > abs(a1)
    assert abs(b2) > abs(b1)
    assert abs(b2) > abs(a2)

    correction = (b2 * 2.0 * (1.0 - sample.geometry.eps)
                  / sample.geometry.sma / sample.gradient)
    new_eps = sample.geometry.eps - correction

    # got closer to test data (eps=0.2)
    assert_allclose(new_eps, 0.21, atol=0.01)


def test_fitting_small_radii():
    sample = EllipseSample(DATA, 2.0)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    assert isophote.ndata == 13


def test_fitting_eps():
    # initial guess is off in the eps parameter
    sample = EllipseSample(DATA, 40.0, eps=2 * 0.2)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    g = isophote.sample.geometry
    assert g.eps >= 0.19
    assert g.eps <= 0.21


def test_fitting_pa():
    data = make_test_image(pa=np.pi / 4, noise=0.01, seed=0)

    # initial guess is off in the pa parameter
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert g.pa >= (np.pi / 4 - 0.05)
    assert g.pa <= (np.pi / 4 + 0.05)


def test_fitting_xy():
    pos = DEFAULT_POS - 5
    data = make_test_image(x0=pos, y0=pos, seed=0)

    # initial guess is off in the x0 and y0 parameters
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert g.x0 >= (pos - 1)
    assert g.x0 <= (pos + 1)
    assert g.y0 >= (pos - 1)
    assert g.y0 <= (pos + 1)


def test_fitting_all():
    # build test image that is off from the defaults
    # assumed by the EllipseSample constructor.
    pos = DEFAULT_POS - 5
    angle = np.pi / 4
    eps = 2 * 0.2
    data = make_test_image(x0=pos, y0=pos, eps=eps, pa=angle, seed=0)
    sma = 60.0

    # initial guess is off in all parameters. We find that the initial
    # guesses, especially for position angle, must be kinda close to the
    # actual value. 20% off max seems to work in this case of high SNR.
    sample = EllipseSample(data, sma, position_angle=(1.2 * angle))
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isophote.stop_code == 0

    g = isophote.sample.geometry
    assert g.x0 >= (pos - 1.5)  # position within 1.5 pixel
    assert g.x0 <= (pos + 1.5)
    assert g.y0 >= (pos - 1.5)
    assert g.y0 <= (pos + 1.5)
    assert g.eps >= (eps - 0.01)  # eps within 0.01
    assert g.eps <= (eps + 0.01)
    assert g.pa >= (angle - 0.05)  # pa within 5 deg
    assert g.pa <= (angle + 0.05)

    sample_m = EllipseSample(data, sma, position_angle=(1.2 * angle),
                             integrmode=MEAN)
    fitter_m = EllipseFitter(sample_m)
    isophote_m = fitter_m.fit()

    assert isophote_m.stop_code == 0


@pytest.mark.remote_data
class TestM51:
    def setup_class(self):
        path = get_path('isophote/M51.fits', location='photutils-datasets',
                        cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def test_m51(self):
        # Here we evaluate the detailed convergence behavior
        # for a particular ellipse where we can see the eps
        # parameter jumping back and forth.
        # We start the fit with initial values taken from
        # previous isophote, as determined by the old code.

        # sample taken in high SNR region
        sample = EllipseSample(self.data, 21.44, eps=0.18,
                               position_angle=(36.0 / 180.0 * np.pi))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.ndata == 119
        assert_allclose(isophote.intens, 685.4, atol=0.1)

        # last sample taken by the original code, before turning inwards.
        sample = EllipseSample(self.data, 61.16, eps=0.219,
                               position_angle=((77.5 + 90) / 180 * np.pi))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.ndata == 382
        assert_allclose(isophote.intens, 155.0, atol=0.1)

    def test_m51_outer(self):
        # sample taken at the outskirts of the image, so many
        # data points lay outside the image frame. This checks
        # for the presence of gaps in the sample arrays.
        sample = EllipseSample(self.data, 330.0, eps=0.2,
                               position_angle=((90) / 180 * np.pi),
                               integrmode='median')
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert not np.any(isophote.sample.values[2] == 0)

    def test_m51_central(self):
        # this code finds central x and y offset by about 0.1 pixel wrt the
        # spp code. In here we use as input the position computed by this
        # code, thus this test is checking just the extraction algorithm.
        g = EllipseGeometry(257.02, 258.1, 0.0, 0.0, 0.0, 0.1,
                            linear_growth=False)
        sample = CentralEllipseSample(self.data, 0.0, geometry=g)
        fitter = CentralEllipseFitter(sample)
        isophote = fitter.fit()

        # the central pixel intensity is about 3% larger than
        # found by the spp code.
        assert isophote.ndata == 1
        assert isophote.intens <= 7560.0
        assert isophote.intens >= 7550.0
