from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest

import numpy as np
from astropy.io import fits

from photutils.isophote import build_test_data
from photutils.isophote.build_test_data import DEFAULT_POS
from photutils.isophote.geometry import Geometry, DEFAULT_EPS
from photutils.isophote.integrator import MEAN, MEDIAN
from photutils.isophote.harmonics import fit_1st_and_2nd_harmonics
from photutils.isophote.sample import Sample, CentralSample
from photutils.isophote.isophote import Isophote
from photutils.isophote.fitter import Fitter, CentralFitter
from photutils.isophote.tests.test_data import TEST_DATA

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_gradient():

    test_data = build_test_data.build()

    sample = Sample(test_data, 40.)
    sample.update()

    assert sample.mean == pytest.approx(200.02, abs=0.01)
    assert sample.gradient == pytest.approx(-4.222, abs=0.001)
    assert sample.gradient_error == pytest.approx(0.0003, abs=0.0001)
    assert sample.gradient_relative_error == pytest.approx(7.45e-05, abs=1.e-5)
    assert sample.sector_area == pytest.approx(2.00, abs=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_raw():
    # this test performs a raw (no Fitter), 1-step
    # correction in one single ellipse coefficient.

    test_data = build_test_data.build()

    # pick first guess ellipse that is off in just
    # one of the parameters (eps).
    sample = Sample(test_data, 40., eps=2*DEFAULT_EPS)
    sample.update()
    s = sample.extract()

    harmonics = fit_1st_and_2nd_harmonics(s[0], s[2])
    y0, a1, b1, a2, b2 = harmonics[0]

    # when eps is off, b2 is the largest (in absolute value).
    assert abs(b2) > abs(a1)
    assert abs(b2) > abs(b1)
    assert abs(b2) > abs(a2)
    correction = b2 * 2. * (1. - sample.geometry.eps) / sample.geometry.sma / sample.gradient
    new_eps = sample.geometry.eps - correction

    # got closer to test data (eps=0.2)
    assert new_eps == pytest.approx(0.21, abs=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_small_radii():

    test_data = build_test_data.build()

    sample = Sample(test_data, 2.)
    fitter = Fitter(sample)

    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    assert isophote.ndata == 13


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_eps():

    test_data = build_test_data.build()

    # initial guess is off in the eps parameter
    sample = Sample(test_data, 40., eps=2*DEFAULT_EPS)
    fitter = Fitter(sample)

    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    g = isophote.sample.geometry
    assert g.eps >= 0.19
    assert g.eps <= 0.21


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_pa():

    test_data = build_test_data.build(pa=np.pi/4, noise=0.01)

    # initial guess is off in the pa parameter
    sample = Sample(test_data, 40)
    fitter = Fitter(sample)

    isophote = fitter.fit()

    g = isophote.sample.geometry
    assert g.pa >= (np.pi/4 - 0.05)
    assert g.pa <= (np.pi/4 + 0.05)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_xy():

    pos_ = DEFAULT_POS - 5
    test_data = build_test_data.build(x0=pos_, y0=pos_)

    # initial guess is off in the x0 and y0 parameters
    sample = Sample(test_data, 40)
    fitter = Fitter(sample)

    isophote = fitter.fit()

    g = isophote.sample.geometry
    assert g.x0 >= (pos_ - 1)
    assert g.x0 <= (pos_ + 1)
    assert g.y0 >= (pos_ - 1)
    assert g.y0 <= (pos_ + 1)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_all():

    # build test image that is off from the defaults
    # assumed by the Sample constructor.
    POS = DEFAULT_POS - 5
    ANGLE = np.pi / 4
    EPS = 2 * DEFAULT_EPS
    test_data = build_test_data.build(x0=POS, y0=POS, eps=EPS, pa=ANGLE)

    sma = 60.

    # initial guess is off in all parameters. We find that the initial
    # guesses, especially for position angle, must be kinda close to the
    # actual value. 20% off max seems to work in this case of high SNR.
    sample = Sample(test_data, sma, position_angle=(1.2 * ANGLE))

    fitter = Fitter(sample)
    isophote = fitter.fit()

    assert isophote.stop_code == 0

    g = isophote.sample.geometry
    assert g.x0 >= (POS - 1.5)      # position within 1.5 pixel
    assert g.x0 <= (POS + 1.5)
    assert g.y0 >= (POS - 1.5)
    assert g.y0 <= (POS + 1.5)
    assert g.eps >= (EPS - 0.01)    # eps within 0.01
    assert g.eps <= (EPS + 0.01)
    assert g.pa >= (ANGLE - 0.05)   # pa within 5 deg
    assert g.pa <= (ANGLE + 0.05)

    sample_m = Sample(test_data, sma, position_angle=(1.2 * ANGLE), integrmode=MEAN)

    fitter_m = Fitter(sample_m)
    isophote_m = fitter_m.fit()

    assert isophote_m.stop_code == 0


@pytest.mark.skipif('not HAS_SCIPY')
def test_m51():
    image = fits.open(TEST_DATA + "M51.fits")
    test_data = image[0].data

    #
    # # here we evaluate the detailed convergency behavior
    # # for a particular ellipse where we can see the eps
    # # parameter jumping back and forth.
    # sample = Sample(test_data, 13.31000001, eps=0.16, position_angle=((-37.5+90)/180.*np.pi))
    # sample.update()
    # fitter = Fitter(sample)
    # isophote = fitter.fit()
    #

    # we start the fit with initial values taken from
    # previous isophote, as determined by the old code.

    # sample taken in high SNR region
    sample = Sample(test_data, 21.44, eps=0.18, position_angle=(36./180.*np.pi))
    fitter = Fitter(sample)
    isophote = fitter.fit()

    assert isophote.ndata == 119
    assert isophote.intens == pytest.approx(685.4, abs=0.1)

    # last sample taken by the original code, before turning inwards.
    sample = Sample(test_data, 61.16, eps=0.219, position_angle=((77.5+90)/180*np.pi))
    fitter = Fitter(sample)
    isophote = fitter.fit()

    assert isophote.ndata == 382
    assert isophote.intens == pytest.approx(155.0, abs=0.1)


@pytest.mark.skipif('not HAS_SCIPY')
def test_m51_outer():
    image = fits.open(TEST_DATA + "M51.fits")
    test_data = image[0].data

    # sample taken at the outskirts of the image, so many
    # data points lay outside the image frame. This checks
    # for the presence of gaps in the sample arrays.
    sample = Sample(test_data, 330., eps=0.2, position_angle=((90)/180*np.pi), integrmode='median')
    fitter = Fitter(sample)
    isophote = fitter.fit()

    assert not np.any(isophote.sample.values[2] == 0)


def test_m51_central():
    image = fits.open(TEST_DATA + "M51.fits")
    test_data = image[0].data

    # this code finds central x and y offset by about 0.1 pixel wrt the
    # spp code. In here we use as input the position computed by this
    # code, thus this test is checking just the extraction algorithm.
    g = Geometry(257.02, 258.1, 0.0, 0.0, 0.0, 0.1, False)
    sample = CentralSample(test_data, 0.0, geometry=g)
    fitter = CentralFitter(sample)

    isophote = fitter.fit()

    # the central pixel intensity is about 3% larger than
    # found by the spp code.
    assert isophote.ndata == 1
    assert isophote.intens <= 7560.
    assert isophote.intens >= 7550.
