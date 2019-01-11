# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.io import fits

from .make_test_data import make_test_image
from ..fitter import EllipseFitter, CentralEllipseFitter
from ..geometry import EllipseGeometry
from ..harmonics import fit_first_and_second_harmonics
from ..integrator import MEAN
from ..isophote import Isophote
from ..sample import EllipseSample, CentralEllipseSample
from ...datasets import get_path

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


DATA = make_test_image(random_state=123)
DEFAULT_POS = 256


def test_gradient():
    sample = EllipseSample(DATA, 40.)
    sample.update()

    assert_allclose(sample.mean, 200.02, atol=0.01)
    assert_allclose(sample.gradient, -4.222, atol=0.001)
    assert_allclose(sample.gradient_error, 0.0003, atol=0.0001)
    assert_allclose(sample.gradient_relative_error, 7.45e-05, atol=1.e-5)
    assert_allclose(sample.sector_area, 2.00, atol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_raw():
    """
    This test performs a raw (no EllipseFitter), 1-step correction in
    one single ellipse coefficient.
    """

    # pick first guess ellipse that is off in just
    # one of the parameters (eps).
    sample = EllipseSample(DATA, 40., eps=2*0.2)
    sample.update()
    s = sample.extract()

    harmonics = fit_first_and_second_harmonics(s[0], s[2])
    y0, a1, b1, a2, b2 = harmonics[0]

    # when eps is off, b2 is the largest (in absolute value).
    assert abs(b2) > abs(a1)
    assert abs(b2) > abs(b1)
    assert abs(b2) > abs(a2)

    correction = (b2 * 2. * (1. - sample.geometry.eps) /
                  sample.geometry.sma / sample.gradient)
    new_eps = sample.geometry.eps - correction

    # got closer to test data (eps=0.2)
    assert_allclose(new_eps, 0.21, atol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_small_radii():
    sample = EllipseSample(DATA, 2.)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    assert isophote.ndata == 13


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_eps():
    # initial guess is off in the eps parameter
    sample = EllipseSample(DATA, 40., eps=2*0.2)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()

    assert isinstance(isophote, Isophote)
    g = isophote.sample.geometry
    assert g.eps >= 0.19
    assert g.eps <= 0.21


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_pa():
    data = make_test_image(pa=np.pi/4, noise=0.01, random_state=123)

    # initial guess is off in the pa parameter
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert g.pa >= (np.pi/4 - 0.05)
    assert g.pa <= (np.pi/4 + 0.05)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_xy():
    pos = DEFAULT_POS - 5
    data = make_test_image(x0=pos, y0=pos, random_state=123)

    # initial guess is off in the x0 and y0 parameters
    sample = EllipseSample(data, 40)
    fitter = EllipseFitter(sample)
    isophote = fitter.fit()
    g = isophote.sample.geometry

    assert g.x0 >= (pos - 1)
    assert g.x0 <= (pos + 1)
    assert g.y0 >= (pos - 1)
    assert g.y0 <= (pos + 1)


@pytest.mark.skipif('not HAS_SCIPY')
def test_fitting_all():
    # build test image that is off from the defaults
    # assumed by the EllipseSample constructor.
    POS = DEFAULT_POS - 5
    ANGLE = np.pi / 4
    EPS = 2 * 0.2
    data = make_test_image(x0=POS, y0=POS, eps=EPS, pa=ANGLE,
                           random_state=123)
    sma = 60.

    # initial guess is off in all parameters. We find that the initial
    # guesses, especially for position angle, must be kinda close to the
    # actual value. 20% off max seems to work in this case of high SNR.
    sample = EllipseSample(data, sma, position_angle=(1.2 * ANGLE))
    fitter = EllipseFitter(sample)
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

    sample_m = EllipseSample(data, sma, position_angle=(1.2 * ANGLE),
                             integrmode=MEAN)
    fitter_m = EllipseFitter(sample_m)
    isophote_m = fitter_m.fit()

    assert isophote_m.stop_code == 0


@pytest.mark.remote_data
@pytest.mark.skipif('not HAS_SCIPY')
class TestM51:
    def setup_class(self):
        path = get_path('isophote/M51.fits', location='photutils-datasets',
                        cache=True)
        hdu = fits.open(path)
        self.data = hdu[0].data
        hdu.close()

    def test_m51(self):
        # here we evaluate the detailed convergence behavior
        # for a particular ellipse where we can see the eps
        # parameter jumping back and forth.
        # sample = EllipseSample(self.data, 13.31000001, eps=0.16,
        #                        position_angle=((-37.5+90)/180.*np.pi))
        # sample.update()
        # fitter = EllipseFitter(sample)
        # isophote = fitter.fit()

        # we start the fit with initial values taken from
        # previous isophote, as determined by the old code.

        # sample taken in high SNR region
        sample = EllipseSample(self.data, 21.44, eps=0.18,
                               position_angle=(36./180.*np.pi))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.ndata == 119
        assert_allclose(isophote.intens, 685.4, atol=0.1)

        # last sample taken by the original code, before turning inwards.
        sample = EllipseSample(self.data, 61.16, eps=0.219,
                               position_angle=((77.5+90)/180*np.pi))
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert isophote.ndata == 382
        assert_allclose(isophote.intens, 155.0, atol=0.1)

    def test_m51_outer(self):
        # sample taken at the outskirts of the image, so many
        # data points lay outside the image frame. This checks
        # for the presence of gaps in the sample arrays.
        sample = EllipseSample(self.data, 330., eps=0.2,
                               position_angle=((90)/180*np.pi),
                               integrmode='median')
        fitter = EllipseFitter(sample)
        isophote = fitter.fit()

        assert not np.any(isophote.sample.values[2] == 0)

    def test_m51_central(self):
        # this code finds central x and y offset by about 0.1 pixel wrt the
        # spp code. In here we use as input the position computed by this
        # code, thus this test is checking just the extraction algorithm.
        g = EllipseGeometry(257.02, 258.1, 0.0, 0.0, 0.0, 0.1, False)
        sample = CentralEllipseSample(self.data, 0.0, geometry=g)
        fitter = CentralEllipseFitter(sample)
        isophote = fitter.fit()

        # the central pixel intensity is about 3% larger than
        # found by the spp code.
        assert isophote.ndata == 1
        assert isophote.intens <= 7560.
        assert isophote.intens >= 7550.
