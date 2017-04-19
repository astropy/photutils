from __future__ import (absolute_import, division, print_function, unicode_literals)

import pytest

import numpy as np
from astropy.io import fits

from photutils.isophote import build_test_data
from photutils.isophote.build_test_data import DEFAULT_POS, DEFAULT_SIZE
from photutils.isophote.geometry import Geometry, DEFAULT_EPS
from photutils.isophote.fitter import NORMAL_FIT, TOO_MANY_FLAGGED
from photutils.isophote.ellipse import Ellipse, FIXED_ELLIPSE, FAILED_FIT
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.tests.test_data import TEST_DATA

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# define an off-center position and a tilted sma
POS = DEFAULT_POS + DEFAULT_SIZE / 4
PA = 10. / 180. * np.pi

# build off-center test data. It's fine to have a single np array
# to use in all tests that need it, but do not use a single instance
# of Geometry. The code may eventually modify it's contents. The safe
# bet is to build it wherever it's needed. The cost is negligible.
OFFSET_GALAXY = build_test_data.build(x0=POS, y0=POS, pa=PA, noise=1.E-12)

verb = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestEllipse(object):

    def test_basic(self):
        # centered, tilted galaxy.
        test_data = build_test_data.build(pa=PA)

        ellipse = Ellipse(test_data, verbose=verb)
        isophote_list = ellipse.fit_image(verbose=verb)

        assert isinstance(isophote_list, IsophoteList)
        assert len(isophote_list) > 1
        assert isinstance(isophote_list[0], Isophote)

        # verify that the list is properly sorted in sem-major axis length
        assert isophote_list[-1] > isophote_list[0]

        # the fit should stop where gradient looses reliability.
        assert len(isophote_list) == 67
        assert isophote_list[-1].stop_code == FAILED_FIT

    def test_fit_one_ellipse(self):
        test_data = build_test_data.build(pa=PA)

        ellipse = Ellipse(test_data, verbose=verb)
        isophote = ellipse.fit_isophote(40.)

        assert isinstance(isophote, Isophote)
        assert isophote.valid

    def test_offcenter_fail(self):
        # A first guess ellipse that is centered in the image frame.
        # This should result in failure since the real galaxy
        # image is off-center by a large offset.
        ellipse = Ellipse(OFFSET_GALAXY, verbose=verb)
        isophote_list = ellipse.fit_image(verbose=verb)

        assert len(isophote_list) == 0

    def test_offcenter_fit(self):
        # A first guess ellipse that is roughly centered on the
        # offset galaxy image.
        g = Geometry(POS+5, POS+5, 10., DEFAULT_EPS, PA, 0.1, verb)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g, verbose=verb)
        isophote_list = ellipse.fit_image(verbose=verb)

        # the fit should stop when too many potential sample
        # points fall outside the image frame.
        assert len(isophote_list) == 63
        assert isophote_list[-1].stop_code == TOO_MANY_FLAGGED

    def test_offcenter_go_beyond_frame(self):
        # Same as before, but now force the fit to goo
        # beyond the image frame limits.
        g = Geometry(POS+5, POS+5, 10., DEFAULT_EPS, PA, 0.1, verb)
        ellipse = Ellipse(OFFSET_GALAXY, geometry=g, verbose=verb)
        isophote_list = ellipse.fit_image(maxsma=400., verbose=verb)

        # the fit should go to maxsma, but with fixed geometry
        assert len(isophote_list) == 71
        assert isophote_list[-1].stop_code == FIXED_ELLIPSE


@pytest.mark.skipif('not HAS_SCIPY')
class TestOnRealData(object):

    def test_basic(self):
        image = fits.open(TEST_DATA + "M105-S001-RGB.fits")
        test_data = image[0].data[0]

        g = Geometry(530., 511, 50., 0.2, 20./180.*3.14)

        ellipse = Ellipse(test_data, geometry=g, verbose=verb)
        isophote_list = ellipse.fit_image(verbose=verb)

        assert len(isophote_list) == 58

        # check that isophote at about sma=70 got an uneventful fit
        assert isophote_list.get_closest(70.).stop_code == NORMAL_FIT
