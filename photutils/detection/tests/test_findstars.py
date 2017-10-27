# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os.path as op
import itertools
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from ..findstars import DAOStarFinder, IRAFStarFinder
from ...datasets import make_100gaussians_image

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage    # noqa
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0]
FWHMS = [1.0, 1.5, 2.0]
warnings.simplefilter('always', AstropyUserWarning)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestDAOStarFinder(object):
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_daofind(self, threshold, fwhm):
        starfinder = DAOStarFinder(threshold, fwhm, sigma_radius=1.5)
        t = starfinder(DATA)
        datafn = ('daofind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')
        assert_allclose(np.array(t).astype(np.float),
                        np.array(t_ref).astype(np.float))

    def test_daofind_include_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=False)
        t = starfinder(DATA)
        assert len(t) == 20

    def test_daofind_exclude_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=True)
        t = starfinder(DATA)
        assert len(t) == 19

    def test_daofind_nosources(self):
        data = np.ones((3, 3))
        starfinder = DAOStarFinder(threshold=10, fwhm=1)
        t = starfinder(data)
        assert len(t) == 0

    def test_daofind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        starfinder = DAOStarFinder(threshold=50, fwhm=1.0, sharplo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_daofind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        starfinder = DAOStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_daofind_flux_negative(self):
        """Test handling of negative flux (here created by large sky)."""
        data = np.ones((5, 5))
        data[2, 2] = 10.
        starfinder = DAOStarFinder(threshold=0.1, fwhm=1.0, sky=10)
        t = starfinder(data)
        assert not np.isfinite(t['mag'])


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestIRAFStarFinder(object):
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_irafstarfind(self, threshold, fwhm):
        starfinder = IRAFStarFinder(threshold, fwhm, sigma_radius=1.5)
        t = starfinder(DATA)
        datafn = ('irafstarfind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')
        assert_allclose(np.array(t).astype(np.float),
                        np.array(t_ref).astype(np.float))

    def test_irafstarfind_nosources(self):
        data = np.ones((3, 3))
        starfinder = IRAFStarFinder(threshold=10, fwhm=1)
        t = starfinder(data)
        assert len(t) == 0

    def test_irafstarfind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, sharplo=2.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_irafstarfind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
        t = starfinder(DATA)
        assert len(t) == 0

    def test_irafstarfind_sky(self):
        starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=10.)
        t = starfinder(DATA)
        assert len(t) == 4

    def test_irafstarfind_largesky(self):
        starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=100.)
        t = starfinder(DATA)
        assert len(t) == 0
