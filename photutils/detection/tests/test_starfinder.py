# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for StarFinder.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D

from photutils.datasets import make_100gaussians_image
from photutils.detection.starfinder import StarFinder
from photutils.utils._optional_deps import HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning

DATA = make_100gaussians_image()
y, x = np.mgrid[0:25, 0:25]
g = Gaussian2D(1, 12, 12, 3, 2, theta=np.pi / 6.0)
PSF = g(x, y)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestStarFinder:
    def test_starfind(self):
        finder1 = StarFinder(10, PSF)
        finder2 = StarFinder(30, PSF)
        tbl1 = finder1(DATA)
        tbl2 = finder2(DATA)
        assert len(tbl1) > len(tbl2)

    def test_inputs(self):
        with pytest.raises(ValueError):
            StarFinder(10, PSF, min_separation=-1)
        with pytest.raises(ValueError):
            StarFinder(10, PSF, brightest=-1)
        with pytest.raises(ValueError):
            StarFinder(10, PSF, brightest=3.1)

    def test_nosources(self):
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            finder = StarFinder(100, PSF)
            tbl = finder(DATA)
            assert tbl is None

    def test_min_separation(self):
        finder1 = StarFinder(10, PSF, min_separation=0)
        finder2 = StarFinder(10, PSF, min_separation=50)
        tbl1 = finder1(DATA)
        tbl2 = finder2(DATA)
        assert len(tbl1) > len(tbl2)

    def test_peakmax(self):
        finder1 = StarFinder(10, PSF, peakmax=None)
        finder2 = StarFinder(10, PSF, peakmax=50)
        tbl1 = finder1(DATA)
        tbl2 = finder2(DATA)
        assert len(tbl1) > len(tbl2)

        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = StarFinder(10, PSF, peakmax=5)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_brightest(self):
        finder = StarFinder(10, PSF, brightest=10)
        tbl = finder(DATA)
        assert len(tbl) == 10
        fluxes = tbl['flux']
        assert fluxes[0] == np.max(fluxes)

        finder = StarFinder(40, PSF, peakmax=120)
        tbl = finder(DATA)
        assert len(tbl) == 1

    def test_mask(self):
        starfinder = StarFinder(10, PSF)
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[0:100] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)
        assert min(tbl2['ycentroid']) > 100
