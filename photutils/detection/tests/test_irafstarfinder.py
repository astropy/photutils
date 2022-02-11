# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for IRAFStarFinder.
"""

import itertools
import os.path as op

from astropy.table import Table
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..irafstarfinder import IRAFStarFinder
from ...datasets import make_100gaussians_image
from ...utils.exceptions import NoDetectionsWarning
from ...utils._optional_deps import HAS_SCIPY  # noqa


DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0, 8.*np.ones_like(DATA)]
FWHMS = [1.0, 1.5, 2.0]


@pytest.mark.skipif('not HAS_SCIPY')
class TestIRAFStarFinder:
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_irafstarfind(self, threshold, fwhm):
        starfinder = IRAFStarFinder(threshold, fwhm, sigma_radius=1.5)
        tbl = starfinder(DATA)
        if not np.isscalar(threshold):
            threshold = threshold.flatten()[0]
        datafn = (f'irafstarfind_test_thresh{threshold:04.1f}_'
                  f'fwhm{fwhm:04.1f}.txt')
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        tbl_ref = Table.read(datafn, format='ascii')

        assert tbl.colnames == tbl_ref.colnames
        for col in tbl.colnames:
            assert_allclose(tbl[col], tbl_ref[col])

    def test_irafstarfind_fwhm_input(self):
        with pytest.raises(TypeError):
            IRAFStarFinder(threshold=3., fwhm=np.ones((2, 2)))

    def test_irafstarfind_nosources(self):
        data = np.ones((3, 3))
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            starfinder = IRAFStarFinder(threshold=10, fwhm=1)
            tbl = starfinder(data)
            assert tbl is None

    def test_irafstarfind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, sharplo=2.)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_irafstarfind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, roundlo=1.)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_irafstarfind_peakmax(self):
        """Sources found, but none pass the peakmax criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = IRAFStarFinder(threshold=50, fwhm=1.0, peakmax=1.)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_irafstarfind_sky(self):
        starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=10.)
        tbl = starfinder(DATA)
        assert len(tbl) == 4

    def test_irafstarfind_largesky(self):
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = IRAFStarFinder(threshold=25.0, fwhm=2.0, sky=100.)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_irafstarfind_peakmax_filtering(self):
        """
        Regression test that objects with ``peak`` >= ``peakmax`` are
        filtered out.
        """
        peakmax = 20
        starfinder = IRAFStarFinder(threshold=7., fwhm=2, roundlo=-np.inf,
                                    roundhi=np.inf, sharplo=-np.inf,
                                    sharphi=np.inf, peakmax=peakmax)
        tbl = starfinder(DATA)
        assert len(tbl) == 117
        assert all(tbl['peak'] < peakmax)

    def test_irafstarfind_brightest_filtering(self):
        """
        Regression test that only top ``brightest`` objects are selected.
        """
        brightest = 40
        starfinder = IRAFStarFinder(threshold=7., fwhm=2, roundlo=-np.inf,
                                    roundhi=np.inf, sharplo=-np.inf,
                                    sharphi=np.inf, brightest=brightest)
        tbl = starfinder(DATA)
        assert len(tbl) == brightest

    def test_irafstarfind_mask(self):
        """Test IRAFStarFinder with a mask."""

        starfinder = IRAFStarFinder(threshold=10, fwhm=1.5)
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[100:200] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)

    def test_inputs(self):
        with pytest.raises(ValueError):
            IRAFStarFinder(10, 1.5, brightest=-1)
        with pytest.raises(ValueError):
            IRAFStarFinder(10, 1.5, brightest=3.1)

    def test_xycoords(self):
        starfinder1 = IRAFStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                     exclude_border=True)
        tbl1 = starfinder1(DATA)

        xycoords = np.array([[145, 169], [395, 187], [427, 211], [11, 224]])
        starfinder2 = IRAFStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                     exclude_border=True, xycoords=xycoords)
        tbl2 = starfinder2(DATA)
        assert np.all(tbl1 == tbl2)

    def test_invalid_xycoords(self):
        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with pytest.raises(ValueError):
            IRAFStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)
