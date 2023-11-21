# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for DAOStarFinder.
"""

import itertools
import os.path as op

import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose

from photutils.datasets import make_100gaussians_image
from photutils.detection.daofinder import DAOStarFinder
from photutils.utils._optional_deps import HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning

DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0]
FWHMS = [1.0, 1.5, 2.0]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestDAOStarFinder:
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_daofind(self, threshold, fwhm):
        starfinder = DAOStarFinder(threshold, fwhm, sigma_radius=1.5)
        tbl = starfinder(DATA)
        datafn = f'daofind_test_thresh{threshold:04.1f}_fwhm{fwhm:04.1f}.txt'
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        tbl_ref = Table.read(datafn, format='ascii')

        assert tbl.colnames == tbl_ref.colnames
        for col in tbl.colnames:
            assert_allclose(tbl[col], tbl_ref[col])

    def test_daofind_threshold_fwhm_inputs(self):
        with pytest.raises(TypeError):
            DAOStarFinder(threshold=np.ones((2, 2)), fwhm=3.0)

        with pytest.raises(TypeError):
            DAOStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

    def test_daofind_include_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=False)
        tbl = starfinder(DATA)
        assert len(tbl) == 20

        # test when no detections
        starfinder = DAOStarFinder(threshold=100, fwhm=2, sigma_radius=1.5,
                                   exclude_border=False)
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            tbl = starfinder(DATA)
            assert tbl is None

    def test_daofind_exclude_border(self):
        starfinder = DAOStarFinder(threshold=10, fwhm=2, sigma_radius=1.5,
                                   exclude_border=True)
        tbl = starfinder(DATA)
        assert len(tbl) == 19

        # test when no detections
        starfinder = DAOStarFinder(threshold=100, fwhm=2, sigma_radius=1.5,
                                   exclude_border=True)
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            tbl = starfinder(DATA)
            assert tbl is None

    def test_daofind_nosources(self):
        data = np.ones((3, 3))
        with pytest.warns(NoDetectionsWarning, match='No sources were found'):
            starfinder = DAOStarFinder(threshold=10, fwhm=1)
            tbl = starfinder(data)
            assert tbl is None

    def test_daofind_sharpness(self):
        """Sources found, but none pass the sharpness criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, sharplo=1.0)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_daofind_roundness(self):
        """Sources found, but none pass the roundness criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, roundlo=1.0)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_daofind_peakmax(self):
        """Sources found, but none pass the peakmax criteria."""
        with pytest.warns(NoDetectionsWarning,
                          match='Sources were found, but none pass'):
            starfinder = DAOStarFinder(threshold=50, fwhm=1.0, peakmax=1.0)
            tbl = starfinder(DATA)
            assert tbl is None

    def test_daofind_flux_negative(self):
        """Test handling of negative flux (here created by large sky)."""
        data = np.ones((5, 5))
        data[2, 2] = 10.0
        starfinder = DAOStarFinder(threshold=0.1, fwhm=1.0, sky=10)
        tbl = starfinder(data)
        assert not np.isfinite(tbl['mag'])

    def test_daofind_negative_fit_peak(self):
        """
        Regression test that sources with negative fit peaks (i.e.,
        hx/hy<=0) are excluded.
        """
        starfinder = DAOStarFinder(threshold=7.0, fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf)
        tbl = starfinder(DATA)
        assert len(tbl) == 102

    def test_daofind_peakmax_filtering(self):
        """
        Regression test that objects with ``peak`` >= ``peakmax`` are
        filtered out.
        """
        peakmax = 20
        starfinder = DAOStarFinder(threshold=7.0, fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, peakmax=peakmax)
        tbl = starfinder(DATA)
        assert len(tbl) == 37
        assert all(tbl['peak'] < peakmax)

    def test_daofind_brightest_filtering(self):
        """
        Regression test that only top ``brightest`` objects are
        selected.
        """
        brightest = 40
        peakmax = 20
        starfinder = DAOStarFinder(threshold=7.0, fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, brightest=brightest)
        tbl = starfinder(DATA)
        # combined with peakmax
        assert len(tbl) == brightest
        starfinder = DAOStarFinder(threshold=7.0, fwhm=1.5, roundlo=-np.inf,
                                   roundhi=np.inf, sharplo=-np.inf,
                                   sharphi=np.inf, brightest=brightest,
                                   peakmax=peakmax)
        tbl = starfinder(DATA)
        assert len(tbl) == 37

    def test_daofind_mask(self):
        """Test DAOStarFinder with a mask."""
        starfinder = DAOStarFinder(threshold=10, fwhm=1.5)
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[100:200] = True
        tbl1 = starfinder(DATA)
        tbl2 = starfinder(DATA, mask=mask)
        assert len(tbl1) > len(tbl2)

    def test_inputs(self):
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=-1)
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=3.1)

    def test_xycoords(self):
        starfinder1 = DAOStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                    exclude_border=True)
        tbl1 = starfinder1(DATA)

        xycoords = np.array([[145, 169], [395, 187], [427, 211], [11, 224]])
        starfinder2 = DAOStarFinder(threshold=30, fwhm=2, sigma_radius=1.5,
                                    exclude_border=True, xycoords=xycoords)
        tbl2 = starfinder2(DATA)
        assert np.all(tbl1 == tbl2)

    def test_invalid_xycoords(self):
        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        with pytest.raises(ValueError):
            DAOStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

    def test_min_separation(self):
        threshold = 5
        fwhm = 1.0
        finder1 = DAOStarFinder(threshold, fwhm, sigma_radius=1.5)
        tbl1 = finder1(DATA)
        finder2 = DAOStarFinder(threshold, fwhm, sigma_radius=1.5,
                                min_separation=3.0)
        tbl2 = finder2(DATA)
        assert len(tbl1) > len(tbl2)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)
