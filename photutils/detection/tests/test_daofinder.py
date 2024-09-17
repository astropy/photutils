# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for DAOStarFinder.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.detection.daofinder import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning


class TestDAOStarFinder:
    def test_daostarfind(self, data):
        units = u.Jy
        threshold = 5.0
        fwhm = 1.0
        finder0 = DAOStarFinder(threshold, fwhm)
        finder1 = DAOStarFinder(threshold * units, fwhm)

        tbl0 = finder0(data)
        tbl1 = finder1(data << units)
        assert_array_equal(tbl0, tbl1)

        assert np.min(tbl0['flux']) > 150

        # test that sources are returned with threshold = 0
        finder = DAOStarFinder(0, fwhm)
        tbl = finder(data)
        assert len(tbl) == 25

    def test_daofind_inputs(self):
        match = 'threshold must be a scalar value'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=np.ones((2, 2)), fwhm=3.0)

        match = 'fwhm must be a scalar value'
        with pytest.raises(TypeError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=np.ones((2, 2)))

        match = 'fwhm must be positive'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=-10)

        match = 'ratio must be positive and less or equal than 1'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, ratio=-10)

        match = 'sigma_radius must be positive'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=3.0, fwhm=2, sigma_radius=-10)

        match = 'brightest must be >= 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=-1)

        match = 'brightest must be an integer'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, brightest=3.1)

        xycoords = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        match = 'xycoords must be shaped as a Nx2 array'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, xycoords=xycoords)

    def test_daofind_nosources(self, data):
        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = DAOStarFinder(threshold=100, fwhm=2)
            tbl = finder(data)
            assert tbl is None

        with pytest.warns(NoDetectionsWarning, match=match):
            finder = DAOStarFinder(threshold=1, fwhm=2)
            tbl = finder(-data)
            assert tbl is None

    def test_daofind_exclude_border(self):
        data = np.zeros((9, 9))
        data[0, 0] = 1
        data[2, 2] = 1
        data[4, 4] = 1
        data[6, 6] = 1

        finder0 = DAOStarFinder(threshold=0.1, fwhm=0.5, exclude_border=False)
        finder1 = DAOStarFinder(threshold=0.1, fwhm=0.5, exclude_border=True)
        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)

    def test_daofind_sharpness(self, data):
        """
        Sources found, but none pass the sharpness criteria.
        """
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = DAOStarFinder(threshold=1, fwhm=1.0, sharplo=1.0)
            tbl = finder(data)
            assert tbl is None

    def test_daofind_roundness(self, data):
        """
        Sources found, but none pass the roundness criteria.
        """
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = DAOStarFinder(threshold=1, fwhm=1.0, roundlo=1.0)
            tbl = finder(data)
            assert tbl is None

    def test_daofind_peakmax(self, data):
        """
        Sources found, but none pass the peakmax criteria.
        """
        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = DAOStarFinder(threshold=1, fwhm=1.0, peakmax=1.0)
            tbl = finder(data)
            assert tbl is None

    def test_daofind_peakmax_filtering(self, data):
        """
        Regression test that objects with peak >= peakmax are filtered
        out.
        """
        peakmax = 8
        finder0 = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                                roundhi=np.inf, sharplo=-np.inf,
                                sharphi=np.inf)
        finder1 = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                                roundhi=np.inf, sharplo=-np.inf,
                                sharphi=np.inf, peakmax=peakmax)

        tbl0 = finder0(data)
        tbl1 = finder1(data)
        assert len(tbl0) > len(tbl1)
        assert all(tbl1['peak'] < peakmax)

    def test_daofind_brightest_filtering(self, data):
        """
        Regression test that only top ``brightest`` objects are
        selected.
        """
        brightest = 10
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                               roundhi=np.inf, sharplo=-np.inf,
                               sharphi=np.inf, brightest=brightest)
        tbl = finder(data)
        assert len(tbl) == brightest

        # combined with peakmax
        peakmax = 8
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5, roundlo=-np.inf,
                               roundhi=np.inf, sharplo=-np.inf,
                               sharphi=np.inf, brightest=brightest,
                               peakmax=peakmax)
        tbl = finder(data)
        assert len(tbl) == 5

    def test_daofind_mask(self, data):
        """
        Test DAOStarFinder with a mask.
        """
        finder = DAOStarFinder(threshold=1.0, fwhm=1.5)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50, :] = True
        tbl0 = finder(data)
        tbl1 = finder(data, mask=mask)
        assert len(tbl0) > len(tbl1)

    def test_xycoords(self, data):
        finder0 = DAOStarFinder(threshold=8.0, fwhm=2)
        tbl0 = finder0(data)
        xycoords = list(zip(tbl0['xcentroid'], tbl0['ycentroid'], strict=True))
        xycoords = np.round(xycoords).astype(int)

        finder1 = DAOStarFinder(threshold=8.0, fwhm=2, xycoords=xycoords)
        tbl1 = finder1(data)
        assert_array_equal(tbl0, tbl1)

    def test_min_separation(self, data):
        threshold = 1.0
        fwhm = 1.0
        finder1 = DAOStarFinder(threshold, fwhm)
        tbl1 = finder1(data)
        finder2 = DAOStarFinder(threshold, fwhm, min_separation=10.0)
        tbl2 = finder2(data)
        assert len(tbl1) > len(tbl2)

        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            DAOStarFinder(threshold=10, fwhm=1.5, min_separation=-1.0)

    def test_single_detected_source(self, data):
        finder = DAOStarFinder(7.9, 2, brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux
