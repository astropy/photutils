# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for StarFinder.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_equal

from photutils.detection import StarFinder
from photutils.utils._optional_deps import HAS_SCIPY
from photutils.utils.exceptions import NoDetectionsWarning


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestStarFinder:
    def test_starfind(self, data, kernel):
        finder1 = StarFinder(1, kernel)
        finder2 = StarFinder(10, kernel)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert isinstance(tbl1, Table)
        assert len(tbl1) == 25
        assert len(tbl2) == 9

        # test with units
        unit = u.Jy
        finder3 = StarFinder(1 * unit, kernel)
        tbl3 = finder3(data << unit)
        assert isinstance(tbl3, Table)
        assert len(tbl3) == 25
        assert tbl3['flux'].unit == unit
        assert tbl3['max_value'].unit == unit
        for col in tbl3.colnames:
            if col not in ('flux', 'max_value'):
                assert_equal(tbl3[col], tbl1[col])

    def test_inputs(self, kernel):
        with pytest.raises(ValueError):
            StarFinder(1, kernel, min_separation=-1)
        with pytest.raises(ValueError):
            StarFinder(1, kernel, brightest=-1)
        with pytest.raises(ValueError):
            StarFinder(1, kernel, brightest=3.1)

    def test_nosources(self, data, kernel):
        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = StarFinder(100, kernel)
            tbl = finder(data)
            assert tbl is None

        data = np.ones((5, 5))
        data[2, 2] = 10.0
        with pytest.warns(NoDetectionsWarning, match=match):
            finder = StarFinder(1, kernel)
            tbl = finder(-data)
            assert tbl is None

    def test_min_separation(self, data, kernel):
        finder1 = StarFinder(1, kernel, min_separation=0)
        finder2 = StarFinder(1, kernel, min_separation=10)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert len(tbl1) == 25
        assert len(tbl2) == 20

    def test_peakmax(self, data, kernel):
        finder1 = StarFinder(1, kernel, peakmax=None)
        finder2 = StarFinder(1, kernel, peakmax=11)
        tbl1 = finder1(data)
        tbl2 = finder2(data)
        assert len(tbl1) == 25
        assert len(tbl2) == 16

        match = 'Sources were found, but none pass'
        with pytest.warns(NoDetectionsWarning, match=match):
            starfinder = StarFinder(10, kernel, peakmax=5)
            tbl = starfinder(data)
            assert tbl is None

    def test_brightest(self, data, kernel):
        finder = StarFinder(1, kernel, brightest=10)
        tbl = finder(data)
        assert len(tbl) == 10
        fluxes = tbl['flux']
        assert fluxes[0] == np.max(fluxes)

    def test_single_detected_source(self, data, kernel):
        finder = StarFinder(11.5, kernel, brightest=1)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl = finder(data, mask=mask)
        assert len(tbl) == 1

        # test slicing with scalar catalog to improve coverage
        cat = finder._get_raw_catalog(data, mask=mask)
        assert cat.isscalar
        flux = cat.flux[0]  # evaluate the flux so it can be sliced
        assert cat[0].flux == flux

    def test_mask(self, data, kernel):
        starfinder = StarFinder(1, kernel)
        mask = np.zeros(data.shape, dtype=bool)
        mask[0:50] = True
        tbl1 = starfinder(data)
        tbl2 = starfinder(data, mask=mask)
        assert len(tbl1) == 25
        assert len(tbl2) == 13
        assert min(tbl2['ycentroid']) > 50
