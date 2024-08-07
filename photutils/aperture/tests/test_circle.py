# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circle module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAnnulus, SkyCircularAperture)
from photutils.aperture.tests.test_aperture_common import BaseTestAperture
from photutils.utils._optional_deps import HAS_MATPLOTLIB

POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class TestCircularAperture(BaseTestAperture):
    aperture = CircularAperture(POSITIONS, r=3.0)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self):
        self.aperture.plot()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_returns_patches(self):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        my_patches = self.aperture.plot()
        assert isinstance(my_patches, list)
        for patch in my_patches:
            assert isinstance(patch, Patch)

        # test creating a legend with these patches
        plt.legend(my_patches, list(range(len(my_patches))))

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'r' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            CircularAperture(POSITIONS, radius)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.r = 2.0
        assert aper != self.aperture


class TestCircularAnnulus(BaseTestAperture):
    aperture = CircularAnnulus(POSITIONS, r_in=3.0, r_out=7.0)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self):
        self.aperture.plot()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_returns_patches(self):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        my_patches = self.aperture.plot()
        assert isinstance(my_patches, list)

        for p in my_patches:
            assert isinstance(p, Patch)

        # make sure I can create a legend with these patches
        labels = list(range(len(my_patches)))
        plt.legend(my_patches, labels)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'r_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            CircularAnnulus(POSITIONS, r_in=radius, r_out=7.0)

        match = 'r_out must be greater than r_in'
        with pytest.raises(ValueError, match=match):
            CircularAnnulus(POSITIONS, r_in=3.0, r_out=radius)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.r_in = 2.0
        assert aper != self.aperture


class TestSkyCircularAperture(BaseTestAperture):
    aperture = SkyCircularAperture(SKYCOORD, r=3.0 * UNIT)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'r' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyCircularAperture(SKYCOORD, r=radius * UNIT)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.r = 2.0 * UNIT
        assert aper != self.aperture


class TestSkyCircularAnnulus(BaseTestAperture):
    aperture = SkyCircularAnnulus(SKYCOORD, r_in=3.0 * UNIT, r_out=7.0 * UNIT)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'r_in' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyCircularAnnulus(SKYCOORD, r_in=radius * UNIT, r_out=7.0 * UNIT)

        match = "'r_out' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyCircularAnnulus(SKYCOORD, r_in=3.0 * UNIT, r_out=radius * UNIT)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.r_in = 2.0 * UNIT
        assert aper != self.aperture


def test_slicing():
    xypos = [(10, 10), (20, 20), (30, 30)]
    aper1 = CircularAperture(xypos, r=3)
    aper2 = aper1[0:2]
    assert len(aper2) == 2

    aper3 = aper1[0]
    assert aper3.isscalar
    match = "A scalar 'CircularAperture' object has no len"
    with pytest.raises(TypeError, match=match):
        len(aper3)

    match = "A scalar 'CircularAperture' object cannot be indexed"
    with pytest.raises(TypeError, match=match):
        _ = aper3[0]


def test_area_overlap():
    data = np.ones((11, 11))
    xypos = [(0, 0), (5, 5), (50, 50)]
    aper = CircularAperture(xypos, r=3)
    areas = aper.area_overlap(data)
    assert_allclose(areas, [10.304636, np.pi * 9.0, np.nan])

    data2 = np.ones((11, 11)) * u.Jy
    areas = aper.area_overlap(data2)
    assert not isinstance(areas[0], u.Quantity)
    assert_allclose(areas, [10.304636, np.pi * 9.0, np.nan])

    aper2 = CircularAperture(xypos[1], r=3)
    area2 = aper2.area_overlap(data)
    assert_allclose(area2, np.pi * 9.0)

    area2 = aper2.area_overlap(data2)
    assert not isinstance(area2, u.Quantity)
    assert_allclose(area2, np.pi * 9.0)


def test_area_overlap_mask():
    data = np.ones((11, 11))
    mask = np.zeros((11, 11), dtype=bool)
    mask[0, 0:2] = True
    mask[5, 5:7] = True
    xypos = [(0, 0), (5, 5), (50, 50)]
    aper = CircularAperture(xypos, r=3)
    areas = aper.area_overlap(data, mask=mask)
    areas_exp = np.array([10.304636, np.pi * 9.0, np.nan]) - 2.0
    assert_allclose(areas, areas_exp)

    mask = np.zeros((3, 3), dtype=bool)
    match = 'mask and data must have the same shape'
    with pytest.raises(ValueError, match=match):
        aper.area_overlap(data, mask=mask)


def test_invalid_positions():
    match = r"'positions' must be a \(x, y\) pixel position or a list"
    with pytest.raises(ValueError, match=match):
        _ = CircularAperture([], r=3)

    with pytest.raises(ValueError, match=match):
        _ = CircularAperture([1], r=3)

    with pytest.raises(ValueError, match=match):
        _ = CircularAperture([[1]], r=3)

    with pytest.raises(ValueError, match=match):
        _ = CircularAperture([1, 2, 3], r=3)

    with pytest.raises(ValueError, match=match):
        _ = CircularAperture([[1, 2, 3]], r=3)

    x = np.arange(3)
    y = np.arange(3)
    xypos = np.transpose((x, y)) * u.pix
    match = "'positions' must not be a Quantity"
    with pytest.raises(TypeError, match=match):
        _ = CircularAperture(xypos, r=3)

    x = np.arange(3) * u.pix
    y = np.arange(3)
    xypos = zip(x, y, strict=True)
    with pytest.raises(TypeError, match=match):
        _ = CircularAperture(xypos, r=3)

    x = np.arange(3) * u.pix
    y = np.arange(3) * u.pix
    xypos = zip(x, y, strict=True)
    with pytest.raises(TypeError, match=match):
        _ = CircularAperture(xypos, r=3)
