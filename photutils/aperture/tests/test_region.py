# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the region module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord

from photutils.aperture.region import RegionalAperture, SkyRegionalAperture
from photutils.utils._optional_deps import HAS_MATPLOTLIB

X, Y = (10, 20)
POSITION = (X, Y)
PIXELCOORD = PixCoord(X, Y)
RA, DEC = X, Y
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class TestRegionalAperture():
    region = CirclePixelRegion(PIXELCOORD, radius=3.0)
    aperture = RegionalAperture(POSITION, region)

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
        with pytest.raises(ValueError):
            reg = CirclePixelRegion(PIXELCOORD, radius)
            RegionalAperture(POSITION, reg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.region.radius = 2.0
        assert aper != self.aperture


class TestSkyRegionalAperture():
    region = CircleSkyRegion(SKYCOORD, radius=3.0 * UNIT)
    aperture = SkyRegionalAperture(SKYCOORD, region)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        with pytest.raises(ValueError):
            reg = CircleSkyRegion(SKYCOORD, radius=radius * UNIT)
            SkyRegionalAperture(SKYCOORD, reg)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.region.radius = 2.0 * UNIT
        assert aper != self.aperture


@pytest.mark.parametrize('x,y,expected_overlap',
                         [(0, 0, 10.304636), (5, 5, np.pi * 9.0),
                          (50, 50, np.nan)])
def test_area_overlap(x, y, expected_overlap):
    data = np.ones((11, 11))
    region = CirclePixelRegion(PixCoord(x, y), 3.0)
    aper = RegionalAperture((x, y), region)
    area = aper.area_overlap(data)
    assert_allclose(area, expected_overlap)

    data2 = np.ones((11, 11)) * u.Jy
    area = aper.area_overlap(data2)
    assert not isinstance(area, u.Quantity)
    assert_allclose(area, expected_overlap)


@pytest.mark.parametrize('x,y,expected_overlap',
                         [(0, 0, 10.304636 - 2), (5, 5, np.pi * 9.0 - 2),
                          (50, 50, np.nan)])
def test_area_overlap_mask(x, y, expected_overlap):
    data = np.ones((11, 11))
    mask = np.zeros((11, 11), dtype=bool)
    mask[0, 0:2] = True
    mask[5, 5:7] = True
    region = CirclePixelRegion(PixCoord(x, y), 3.0)
    aper = RegionalAperture((x, y), region)
    area = aper.area_overlap(data, mask=mask)
    assert_allclose(area, expected_overlap)

    with pytest.raises(ValueError):
        mask = np.zeros((3, 3), dtype=bool)
        aper.area_overlap(data, mask=mask)


def test_area_overlap_different_centers():
    data = np.ones((11, 11))
    region = CirclePixelRegion(PixCoord(5, 5), 3.0)
    aper1 = RegionalAperture((5, 5), region)
    aper2 = RegionalAperture((6, 6), region)
    area1 = aper1.area_overlap(data)
    area2 = aper2.area_overlap(data)
    assert area1 == area2


def test_invalid_positions():
    region = CirclePixelRegion(PixCoord(1, 2), 3)
    with pytest.raises(ValueError):
        _ = RegionalAperture([], region)

    with pytest.raises(ValueError):
        _ = RegionalAperture([1], region)

    with pytest.raises(ValueError):
        _ = RegionalAperture([[1]], region)

    with pytest.raises(ValueError):
        _ = RegionalAperture([1, 2, 3], region)

    with pytest.raises(ValueError):
        _ = RegionalAperture([[1, 2, 3]], region)

    with pytest.raises(ValueError):
        _ = RegionalAperture([[1, 2], [3, 4]], region)

    with pytest.raises(TypeError):
        xypos = np.array((1, 2)) * u.pix
        _ = RegionalAperture(xypos, region)

    with pytest.raises(TypeError):
        x = 1 * u.pix
        y = 2
        _ = RegionalAperture((x, y), region)

    with pytest.raises(TypeError):
        x = 1 * u.pix
        y = 2 * u.pix
        _ = RegionalAperture((x, y), region)
