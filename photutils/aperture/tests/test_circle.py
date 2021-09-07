# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circle module.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
import pytest

from .test_aperture_common import BaseTestAperture
from ..circle import (CircularAperture, CircularAnnulus, SkyCircularAperture,
                      SkyCircularAnnulus)
from ...utils._optional_deps import HAS_MATPLOTLIB  # noqa


POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec


class TestCircularAperture(BaseTestAperture):
    aperture = CircularAperture(POSITIONS, r=3.)

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_plot(self):
        self.aperture.plot()

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_plot_returns_patches(self):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        my_patches = self.aperture.plot()
        assert isinstance(my_patches, list)
        for patch in my_patches:
            assert isinstance(patch, Patch)

        # test creating a legend with these patches
        plt.legend(my_patches, list(range(len(my_patches))))


class TestCircularAnnulus(BaseTestAperture):
    aperture = CircularAnnulus(POSITIONS, r_in=3., r_out=7.)

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_plot(self):
        self.aperture.plot()

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
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


class TestSkyCircularAperture(BaseTestAperture):
    aperture = SkyCircularAperture(SKYCOORD, r=3.*UNIT)


class TestSkyCircularAnnulus(BaseTestAperture):
    aperture = SkyCircularAnnulus(SKYCOORD, r_in=3.*UNIT, r_out=7.*UNIT)


def test_slicing():
    xypos = [(10, 10), (20, 20), (30, 30)]
    aper1 = CircularAperture(xypos, r=3)
    aper2 = aper1[0:2]
    assert len(aper2) == 2

    aper3 = aper1[0]
    assert aper3.isscalar
    with pytest.raises(TypeError):
        len(aper3)

    with pytest.raises(TypeError):
        _ = aper3[0]


def test_area_overlap():
    data = np.ones((11, 11))
    xypos = [(0, 0), (5, 5), (50, 50)]
    aper = CircularAperture(xypos, r=3)
    areas = aper.area_overlap(data)
    assert_allclose(areas, [10.304636, np.pi*9., np.nan])

    aper2 = CircularAperture(xypos[1], r=3)
    area2 = aper2.area_overlap(data)
    assert_allclose(area2, np.pi * 9.)


def test_area_overlap_mask():
    data = np.ones((11, 11))
    mask = np.zeros((11, 11), dtype=bool)
    mask[0, 0:2] = True
    mask[5, 5:7] = True
    xypos = [(0, 0), (5, 5), (50, 50)]
    aper = CircularAperture(xypos, r=3)
    areas = aper.area_overlap(data, mask=mask)
    areas_exp = np.array([10.304636, np.pi*9., np.nan]) - 2.
    assert_allclose(areas, areas_exp)

    with pytest.raises(ValueError):
        mask = np.zeros((3, 3), dtype=bool)
        aper.area_overlap(data, mask=mask)
