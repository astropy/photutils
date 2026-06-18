# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the circle module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture import PolygonAperture, SkyPolygonAperture
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

        # Test creating a legend with these patches
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

        # Test creating a legend with these patches
        labels = list(range(len(my_patches)))
        _, ax = plt.subplots()
        ax.legend(my_patches, labels)

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'r_in' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            CircularAnnulus(POSITIONS, r_in=radius, r_out=7.0)

        match = "'r_out' must be greater than 'r_in'"
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

        match = "'r_out' must be greater than 'r_in'"
        with pytest.raises(ValueError, match=match):
            SkyCircularAnnulus(SKYCOORD, r_in=3.0 * UNIT, r_out=radius * UNIT)

    def test_copy_eq(self):
        aper = self.aperture.copy()
        assert aper == self.aperture
        aper.r_in = 2.0 * UNIT
        assert aper != self.aperture

    @staticmethod
    def test_r_out_less_than_r_in():
        """
        Test that a ValueError is raised when r_out <= r_in.
        """
        match = "'r_out' must be greater than 'r_in'"
        with pytest.raises(ValueError, match=match):
            SkyCircularAnnulus(SKYCOORD, r_in=7.0 * UNIT, r_out=3.0 * UNIT)

    @staticmethod
    def test_non_angle_quantity():
        """
        Test that a ValueError is raised when r_in has non-angular
        units.
        """
        match = "'r_in' must have angular units"
        with pytest.raises(ValueError, match=match):
            SkyCircularAnnulus(SKYCOORD, r_in=0.5 * u.pix, r_out=7.0 * u.pix)


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


def test_circular_aperture_to_polygon():
    aper = CircularAperture((10.0, 20.0), r=5.0)
    poly = aper.to_polygon(n_vertices=200)
    assert isinstance(poly, PolygonAperture)
    assert poly.n_vertices == 200
    assert poly.is_regular
    assert abs(poly.area - aper.area) / aper.area < 1e-3


def test_circular_aperture_to_polygon_multi_position():
    aper = CircularAperture([(10.0, 20.0), (30.0, 40.0)], r=3.0)
    poly = aper.to_polygon(n_vertices=50)
    assert poly.vertices.shape == (2, 50, 2)


def test_circular_aperture_to_polygon_invalid_n_vertices():
    aper = CircularAperture((0.0, 0.0), r=1.0)
    match = 'n_vertices must be at least 3'
    with pytest.raises(ValueError, match=match):
        aper.to_polygon(n_vertices=2)


def test_sky_circular_aperture_to_polygon():
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyCircularAperture(pos, r=5.0 * u.arcsec)
    poly = aper.to_polygon(n_vertices=200)
    assert isinstance(poly, SkyPolygonAperture)
    assert poly.n_vertices == 200
    assert poly.is_regular
    assert_allclose(poly.outer_radius.to_value(u.arcsec), 5.0)
    assert poly.vertices.shape == (200,)


def test_sky_circular_aperture_to_polygon_multi_position():
    pos = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    aper = SkyCircularAperture(pos, r=3.0 * u.arcsec)
    poly = aper.to_polygon(n_vertices=50)
    assert poly.vertices.shape == (2, 50)


def test_sky_circular_aperture_to_polygon_invalid_n_vertices():
    pos = SkyCoord(ra=0.0, dec=0.0, unit='deg')
    aper = SkyCircularAperture(pos, r=1.0 * u.arcsec)
    match = 'n_vertices must be at least 3'
    with pytest.raises(ValueError, match=match):
        aper.to_polygon(n_vertices=2)


def _make_round_trip_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [10.0, 30.0]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


def test_sky_circular_aperture_to_polygon_wcs_round_trip():
    """
    Test that converting a SkyCircularAperture to a polygon and then to
    pixels is equivalent to converting to pixels and then to a polygon.
    """
    wcs = _make_round_trip_wcs()
    pos = SkyCoord(ra=10.0, dec=30.0, unit='deg')
    aper = SkyCircularAperture(pos, r=5.0 * u.arcsec)
    poly_from_sky = aper.to_polygon(n_vertices=200).to_pixel(wcs)
    poly_from_pix = aper.to_pixel(wcs).to_polygon(n_vertices=200)
    assert_allclose(poly_from_sky.area, poly_from_pix.area, rtol=1e-6)
    assert_allclose(poly_from_sky._xy_extents, poly_from_pix._xy_extents,
                    rtol=1e-6)


def test_circular_aperture_to_polygon_to_sky_round_trip():
    """
    Test that converting a CircularAperture to a polygon and then to sky
    is equivalent to converting to sky and then to a polygon.
    """
    # The bounding-box orientation can differ at the ~1e-5 level because
    # the sky shape parameters from ``to_sky`` are derived from a local
    # linear (SVD) approximation, while the polygon ``to_sky`` maps each
    # vertex exactly, so only rotation-invariant quantities are compared.
    wcs = _make_round_trip_wcs()
    aper = CircularAperture((50.0, 50.0), r=5.0)
    sky_from_poly = aper.to_polygon(n_vertices=200).to_sky(wcs)
    sky_from_aper = aper.to_sky(wcs).to_polygon(n_vertices=200)
    assert_allclose(sky_from_poly.positions.ra.deg,
                    sky_from_aper.positions.ra.deg)
    assert_allclose(sky_from_poly.positions.dec.deg,
                    sky_from_aper.positions.dec.deg)
    assert_allclose(sky_from_poly.perimeter.to_value(u.arcsec),
                    sky_from_aper.perimeter.to_value(u.arcsec), rtol=1e-6)
    assert_allclose(sky_from_poly.to_pixel(wcs).area,
                    sky_from_aper.to_pixel(wcs).area, rtol=1e-6)
