# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the mask module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.circle import CircularAnnulus, CircularAperture
from photutils.aperture.ellipse import EllipticalAnnulus
from photutils.aperture.mask import ApertureMask
from photutils.aperture.rectangle import RectangularAnnulus

# Positions with no or partial overlap with a 50x50 array
POSITIONS = [(-20, -20), (-20, 20), (20, -20), (60, 60)]


class TestApertureMaskConstruction:
    """
    Tests for creating ApertureMask objects and converting them to
    arrays.
    """

    def test_input_shapes(self):
        mask_data = np.ones((10, 10))
        bbox = BoundingBox(5, 10, 5, 10)
        match = 'mask data and bounding box must have the same shape'
        with pytest.raises(ValueError, match=match):
            ApertureMask(mask_data, bbox)

    def test_array(self):
        mask_data = np.ones((10, 10))
        bbox = BoundingBox(5, 15, 5, 15)
        mask = ApertureMask(mask_data, bbox)
        data = np.array(mask)
        assert_allclose(data, mask.data)

    def test_copy(self):
        bbox = BoundingBox(5, 15, 5, 15)

        mask = ApertureMask(np.ones((10, 10)), bbox)
        mask_copy = np.array(mask, copy=True)
        mask_copy[0, 0] = 100.0
        assert mask.data[0, 0] == 1.0

        mask = ApertureMask(np.ones((10, 10)), bbox)
        mask_copy = np.array(mask, copy=False)
        mask_copy[0, 0] = 100.0
        assert mask.data[0, 0] == 100.0

        # No copy: copy=None returns a copy only if __array__ returns a
        # copy (copy=None was introduced in NumPy 2.0)
        mask = ApertureMask(np.ones((10, 10)), bbox)
        mask_copy = np.array(mask, copy=None)
        mask_copy[0, 0] = 100.0
        assert mask.data[0, 0] == 100.0

        # No copy
        mask = ApertureMask(np.ones((10, 10)), bbox)
        mask_copy = np.asarray(mask)
        mask_copy[0, 0] = 100.0
        assert mask.data[0, 0] == 100.0

        # Needs to copy because of the dtype change
        mask = ApertureMask(np.ones((10, 10)), bbox)
        mask_copy = np.asarray(mask, dtype=int)
        mask_copy[0, 0] = 100.0
        assert mask.data[0, 0] == 1.0


class TestCutout:
    """
    Tests for the ApertureMask cutout, to_image, and overlap-slice
    methods.
    """

    def test_get_overlap_slices(self):
        aper = CircularAperture((5, 5), r=10.0)
        mask = aper.to_mask()
        slc = ((slice(0, 16, None), slice(0, 16, None)),
               (slice(5, 21, None), slice(5, 21, None)))
        assert mask.get_overlap_slices((25, 25)) == slc

    def test_cutout_shape(self):
        mask_data = np.ones((10, 10))
        bbox = BoundingBox(5, 15, 5, 15)
        mask = ApertureMask(mask_data, bbox)

        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            mask.cutout(np.arange(10))

        match = 'input shape must have 2 elements'
        with pytest.raises(ValueError, match=match):
            mask.to_image((10,))

    def test_cutout_copy(self):
        data = np.ones((50, 50))
        aper = CircularAperture((25, 25), r=10.0)
        mask = aper.to_mask()
        cutout = mask.cutout(data, copy=True)
        data[25, 25] = 100.0
        assert cutout[10, 10] == 1.0

        # Test quantity data
        data2 = np.ones((50, 50)) * u.adu
        cutout2 = mask.cutout(data2, copy=True)
        assert cutout2.unit == data2.unit
        data2[25, 25] = 100.0 * u.adu
        assert cutout2[10, 10].value == 1.0

    @pytest.mark.parametrize('position', POSITIONS)
    def test_cutout_no_overlap(self, position):
        data = np.ones((50, 50))
        aper = CircularAperture(position, r=10.0)
        mask = aper.to_mask()

        cutout = mask.cutout(data)
        assert cutout is None

        weighted_data = mask.multiply(data)
        assert weighted_data is None

        image = mask.to_image(data.shape)
        assert image is None

    @pytest.mark.parametrize('position', POSITIONS)
    def test_cutout_partial_overlap(self, position):
        data = np.ones((50, 50))
        aper = CircularAperture(position, r=30.0)
        mask = aper.to_mask()

        cutout = mask.cutout(data)
        assert cutout.shape == mask.shape

        weighted_data = mask.multiply(data)
        assert weighted_data.shape == mask.shape

        image = mask.to_image(data.shape)
        assert image.shape == data.shape

    def test_cutout_partial_overlap_quantity(self):
        """
        Test that cutout with a Quantity array and partial overlap
        applies the data unit to the output cutout (covers the `cutout
        <<= data.unit` branch).
        """
        aper = CircularAperture((-20, -20), r=30.0)
        mask = aper.to_mask()
        data = np.ones((50, 50)) * u.adu
        cutout = mask.cutout(data)
        assert isinstance(cutout, u.Quantity)
        assert cutout.unit == u.adu

    @pytest.mark.parametrize('value', [np.nan, np.inf])
    def test_nonfinite_fill_value(self, value):
        aper = CircularAnnulus((0, 0), 10, 20)
        data = np.ones((101, 101)).astype(int)
        cutout = aper.to_mask().cutout(data, fill_value=value)
        assert ~np.isfinite(cutout[0, 0])


class TestMultiply:
    """
    Tests for the ApertureMask multiply method.
    """

    def test_multiply(self):
        radius = 10.0
        data = np.ones((50, 50))
        aper = CircularAperture((25, 25), r=radius)
        mask = aper.to_mask()
        data_weighted = mask.multiply(data)
        assert_almost_equal(np.sum(data_weighted), np.pi * radius**2)

        # Test that multiply() returns a copy
        data[25, 25] = 100.0
        assert data_weighted[10, 10] == 1.0

    def test_multiply_quantity(self):
        radius = 10.0
        data = np.ones((50, 50)) * u.adu
        aper = CircularAperture((25, 25), r=radius)
        mask = aper.to_mask()
        data_weighted = mask.multiply(data)
        assert data_weighted.unit == u.adu
        assert_almost_equal(np.sum(data_weighted.value), np.pi * radius**2)

        # Test that multiply() returns a copy
        data[25, 25] = 100.0 * u.adu
        assert data_weighted[10, 10].value == 1.0

    def test_multiply_fill_value(self):
        aper = CircularAnnulus((0, 0), 10, 20)
        data = np.ones((101, 101)).astype(int)
        cutout = aper.to_mask().multiply(data, fill_value=np.nan)
        xypos = ((20, 20), (5, 5), (5, 35), (35, 5), (35, 35))
        for x, y in xypos:
            assert np.isnan(cutout[y, x])

    def test_nonfinite_in_bbox(self):
        """
        Regression test that non-finite data values outside the mask but
        within the bounding box are set to zero.
        """
        data = np.ones((101, 101))
        data[33, 33] = np.nan
        data[67, 67] = np.inf
        data[33, 67] = -np.inf
        data[22, 22] = np.nan
        data[22, 23] = np.inf

        radius = 20.0
        aper1 = CircularAperture((50, 50), r=radius)
        aper2 = CircularAperture((5, 5), r=radius)

        wdata1 = aper1.to_mask(method='exact').multiply(data)
        assert_allclose(np.sum(wdata1), np.pi * radius**2)

        wdata2 = aper2.to_mask(method='exact').multiply(data)
        assert_allclose(np.sum(wdata2), 561.6040111923013)


class TestGetValues:
    """
    Tests for the ApertureMask get_values method.
    """

    def test_get_values(self):
        aper = CircularAnnulus(((0, 0), (50, 50), (100, 100)), 10, 20)
        data = np.ones((101, 101))
        values = [mask.get_values(data) for mask in aper.to_mask()]
        shapes = [val.shape for val in values]
        sums = [np.sum(val) for val in values]
        assert shapes[0] == (278,)
        assert shapes[1] == (1068,)
        assert shapes[2] == (278,)
        sums_expected = (245.621534, 942.477796, 245.621534)
        assert_allclose(sums, sums_expected)

    def test_get_values_no_overlap(self):
        aper = CircularAperture((-100, -100), r=3)
        data = np.ones((51, 51))
        values = aper.to_mask().get_values(data)
        assert values.shape == (0,)

    def test_get_values_units(self):
        """
        Test that the result is a Quantity with the data units for both
        the overlap and no-overlap cases.
        """
        data = np.ones((51, 51)) * u.Jy
        values = CircularAperture((25, 25), r=3).to_mask().get_values(data)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.Jy

        values = CircularAperture((-100, -100), r=3).to_mask().get_values(
            data)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.Jy
        assert values.shape == (0,)

    def test_get_values_mask(self):
        aper = CircularAperture((24.5, 24.5), r=10.0)
        data = np.ones((51, 51))
        mask = aper.to_mask()
        match = 'mask and data must have the same shape'
        with pytest.raises(ValueError, match=match):
            mask.get_values(data, mask=np.ones(3))

        arr = mask.get_values(data, mask=None)
        assert_allclose(np.sum(arr), 100.0 * np.pi)

        data_mask = np.zeros(data.shape, dtype=bool)
        data_mask[25:] = True
        arr2 = mask.get_values(data, mask=data_mask)
        assert_allclose(np.sum(arr2), 100.0 * np.pi / 2.0)


class TestAnnulusMasks:
    """
    Tests for the masks produced by the annulus apertures.
    """

    def test_rectangular_annulus_hin(self):
        aper = RectangularAnnulus((25, 25), 2, 4, 20, h_in=18, theta=0)
        mask = aper.to_mask(method='center')
        assert mask.data.shape == (21, 5)
        assert np.count_nonzero(mask.data) == 40

    @pytest.mark.parametrize('method', ['center', 'subpixel', 'exact'])
    def test_annulus_mask_nonnegative(self, method):
        """
        Regression test that annulus aperture masks never contain
        negative overlap fractions.

        An annulus overlap is computed as the difference of the outer
        and inner overlaps, which can leave a boundary pixel with a tiny
        negative value from floating-point noise. These must be clipped
        to zero.
        """
        # Fractional positions place pixel centers near the annulus
        # edges, where the subtraction is prone to tiny negative
        # floating-point values.
        positions = [(24.3, 27.7), (25.55, 22.15), (28.1, 29.9)]
        apertures = [
            CircularAnnulus(positions, r_in=3.0, r_out=6.0),
            EllipticalAnnulus(positions, a_in=3.0, a_out=6.0, b_out=4.0,
                              theta=0.5),
            RectangularAnnulus(positions, w_in=3.0, w_out=6.0, h_out=4.0,
                               theta=0.5),
        ]
        for aperture in apertures:
            for mask in aperture.to_mask(method=method):
                assert mask.data.min() >= 0.0
