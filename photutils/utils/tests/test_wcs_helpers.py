# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _wcs_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.utils._wcs_helpers import _pixel_scale_angle_at_skycoord


def _make_simple_wcs(crpix=(50, 50), cdelt=(-0.001, 0.001),
                     crval=(180.0, 45.0)):
    """
    Create a simple TAN WCS for testing.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = cdelt
    wcs.wcs.crval = crval
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


class TestPixelScaleAngle:
    def setup_class(self):
        self.wcs = _make_simple_wcs()
        self.skycoord = SkyCoord(180.0, 45.0, unit='deg')

    def test_basic(self):
        """
        Test pixel scale and angle at the reference position.
        """
        xypos, scale, angle = _pixel_scale_angle_at_skycoord(
            self.skycoord, self.wcs)

        # Check that the returned pixel position is near the CRPIX
        assert_allclose(xypos[0], 49.0, atol=1.0)
        assert_allclose(xypos[1], 49.0, atol=1.0)

        # Scale should be close to |cdelt| in arcsec/pixel
        expected_scale = 0.001 * 3600  # 3.6 arcsec/pixel
        assert_allclose(scale.value, expected_scale, rtol=0.01)
        assert scale.unit == u.arcsec / u.pixel

        # Angle should be finite
        assert np.isfinite(angle.value)
        assert angle.unit == u.deg

    def test_offset_position(self):
        """
        Test at a position offset from the reference pixel.
        """
        skycoord = self.wcs.pixel_to_world(30, 30)
        xypos, scale, angle = _pixel_scale_angle_at_skycoord(
            skycoord, self.wcs)
        assert_allclose(xypos[0], 30.0, atol=0.5)
        assert_allclose(xypos[1], 30.0, atol=0.5)
        assert np.isfinite(scale.value)
        assert np.isfinite(angle.value)

    def test_custom_offset(self):
        """
        Test with a custom angular offset.
        """
        _, scale1, _ = _pixel_scale_angle_at_skycoord(
            self.skycoord, self.wcs, offset=1 * u.arcsec)
        _, scale2, _ = _pixel_scale_angle_at_skycoord(
            self.skycoord, self.wcs, offset=10 * u.arcsec)

        # The scales should be very similar regardless of offset
        assert_allclose(scale1.value, scale2.value, rtol=0.01)

    def test_degenerate_wcs(self):
        """
        Test that a degenerate WCS raises an error.
        """
        wcs = _make_simple_wcs(cdelt=(0.0, 0.0))
        skycoord = SkyCoord(180.0, 45.0, unit='deg')
        with pytest.raises((ValueError, Exception)):
            _pixel_scale_angle_at_skycoord(skycoord, wcs)

    def test_nan_pixel_scale(self):
        """
        Test that a ValueError is raised when pixel_sep is zero or
        non-finite.
        """
        from unittest.mock import MagicMock

        mock_wcs = MagicMock()
        # The mock returns the same pixel coords for both calls,
        # resulting in pixel_sep of 0
        mock_wcs.world_to_pixel.return_value = (50.0, 50.0)
        skycoord = SkyCoord(180.0, 45.0, unit='deg')
        match = 'Cannot compute pixel scale'
        with pytest.raises(ValueError, match=match):
            _pixel_scale_angle_at_skycoord(skycoord, mock_wcs)
