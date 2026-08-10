# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Base classes for aperture tests.

The shape-specific test modules (e.g., ``test_circle``, ``test_ellipse``,
``test_rectangle``) define one class per aperture class. Each of those
classes sets the ``aperture`` attribute (a multi-position aperture) and
the ``copy_param``/``copy_value`` pair naming a parameter to change, and
inherits the tests that apply to every aperture of that kind from the
base classes here.
"""

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_equal

from photutils.utils._optional_deps import HAS_MATPLOTLIB


class BaseTestApertureParams:
    index = 2
    slc = slice(0, 2)
    expected_slc_len = 2


class BaseTestAperture(BaseTestApertureParams):
    """
    Tests that apply to every aperture, pixel or sky.
    """

    # The name and new value of an aperture parameter to change in
    # test_copy_eq
    copy_param = None
    copy_value = None

    def test_index(self):
        aper = self.aperture[self.index]
        assert isinstance(aper, self.aperture.__class__)
        assert aper.isscalar
        expected_positions = self.aperture.positions[self.index]

        for param in aper._params:
            if param == 'positions':
                if isinstance(expected_positions, SkyCoord):
                    assert_quantity_allclose(aper.positions.ra,
                                             expected_positions.ra)
                    assert_quantity_allclose(aper.positions.dec,
                                             expected_positions.dec)
                else:
                    assert_equal(getattr(aper, param), expected_positions)
            else:
                assert (getattr(aper, param)
                        == getattr(self.aperture, param))

    def test_slice(self):
        aper = self.aperture[self.slc]
        assert isinstance(aper, self.aperture.__class__)
        assert len(aper) == self.expected_slc_len
        expected_positions = self.aperture.positions[self.slc]

        for param in aper._params:
            if param == 'positions':
                if isinstance(expected_positions, SkyCoord):
                    assert_quantity_allclose(aper.positions.ra,
                                             expected_positions.ra)
                    assert_quantity_allclose(aper.positions.dec,
                                             expected_positions.dec)
                else:
                    assert_equal(getattr(aper, param), expected_positions)
            else:
                assert (getattr(aper, param)
                        == getattr(self.aperture, param))

    def test_copy_eq(self):
        """
        Test that a copy compares equal until one of its parameters is
        changed.
        """
        aper = self.aperture.copy()
        assert aper == self.aperture
        setattr(aper, self.copy_param, self.copy_value)
        assert aper != self.aperture


class BaseTestPixelAperture(BaseTestAperture):
    """
    Tests that apply to every pixel aperture.
    """

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self):
        self.aperture.plot()

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_returns_patches(self):
        """
        Test that plot returns a list of patches usable in a legend.
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        patches = self.aperture.plot()
        assert isinstance(patches, list)
        for patch in patches:
            assert isinstance(patch, Patch)

        _, ax = plt.subplots()
        ax.legend(patches, list(range(len(patches))))

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_to_patch_nonscalar(self):
        """
        Test that _to_patch returns a list for non-scalar apertures.
        """
        patches = self.aperture._to_patch()
        assert isinstance(patches, list)


class BaseTestRotatedPixelAperture(BaseTestPixelAperture):
    """
    Tests that apply to every pixel aperture with a rotation angle.
    """

    def test_theta(self):
        """
        Test that a theta input as a float is stored as a Quantity in
        radians.
        """
        assert isinstance(self.aperture.theta, u.Quantity)
        assert self.aperture.theta.unit == u.rad
