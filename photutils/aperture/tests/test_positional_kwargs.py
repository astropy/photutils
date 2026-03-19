# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deprecation of positional optional arguments in the
aperture package.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAnnulus, SkyCircularAperture)
from photutils.aperture.core import SkyAperture
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.photometry import aperture_photometry
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)
from photutils.utils._optional_deps import HAS_MATPLOTLIB

DATA = np.ones((101, 101))
POSITION = (50, 50)
SKY_POSITION = SkyCoord(ra=50.0, dec=50.0, unit='deg')

PIXEL_APERTURE_CL = [CircularAperture,
                     CircularAnnulus,
                     EllipticalAperture,
                     EllipticalAnnulus,
                     RectangularAperture,
                     RectangularAnnulus]

SKY_APERTURE_CL = [SkyCircularAperture,
                   SkyCircularAnnulus,
                   SkyEllipticalAperture,
                   SkyEllipticalAnnulus,
                   SkyRectangularAperture,
                   SkyRectangularAnnulus]

APERTURE_CL = PIXEL_APERTURE_CL + SKY_APERTURE_CL

PIXEL_TEST_APERTURES = list(zip(PIXEL_APERTURE_CL,
                                ({'r': 3.0},
                                 {'r_in': 3.0, 'r_out': 5.0},
                                 {'a': 3.0, 'b': 5.0, 'theta': 1.0},
                                 {'a_in': 3.0, 'a_out': 5.0, 'b_out': 4.0,
                                  'b_in': 12.0 / 5.0, 'theta': 1.0},
                                 {'w': 5, 'h': 8, 'theta': np.pi / 4},
                                 {'w_in': 8, 'w_out': 12, 'h_out': 8,
                                  'h_in': 16.0 / 3.0, 'theta': np.pi / 8}),
                                strict=True))

SKY_TEST_APERTURES = list(zip(SKY_APERTURE_CL,
                              ({'r': 3.0 * u.arcsec},
                               {'r_in': 3.0 * u.arcsec,
                                'r_out': 5.0 * u.arcsec},
                               {'a': 3.0 * u.arcsec, 'b': 5.0 * u.arcsec,
                                'theta': 1.0 * u.deg},
                               {'a_in': 3.0 * u.arcsec,
                                'a_out': 5.0 * u.arcsec,
                                'b_out': 4.0 * u.arcsec,
                                'b_in': 2.4 * u.arcsec,
                                'theta': 1.0 * u.deg},
                               {'w': 5.0 * u.arcsec, 'h': 8.0 * u.arcsec,
                                'theta': 45.0 * u.deg},
                               {'w_in': 8.0 * u.arcsec,
                                'w_out': 12.0 * u.arcsec,
                                'h_out': 8.0 * u.arcsec,
                                'h_in': 16.0 / 3.0 * u.arcsec,
                                'theta': 22.5 * u.deg}),
                              strict=True))

TEST_APERTURES = PIXEL_TEST_APERTURES + SKY_TEST_APERTURES

# Apertures with @deprecated_positional_kwargs on __init__
# (excludes circular apertures which have no optional kwargs)
INIT_WARN_APERTURES = [
    (cls, POSITION, params)
    for cls, params in PIXEL_TEST_APERTURES
    if cls not in (CircularAperture, CircularAnnulus)
] + [
    (cls, SKY_POSITION, params)
    for cls, params in SKY_TEST_APERTURES
    if cls not in (SkyCircularAperture, SkyCircularAnnulus)
]


class TestCircularMaskMixinPositionalKwargs:
    def test_to_mask_no_warning(self):
        aper = CircularAperture(POSITION, r=5)
        aper.to_mask(method='exact', subpixels=5)

    def test_to_mask_positional_warning(self):
        aper = CircularAperture(POSITION, r=5)
        match = 'to_mask'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aper.to_mask('exact')


class TestApertureMethodsPositionalKwargs:
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_plot_no_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        aper.plot(ax=None)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_plot_positional_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        match = 'plot'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aper.plot(None)

    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_to_mask_no_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        aper.to_mask(method='exact', subpixels=5)

    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_to_mask_positional_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        match = 'to_mask'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aper.to_mask('exact')

    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_do_photometry_no_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        aper.do_photometry(DATA, error=None, mask=None)

    @pytest.mark.parametrize(('aperture_class', 'params'),
                             PIXEL_TEST_APERTURES)
    def test_do_photometry_positional_warning(self, aperture_class, params):
        aper = aperture_class(POSITION, **params)
        error = np.ones_like(DATA)
        match = 'do_photometry'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aper.do_photometry(DATA, error)


class TestApertureInitPositionalKwargs:
    @pytest.mark.parametrize(('aperture_class', 'params'),
                             TEST_APERTURES)
    def test_init_no_warning(self, aperture_class, params):
        position = (SKY_POSITION if issubclass(aperture_class, SkyAperture)
                    else POSITION)
        aperture_class(position, **params)

    @pytest.mark.parametrize(('aperture_class', 'position', 'params'),
                             INIT_WARN_APERTURES)
    def test_init_positional_warning(self, aperture_class, position, params):
        match = '__init__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aperture_class(position, *params.values())


class TestBoundingBoxPositionalKwargs:
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_no_warning(self):
        bbox = BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
        # Keyword use should not warn
        bbox.plot(ax=None, origin=(0, 0))

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_positional_warning(self):
        bbox = BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
        match = 'plot'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bbox.plot(None)


class TestApertureMaskPositionalKwargs:
    def setup_method(self):
        aper = CircularAperture(POSITION, r=5)
        self.mask = aper.to_mask(method='exact')

    def test_to_image_no_warning(self):
        self.mask.to_image(shape=DATA.shape, dtype=float)

    def test_to_image_positional_warning(self):
        match = 'to_image'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            self.mask.to_image(DATA.shape, int)

    def test_cutout_no_warning(self):
        self.mask.cutout(DATA, fill_value=0.0, copy=False)

    def test_cutout_positional_warning(self):
        match = 'cutout'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            self.mask.cutout(DATA, 0.0)

    def test_multiply_no_warning(self):
        self.mask.multiply(DATA, fill_value=0.0)

    def test_multiply_positional_warning(self):
        match = 'multiply'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            self.mask.multiply(DATA, 0.0)

    def test_get_values_no_warning(self):
        self.mask.get_values(DATA, mask=None)

    def test_get_values_positional_warning(self):
        match = 'get_values'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            self.mask.get_values(DATA, None)


class TestAperturePhotometryPositionalKwargs:
    def test_no_warning(self):
        aper = CircularAperture(POSITION, r=5)
        aperture_photometry(DATA, aper, error=None, mask=None)

    def test_positional_warning(self):
        aper = CircularAperture(POSITION, r=5)
        error = np.ones_like(DATA)
        match = 'aperture_photometry'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            aperture_photometry(DATA, aper, error)
