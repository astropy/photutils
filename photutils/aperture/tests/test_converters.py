# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for converting regions.Region to Aperture.
"""

import pytest
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                SkyCircularAnnulus, SkyCircularAperture,
                                SkyEllipticalAnnulus, SkyEllipticalAperture,
                                SkyRectangularAnnulus, SkyRectangularAperture)
from photutils.aperture.converters import region_to_aperture
from photutils.utils._optional_deps import HAS_REGIONS


@pytest.fixture
def image_2d_wcs():
    return WCS(
        {
            'CTYPE1': 'RA---TAN',
            'CUNIT1': 'deg',
            'CDELT1': -0.0002777777778,
            'CRPIX1': 1,
            'CRVAL1': 337.5202808,
            'CTYPE2': 'DEC--TAN',
            'CUNIT2': 'deg',
            'CDELT2': 0.0002777777778,
            'CRPIX2': 1,
            'CRVAL2': -20.833333059999998,
        }
    )


def compare_region_shapes(reg1, reg2):
    from regions import PixCoord

    assert reg1.__class__ == reg2.__class__
    for param in reg1._params:
        par1 = getattr(reg1, param)
        par2 = getattr(reg2, param)
        if isinstance(par1, PixCoord):
            assert_allclose(par1.xy, par2.xy)
        elif isinstance(par1, SkyCoord):
            assert par1 == par2
        elif isinstance(par1, u.Quantity):
            assert_quantity_allclose(par1, par2)
        else:
            assert_allclose(par1, par2)


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_circle(image_2d_wcs):
    from regions import CirclePixelRegion, PixCoord

    region_shape = CirclePixelRegion(center=PixCoord(x=42, y=43), radius=4.2)
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, CircularAperture)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.r, region_shape.radius)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyCircularAperture)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.r, region_sky.radius)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        CirclePixelRegion(center=PixCoord(x=[0, 42], y=[1, 43]), radius=4.2)
    match = 'must be a strictly positive scalar'
    with pytest.raises(ValueError, match=match):
        CirclePixelRegion(center=PixCoord(x=42, y=43), radius=[1, 4.2])


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_ellipse(image_2d_wcs):
    from regions import EllipsePixelRegion, PixCoord

    region_shape = EllipsePixelRegion(
        center=PixCoord(x=42, y=43), width=16, height=10,
        angle=Angle(30, 'deg')
    )
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, EllipticalAperture)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.a * 2, region_shape.width)
    assert_allclose(aperture.b * 2, region_shape.height)
    assert_allclose(aperture.theta, region_shape.angle.radian)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyEllipticalAperture)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.a * 2, region_sky.width)
    assert_quantity_allclose(aperture_sky.b * 2, region_sky.height)
    assert_quantity_allclose(aperture_sky.theta + (90 * u.deg),
                             region_sky.angle)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        EllipsePixelRegion(
            center=PixCoord(x=[0, 42], y=[1, 43]),
            width=16,
            height=10,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipsePixelRegion(
            center=PixCoord(x=42, y=43),
            width=[1, 16],
            height=10,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipsePixelRegion(
            center=PixCoord(x=42, y=43),
            width=16,
            height=[1, 10],
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipsePixelRegion(
            center=PixCoord(x=42, y=43),
            width=16,
            height=10,
            angle=Angle([0, 30], 'deg'),
        )


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_rectangle(image_2d_wcs):
    from regions import PixCoord, RectanglePixelRegion

    region_shape = RectanglePixelRegion(
        center=PixCoord(x=42, y=43), width=16, height=10,
        angle=Angle(30, 'deg')
    )
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, RectangularAperture)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.w, region_shape.width)
    assert_allclose(aperture.h, region_shape.height)
    assert_allclose(aperture.theta, region_shape.angle.radian)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyRectangularAperture)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.w, region_sky.width)
    assert_quantity_allclose(aperture_sky.h, region_sky.height)
    assert_quantity_allclose(aperture_sky.theta + (90 * u.deg),
                             region_sky.angle)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        RectanglePixelRegion(
            center=PixCoord(x=[0, 42], y=[1, 43]),
            width=16,
            height=10,
            angle=Angle(30, 'deg'),
        )

    match = 'must be a strictly positive scalar'
    with pytest.raises(ValueError, match=match):
        RectanglePixelRegion(
            center=PixCoord(x=42, y=43),
            width=[1, 16],
            height=10,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=match):
        RectanglePixelRegion(
            center=PixCoord(x=42, y=43),
            width=16,
            height=[1, 10],
            angle=Angle(30, 'deg'),
        )
    match = 'must be a scalar'
    with pytest.raises(ValueError, match=match):
        RectanglePixelRegion(
            center=PixCoord(x=42, y=43),
            width=16,
            height=10,
            angle=Angle([0, 30], 'deg'),
        )


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_circle_annulus(image_2d_wcs):
    from regions import CircleAnnulusPixelRegion, PixCoord

    region_shape = CircleAnnulusPixelRegion(
        center=PixCoord(x=42, y=43), inner_radius=5, outer_radius=8
    )
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, CircularAnnulus)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.r_in, region_shape.inner_radius)
    assert_allclose(aperture.r_out, region_shape.outer_radius)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyCircularAnnulus)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.r_in, region_sky.inner_radius)
    assert_quantity_allclose(aperture_sky.r_out, region_sky.outer_radius)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        CircleAnnulusPixelRegion(
            center=PixCoord(x=[0, 42], y=[1, 43]), inner_radius=5,
            outer_radius=8
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        CircleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43), inner_radius=[1, 5], outer_radius=8
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        CircleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43), inner_radius=5, outer_radius=[8, 10]
        )


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_ellipse_annulus(image_2d_wcs):
    from regions import EllipseAnnulusPixelRegion, PixCoord

    region_shape = EllipseAnnulusPixelRegion(
        center=PixCoord(x=42, y=43),
        inner_width=5.5,
        inner_height=3.5,
        outer_width=8.5,
        outer_height=6.5,
        angle=Angle(30, 'deg'),
    )
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, EllipticalAnnulus)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.a_in * 2, region_shape.inner_width)
    assert_allclose(aperture.a_out * 2, region_shape.outer_width)
    assert_allclose(aperture.b_in * 2, region_shape.inner_height)
    assert_allclose(aperture.b_out * 2, region_shape.outer_height)
    assert_allclose(aperture.theta, region_shape.angle.radian)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyEllipticalAnnulus)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.a_in * 2, region_sky.inner_width)
    assert_quantity_allclose(aperture_sky.a_out * 2, region_sky.outer_width)
    assert_quantity_allclose(aperture_sky.b_in * 2, region_sky.inner_height)
    assert_quantity_allclose(aperture_sky.b_out * 2, region_sky.outer_height)
    assert_quantity_allclose(aperture_sky.theta + (90 * u.deg),
                             region_sky.angle)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=[0, 42], y=[1, 43]),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    match = 'must be a strictly positive scalar'
    with pytest.raises(ValueError, match=match):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=[1, 5.5],
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=match):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=[1, 3.5],
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=[8.5, 10],
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=[6.5, 10],
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        EllipseAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle([0, 30], 'deg'),
        )


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_rectangle_annulus(image_2d_wcs):
    from regions import PixCoord, RectangleAnnulusPixelRegion

    region_shape = RectangleAnnulusPixelRegion(
        center=PixCoord(x=42, y=43),
        inner_width=5.5,
        inner_height=3.5,
        outer_width=8.5,
        outer_height=6.5,
        angle=Angle(30, 'deg'),
    )
    aperture = region_to_aperture(region_shape)
    assert isinstance(aperture, RectangularAnnulus)
    assert_allclose(aperture.positions, region_shape.center.xy)
    assert_allclose(aperture.w_in, region_shape.inner_width)
    assert_allclose(aperture.w_out, region_shape.outer_width)
    assert_allclose(aperture.h_in, region_shape.inner_height)
    assert_allclose(aperture.h_out, region_shape.outer_height)
    assert_allclose(aperture.theta, region_shape.angle.radian)

    region_sky = region_shape.to_sky(image_2d_wcs)
    aperture_sky = region_to_aperture(region_sky)
    assert isinstance(aperture_sky, SkyRectangularAnnulus)
    assert aperture_sky.positions == region_sky.center  # SkyCoord
    assert_quantity_allclose(aperture_sky.w_in, region_sky.inner_width)
    assert_quantity_allclose(aperture_sky.w_out, region_sky.outer_width)
    assert_quantity_allclose(aperture_sky.h_in, region_sky.inner_height)
    assert_quantity_allclose(aperture_sky.h_out, region_sky.outer_height)
    assert_quantity_allclose(aperture_sky.theta + (90 * u.deg),
                             region_sky.angle)

    # NOTE: If these no longer fail, we also have to account for
    # non-scalar inputs. Assume this is representative for the sky
    # counterpart too.
    match = 'must be a scalar PixCoord'
    with pytest.raises(ValueError, match=match):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=[0, 42], y=[1, 43]),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )

    match = 'must be a strictly positive scalar'
    with pytest.raises(ValueError, match=match):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=[1, 5.5],
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=match):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=[1, 3.5],
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=[8.5, 10],
            outer_height=6.5,
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=[6.5, 10],
            angle=Angle(30, 'deg'),
        )
    with pytest.raises(ValueError, match=r'must be .* scalar'):
        RectangleAnnulusPixelRegion(
            center=PixCoord(x=42, y=43),
            inner_width=5.5,
            inner_height=3.5,
            outer_width=8.5,
            outer_height=6.5,
            angle=Angle([0, 30], 'deg'),
        )


@pytest.mark.skipif(not HAS_REGIONS, reason='regions is required')
def test_translation_polygon():
    from regions import PixCoord, PolygonPixelRegion

    region_shape = PolygonPixelRegion(vertices=PixCoord(x=[1, 2, 2],
                                                        y=[1, 1, 2]))
    match = r'Cannot convert .* to an Aperture object'
    with pytest.raises(TypeError, match=match):
        region_to_aperture(region_shape)
