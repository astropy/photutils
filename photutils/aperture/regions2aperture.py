# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to convert regions.Region to Aperture.
"""

import astropy.units as u

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture,
                                SkyCircularAnnulus, SkyCircularAperture,
                                SkyEllipticalAnnulus, SkyEllipticalAperture,
                                SkyRectangularAnnulus, SkyRectangularAperture)


def regions2aperture(region_shape):
    """
    Convert a given `regions` shape to ``photutils`` aperture.

    Parameters
    ----------
    region_shape : `regions.Region`
        A supported `regions` shape.

    Returns
    -------
    aperture : `~photutils.aperture.Aperture`
        An equivalent ``photutils`` aperture.

    Raises
    ------
    NotImplementedError
        The given `regions` shape is not supported.
    """
    from regions import (CircleAnnulusPixelRegion, CircleAnnulusSkyRegion,
                         CirclePixelRegion, CircleSkyRegion,
                         EllipseAnnulusPixelRegion, EllipseAnnulusSkyRegion,
                         EllipsePixelRegion, EllipseSkyRegion,
                         RectangleAnnulusPixelRegion,
                         RectangleAnnulusSkyRegion, RectanglePixelRegion,
                         RectangleSkyRegion)

    if isinstance(region_shape, CirclePixelRegion):
        aperture = CircularAperture(region_shape.center.xy, region_shape.radius)

    elif isinstance(region_shape, CircleSkyRegion):
        aperture = SkyCircularAperture(region_shape.center, region_shape.radius)

    elif isinstance(region_shape, EllipsePixelRegion):
        aperture = EllipticalAperture(
            region_shape.center.xy, region_shape.width * 0.5, region_shape.height * 0.5,
            theta=region_shape.angle.to_value(u.radian))

    elif isinstance(region_shape, EllipseSkyRegion):
        aperture = SkyEllipticalAperture(
            region_shape.center, region_shape.width * 0.5, region_shape.height * 0.5,
            theta=(region_shape.angle - (90 * u.deg)))

    elif isinstance(region_shape, RectanglePixelRegion):
        aperture = RectangularAperture(
            region_shape.center.xy, region_shape.width, region_shape.height,
            theta=region_shape.angle.to_value(u.radian))

    elif isinstance(region_shape, RectangleSkyRegion):
        aperture = SkyRectangularAperture(
            region_shape.center, region_shape.width, region_shape.height,
            theta=(region_shape.angle - (90 * u.deg)))

    elif isinstance(region_shape, CircleAnnulusPixelRegion):
        aperture = CircularAnnulus(
            region_shape.center.xy,
            region_shape.inner_radius, region_shape.outer_radius)

    elif isinstance(region_shape, CircleAnnulusSkyRegion):
        aperture = SkyCircularAnnulus(
            region_shape.center,
            region_shape.inner_radius, region_shape.outer_radius)

    elif isinstance(region_shape, EllipseAnnulusPixelRegion):
        aperture = EllipticalAnnulus(
            region_shape.center.xy,
            region_shape.inner_width * 0.5, region_shape.outer_width * 0.5,
            region_shape.outer_height * 0.5, b_in=region_shape.inner_height * 0.5,
            theta=region_shape.angle.to_value(u.radian))

    elif isinstance(region_shape, EllipseAnnulusSkyRegion):
        aperture = SkyEllipticalAnnulus(
            region_shape.center,
            region_shape.inner_width * 0.5, region_shape.outer_width * 0.5,
            region_shape.outer_height * 0.5, b_in=region_shape.inner_height * 0.5,
            theta=(region_shape.angle - (90 * u.deg)))

    elif isinstance(region_shape, RectangleAnnulusPixelRegion):
        aperture = RectangularAnnulus(
            region_shape.center.xy,
            region_shape.inner_width, region_shape.outer_width,
            region_shape.outer_height, h_in=region_shape.inner_height,
            theta=region_shape.angle.to_value(u.radian))

    elif isinstance(region_shape, RectangleAnnulusSkyRegion):
        aperture = SkyRectangularAnnulus(
            region_shape.center,
            region_shape.inner_width, region_shape.outer_width,
            region_shape.outer_height, h_in=region_shape.inner_height,
            theta=(region_shape.angle - (90 * u.deg)))

    else:
        raise NotImplementedError(f'{region_shape.__class__.__name__}'
                                  ' is not supported')

    return aperture
