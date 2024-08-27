# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines tools to convert `regions.Region` objects to
Aperture objects.
"""

import astropy.units as u

from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAnnulus, SkyCircularAperture)
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)

__all__ = ['region_to_aperture']

__doctest_requires__ = {'region_to_aperture': ['regions']}


def region_to_aperture(region):
    """
    Convert a given `regions.Region` object to an
    `~photutils.aperture.Aperture` object.

    Parameters
    ----------
    region : `regions.Region`
        A supported `regions.Region` object.

    Returns
    -------
    aperture : `~photutils.aperture.Aperture`
        An equivalent ``photutils`` aperture.

    Raises
    ------
    `TypeError`
        The given `regions.Region` object is not supported.

    Notes
    -----
    .. |rarr| unicode:: U+0279E .. RIGHTWARDS ARROW

    The following `regions.Region` objects are supported, shown with
    their equivalent `~photutils.aperture.Aperture` object:

    - `~regions.CirclePixelRegion` |rarr|
      `~photutils.aperture.CircularAperture`
    - `~regions.CircleSkyRegion` |rarr|
      `~photutils.aperture.SkyCircularAperture`
    - `~regions.EllipsePixelRegion` |rarr|
      `~photutils.aperture.EllipticalAperture`
    - `~regions.EllipseSkyRegion` |rarr|
      `~photutils.aperture.SkyEllipticalAperture`
    - `~regions.RectanglePixelRegion` |rarr|
      `~photutils.aperture.RectangularAperture`
    - `~regions.RectangleSkyRegion` |rarr|
      `~photutils.aperture.SkyRectangularAperture`
    - `~regions.CircleAnnulusPixelRegion` |rarr|
      `~photutils.aperture.CircularAnnulus`
    - `~regions.CircleAnnulusSkyRegion` |rarr|
      `~photutils.aperture.SkyCircularAnnulus`
    - `~regions.EllipseAnnulusPixelRegion` |rarr|
      `~photutils.aperture.EllipticalAnnulus`
    - `~regions.EllipseAnnulusSkyRegion` |rarr|
      `~photutils.aperture.SkyEllipticalAnnulus`
    - `~regions.RectangleAnnulusPixelRegion` |rarr|
      `~photutils.aperture.RectangularAnnulus`
    - `~regions.RectangleAnnulusSkyRegion` |rarr|
      `~photutils.aperture.SkyRectangularAnnulus`

    Examples
    --------
    >>> from regions import CirclePixelRegion, PixCoord
    >>> from photutils.aperture import region_to_aperture
    >>> region = CirclePixelRegion(center=PixCoord(x=10, y=20), radius=5)
    >>> aperture = region_to_aperture(region)
    >>> aperture
    <CircularAperture([10., 20.], r=5.0)>
    """
    from regions import (CircleAnnulusPixelRegion, CircleAnnulusSkyRegion,
                         CirclePixelRegion, CircleSkyRegion,
                         EllipseAnnulusPixelRegion, EllipseAnnulusSkyRegion,
                         EllipsePixelRegion, EllipseSkyRegion,
                         RectangleAnnulusPixelRegion,
                         RectangleAnnulusSkyRegion, RectanglePixelRegion,
                         RectangleSkyRegion)

    if isinstance(region, CirclePixelRegion):
        aperture = CircularAperture(region.center.xy, region.radius)

    elif isinstance(region, CircleSkyRegion):
        aperture = SkyCircularAperture(region.center, region.radius)

    elif isinstance(region, EllipsePixelRegion):
        aperture = EllipticalAperture(
            region.center.xy, region.width * 0.5, region.height * 0.5,
            theta=region.angle.to_value(u.radian))

    elif isinstance(region, EllipseSkyRegion):
        aperture = SkyEllipticalAperture(
            region.center, region.width * 0.5, region.height * 0.5,
            theta=(region.angle - (90 * u.deg)))

    elif isinstance(region, RectanglePixelRegion):
        aperture = RectangularAperture(
            region.center.xy, region.width, region.height,
            theta=region.angle.to_value(u.radian))

    elif isinstance(region, RectangleSkyRegion):
        aperture = SkyRectangularAperture(
            region.center, region.width, region.height,
            theta=(region.angle - (90 * u.deg)))

    elif isinstance(region, CircleAnnulusPixelRegion):
        aperture = CircularAnnulus(
            region.center.xy,
            region.inner_radius, region.outer_radius)

    elif isinstance(region, CircleAnnulusSkyRegion):
        aperture = SkyCircularAnnulus(
            region.center,
            region.inner_radius, region.outer_radius)

    elif isinstance(region, EllipseAnnulusPixelRegion):
        aperture = EllipticalAnnulus(
            region.center.xy,
            region.inner_width * 0.5, region.outer_width * 0.5,
            region.outer_height * 0.5, b_in=region.inner_height * 0.5,
            theta=region.angle.to_value(u.radian))

    elif isinstance(region, EllipseAnnulusSkyRegion):
        aperture = SkyEllipticalAnnulus(
            region.center,
            region.inner_width * 0.5, region.outer_width * 0.5,
            region.outer_height * 0.5, b_in=region.inner_height * 0.5,
            theta=(region.angle - (90 * u.deg)))

    elif isinstance(region, RectangleAnnulusPixelRegion):
        aperture = RectangularAnnulus(
            region.center.xy,
            region.inner_width, region.outer_width,
            region.outer_height, h_in=region.inner_height,
            theta=region.angle.to_value(u.radian))

    elif isinstance(region, RectangleAnnulusSkyRegion):
        aperture = SkyRectangularAnnulus(
            region.center,
            region.inner_width, region.outer_width,
            region.outer_height, h_in=region.inner_height,
            theta=(region.angle - (90 * u.deg)))

    else:
        raise TypeError(f'Cannot convert {region.__class__.__name__!r} to '
                        'an Aperture object')

    return aperture
