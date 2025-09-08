# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to convert between `regions.Region` and Aperture objects
and between `shapely.geometry.Polygon` and `regions.PolygonRegion`
objects.
"""

import astropy.units as u
import numpy as np

# prevent circular imports
from photutils.aperture.circle import (CircularAnnulus, CircularAperture,
                                       SkyCircularAnnulus, SkyCircularAperture)
from photutils.aperture.core import Aperture
from photutils.aperture.ellipse import (EllipticalAnnulus, EllipticalAperture,
                                        SkyEllipticalAnnulus,
                                        SkyEllipticalAperture)
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture,
                                          SkyRectangularAnnulus,
                                          SkyRectangularAperture)

__all__ = ['aperture_to_region', 'region_to_aperture']

__doctest_requires__ = {'region_to_aperture': ['regions'],
                        'aperture_to_region': ['regions'],
                        '_scalar_aperture_to_region': ['regions'],
                        '_shapely_polygon_to_region': ['regions', 'shapely']}


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
    The ellipse ``width`` and ``height`` region parameters represent the
    full extent of the shapes and thus are divided by 2 when converting
    to elliptical aperture objects, which are defined using the
    semi-major (``a``) and semi-minor (``b``) axes. The ``width`` and
    ``height`` parameters are mapped to the the semi-major (``a``) and
    semi-minor (``b``) axes parameters, respectively, of the elliptical
    apertures.

    The region ``angle`` for sky-based regions is defined as the angle
    of the ``width`` axis relative to WCS longitude axis (PA=90).
    However, the sky-based apertures define the ``theta`` as the
    position angle of the semimajor axis relative to the North celestial
    pole (PA=0). Therefore, for sky-based regions the region ``angle``
    is converted to the aperture ``theta`` parameter by subtracting 90
    degrees.

    .. |rarr| unicode:: U+0279E .. RIGHTWARDS ARROW

    The following `regions.Region` objects are supported, shown with
    their equivalent `~photutils.aperture.Aperture` object:

    * `~regions.CirclePixelRegion` |rarr|
      `~photutils.aperture.CircularAperture`
    * `~regions.CircleSkyRegion` |rarr|
      `~photutils.aperture.SkyCircularAperture`
    * `~regions.EllipsePixelRegion` |rarr|
      `~photutils.aperture.EllipticalAperture`
    * `~regions.EllipseSkyRegion` |rarr|
      `~photutils.aperture.SkyEllipticalAperture`
    * `~regions.RectanglePixelRegion` |rarr|
      `~photutils.aperture.RectangularAperture`
    * `~regions.RectangleSkyRegion` |rarr|
      `~photutils.aperture.SkyRectangularAperture`
    * `~regions.CircleAnnulusPixelRegion` |rarr|
      `~photutils.aperture.CircularAnnulus`
    * `~regions.CircleAnnulusSkyRegion` |rarr|
      `~photutils.aperture.SkyCircularAnnulus`
    * `~regions.EllipseAnnulusPixelRegion` |rarr|
      `~photutils.aperture.EllipticalAnnulus`
    * `~regions.EllipseAnnulusSkyRegion` |rarr|
      `~photutils.aperture.SkyEllipticalAnnulus`
    * `~regions.RectangleAnnulusPixelRegion` |rarr|
      `~photutils.aperture.RectangularAnnulus`
    * `~regions.RectangleAnnulusSkyRegion` |rarr|
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
                         RectangleSkyRegion, Region)

    if not isinstance(region, Region):
        msg = 'Input region must be a Region object'
        raise TypeError(msg)

    if isinstance(region, CirclePixelRegion):
        aperture = CircularAperture(region.center.xy, region.radius)

    elif isinstance(region, CircleSkyRegion):
        aperture = SkyCircularAperture(region.center, region.radius)

    elif isinstance(region, EllipsePixelRegion):
        aperture = EllipticalAperture(
            region.center.xy, region.width * 0.5, region.height * 0.5,
            theta=region.angle)

    elif isinstance(region, EllipseSkyRegion):
        aperture = SkyEllipticalAperture(
            region.center, region.width * 0.5, region.height * 0.5,
            theta=(region.angle - (90 * u.deg)))

    elif isinstance(region, RectanglePixelRegion):
        aperture = RectangularAperture(
            region.center.xy, region.width, region.height,
            theta=region.angle)

    elif isinstance(region, RectangleSkyRegion):
        aperture = SkyRectangularAperture(
            region.center, region.width, region.height,
            theta=(region.angle - (90 * u.deg)))

    elif isinstance(region, CircleAnnulusPixelRegion):
        aperture = CircularAnnulus(
            region.center.xy, region.inner_radius, region.outer_radius)

    elif isinstance(region, CircleAnnulusSkyRegion):
        aperture = SkyCircularAnnulus(
            region.center, region.inner_radius, region.outer_radius)

    elif isinstance(region, EllipseAnnulusPixelRegion):
        aperture = EllipticalAnnulus(
            region.center.xy,
            region.inner_width * 0.5, region.outer_width * 0.5,
            region.outer_height * 0.5, b_in=region.inner_height * 0.5,
            theta=region.angle)

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
            theta=region.angle)

    elif isinstance(region, RectangleAnnulusSkyRegion):
        aperture = SkyRectangularAnnulus(
            region.center,
            region.inner_width, region.outer_width,
            region.outer_height, h_in=region.inner_height,
            theta=(region.angle - (90 * u.deg)))

    else:
        msg = (f'Cannot convert {region.__class__.__name__!r} to an '
               'Aperture object')
        raise TypeError(msg)

    return aperture


def aperture_to_region(aperture):
    """
    Convert a given `~photutils.aperture.Aperture` object to a
    `regions.Region` or `regions.Regions` object.

    Because a `regions.Region` object can only have one position, a
    `regions.Regions` object will be returned if the input ``aperture``
    has more than one position. Otherwise, a `regions.Region` object
    will be returned.

    Parameters
    ----------
    aperture : `~photutils.aperture.Aperture`
        An `~photutils.aperture.Aperture` object to convert.

    Returns
    -------
    region : `regions.Region` or `regions.Regions`
        An equivalent `regions.Region` object. If the input ``aperture``
        has more than one position then a `regions.Regions` will be
        returned.

    Notes
    -----
    The elliptical aperture ``a`` and ``b`` parameters represent the
    semi-major and semi-minor axes, respectively. The ``a`` and ``b``
    parameters are mapped to the ellipse ``width`` and ``height`` region
    parameters, respectively, by multiplying by 2 because they represent
    the full extent of the ellipse.

    The region ``angle`` for sky-based regions is defined as the angle
    of the ``width`` axis relative to WCS longitude axis (PA=90).
    However, the sky-based apertures define the ``theta`` as the
    position angle of the semimajor axis relative to the North celestial
    pole (PA=0). Therefore, for sky-based apertures the ``theta``
    parameter is converted to the region ``angle`` by adding 90 degrees.

    .. |rarr| unicode:: U+0279E .. RIGHTWARDS ARROW

    The following `~photutils.aperture.Aperture` objects are supported,
    shown with their equivalent `regions.Region` object:

    * `~photutils.aperture.CircularAperture` |rarr|
      `~regions.CirclePixelRegion`
    * `~photutils.aperture.SkyCircularAperture` |rarr|
      `~regions.CircleSkyRegion`
    * `~photutils.aperture.EllipticalAperture` |rarr|
      `~regions.EllipsePixelRegion`
    * `~photutils.aperture.SkyEllipticalAperture` |rarr|
      `~regions.EllipseSkyRegion`
    * `~photutils.aperture.RectangularAperture` |rarr|
      `~regions.RectanglePixelRegion`
    * `~photutils.aperture.SkyRectangularAperture` |rarr|
      `~regions.RectangleSkyRegion`
    * `~photutils.aperture.CircularAnnulus` |rarr|
      `~regions.CircleAnnulusPixelRegion`
    * `~photutils.aperture.SkyCircularAnnulus` |rarr|
      `~regions.CircleAnnulusSkyRegion`
    * `~photutils.aperture.EllipticalAnnulus` |rarr|
      `~regions.EllipseAnnulusPixelRegion`
    * `~photutils.aperture.SkyEllipticalAnnulus` |rarr|
      `~regions.EllipseAnnulusSkyRegion`
    * `~photutils.aperture.RectangularAnnulus` |rarr|
      `~regions.RectangleAnnulusPixelRegion`
    * `~photutils.aperture.SkyRectangularAnnulus` |rarr|
      `~regions.RectangleAnnulusSkyRegion`

    Examples
    --------
    >>> from photutils.aperture import CircularAperture, aperture_to_region
    >>> aperture = CircularAperture((10, 20), r=5)
    >>> region = aperture_to_region(aperture)
    >>> region
    <CirclePixelRegion(center=PixCoord(x=10.0, y=20.0), radius=5.0)>

    >>> aperture = CircularAperture(((10, 20), (30, 40)), r=5)
    >>> region = aperture_to_region(aperture)
    >>> region
    <Regions([<CirclePixelRegion(center=PixCoord(x=10.0, y=20.0), radius=5.0)>,
    <CirclePixelRegion(center=PixCoord(x=30.0, y=40.0), radius=5.0)>])>
    """
    from regions import Regions

    if not isinstance(aperture, Aperture):
        msg = 'Input aperture must be an Aperture object'
        raise TypeError(msg)

    if aperture.shape == ():
        return _scalar_aperture_to_region(aperture)

    # multiple aperture positions return a Regions object
    regs = [_scalar_aperture_to_region(aper) for aper in aperture]
    return Regions(regs)


def _scalar_aperture_to_region(aperture):
    """
    Convert a given scalar `~photutils.aperture.Aperture` object to a
    `regions.Region` object.

    Parameters
    ----------
    aperture : `~photutils.aperture.Aperture`
        An `~photutils.aperture.Aperture` object to convert. The
        ``aperture`` must have a single position (scalar).

    Returns
    -------
    region : `regions.Region` or `regions.Regions`
        An equivalent `regions.Region` object.
    """
    from regions import (CircleAnnulusPixelRegion, CircleAnnulusSkyRegion,
                         CirclePixelRegion, CircleSkyRegion,
                         EllipseAnnulusPixelRegion, EllipseAnnulusSkyRegion,
                         EllipsePixelRegion, EllipseSkyRegion, PixCoord,
                         RectangleAnnulusPixelRegion,
                         RectangleAnnulusSkyRegion, RectanglePixelRegion,
                         RectangleSkyRegion)

    if aperture.shape != ():
        msg = 'Only scalar (single-position) apertures are supported'
        raise ValueError(msg)

    if isinstance(aperture, CircularAperture):
        region = CirclePixelRegion(PixCoord(*aperture.positions), aperture.r)

    elif isinstance(aperture, SkyCircularAperture):
        region = CircleSkyRegion(aperture.positions, aperture.r)

    elif isinstance(aperture, EllipticalAperture):
        region = EllipsePixelRegion(
            PixCoord(*aperture.positions), aperture.a * 2, aperture.b * 2,
            angle=aperture.theta)

    elif isinstance(aperture, SkyEllipticalAperture):
        region = EllipseSkyRegion(
            aperture.positions, aperture.a * 2, aperture.b * 2,
            angle=(aperture.theta + (90 * u.deg)))

    elif isinstance(aperture, RectangularAperture):
        region = RectanglePixelRegion(
            PixCoord(*aperture.positions), aperture.w, aperture.h,
            angle=aperture.theta)

    elif isinstance(aperture, SkyRectangularAperture):
        region = RectangleSkyRegion(
            aperture.positions, aperture.w, aperture.h,
            angle=(aperture.theta + (90 * u.deg)))

    elif isinstance(aperture, CircularAnnulus):
        region = CircleAnnulusPixelRegion(
            PixCoord(*aperture.positions), aperture.r_in, aperture.r_out)

    elif isinstance(aperture, SkyCircularAnnulus):
        region = CircleAnnulusSkyRegion(
            aperture.positions, aperture.r_in, aperture.r_out)

    elif isinstance(aperture, EllipticalAnnulus):
        region = EllipseAnnulusPixelRegion(
            PixCoord(*aperture.positions),
            aperture.a_in * 2, aperture.a_out * 2,
            aperture.b_in * 2, aperture.b_out * 2,
            angle=aperture.theta)

    elif isinstance(aperture, SkyEllipticalAnnulus):
        region = EllipseAnnulusSkyRegion(
            aperture.positions,
            aperture.a_in * 2, aperture.a_out * 2,
            aperture.b_in * 2, aperture.b_out * 2,
            angle=(aperture.theta + (90 * u.deg)))

    elif isinstance(aperture, RectangularAnnulus):
        region = RectangleAnnulusPixelRegion(
            PixCoord(*aperture.positions),
            aperture.w_in, aperture.w_out,
            aperture.h_in, aperture.h_out,
            angle=aperture.theta)

    elif isinstance(aperture, SkyRectangularAnnulus):
        region = RectangleAnnulusSkyRegion(
            aperture.positions,
            aperture.w_in, aperture.w_out,
            aperture.h_in, aperture.h_out,
            angle=(aperture.theta + (90 * u.deg)))

    else:  # pragma: no cover
        msg = 'Cannot convert input aperture to a Region object'
        raise TypeError(msg)

    return region


def _shapely_polygon_to_region(polygon, label=None):
    """
    Convert a `shapely.geometry.polygon.Polygon` object to a
    `regions.PolygonPixelRegion` object.

    Parameters
    ----------
    polygon : `shapely.geometry.polygon.Polygon` \
            or `shapely.geometry.MultiPolygon`
        A Shapely Polygon or MultiPolygon object.

    label : str or `None`, optional
        A label for the region. If provided, it will be stored in the
        meta attribute of the returned `regions.PolygonPixelRegion`
        objects.

    Returns
    -------
    result : list of `regions.PolygonPixelRegion` or `regions.Regions`
        If the polygon is a `shapely.geometry.polygon.Polygon`,
        then a `regions.PolygonPixelRegion` object is returned. If
        the polygon is a `shapely.geometry.MultiPolygon`, then a
        `regions.Regions` object is returned containing one or more
        `regions.PolygonPixelRegion` objects.

    Notes
    -----
    The `regions.PolygonPixelRegion` object does not include the
    last Shapely vertex, which is the same as the first vertex. The
    `regions.PolygonPixelRegion` does not need to include the last
    vertex to close the polygon.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from photutils.aperture.converters import _shapely_polygon_to_region
    >>> polygon = Polygon([(1, 1), (3, 1), (2, 4), (1, 2)])
    >>> region = _shapely_polygon_to_region(polygon)
    >>> region
    <PolygonPixelRegion(vertices=PixCoord(x=[1. 3. 2. 1.], y=[1. 1. 4. 2.]))>
    """
    from regions import PixCoord, PolygonPixelRegion, Regions
    from shapely.geometry import MultiPolygon, Polygon

    meta = {'label': label} if label is not None else None

    if isinstance(polygon, Polygon):
        x, y = np.transpose(polygon.exterior.coords[:-1])
        return PolygonPixelRegion(vertices=PixCoord(x=x, y=y), meta=meta)
    if isinstance(polygon, MultiPolygon):
        geoms = []
        for poly in polygon.geoms:
            x, y = np.transpose(poly.exterior.coords[:-1])
            geoms.append(PolygonPixelRegion(vertices=PixCoord(x=x, y=y),
                                            meta=meta))
        return Regions(geoms)
    msg = 'Input must be a Polygon or MultiPolygon object'
    raise TypeError(msg)
