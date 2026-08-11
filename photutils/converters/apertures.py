# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Converters to and from the ASDF data format for photutils apertures.
"""

from asdf.extension import Converter
from astropy.coordinates import SkyCoord

from photutils.converters._utils import optional_params

__all__ = [
    'CircularAnnulusConverter',
    'CircularApertureConverter',
    'EllipticalAnnulusConverter',
    'EllipticalApertureConverter',
    'PolygonApertureConverter',
    'RectangularAnnulusConverter',
    'RectangularApertureConverter',
]


class _ApertureConverter(Converter):
    """
    Base class for the photutils aperture converters.

    The pixel and sky variants of an aperture share a tag, so a single
    converter handles both. Subclasses list the pixel class first and
    the sky class second in ``types``, and name the aperture parameters
    that accompany ``positions``.

    Every parameter is written to the file. Those in
    ``optional_aperture_params`` may be absent when reading, in which
    case the aperture class supplies its default. They are omitted
    rather than passed as `None` because the defaults are not always
    `None` and differ between the pixel and sky apertures.
    """

    aperture_params = ()
    optional_aperture_params = ()

    def _aperture_class(self, positions):
        """
        Return the pixel or sky aperture class for ``positions``.
        """
        from photutils import aperture

        index = 1 if isinstance(positions, SkyCoord) else 0
        return getattr(aperture, self.types[index].rsplit('.', 1)[-1])

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        positions = obj.positions
        if not isinstance(positions, SkyCoord) and positions.shape == (2,):
            # store a single pixel position as a plain (x, y) pair
            positions = positions.tolist()

        node = {'positions': positions}
        for name in (*self.aperture_params, *self.optional_aperture_params):
            node[name] = getattr(obj, name)
        return node

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        positions = node['positions']
        params = {name: node[name] for name in self.aperture_params}

        return self._aperture_class(positions)(
            positions=positions,
            **params,
            **optional_params(node, *self.optional_aperture_params),
        )


class CircularApertureConverter(_ApertureConverter):
    """
    ASDF converter for circular apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/circular_aperture-*',)
    types = ('photutils.aperture.CircularAperture',
             'photutils.aperture.SkyCircularAperture',
             )
    aperture_params = ('r',)


class CircularAnnulusConverter(_ApertureConverter):
    """
    ASDF converter for circular annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/circular_annulus-*',)
    types = ('photutils.aperture.CircularAnnulus',
             'photutils.aperture.SkyCircularAnnulus',
             )
    aperture_params = ('r_in', 'r_out')


class EllipticalApertureConverter(_ApertureConverter):
    """
    ASDF converter for elliptical apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/elliptical_aperture-*',)
    types = ('photutils.aperture.EllipticalAperture',
             'photutils.aperture.SkyEllipticalAperture',
             )
    aperture_params = ('a', 'b')
    optional_aperture_params = ('theta',)


class EllipticalAnnulusConverter(_ApertureConverter):
    """
    ASDF converter for elliptical annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/elliptical_annulus-*',)
    types = ('photutils.aperture.EllipticalAnnulus',
             'photutils.aperture.SkyEllipticalAnnulus',
             )
    aperture_params = ('a_in', 'a_out', 'b_out')
    optional_aperture_params = ('b_in', 'theta')


class PolygonApertureConverter(_ApertureConverter):
    """
    ASDF converter for polygon apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/polygon_aperture-*',)
    types = ('photutils.aperture.PolygonAperture',
             'photutils.aperture.SkyPolygonAperture',
             )
    aperture_params = ('vertex_offsets',)


class RectangularApertureConverter(_ApertureConverter):
    """
    ASDF converter for rectangular apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/rectangular_aperture-*',)
    types = ('photutils.aperture.RectangularAperture',
             'photutils.aperture.SkyRectangularAperture',
             )
    aperture_params = ('w', 'h')
    optional_aperture_params = ('theta',)


class RectangularAnnulusConverter(_ApertureConverter):
    """
    ASDF converter for rectangular annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/rectangular_annulus-*',)
    types = ('photutils.aperture.RectangularAnnulus',
             'photutils.aperture.SkyRectangularAnnulus',
             )
    aperture_params = ('w_in', 'w_out', 'h_out')
    optional_aperture_params = ('h_in', 'theta')
