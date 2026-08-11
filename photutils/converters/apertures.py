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


class CircularApertureConverter(Converter):
    """
    ASDF converter for circular apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/circular_aperture-*',)
    types = ('photutils.aperture.CircularAperture',
             'photutils.aperture.SkyCircularAperture',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'r': obj.r,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyCircularAperture as Aper
        else:
            from photutils.aperture import CircularAperture as Aper

        return Aper(
            positions=node['positions'],
            r=node['r'],
        )


class CircularAnnulusConverter(Converter):
    """
    ASDF converter for circular annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/circular_annulus-*',)
    types = ('photutils.aperture.CircularAnnulus',
             'photutils.aperture.SkyCircularAnnulus',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'r_in': obj.r_in,
            'r_out': obj.r_out,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyCircularAnnulus as Aper
        else:
            from photutils.aperture import CircularAnnulus as Aper

        return Aper(
            positions=node['positions'],
            r_in=node['r_in'],
            r_out=node['r_out'],
        )


class EllipticalApertureConverter(Converter):
    """
    ASDF converter for elliptical apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/elliptical_aperture-*',)
    types = ('photutils.aperture.EllipticalAperture',
             'photutils.aperture.SkyEllipticalAperture',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'a': obj.a,
            'b': obj.b,
            'theta': obj.theta,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyEllipticalAperture as Aper
        else:
            from photutils.aperture import EllipticalAperture as Aper

        return Aper(
            positions=node['positions'],
            a=node['a'],
            b=node['b'],
            **optional_params(node, 'theta'),
        )


class EllipticalAnnulusConverter(Converter):
    """
    ASDF converter for elliptical annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/elliptical_annulus-*',)
    types = ('photutils.aperture.EllipticalAnnulus',
             'photutils.aperture.SkyEllipticalAnnulus',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'a_in': obj.a_in,
            'a_out': obj.a_out,
            'b_in': obj.b_in,
            'b_out': obj.b_out,
            'theta': obj.theta,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyEllipticalAnnulus as Aper
        else:
            from photutils.aperture import EllipticalAnnulus as Aper

        return Aper(
            positions=node['positions'],
            a_in=node['a_in'],
            a_out=node['a_out'],
            b_out=node['b_out'],
            **optional_params(node, 'b_in', 'theta'),
        )


class PolygonApertureConverter(Converter):
    """
    ASDF converter for polygon apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/polygon_aperture-*',)
    types = ('photutils.aperture.PolygonAperture',
             'photutils.aperture.SkyPolygonAperture',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'vertex_offsets': obj.vertex_offsets,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyPolygonAperture as Aper
        else:
            from photutils.aperture import PolygonAperture as Aper

        return Aper(
            positions=node['positions'],
            vertex_offsets=node['vertex_offsets'],
        )


class RectangularApertureConverter(Converter):
    """
    ASDF converter for rectangular apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/rectangular_aperture-*',)
    types = ('photutils.aperture.RectangularAperture',
             'photutils.aperture.SkyRectangularAperture',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'w': obj.w,
            'h': obj.h,
            'theta': obj.theta,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyRectangularAperture as Aper
        else:
            from photutils.aperture import RectangularAperture as Aper

        return Aper(
            positions=node['positions'],
            w=node['w'],
            h=node['h'],
            **optional_params(node, 'theta'),
        )


class RectangularAnnulusConverter(Converter):
    """
    ASDF converter for rectangular annulus apertures.
    """

    tags = ('tag:astropy.org:photutils/aperture/rectangular_annulus-*',)
    types = ('photutils.aperture.RectangularAnnulus',
             'photutils.aperture.SkyRectangularAnnulus',
             )

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        pos = obj.positions
        if not isinstance(pos, SkyCoord) and obj.positions.shape == (2,):
            pos = obj.positions.tolist()

        return {
            'positions': pos,
            'w_in': obj.w_in,
            'w_out': obj.w_out,
            'h_in': obj.h_in,
            'h_out': obj.h_out,
            'theta': obj.theta,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        if isinstance(node['positions'], SkyCoord):
            from photutils.aperture import SkyRectangularAnnulus as Aper
        else:
            from photutils.aperture import RectangularAnnulus as Aper

        return Aper(
            positions=node['positions'],
            w_in=node['w_in'],
            w_out=node['w_out'],
            h_out=node['h_out'],
            **optional_params(node, 'h_in', 'theta'),
        )
