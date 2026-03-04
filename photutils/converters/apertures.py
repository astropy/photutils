# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Converters to and from the ASDF data format for photutils apertures.
"""

from asdf.extension import Converter

__all__ = ["CircularApertureConverter"]


class CircularApertureConverter(Converter):
    """
    Base class for aperture converters.
    """
    tags = ["tag:astropy.org:photutils/aperture/circular_aperture-*"]
    types = ["photutils.aperture.circle.CircularAperture"]

    def to_yaml_tree(self, obj, tag, ctx):
        if obj.positions.shape == (2,):
            pos = obj.positions.tolist()
        else:
            pos = obj.positions

        return {
            "positions": pos,
            "r": obj.r,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from photutils.aperture.circle import CircularAperture

        return CircularAperture(
            positions=node["positions"],
            r=node["r"],
        )
