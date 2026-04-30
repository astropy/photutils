# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Converters to and from the ASDF data format for photutils apertures.
"""

from asdf.extension import Converter

__all__ = ['CircularApertureConverter']


class CircularApertureConverter(Converter):
    """
    ASDF converter for CircularAperture.
    """

    tags = ('tag:astropy.org:photutils/aperture/circular_aperture-*',)
    types = ('photutils.aperture.CircularAperture',)

    def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
        if obj.positions.shape == (2,):
            pos = obj.positions.tolist()
        else:
            pos = obj.positions

        return {
            'positions': pos,
            'r': obj.r,
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        from photutils.aperture import CircularAperture

        return CircularAperture(
            positions=node['positions'],
            r=node['r'],
        )
