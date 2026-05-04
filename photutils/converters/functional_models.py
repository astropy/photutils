# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Converters to and from the ASDF format for photutils.psf.functional_models.
"""

try:
    from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                        parameter_to_value)

    _ASDF_ASTROPY_AVAILABLE = True
except ImportError:
    from asdf.extension import Converter as TransformConverterBase

    _ASDF_ASTROPY_AVAILABLE = False

__all__ = ['AiryDiskPSFConverter']


class AiryDiskPSFConverter(TransformConverterBase):
    """
    Converter for AiryDiskPSF.
    """

    tags = ('tag:astropy.org:photutils/psf/airy_disk_psf-*',)
    types = ('photutils.psf.AiryDiskPSF',)

    if _ASDF_ASTROPY_AVAILABLE:
        def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
            return {
                'flux': parameter_to_value(model.flux),
                'x_0': parameter_to_value(model.x_0),
                'y_0': parameter_to_value(model.y_0),
                'radius': parameter_to_value(model.radius),
                'bbox_factor': model.bbox_factor,
            }

        def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
            from photutils.psf import AiryDiskPSF

            return AiryDiskPSF(
                flux=node['flux'],
                x_0=node['x_0'],
                y_0=node['y_0'],
                radius=node['radius'],
                bbox_factor=node['bbox_factor'],
            )
    else:
        def to_yaml_tree(self, obj, tag, ctx):  # noqa: ARG002
            msg = (
                'asdf-astropy must be installed to serialize AiryDiskPSF '
                'to ASDF format. Install it with:\n'
                '    pip install asdf-astropy'
            )
            raise ImportError(msg)

        def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
            msg = (
                'asdf-astropy must be installed to deserialize AiryDiskPSF '
                'from ASDF format. Install it with:\n'
                '    pip install asdf-astropy'
            )
            raise ImportError(msg)
