# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Converters to and from the ASDF format for photutils.psf.image_models.
"""
from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                    parameter_to_value)

__all__ = ['ImagePSFConverter']

class ImagePSFConverter(TransformConverterBase):
    """
    ASDF converter for ImagePSF model.
    """

    tags = ('tag:astropy.org:photutils/psf/image_psf-*',)
    types = ('photutils.psf.ImagePSF',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'data': model.data,
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'oversampling': model.oversampling,
            'fill_value': model.fill_value,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import ImagePSF

        return ImagePSF(
            data=node['data'],
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            oversampling=node['oversampling'],
            fill_value=node['fill_value'],
        )
