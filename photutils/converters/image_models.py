# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Converters to and from the ASDF format for photutils.psf.image_models
and photutils.psf.gridded_models.
"""
import numpy as np
from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                    parameter_to_value)

__all__ = ['GriddedPSFConverter', 'ImagePSFConverter']


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
            'origin': model.origin,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import ImagePSF

        return ImagePSF(
            # Wrapping "data" as an ndarray to ensure the data is properly
            # converted from NDArrayType. This is necessary because
            # the validation in ImagePSF happens before "data" is assigned.
            data=np.array(node['data']),
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            oversampling=node['oversampling'],
            fill_value=node['fill_value'],
            origin=node['origin'],
        )


class GriddedPSFConverter(TransformConverterBase):
    """
    ASDF converter for GriddedPSFModel.
    """

    tags = ('tag:astropy.org:photutils/psf/gridded_psf-*',)
    types = ('photutils.psf.GriddedPSFModel',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        return {
            'data': model.data,
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'oversampling': model.oversampling,
            'fill_value': model.fill_value,
            'grid_xypos': model.grid_xypos,
        }

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from astropy.nddata import NDData

        from photutils.psf import GriddedPSFModel

        nd_data = NDData(
            data=np.array(node['data']),
            meta={'grid_xypos': node['grid_xypos'],
                  'oversampling': node['oversampling'],
                  },
        )

        return GriddedPSFModel(
            nddata=nd_data,
            flux=node['flux'],
            x_0=node['x_0'],
            y_0=node['y_0'],
            fill_value=node['fill_value'],
        )
