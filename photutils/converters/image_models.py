# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Converters to and from the ASDF format for photutils.psf.image_models
and photutils.psf.gridded_models.
"""

import numpy as np
from asdf.extension import Converter
from asdf_astropy.converters.transform.core import (TransformConverterBase,
                                                    parameter_to_value)

from photutils.converters._utils import optional_params

__all__ = ['GriddedPSFModelConverter',
           'ImagePSFConverter',
           'STDPSFGridConverter',
           ]


class ImagePSFConverter(TransformConverterBase):
    """
    ASDF converter for ImagePSF.
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
            # Wrapping "data" as an ndarray to ensure the data is
            # properly converted from NDArrayType. This is necessary
            # because the validation in ImagePSF happens before "data"
            # is assigned.
            data=np.array(node['data']),
            **optional_params(node, 'flux', 'x_0', 'y_0', 'oversampling',
                              'fill_value', 'origin'),
        )


class GriddedPSFModelConverter(TransformConverterBase):
    """
    ASDF converter for GriddedPSFModel.
    """

    tags = ('tag:astropy.org:photutils/psf/gridded_psf_model-*',)
    types = ('photutils.psf.GriddedPSFModel',)

    def to_yaml_tree_transform(self, model, tag, ctx):  # noqa: ARG002
        node = {
            'data': model.data,
            'flux': parameter_to_value(model.flux),
            'x_0': parameter_to_value(model.x_0),
            'y_0': parameter_to_value(model.y_0),
            'oversampling': model.oversampling,
            'fill_value': model.fill_value,
            'grid_xypos': model.grid_xypos,
        }

        # Preserve any additional meta items (e.g., 'STDPSF', 'detector',
        # 'filter'). 'grid_xypos' and 'oversampling' are stored above.
        node['meta'] = {
            key: value for key, value in model.meta.items()
            if key not in ('grid_xypos', 'oversampling')
        }

        return node

    def from_yaml_tree_transform(self, node, tag, ctx):  # noqa: ARG002
        from astropy.nddata import NDData

        from photutils.psf import GriddedPSFModel

        meta = dict(node['meta'])
        meta['grid_xypos'] = node['grid_xypos']
        meta['oversampling'] = node['oversampling']

        nd_data = NDData(data=np.array(node['data']), meta=meta)

        return GriddedPSFModel(
            nddata=nd_data,
            **optional_params(node, 'flux', 'x_0', 'y_0', 'fill_value'),
        )


class STDPSFGridConverter(Converter):
    """
    ASDF converter for STDPSFGrid.
    """

    tags = ('tag:astropy.org:photutils/psf/stdpsf_grid-*',)
    types = ('photutils.psf.STDPSFGrid',)

    def to_yaml_tree(self, model, tag, ctx):  # noqa: ARG002
        # The grid structure is stored as separate properties, as it is
        # for GriddedPSFModel, leaving 'meta' for the remaining metadata
        # (e.g., 'STDPSF', 'detector', 'filter'). STDPSFGrid consumes
        # these three items when it is initialized, so they never appear
        # in its meta attribute.
        return {
            'data': model.data,
            'grid_xypos': np.array(model.grid_xypos),
            'grid_shape': model.grid_shape,
            'oversampling': tuple(int(value)
                                  for value in model.oversampling),
            'meta': dict(model.meta),
        }

    def from_yaml_tree(self, node, tag, ctx):  # noqa: ARG002
        from photutils.psf import STDPSFGrid

        meta = dict(node['meta'])
        # Load the lazily-read grid positions into memory
        meta['grid_xypos'] = np.asarray(node['grid_xypos'])
        meta['grid_shape'] = node['grid_shape']
        meta['oversampling'] = node['oversampling']

        return STDPSFGrid._from_asdf(node['data'], meta)
