# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.convolution import discretize_model
from astropy.modeling import Model
from astropy.nddata import overlap_slices
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table

from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar

__all__ = ['make_model_image']


def make_model_image(shape, model, params_table, *, model_shape=None,
                     xname='x_0', yname='y_0', discretize_method='center',
                     discretize_oversample=10, progress_bar=False):

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError('shape must be a 2-tuple')

    if not isinstance(model, Model):
        raise ValueError('model must be a Model instance')
    if model.n_inputs != 2 or model.n_outputs != 1:
        raise ValueError('model must be a 2D model')
    if xname not in model.param_names:
        raise ValueError(f'xname "{xname}" not in model parameter names')
    if yname not in model.param_names:
        raise ValueError(f'yname "{yname}" not in model parameter names')

    if not isinstance(params_table, Table):
        raise ValueError('params_table must be an astropy Table')
    if xname not in params_table.colnames:
        raise ValueError(f'xname "{xname}" not in psf_params column names')
    if yname not in params_table.colnames:
        raise ValueError(f'yname "{yname}" not in psf_params column names')

    if model_shape is not None:
        model_shape = as_pair('model_shape', model_shape, lower_bound=(0, 1))

    variable_shape = False
    if 'model_shape' in params_table.colnames:
        model_shape = np.array(params_table['model_shape'])
        if model_shape.ndim == 1:
            model_shape = np.array([as_pair('model_shape', shape)
                                    for shape in model_shape])
        variable_shape = True

    if model_shape is None:
        try:
            bbox = model.bounding_box.bounding_box()
            model_shape = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0])
        except NotImplementedError:
            raise ValueError('model_shape must be specified if the model '
                             'does not have a bounding_box attribute')

    if 'local_bkg' in params_table.colnames:
        local_bkg = params_table['local_bkg']
    else:
        local_bkg = np.zeros(len(params_table))

    # include only column names that are model parameters
    params_to_set = set(params_table.colnames) & set(model.param_names)

    # save the model initial parameter values so we can restore them
    init_params = {param: getattr(model, param) for param in params_to_set}

    if progress_bar:  # pragma: no cover
        desc = 'Add model sources'
        params_table = add_progress_bar(params_table, desc=desc)

    image = np.zeros(shape, dtype=float)
    for i, source in enumerate(params_table):
        for param in params_to_set:
            setattr(model, param, source[param])

        x0 = getattr(model, xname).value
        y0 = getattr(model, yname).value

        if variable_shape:
            mod_shape = model_shape[i]
        else:
            mod_shape = model_shape

        try:
            slc_lg, _ = overlap_slices(shape, mod_shape, (y0, x0), mode='trim')

            if discretize_method == 'center':
                yy, xx = np.mgrid[slc_lg]
                subimg = model(xx, yy)
            else:
                if discretize_method == 'interp':
                    discretize_method = 'linear_interp'
                x_range = (slc_lg[1].start, slc_lg[1].stop)
                y_range = (slc_lg[0].start, slc_lg[0].stop)
                subimg = discretize_model(model, x_range=x_range,
                                          y_range=y_range,
                                          mode=discretize_method,
                                          factor=discretize_oversample)

            image[slc_lg] += subimg + local_bkg[i]

        except NoOverlapError:
            continue

    # restore the model initial parameter values
    for param, value in init_params.items():
        setattr(model, param, value)

    return image
