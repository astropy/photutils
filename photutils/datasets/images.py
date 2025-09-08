# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide tools for making simulated images for documentation examples and
tests.
"""

import astropy.units as u
import numpy as np
from astropy.convolution import discretize_model
from astropy.modeling import Model
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table

from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['make_model_image']


def make_model_image(shape, model, params_table, *, model_shape=None,
                     bbox_factor=None, x_name='x_0', y_name='y_0',
                     params_map=None, discretize_method='center',
                     discretize_oversample=10, progress_bar=False):
    """
    Make a 2D image containing sources generated from a user-specified
    astropy 2D model.

    The model parameters for each source are taken from the input
    ``params_table`` table. By default, the table is searched for column
    names that match model parameter names and the values specified by
    ``x_name`` and ``y_name``. However, the user can specify a different
    mapping between model parameter names and column names using the
    ``params_map`` keyword.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    model : 2D `astropy.modeling.Model`
        The 2D model to be used to render the sources. The model must
        be two-dimensional where it accepts 2 inputs (i.e., (x, y)) and
        has 1 output. The model must have parameters for the x and y
        positions of the sources. Typically, these parameters are named
        'x_0' and 'y_0', but the parameter names can be specified using
        the ``x_name`` and ``y_name`` keywords.

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names, which must match the model
        parameter names. The table must contain columns for the x
        and y positions of the sources. The column names for the x
        and y positions can be specified using the ``x_name`` and
        ``y_name`` keywords. Model parameters not defined in the table
        or ``params_maps`` will be set to the ``model`` default value.
        To attach units to model parameters, ``params_table`` must
        be input as a `~astropy.table.QTable`. Rows that contain any
        non-finite model parameters will be skipped.

        If the table contains a column named 'model_shape', then
        the values in that column will be used to override the
        ``model_shape`` keyword and model ``bounding_box`` for each
        source. This can be used to render each source with a different
        shape.

        If the table contains a column named 'local_bkg', then the
        per-pixel local background values in that column will be used
        to added to each model source over the region defined by its
        ``model_shape``. The 'local_bkg' column must have the same
        flux units as the output image (e.g., if the input ``model``
        has 'amplitude' or 'flux' parameters with units). Including
        'local_bkg' should be used with care, especially in crowded
        fields where the ``model_shape`` of sources overlap (see Notes
        below).

        Except for ``model_shape`` and ``local_bkg`` column names,
        column names that do not match model parameters will be ignored
        unless ``params_map`` is input.

    model_shape : 2-tuple of int, int, or `None`, optional
        The shape around the (x, y) center of each source that will
        used to evaluate the ``model``. If ``model_shape`` is a scalar
        integer, then a square shape of size ``model_shape`` will be
        used. If `None`, then the bounding box of the model will be
        used (which can optionally be scaled using the ``bbox_factor``
        keyword if the model supports it). This keyword must be
        specified if the model does not have a ``bounding_box``
        attribute. If specified, this keyword overrides the model
        ``bounding_box`` attribute. To use a different shape for
        each source, include a column named ``'model_shape'`` in the
        ``params_table``. For that case, this keyword is ignored.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. If the model
        ``bounding_box`` method does not accept a ``factor`` keyword,
        then this keyword is ignored. If `None`, the default model
        bounding box will be used. This keyword is ignored if
        ``model_shape`` is specified or if the ``params_table`` contains
        a ``'model_shape'`` column. Note that some Photutils PSF models
        have a ``bbox_factor`` keyword that is be used to define the
        model bounding box. In that case, this keyword is ignored.

    x_name : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources. If ``param_map`` is not input, then
        this value must also be a column name in ``params_table``.

    y_name : str, optional
        The name of the ``model`` parameter that corresponds to the y
        position of the sources. If ``param_map`` is not input, then
        this value must also be a column name in ``params_table``.

    params_map : dict or None, optional
        A dictionary mapping the model parameter names to the column
        names in the input ``params_table``. The dictionary keys are
        the model parameter names and the values are the column names
        in the input ``params_table``. This can be used to map column
        names to model parameter names that are different. For example,
        if the input column name is 'flux_f200w' and the model parameter
        name is 'flux', then use ``column_map={'flux': 'flux_f200w'}``.
        This table may also be used if you want to map the model x and y
        parameters to different columns than ``x_name`` and ``y_name``,
        but the ``x_name`` and ``y_name`` keys must be included in the
        dictionary.

    discretize_method : {'center', 'interp', 'oversample', 'integrate'}, \
            optional
        One of the following methods for discretizing the model on the
        pixel grid:

        * ``'center'`` (default)
          Discretize model by taking the value at the center of the
          pixel bins. This method should be used for ePSF/PRF single or
          gridded models.

        * ``'interp'``
          Discretize model by bilinearly interpolating between the
          values at the corners of the pixel bins.

        * ``'oversample'``
          Discretize model by taking the average of model values
          in the pixel bins on an oversampled grid. Use the
          ``discretize_oversample`` keyword to set the integer
          oversampling factor.

        * ``'integrate'``
          Discretize model by integrating the model over the pixel bins
          using `scipy.integrate.quad`. This mode conserves the model
          integral on a subpixel scale, but it is *extremely* slow.

    discretize_oversample : int, optional
        The integer oversampling factor used when
        ``descretize_method='oversample'``. This keyword is ignored
        otherwise.

    progress_bar : bool, optional
        Whether to display a progress bar while adding the sources
        to the image. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Returns
    -------
    array : 2D `~numpy.ndarray`
        The rendered image containing the model sources.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.modeling.models import Moffat2D
        from photutils.datasets import (make_model_image,
                                        make_random_models_table)

        model = Moffat2D()
        n_sources = 25
        shape = (100, 100)
        param_ranges = {'amplitude': [100, 200],
                        'x_0': [0, shape[1]],
                        'y_0': [0, shape[0]],
                        'gamma': [1, 2],
                        'alpha': [1, 2]}
        params = make_random_models_table(n_sources, param_ranges, seed=0)

        model_shape = (15, 15)
        data = make_model_image(shape, model, params, model_shape=model_shape)

        plt.imshow(data, origin='lower')
        plt.tight_layout()

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from photutils.datasets import make_model_image, make_model_params

        model = Gaussian2D()
        shape = (500, 500)
        n_sources = 100
        params = make_model_params(shape, n_sources, x_name='x_mean',
                                   y_name='y_mean', min_separation=25,
                                   amplitude=(100, 500), x_stddev=(1, 3),
                                   y_stddev=(1, 3), theta=(0, np.pi))
        model_shape = (25, 25)
        data = make_model_image(shape, model, params, model_shape=model_shape,
                                x_name='x_mean', y_name='y_mean')

        plt.imshow(data, origin='lower')
        plt.tight_layout()

    Notes
    -----
    The local background value around each source is optionally included
    using the ``local_bkg`` column in the input ``params_table``. This
    local background added to each source over its ``model_shape``
    region. In regions where the ``model_shape`` of source overlap, the
    local background will be added multiple times. This is not an issue
    if the sources are well-separated, but for crowded fields, this
    option should be used with care.
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        msg = 'shape must be a 2-tuple'
        raise ValueError(msg)

    if not isinstance(model, Model):
        msg = 'model must be a Model instance'
        raise TypeError(msg)
    if model.n_inputs != 2 or model.n_outputs != 1:
        msg = 'model must be a 2D model'
        raise ValueError(msg)
    if not isinstance(params_table, Table):
        msg = 'params_table must be an astropy Table'
        raise TypeError(msg)

    xypos_map = {x_name: x_name, y_name: y_name}

    # by default, use the model parameter names as the column names
    # if they are in the table
    params_to_set = set(params_table.colnames) & set(model.param_names)
    xypos_map.update({param: param for param in params_to_set})

    if params_map is not None:
        # params_map takes precedence over x_name and y_name and
        # any matching column names in params_table
        xypos_map.update(params_map)
    params_map = xypos_map

    for key, value in params_map.items():
        if key not in model.param_names:
            msg = f'key "{key}" not in model parameter names'
            raise ValueError(msg)
        if value not in params_table.colnames:
            msg = f'value "{value}" not in params_table column names'
            raise ValueError(msg)

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
            _ = model.bounding_box
        except NotImplementedError as exc:
            msg = ('model_shape must be specified if the model does not '
                   'have a bounding_box attribute')
            raise ValueError(msg) from exc

    if 'local_bkg' in params_table.colnames:
        local_bkg = params_table['local_bkg']
    else:
        local_bkg = np.zeros(len(params_table))

    # copy the input model to leave it unchanged
    model = model.copy()

    if progress_bar:  # pragma: no cover
        desc = 'Add model sources'
        params_table = add_progress_bar(params_table, desc=desc)

    image = np.zeros(shape, dtype=float)
    apply_units = True
    for i, source in enumerate(params_table):
        for key, param in params_map.items():
            value = source[param]
            # skip if any parameter value is not finite
            if not np.isfinite(value):
                break
            setattr(model, key, value)

        else:  # all parameters are finite for the source
            # This assumes that if the user also uses params_table to
            # override the (x/y)_name mapping that the x_name and y_name
            # values are correct (i.e., the mapping keys include x_name
            # and y_name). There is no good way to check/enforce this.
            x0 = getattr(model, x_name).value
            y0 = getattr(model, y_name).value

            if variable_shape:
                mod_shape = model_shape[i]
            elif model_shape is None:
                # the bounding box size generally depends on model
                # parameters, so needs to be calculated for each source
                mod_shape = _model_shape_from_bbox(model,
                                                   bbox_factor=bbox_factor)
            else:
                mod_shape = model_shape

            try:
                slc_lg, _ = overlap_slices(shape, mod_shape, (y0, x0),
                                           mode='trim')

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

                # if the model is a Quantity, then the output image
                # should also be a Quantity with the same units;
                # but apply the units only once
                if apply_units and isinstance(subimg, u.Quantity):
                    apply_units = False
                    image <<= subimg.unit

                try:
                    image[slc_lg] += subimg + local_bkg[i]
                except u.UnitConversionError as exc:
                    msg = ('The local_bkg column must have the same flux '
                           'units as the output image')
                    raise ValueError(msg) from exc

            except NoOverlapError:
                # evaluate the model to get the model output units
                result = model(0, 0)
                if isinstance(result, u.Quantity):
                    image <<= result.unit
                continue

    return image


def _model_shape_from_bbox(model, bbox_factor=None):
    """
    Calculate the model shape from the model bounding box.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model to be used to render the sources.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. If the model
        ``bounding_box`` method does not accept a ``factor`` keyword,
        then this keyword is ignored. If `None`, the default model
        bounding box will be used.

    Returns
    -------
    model_shape : 2-tuple of int
        The shape around the (x, y) center of the model that will used
        to evaluate the model.

    Raises
    ------
    ValueError
        If the model does not have a bounding_box attribute.
    """
    try:
        hasattr(model, 'bounding_box')
    except NotImplementedError as exc:
        msg = 'model does not have a bounding_box attribute'
        raise ValueError(msg) from exc

    if bbox_factor is not None:
        try:
            bbox = model.bounding_box(factor=bbox_factor)
        except NotImplementedError:
            bbox = model.bounding_box.bounding_box()
    else:
        bbox = model.bounding_box.bounding_box()

    return (int(np.ceil(bbox[0][1] - bbox[0][0])),
            int(np.ceil(bbox[1][1] - bbox[1][0])))
