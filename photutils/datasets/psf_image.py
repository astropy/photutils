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
    """
    Make a 2D image containing sources generated from a user-specified
    astropy 2D model.

    The model parameters for each source are taken from the input
    ``params_table`` table.

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
        the ``xname`` and ``yname`` keywords.

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names, which must match the model
        parameter names. The table must contain columns for the x and
        y positions of the sources. The column names for the x and y
        positions can be specified using the ``xname`` and ``yname``
        keywords. Model parameters not defined in the table will be set
        to the ``model`` default value. Any units in the table will be
        ignored.

        If the table contains a column named 'model_shape', then
        the values in that column will be used to override the
        ``model_shape`` keyword and model ``bounding_box`` for each
        source. This can be used to render each source with a different
        shape.

        If the table contains a column named 'local_bkg', then the
        per-pixel local background values in that column will be used
        to added to each model source over the region defined by its
        ``model_shape``. This option should be used with care,
        especially in crowded fields where the ``model_shape`` of
        sources overlap (see Notes below).

        Except for ``model_shape`` and ``local_bkg`` column names,
        column names that do not match model parameters will be ignored.

    model_shape : 2-tuple of int, int, or `None`, optional
        The shape around the (x, y) center of each source that will
        used to evaluate the ``model``. If ``model_shape`` is a scalar
        integer, then a square shape of size ``model_shape`` will
        be used. If `None`, then the bounding box of the model will
        be used. This keyword must be specified if the model does
        not have a ``bounding_box`` attribute. If specified, this
        keyword overrides the model ``bounding_box`` attribute. To
        use a different shape for each source, include a column named
        ``'model_shape'`` in the ``params_table``. For that case, this
        keyword is ignored.

    xname : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources. This parameter must also be a column
        name in ``params_table``.

    yname : str, optional
        The name of the ``model`` parameter that corresponds to the y
        position of the sources. This parameter must also be a column
        name in ``params_table``.

    discretize_method : {'center', 'interp', 'oversample', 'integrate'}, optional
        One of the following methods for discretizing the model on the
        pixel grid:

            * ``'center'`` (default)
                Discretize model by taking the value at the center of
                the pixel bins. This method should be used for ePSF/PRF
                single or gridded models.

            * ``'interp'``
                Discretize model by bilinearly interpolating between the
                values at the corners of the pixel bins.

            * ``'oversample'``
                Discretize model by taking the average of model values
                in the pixel bins on an oversampled grid. Use the
                ``discretize_oversample`` keyword to set the integer
                oversampling factor.

            * ``'integrate'``
                Discretize model by integrating the model over the pixel
                bins using `scipy.integrate.quad`. This mode conserves
                the model integral on a subpixel scale, but it is
                *extremely* slow.

    discretize_oversample : int, optional
        The integer oversampling factor used when
        ``descretize_method='oversample'``. This keyword is ignored
        otherwise.

    progress_bar : bool, optional
        Whether to display a progress bar while adding the sources
        to the image. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.
        Note that the progress bar does not currently work in the
        Jupyter console due to limitations in ``tqdm``.

    Returns
    -------
    array : 2D `~numpy.ndarray`
        The rendered image containing the model sources.

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
