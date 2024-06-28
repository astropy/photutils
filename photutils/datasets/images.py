# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for making simulated images for documentation
examples and tests.
"""

import math
import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import discretize_model
from astropy.modeling import Model
from astropy.modeling.models import Gaussian2D
from astropy.nddata.utils import NoOverlapError
from astropy.table import QTable, Table
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._coords import make_random_xycoords
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['make_model_image', 'make_model_sources_image',
           'make_gaussian_sources_image', 'make_gaussian_prf_sources_image',
           'make_test_psf_data']


def make_model_image(shape, model, params_table, *, model_shape=None,
                     bbox_factor=None, x_name='x_0', y_name='y_0',
                     discretize_method='center', discretize_oversample=10,
                     progress_bar=False):
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
        the ``x_name`` and ``y_name`` keywords.

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source.
        Each row of the table corresponds to a source whose model
        parameters are defined by the column names, which must match
        the model parameter names. The table must contain columns for
        the x and y positions of the sources. The column names for
        the x and y positions can be specified using the ``x_name``
        and ``y_name`` keywords. Model parameters not defined in the
        table will be set to the ``model`` default value. To attach
        units to model parameters, ``params_table`` must be input as a
        `~astropy.table.QTable`.

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
        column names that do not match model parameters will be ignored.

    model_shape : 2-tuple of int, int, or `None`, optional
        The shape around the (x, y) center of each source that will
        used to evaluate the ``model``. If ``model_shape`` is a scalar
        integer, then a square shape of size ``model_shape`` will be
        used. If `None`, then the bounding box of the model will be
        used (which can optionally be scaled using the ``bbox_factor``
        keyword). This keyword must be specified if the model does
        not have a ``bounding_box`` attribute. If specified, this
        keyword overrides the model ``bounding_box`` attribute. To
        use a different shape for each source, include a column named
        ``'model_shape'`` in the ``params_table``. For that case, this
        keyword is ignored.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. If `None`, the default
        model bounding box will be used. This keyword is ignored if
        ``model_shape`` is specified or if the ``params_table`` contains
        a ``'model_shape'`` column.

    x_name : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources. This parameter must also be a column
        name in ``params_table``.

    y_name : str, optional
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
        raise ValueError('shape must be a 2-tuple')

    if not isinstance(model, Model):
        raise ValueError('model must be a Model instance')
    if model.n_inputs != 2 or model.n_outputs != 1:
        raise ValueError('model must be a 2D model')
    if x_name not in model.param_names:
        raise ValueError(f'x_name "{x_name}" not in model parameter names')
    if y_name not in model.param_names:
        raise ValueError(f'y_name "{y_name}" not in model parameter names')

    if not isinstance(params_table, Table):
        raise ValueError('params_table must be an astropy Table')
    if x_name not in params_table.colnames:
        raise ValueError(f'x_name "{x_name}" not in psf_params column names')
    if y_name not in params_table.colnames:
        raise ValueError(f'y_name "{y_name}" not in psf_params column names')

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
        except NotImplementedError:
            raise ValueError('model_shape must be specified if the model '
                             'does not have a bounding_box attribute')

    if 'local_bkg' in params_table.colnames:
        local_bkg = params_table['local_bkg']
    else:
        local_bkg = np.zeros(len(params_table))

    # include only column names that are model parameters
    params_to_set = set(params_table.colnames) & set(model.param_names)

    # copy the input model to leave it unchanged
    model = model.copy()

    if progress_bar:  # pragma: no cover
        desc = 'Add model sources'
        params_table = add_progress_bar(params_table, desc=desc)

    image = np.zeros(shape, dtype=float)
    for i, source in enumerate(params_table):
        for param in params_to_set:
            setattr(model, param, source[param])

        x0 = getattr(model, x_name).value
        y0 = getattr(model, y_name).value

        if variable_shape:
            mod_shape = model_shape[i]
        else:
            if model_shape is None:
                # the bounding box size generally depends on model parameters,
                # so needs to be calculated for each source
                if bbox_factor is not None:
                    bbox = model.bounding_box(factor=bbox_factor)
                else:
                    bbox = model.bounding_box.bounding_box()
                mod_shape = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0])
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

            if i == 0 and isinstance(subimg, u.Quantity):
                image <<= subimg.unit
            try:
                image[slc_lg] += subimg + local_bkg[i]
            except u.UnitConversionError:
                raise ValueError('The local_bkg column must have the same '
                                 'flux units as the output image')

        except NoOverlapError:
            continue

    return image


@deprecated('1.13.0', alternative='make_model_image')
def make_model_sources_image(shape, model, source_table, oversample=1):
    """
    Make an image containing sources generated from a user-specified
    model.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output 2D image.

    model : 2D astropy.modeling.models object
        The model to be used for rendering the sources.

    source_table : `~astropy.table.Table`
        Table of parameters for the sources. Each row of the table
        corresponds to a source whose model parameters are defined by
        the column names, which must match the model parameter names.
        Column names that do not match model parameters will be ignored.
        Model parameters not defined in the table will be set to the
        ``model`` default value.

    oversample : float, optional
        The sampling factor used to discretize the models on a pixel
        grid. If the value is 1.0 (the default), then the models will
        be discretized by taking the value at the center of the pixel
        bin. Note that this method will not preserve the total flux of
        very small sources. Otherwise, the models will be discretized by
        taking the average over an oversampled grid. The pixels will be
        oversampled by the ``oversample`` factor.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing model sources.
    """
    image = np.zeros(shape, dtype=float)
    yidx, xidx = np.indices(shape)

    params_to_set = []
    for param in source_table.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    # Save the initial parameter values so we can set them back when
    # done with the loop. It's best not to copy a model, because some
    # models (e.g., PSF models) may have substantial amounts of data in
    # them.
    init_params = {param: getattr(model, param) for param in params_to_set}

    try:
        for source in source_table:
            for param in params_to_set:
                setattr(model, param, source[param])

            if oversample == 1:
                image += model(xidx, yidx)
            else:
                image += discretize_model(model, (0, shape[1]),
                                          (0, shape[0]), mode='oversample',
                                          factor=oversample)
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return image


@deprecated('1.13.0', alternative='make_model_image')
def make_gaussian_sources_image(shape, source_table, oversample=1):
    r"""
    Make an image containing 2D Gaussian sources.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output 2D image.

    source_table : `~astropy.table.Table`
        Table of parameters for the Gaussian sources. Each row of the
        table corresponds to a Gaussian source whose parameters are
        defined by the column names. With the exception of ``'flux'``,
        column names that do not match model parameters will be ignored
        (flux will be converted to amplitude). If both ``'flux'`` and
        ``'amplitude'`` are present, then ``'flux'`` will be ignored.
        Model parameters not defined in the table will be set to the
        default value.

    oversample : float, optional
        The sampling factor used to discretize the models on a pixel
        grid. If the value is 1.0 (the default), then the models will
        be discretized by taking the value at the center of the pixel
        bin. Note that this method will not preserve the total flux of
        very small sources. Otherwise, the models will be discretized by
        taking the average over an oversampled grid. The pixels will be
        oversampled by the ``oversample`` factor.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 2D Gaussian sources.
    """
    model = Gaussian2D(x_stddev=1, y_stddev=1)

    if 'x_stddev' in source_table.colnames:
        xstd = source_table['x_stddev']
    else:
        xstd = model.x_stddev.value  # default
    if 'y_stddev' in source_table.colnames:
        ystd = source_table['y_stddev']
    else:
        ystd = model.y_stddev.value  # default

    colnames = source_table.colnames
    if 'flux' in colnames and 'amplitude' not in colnames:
        source_table = source_table.copy()
        source_table['amplitude'] = (source_table['flux']
                                     / (2.0 * np.pi * xstd * ystd))

    return make_model_image(shape, model, source_table, x_name='x_mean',
                            y_name='y_mean', discretize_oversample=oversample)


@deprecated('1.13.0', alternative='make_psf_model_image')
def make_gaussian_prf_sources_image(shape, source_table):
    r"""
    Make an image containing 2D Gaussian sources.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output 2D image.

    source_table : `~astropy.table.Table`
        Table of parameters for the Gaussian sources. Each row of the
        table corresponds to a Gaussian source whose parameters are
        defined by the column names. With the exception of ``'flux'``,
        column names that do not match model parameters will be ignored
        (flux will be converted to amplitude). If both ``'flux'`` and
        ``'amplitude'`` are present, then ``'flux'`` will be ignored.
        Model parameters not defined in the table will be set to the
        default value.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 2D Gaussian sources.
    """
    from photutils.psf import IntegratedGaussianPRF

    model = IntegratedGaussianPRF(sigma=1)

    if 'sigma' in source_table.colnames:
        sigma = source_table['sigma']
    else:
        sigma = model.sigma.value  # default

    colnames = source_table.colnames
    if 'flux' not in colnames and 'amplitude' in colnames:
        source_table = source_table.copy()
        source_table['flux'] = (source_table['amplitude']
                                * (2.0 * np.pi * sigma * sigma))

    return make_model_image(shape, model, source_table)


def _define_psf_shape(psf_model, psf_shape):  # pragma: no cover
    """
    Define the shape of the model to evaluate, including the
    oversampling.

    Deprecated with make_test_psf_data.
    """
    try:
        model_ndim = psf_model.data.ndim
    except AttributeError:
        model_ndim = None

    try:
        model_bbox = psf_model.bounding_box
    except NotImplementedError:
        model_bbox = None

    if model_ndim is not None:
        if model_ndim == 3:
            model_shape = psf_model.data.shape[1:]
        elif model_ndim == 2:
            model_shape = psf_model.data.shape

        try:
            oversampling = psf_model.oversampling
        except AttributeError:
            oversampling = 1
        oversampling = as_pair('oversampling', oversampling)

        model_shape = tuple(np.array(model_shape) // oversampling)

        if np.any(psf_shape > model_shape):
            psf_shape = tuple(np.min([model_shape, psf_shape], axis=0))
            warnings.warn('The input psf_shape is larger than the size of the '
                          'evaluated PSF model (including oversampling). The '
                          f'psf_shape was changed to {psf_shape!r}.',
                          AstropyUserWarning)

    elif model_bbox is not None:
        ixmin = math.floor(model_bbox['x'].lower + 0.5)
        ixmax = math.ceil(model_bbox['x'].upper + 0.5)
        iymin = math.floor(model_bbox['y'].lower + 0.5)
        iymax = math.ceil(model_bbox['y'].upper + 0.5)
        model_shape = (iymax - iymin, ixmax - ixmin)

        if np.any(psf_shape > model_shape):
            psf_shape = tuple(np.min([model_shape, psf_shape], axis=0))
            warnings.warn('The input psf_shape is larger than the bounding '
                          'box size of the PSF model. The psf_shape was '
                          f'changed to {psf_shape!r}.', AstropyUserWarning)

    return psf_shape


@deprecated('1.13.0', alternative='make_psf_model_image')
def make_test_psf_data(shape, psf_model, psf_shape, nsources, *,
                       flux_range=(100, 1000), min_separation=1, seed=0,
                       border_size=None, progress_bar=False):
    """
    Make an example image containing PSF model images.

    Source positions and fluxes are randomly generated using an optional
    ``seed``.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model.

    psf_shape : 2-tuple of int
        The shape around the center of the star that will used to
        evaluate the ``psf_model``.

    nsources : int
        The number of sources to generate.

    flux_range : tuple, optional
        The lower and upper bounds of the flux range.

    min_separation : float, optional
        The minimum separation between the centers of two sources. Note
        that if the minimum separation is too large, the number of
        sources generated may be less than ``nsources``.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    border_size : tuple of 2 int, optional
        The (ny, nx) size of the border around the image where no
        sources will be generated (i.e., the source center will not be
        located within the border). If `None`, then a border size equal
        to half the (y, x) size of the evaluated PSF model (i.e., taking
        into account oversampling) will be used.

    progress_bar : bool, optional
        Whether to display a progress bar when creating the sources. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed. Note that the progress
        bar does not currently work in the Jupyter console due to
        limitations in ``tqdm``.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The simulated image.

    table : `~astropy.table.Table`
        A table containing the parameters of the generated sources.
    """
    psf_shape = _define_psf_shape(psf_model, psf_shape)

    if border_size is None:
        hshape = (np.array(psf_shape) - 1) // 2
    else:
        hshape = border_size
    xrange = (hshape[1], shape[1] - hshape[1])
    yrange = (hshape[0], shape[0] - hshape[0])

    xycoords = make_random_xycoords(nsources, xrange, yrange,
                                    min_separation=min_separation,
                                    seed=seed)
    x, y = np.transpose(xycoords)

    rng = np.random.default_rng(seed)
    flux = rng.uniform(flux_range[0], flux_range[1], nsources)
    flux = flux[:len(x)]

    sources = QTable()
    sources['x_0'] = x
    sources['y_0'] = y
    sources['flux'] = flux

    sources_iter = sources
    if progress_bar:  # pragma: no cover
        desc = 'Adding sources'
        sources_iter = add_progress_bar(sources, desc=desc)

    data = np.zeros(shape, dtype=float)
    for source in sources_iter:
        for param in ('x_0', 'y_0', 'flux'):
            setattr(psf_model, param, source[param])
        xcen = source['x_0']
        ycen = source['y_0']
        slc_lg, _ = overlap_slices(shape, psf_shape, (ycen, xcen), mode='trim')
        yy, xx = np.mgrid[slc_lg]
        data[slc_lg] += psf_model(xx, yy)

    sources.rename_column('x_0', 'x')
    sources.rename_column('y_0', 'y')
    sources.rename_column('flux', 'flux')

    return data, sources
