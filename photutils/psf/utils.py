# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utilities for PSF-fitting photometry.
"""

import re

import numpy as np
from astropy.modeling import CompoundModel, Model
from astropy.modeling.models import Const2D, Identity, Shift
from astropy.nddata import NDData

from photutils.datasets import make_model_image, make_model_params
from photutils.utils._parameters import as_pair

__all__ = ['make_psf_model', 'grid_from_epsfs', 'make_psf_model_image']

__doctest_requires__ = {('make_psf_model', 'make_psf_model_image'): ['scipy']}


def _interpolate_missing_data(data, mask, method='cubic'):
    """
    Interpolate missing data as identified by the ``mask`` keyword.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        An array containing the 2D image.

    mask : 2D bool `~numpy.ndarray`
        A 2D boolean mask array with the same shape as the input
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked. The masked data points are those
        that will be interpolated.

    method : {'cubic', 'nearest'}, optional
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
            splines. This is the default.

        * ``'nearest'``:  Masked data are interpolated using
            nearest-neighbor interpolation.

    Returns
    -------
    data_interp : 2D `~numpy.ndarray`
        The interpolated 2D image.
    """
    from scipy import interpolate

    data_interp = np.array(data, copy=True)

    if len(data_interp.shape) != 2:
        raise ValueError("'data' must be a 2D array.")

    if mask.shape != data.shape:
        raise ValueError("'mask' and 'data' must have the same shape.")

    y, x = np.indices(data_interp.shape)
    xy = np.dstack((x[~mask].ravel(), y[~mask].ravel()))[0]
    z = data_interp[~mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)
    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)
    else:
        raise ValueError('Unsupported interpolation method.')

    xy_missing = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    data_interp[mask] = interpol(xy_missing)

    return data_interp


class _InverseShift(Shift):
    """
    A model that is the inverse of the normal
    `astropy.modeling.functional_models.Shift` model.
    """

    @staticmethod
    def evaluate(x, offset):
        return x - offset

    @staticmethod
    def fit_deriv(x, *params):
        """
        One dimensional Shift model derivative with respect to parameter.
        """
        d_offset = -np.ones_like(x)
        return [d_offset]


def _integrate_model(model, x_name=None, y_name=None, dx=50, dy=50,
                     subsample=100, use_dblquad=False):
    """
    Integrate a model over a 2D grid.

    By default, the model is discretized on a grid of size ``dx``
    x ``dy`` from the model center with a subsampling factor of
    ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        The Astropy 2D model.

    x_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        x-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    y_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    dx, dy : odd int, optional
        The size of the integration grid in x and y. Must be odd.
        These keywords are ignored if ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis.
        Each pixel will be sampled ``subsample`` x ``subsample`` times.
        This keyword is ignored if ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model. This is *much* slower than the default integration of
        the evaluated model, but it is more accurate.

    Returns
    -------
    integral : float
        The integral of the model over the 2D grid.
    """
    if use_dblquad:
        from scipy.integrate import dblquad

        return dblquad(model, -np.inf, np.inf, -np.inf, np.inf)[0]

    from scipy.integrate import trapezoid

    if dx <= 0 or dy <= 0:
        raise ValueError('dx and dy must be > 0')
    if subsample < 1:
        raise ValueError('subsample must be >= 1')

    xc = getattr(model, x_name)
    yc = getattr(model, y_name)

    if np.any(~np.isfinite((xc.value, yc.value))):
        raise ValueError('model x and y positions must be finite')

    hx = (dx - 1) / 2
    hy = (dy - 1) / 2
    nxpts = int(dx * subsample)
    nypts = int(dy * subsample)
    xvals = np.linspace(xc - hx, xc + hx, nxpts)
    yvals = np.linspace(yc - hy, yc + hy, nypts)

    # evaluate the model on the subsampled grid
    data = model(xvals.reshape(-1, 1), yvals.reshape(1, -1))

    # now integrate over the subsampled grid (first over x, then over y)
    int_func = trapezoid

    return int_func([int_func(row, xvals) for row in data], yvals)


def _shift_model_param(model, param_name, shift=2):
    if isinstance(model, CompoundModel):
        # for CompoundModel, add "shift" to the parameter suffix
        out = re.search(r'(.*)_([\d]*)$', param_name)
        new_name = out.groups()[0] + '_' + str(int(out.groups()[1]) + 2)
    else:
        # simply add the shift to the parameter name
        new_name = param_name + '_' + str(shift)

    return new_name


def make_psf_model(model, *, x_name=None, y_name=None, flux_name=None,
                   normalize=True, dx=50, dy=50, subsample=100,
                   use_dblquad=False):
    """
    Make a PSF model that can be used with the PSF photometry classes
    (`PSFPhotometry` or `IterativePSFPhotometry`) from an Astropy
    fittable 2D model.

    If the ``x_name``, ``y_name``, or ``flux_name`` keywords are input,
    this function will map those ``model`` parameter names to ``x_0``,
    ``y_0``, or ``flux``, respectively.

    If any of the ``x_name``, ``y_name``, or ``flux_name`` keywords
    are `None`, then a new parameter will be added to the model
    corresponding to the missing parameter. Any new position parameters
    will be set to a default value of 0, and any new flux parameter will
    be set to a default value of 1.

    The output PSF model will have ``x_name``, ``y_name``, and
    ``flux_name`` attributes that contain the name of the corresponding
    model parameter.

    .. note::

        This function is needed only in cases where the 2D PSF model
        does not have ``x_0``, ``y_0``, and ``flux`` parameters.

        It is *not* needed for any of the PSF models provided
        by Photutils (e.g., `~photutils.psf.GriddedPSFModel`,
        `~photutils.psf.IntegratedGaussianPRF`).

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        An Astropy fittable 2D model to use as a PSF.

    x_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the x
        center of the PSF. If `None`, the model will be assumed to be
        centered at x=0, and a new model parameter called ``xpos_0``
        will be added for the x position.

    y_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y center of the PSF. If `None`, the model will be assumed
        to be centered at y=0, and a new parameter called ``ypos_1``
        will be added for the y position.

    flux_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        total flux of a source. If `None`, a new model parameter called
        ``flux_3`` will be added for model flux.

    normalize : bool, optional
        If `True`, the input ``model`` will be integrated and rescaled
        so that its sum integrates to 1. This normalization occurs only
        once for the input ``model``. If the total flux of ``model``
        somehow depends on (x, y) position, then one will need to
        correct the fitted model fluxes for this effect.

    dx, dy : odd int, optional
        The size of the integration grid in x and y for normalization.
        Must be odd. These keywords are ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis
        for normalization. Each pixel will be sampled ``subsample`` x
        ``subsample`` times. This keyword is ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model for normalization. This is *much* slower than the default
        integration of the evaluated model, but it is more accurate.
        This keyword is ignored if ``normalize`` is `False`.

    Returns
    -------
    result : `~astropy.modeling.CompoundModel`
        A PSF model that can be used with the PSF photometry classes.
        The returned model will always be an Astropy compound model.

    Notes
    -----
    To normalize the model, by default it is discretized on a grid of
    size ``dx`` x ``dy`` from the model center with a subsampling factor
    of ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.

    Examples
    --------
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.psf import make_psf_model
    >>> model = Gaussian2D(x_stddev=2, y_stddev=2)
    >>> psf_model = make_psf_model(model, x_name='x_mean', y_name='y_mean')
    >>> print(psf_model.param_names)  # doctest: +SKIP
    ('amplitude_2', 'x_mean_2', 'y_mean_2', 'x_stddev_2', 'y_stddev_2',
     'theta_2', 'amplitude_3', 'amplitude_4')
    """
    input_model = model.copy()

    if x_name is None:
        x_model = _InverseShift(0, name='x_position')
        # "offset" is the _InverseShift parameter name;
        # the x inverse shift model is always the first submodel
        x_name = 'offset_0'
    else:
        if x_name not in input_model.param_names:
            raise ValueError(f'{x_name!r} parameter name not found in the '
                             'input model.')

        x_model = Identity(1)
        x_name = _shift_model_param(input_model, x_name, shift=2)

    if y_name is None:
        y_model = _InverseShift(0, name='y_position')
        # "offset" is the _InverseShift parameter name;
        # the y inverse shift model is always the second submodel
        y_name = 'offset_1'
    else:
        if y_name not in input_model.param_names:
            raise ValueError(f'{y_name!r} parameter name not found in the '
                             'input model.')

        y_model = Identity(1)
        y_name = _shift_model_param(input_model, y_name, shift=2)

    x_model.fittable = True
    y_model.fittable = True
    psf_model = (x_model & y_model) | input_model

    if flux_name is None:
        psf_model *= Const2D(1.0, name='flux')
        # "amplitude" is the Const2D parameter name;
        # the flux scaling is always the last component
        flux_name = psf_model.param_names[-1]
    else:
        flux_name = _shift_model_param(input_model, flux_name, shift=2)

    if normalize:
        integral = _integrate_model(psf_model, x_name=x_name, y_name=y_name,
                                    dx=dx, dy=dy, subsample=subsample,
                                    use_dblquad=use_dblquad)

        if integral == 0:
            raise ValueError('Cannot normalize the model because the '
                             'integrated flux is zero.')

        psf_model *= Const2D(1.0 / integral, name='normalization_scaling')

    # fix all the output model parameters that are not x, y, or flux
    for name in psf_model.param_names:
        psf_model.fixed[name] = name not in (x_name, y_name, flux_name)

    # final check that the x, y, and flux parameter names are in the
    # output model
    names = (x_name, y_name, flux_name)
    for name in names:
        if name not in psf_model.param_names:
            raise ValueError(f'{name!r} parameter name not found in the '
                             'output model.')

    # set the parameter names for the PSF photometry classes
    psf_model.x_name = x_name
    psf_model.y_name = y_name
    psf_model.flux_name = flux_name

    # set aliases
    psf_model.x_0 = getattr(psf_model, x_name)
    psf_model.y_0 = getattr(psf_model, y_name)
    psf_model.flux = getattr(psf_model, flux_name)

    return psf_model


def grid_from_epsfs(epsfs, grid_xypos=None, meta=None):
    """
    Create a GriddedPSFModel from a list of EPSFModels.

    Given a list of EPSFModels, this function will return a
    GriddedPSFModel. The fiducial points for each input EPSFModel can
    either be set on each individual model by setting the 'x_0' and
    'y_0' attributes, or provided as a list of tuples (``grid_xypos``).
    If a ``grid_xypos`` list is provided, it must match the length of
    input EPSFs. In either case, the fiducial points must be on a grid.

    Optionally, a ``meta`` dictionary may be provided for the output
    GriddedPSFModel. If this dictionary contains the keys 'grid_xypos',
    'oversampling', or 'fill_value', they will be overridden.

    Note: If set on the input EPSFModel (x_0, y_0), then ``origin``
    must be the same for each input EPSF. Additionally data units and
    dimensions must be for each input EPSF, and values for ``flux`` and
    ``oversampling``, and ``fill_value`` must match as well.

    Parameters
    ----------
    epsfs : list of `photutils.psf.models.EPSFModel`
        A list of EPSFModels representing the individual PSFs.
    grid_xypos : list, optional
        A list of fiducial points (x_0, y_0) for each PSF. If not
        provided, the x_0 and y_0 of each input EPSF will be considered
        the fiducial point for that PSF. Default is None.
    meta : dict, optional
        Additional metadata for the GriddedPSFModel. Note that, if
        they exist in the supplied ``meta``, any values under the keys
        ``grid_xypos`` , ``oversampling``, or ``fill_value`` will be
        overridden. Default is None.

    Returns
    -------
    GriddedPSFModel: `photutils.psf.GriddedPSFModel`
        The gridded PSF model created from the input EPSFs.
    """
    # prevent circular imports
    from photutils.psf import EPSFModel, GriddedPSFModel

    # optional, to store fiducial from input if `grid_xypos` is None
    x_0s = []
    y_0s = []
    data_arrs = []
    oversampling = None
    fill_value = None
    dat_unit = None
    origin = None
    flux = None

    # make sure, if provided, that ``grid_xypos`` is the same length as
    # ``epsfs``
    if grid_xypos is not None:
        if len(grid_xypos) != len(epsfs):
            raise ValueError('``grid_xypos`` must be the same length as '
                             '``epsfs``.')

    # loop over input once
    for i, epsf in enumerate(epsfs):

        # check input type
        if not isinstance(epsf, EPSFModel):
            raise ValueError('All input `epsfs` must be of type '
                             '`photutils.psf.models.EPSFModel`.')

        # get data array from EPSF
        data_arrs.append(epsf.data)

        if i == 0:
            oversampling = epsf.oversampling

            # same for fill value and flux, grid will have a single value
            # so it should be the same for all input, and error if not.
            fill_value = epsf.fill_value

            # check that origins are the same
            if grid_xypos is None:
                origin = epsf.origin

            flux = epsf.flux

            # if there's a unit, those should also all be the same
            try:
                dat_unit = epsf.data.unit
            except AttributeError:
                pass  # just keep as None

        else:
            if np.any(epsf.oversampling != oversampling):
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``oversampling``.')

            if epsf.fill_value != fill_value:
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``fill_value``.')

            if epsf.data.ndim != data_arrs[0].ndim:
                raise ValueError('All input EPSFModels must have data with '
                                 'the same dimensions.')

            try:
                unitt = epsf.data_unit
                if unitt != dat_unit:
                    raise ValueError('All input data must have the same unit.')
            except AttributeError as exc:
                if dat_unit is not None:
                    raise ValueError('All input data must have the same '
                                     'unit.') from exc

            if epsf.flux != flux:
                raise ValueError('All input EPSFModels must have the same '
                                 'value for ``flux``.')

        if grid_xypos is None:  # get gridxy_pos from x_0, y_0 if not provided
            x_0s.append(epsf.x_0.value)
            y_0s.append(epsf.y_0.value)

            # also check that origin is the same, if using x_0s and y_0s
            # from input
            if epsf.origin != origin:
                raise ValueError('If using ``x_0``, ``y_0`` as fiducial point,'
                                 '``origin`` must match for each input EPSF.')

    # if not supplied, use from x_0, y_0 of input EPSFs as fiducuals
    # these are checked when GriddedPSFModel is created to make sure they
    # are actually on a grid.
    if grid_xypos is None:
        grid_xypos = list(zip(x_0s, y_0s))

    data_cube = np.stack(data_arrs, axis=0)

    if meta is None:
        meta = {}
    # add required keywords to meta
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = oversampling
    meta['fill_value'] = fill_value

    data = NDData(data_cube, meta=meta)

    grid = GriddedPSFModel(data, fill_value=fill_value)

    return grid


def _validate_psf_model(psf_model):
    """
    Validate the PSF model.

    The PSF model must be a subclass of `astropy
    .modeling.Fittable2DModel`. It must also be two-dimensional and
    have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model to validate.

    Returns
    -------
    psf_model : `astropy.modeling.Model`
        The validated PSF model.

    Raises
    ------
    TypeError
        If the PSF model is not an Astropy Model subclass.

    ValueError
        If the PSF model is not two-dimensional with n_inputs=2 and
        n_outputs=1.
    """
    if not isinstance(psf_model, Model):
        raise TypeError('psf_model must be an Astropy Model subclass.')

    if psf_model.n_inputs != 2 or psf_model.n_outputs != 1:
        raise ValueError('psf_model must be two-dimensional with '
                         'n_inputs=2 and n_outputs=1.')

    return psf_model


def _get_psf_model_params(psf_model):
    """
    Get the names of the PSF model parameters corresponding to
    x, y, and flux.

    The PSF model must have parameters called 'x_0', 'y_0', and
    'flux' or it must have 'x_name', 'y_name', and 'flux_name'
    attributes (i.e., output from `make_psf_model`). Otherwise, a
    `ValueError` is raised.

    The PSF model must be a subclass of `astropy.modeling.Model`. It
    must also be two-dimensional and have a single output.

    Parameters
    ----------
    psf_model : `astropy.modeling.Model`
        The PSF model to validate.

    Returns
    -------
    model_params : tuple
        A tuple of the PSF model parameter names.
    """
    psf_model = _validate_psf_model(psf_model)

    params1 = ('x_0', 'y_0', 'flux')
    params2 = ('x_name', 'y_name', 'flux_name')
    if all(name in psf_model.param_names for name in params1):
        model_params = params1
    elif all(params := [getattr(psf_model, name, None) for name in params2]):
        model_params = tuple(params)
    else:
        msg = 'Invalid PSF model - could not find PSF parameter names.'
        raise ValueError(msg)

    return model_params


def make_psf_model_image(shape, psf_model, n_sources, *, model_shape=None,
                         min_separation=1, border_size=None, seed=0,
                         progress_bar=False, **kwargs):
    """
    Make an example image containing PSF model images.

    Source parameters are randomly generated using an optional ``seed``.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    psf_model : 2D `astropy.modeling.Model`
        The PSF model. The model must have parameters named ``x_0``,
        ``y_0``, and ``flux``, corresponding to the center (x, y)
        position and flux, or it must have 'x_name', 'y_name', and
        'flux_name' attributes that map to the x, y, and flux parameters
        (i.e., a model output from `make_psf_model`). The model must be
        two-dimensional such that it accepts 2 inputs (e.g., x and y)
        and provides 1 output.

    n_sources : int
        The number of sources to generate. If ``min_separation`` is too
        large, the number of requested sources may not fit within the
        given ``shape`` and therefore the number of sources generated
        may be less than ``n_sources``.

    model_shape : `None` or 2-tuple of int, optional
        The shape around the center (x, y) position that will used to
        evaluate the ``psf_model``. If `None`, then the shape will be
        determined from the ``psf_model`` bounding box (an error will be
        raised if the model does not have a bounding box).

    min_separation : float, optional
        The minimum separation between the centers of two sources. Note
        that if the minimum separation is too large, the number of
        sources generated may be less than ``n_sources``.

    border_size : `None`, tuple of 2 int, or int, optional
        The (ny, nx) size of the exclusion border around the image edges
        where no sources will be generated that have centers within
        the border region. If a single integer is provided, it will be
        used for both dimensions. If `None`, then a border size equal
        to half the (y, x) size of the evaluated PSF model (taking any
        oversampling into account) will be used.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    progress_bar : bool, optional
        Whether to display a progress bar when creating the sources. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed. Note that the progress
        bar does not currently work in the Jupyter console due to
        limitations in ``tqdm``.

    **kwargs
        Keyword arguments are accepted for additional model parameters.
        The values should be 2-tuples of the lower and upper bounds for
        the parameter range. The parameter values will be uniformly
        distributed between the lower and upper bounds, inclusively. If
        the parameter is not in the input ``psf_model`` parameter names,
        it will be ignored.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The simulated image.

    table : `~astropy.table.Table`
        A table containing the (x, y, flux) parameters of the generated
        sources. The column names will correspond to the names of the
        input ``psf_model`` (x, y, flux) parameter names. The table will
        also contain an ``'id'`` column with unique source IDs

    Examples
    --------
    >>> from photutils.psf import IntegratedGaussianPRF, make_psf_model_image
    >>> shape = (150, 200)
    >>> psf_model= IntegratedGaussianPRF(sigma=1.5)
    >>> n_sources = 10
    >>> data, params = make_psf_model_image(shape, psf_model, n_sources,
    ...                                     flux=(100, 250),
    ...                                     min_separation=10,
    ...                                     seed=0)
    >>> params['x_0'].info.format = '.4f'  # optional format
    >>> params['y_0'].info.format = '.4f'
    >>> params['flux'].info.format = '.4f'
    >>> print(params)  # doctest: +FLOAT_CMP
     id   x_0      y_0      flux
    --- -------- -------- --------
      1 125.4749  72.2784 147.9522
      2  57.1803  38.6027 128.1262
      3  14.6211 116.0558 200.8790
      4  10.0741 132.6001 129.2661
      5 158.2683  43.1937 186.6532
      6 176.7725  80.2951 190.3359
      7 142.6864 133.6184 244.3635
      8 108.1142  12.5095 110.8398
      9 180.9235 106.5528 174.9959
     10 158.7488  90.5548 211.6146

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf import IntegratedGaussianPRF, make_psf_model_image
        shape = (150, 200)
        psf_model= IntegratedGaussianPRF(sigma=1.5)
        n_sources = 10
        data, params = make_psf_model_image(shape, psf_model, n_sources,
                                            flux=(100, 250),
                                            min_separation=10,
                                            seed=0)
        plt.imshow(data, origin='lower')

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf import IntegratedGaussianPRF, make_psf_model_image
        shape = (150, 200)
        psf_model= IntegratedGaussianPRF(sigma=1.5)
        n_sources = 10
        data, params = make_psf_model_image(shape, psf_model, n_sources,
                                            flux=(100, 250),
                                            min_separation=10,
                                            seed=0, sigma=(1, 2))
        plt.imshow(data, origin='lower')
    """
    psf_params = _get_psf_model_params(psf_model)

    if model_shape is not None:
        model_shape = as_pair('model_shape', model_shape, lower_bound=(0, 1))
    else:
        try:
            bbox = psf_model.bounding_box.bounding_box()
            model_shape = (int(np.round(bbox[0][1] - bbox[0][0])),
                           int(np.round(bbox[1][1] - bbox[1][0])))

        except NotImplementedError:
            raise ValueError('model_shape must be specified if the model '
                             'does not have a bounding_box attribute')

    if border_size is None:
        border_size = (np.array(model_shape) - 1) // 2

    other_params = {}
    if kwargs:
        # include only kwargs that are not x, y, or flux
        for key, val in kwargs.items():
            if key not in psf_model.param_names or key in psf_params[0:2]:
                continue  # skip the x, y parameters
            other_params[key] = val

    x_name, y_name = psf_params[0:2]
    params = make_model_params(shape, n_sources, x_name=x_name, y_name=y_name,
                               min_separation=min_separation,
                               border_size=border_size, seed=seed,
                               **other_params)

    data = make_model_image(shape, psf_model, params, model_shape=model_shape,
                            x_name=x_name, y_name=y_name,
                            progress_bar=progress_bar)

    return data, params
