# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making PSF models.
"""

import contextlib
import re

import numpy as np
from astropy.modeling import CompoundModel
from astropy.modeling.models import Const2D, Identity, Shift
from astropy.nddata import NDData
from astropy.units import Quantity
from astropy.utils.decorators import deprecated
from scipy.integrate import dblquad, trapezoid
from scipy.interpolate import make_interp_spline

from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._parameters import as_pair
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['grid_from_epsfs', 'make_epsf_from_psf', 'make_psf_model']


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

        It is *not* needed for any of the PSF models provided by
        Photutils.

    The output PSF model is evaluated at the input positions like
    the input ``model``. It is not integrated over the detector pixels
    (see the Notes section).

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        An Astropy fittable 2D model to use as a PSF.

    x_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the x
        center of the PSF. If `None`, the model will be assumed to be
        centered at x=0, and a new model parameter called ``offset_0``
        will be added for the x position.

    y_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the y
        center of the PSF. If `None`, the model will be assumed to be
        centered at y=0, and a new parameter called ``offset_1`` will be
        added for the y position.

    flux_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        total flux of a source. If `None`, a new model parameter called
        ``amplitude_3`` will be added for model flux.

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

    That integration is used only to normalize the model. The values
    of the output model on a grid of detector pixels are the values of
    the input ``model`` at the pixel centers, not the fluxes in the
    pixels. For a PSF that is undersampled by the detector pixels, such
    a model is sharper than a source in the data and the sum of its
    values over the pixels depends on the subpixel position of the
    source. To integrate a model over the pixels, evaluate it on an
    oversampled grid, make an ePSF image from the result with
    `make_epsf_from_psf`, and use that image with `ImagePSF`.

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
        # The _InverseShift model is always the first submodel, so the x
        # position parameter name is always "offset_0".
        x_name = 'offset_0'
    else:
        if x_name not in input_model.param_names:
            msg = f'{x_name!r} parameter name not found in the input model'
            raise ValueError(msg)

        x_model = Identity(1)
        x_name = _shift_model_param(input_model, x_name, shift=2)

    if y_name is None:
        y_model = _InverseShift(0, name='y_position')
        # The _InverseShift model is always the second submodel, so the y
        # position parameter name is always "offset_1".
        y_name = 'offset_1'
    else:
        if y_name not in input_model.param_names:
            msg = f'{y_name!r} parameter name not found in the input model'
            raise ValueError(msg)

        y_model = Identity(1)
        y_name = _shift_model_param(input_model, y_name, shift=2)

    x_model.fittable = True
    y_model.fittable = True
    psf_model = (x_model & y_model) | input_model

    if flux_name is None:
        psf_model *= Const2D(1.0, name='flux')
        # The Const2D model is always the last submodel, so the flux
        # parameter name is always "amplitude_3" (or "amplitude" if the
        # input model is a CompoundModel).
        flux_name = psf_model.param_names[-1]
    else:
        if flux_name not in input_model.param_names:
            msg = f'{flux_name!r} parameter name not found in the input model'
            raise ValueError(msg)

        flux_name = _shift_model_param(input_model, flux_name, shift=2)

    if normalize:
        integral = _integrate_model(psf_model, x_name=x_name, y_name=y_name,
                                    dx=dx, dy=dy, subsample=subsample,
                                    use_dblquad=use_dblquad)

        if integral == 0:
            msg = ('Cannot normalize the model because the integrated flux '
                   'is zero')
            raise ValueError(msg)

        psf_model *= Const2D(1.0 / integral, name='normalization_scaling')

    # Set all the other parameters to be fixed so that they are not fit
    # during PSF photometry.
    for name in psf_model.param_names:
        psf_model.fixed[name] = name not in (x_name, y_name, flux_name)

    # Set the parameter names for the PSF photometry classes
    psf_model.x_name = x_name
    psf_model.y_name = y_name
    psf_model.flux_name = flux_name

    # Set aliases
    psf_model.x_0 = getattr(psf_model, x_name)
    psf_model.y_0 = getattr(psf_model, y_name)
    psf_model.flux = getattr(psf_model, flux_name)

    return psf_model


class _InverseShift(Shift):
    """
    A model that is the inverse of the normal
    `astropy.modeling.functional_models.Shift` model.
    """

    @staticmethod
    def evaluate(x, offset):
        return x - offset

    @staticmethod
    def fit_deriv(x, offset):
        """
        One dimensional Shift model derivative with respect to
        parameter.
        """
        d_offset = -np.ones_like(x) + offset * 0.0
        return [d_offset]


def _integrate_model(model, *, x_name=None, y_name=None, dx=50, dy=50,
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
        return dblquad(model, -np.inf, np.inf, -np.inf, np.inf)[0]

    if dx <= 0 or dy <= 0:
        msg = 'dx and dy must be > 0'
        raise ValueError(msg)
    if subsample < 1:
        msg = 'subsample must be >= 1'
        raise ValueError(msg)

    xc = getattr(model, x_name)
    yc = getattr(model, y_name)

    if np.any(~np.isfinite((xc.value, yc.value))):
        msg = 'model x and y positions must be finite'
        raise ValueError(msg)

    hx = (dx - 1) / 2
    hy = (dy - 1) / 2
    nx_pts = int(dx * subsample)
    ny_pts = int(dy * subsample)
    xvals = np.linspace(xc - hx, xc + hx, nx_pts)
    yvals = np.linspace(yc - hy, yc + hy, ny_pts)

    # Evaluate the model on the subsampled grid
    data = model(xvals.reshape(-1, 1), yvals.reshape(1, -1))
    if isinstance(data, Quantity):
        data = data.value

    # Now integrate over the subsampled grid (first over y because
    # each row varies over y, then over x)
    int_func = trapezoid

    return int_func([int_func(row, yvals) for row in data], xvals)


def _shift_model_param(model, param_name, *, shift=2):
    if isinstance(model, CompoundModel):
        # For CompoundModel, add "shift" to the parameter suffix
        out = re.search(r'(.*)_([\d]+)$', param_name)
        new_name = out.groups()[0] + '_' + str(int(out.groups()[1]) + shift)
    else:
        # Simply add the shift to the parameter name
        new_name = param_name + '_' + str(shift)

    return new_name


@deprecated_positional_kwargs(since='3.0', until='4.0')
@deprecated(since='3.0', alternative='`GriddedPSFModel`',
            warning_type=PhotutilsDeprecationWarning)
def grid_from_epsfs(epsfs, grid_xypos=None, meta=None):  # pragma: no cover
    """
    Create a GriddedPSFModel from a list of ImagePSF models.

    Given a list of `~photutils.psf.ImagePSF` models, this function will
    return a `~photutils.psf.GriddedPSFModel`. The fiducial points for
    each input ImagePSF can either be set on each individual model by
    setting the 'x_0' and 'y_0' attributes, or provided as a list of
    tuples (``grid_xypos``). If a ``grid_xypos`` list is provided, it
    must match the length of input EPSFs. In either case, the fiducial
    points must be on a grid.

    Optionally, a ``meta`` dictionary may be provided for the output
    GriddedPSFModel. If this dictionary contains the keys 'grid_xypos',
    'oversampling', or 'fill_value', they will be overridden.

    Note: If set on the input ImagePSF (x_0, y_0), then ``origin``
    must be the same for each input EPSF. Additionally, data units and
    dimensions must be the same for each input EPSF, and values for
    ``flux``, ``oversampling``, and ``fill_value`` must match as well.

    Parameters
    ----------
    epsfs : list of `photutils.psf.ImagePSF`
        A list of ImagePSF models representing the individual PSFs.
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
    # Prevent circular imports
    from photutils.psf import GriddedPSFModel, ImagePSF

    # Optional, to store fiducial from input if `grid_xypos` is None
    x_0s = []
    y_0s = []
    data_arrs = []
    oversampling = None
    fill_value = None
    dat_unit = None
    origin = None
    flux = None

    # Make sure that ``grid_xypos`` has the same length as epsfs
    if grid_xypos is not None and len(grid_xypos) != len(epsfs):
        msg = 'grid_xypos must be the same length as epsfs'
        raise ValueError(msg)

    # Loop over input once
    for i, epsf in enumerate(epsfs):

        # Check input type
        if not isinstance(epsf, ImagePSF):
            msg = 'All input epsfs must be of type ImagePSF'
            raise TypeError(msg)

        # Get data array from EPSF
        data_arrs.append(epsf.data)

        if i == 0:
            oversampling = epsf.oversampling

            # Same for fill value and flux, grid will have a single value
            # so it should be the same for all input, and error if not.
            fill_value = epsf.fill_value

            # Check that origins are the same
            if grid_xypos is None:
                origin = epsf.origin

            flux = epsf.flux

            # If there's a unit, those should also all be the same
            with contextlib.suppress(AttributeError):
                dat_unit = epsf.data.unit
        else:
            if np.any(epsf.oversampling != oversampling):
                msg = ('All input ImagePSF models must have the same value '
                       'for oversampling')
                raise ValueError(msg)

            if epsf.fill_value != fill_value:
                msg = ('All input ImagePSF models must have the same value '
                       'for fill_value')
                raise ValueError(msg)

            if epsf.data.ndim != data_arrs[0].ndim:
                msg = ('All input ImagePSF models must have data with the '
                       'same dimensions')
                raise ValueError(msg)

            unitt = None
            with contextlib.suppress(AttributeError):
                unitt = epsf.data.unit
            if unitt != dat_unit:
                msg = 'All input data must have the same unit'
                raise ValueError(msg)

            if epsf.flux != flux:
                msg = ('All input ImagePSF models must have the same value '
                       'for flux')
                raise ValueError(msg)

        if grid_xypos is None:  # get gridxy_pos from x_0, y_0 if not provided
            x_0s.append(epsf.x_0.value)
            y_0s.append(epsf.y_0.value)

            # Also check that origin is the same, if using x_0s and y_0s
            # from input
            if np.any(epsf.origin != origin):
                msg = ('If using (x_0, y_0) as fiducial point, origin must '
                       'match for each input EPSF')
                raise ValueError(msg)

    # If not supplied, use from x_0, y_0 of input EPSFs as fiducials.
    # These are checked when GriddedPSFModel is created to make sure
    # they are actually on a grid.
    if grid_xypos is None:
        grid_xypos = list(zip(x_0s, y_0s, strict=True))

    data_cube = np.stack(data_arrs, axis=0)

    if meta is None:
        meta = {}
    # Add required keywords to meta
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = oversampling
    meta['fill_value'] = fill_value

    data = NDData(data_cube, meta=meta)

    return GriddedPSFModel(data, fill_value=fill_value)


def _integrate_pixel_along_axis(data, half_width, axis):
    """
    Integrate the cubic spline through ``data`` over a window centered
    at each grid point along one axis.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data array.

    half_width : float
        The half width of the integration window in grid points.

    axis : int
        The axis along which to integrate.

    Returns
    -------
    result : `~numpy.ndarray`
        The integrals, with the same shape as ``data``. The window is
        truncated at the first and last grid points.
    """
    n_points = data.shape[axis]
    points = np.arange(n_points, dtype=float)
    antiderivative = make_interp_spline(points, data, k=3,
                                        axis=axis).antiderivative()
    lower = np.clip(points - half_width, 0, n_points - 1)
    upper = np.clip(points + half_width, 0, n_points - 1)
    return antiderivative(upper) - antiderivative(lower)


def make_epsf_from_psf(data, *, oversampling):
    """
    Make an effective PSF (ePSF) image from an oversampled PSF image
    that is not integrated over the detector pixels.

    The image-based PSF models (`ImagePSF` and `GriddedPSFModel`)
    require ePSF images. Each value of an ePSF is the fraction of the
    source flux that falls in a whole detector pixel centered at that
    position relative to the source. This function makes such an image
    from an oversampled PSF whose values are samples of the PSF at the
    grid points, such as the output of an optical model.

    Parameters
    ----------
    data : 2D or 3D `~numpy.ndarray`
        The oversampled PSF image. A 3D array is a stack of PSF images
        with shape ``(n_psfs, ny, nx)``, such as the data of a
        `GriddedPSFModel`. The x and y dimensions must both be at
        least 4 pixels. All values must be finite.

    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the PSF image. If a
        scalar is provided, it is applied to both axes. If two values
        are provided, they must be in ``(y, x)`` order.

    Returns
    -------
    result : `~numpy.ndarray`
        The ePSF image(s), with the same shape, grid, and oversampling
        as ``data``.

    See Also
    --------
    ImagePSF, GriddedPSFModel

    Notes
    -----
    The input image is interpolated with a bicubic spline. The spline
    is integrated exactly over the area of one detector pixel centered
    at each grid point, and the result is divided by the number of grid
    points in a detector pixel. The normalization of the input image is
    therefore preserved. An input image whose values sum to the product
    of the oversampling factors gives an ePSF with the same sum, which
    is the normalization that `ImagePSF` requires.

    The accuracy of the result is set by how well the spline through
    the input values represents the PSF. The input grid must therefore
    sample the PSF well, which generally requires an oversampled image
    for a PSF that is undersampled by the detector pixels.

    The input image is taken to be zero outside of its grid. The
    output values within half of a detector pixel of the image edges
    are therefore integrals over only the part of the pixel that is
    inside the grid.

    The input values must be samples of the PSF at the grid points.
    The result is less accurate for an image whose values are the
    fluxes in the cells of the oversampled grid, because such an image
    is already integrated over the area of one cell.

    A model made from the output image conserves flux. The model
    values on a grid of detector pixels sum to the model flux for any
    subpixel position of the source, apart from the flux that falls
    outside of the image. A model made directly from a PSF that is
    sampled at points does not have this property when the PSF is
    undersampled by the detector pixels.

    Examples
    --------
    Make an ePSF from a narrow Gaussian PSF that is sampled at the
    points of a grid that is oversampled by a factor of 5:

    >>> import numpy as np
    >>> from photutils.psf import ImagePSF, make_epsf_from_psf
    >>> oversampling = 5
    >>> yy, xx = np.mgrid[-37:38, -37:38] / oversampling
    >>> sigma = 0.42  # detector pixels
    >>> psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    >>> psf *= oversampling**2 / psf.sum()
    >>> epsf = make_epsf_from_psf(psf, oversampling=oversampling)

    The sum of the `ImagePSF` model over the detector pixels depends
    on the subpixel position of the source for the sampled PSF, but
    not for the ePSF:

    >>> yy, xx = np.mgrid[-7:8, -7:8]
    >>> for data in (psf, epsf):
    ...     model = ImagePSF(data, oversampling=oversampling)
    ...     for x_0 in (0.0, 0.5):
    ...         model.x_0 = x_0
    ...         print(f'{model(xx, yy).sum():.3f}')
    1.127
    0.997
    1.000
    1.000
    """
    data = np.asarray(data, dtype=float)
    if data.ndim not in (2, 3):
        msg = 'data must be a 2D or 3D array'
        raise ValueError(msg)
    if data.shape[-2] < 4 or data.shape[-1] < 4:
        msg = 'The x and y dimensions of data must both be at least 4'
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = 'All elements of data must be finite'
        raise ValueError(msg)
    oversampling = as_pair('oversampling', oversampling, lower_bound=(0, 0))

    result = data
    for axis, factor in zip((-2, -1), oversampling, strict=True):
        result = _integrate_pixel_along_axis(result, factor / 2, axis)
        result /= factor
    return result
