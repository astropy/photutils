# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for centroiding sources and measuring their morphological
properties.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import numpy as np
from astropy.modeling.models import Gaussian1D, Gaussian2D, Const1D, Const2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import overlap_slices
from .segmentation import SourceProperties
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['GaussianConst2D', 'centroid_com', 'gaussian1d_moments',
           'marginalize_data2d', 'centroid_1dg', 'centroid_2dg',
           'fit_2dgaussian', 'data_properties', 'cutout_footprint']


class _GaussianConst1D(Const1D + Gaussian1D):
    """A 1D Gaussian plus a constant model."""


class GaussianConst2D(Const2D + Gaussian2D):
    """
    A 2D Gaussian plus a constant model.

    Parameters
    ----------
    amplitude_0 : float
        Value of the constant.
    amplitude_1 : float
        Amplitude of the Gaussian.
    x_mean_1 : float
        Mean of the Gaussian in x.
    y_mean_1 : float
        Mean of the Gaussian in y.
    x_stddev_1 : float
        Standard deviation of the Gaussian in x.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    y_stddev_1 : float
        Standard deviation of the Gaussian in y.
        ``x_stddev`` and ``y_stddev`` must be specified unless a covariance
        matrix (``cov_matrix``) is input.
    theta_1 : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.
    cov_matrix_1 : ndarray, optional
        A 2x2 covariance matrix. If specified, overrides the ``x_stddev``,
        ``y_stddev``, and ``theta`` specification.
    """


def _convert_image(data, mask=None):
    """
    Convert the input data to a float64 (double) `numpy.ndarray`,
    required for input to `skimage.measure.moments` and
    `skimage.measure.moments_central`.

    The input ``data`` is copied unless it already has that
    `numpy.dtype`.

    If ``mask`` is input, then masked pixels are set to zero in the
    output ``data``.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are set to zero in the output ``data``.

    Returns
    -------
    image : `numpy.ndarray`, float64
        The converted 2D array of the image, where masked pixels have
        been set to zero.
    """
    if mask is None:
        copy = False
    else:
        copy = True
    image = np.asarray(data).astype(np.float, copy=copy)
    if mask is not None:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        image[mask] = 0.0
    return image


def centroid_com(data, mask=None):
    """
    Calculate the centroid of a 2D array as its center of mass
    determined from image moments.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    from skimage.measure import moments
    data = _convert_image(data, mask=mask)
    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    return xcen, ycen


def gaussian1d_moments(data, mask=None):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.

    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : array_like (1D)
        The 1D array.

    mask : array_like (1D bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """

    if mask is not None:
        mask = np.asanyarray(mask)
        data = data.copy()
        data[mask] = 0.
    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean)**2) / np.sum(data)))
    amplitude = np.nanmax(data) - np.nanmin(data)
    return amplitude, x_mean, x_stddev


def marginalize_data2d(data, error=None, mask=None):
    """
    Generate the marginal x and y distributions from a 2D data array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    marginal_data : list of `~numpy.ndarray`
        The marginal x and y distributions of the input ``data``.

    marginal_error : list of `~numpy.ndarray`
        The marginal x and y distributions of the input ``error``.

    marginal_mask : list of `~numpy.ndarray` (bool)
        The marginal x and y distributions of the input ``mask``.
    """

    if error is not None:
        marginal_error = np.array(
            [np.sqrt(np.sum(error**2, axis=i)) for i in [0, 1]])
    else:
        marginal_error = [None, None]

    if mask is not None:
        mask = np.asanyarray(mask)
        marginal_mask = [np.sum(mask, axis=i).astype(np.bool) for i in [0, 1]]
    else:
        marginal_mask = [None, None]

    marginal_data = [np.sum(data, axis=i) for i in [0, 1]]

    return marginal_data, marginal_error, marginal_mask


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal x and y distributions of the array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    mdata, merror, mmask = marginalize_data2d(data, error=error, mask=mask)

    if merror[0] is None and mmask[0] is None:
        mweights = [None, None]
    else:
        if merror[0] is not None:
            mweights = [(1.0 / merror[i].clip(min=1.e-30)) for i in [0, 1]]
        else:
            mweights = np.array([np.ones(data.shape[1]),
                                 np.ones(data.shape[0])])
        # down-weight masked pixels
        for i in [0, 1]:
            mweights[i][mmask[i]] = 1.e-20

    const_init = np.min(data)
    centroid = []
    for (mdata_i, mweights_i, mmask_i) in zip(mdata, mweights, mmask):
        params_init = gaussian1d_moments(mdata_i, mask=mmask_i)
        g_init = _GaussianConst1D(const_init, *params_init)
        fitter = LevMarLSQFitter()
        x = np.arange(mdata_i.size)
        g_fit = fitter(g_init, x, mdata_i, weights=mweights_i)
        centroid.append(g_fit.mean_1.value)
    return tuple(centroid)


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian (plus
    a constant) to the array.

    Parameters
    ----------
    data : array_like
        The 2D data array.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    gfit = fit_2dgaussian(data, error=error, mask=mask)
    return gfit.x_mean_1.value, gfit.y_mean_1.value


def fit_2dgaussian(data, error=None, mask=None):
    """
    Fit a 2D Gaussian plus a constant to a 2D image.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    result : A `GaussianConst2D` model instance.
        The best-fitting Gaussian 2D model.
    """

    if data.size < 7:
        warnings.warn('data array must have a least 7 values to fit a 2D '
                      'Gaussian plus a constant', AstropyUserWarning)
        return None

    if error is not None:
        weights = 1.0 / error
    else:
        weights = None

    if mask is not None:
        mask = np.asanyarray(mask)
        if weights is None:
            weights = np.ones_like(data)
        # down-weight masked pixels
        weights[mask] = 1.e-30

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    shift = np.min(data)
    data = np.copy(data) - shift
    props = data_properties(data, mask=mask)
    init_values = np.array([props.xcentroid.value, props.ycentroid.value,
                            props.semimajor_axis_sigma.value,
                            props.semiminor_axis_sigma.value,
                            props.orientation.value])

    init_const = 0.    # subtracted data minimum above
    init_amplitude = np.nanmax(data) - np.nanmin(data)
    g_init = GaussianConst2D(init_const, init_amplitude, *init_values)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)
    gfit.amplitude_0 = gfit.amplitude_0 + shift
    return gfit


def data_properties(data, mask=None, background=None):
    """
    Calculate the centroid and morphological properties of a 2D array,
    e.g., an image cutout of an object.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was previously present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.

    Returns
    -------
    result : `~photutils.segmentation.SourceProperties` instance
        A `~photutils.segmentation.SourceProperties` object.
    """

    segment_image = np.ones(data.shape, dtype=np.int)
    return SourceProperties(data, segment_image, label=1, mask=mask,
                            background=background)


def cutout_footprint(data, position, box_size=3, footprint=None, mask=None,
                     error=None):
    """
    Cut out a region from data (and optional mask and error) centered at
    specified (x, y) position.

    The size of the region is specified via the ``box_size`` or
    ``footprint`` keywords.  The output mask for the cutout region
    represents the combination of the input mask and footprint mask.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    position : 2 tuple
        The ``(x, y)`` pixel coordinate of the center of the region.

    box_size : scalar or tuple, optional
        The size of the region to cutout from ``data``.  If ``box_size``
        is a scalar, then the region shape will be ``(box_size,
        box_size)``.  Either ``box_size`` or ``footprint`` must be
        defined.  If they are both defined, then ``footprint`` overrides
        ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local footprint
        region.  ``box_size=(n, m)`` is equivalent to
        ``footprint=np.ones((n, m))``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    Returns
    -------
    region_data : `~numpy.ndarray`
        The ``data`` cutout.

    region_mask : `~numpy.ndarray`
        The ``mask`` cutout.

    region_error : `~numpy.ndarray`
        The ``error`` cutout.

    slices : tuple of slices
        Slices in each dimension of the ``data`` array used to define
        the cutout region.
    """

    if len(position) != 2:
        raise ValueError('position must have a length of 2')

    if footprint is None:
        if box_size is None:
            raise ValueError('box_size or footprint must be defined.')
        if not isinstance(box_size, collections.Iterable):
            shape = (box_size, box_size)
        else:
            if len(box_size) != 2:
                raise ValueError('box_size must have a length of 2')
            shape = box_size
        footprint = np.ones(shape, dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)

    slices_large, slices_small = overlap_slices(data.shape, footprint.shape,
                                                position[::-1])
    region_data = data[slices_large]

    if error is not None:
        region_error = error[slices_large]
    else:
        region_error = None

    if mask is not None:
        region_mask = mask[slices_large]
    else:
        region_mask = np.zeros_like(region_data, dtype=bool)
    footprint_mask = ~footprint
    footprint_mask = footprint_mask[slices_small]    # trim if necessary
    region_mask = np.logical_or(region_mask, footprint_mask)

    return region_data, region_mask, region_error, slices_large
