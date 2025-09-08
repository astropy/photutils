# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for centroiding sources using Gaussians.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._quantity_helpers import process_quantities

__all__ = ['centroid_1dg', 'centroid_2dg']


def centroid_1dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal ``x`` and ``y`` distributions of the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.centroids import centroid_1dg
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> x1, y1 = centroid_1dg(data)
    >>> print(np.array((x1, y1)))
    [19.96553246 20.04952841]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_1dg
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        data = data[40:80, 70:110]
        xycen = centroid_1dg(data)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(data, origin='lower', interpolation='nearest')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    (data, error), _ = process_quantities((data, error), ('data', 'error'))

    data = np.ma.asanyarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            msg = 'data and mask must have the same shape'
            raise ValueError(msg)
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            msg = 'data and error must have the same shape'
            raise ValueError(msg)
        data.mask |= error.mask
        error.mask = data.mask

        xy_error = [np.sqrt(np.ma.sum(error**2, axis=i)) for i in (0, 1)]
        xy_weights = [(1.0 / xy_error[i].clip(min=1.0e-30)) for i in (0, 1)]
    else:
        xy_weights = [np.ones(data.shape[i]) for i in (1, 0)]

    # assign zero weight where an entire row or column is masked
    if np.any(data.mask):
        bad_idx = [np.all(data.mask, axis=i) for i in (0, 1)]
        for i in (0, 1):
            xy_weights[i][bad_idx[i]] = 0.0

    xy_data = [np.ma.sum(data, axis=i).data for i in (0, 1)]

    # Gaussian1D stddev is bounded to be strictly positive
    fitter = TRFLSQFitter()

    centroid = []
    for (data_i, weights_i) in zip(xy_data, xy_weights, strict=True):
        params_init = _gaussian1d_moments(data_i)
        g_init = Gaussian1D(*params_init)
        x = np.arange(data_i.size)
        g_fit = fitter(g_init, x, data_i, weights=weights_i)
        centroid.append(g_fit.mean.value)

    return np.array(centroid)


def _gaussian1d_moments(data, mask=None):
    """
    Estimate 1D Gaussian parameters from the moments of 1D data.

    This function can be useful for providing initial parameter values
    when fitting a 1D Gaussian to the ``data``.

    Parameters
    ----------
    data : 1D `~numpy.ndarray`
        The 1D data array.

    mask : 1D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    amplitude, mean, stddev : float
        The estimated parameters of a 1D Gaussian.
    """
    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)
    else:
        data = np.ma.array(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            msg = 'data and mask must have the same shape'
            raise ValueError(msg)
        data.mask |= mask

    data.fill_value = 0.0
    data = data.filled()

    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean) ** 2) / np.sum(data)))
    amplitude = np.ptp(data)

    return amplitude, x_mean, x_stddev


def centroid_2dg(data, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian to the
    array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.centroids import centroid_2dg
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> x1, y1 = centroid_2dg(data)
    >>> print(np.array((x1, y1)))
    [19.9851944  20.01490157]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        data = data[40:80, 70:110]
        xycen = centroid_2dg(data)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(data, origin='lower', interpolation='nearest')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    # prevent circular import
    from photutils.morphology import data_properties

    (data, error), _ = process_quantities((data, error), ('data', 'error'))

    data = np.ma.asanyarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            msg = 'data and mask must have the same shape'
            raise ValueError(msg)
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            msg = 'data and error must have the same shape'
            raise ValueError(msg)
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.0e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 6:
        msg = ('Input data must have a least 6 unmasked values to fit a '
               '2D Gaussian.')
        raise ValueError(msg)

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.0

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data to make the data values positive.
    # This prevents issues with the moment estimation in data_properties.
    # Moments from negative data values can yield undefined Gaussian
    # parameters, e.g., x/y_stddev.
    props = data_properties(data - np.min(data), mask=mask)

    g_init = Gaussian2D(amplitude=np.ptp(data),
                        x_mean=props.xcentroid,
                        y_mean=props.ycentroid,
                        x_stddev=props.semimajor_sigma.value,
                        y_stddev=props.semiminor_sigma.value,
                        theta=props.orientation)

    # Gaussian2D [x/y]_stddev are bounded to be strictly positive
    fitter = TRFLSQFitter()

    y, x = np.indices(data.shape)

    with warnings.catch_warnings(record=True) as fit_warnings:
        gfit = fitter(g_init, x, y, data, weights=weights)

    if len(fit_warnings) > 0:
        warnings.warn('The fit may not have converged. Please check your '
                      'results.', AstropyUserWarning)

    return np.array([gfit.x_mean.value, gfit.y_mean.value])
