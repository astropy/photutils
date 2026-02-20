# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for centroiding sources using Gaussians.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning

from photutils.centroids._utils import (_gaussian1d_moments,
                                        _gaussian2d_moments,
                                        _validate_gaussian_inputs)
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['centroid_1dg', 'centroid_2dg']


def centroid_1dg(data, *, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting 1D Gaussians to the
    marginal ``x`` and ``y`` distributions of the array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D array_like
        The 2D image data. ``data`` can be a `~numpy.ma.MaskedArray`.
        The image should be a background-subtracted cutout image
        containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        If ``data`` is a `~numpy.ma.MaskedArray`, its mask will be
        combined (using bitwise OR) with the input ``mask``.

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
    data, mask, error = _validate_gaussian_inputs(data, mask, error)

    if error is not None:
        error_squared = error**2
        xy_error = [np.sqrt(np.sum(error_squared, axis=i)) for i in (0, 1)]
        xy_weights = [1.0 / xy_err.clip(min=1.0e-30) for xy_err in xy_error]
    else:
        xy_weights = [np.ones(data.shape[i]) for i in (1, 0)]

    # Assign zero weight where an entire row or column is masked
    if np.any(mask):
        bad_idx = [np.all(mask, axis=i) for i in (0, 1)]
        for i in (0, 1):
            xy_weights[i][bad_idx[i]] = 0.0

    xy_data = [np.sum(data, axis=i) for i in (0, 1)]

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


def centroid_2dg(data, *, error=None, mask=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D Gaussian to the
    array.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : 2D array_like
        The 2D image data. ``data`` can be a `~numpy.ma.MaskedArray`.
        The image should be a background-subtracted cutout image
        containing a single source.

    error : 2D `~numpy.ndarray`, optional
        The 2D array of the 1-sigma errors of the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        If ``data`` is a `~numpy.ma.MaskedArray`, its mask will be
        combined (using bitwise OR) with the input ``mask``.

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
    (data, error), _ = process_quantities((data, error), ('data', 'error'))
    data, mask, error = _validate_gaussian_inputs(data, mask, error)

    if np.count_nonzero(~mask) < 6:
        msg = ('Input data must have a least 6 unmasked values to fit a '
               '2D Gaussian.')
        raise ValueError(msg)

    if error is not None:
        weights = 1.0 / error.clip(min=1.0e-30)
    else:
        weights = np.ones(data.shape)

    # Assign zero weight to masked pixels
    if np.any(mask):
        weights[mask] = 0.0

    # Subtract the minimum of the data to make the data values positive.
    # Moments from negative data values can yield undefined Gaussian
    # parameters, e.g., x_stddev and y_stddev.
    amplitude, x_mean, y_mean, x_stddev, y_stddev, theta = _gaussian2d_moments(
        data - np.min(data))

    g_init = Gaussian2D(amplitude=amplitude,
                        x_mean=x_mean,
                        y_mean=y_mean,
                        x_stddev=x_stddev,
                        y_stddev=y_stddev,
                        theta=theta)
    fitter = TRFLSQFitter()

    y, x = np.indices(data.shape)

    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter('always', AstropyUserWarning)
        gfit = fitter(g_init, x, y, data, weights=weights)

    if any(issubclass(w.category, AstropyUserWarning) for w in fit_warnings):
        msg = 'The fit may not have converged. Please check your results.'
        warnings.warn(msg, AstropyUserWarning)

    return np.array([gfit.x_mean.value, gfit.y_mean.value])
