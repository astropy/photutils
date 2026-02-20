# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Private utility functions for centroiding.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning


def _validate_data(data, ndim=2):
    """
    Validate and convert the input data array.

    Parameters
    ----------
    data : array_like
        The input data array.

    ndim : int or `None`, optional
        The required number of dimensions. If `None`, no dimensionality
        check is performed.

    Returns
    -------
    data : `~numpy.ndarray`
        The input data converted to a float array.
    """
    data = np.asanyarray(data, dtype=float)
    if ndim is not None and data.ndim != ndim:
        msg = f'data must be a {ndim}D array'
        raise ValueError(msg)
    return data


def _validate_mask_shape(data, mask):
    """
    Validate that the data and mask have the same shape.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The input data array.

    mask : bool `~numpy.ndarray` or `None`
        The input mask array.
    """
    if mask is not None and data.shape != mask.shape:
        msg = 'data and mask must have the same shape'
        raise ValueError(msg)


def _process_data_mask(data, mask, ndim=2, fill_value=np.nan):
    """
    Process the input data and mask.

    This function validates the input data and mask, handles non-finite
    values, and returns the processed data. The input ``mask`` is never
    mutated; a new array is created if the mask needs to be combined
    with a `~numpy.ma.MaskedArray` mask. Copies of ``data`` are made
    only when modifications are required.

    Parameters
    ----------
    data : array_like
        The input data array.

    mask : bool `~numpy.ndarray` or `None`
        A boolean mask where `True` indicates a masked (invalid) pixel.

    ndim : int or `None`, optional
        The required number of dimensions for ``data``. If `None`, no
        dimensionality check is performed.

    fill_value : float, optional
        The value used to replace masked or non-finite data values.

    Returns
    -------
    data : `~numpy.ndarray`
        Processed data with masked and non-finite values replaced by
        ``fill_value``. Always returned as a plain `~numpy.ndarray`
        (never a `~numpy.ma.MaskedArray`).
    """
    data = _validate_data(data, ndim=ndim)
    is_copied = False
    _validate_mask_shape(data, mask)

    badmask = ~np.isfinite(data)

    if np.ma.is_masked(data):
        mask2 = data.mask
        mask = mask2 if mask is None else mask | mask2

    if mask is not None:
        if np.any(mask):
            data = data.copy()
            is_copied = True
            data[mask] = fill_value
        badmask &= ~mask

    if np.any(badmask):
        msg = ('Input data contains non-finite values (e.g., NaN or inf) '
               'that were automatically masked.')
        warnings.warn(msg, AstropyUserWarning)
        if not is_copied:
            data = data.copy()
        data[badmask] = fill_value

    # If the input was a MaskedArray, return a plain ndarray; the mask
    # has already been applied to the data above.
    if isinstance(data, np.ma.MaskedArray):
        data = np.asarray(data)

    return data


def _validate_gaussian_inputs(data, mask, error):
    """
    Process and validate the data, mask, and optional error inputs for
    Gaussian centroid functions.

    The input ``mask`` and ``error`` arrays are not mutated; copies are
    made only when modifications are needed.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The input data array.

    mask : 2D bool `~numpy.ndarray` or `None`
        A boolean mask where `True` indicates a masked (invalid) pixel.

    error : 2D `~numpy.ndarray` or `None`
        The 1-sigma error array.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        Processed data with all invalid pixels (from the mask,
        non-finite data, and non-finite error) set to zero.

    combined_mask : 2D bool `~numpy.ndarray`
        Boolean mask of all invalid pixels.

    error : 2D `~numpy.ndarray` or `None`
        Error array with all invalid pixels set to zero (copied only
        if a modification was required), or `None` if no error was
        provided.
    """
    data = _process_data_mask(data, mask, fill_value=np.nan)
    combined_mask = ~np.isfinite(data)

    if error is not None:
        error = np.asanyarray(error, dtype=float)
        if data.shape != error.shape:
            msg = 'data and error must have the same shape'
            raise ValueError(msg)

        error_mask = ~np.isfinite(error)
        if np.any(error_mask):
            combined_mask |= error_mask

        # Zero error at all invalid pixel positions; copy only if needed
        if np.any(combined_mask):
            error = error.copy()
            error[combined_mask] = 0.0

    # Apply the full combined mask to data once
    if np.any(combined_mask):
        data[combined_mask] = 0.0

    return data, combined_mask, error


def _gaussian1d_moments(data, *, mask=None):
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
    data = _process_data_mask(data, mask, ndim=1, fill_value=0.0)

    x = np.arange(data.size)
    x_mean = np.sum(x * data) / np.sum(data)
    x_stddev = np.sqrt(abs(np.sum(data * (x - x_mean) ** 2) / np.sum(data)))
    amplitude = np.ptp(data)

    return amplitude, x_mean, x_stddev


def _gaussian2d_moments(data):
    """
    Estimate 2D Gaussian parameters from the moments of 2D data.

    This function can be useful for providing initial parameter
    values when fitting a 2D Gaussian to the ``data``.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. Values at masked pixels must already be set
        to zero before calling this function.

    Returns
    -------
    x_mean, y_mean, x_stddev, y_stddev, theta : float
        The estimated centroid (``x_mean``, ``y_mean``), axis standard
        deviations (``x_stddev``, ``y_stddev``), and rotation angle in
        radians (``theta``) of a 2D Gaussian.
    """
    y, x = np.indices(data.shape)
    total = np.sum(data)

    # 1st-order moments (centroid)
    x_mean = np.sum(x * data) / total
    y_mean = np.sum(y * data) / total

    # 2nd-order central moments
    dx = x - x_mean
    dy = y - y_mean
    mu_20 = np.sum(dx**2 * data) / total
    mu_02 = np.sum(dy**2 * data) / total
    mu_11 = np.sum(dx * dy * data) / total

    # Covariance matrix
    covar = np.array([[mu_02, mu_11], [mu_11, mu_20]])

    # Eigenvalues in descending order give semimajor/semiminor sigma^2
    eigvals = np.linalg.eigvalsh(covar)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    x_stddev = np.sqrt(eigvals[0])
    y_stddev = np.sqrt(eigvals[1])

    # Orientation angle (radians) between x-axis and the major axis.
    # When the distribution is nearly isotropic (mu_20 ~ mu_02, mu_11 ~
    # 0), the angle is undefined; guard against floating-point noise by
    # returning theta = 0 in that case.
    anisotropy = np.sqrt((mu_20 - mu_02) ** 2 + (2.0 * mu_11) ** 2)
    if anisotropy < 1e-6 * (mu_20 + mu_02):
        theta = 0.0
    else:
        theta = 0.5 * np.arctan2(2.0 * mu_11, mu_20 - mu_02)

    return x_mean, y_mean, x_stddev, y_stddev, theta
