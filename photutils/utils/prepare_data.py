# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable
import copy
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['calculate_total_error', 'subtract_background',
           'interpolate_masked_data']


def calculate_total_error(data, error, effective_gain):
    """
    Calculate a total error array, combining a background error array
    with the Poisson noise of sources.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array data array.

    error : array_like or `~astropy.units.Quantity`
        The pixel-wise Gaussian 1-sigma background errors of the input
        ``data``.  ``error`` should include all sources of "background"
        error but *exclude* the Poisson error of the sources.  ``error``
        must have the same shape as ``data``.

    effective_gain : float, array-like, or `~astropy.units.Quantity`
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.

    Notes
    -----
    The total error array, :math:`\sigma_{\mathrm{tot}}` is:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                  \\frac{I}{g}}

    where :math:`\sigma_b`, :math:`I`, and :math:`g` are the background
    ``error`` image, ``data`` image, and ``effective_gain``,
    respectively.

    ``data`` here should not be background-subtracted.  ``data`` should
    include all detected counts (electrons or photons), including the
    background, to properly calculate the Poisson errors of sources.
    """

    has_unit = [hasattr(x, 'unit') for x in [data, effective_gain]]
    if any(has_unit) and not all(has_unit):
        raise ValueError('If either data or effective_gain has units, then '
                         'they both must have units.')
    if all(has_unit):
        count_units = [u.electron, u.photon]
        datagain_unit = (data * effective_gain).unit
        if datagain_unit not in count_units:
            raise u.UnitsError('(data * effective_gain) has units of "{0}", '
                               'but it must have count units (u.electron '
                               'or u.photon).'.format(datagain_unit))

    if not isiterable(effective_gain):
        # NOTE: np.broadcast_arrays() never returns a Quantity
        # effective_gain = np.broadcast_arrays(effective_gain, data)[0]
        effective_gain = np.zeros(data.shape) + effective_gain
    else:
        if effective_gain.shape != data.shape:
            raise ValueError('If input effective_gain is 2D, then it must '
                             'have the same shape as the input data.')
    if np.any(effective_gain <= 0):
        raise ValueError('effective_gain must be positive everywhere')

    if all(has_unit):
        variance_total = np.maximum(
            error**2 + ((data * data.unit) / effective_gain.value),
            0. * error.unit**2)
    else:
        variance_total = np.maximum(error**2 + (data / effective_gain), 0)
    return np.sqrt(variance_total)


def subtract_background(data, background):
    """
    Subtract background from data and generate a 2D pixel-wise
    background image.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D data array from which to subtract ``background``.

    background : float, array_like, or `~astropy.units.Quantity`
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.

    Returns
    -------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The background subtracted data.

    background : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The pixel-wise background array.
    """

    if not isiterable(background):
        # NOTE: np.broadcast_arrays() never returns a Quantity
        # background = np.broadcast_arrays(background, data)[0]
        background = np.zeros(data.shape) + background
    else:
        if background.shape != data.shape:
            raise ValueError('If input background is 2D, then it must '
                             'have the same shape as the input data.')
    return (data - background), background


def interpolate_masked_data(data, mask, error=None, background=None):
    """
    Interpolate over masked pixels in data and optional error or
    background images.

    The value of masked pixels are replaced by the mean value of the
    8-connected neighboring non-masked pixels.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D data array.

    mask : array_like (bool)
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` must have the same shape as ``data``.

    background : array_like, or `~astropy.units.Quantity`, optional
        The pixel-wise background level of the input ``data``.
        ``background`` must have the same shape as ``data``.

    Returns
    -------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``data`` with interpolated masked pixels.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``error`` with interpolated masked pixels.  `None` if
        input ``error`` is not input.

    background : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``background`` with interpolated masked pixels.  `None` if
        input ``background`` is not input.
    """

    if data.shape != mask.shape:
        raise ValueError('data and mask must have the same shape')

    data_out = copy.deepcopy(data)    # do not alter input data
    mask_idx = mask.nonzero()
    for j, i in zip(*mask_idx):
        y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
        x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
        goodpix = ~mask[y0:y1, x0:x1]
        if not np.any(goodpix):
            warnings.warn('The masked pixel at "({0}, {1})" is completely '
                          'surrounded by (8-connected) masked pixels, '
                          'thus unable to interpolate'.format(i, j),
                          AstropyUserWarning)
            continue
        data_out[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])

        if background is not None:
            if background.shape != data.shape:
                raise ValueError('background and data must have the same '
                                 'shape')
            background_out = copy.deepcopy(background)
            background_out[j, i] = np.mean(background[y0:y1, x0:x1][goodpix])
        else:
            background_out = None

        if error is not None:
            if error.shape != data.shape:
                raise ValueError('error and data must have the same '
                                 'shape')
            error_out = copy.deepcopy(error)
            error_out[j, i] = np.sqrt(
                np.mean(error[y0:y1, x0:x1][goodpix]**2))
        else:
            error_out = None

    return data_out, error_out, background_out


def _check_units(inputs):
    """Check for consistent units on data, error, and background."""
    has_unit = [hasattr(x, 'unit') for x in inputs]
    if any(has_unit) and not all(has_unit):
        raise ValueError('If any of data, error, or background has units, '
                         'then they all must all have units.')
    if all(has_unit):
        if any([inputs[0].unit != getattr(x, 'unit') for x in inputs[1:]]):
            raise u.UnitsError(
                'data, error, and background units do not match.')


def _prepare_data(data, error=None, effective_gain=None, background=None):
    """
    Prepare the data, error, and background arrays.

    If any of ``data``, ``error``, and ``background`` have units, then
    they all are checked that they have units and the units are the
    same.

    If ``effective_gain`` is input, then the total error array including
    source Poisson noise is calculated.

    If ``background`` is input, it is subtracted from ``data``.
    ``background`` is always returned as a 2D array with the same shape
    as ``data``.

    Notes
    -----
    ``data``, ``error``, and ``background`` must all have the same units
    if they are `~astropy.units.Quantity`\s.

    If ``effective_gain`` is a `~astropy.units.Quantity`, then it must
    have units such that ``effective_gain * data`` is in units of counts
    (e.g. counts, electrons, or photons).
    """

    inputs = [data, error, background]
    _check_units(inputs)

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        if effective_gain is not None:
            # data here should include the background
            error = calculate_total_error(data, error, effective_gain)

    # subtract background after calculating total variance
    if background is not None:
        data, background = subtract_background(data, background)

    return data, error, background
