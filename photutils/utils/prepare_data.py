# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable


__all__ = ['calculate_total_error']


def calculate_total_error(data, error, effective_gain):
    """
    Calculate a total error array, combining a background error array
    with the Poisson noise of sources.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The (background-subtracted) data array.

    error : array_like or `~astropy.units.Quantity`
        The pixel-wise Gaussian 1-sigma background errors of the input
        ``data``.  ``error`` should include all sources of "background"
        error but *exclude* the Poisson error of the sources.  ``error``
        must have the same shape as ``data``.

    effective_gain : float, array-like, or `~astropy.units.Quantity`
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.

    Returns
    -------
    total_error : `~numpy.ndarray` or `~astropy.units.Quantity`
        Total error.

    Notes
    -----
    The total error array, :math:`\sigma_{\mathrm{tot}}` is:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                  \\frac{I}{g}}

    where :math:`\sigma_b`, :math:`I`, and :math:`g` are the background
    ``error`` image, (background-subtracted) ``data`` image, and
    ``effective_gain``, respectively.

    ``data`` here should be background-subtracted to match SExtractor.
    """

    data = np.asanyarray(data)
    error = np.asanyarray(error)

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
        effective_gain = np.zeros(data.shape) + effective_gain
    else:
        effective_gain = np.asanyarray(effective_gain)
        if effective_gain.shape != data.shape:
            raise ValueError('If input effective_gain is 2D, then it must '
                             'have the same shape as the input data.')
    if np.any(effective_gain <= 0):
        raise ValueError('effective_gain must be strictly positive '
                         'everywhere')

    if all(has_unit):
        source_variance = np.maximum((data * data.unit) /
                                     effective_gain.value, 0. * error.unit**2)
        variance_total = error**2 + source_variance
    else:
        source_variance = np.maximum(data / effective_gain, 0)
        variance_total = error**2 + source_variance

    return np.sqrt(variance_total)


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

    If ``background`` is input, then it is returned as a 2D array with
    the same shape as ``data`` (if necessary).  It is *not* subtracted
    from the input ``data``.

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

    # generate a 2D background array, if necessary
    if background is not None:
        if not isiterable(background):
            background = np.zeros(data.shape) + background
        else:
            if background.shape != data.shape:
                raise ValueError('If input background is 2D, then it must '
                                 'have the same shape as the input data.')

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        if effective_gain is not None:
            error = calculate_total_error(data, error, effective_gain)

    return data, error, background
