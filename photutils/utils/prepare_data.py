# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.utils.misc import isiterable


__all__ = ['calc_total_error']


def calc_total_error(data, bkg_error, effective_gain):
    """
    Calculate a total error array, combining a background-only error
    array with the Poisson noise of sources.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The data array.

    bkg_error : array_like or `~astropy.units.Quantity`
        The pixel-wise Gaussian 1-sigma background-only errors of the
        input ``data``.  ``error`` should include all sources of
        "background" error but *exclude* the Poisson error of the
        sources.  ``error`` must have the same shape as ``data``.

    effective_gain : float, array-like, or `~astropy.units.Quantity`
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.

    Returns
    -------
    total_error : `~numpy.ndarray` or `~astropy.units.Quantity`
        The total error array.  If ``data``, ``bkg_error``, and
        ``effective_gain`` are all `~astropy.units.Quantity` objects,
        then ``total_error`` will also be returned as a
        `~astropy.units.Quantity` object.  Otherwise, a `~numpy.ndarray`
        will be returned.

    Notes
    -----
    To use units, ``data``, ``bkg_error``, and ``effective_gain`` must
    *all* be `~astropy.units.Quantity` objects.  A `ValueError` will be
    raised if only some of the inputs are `~astropy.units.Quantity`
    objects.

    The total error array, :math:`\sigma_{\mathrm{tot}}` is:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                  \\frac{I}{g}}

    where :math:`\sigma_b`, :math:`I`, and :math:`g` are the background
    ``bkg_error`` image, ``data`` image, and ``effective_gain``,
    respectively.

    Pixels where ``data`` (:math:`I_i)` is negative do not contribute
    additional Poisson noise to the total error, i.e.
    :math:`\sigma_{\mathrm{tot}, i} = \sigma_{\mathrm{b}, i}`.  Note
    that this is different from `SExtractor`_, which sums the total
    variance in the segment, including pixels where :math:`I_i` is
    negative.  In such cases, `SExtractor`_ underestimates the total
    errors.  Also note that ``data`` should be background-subtracted to
    match SExtractor's errors.

    ``effective_gain`` can either be a scalar value or a 2D image with
    the same shape as the ``data``.  A 2D image is useful with mosaic
    images that have variable depths (i.e., exposure times) across the
    field. For example, one should use an exposure-time map as the
    ``effective_gain`` for a variable depth mosaic image in count-rate
    units.

    If your input ``data`` are in units of ADU, then ``effective_gain``
    should represent electrons/ADU.  If your input ``data`` are in units
    of electrons/s then ``effective_gain`` should be the exposure time
    or an exposure time map (e.g., for mosaics with non-uniform exposure
    times).

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    data = np.asanyarray(data)
    bkg_error = np.asanyarray(bkg_error)

    inputs = [data, bkg_error, effective_gain]
    has_unit = [hasattr(x, 'unit') for x in inputs]
    use_units = all(has_unit)
    if any(has_unit) and not all(has_unit):
        raise ValueError('If any of data, bkg_error, or effective_gain has '
                         'units, then they all must all have units.')

    if use_units:
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

    if use_units:
        source_variance = np.maximum((data * data.unit) /
                                     effective_gain.value,
                                     0. * bkg_error.unit**2)
    else:
        source_variance = np.maximum(data / effective_gain, 0)

    return np.sqrt(bkg_error**2 + source_variance)


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
            error = calc_total_error(data, error, effective_gain)

    return data, error, background
