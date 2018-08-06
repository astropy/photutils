# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
        input ``data``.  ``bkg_error`` should include all sources of
        "background" error but *exclude* the Poisson error of the
        sources.  ``bkg_error`` must have the same shape as ``data``.
        If ``data`` and ``bkg_error`` are `~astropy.units.Quantity`
        objects, then they must have the same units.

    effective_gain : float, array-like, or `~astropy.units.Quantity`
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.

    Returns
    -------
    total_error : `~numpy.ndarray` or `~astropy.units.Quantity`
        The total error array.  If ``data``, ``bkg_error``, and
        ``effective_gain`` are all `~astropy.units.Quantity` objects,
        then ``total_error`` will also be returned as a
        `~astropy.units.Quantity` object with the same units as the
        input ``data``.  Otherwise, a `~numpy.ndarray` will be returned.

    Notes
    -----
    To use units, ``data``, ``bkg_error``, and ``effective_gain`` must
    *all* be `~astropy.units.Quantity` objects.  ``data`` and
    ``bkg_error`` must have the same units.  A `ValueError` will be
    raised if only some of the inputs are `~astropy.units.Quantity`
    objects or if the ``data`` and ``bkg_error`` units differ.

    The total error array, :math:`\\sigma_{\\mathrm{tot}}` is:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                  \\frac{I}{g}}

    where :math:`\\sigma_b`, :math:`I`, and :math:`g` are the background
    ``bkg_error`` image, ``data`` image, and ``effective_gain``,
    respectively.

    Pixels where ``data`` (:math:`I_i)` is negative do not contribute
    additional Poisson noise to the total error, i.e.
    :math:`\\sigma_{\\mathrm{tot}, i} = \\sigma_{\\mathrm{b}, i}`.  Note
    that this is different from `SExtractor`_, which sums the total
    variance in the segment, including pixels where :math:`I_i` is
    negative.  In such cases, `SExtractor`_ underestimates the total
    errors.  Also note that SExtractor computes Poisson errors from
    background-subtracted data, which also results in an underestimation
    of the Poisson noise.

    ``effective_gain`` can either be a scalar value or a 2D image with
    the same shape as the ``data``.  A 2D image is useful with mosaic
    images that have variable depths (i.e., exposure times) across the
    field.  For example, one should use an exposure-time map as the
    ``effective_gain`` for a variable depth mosaic image in count-rate
    units.

    As an example, if your input ``data`` are in units of ADU, then
    ``effective_gain`` should be in units of electrons/ADU (or
    photons/ADU).  If your input ``data`` are in units of electrons/s
    then ``effective_gain`` should be the exposure time or an exposure
    time map (e.g., for mosaics with non-uniform exposure times).

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    data = np.asanyarray(data)
    bkg_error = np.asanyarray(bkg_error)

    inputs = [data, bkg_error, effective_gain]
    has_unit = [hasattr(x, 'unit') for x in inputs]
    use_units = all(has_unit)
    if any(has_unit) and not use_units:
        raise ValueError('If any of data, bkg_error, or effective_gain has '
                         'units, then they all must all have units.')

    if use_units:
        if data.unit != bkg_error.unit:
            raise ValueError('data and bkg_error must have the same units.')

        count_units = [u.electron, u.photon]
        datagain_unit = data.unit * effective_gain.unit
        if datagain_unit not in count_units:
            raise u.UnitsError('(data * effective_gain) has units of "{0}", '
                               'but it must have count units (e.g. '
                               'u.electron or u.photon).'
                               .format(datagain_unit))

    if not isiterable(effective_gain):
        effective_gain = np.zeros(data.shape) + effective_gain
    else:
        effective_gain = np.asanyarray(effective_gain)
        if effective_gain.shape != data.shape:
            raise ValueError('If input effective_gain is 2D, then it must '
                             'have the same shape as the input data.')
    if np.any(effective_gain <= 0):
        raise ValueError('effective_gain must be strictly positive '
                         'everywhere.')

    # This calculation assumes that data and bkg_error have the same
    # units.  source_variance is calculated to have units of
    # (data.unit)**2 so that it can be added with bkg_error**2 below.  The
    # final returned error will have units of data.unit.  np.maximum is
    # used to ensure that negative data values do not contribute to the
    # Poisson noise.
    if use_units:
        unit = data.unit
        data = data.value
        effective_gain = effective_gain.value
        source_variance = np.maximum(data / effective_gain, 0) * unit**2
    else:
        source_variance = np.maximum(data / effective_gain, 0)

    return np.sqrt(bkg_error**2 + source_variance)
