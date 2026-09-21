# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating total error arrays.
"""

import astropy.units as u
import numpy as np

__all__ = ['calc_total_error']


def calc_total_error(data, bkg_error, effective_gain):
    r"""
    Calculate a total error array by combining a background-only error
    array with the Poisson noise of sources.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The background-subtracted data array.

    bkg_error : array_like or `~astropy.units.Quantity`
        The 1-sigma background-only errors of the input ``data``.
        ``bkg_error`` should include all sources of "background" error
        but *exclude* the Poisson error of the sources. ``bkg_error``
        must have the same shape as ``data``. If ``data`` and
        ``bkg_error`` are `~astropy.units.Quantity` objects, then they
        must have the same units.

    effective_gain : float, array_like, or `~astropy.units.Quantity`
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.
        If ``effective_gain`` is zero (or contains zero values in
        an array), then the source Poisson noise component will not
        be included. In other words, the returned total error value
        will simply be the ``bkg_error`` value for pixels where
        ``effective_gain`` is zero. ``effective_gain`` cannot not be
        negative or contain negative values.

    Returns
    -------
    total_error : `~numpy.ndarray` or `~astropy.units.Quantity`
        The total error array. If ``data``, ``bkg_error``, and
        ``effective_gain`` are all `~astropy.units.Quantity`
        objects, then ``total_error`` will also be returned as a
        `~astropy.units.Quantity` object with the same units as the
        input ``data``. Otherwise, a `~numpy.ndarray` will be returned.
        The dtype follows the NumPy type promotion of the floating-point
        inputs (see Notes).

    Notes
    -----
    To use units, ``data``, ``bkg_error``, and ``effective_gain``
    must *all* be `~astropy.units.Quantity` objects. ``data`` and
    ``bkg_error`` must have the same units. A `ValueError` will be
    raised if only some of the inputs are `~astropy.units.Quantity`
    objects or if the ``data`` and ``bkg_error`` units differ.

    The source Poisson error in countable units (e.g., electrons or
    photons) is:

    .. math::

        \sigma_{\mathrm{src}}  = \sqrt{g_{\mathrm{eff}} I}

    where :math:`g_{\mathrm{eff}}` is the effective gain
    (``effective_gain``, an image or scalar) and :math:`I` is the
    ``data`` image.

    The total error is the combination of the background-only
    error and the source Poisson error. The total error array
    :math:`\sigma_{\mathrm{tot}}` in countable units (e.g., electrons or
    photons) is therefore:

    .. math::

        \sigma_{\mathrm{tot}}  = \sqrt{g_{\mathrm{eff}}^2
        \sigma_{\mathrm{bkg}}^2 + g_{\mathrm{eff}} I}

    where :math:`\sigma_{\mathrm{bkg}}` is the background-only error
    image (``bkg_error``).

    Converting back to the input ``data`` units gives:

    .. math::

        \sigma_{\mathrm{tot}}  = \frac{1}{g_{\mathrm{eff}}}
        \sqrt{g_{\mathrm{eff}}^2
        \sigma_{\mathrm{bkg}}^2 + g_{\mathrm{eff}} I}

    .. math::

        \sigma_{\mathrm{tot}} = \sqrt{\sigma_{\mathrm{bkg}}^2 +
                  \frac{I}{g_{\mathrm{eff}}}}

    The dtype of the returned array follows the NumPy type promotion of
    the ``data`` and ``bkg_error`` arrays, and of ``effective_gain``
    if it is an array, with a minimum of float32. For example, float32
    ``data`` and ``bkg_error`` arrays give a float32 total error, while
    float32 ``data`` with a float64 ``bkg_error`` array gives a float64
    total error. A scalar ``effective_gain`` does not affect the dtype.
    If any of the input arrays has a non-floating-point dtype (e.g.,
    integer ``data``), the returned array is float64. The inputs are
    converted to the output dtype before the calculation.

    ``effective_gain`` can either be a scalar value or a 2D image with
    the same shape as the ``data``. A 2D ``effective_gain`` image is
    useful when the input ``data`` has variable depths across the field
    (e.g., a mosaic image with non-uniform exposure times). For example,
    if your input ``data`` are in units of electrons/s then ideally
    ``effective_gain`` should be an exposure-time map.

    The Poisson noise component is not included in the output total
    error for pixels where ``data`` (:math:`I_i)` is negative. For such
    pixels, :math:`\sigma_{\mathrm{tot}, i} = \sigma_{\mathrm{bkg}, i}`.

    The Poisson noise component is also not included in the output total
    error for pixels where the effective gain (:math:`g_{\mathrm{eff},
    i}`) is zero. For such pixels, :math:`\sigma_{\mathrm{tot}, i} =
    \sigma_{\mathrm{bkg}, i}`.

    To replicate `SourceExtractor`_ errors when it is configured to
    consider weight maps as gain maps (i.e., 'WEIGHT_GAIN=Y', which is
    the default), one should input an ``effective_gain`` calculated as:

    .. math::

        g_{\mathrm{eff}}^{\prime} = g_{\mathrm{eff}} \left(
        \frac{\mathrm{RMS_{\mathrm{median}}^2}}{\sigma_{\mathrm{bkg}}^2}
        \right)

    where :math:`g_{\mathrm{eff}}` is the effective gain,
    :math:`\sigma_{\mathrm{bkg}}` are the background-only errors,
    and :math:`\mathrm{RMS_{\mathrm{median}}}` is the median
    value of the low-resolution background RMS map generated by
    `SourceExtractor`_. When running `SourceExtractor`_, this
    value is printed to stdout as "(M+D) RMS: <value>". If you are
    using `~photutils.background.Background2D`, the median value
    of the low-resolution background RMS map is returned via the
    `~photutils.background.Background2D.background_rms_median`
    attribute.

    In that case the total error is:

    .. math::

        \sigma_{\mathrm{tot}} = \sqrt{\sigma_{\mathrm{bkg}}^2 +
        \left(\frac{I}{g_{\mathrm{eff}}}\right)
        \left(\frac{\sigma_{\mathrm{bkg}}^2}
        {\mathrm{RMS_{\mathrm{median}}^2}}\right)}

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """
    data = np.asanyarray(data)
    bkg_error = np.asanyarray(bkg_error)

    inputs = [data, bkg_error, effective_gain]
    has_unit = [hasattr(x, 'unit') for x in inputs]
    use_units = all(has_unit)
    if any(has_unit) and not use_units:
        msg = ('If any of data, bkg_error, or effective_gain has units, '
               'then they all must have units.')
        raise ValueError(msg)

    if bkg_error.shape != data.shape:
        msg = ('bkg_error must have the same shape as the input data.')
        raise ValueError(msg)

    unit = None
    if use_units:
        if data.unit != bkg_error.unit:
            msg = 'data and bkg_error must have the same units'
            raise ValueError(msg)

        count_units = [u.electron, u.photon]
        datagain_unit = data.unit * effective_gain.unit
        if datagain_unit not in count_units:
            msg = ('(data * effective_gain) has units of '
                   f'{datagain_unit}, but it must have count units '
                   '(e.g., u.electron or u.photon).')
            raise u.UnitsError(msg)

        unit = data.unit
        data = data.value
        bkg_error = bkg_error.value
        effective_gain = effective_gain.value

    scalar_gain = np.ndim(effective_gain) == 0
    if not scalar_gain:
        effective_gain = np.asanyarray(effective_gain)
        if effective_gain.shape != data.shape:
            msg = ('If input effective_gain is 2D, then it must have '
                   'the same shape as the input data.')
            raise ValueError(msg)

    if np.any(effective_gain < 0):
        msg = 'effective_gain must be non-negative everywhere'
        raise ValueError(msg)

    # The arrays are converted to the output dtype before any
    # arithmetic, so that the precision of the result is not limited by
    # a lower-precision input.
    arrays = [data, bkg_error]
    if not scalar_gain:
        arrays.append(effective_gain)
    dtype = _total_error_dtype(*arrays)

    # Do not include source variance where effective_gain = 0.
    # source_variance is a copy of data, which is not modified.
    source_variance = data.astype(dtype)
    if scalar_gain:
        if effective_gain == 0:
            source_variance[...] = 0.0
        else:
            source_variance /= dtype.type(effective_gain)
    else:
        effective_gain = effective_gain.astype(dtype, copy=False)
        mask = effective_gain != 0
        source_variance[mask] /= effective_gain[mask]
        source_variance[~mask] = 0.0

    # Do not include source variance where data is negative (note that
    # effective_gain cannot be negative)
    np.maximum(source_variance, 0, out=source_variance)

    # source_variance has units of (data.unit)**2 so that it can be
    # added to bkg_error**2. The returned total error has the same units
    # as the data.
    total_error = source_variance
    total_error += bkg_error.astype(dtype, copy=False) ** 2
    np.sqrt(total_error, out=total_error)

    if unit is not None:
        total_error <<= unit
    return total_error


def _total_error_dtype(*arrays):
    """
    Return the dtype of the total error calculated from the input
    arrays.

    Parameters
    ----------
    *arrays : `~numpy.ndarray`
        The ``data`` and ``bkg_error`` arrays, and the
        ``effective_gain`` array if it is not a scalar.

    Returns
    -------
    dtype : `numpy.dtype`
        The NumPy promotion of the input dtypes, with a minimum of
        float32, if all of the arrays are floating-point arrays.
        Otherwise, float64.
    """
    if any(array.dtype.kind != 'f' for array in arrays):
        return np.dtype(np.float64)
    return np.result_type(np.float32, *[array.dtype for array in arrays])
