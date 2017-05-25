# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains an updated version of ``mad_std`` from
astropy.stats.  This version supports the ``axis`` keyword, which is
needed by the background classes.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from distutils.version import LooseVersion
from warnings import warn
import numpy as np
from astropy.utils import isiterable


def mad_std(data, axis=None, func=None, ignore_nan=False):
    r"""
    Calculate a robust standard deviation using the `median absolute
    deviation (MAD)
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
            \approx 1.4826 \ \textrm{MAD}

    where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.

    Parameters
    ----------
    data : array-like
        Data array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis along which the robust standard deviations are computed.
        The default (`None`) is to compute the robust standard deviation
        of the flattened array.
    func : callable, optional
        The function used to compute the median. Defaults to `numpy.ma.median`
        for masked arrays, otherwise to `numpy.median`.
    ignore_nan : bool
        Ignore NaN values (treat them as if they are not in the array) when
        computing the median.  This will use `numpy.ma.median` if ``axis`` is
        specified, or `numpy.nanmedian` if ``axis=None`` and numpy's version is
        >1.10 because nanmedian is slightly faster in this case.

    Returns
    -------
    mad_std : float or `~numpy.ndarray`
        The robust standard deviation of the input data.  If ``axis`` is
        `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.
    """

    # NOTE: 1. / scipy.stats.norm.ppf(0.75) = 1.482602218505602
    MAD = median_absolute_deviation(data, axis=axis, func=func,
                                    ignore_nan=ignore_nan)
    return MAD * 1.482602218505602


def median_absolute_deviation(data, axis=None, func=None, ignore_nan=False):
    """
    Calculate the median absolute deviation (MAD).

    The MAD is defined as ``median(abs(a - median(a)))``.

    Parameters
    ----------
    data : array-like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis along which the MADs are computed.  The default (`None`) is
        to compute the MAD of the flattened array.
    func : callable, optional
        The function used to compute the median. Defaults to `numpy.ma.median`
        for masked arrays, otherwise to `numpy.median`.
    ignore_nan : bool
        Ignore NaN values (treat them as if they are not in the array) when
        computing the median.  This will use `numpy.ma.median` if ``axis`` is
        specified, or `numpy.nanmedian` if ``axis==None`` and numpy's version
        is >1.10 because nanmedian is slightly faster in this case.

    Returns
    -------
    mad : float or `~numpy.ndarray`
        The median absolute deviation of the input array.  If ``axis``
        is `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.
    """

    if func is None:
        # Check if the array has a mask and if so use np.ma.median
        # See https://github.com/numpy/numpy/issues/7330 why using np.ma.median
        # for normal arrays should not be done (summary: np.ma.median always
        # returns an masked array even if the result should be scalar). (#4658)
        if isinstance(data, np.ma.MaskedArray):
            is_masked = True
            func = np.ma.median
            if ignore_nan:
                data = np.ma.masked_invalid(data)
        elif ignore_nan:
            is_masked = False
            func = np.nanmedian
        else:
            is_masked = False
            func = np.median
    else:
        is_masked = None

    if (not ignore_nan and
            (LooseVersion(np.__version__) < LooseVersion('1.10'))
            and np.any(np.isnan(data))):
        warn("Numpy versions <1.10 will return a number rather than "
             "NaN for the median of arrays containing NaNs.  This "
             "behavior is unlikely to be what you expect.")

    data = np.asanyarray(data)
    # np.nanmedian has `keepdims`, which is a good option if we're not allowing
    # user-passed functions here
    data_median = func(data, axis=axis)

    # broadcast the median array before subtraction
    if axis is not None:
        if isiterable(axis):
            for ax in sorted(list(axis)):
                data_median = np.expand_dims(data_median, axis=ax)
        else:
            data_median = np.expand_dims(data_median, axis=axis)

    result = func(np.abs(data - data_median), axis=axis, overwrite_input=True)

    if axis is None and np.ma.isMaskedArray(result):
        # return scalar version
        result = result.item()
    elif np.ma.isMaskedArray(result) and not is_masked:
        # if the input array was not a masked array, we don't want to return a
        # masked array
        result = result.filled(fill_value=np.nan)

    return result
