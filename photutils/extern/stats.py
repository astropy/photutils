# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains an updated version of ``mad_std`` from
astropy.stats.  This version supports the ``axis`` keyword, which is
needed by the background classes.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.stats import median_absolute_deviation


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
