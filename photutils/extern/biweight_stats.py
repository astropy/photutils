# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains new versions of the ``biweight_location`` and
``biweight_midvariance`` functions from astropy.stats.  These versions
support the ``axis`` keyword, which is needed by the background classes.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import median_absolute_deviation


__all__ = ['biweight_location', 'biweight_midvariance']


def biweight_location(a, c=6.0, M=None, axis=None):
    """
    Compute the biweight location.

    The biweight location is a robust statistic for determining the
    central location of a distribution.  It is given by:

    .. math::

        C_{bl}= M+\\frac{\Sigma_{\|u_i\|<1} (x_i-M)(1-u_i^2)^2}
        {\Sigma_{\|u_i\|<1} (1-u_i^2)^2}

    where :math:`M` is the sample median (or the input initial guess)
    and :math:`u_i` is given by:

    .. math::

        u_{i} = \\frac{(x_i-M)}{c\ MAD}

    where :math:`c` is the tuning constant and :math:`MAD` is the median
    absolute deviation.

    For more details, see `Beers, Flynn, and Gebhardt (1990); AJ 100, 32
    <http://adsabs.harvard.edu/abs/1990AJ....100...32B>`_.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    c : float, optional
        Tuning constant for the biweight estimator.  Default value is 6.0.
    M : float or array-like, optional
        Initial guess for the biweight location.  An array can be input
        when using the ``axis`` keyword.
    axis : int, optional
        Axis along which the biweight locations are computed.  The
        default (`None`) is to compute the biweight location of the
        flattened array.

    Returns
    -------
    biweight_location : float or `~numpy.ndarray`
        The biweight location of the input data.  If ``axis`` is `None`
        then a scalar will be returned, otherwise a `~numpy.ndarray`
        will be returned.

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    biweight location of the distribution::

        >>> import numpy as np
        >>> from astropy.stats import biweight_location
        >>> rand = np.random.RandomState(12345)
        >>> from numpy.random import randn
        >>> loc = biweight_location(rand.randn(1000))
        >>> print(loc)    # doctest: +FLOAT_CMP
        -0.0175741540445

    See Also
    --------
    biweight_midvariance, median_absolute_deviation, mad_std
    """

    a = np.asanyarray(a)

    if M is None:
        M = np.median(a, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = a - M

    # set up the weighting
    mad = median_absolute_deviation(a, axis=axis)
    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    u = (1 - u ** 2) ** 2
    u[mask] = 0

    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)


def biweight_midvariance(a, c=9.0, M=None, axis=None):
    """
    Compute the biweight midvariance.

    The biweight midvariance is a robust statistic for determining the
    midvariance (i.e. the standard deviation) of a distribution.  It is
    given by:

    .. math::

      C_{bl}= (n')^{1/2} \\frac{[\Sigma_{|u_i|<1} (x_i-M)^2(1-u_i^2)^4]^{0.5}}
      {|\Sigma_{|u_i|<1} (1-u_i^2)(1-5u_i^2)|}

    where :math:`u_i` is given by

    .. math::

        u_{i} = \\frac{(x_i-M)}{c MAD}

    where :math:`c` is the tuning constant and :math:`MAD` is the median
    absolute deviation.  The midvariance tuning constant ``c`` is
    typically 9.0.

    :math:`n'` is the number of points for which :math:`|u_i| < 1`
    holds, while the summations are over all :math:`i` up to :math:`n`:

    .. math::

        n' = \Sigma_{|u_i|<1}^n 1

    This is slightly different than given in the reference below, but
    results in a value closer to the true midvariance.
    For more details, see `Beers, Flynn, and Gebhardt (1990); AJ 100, 32
    <http://adsabs.harvard.edu/abs/1990AJ....100...32B>`_.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    c : float, optional
        Tuning constant for the biweight estimator.  Default value is 9.0.
    M : float or array-like, optional
        Initial guess for the biweight location.  An array can be input
        when using the ``axis`` keyword.
    axis : int, optional
        Axis along which the biweight midvariances are computed.  The
        default (`None`) is to compute the biweight midvariance of the
        flattened array.

    Returns
    -------
    biweight_midvariance : float or `~numpy.ndarray`
        The biweight midvariance of the input data.  If ``axis`` is
        `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    biweight midvariance of the distribution::

        >>> import numpy as np
        >>> from astropy.stats import biweight_midvariance
        >>> rand = np.random.RandomState(12345)
        >>> from numpy.random import randn
        >>> bmv = biweight_midvariance(rand.randn(1000))
        >>> print(bmv)    # doctest: +FLOAT_CMP
        0.986726249291

    See Also
    --------
    biweight_location, mad_std, median_absolute_deviation
    """

    a = np.asanyarray(a)

    if M is None:
        M = np.median(a, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = a - M

    # set up the weighting
    mad = median_absolute_deviation(a, axis=axis)
    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
    u = d / (c * mad)

    # now remove the outlier points
    mask = np.abs(u) < 1
    u = u ** 2
    n = mask.sum(axis=axis)

    f1 = d * d * (1. - u)**4
    f1[~mask] = 0.
    f1 = f1.sum(axis=axis) ** 0.5
    f2 = (1. - u) * (1. - 5.*u)
    f2[~mask] = 0.
    f2 = np.abs(f2.sum(axis=axis))

    return (n ** 0.5) * f1 / f2
