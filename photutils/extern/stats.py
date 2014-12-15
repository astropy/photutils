# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import median_absolute_deviation as mad


__all__ = ['mad_std', 'fwhm2sigma']


def mad_std(data):
    """
    Calculate a robust standard deviation using the `median absolute
    deviation (MAD)
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \\sigma \\approx \\frac{\\textrm{MAD}}{\Phi^{-1}(3/4)} \\approx 1.4826 \ \\textrm{MAD}

    where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
    distribution function evaulated at probability :math:`P = 3/4`.

    Parameters
    ----------
    data : array-like
        2D data array

    Returns
    -------
    result : float
        The robust standard deviation of the data.

    Examples
    --------
    >>> from photutils.utils import check_random_state
    >>> from photutils.extern.stats import mad_std
    >>> prng = check_random_state(12345)
    >>> data = prng.normal(5, 2, size=(100, 100))
    >>> print(mad_std(data))    # doctest: +FLOAT_CMP
    2.02327646594
    """

    from scipy.stats import norm
    return mad(data) / norm.ppf(0.75)


def fwhm2sigma(fwhm):
    """
    Convert a Gaussian full-width at half-maximum (FWHM) to the 1-sigma
    standard deviation.

    Parameters
    ----------
    fwhm : float
        The Gaussian FWHM.

    Returns
    -------
    sigma : float
        The Gaussian 1-sigma standard deviation.

    Examples
    --------
    >>> from photutils.extern.stats import fwhm2sigma
    >>> print(fwhm2sigma(3.0))    # doctest: +FLOAT_CMP
    1.2739827004320285
    """

    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
