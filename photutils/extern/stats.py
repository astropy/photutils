# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import median_absolute_deviation as mad


__all__ = ['mad_std', 'gaussian_fwhm_to_sigma', 'gaussian_sigma_to_fwhm']


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


def gaussian_fwhm_to_sigma(fwhm):
    """
    Convert Gaussian full-width at half-maximum(s) (FWHM) to the 1-sigma
    standard deviation(s).

    Parameters
    ----------
    fwhm : float, array-like
        The Gaussian FWHM(s).

    Returns
    -------
    sigma : float, array-like
        The Gaussian 1-sigma standard deviation(s).

    Examples
    --------
    >>> from photutils.extern.stats import gaussian_fwhm_to_sigma
    >>> gaussian_fwhm_to_sigma(3.0)    # doctest: +FLOAT_CMP
    1.27398270043
    """

    return np.array(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_sigma_to_fwhm(sigma):
    """
    Convert Gaussian 1-sigma standard deviation(s) to full-width at
    half-maximum(s) (FWHM).

    Parameters
    ----------
    sigma : float, array-like
        The Gaussian 1-sigma standard deviation(s).

    Returns
    -------
    fwhm : float, array-like
        The Gaussian FWHM(s).

    Examples
    --------
    >>> from photutils.extern.stats import gaussian_sigma_to_fwhm
    >>> gaussian_sigma_to_fwhm(3.0)    # doctest: +FLOAT_CMP
    7.06446013509
    """

    return np.array(sigma) * (2.0 * np.sqrt(2.0 * np.log(2.0)))
