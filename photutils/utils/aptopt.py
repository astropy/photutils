# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module contains the centering routine (``aptopt.x``)
used by ``ofilter`` algorithm in IRAF.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# THIRD-PARTY
import numpy as np
from astropy.stats.funcs import gaussian_sigma_to_fwhm

__all__ = ['find_center']


def find_center(data, center, sigma, loc=None, maxiter=10, tol=0.001,
                max_search=3):
    """One-dimensional centering routine using repeated convolutions to
    locate image center.

    Parameters
    ----------
    data : array-like
        1D array to fit.

    center : float
        Initial guess at center. Must correspond to ``loc``, if given.

    sigma : float
        Sigma of Gaussian.

    loc : array-like or `None`
        1D array of bin centers or pixel values.

    maxiter : int, optional
        Maximum number of iterations.

    tol : float, optional
        Gap tolerance for sigma.

    max_search : int, optional
        Max initial search steps.

    Returns
    -------
    result : float
        Calculated center.

    niter : int
        Number of iterations used.

    Raises
    ------
    ValueError
        Calculations failed.

    """
    if sigma <= 0:
        raise ValueError('sigma must be greater than zero')

    # Convert center to index space.
    if loc is not None:
        min_loc = min(loc)
        max_loc = max(loc)
        center = apmapr(center, min_loc, max_loc, 1.0, data.size)

    # Initialize.
    wgt = ap_tprofder(data.size, center, sigma)[1]
    s = np.zeros(3)
    s[0] = np.dot(wgt, data)
    if s[0] == 0:
        return center, 0

    x = np.zeros(3)
    x[0] = center
    s[2] = s[0]

    # Search for the correct interval.
    i = 0
    while (i < max_search) and (s[2] * s[0] >= 0):
        s[2] = s[0]
        x[2] = x[0]
        x[0] = x[2] + apply_sign(sigma, s[2])
        wgt = ap_tprofder(data.size, x[0], sigma)[1]
        s[0] = np.dot(wgt, data)

        if s[0] == 0:
            return x[0], 0

        i += 1

    if s[2] * s[0] > 0:
        raise ValueError('Location not bracketed')

    # Intialize the quadratic search.
    delx = x[0] - x[2]
    x[1] = x[2] - s[2] * delx / (s[0] - s[2])
    wgt = ap_tprofder(data.size, x[1], sigma)[1]
    s[1] = np.dot(wgt, data)
    if s[1] == 0:
        return x[1], 1

    # Search quadratically.
    for niter in range(1, maxiter):

        # Check for completion.
        if s[1] == 0 or np.any(abs(x[1:] - x[:-1]) <= tol):
            break

        # Compute new intermediate value.
        newx = x[0] + apqzero(x, s)
        wgt = ap_tprofder(data.size, newx, sigma)[1]
        news = np.dot(wgt, data)

        if s[0] * s[1] > 0:
            s[0] = s[1]
            x[0] = x[1]
            s[1] = news
            x[1] = newx
        else:
            s[2] = s[1]
            x[2] = x[1]
            s[1] = news
            x[1] = newx

    # Convert center to index space.
    if loc is not None:
        result = apmapr(x[1], 1.0, data.size, min_loc, max_loc)
    else:
        result = x[1]

    return result, niter + 1


# Helper functions for the centering routine.


def apmapr(a, a1, a2, b1, b2):
    """Vector linear transformation.

    Map the range of pixel values ``a1, a2`` from ``a``
    into the range ``b1, b2`` into ``b``.
    It is assumed that ``a1 < a2`` and ``b1 < b2``.

    Parameters
    ----------
    a : float
        The value to be mapped.

    a1, a2 : float
        The numbers specifying the input data range.

    b1, b2 : float
        The numbers specifying the output data range.

    Returns
    -------
    b : float
        Mapped value.

    """
    scalar = (b2 - b1) / (a2 - a1)
    return max(b1, min(b2, (a - a1) * scalar + b1))


def ap_tprofder(npix, center, sigma, ampl=1.0):
    """Estimate the approximating triangle function and its derivatives.

    Parameters
    ----------
    npix : int
        Number of elements in output arrays.

    center, sigma : float
        Center and sigma of input Gaussian function.

    ampl : float, optional
        Amplitude.

    Returns
    -------
    data : array-like
        Output data.

    der : array-like
        Derivatives.

    """
    data = np.zeros(npix)
    der = np.zeros(npix)

    x = (np.arange(npix) - center + 0.5) / (sigma * gaussian_sigma_to_fwhm)
    xabs = np.abs(x)

    mask = xabs <= 1
    data[mask] = ampl * (1 - xabs[mask])
    der[mask] = x[mask] * data[mask]

    return data, der


def apply_sign(x, y):
    """Return the absolute value of ``x`` multiplied by
    the sign (i.e., +1 or -1) of ``y``."""
    if y < 0:
        fac = -1.0
    else:
        fac = 1.0
    return abs(x) * fac


def apqzero(x, y, qtol=0.125):
    """Return the root of a quadratic function defined by three points."""
    if len(x) != 3 or len(y) != 3:
        raise ValueError('This function only accepts 3 points')

    # Compute the determinant.
    x2 = x[1] - x[0]
    x3 = x[2] - x[0]
    y2 = y[1] - y[0]
    y3 = y[2] - y[0]
    det = x2 * x3 * (x2 - x3)

    # Compute the shift in x.
    if abs(det) > 0:
        a = (x3 * y2 - x2 * y3) / det
        b = -(x3 * x3 * y2 - x2 * x2 * y3) / det
        c =  a * y[0] / (b * b)
        if abs(c) > qtol:
            dx = (-b / (2.0 * a)) * (1.0 - np.sqrt(1.0 - 4.0 * c))
        else:
            dx = -(y[0] / b) * (1.0 + c)
    elif abs(y3) > 0:
        dx = -y[0] * x3 / y3
    else:
        dx = 0.0

    return dx
