# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for computing and fitting harmonic functions.
"""

import numpy as np

__all__ = ['first_and_second_harmonic_function',
           'fit_first_and_second_harmonics', 'fit_upper_harmonic']


def _least_squares_fit(optimize_func, parameters):
    # call the least squares fitting
    # function and handle the result.
    from scipy.optimize import leastsq

    solution = leastsq(optimize_func, parameters, full_output=True)

    if solution[4] > 4:
        raise RuntimeError('Error in least squares fit: ' + solution[3])

    # return coefficients and covariance matrix
    return (solution[0], solution[1])


def first_and_second_harmonic_function(phi, c):
    r"""
    Compute the harmonic function value used to calculate the
    corrections for ellipse fitting.

    This function includes simultaneously both the first and second
    order harmonics:

    .. math::

        f(phi) = c[0] + c[1]*\sin(phi) + c[2]*\cos(phi) +
                 c[3]*\sin(2*phi) + c[4]*\cos(2*phi)

    Parameters
    ----------
    phi : float or `~numpy.ndarray`
        The angle(s) along the elliptical path, going towards the positive
        y axis, starting coincident with the position angle. That is, the
        angles are defined from the semimajor axis that lies in
        the positive x quadrant.
    c : `~numpy.ndarray` of shape (5,)
        Array containing the five harmonic coefficients.

    Returns
    -------
    result : float or `~numpy.ndarray`
        The function value(s) at the given input angle(s).
    """
    return (c[0] + c[1] * np.sin(phi) + c[2] * np.cos(phi)
            + c[3] * np.sin(2 * phi) + c[4] * np.cos(2 * phi))


def fit_first_and_second_harmonics(phi, intensities):
    r"""
    Fit the first and second harmonic function values to a set of
    (angle, intensity) pairs.

    This function is used to compute corrections for ellipse fitting:

    .. math::

        f(phi) = y0 + a1*\sin(phi) + b1*\cos(phi) + a2*\sin(2*phi) +
                 b2*\cos(2*phi)

    Parameters
    ----------
    phi : float or `~numpy.ndarray`
        The angle(s) along the elliptical path, going towards the positive
        y axis, starting coincident with the position angle. That is, the
        angles are defined from the semimajor axis that lies in
        the positive x quadrant.
    intensities : `~numpy.ndarray`
        The intensities measured along the elliptical path, at the
        angles defined by the ``phi`` parameter.

    Returns
    -------
    y0, a1, b1, a2, b2 : float
        The fitted harmonic coefficient values.
    """
    a1 = b1 = a2 = b2 = 1.0

    def optimize_func(x):
        return first_and_second_harmonic_function(
            phi, np.array([x[0], x[1], x[2], x[3], x[4]])) - intensities

    return _least_squares_fit(optimize_func, [np.mean(intensities), a1, b1,
                                              a2, b2])


def fit_upper_harmonic(phi, intensities, order):
    r"""
    Fit upper harmonic function to a set of (angle, intensity) pairs.

    With ``order`` set to 3 or 4, the resulting amplitudes, divided by
    the semimajor axis length and local gradient, measure the deviations
    from perfect ellipticity.

    The harmonic function that is fit is:

    .. math::
        y(phi, order) = y0 + An*\sin(order*phi) + Bn*\cos(order*phi)

    Parameters
    ----------
    phi : float or `~numpy.ndarray`
        The angle(s) along the elliptical path, going towards the positive
        y axis, starting coincident with the position angle. That is, the
        angles are defined from the semimajor axis that lies in
        the positive x quadrant.
    intensities : `~numpy.ndarray`
        The intensities measured along the elliptical path, at the
        angles defined by the ``phi`` parameter.
    order : int
        The order of the harmonic to be fitted.

    Returns
    -------
    y0, An, Bn : float
        The fitted harmonic values.
    """
    an = bn = 1.0

    def optimize_func(x):
        return (x[0] + x[1] * np.sin(order * phi)
                + x[2] * np.cos(order * phi) - intensities)

    return _least_squares_fit(optimize_func, [np.mean(intensities), an, bn])
