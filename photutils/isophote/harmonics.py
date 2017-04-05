from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.optimize import leastsq


def _dofit(optimize_func, parameters):
    # call the least squares fitting
    # function and handle the result.
    solution = leastsq(optimize_func, parameters, full_output=True)

    if solution[4] > 4:
        raise RuntimeError("Error in least squares fit: " + solution[3])

    # return coefficients and covariance matrix
    return (solution[0], solution[1])


def first_and_2nd_harmonic_function(phi, c):
    """
    Computes harmonic function used to calculate the
    corrections for ellipse fitting. This function includes
    simultaneously both the 1st and 2nd order harmonics.

    function = c[0] + c[1]*sin(phi) + c[2]*cos(phi) + c[3]*sin(2*phi) + c[4]*cos(2*phi)

    Parameters
    ----------
    phi : float or np.array
        angle(s) along the elliptical path, going towards the +Y axis,
        starting coincident with the position angle. That is, the
        angles are defined from the semi-major axis that lies in
        the +X quadrant.
    c : np array of shape (5)
        containing the five harmonic coefficients

    Returns
    -------
    float or np.array
        function value(s) at the given input angle(s)
    """
    return c[0] + c[1]*np.sin(phi) + c[2]*np.cos(phi) + c[3]*np.sin(2*phi) + c[4]*np.cos(2*phi)


def fit_1st_and_2nd_harmonics(phi, intensities):
    """
    Fits 1st and 2nd harmonic function to a set of angle,intensity pairs.
    This is used to compute the corrections for ellipse fitting.

    Parameters
    ----------
    phi : np.array
        angles defined in the same way as in harmonic_function
    intensities : np.array
        intensities measured along the elliptical path, at the angles defined in parameter `phi`

    Returns
    -------
    5 float values
        fitted values for y0, a1, b1, a2, b2
    """
    a1 = b1 = a2 = b2 = 1.

    optimize_func = lambda x: first_and_2nd_harmonic_function(phi, np.array([x[0], x[1], x[2], x[3], x[4]])) - intensities

    return _dofit(optimize_func, [np.mean(intensities), a1, b1, a2, b2])


def fit_upper_harmonic(phi, intensities, order):
    """
    Fits upper harmonic function to a set of angle,intensity pairs.
    With `order` set to 3 or 4, the resulting amplitudes, divided
    by the semi-major axis length and local gradient, measure the
    deviations from perfect ellipticity.

    function = y0 + c[0]*sin(order*phi) + c[1]*cos(order*phi)

    Parameters
    ----------
    phi : np.array
        angles defined in the same way as in harmonic_function
    intensities : np.array
        intensities measured along the elliptical path, at the angles defined in parameter `phi`
    order : int
        the order of the harmonic to be fitted.

    Returns
    -------
    3 float values
        fitted values for y0, an, bn
    """
    an = bn = 1.

    optimize_func = lambda x: x[0] + x[1]*np.sin(order*phi) + x[2]*np.cos(order*phi) - intensities

    return _dofit(optimize_func, [np.mean(intensities), an, bn])
