# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for computing and fitting harmonic functions.
"""

import numpy as np

__all__ = ['first_and_second_harmonic_function',
           'fit_first_and_second_harmonics', 'fit_upper_harmonic']


def _linear_least_squares_fit(design, values):
    """
    Solve a linear least-squares problem.

    The harmonic functions are linear in their coefficients, so they
    are fit directly instead of with an iterative nonlinear fitter.

    Parameters
    ----------
    design : 2D `~numpy.ndarray`
        The design matrix, with one column per coefficient.

    values : 1D `~numpy.ndarray`
        The data values.

    Returns
    -------
    coeffs : 1D `~numpy.ndarray`
        The fitted coefficients.

    covariance : 2D `~numpy.ndarray` or `None`
        The unscaled covariance matrix of the coefficients, i.e., the
        inverse of the normal matrix. It must be multiplied by the
        residual variance to get the covariance matrix. It is `None`
        if the data points do not constrain all of the coefficients.

    Raises
    ------
    ValueError
        If there are fewer data values than coefficients.
    """
    n_coeffs = design.shape[1]
    if len(values) < n_coeffs:
        msg = (f'At least {n_coeffs} data points are required for the '
               f'harmonic fit, got {len(values)}')
        raise ValueError(msg)

    # The singular value decomposition of the design matrix gives both
    # the coefficients and their covariance matrix. Inverting the
    # normal matrix instead would square the condition number, which
    # is large if the angles cover only a short arc.
    u, s, vt = np.linalg.svd(design, full_matrices=False)

    # Singular values below this limit are zero to within the rounding
    # errors. Their directions are excluded from the solution, which
    # is then the minimum-norm solution.
    limit = s[0] * max(design.shape) * np.finfo(float).eps
    is_full_rank = s[-1] > limit
    s_inv = np.zeros_like(s)
    np.divide(1.0, s, out=s_inv, where=s > limit)

    coeffs = (vt.T * s_inv) @ (u.T @ values)
    covariance = (vt.T * s_inv**2) @ vt if is_full_rank else None

    return coeffs, covariance


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
        y-axis, starting coincident with the position angle. That is, the
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
        y-axis, starting coincident with the position angle. That is, the
        angles are defined from the semimajor axis that lies in
        the positive x quadrant.
    intensities : 1D `~numpy.ndarray`
        The intensities measured along the elliptical path, at the
        angles defined by the ``phi`` parameter.

    Returns
    -------
    coeffs : 1D `~numpy.ndarray`
        The fitted harmonic coefficient values ``(y0, a1, b1, a2,
        b2)``.

    covariance : 2D `~numpy.ndarray` or `None`
        The unscaled covariance matrix of the coefficients. It must be
        multiplied by the variance of the fit residuals to get the
        covariance matrix. It is `None` if the angles do not constrain
        all of the coefficients.

    Raises
    ------
    ValueError
        If there are fewer than 5 data points.
    """
    # The isophote fitter relies on the coefficient order (mean, A1,
    # B1, A2, B2). See _HARMONIC_PARAMETERS in the geometry module.
    phi = np.asarray(phi)
    design = np.column_stack([np.ones_like(phi), np.sin(phi), np.cos(phi),
                              np.sin(2 * phi), np.cos(2 * phi)])
    return _linear_least_squares_fit(design, intensities)


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
        y-axis, starting coincident with the position angle. That is, the
        angles are defined from the semimajor axis that lies in
        the positive x quadrant.
    intensities : 1D `~numpy.ndarray`
        The intensities measured along the elliptical path, at the
        angles defined by the ``phi`` parameter.
    order : int
        The order of the harmonic to be fitted.

    Returns
    -------
    coeffs : 1D `~numpy.ndarray`
        The fitted harmonic values ``(y0, An, Bn)``.

    covariance : 2D `~numpy.ndarray` or `None`
        The unscaled covariance matrix of the coefficients. It must be
        multiplied by the variance of the fit residuals to get the
        covariance matrix. It is `None` if the angles do not constrain
        all of the coefficients.

    Raises
    ------
    ValueError
        If there are fewer than 3 data points.
    """
    phi = np.asarray(phi)
    design = np.column_stack([np.ones_like(phi), np.sin(order * phi),
                              np.cos(order * phi)])
    return _linear_least_squares_fit(design, intensities)
