# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Faster computation of the elliptical sampling path from sample.py.
"""

import numpy as np

from cpython.exc cimport PyErr_CheckSignals


cdef extern from "math.h" nogil:
    double cos(double x)
    double sin(double x)
    double sqrt(double x)

DTYPE = np.float64


cdef Py_ssize_t walk_path(
    double phi,
    double radius,
    double phi_max,
    double sma_q,
    double q,
    double[::1] phis,
    double[::1] radii,
    bint store,
) except -1 nogil:
    """
    Walk along the elliptical path and return the number of points.

    The points are stored in ``phis`` and ``radii`` only if ``store``
    is true. The walk of a very long path (a very large semimajor
    axis length) can be interrupted.
    """
    cdef Py_ssize_t n = 0
    cdef double step, next_phi, cos2, sin2

    while phi < phi_max:
        if store:
            phis[n] = phi
            radii[n] = radius
        n += 1

        # Check for a keyboard interrupt about every million points.
        if (n & 0xFFFFF) == 0:
            with gil:
                PyErr_CheckSignals()

        # The angle step is about one pixel along the path, but at most
        # 0.5 radians. Stop if the path cannot advance (a nonpositive or
        # NaN radius) or if the step is below the resolution of the
        # angles at the end of the path (a radius so large that the path
        # could not be completed).
        step = 1.0 / radius
        if step > 0.5:
            step = 0.5
        next_phi = phi + step
        if not (next_phi > phi and phi_max + step > phi_max):
            break

        # The squares are computed in separate statements so that a
        # compiler that contracts floating-point expressions only within
        # a statement (e.g., clang) does not fuse them with the sum. A
        # fused multiply-add rounds differently.
        phi = next_phi
        cos2 = q * cos(phi)
        cos2 = cos2 * cos2
        sin2 = sin(phi)
        sin2 = sin2 * sin2
        radius = sma_q / sqrt(cos2 + sin2)

    return n


def compute_polar_path(
    double phi,
    double radius,
    double phi_max,
    double sma,
    double eps,
):
    """
    Compute the polar angles and radii of the points along an
    elliptical path.

    The angle step between consecutive points is the inverse of the
    polar radius (about one pixel along the path), but at most 0.5
    radians.

    Parameters
    ----------
    phi : float
        The polar angle (radians) of the first point.

    radius : float
        The polar radius (pixels) of the first point.

    phi_max : float
        The polar angle (radians) where the path ends. The path
        includes only angles smaller than this value.

    sma : float
        The semimajor axis length (pixels) of the ellipse.

    eps : float
        The ellipticity of the ellipse.

    Returns
    -------
    phis, radii : 1D `~numpy.ndarray`
        The polar angles and polar radii of the points.
    """
    cdef double q = 1.0 - eps
    cdef double sma_q = sma * q
    cdef Py_ssize_t n
    cdef double[::1] phis
    cdef double[::1] radii

    # The number of points is not known in advance, so the path is
    # walked twice, first to count the points and then to store them.
    with nogil:
        n = walk_path(phi, radius, phi_max, sma_q, q, None, None, False)

    phis = np.empty(n, dtype=DTYPE)
    radii = np.empty(n, dtype=DTYPE)
    with nogil:
        walk_path(phi, radius, phi_max, sma_q, q, phis, radii, True)

    return np.asarray(phis), np.asarray(radii)
