# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
The functions defined here allow one to determine the exact area of
overlap of an ellipse and a triangle (written by Thomas Robitaille).
The approach is to divide the rectangle into two triangles, and
reproject these so that the ellipse is a unit circle, then compute the
intersection of a triangle with a unit circle.
"""

import numpy as np
cimport numpy as np

__all__ = ['elliptical_overlap_grid']


cdef extern from "math.h":

    double asin(double x)
    double sin(double x)
    double cos(double x)
    double sqrt(double x)

from cpython cimport bool

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

# NOTE: Here we need to make sure we use cimport to import the C functions from
# core (since these were defined with cdef). This also requires the core.pxd
# file to exist with the function signatures.
from .core cimport distance, area_triangle, overlap_area_triangle_unit_circle


def elliptical_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                            int nx, int ny, double rx, double ry, double theta,
                            int use_exact, int subpixels):
    """
    elliptical_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, rx, ry,
                             use_exact, subpixels)

    Area of overlap between an ellipse and a pixel grid. The ellipse is
    centered on the origin.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Extent of the grid in the x and y direction.
    nx, ny : int
        Grid dimensions.
    rx : float
        The semimajor axis of the ellipse.
    ry : float
        The semiminor axis of the ellipse.
    theta : float
        The position angle of the semimajor axis in radians (counterclockwise).
    use_exact : 0 or 1
        If set to 1, calculates the exact overlap, while if set to 0, uses a
        subpixel sampling method with ``subpixel`` subpixels in each direction.
    subpixels : int
        If ``use_exact`` is 0, each pixel is resampled by this factor in each
        dimension. Thus, each pixel is divided into ``subpixels ** 2``
        subpixels.

    Returns
    -------
    frac : `~numpy.ndarray`
        2-d array giving the fraction of the overlap.
    """

    cdef unsigned int i, j
    cdef double x, y, dx, dy
    cdef double bxmin, bxmax, bymin, bymax
    cdef double pxmin, pxmax, pymin, pymax
    cdef double norm

    # Define output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)

    # Find the width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    norm = 1. / (dx * dy)

    # For now we use a bounding circle and then use that to find a bounding box
    # but of course this is inefficient and could be done better.

    # Find bounding circle radius
    r = max(rx, ry)

    # Define bounding box
    bxmin = -r - 0.5 * dx
    bxmax = +r + 0.5 * dx
    bymin = -r - 0.5 * dy
    bymax = +r + 0.5 * dy

    for i in range(nx):
        pxmin = xmin + i * dx  # lower end of pixel
        pxmax = pxmin + dx  # upper end of pixel
        if pxmax > bxmin and pxmin < bxmax:
            for j in range(ny):
                pymin = ymin + j * dy
                pymax = pymin + dy
                if pymax > bymin and pymin < bymax:
                    if use_exact:
                        frac[j, i] = elliptical_overlap_single_exact(
                            pxmin, pymin, pxmax, pymax, rx, ry, theta) * norm
                    else:
                        frac[j, i] = elliptical_overlap_single_subpixel(
                            pxmin, pymin, pxmax, pymax, rx, ry, theta,
                            subpixels)
    return frac


# NOTE: The following two functions use cdef because they are not
# intended to be called from the Python code. Using def makes them
# callable from outside, but also slower. In any case, these aren't useful
# to call from outside because they only operate on a single pixel.


cdef double elliptical_overlap_single_subpixel(double x0, double y0,
                                               double x1, double y1,
                                               double rx, double ry,
                                               double theta, int subpixels):
    """
    Return the fraction of overlap between a ellipse and a single pixel with
    given extent, using a sub-pixel sampling method.
    """

    cdef unsigned int i, j
    cdef double x, y
    cdef double frac = 0.  # Accumulator.
    cdef double inv_rx_sq, inv_ry_sq
    cdef double cos_theta = cos(theta)
    cdef double sin_theta = sin(theta)
    cdef double dx, dy
    cdef double x_tr, y_tr

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels

    inv_rx_sq = 1. / (rx * rx)
    inv_ry_sq = 1. / (ry * ry)

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy

            # Transform into frame of rotated ellipse
            x_tr = y * sin_theta + x * cos_theta
            y_tr = y * cos_theta - x * sin_theta

            if x_tr * x_tr * inv_rx_sq + y_tr * y_tr * inv_ry_sq < 1.:
                frac += 1.

    return frac / (subpixels * subpixels)


cdef double elliptical_overlap_single_exact(double xmin, double ymin,
                                            double xmax, double ymax,
                                            double rx, double ry,
                                            double theta):
    """
    Given a rectangle defined by (xmin, ymin, xmax, ymax) and an ellipse
    with major and minor axes rx and ry respectively, position angle theta,
    and centered at the origin, find the area of overlap.
    """

    cdef double cos_m_theta = cos(-theta)
    cdef double sin_m_theta = sin(-theta)
    cdef double scale

    # Find scale by which the areas will be shrunk
    scale = rx * ry

    # Reproject rectangle to frame of reference in which ellipse is a
    # unit circle
    x1, y1 = ((xmin * cos_m_theta - ymin * sin_m_theta) / rx,
              (xmin * sin_m_theta + ymin * cos_m_theta) / ry)
    x2, y2 = ((xmax * cos_m_theta - ymin * sin_m_theta) / rx,
              (xmax * sin_m_theta + ymin * cos_m_theta) / ry)
    x3, y3 = ((xmax * cos_m_theta - ymax * sin_m_theta) / rx,
              (xmax * sin_m_theta + ymax * cos_m_theta) / ry)
    x4, y4 = ((xmin * cos_m_theta - ymax * sin_m_theta) / rx,
              (xmin * sin_m_theta + ymax * cos_m_theta) / ry)

    # Divide resulting quadrilateral into two triangles and find
    # intersection with unit circle
    return (overlap_area_triangle_unit_circle(x1, y1, x2, y2, x3, y3) +
            overlap_area_triangle_unit_circle(x1, y1, x4, y4, x3, y3)) * scale
