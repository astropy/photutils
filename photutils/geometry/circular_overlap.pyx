# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
The functions defined here allow one to determine the exact area of
overlap of a rectangle and a circle (written by Thomas Robitaille).
"""

import numpy as np
cimport numpy as np

__all__ = ['circular_overlap_grid']


cdef extern from "math.h":

    double asin(double x)
    double sin(double x)
    double sqrt(double x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# NOTE: Here we need to make sure we use cimport to import the C functions from
# core (since these were defined with cdef). This also requires the core.pxd
# file to exist with the function signatures.
from .core cimport area_arc, area_triangle, floor_sqrt


def circular_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                          int nx, int ny, double r, int use_exact,
                          int subpixels):
    """
    circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r,
                             use_exact, subpixels)

    Area of overlap between a circle and a pixel grid. The circle is centered
    on the origin.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Extent of the grid in the x and y direction.
    nx, ny : int
        Grid dimensions.
    r : float
        The radius of the circle.
    use_exact : 0 or 1
        If ``1`` calculates exact overlap, if ``0`` uses ``subpixel`` number
        of subpixels to calculate the overlap.
    subpixels : int
        Each pixel resampled by this factor in each dimension, thus each
        pixel is divided into ``subpixels ** 2`` subpixels.

    Returns
    -------
    frac : `~numpy.ndarray` (float)
        2-d array of shape (ny, nx) giving the fraction of the overlap.
    """

    cdef unsigned int i, j
    cdef double x, y, dx, dy, d, pixel_radius
    cdef double bxmin, bxmax, bymin, bymax
    cdef double pxmin, pxcen, pxmax, pymin, pycen, pymax

    # Define output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)

    # Find the width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Find the radius of a single pixel
    pixel_radius = 0.5 * sqrt(dx * dx + dy * dy)

    # Define bounding box
    bxmin = -r - 0.5 * dx
    bxmax = +r + 0.5 * dx
    bymin = -r - 0.5 * dy
    bymax = +r + 0.5 * dy

    for i in range(nx):
        pxmin = xmin + i * dx  # lower end of pixel
        pxcen = pxmin + dx * 0.5
        pxmax = pxmin + dx  # upper end of pixel
        if pxmax > bxmin and pxmin < bxmax:
            for j in range(ny):
                pymin = ymin + j * dy
                pycen = pymin + dy * 0.5
                pymax = pymin + dy
                if pymax > bymin and pymin < bymax:

                    # Distance from circle center to pixel center.
                    d = sqrt(pxcen * pxcen + pycen * pycen)

                    # If pixel center is "well within" circle, count full
                    # pixel.
                    if d < r - pixel_radius:
                        frac[j, i] = 1.

                    # If pixel center is "close" to circle border, find
                    # overlap.
                    elif d < r + pixel_radius:

                        # Either do exact calculation or use subpixel
                        # sampling:
                        if use_exact:
                            frac[j, i] = circular_overlap_single_exact(
                                pxmin, pymin, pxmax, pymax, r) / (dx * dy)
                        else:
                            frac[j, i] = circular_overlap_single_subpixel(
                                pxmin, pymin, pxmax, pymax, r, subpixels)

                    # Otherwise, it is fully outside circle.
                    # No action needed.

    return frac


# NOTE: The following two functions use cdef because they are not
# intended to be called from the Python code. Using def makes them
# callable from outside, but also slower. In any case, these aren't useful
# to call from outside because they only operate on a single pixel.


cdef double circular_overlap_single_subpixel(double x0, double y0,
                                             double x1, double y1,
                                             double r, int subpixels):
    """Return the fraction of overlap between a circle and a single pixel
    with given extent, using a sub-pixel sampling method."""

    cdef unsigned int i, j
    cdef double x, y, dx, dy, r_squared
    cdef double frac = 0.  # Accumulator.

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels
    r_squared = r ** 2

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy
            if x * x + y * y < r_squared:
                frac += 1.

    return frac / (subpixels * subpixels)


cdef double circular_overlap_single_exact(double xmin, double ymin,
                                          double xmax, double ymax,
                                          double r):
    """
    Area of overlap of a rectangle and a circle
    """
    if 0. <= xmin:
        if 0. <= ymin:
            return circular_overlap_core(xmin, ymin, xmax, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-ymax, xmin, -ymin, xmax, r)
        else:
            return circular_overlap_single_exact(xmin, ymin, xmax, 0., r) \
                + circular_overlap_single_exact(xmin, 0., xmax, ymax, r)
    elif 0. >= xmax:
        if 0. <= ymin:
            return circular_overlap_core(-xmax, ymin, -xmin, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        else:
            return circular_overlap_single_exact(xmin, ymin, xmax, 0., r) \
                + circular_overlap_single_exact(xmin, 0., xmax, ymax, r)
    else:
        if 0. <= ymin:
            return circular_overlap_single_exact(xmin, ymin, 0., ymax, r) \
                + circular_overlap_single_exact(0., ymin, xmax, ymax, r)
        if 0. >= ymax:
            return circular_overlap_single_exact(xmin, ymin, 0., ymax, r) \
                + circular_overlap_single_exact(0., ymin, xmax, ymax, r)
        else:
            return circular_overlap_single_exact(xmin, ymin, 0., 0., r) \
                + circular_overlap_single_exact(0., ymin, xmax, 0., r) \
                + circular_overlap_single_exact(xmin, 0., 0., ymax, r) \
                + circular_overlap_single_exact(0., 0., xmax, ymax, r)


cdef double circular_overlap_core(double xmin, double ymin, double xmax, double ymax,
                          double r):
    """
    Assumes that the center of the circle is <= xmin,
    ymin (can always modify input to conform to this).
    """

    cdef double area, d1, d2, x1, x2, y1, y2

    if xmin * xmin + ymin * ymin > r * r:
        area = 0.
    elif xmax * xmax + ymax * ymax < r * r:
        area = (xmax - xmin) * (ymax - ymin)
    else:
        area = 0.
        d1 = floor_sqrt(xmax * xmax + ymin * ymin)
        d2 = floor_sqrt(xmin * xmin + ymax * ymax)
        if d1 < r and d2 < r:
            x1, y1 = floor_sqrt(r * r - ymax * ymax), ymax
            x2, y2 = xmax, floor_sqrt(r * r - xmax * xmax)
            area = ((xmax - xmin) * (ymax - ymin) -
                    area_triangle(x1, y1, x2, y2, xmax, ymax) +
                    area_arc(x1, y1, x2, y2, r))
        elif d1 < r:
            x1, y1 = xmin, floor_sqrt(r * r - xmin * xmin)
            x2, y2 = xmax, floor_sqrt(r * r - xmax * xmax)
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, x1, ymin, xmax, ymin) +
                    area_triangle(x1, y1, x2, ymin, x2, y2))
        elif d2 < r:
            x1, y1 = floor_sqrt(r * r - ymin * ymin), ymin
            x2, y2 = floor_sqrt(r * r - ymax * ymax), ymax
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, xmin, y1, xmin, ymax) +
                    area_triangle(x1, y1, xmin, y2, x2, y2))
        else:
            x1, y1 = floor_sqrt(r * r - ymin * ymin), ymin
            x2, y2 = xmin, floor_sqrt(r * r - xmin * xmin)
            area = (area_arc(x1, y1, x2, y2, r) +
                    area_triangle(x1, y1, x2, y2, xmin, ymin))

    return area
