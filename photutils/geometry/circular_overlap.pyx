# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The functions defined here allow one to determine the exact area of
# overlap of a rectangle and a circle (written by Thomas Robitaille).

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
cimport numpy as np

cdef extern from "math.h":

    double asin(double x)
    double sin(double x)
    double sqrt(double x)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def distance(double x1, double y1, double x2, double y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def area_arc(double x1, double y1, double x2, double y2, double R):
    """Area of a circle arc with radius R between points (x1, y1) and (x2, y2).

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """

    cdef double a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2. * asin(0.5 * a / R)
    return 0.5 * R * R * (theta - sin(theta))


def area_triangle(double x1, double y1, double x2, double y2, double x3,
                  double y3):
    """Area of a triangle defined by three vertices.
    """
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def circular_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                          int nx, int ny, double R, int use_exact,
                          int subpixels):
    """For a circle of radius R, find the area of overlap in each element on
    a given grid of pixels, using either an exact overlap method, or by
    subsampling a pixel."""

    cdef unsigned int i, j
    cdef double x, y, dx, dy, d, pixrad, xlim0, xlim1, ylim0, ylim1

    # Output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)

    # Width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Define these here to speed computation below
    pixrad = 0.5 * sqrt(dx * dx + dy * dy)  # Radius of a single pixel
    xlim0 = -R - 0.5 * dx                   # Extent of circle + half pixel
    xlim1 = R + 0.5 * dx                    # ...
    ylim0 = -R - 0.5 * dy                   # ...
    ylim1 = R + 0.5 * dy                    # ...

    for i in range(nx):
        x = xmin + (i + 0.5) * dx  # x coordinate of pixel center
        if x > xlim0 and x < xlim1:
            for j in range(ny):
                y = ymin + (j + 0.5) * dy  # y coordinate of pixel center
                if y > ylim0 and y < ylim1:

                    # Distance from circle center to pixel center.
                    d = sqrt(x * x + y * y)

                    # If pixel center is "well within" circle, count full pixel.
                    if d < R - pixrad:
                        frac[j, i] = 1.

                    # If pixel center is "close" to circle border, find overlap.
                    elif d < R + pixrad:

                        # Either do exact calculation...
                        if use_exact:
                            frac[j, i] = overlap_single_exact(x - 0.5 * dx, \
                                y - 0.5 * dy, x + 0.5 * dx, y + 0.5 * dy, R) \
                                / (dx * dy)

                        # or use subpixel samping.
                        else:
                            frac[j, i] = overlap_single_subpixel(x - 0.5 * dx, \
                                y - 0.5 * dy, x + 0.5 * dx, y + 0.5 * dy, R,
                                subpixels)

                        # Otherwise, it is fully outside circle.
                        # No action needed.

    return frac


def overlap_single_subpixel(double x0, double y0, double x1, double y1,
                            double R, int subpixels):
    """Return the fraction of overlap between a circle and a single pixel
    with given extent, using a sub-pixel sampling method."""

    cdef unsigned int i, j
    cdef double x, y, dx, dy, R_squared
    cdef double frac = 0.  # Accumulator.

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels
    R_squared = R ** 2

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy
            if x * x + y * y < R_squared:
                frac += 1.

    return frac / (subpixels * subpixels)


def overlap_single_exact(double xmin, double ymin, double xmax, double ymax,
                         double r):
    '''
    Area of overlap of a rectangle and a circle
    '''
    if 0. <= xmin:
        if 0. <= ymin:
            return circular_overlap_core(xmin, ymin, xmax, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-ymax, xmin, -ymin, xmax, r)
        else:
            return overlap_single_exact(xmin, ymin, xmax, 0., r) \
                 + overlap_single_exact(xmin, 0., xmax, ymax, r)
    elif 0. >= xmax:
        if 0. <= ymin:
            return circular_overlap_core(-xmax, ymin, -xmin, ymax, r)
        elif 0. >= ymax:
            return circular_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        else:
            return overlap_single_exact(xmin, ymin, xmax, 0., r) \
                 + overlap_single_exact(xmin, 0., xmax, ymax, r)
    else:
        if 0. <= ymin:
            return overlap_single_exact(xmin, ymin, 0., ymax, r) \
                 + overlap_single_exact(0., ymin, xmax, ymax, r)
        if 0. >= ymax:
            return overlap_single_exact(xmin, ymin, 0., ymax, r) \
                 + overlap_single_exact(0., ymin, xmax, ymax, r)
        else:
            return overlap_single_exact(xmin, ymin, 0., 0., r) \
                 + overlap_single_exact(0., ymin, xmax, 0., r) \
                 + overlap_single_exact(xmin, 0., 0., ymax, r) \
                 + overlap_single_exact(0., 0., xmax, ymax, r)


def circular_overlap_core(double xmin, double ymin, double xmax, double ymax,
                          double R):
    """Assumes that the center of the circle is <= xmin,
    ymin (can always modify input to conform to this).
    """

    cdef double area, d1, d2, x1, x2, y1, y2

    if xmin * xmin + ymin * ymin > R * R:
        area = 0.
    elif xmax * xmax + ymax * ymax < R * R:
        area = (xmax - xmin) * (ymax - ymin)
    else:
        area = 0.
        d1 = sqrt(xmax * xmax + ymin * ymin)
        d2 = sqrt(xmin * xmin + ymax * ymax)
        if d1 < R and d2 < R:
            x1, y1 = sqrt(R * R - ymax * ymax), ymax
            x2, y2 = xmax, sqrt(R * R - xmax * xmax)
            area = (xmax - xmin) * (ymax - ymin) - area_triangle(x1, y1, x2, y2, xmax, ymax) + area_arc(x1, y1, x2, y2, R)
        elif d1 < R:
            x1, y1 = xmin, sqrt(R * R - xmin * xmin)
            x2, y2 = xmax, sqrt(R * R - xmax * xmax)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x1, ymin, xmax, ymin) + area_triangle(x1, y1, x2, ymin, x2, y2)
        elif d2 < R:
            x1, y1 = sqrt(R * R - ymin * ymin), ymin
            x2, y2 = sqrt(R * R - ymax * ymax), ymax
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, xmin, y1, xmin, ymax) + area_triangle(x1, y1, xmin, y2, x2, y2)
        else:
            x1, y1 = sqrt(R * R - ymin * ymin), ymin
            x2, y2 = xmin, sqrt(R * R - xmin * xmin)
            area = area_arc(x1, y1, x2, y2, R) + area_triangle(x1, y1, x2, y2, xmin, ymin)

    return area
