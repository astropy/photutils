# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to calculate the area of overlap between a circle and a pixel
grid.

The cdef functions are not intended to be called from Python code.
They are pure C math functions declared ``noexcept nogil`` so they can
be called without the GIL (e.g., from the batch aperture photometry
driver), including from multiple threads on free-threaded Python builds.
Their signatures are exported via circle_overlap.pxd.

NOTE: The ``circular_overlap_grid`` function should be named
``circle_overlap_grid``, but it has been public for a long time and
changing the name would break backwards compatibility.
"""

import numpy as np

cimport numpy as np

from .core cimport area_arc, area_triangle, floor_sqrt

__all__ = ['circular_overlap_grid']


cdef extern from "math.h" nogil:
    double sqrt(double x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def circular_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                          int nx, int ny, double r, int use_exact,
                          int subpixels):
    """
    circle_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r, use_exact,
                        subpixels)

    Calculate the fractional overlap between a circle and a pixel grid.

    The circle is centered on the origin.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        The extent of the grid in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax).

    nx, ny : int
        The grid dimensions in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax) and is divided into nx and ny pixels in the x and y
        direction, respectively.

    r : float
        The radius of the circle.

    use_exact : 0 or 1
        Set to ``1`` to use an exact method to calculate the overlap
        between the circle and each pixel. Set to ``0`` to use a
        sub-pixel sampling method to calculate the overlap, where each
        pixel is divided into ``subpixels ** 2`` subpixels and the
        fraction of subpixels that are within the circle is used to
        estimate the overlap.

    subpixels : int
        The number of subpixels to use in each dimension when using
        the sub-pixel sampling method. Each pixel is resampled by this
        factor in each dimension; thus, each pixel is divided into
        ``subpixels ** 2`` subpixels.

        A subpixel is included only if its center lies strictly inside
        the circle; subpixel centers lying exactly on the circle
        boundary are excluded (weight 0).

    Returns
    -------
    result : `~numpy.ndarray` (float)
        A 2D array of shape (ny, nx) giving the fraction of each
        pixel's area that overlaps with the circle, ranging from 0 to
        1. The element at index (j, i) corresponds to the pixel with
        corners at (xmin + i * dx, ymin + j * dy) and (xmin + (i + 1)
        * dx, ymin + (j + 1) * dy), where dx and dy are the width of
        each pixel in the x and y direction, respectively.
    """
    cdef unsigned int i, j
    cdef double dx, dy, d, pixel_radius
    cdef double bxmin, bxmax, bymin, bymax
    cdef double pxmin, pxcen, pxmax, pymin, pycen, pymax

    # Define output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac

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

    with nogil:
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

                        # If pixel center is "well within" circle,
                        # count full pixel.
                        if d < r - pixel_radius:
                            frac_view[j, i] = 1.0

                        # If pixel center is "close" to circle border,
                        # find overlap.
                        elif d < r + pixel_radius:
                            # Either do exact calculation or use
                            # subpixel sampling:
                            if use_exact:
                                frac_view[j, i] = (
                                    circle_overlap_single_exact(
                                        pxmin, pymin, pxmax, pymax, r)
                                    / (dx * dy))
                            else:
                                frac_view[j, i] = (
                                    circle_overlap_single_subpixel(
                                        pxmin, pymin, pxmax, pymax, r,
                                        subpixels))

                        # Otherwise, it is fully outside circle.
                        # No action needed.

    return frac


cdef double circle_overlap_single_subpixel(double x0, double y0,
                                           double x1, double y1,
                                           double r,
                                           int subpixels) noexcept nogil:
    """
    Return the fraction of overlap between a circle and a single pixel
    with given extent, using a sub-pixel sampling method.
    """
    cdef unsigned int _i, _j
    cdef double x, y, dx, dy, r_squared
    cdef double frac = 0.0  # accumulator

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels
    r_squared = r ** 2

    x = x0 - 0.5 * dx
    for _i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for _j in range(subpixels):
            y += dy
            if x * x + y * y < r_squared:
                frac += 1.0

    return frac / (subpixels * subpixels)


cdef double circle_overlap_single_exact(double xmin, double ymin,
                                        double xmax, double ymax,
                                        double r) noexcept nogil:
    """
    Calculate the area of overlap between a circle and a single pixel
    with given extent, using an exact method.

    The circle is centered on the origin.
    """
    if 0.0 <= xmin:
        if 0.0 <= ymin:
            return circle_overlap_core(xmin, ymin, xmax, ymax, r)
        elif 0.0 >= ymax:
            return circle_overlap_core(-ymax, xmin, -ymin, xmax, r)
        else:
            return circle_overlap_single_exact(xmin, ymin, xmax, 0.0, r) \
                + circle_overlap_single_exact(xmin, 0.0, xmax, ymax, r)
    elif 0.0 >= xmax:
        if 0.0 <= ymin:
            return circle_overlap_core(-xmax, ymin, -xmin, ymax, r)
        elif 0.0 >= ymax:
            return circle_overlap_core(-xmax, -ymax, -xmin, -ymin, r)
        else:
            return circle_overlap_single_exact(xmin, ymin, xmax, 0.0, r) \
                + circle_overlap_single_exact(xmin, 0.0, xmax, ymax, r)
    else:
        if 0.0 <= ymin or 0.0 >= ymax:
            return circle_overlap_single_exact(xmin, ymin, 0.0, ymax, r) \
                + circle_overlap_single_exact(0.0, ymin, xmax, ymax, r)
        else:
            return circle_overlap_single_exact(xmin, ymin, 0.0, 0.0, r) \
                + circle_overlap_single_exact(0.0, ymin, xmax, 0.0, r) \
                + circle_overlap_single_exact(xmin, 0.0, 0.0, ymax, r) \
                + circle_overlap_single_exact(0.0, 0.0, xmax, ymax, r)


cdef double circle_overlap_core(double xmin, double ymin, double xmax,
                                double ymax, double r) noexcept nogil:
    """
    Calculate the area of overlap between a circle and a rectangle,
    where the rectangle is in the first quadrant and the circle is
    centered on the origin.

    Assumes that the center of the circle is <= xmin, ymin (can always
    modify input to conform to this).
    """
    cdef double area, d1, d2, x1, x2, y1, y2

    if xmin * xmin + ymin * ymin > r * r:
        area = 0.0
    elif xmax * xmax + ymax * ymax < r * r:
        area = (xmax - xmin) * (ymax - ymin)
    else:
        area = 0.0
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
