# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
This module provides tools to calculate the area of overlap between a
rectangle and a pixel grid.
"""

import numpy as np

cimport numpy as np

from ._polygon_overlap cimport polygon_pixel_overlap

__all__ = ['rectangular_overlap_grid']


cdef extern from "math.h":
    double sin(double x) nogil
    double cos(double x) nogil
    double fabs(double x) nogil


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def rectangular_overlap_grid(double xmin, double xmax, double ymin,
                             double ymax, int nx, int ny, double width,
                             double height, double theta, int use_exact,
                             int subpixels):
    """
    rectangular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, width, height,
                             theta, use_exact, subpixels)

    Area of overlap between a rectangle and a pixel grid. The rectangle
    is centered on the origin.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Extent of the grid in the x and y direction.
    nx, ny : int
        Grid dimensions.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    theta : float
        The position angle of the rectangle in radians (counterclockwise).
    use_exact : 0 or 1
        If set to 1, calculates the exact overlap, while if set to 0,
        uses a subpixel sampling method with ``subpixels`` subpixels in
        each direction.
    subpixels : int
        If ``use_exact`` is 0, each pixel is resampled by this factor in
        each dimension. Each pixel is divided into ``subpixels ** 2``
        subpixels.

    Returns
    -------
    frac : `~numpy.ndarray`
        2-d array giving the fraction of the overlap.
    """
    cdef unsigned int i, j
    cdef double pxmin, pxmax, pymin, pymax
    cdef double dx, dy
    cdef double half_width = 0.5 * width
    cdef double half_height = 0.5 * height
    cdef double cos_theta = cos(theta)
    cdef double sin_theta = sin(theta)
    cdef double pixel_area
    cdef double bbox_dx, bbox_dy
    cdef double poly_x[4]
    cdef double poly_y[4]
    cdef double buf_a_x[32]
    cdef double buf_a_y[32]
    cdef double buf_b_x[32]
    cdef double buf_b_y[32]

    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    if use_exact == 1:
        # Build the four CCW vertices of the rotated rectangle (centered
        # on the origin). Local frame vertices in CCW order are
        # (-w/2, -h/2), ( w/2, -h/2), ( w/2,  h/2), (-w/2,  h/2).
        # Rotation by theta:  x' = x cos t - y sin t,
        #                     y' = x sin t + y cos t.
        poly_x[0] = -half_width * cos_theta - (-half_height) * sin_theta
        poly_y[0] = -half_width * sin_theta + (-half_height) * cos_theta
        poly_x[1] = half_width * cos_theta - (-half_height) * sin_theta
        poly_y[1] = half_width * sin_theta + (-half_height) * cos_theta
        poly_x[2] = half_width * cos_theta - half_height * sin_theta
        poly_y[2] = half_width * sin_theta + half_height * cos_theta
        poly_x[3] = -half_width * cos_theta - half_height * sin_theta
        poly_y[3] = -half_width * sin_theta + half_height * cos_theta

        # Axis-aligned bounding box of the rotated rectangle for
        # fast pixel rejection
        bbox_dx = (half_width * fabs(cos_theta)
                   + half_height * fabs(sin_theta))
        bbox_dy = (half_width * fabs(sin_theta)
                   + half_height * fabs(cos_theta))
        pixel_area = dx * dy

        with nogil:
            for i in range(nx):
                pxmin = xmin + i * dx
                pxmax = pxmin + dx
                if pxmax <= -bbox_dx or pxmin >= bbox_dx:
                    continue
                for j in range(ny):
                    pymin = ymin + j * dy
                    pymax = pymin + dy
                    if pymax <= -bbox_dy or pymin >= bbox_dy:
                        continue
                    frac_view[j, i] = (
                        polygon_pixel_overlap(pxmin, pymin, pxmax,
                                              pymax, poly_x, poly_y, 4,
                                              buf_a_x, buf_a_y,
                                              buf_b_x, buf_b_y, 32)
                        / pixel_area)
        return frac

    # Subpixel-sampling fallback
    with nogil:
        for i in range(nx):
            pxmin = xmin + i * dx
            pxmax = pxmin + dx
            for j in range(ny):
                pymin = ymin + j * dy
                pymax = pymin + dy
                frac_view[j, i] = rectangular_overlap_single_subpixel(
                    pxmin, pymin, pxmax, pymax, half_width, half_height,
                    cos_theta, sin_theta, subpixels)

    return frac


cdef double rectangular_overlap_single_subpixel(double x0, double y0,
                                                double x1, double y1,
                                                double half_width,
                                                double half_height,
                                                double cos_theta,
                                                double sin_theta,
                                                int subpixels) noexcept nogil:
    """
    Return the fraction of overlap between a rectangle and a single
    pixel with given extent, using a sub-pixel sampling method.
    """
    cdef unsigned int i, j
    cdef double x, y, x_tr, y_tr
    cdef double frac = 0.0
    cdef double dx = (x1 - x0) / subpixels
    cdef double dy = (y1 - y0) / subpixels

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy
            x_tr = y * sin_theta + x * cos_theta
            y_tr = y * cos_theta - x * sin_theta
            if fabs(x_tr) < half_width and fabs(y_tr) < half_height:
                frac += 1.0

    return frac / (subpixels * subpixels)
