# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to calculate the area of overlap between a rectangle and a pixel
grid.

The cdef function is not intended to be called from Python code.
It is a pure C math function declared ``noexcept nogil`` so it can
be called without the GIL (e.g., from the batch aperture photometry
driver), including from multiple threads on free-threaded Python builds.
Its signature is exported via rectangular_overlap.pxd.
"""

import numpy as np

cimport numpy as np

from ._polygon_overlap cimport polygon_pixel_overlap

__all__ = ['rectangular_overlap_grid']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    double floor(double x)
    double ceil(double x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def rectangular_overlap_grid(double xmin, double xmax, double ymin,
                             double ymax, int nx, int ny, double width,
                             double height, double theta, int use_exact,
                             int subpixels):
    """
    rectangular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, width, height,
                             theta, use_exact, subpixels)

    Calculate the fractional overlap between a rectangle and a pixel
    grid.

    The rectangle is centered on the origin.

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

    width : float
        The width of the rectangle.

    height : float
        The height of the rectangle.

    theta : float
        The position angle of the rectangle in radians
        (counterclockwise).

    use_exact : 0 or 1
        Set to ``1`` to use an exact method to calculate the overlap
        between the rectangle and each pixel. Set to ``0`` to use a
        sub-pixel sampling method to calculate the overlap, where each
        pixel is divided into ``subpixels ** 2`` subpixels and the
        fraction of subpixels that are within the rectangle is used to
        estimate the overlap.

    subpixels : int
        The number of subpixels to use in each dimension when using
        the sub-pixel sampling method. Each pixel is resampled by this
        factor in each dimension; thus, each pixel is divided into
        ``subpixels ** 2`` subpixels.

    Returns
    -------
    result : `~numpy.ndarray` (float)
        A 2D array of shape (ny, nx) giving the fraction of each
        pixel's area that overlaps with the rectangle, ranging from 0
        to 1. The element at index (j, i) corresponds to the pixel with
        corners at (xmin + i * dx, ymin + j * dy) and (xmin + (i + 1)
        * dx, ymin + (j + 1) * dy), where dx and dy are the width of
        each pixel in the x and y direction, respectively.
    """
    cdef unsigned int i, j
    cdef int i_min, i_max, j_min, j_max
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

        # Axis-aligned bounding box of the rotated rectangle, used to
        # restrict the pixel loops to the bounding-box index range
        # (pixels outside it have zero overlap). The clamping to
        # [0, nx] and [0, ny] is done in floating point to avoid
        # integer overflow for rectangles far outside the grid.
        bbox_dx = (half_width * fabs(cos_theta)
                   + half_height * fabs(sin_theta))
        bbox_dy = (half_width * fabs(sin_theta)
                   + half_height * fabs(cos_theta))
        pixel_area = dx * dy

        i_min = <int>fmax(0.0, fmin(<double>nx,
                                    floor((-bbox_dx - xmin) / dx)))
        i_max = <int>fmax(0.0, fmin(<double>nx,
                                    ceil((bbox_dx - xmin) / dx)))
        j_min = <int>fmax(0.0, fmin(<double>ny,
                                    floor((-bbox_dy - ymin) / dy)))
        j_max = <int>fmax(0.0, fmin(<double>ny,
                                    ceil((bbox_dy - ymin) / dy)))

        with nogil:
            for i in range(i_min, i_max):
                pxmin = xmin + i * dx
                pxmax = pxmin + dx
                for j in range(j_min, j_max):
                    pymin = ymin + j * dy
                    pymax = pymin + dy
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
