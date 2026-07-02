# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the rectangle overlap function into other
Cython files. This single-pixel function is a pure C math function that
is safe to call without the GIL.
"""

cdef double rectangle_overlap_single_subpixel(double x0, double y0,
                                              double x1, double y1,
                                              double half_width,
                                              double half_height,
                                              double cos_theta,
                                              double sin_theta,
                                              int subpixels) noexcept nogil


cdef inline void rect_vertices(double half_width, double half_height,
                               double cos_theta, double sin_theta,
                               double *poly_x, double *poly_y) noexcept nogil:
    """
    Build the four CCW vertices of a rotated rectangle centered on the
    origin. Shared by ``rectangular_overlap_grid`` and the batch
    rectangle helpers so the vertex arithmetic lives in one place.

    Local-frame vertices in CCW order are (-w/2, -h/2), (w/2, -h/2),
    (w/2, h/2), (-w/2, h/2), rotated by theta:
    ``x' = x cos t - y sin t``, ``y' = x sin t + y cos t``.
    """
    poly_x[0] = -half_width * cos_theta - (-half_height) * sin_theta
    poly_y[0] = -half_width * sin_theta + (-half_height) * cos_theta
    poly_x[1] = half_width * cos_theta - (-half_height) * sin_theta
    poly_y[1] = half_width * sin_theta + (-half_height) * cos_theta
    poly_x[2] = half_width * cos_theta - half_height * sin_theta
    poly_y[2] = half_width * sin_theta + half_height * cos_theta
    poly_x[3] = -half_width * cos_theta - half_height * sin_theta
    poly_y[3] = -half_width * sin_theta + half_height * cos_theta
