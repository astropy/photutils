# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3

# This file is needed in order to be able to cimport functions into
# other Cython files. These single-pixel functions are pure C math
# functions that are safe to call without the GIL.

cdef double polygon_pixel_overlap(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double *poly_x, double *poly_y,
                                  int n_poly,
                                  double *buf_a_x, double *buf_a_y,
                                  double *buf_b_x, double *buf_b_y,
                                  int buf_size) noexcept nogil


cdef int point_in_polygon(double x, double y, double *poly_x,
                          double *poly_y, int n_poly) noexcept nogil
