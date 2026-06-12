# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3


cdef double polygon_pixel_overlap(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double *poly_x, double *poly_y,
                                  int n_poly,
                                  double *buf_a_x, double *buf_a_y,
                                  double *buf_b_x, double *buf_b_y,
                                  int buf_size) noexcept nogil


cdef int point_in_polygon(double x, double y, double *poly_x,
                          double *poly_y, int n_poly) noexcept nogil
