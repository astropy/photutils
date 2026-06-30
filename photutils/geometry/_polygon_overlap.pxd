# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the polygon overlap functions into other
Cython files. These single-pixel functions are pure C math functions
that are safe to call without the GIL.
"""

cdef double polygon_pixel_overlap(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double *poly_x, double *poly_y,
                                  int n_poly,
                                  double *buf_a_x, double *buf_a_y,
                                  double *buf_b_x, double *buf_b_y,
                                  int buf_size) noexcept nogil
cdef int convex_edge_normals(double *poly_x, double *poly_y, int n_poly,
                             double *edge_nx, double *edge_ny,
                             double *edge_c) noexcept nogil
cdef double convex_polygon_pixel_overlap(double pxmin, double pymin,
                                         double pxmax, double pymax,
                                         double *poly_x, double *poly_y,
                                         int n_poly,
                                         double *edge_nx, double *edge_ny,
                                         double *edge_c, int is_convex,
                                         double margin,
                                         double *buf_a_x, double *buf_a_y,
                                         double *buf_b_x, double *buf_b_y,
                                         int buf_size) noexcept nogil
cdef double polygon_overlap_single_subpixel(double x0, double y0,
                                            double x1, double y1,
                                            double *poly_x, double *poly_y,
                                            int n_poly, int subpixels,
                                            double *xint_buf,
                                            double *hxmin_buf,
                                            double *hxmax_buf) noexcept nogil
