# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3

# Maximum number of vertices in any intermediate clipped polygon. The
# function here clips a simple (possibly non-convex) subject polygon
# against the four axis-aligned half-planes of a pixel rectangle. Each
# half-plane clip increases the number of vertices by at most the
# number of input vertices (each input edge contributes at most one
# intersection in addition to the input vertex it leaves), so the final
# vertex count is bounded by 16 * n_input. With an input limit of 512
# vertices the buffers below are sized to 8192 (~264 KB of stack per
# ``polygon_overlap_grid`` call, well within the default 512 KB pthread
# stack on macOS).
cdef enum:
    POLYGON_OVERLAP_MAX_INPUT_VERTICES = 512
    POLYGON_OVERLAP_MAX_VERTICES = 8192


cdef double polygon_pixel_overlap(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double *poly_x, double *poly_y,
                                  int n_poly,
                                  double *buf_a_x, double *buf_a_y,
                                  double *buf_b_x, double *buf_b_y,
                                  int buf_size) noexcept nogil


cdef int point_in_polygon(double x, double y, double *poly_x,
                          double *poly_y, int n_poly) noexcept nogil
