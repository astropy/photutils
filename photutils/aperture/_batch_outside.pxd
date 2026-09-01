# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Declarations of the outside-weight scan of the batch aperture
photometry driver (see ``_batch_outside.pyx``).
"""


ctypedef struct _ShapeSpec:
    # The aperture shape parameters of one batch_aperture_sums call.
    # The pointer members reference scratch buffers local to that call.
    int shape_code
    int use_exact
    int subpixels
    double r_in
    double r_out
    double rx_in
    double ry_in
    double rx_out
    double ry_out
    double cos_theta
    double sin_theta
    double half_width_in
    double half_height_in
    double half_width_out
    double half_height_out
    double bbox_dx_in
    double bbox_dy_in
    double bbox_dx_out
    double bbox_dy_out
    double *poly_x_in
    double *poly_y_in
    double *poly_x_out
    double *poly_y_out
    double *buf_a_x
    double *buf_a_y
    double *buf_b_x
    double *buf_b_y
    double *poly_x
    double *poly_y
    int n_poly
    int poly_buf_size
    int is_poly_convex
    double *pedge_nx
    double *pedge_ny
    double *pedge_c
    double *pbuf_a_x
    double *pbuf_a_y
    double *pbuf_b_x
    double *pbuf_b_y


cdef bint outside_weight(const _ShapeSpec *sp, bint inside_any,
                         double gxmin, double gymin, double dx, double dy,
                         double pixel_radius, double norm,
                         Py_ssize_t ixmin, Py_ssize_t iymin,
                         Py_ssize_t ixmax_full, Py_ssize_t iymax_full,
                         Py_ssize_t ix0, Py_ssize_t ix1, Py_ssize_t iy0,
                         Py_ssize_t iy1) noexcept nogil
