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


cdef inline Py_ssize_t polygon_work_size(int n_poly,
                                         int *buf_size) noexcept nogil:
    """
    Total number of doubles needed for a polygon overlap workspace.

    The workspace holds the polygon vertices, the four Sutherland-
    Hodgman clip buffers, and the three edge-normal arrays used by the
    convex fast path (see ``polygon_work_partition`` for the layout).
    Each Sutherland-Hodgman clip can at most double the vertex count of
    a non-convex polygon, so after the four half-plane clips the vertex
    count is bounded by ``16 * n_poly``.

    Parameters
    ----------
    n_poly : int
        The number of polygon vertices.

    buf_size : int *
        Output. The per-buffer size (``16 * n_poly``) used for each of
        the four Sutherland-Hodgman clip buffers.

    Returns
    -------
    total : Py_ssize_t
        The total number of doubles needed for the workspace.
    """
    buf_size[0] = 16 * n_poly
    return 2 * n_poly + 4 * buf_size[0] + 3 * n_poly


cdef inline void polygon_work_partition(double *base, int n_poly,
                                        int buf_size,
                                        double **poly_x, double **poly_y,
                                        double **buf_a_x, double **buf_a_y,
                                        double **buf_b_x, double **buf_b_y,
                                        double **edge_nx, double **edge_ny,
                                        double **edge_c) noexcept nogil:
    """
    Partition a polygon overlap workspace into its component arrays.

    Centralizing the offset arithmetic here keeps the buffer layout
    consistent wherever it is used. This runs once per call (not per
    pixel), so the pointer-output form has no effect on the per-pixel
    loops.

    Parameters
    ----------
    base : double *
        The workspace buffer, of at least ``polygon_work_size(n_poly)``
        doubles. The layout is the polygon vertex arrays (``n_poly``
        each), the four clip buffers (``buf_size`` each), and the three
        edge-normal arrays (``n_poly`` each).

    n_poly : int
        The number of polygon vertices.

    buf_size : int
        The size of each of the four clip buffers (see
        ``polygon_work_size``).

    poly_x, poly_y : double **
        Output. Pointers into ``base`` for the polygon vertex arrays.

    buf_a_x, buf_a_y, buf_b_x, buf_b_y : double **
        Output. Pointers into ``base`` for the four Sutherland-Hodgman
        clip buffers.

    edge_nx, edge_ny, edge_c : double **
        Output. Pointers into ``base`` for the three convex-fast-path
        edge-normal arrays.
    """
    poly_x[0] = base
    poly_y[0] = base + n_poly
    buf_a_x[0] = base + 2 * n_poly
    buf_a_y[0] = base + 2 * n_poly + buf_size
    buf_b_x[0] = base + 2 * n_poly + 2 * buf_size
    buf_b_y[0] = base + 2 * n_poly + 3 * buf_size
    edge_nx[0] = base + 2 * n_poly + 4 * buf_size
    edge_ny[0] = base + 3 * n_poly + 4 * buf_size
    edge_c[0] = base + 4 * n_poly + 4 * buf_size
