# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to calculate the area of overlap between a simple polygon and a
pixel grid.

The functions here are written so that they can be reused by any
aperture whose footprint is a simple polygon (e.g., rotated rectangles,
regular polygons, or arbitrary user-supplied polygons). Convexity is not
required: the subject is the aperture polygon and the clip polygon is
the (always-convex, axis-aligned) pixel rectangle.
"""

import numpy as np

cimport cython
cimport numpy as np


cdef extern from "math.h":
    double fabs(double x) nogil
    double fmax(double x, double y) nogil
    double fmin(double x, double y) nogil
    double floor(double x) nogil
    double ceil(double x) nogil


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline double _polygon_signed_area(double *xs, double *ys,
                                        int n) noexcept nogil:
    """
    Signed area of a polygon via the shoelace formula.

    Positive for counter-clockwise vertices, negative for clockwise
    vertices. For a Sutherland-Hodgman-clipped polygon the magnitude is
    the area of the intersection even when the subject was non-convex
    (any spurious "bridge" edges traverse zero area in pairs).
    """
    cdef double area = 0.0
    cdef int i, j
    if n < 3:
        return 0.0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return 0.5 * area


cdef inline int _clip_against_axis(double *xs_in, double *ys_in, int n_in,
                                   int axis, double bound, int keep_greater,
                                   double *xs_out,
                                   double *ys_out) noexcept nogil:
    """
    Sutherland-Hodgman clip of a polygon (xs_in, ys_in) against an
    axis-aligned half-plane.

    Parameters
    ----------
    axis : int
        0 for an x-aligned bound, 1 for a y-aligned bound.
    bound : double
        The coordinate value of the clipping line.
    keep_greater : int
        1 to keep points with coordinate >= bound, 0 to keep points
        with coordinate <= bound.
    """
    cdef int i, n_out = 0
    cdef double sx, sy, px, py, sc, pc, t
    cdef double sign

    if n_in == 0:
        return 0

    if keep_greater == 1:
        sign = 1.0
    else:
        sign = -1.0

    sx = xs_in[n_in - 1]
    sy = ys_in[n_in - 1]
    if axis == 0:
        sc = sign * (sx - bound)
    else:
        sc = sign * (sy - bound)

    for i in range(n_in):
        px = xs_in[i]
        py = ys_in[i]
        if axis == 0:
            pc = sign * (px - bound)
        else:
            pc = sign * (py - bound)
        if pc >= 0.0:
            if sc < 0.0:
                # entering: emit intersection, then point
                t = sc / (sc - pc)
                xs_out[n_out] = sx + t * (px - sx)
                ys_out[n_out] = sy + t * (py - sy)
                n_out += 1
            xs_out[n_out] = px
            ys_out[n_out] = py
            n_out += 1
        elif sc >= 0.0:
            # leaving: emit intersection only
            t = sc / (sc - pc)
            xs_out[n_out] = sx + t * (px - sx)
            ys_out[n_out] = sy + t * (py - sy)
            n_out += 1
        sx = px
        sy = py
        sc = pc

    return n_out


cdef double polygon_pixel_overlap(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double *poly_x, double *poly_y,
                                  int n_poly,
                                  double *buf_a_x, double *buf_a_y,
                                  double *buf_b_x, double *buf_b_y,
                                  int buf_size) noexcept nogil:
    """
    Exact area of overlap between the axis-aligned pixel rectangle
    [pxmin, pxmax] x [pymin, pymax] and a simple polygon supplied as
    counter-clockwise vertices.

    The polygon (subject) is clipped against the four edges of the pixel
    rectangle (clip) using the Sutherland-Hodgman algorithm, followed
    by the shoelace area formula. Because the clip polygon is convex,
    the subject polygon may be either convex or non-convex (it must,
    however, be simple, i.e., non-self-intersecting).

    The four ``buf_*`` parameters are caller-allocated working buffers;
    each must hold at least ``buf_size`` doubles. The caller is
    responsible for ensuring ``buf_size`` is large enough; each clip can
    roughly double the vertex count of a non-convex polygon, so 16 times
    the maximum input polygon size is a safe choice (see the analysis in
    ``_polygon_overlap.pxd``). For convex input polygons, each clip adds
    at most one vertex, so input size plus four suffices.
    """
    cdef double *cur_x = buf_a_x
    cdef double *cur_y = buf_a_y
    cdef double *nxt_x = buf_b_x
    cdef double *nxt_y = buf_b_y
    cdef double *tmp
    cdef int n

    if n_poly < 3 or n_poly > buf_size:
        return 0.0

    # Clip against x >= pxmin, x <= pxmax, y >= pymin, y <= pymax.
    # The first clip reads directly from the input polygon, avoiding a
    # per-pixel copy of the input vertices.
    n = _clip_against_axis(poly_x, poly_y, n_poly, 0, pxmin, 1,
                           cur_x, cur_y)
    if n == 0:
        return 0.0

    n = _clip_against_axis(cur_x, cur_y, n, 0, pxmax, 0, nxt_x, nxt_y)
    if n == 0:
        return 0.0
    tmp = cur_x; cur_x = nxt_x; nxt_x = tmp
    tmp = cur_y; cur_y = nxt_y; nxt_y = tmp

    n = _clip_against_axis(cur_x, cur_y, n, 1, pymin, 1, nxt_x, nxt_y)
    if n == 0:
        return 0.0
    tmp = cur_x; cur_x = nxt_x; nxt_x = tmp
    tmp = cur_y; cur_y = nxt_y; nxt_y = tmp

    n = _clip_against_axis(cur_x, cur_y, n, 1, pymax, 0, nxt_x, nxt_y)
    if n == 0:
        return 0.0

    return fabs(_polygon_signed_area(nxt_x, nxt_y, n))


cdef int point_in_polygon(double x, double y, double *poly_x,
                          double *poly_y, int n_poly) noexcept nogil:
    """
    Return 1 if ``(x, y)`` lies inside the simple polygon ``(poly_x,
    poly_y)``, 0 otherwise.

    Uses the standard ray-casting algorithm so the polygon may be convex
    or non-convex. Points exactly on an edge may be classified as either
    inside or outside (the usual half-open behavior of ray casting);
    this is adequate for subpixel sampling.
    """
    cdef int i, j, inside = 0
    cdef double xi, yi, xj, yj
    if n_poly < 3:
        return 0
    j = n_poly - 1
    for i in range(n_poly):
        xi = poly_x[i]
        yi = poly_y[i]
        xj = poly_x[j]
        yj = poly_y[j]
        if ((yi > y) != (yj > y)):
            # x of intersection of polygon edge with horizontal line at y
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = 1 - inside
        j = i
    return inside


cdef void _ensure_ccw(double[::1] xs, double[::1] ys, int n,
                      double *out_x, double *out_y) noexcept nogil:
    """
    Copy ``(xs, ys)`` into ``(out_x, out_y)`` ensuring counter-clockwise
    order via the sign of the signed area.
    """
    cdef double area = _polygon_signed_area(&xs[0], &ys[0], n)
    cdef int i, k
    if area >= 0.0:
        for i in range(n):
            out_x[i] = xs[i]
            out_y[i] = ys[i]
    else:
        for i in range(n):
            k = n - 1 - i
            out_x[i] = xs[k]
            out_y[i] = ys[k]


def polygon_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                         int nx, int ny,
                         np.ndarray[DTYPE_t, ndim=1] vertices_x,
                         np.ndarray[DTYPE_t, ndim=1] vertices_y,
                         int use_exact, int subpixels):
    """
    polygon_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, vertices_x,
                         vertices_y, use_exact, subpixels)

    Area of overlap between a simple polygon and a pixel grid.

    The polygon vertices must define a simple polygon (no
    self-intersections). The polygon may be convex or non-convex. The
    vertex order may be either clockwise or counter-clockwise; clockwise
    input is reversed internally.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Extent of the grid in the x and y direction.
    nx, ny : int
        Grid dimensions.
    vertices_x, vertices_y : 1-d `~numpy.ndarray`
        The x and y coordinates of the polygon vertices (in the same
        coordinate frame as the grid extents). The polygon must have at
        least 3 and at most 512 vertices.
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
        2D array giving the fraction of the overlap.
    """
    cdef int n_poly = vertices_x.shape[0]
    if vertices_y.shape[0] != n_poly:
        msg = 'vertices_x and vertices_y must have the same length'
        raise ValueError(msg)
    if n_poly < 3:
        msg = 'polygon must have at least 3 vertices'
        raise ValueError(msg)
    if n_poly > POLYGON_OVERLAP_MAX_INPUT_VERTICES:
        msg = (f'polygon has too many vertices (max '
               f'{POLYGON_OVERLAP_MAX_INPUT_VERTICES})')
        raise ValueError(msg)
    if use_exact != 1 and subpixels < 1:
        msg = 'subpixels must be a strictly positive integer'
        raise ValueError(msg)

    cdef double poly_x[POLYGON_OVERLAP_MAX_INPUT_VERTICES]
    cdef double poly_y[POLYGON_OVERLAP_MAX_INPUT_VERTICES]
    cdef double buf_a_x[POLYGON_OVERLAP_MAX_VERTICES]
    cdef double buf_a_y[POLYGON_OVERLAP_MAX_VERTICES]
    cdef double buf_b_x[POLYGON_OVERLAP_MAX_VERTICES]
    cdef double buf_b_y[POLYGON_OVERLAP_MAX_VERTICES]
    cdef double[::1] vx_view = np.ascontiguousarray(vertices_x)
    cdef double[::1] vy_view = np.ascontiguousarray(vertices_y)
    _ensure_ccw(vx_view, vy_view, n_poly, poly_x, poly_y)

    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac
    cdef double dx = (xmax - xmin) / nx
    cdef double dy = (ymax - ymin) / ny
    cdef double pxmin, pxmax, pymin, pymax
    cdef double bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax
    cdef double pixel_area = dx * dy
    cdef int i, j, ii, jj, k
    cdef int i_min, i_max, j_min, j_max
    cdef double sub_dx, sub_dy, sx, sy, hits

    # Polygon bounding box.
    bbox_xmin = poly_x[0]
    bbox_xmax = poly_x[0]
    bbox_ymin = poly_y[0]
    bbox_ymax = poly_y[0]
    for k in range(1, n_poly):
        bbox_xmin = fmin(bbox_xmin, poly_x[k])
        bbox_xmax = fmax(bbox_xmax, poly_x[k])
        bbox_ymin = fmin(bbox_ymin, poly_y[k])
        bbox_ymax = fmax(bbox_ymax, poly_y[k])

    # Restrict the pixel loops to the bounding-box index range (pixels
    # outside the bounding box have zero overlap). The clamping to
    # [0, nx] and [0, ny] is done in floating point to avoid integer
    # overflow for polygons far outside the grid.
    i_min = <int>fmax(0.0, fmin(<double>nx, floor((bbox_xmin - xmin) / dx)))
    i_max = <int>fmax(0.0, fmin(<double>nx, ceil((bbox_xmax - xmin) / dx)))
    j_min = <int>fmax(0.0, fmin(<double>ny, floor((bbox_ymin - ymin) / dy)))
    j_max = <int>fmax(0.0, fmin(<double>ny, ceil((bbox_ymax - ymin) / dy)))

    if use_exact == 1:
        with nogil:
            for i in range(i_min, i_max):
                pxmin = xmin + i * dx
                pxmax = pxmin + dx
                for j in range(j_min, j_max):
                    pymin = ymin + j * dy
                    pymax = pymin + dy
                    frac_view[j, i] = (
                        polygon_pixel_overlap(pxmin, pymin, pxmax, pymax,
                                              poly_x, poly_y, n_poly,
                                              buf_a_x, buf_a_y,
                                              buf_b_x, buf_b_y,
                                              POLYGON_OVERLAP_MAX_VERTICES)
                        / pixel_area)
    else:
        sub_dx = dx / subpixels
        sub_dy = dy / subpixels
        with nogil:
            for i in range(i_min, i_max):
                pxmin = xmin + i * dx
                for j in range(j_min, j_max):
                    pymin = ymin + j * dy
                    hits = 0.0
                    sx = pxmin + 0.5 * sub_dx
                    for ii in range(subpixels):
                        sy = pymin + 0.5 * sub_dy
                        for jj in range(subpixels):
                            if point_in_polygon(sx, sy, poly_x, poly_y,
                                                n_poly) == 1:
                                hits += 1.0
                            sy += sub_dy
                        sx += sub_dx
                    frac_view[j, i] = hits / (subpixels * subpixels)

    return frac
