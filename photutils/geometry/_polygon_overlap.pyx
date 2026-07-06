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

The cdef functions are not intended to be called from Python code.
They are pure C math functions declared ``noexcept nogil`` so they can
be called without the GIL (e.g., from the batch aperture photometry
driver), including from multiple threads on free-threaded Python builds.
Their signatures are exported via polygon_overlap.pxd.
"""

import numpy as np

cimport numpy as np

__all__ = ['polygon_overlap_grid']


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    double floor(double x)
    double ceil(double x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def polygon_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                         int nx, int ny,
                         np.ndarray[DTYPE_t, ndim=1] vertices_x,
                         np.ndarray[DTYPE_t, ndim=1] vertices_y,
                         int use_exact, int subpixels):
    """
    polygon_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, vertices_x,
                         vertices_y, use_exact, subpixels)

    Calculate the fractional overlap between a simple polygon and a
    pixel grid.

    The polygon vertices must define a simple polygon (no
    self-intersections). The polygon may be convex or non-convex. The
    vertex order may be either clockwise or counter-clockwise; clockwise
    input is reversed internally.

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

    vertices_x, vertices_y : 1D `~numpy.ndarray`
        The x and y coordinates of the polygon vertices (in the same
        coordinate frame as the grid extents). The polygon must have at
        least 3 vertices.

    use_exact : 0 or 1
        Set to ``1`` to use an exact method to calculate the overlap
        between the polygon and each pixel. Set to ``0`` to use a
        sub-pixel sampling method to calculate the overlap, where each
        pixel is divided into ``subpixels ** 2`` subpixels and the
        fraction of subpixels that are within the polygon is used to
        estimate the overlap.

    subpixels : int
        The number of subpixels to use in each dimension when using
        the sub-pixel sampling method. Each pixel is resampled by this
        factor in each dimension; thus, each pixel is divided into
        ``subpixels ** 2`` subpixels.

        With ``subpixels == 1`` (the ``center`` method) a pixel is
        included only if its center falls strictly inside the polygon.
        Pixel centers lying exactly on a polygon edge are counted
        as outside, consistent with the circular, elliptical, and
        rectangular apertures (see ``point_in_polygon``).

        For ``subpixels > 1`` the same convention applies to each
        subpixel: a subpixel is included only if its center lies
        strictly inside the polygon; subpixel centers lying exactly on a
        polygon edge are excluded (weight 0).

    Returns
    -------
    result : `~numpy.ndarray` (float)
        A 2D array of shape (ny, nx) giving the fraction of each
        pixel's area that overlaps with the polygon, ranging from 0 to
        1. The element at index (j, i) corresponds to the pixel with
        corners at (xmin + i * dx, ymin + j * dy) and (xmin + (i + 1)
        * dx, ymin + (j + 1) * dy), where dx and dy are the width of
        each pixel in the x and y direction, respectively.
    """
    cdef int n_poly = vertices_x.shape[0]
    if vertices_y.shape[0] != n_poly:
        msg = 'vertices_x and vertices_y must have the same length'
        raise ValueError(msg)
    if n_poly < 3:
        msg = 'polygon must have at least 3 vertices'
        raise ValueError(msg)
    if use_exact != 1 and subpixels < 1:
        msg = 'subpixels must be a strictly positive integer'
        raise ValueError(msg)

    # Working buffers, allocated once per call (not per pixel) as a
    # single block. Each Sutherland-Hodgman clip can at most double the
    # vertex count of a non-convex polygon, so after the four half-plane
    # clips the vertex count is bounded by 16 * n_poly.
    cdef int buf_size = 16 * n_poly
    cdef double[::1] work = np.empty(2 * n_poly + 4 * buf_size
                                     + 3 * n_poly, dtype=DTYPE)
    cdef double *poly_x = &work[0]
    cdef double *poly_y = &work[n_poly]
    cdef double *buf_a_x = &work[2 * n_poly]
    cdef double *buf_a_y = &work[2 * n_poly + buf_size]
    cdef double *buf_b_x = &work[2 * n_poly + 2 * buf_size]
    cdef double *buf_b_y = &work[2 * n_poly + 3 * buf_size]
    cdef double *edge_nx = &work[2 * n_poly + 4 * buf_size]
    cdef double *edge_ny = &work[3 * n_poly + 4 * buf_size]
    cdef double *edge_c = &work[4 * n_poly + 4 * buf_size]
    cdef const double[::1] vx_view = np.ascontiguousarray(vertices_x)
    cdef const double[::1] vy_view = np.ascontiguousarray(vertices_y)
    _ensure_ccw(vx_view, vy_view, n_poly, poly_x, poly_y)
    cdef int is_convex = convex_edge_normals(poly_x, poly_y, n_poly,
                                             edge_nx, edge_ny, edge_c)

    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac
    cdef double dx = (xmax - xmin) / nx
    cdef double dy = (ymax - ymin) / ny
    cdef double pxmin, pxmax, pymin, pymax
    cdef double bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax
    cdef double pixel_area = dx * dy
    cdef double margin = 0.5 * sqrt(dx * dx + dy * dy)
    cdef int i, j, k
    cdef int i_min, i_max, j_min, j_max

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
                        convex_polygon_pixel_overlap(
                            pxmin, pymin, pxmax, pymax,
                            poly_x, poly_y, n_poly,
                            edge_nx, edge_ny, edge_c, is_convex, margin,
                            buf_a_x, buf_a_y, buf_b_x, buf_b_y, buf_size)
                        / pixel_area)
    else:
        with nogil:
            for i in range(i_min, i_max):
                pxmin = xmin + i * dx
                for j in range(j_min, j_max):
                    pymin = ymin + j * dy
                    frac_view[j, i] = polygon_overlap_single_subpixel(
                        pxmin, pymin, pxmin + dx, pymin + dy,
                        poly_x, poly_y, n_poly, subpixels,
                        buf_a_x, buf_a_y, buf_b_x)

    return frac


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
    the maximum input polygon size is a safe choice. For convex input
    polygons, each clip adds at most one vertex, so input size plus four
    suffices.
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

    tmp = cur_x
    cur_x = nxt_x
    nxt_x = tmp
    tmp = cur_y
    cur_y = nxt_y
    nxt_y = tmp

    n = _clip_against_axis(cur_x, cur_y, n, 1, pymin, 1, nxt_x, nxt_y)
    if n == 0:
        return 0.0

    tmp = cur_x
    cur_x = nxt_x
    nxt_x = tmp
    tmp = cur_y
    cur_y = nxt_y
    nxt_y = tmp

    n = _clip_against_axis(cur_x, cur_y, n, 1, pymax, 0, nxt_x, nxt_y)
    if n == 0:
        return 0.0

    return fabs(_polygon_signed_area(nxt_x, nxt_y, n))


cdef int convex_edge_normals(double *poly_x, double *poly_y, int n_poly,
                             double *edge_nx, double *edge_ny,
                             double *edge_c) noexcept nogil:
    """
    Test whether a counter-clockwise simple polygon is convex and, if
    so, fill the per-edge unit inward normals and offsets.

    For edge ``k`` (from vertex ``k`` to vertex ``k + 1``) of a
    counter-clockwise convex polygon, the inward unit normal is
    ``(-edge_y, edge_x)`` normalized, and the signed distance of a point
    ``(x, y)`` from the edge line is ``edge_nx[k] * x + edge_ny[k] * y -
    edge_c[k]``, positive on the interior side.

    Returns 1 if the polygon is convex (and the three output arrays,
    each of length ``n_poly``, are filled), 0 otherwise. A clockwise or
    degenerate polygon also returns 0, in which case the caller should
    fall back to the general Sutherland-Hodgman clip.
    """
    cdef int k, k1, k2
    cdef double ex, ey, ex2, ey2, cross, length

    if n_poly < 3:
        return 0

    for k in range(n_poly):
        k1 = k + 1
        if k1 == n_poly:
            k1 = 0
        k2 = k1 + 1
        if k2 == n_poly:
            k2 = 0
        ex = poly_x[k1] - poly_x[k]
        ey = poly_y[k1] - poly_y[k]
        ex2 = poly_x[k2] - poly_x[k1]
        ey2 = poly_y[k2] - poly_y[k1]

        # Cross product of consecutive edges. For a counter-clockwise
        # convex polygon every turn is a left turn (cross >= 0); a
        # negative value marks a reflex vertex (non-convex), and a
        # uniformly negative sign marks a clockwise polygon.
        cross = ex * ey2 - ey * ex2
        if cross < 0.0:
            return 0
        length = sqrt(ex * ex + ey * ey)
        if length == 0.0:
            return 0
        edge_nx[k] = -ey / length
        edge_ny[k] = ex / length
        edge_c[k] = edge_nx[k] * poly_x[k] + edge_ny[k] * poly_y[k]

    return 1


cdef double convex_polygon_pixel_overlap(double pxmin, double pymin,
                                         double pxmax, double pymax,
                                         double *poly_x, double *poly_y,
                                         int n_poly,
                                         double *edge_nx, double *edge_ny,
                                         double *edge_c, int is_convex,
                                         double margin,
                                         double *buf_a_x, double *buf_a_y,
                                         double *buf_b_x, double *buf_b_y,
                                         int buf_size) noexcept nogil:
    """
    Exact area of overlap between the pixel rectangle and a simple
    polygon, with an interior/exterior fast path for convex polygons.

    When ``is_convex`` is 1, ``edge_nx``/``edge_ny``/``edge_c``
    hold the per-edge unit inward normals and offsets from
    ``convex_edge_normals``, and ``margin`` is half the pixel diagonal.
    Because each pixel point lies within ``margin`` of the pixel center,
    a pixel is wholly inside the polygon when the center's smallest
    signed edge distance exceeds ``margin`` (overlap = pixel area), and
    wholly outside when it is below ``-margin`` (overlap = 0). Pixels in
    the boundary band, and all pixels of non-convex polygons, fall back
    to the exact Sutherland-Hodgman clip.
    """
    cdef double pxcen, pycen, min_dist, d
    cdef int k

    if is_convex:
        pxcen = 0.5 * (pxmin + pxmax)
        pycen = 0.5 * (pymin + pymax)
        min_dist = edge_nx[0] * pxcen + edge_ny[0] * pycen - edge_c[0]
        for k in range(1, n_poly):
            d = edge_nx[k] * pxcen + edge_ny[k] * pycen - edge_c[k]
            if d < min_dist:
                min_dist = d
        if min_dist > margin:
            return (pxmax - pxmin) * (pymax - pymin)
        if min_dist < -margin:
            return 0.0

    return polygon_pixel_overlap(pxmin, pymin, pxmax, pymax,
                                 poly_x, poly_y, n_poly,
                                 buf_a_x, buf_a_y, buf_b_x, buf_b_y,
                                 buf_size)


cdef int point_in_polygon(double x, double y, double *poly_x,
                          double *poly_y, int n_poly) noexcept nogil:
    """
    Return 1 if ``(x, y)`` lies inside the simple polygon ``(poly_x,
    poly_y)``, 0 otherwise.

    Uses the standard ray-casting algorithm so the polygon may be convex
    or non-convex.

    Boundary convention
    -------------------
    Points lying exactly on a polygon edge are classified as
    *outside* (return 0). This matches the ``center`` method
    of the circular, elliptical, and rectangular apertures,
    which use a strict inequality (``< r**2``, ``< 1``, and ``<
    half_width/half_height``, respectively) and therefore also exclude
    every on-boundary point. In particular, because a rectangle is
    itself a polygon, a `~photutils.aperture.RectangularAperture` and
    a `~photutils.aperture.PolygonAperture` with the same four corners
    produce identical ``center`` masks.

    The on-edge test is performed first: if ``(x, y)`` lies on any edge
    (within the closed bounding box of that edge and collinear with it),
    the point is reported as outside. Otherwise the standard ray-casting
    parity test is applied for strictly interior/exterior points.

    This boundary handling matters mainly for the ``center`` method
    (``subpixels == 1``), where pixel centers commonly fall exactly
    on an edge (e.g., for integer-coordinate polygons). For subpixel
    sampling (``subpixels > 1``) on-edge samples are rare.
    """
    cdef int i, j, inside = 0
    cdef double xi, yi, xj, yj
    cdef double cross

    if n_poly < 3:
        return 0

    j = n_poly - 1
    for i in range(n_poly):
        xi = poly_x[i]
        yi = poly_y[i]
        xj = poly_x[j]
        yj = poly_y[j]

        # Reject points exactly on this edge (collinear and within the
        # edge's closed bounding box), so that all boundary points are
        # classified as outside, consistent with the other apertures.
        cross = (xj - xi) * (y - yi) - (yj - yi) * (x - xi)
        if (cross == 0.0
                and fmin(xi, xj) <= x <= fmax(xi, xj)
                and fmin(yi, yj) <= y <= fmax(yi, yj)):
            return 0

        if ((yi > y) != (yj > y)):
            # x of intersection of polygon edge with horizontal line at y
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = 1 - inside

        j = i

    return inside


cdef double polygon_overlap_single_subpixel(double x0, double y0,
                                            double x1, double y1,
                                            double *poly_x, double *poly_y,
                                            int n_poly, int subpixels,
                                            double *xint_buf,
                                            double *hxmin_buf,
                                            double *hxmax_buf) noexcept nogil:
    """
    Return the fraction of overlap between a simple polygon and a single
    pixel with the given extent, using a sub-pixel sampling method.

    The pixel ``[x0, x1] x [y0, y1]`` is divided into ``subpixels ** 2``
    subpixels and the fraction of subpixel centers that fall inside the
    polygon is returned. The polygon may be convex or non-convex.

    For ``subpixels == 1`` (the ``center`` method) the single
    pixel-center sample is classified with the full `point_in_polygon`
    test, so that samples lying exactly on a polygon edge are excluded,
    consistent with the circular, elliptical, and rectangular apertures.

    For ``subpixels > 1`` the samples are classified with a scanline
    parity fill: for each subpixel row, the polygon boundary restricted
    to the horizontal line at ``y`` is fully characterized once
    (``O(n_poly)``) as a set of crossing points (from sloped and
    vertical edges) plus a set of closed x-intervals (from any
    horizontal edges lying exactly on that line). The crossings are
    sorted and the row's samples are then classified by a single
    monotonic sweep instead of re-testing every edge for every sample.
    This reduces the per-pixel cost from ``O(subpixels ** 2 * n_poly)``
    to roughly ``O(subpixels * n_poly + subpixels ** 2)`` while matching
    the classification of `point_in_polygon`: interior samples are
    classified by the identical ``x < xint`` parity rule, and every
    on-boundary sample is excluded (classified as outside). A sample
    is on the boundary when its x coincides exactly with a crossing (a
    sloped or vertical edge, including vertices, whose crossing x is
    computed exactly because the ``y - yi`` term vanishes) or when it
    lies within the closed x-span of a horizontal edge at that ``y``.
    This makes the on-boundary exclusion bit-exact for the axis-aligned
    and integer-coordinate polygons where on-boundary samples actually
    occur. For samples on the interior of a sloped edge (a rare event
    for subpixel sampling) the recomputed crossing may differ from the
    sample x by rounding and the exclusion is not guaranteed.

    ``xint_buf``, ``hxmin_buf``, and ``hxmax_buf`` are caller-supplied
    scratch buffers, each of which must hold at least ``n_poly``
    doubles. ``xint_buf`` stores the per-row edge crossings, while
    ``hxmin_buf`` and ``hxmax_buf`` store the endpoints of the per-row
    horizontal-edge intervals. They are written only within this call,
    so the function remains thread safe.

    This is a pure C math function declared ``noexcept nogil`` so it can
    be called without the GIL (e.g., from the batch aperture photometry
    driver), including from multiple threads on free-threaded Python
    builds.
    """
    cdef int _i, _j, e, prev, m, a, b, ptr, h, hh
    cdef double x, y, dx, dy, hits
    cdef double xi, yi, xj, yj, tmp
    cdef bint inside

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels
    hits = 0.0

    if subpixels == 1:
        # Center method: a single sample at the pixel center. Use
        # the full point-in-polygon test so that on-edge samples are
        # classified consistently with the other apertures (excluded).
        if point_in_polygon(x0 + 0.5 * dx, y0 + 0.5 * dy,
                            poly_x, poly_y, n_poly) == 1:
            hits = 1.0
        return hits

    # Subpixel sampling (subpixels > 1): scanline parity fill.
    y = y0 + 0.5 * dy
    for _i in range(subpixels):
        # Characterize the polygon boundary on the horizontal line
        # at ``y``. Sloped and vertical edges that straddle the line
        # each contribute one crossing point (``xint_buf``), using
        # the same edge-straddle test and intersection arithmetic
        # as the parity step of ``point_in_polygon``. Horizontal
        # edges lying exactly on the line cannot straddle it,
        # so they are recorded separately as closed x-intervals
        # (``hxmin_buf``/``hxmax_buf``). Together these two sets fully
        # describe the boundary on the line and let every on-boundary
        # sample be excluded.
        m = 0
        h = 0
        prev = n_poly - 1
        for e in range(n_poly):
            yi = poly_y[e]
            yj = poly_y[prev]
            if yi == y and yj == y:
                xi = poly_x[e]
                xj = poly_x[prev]
                hxmin_buf[h] = fmin(xi, xj)
                hxmax_buf[h] = fmax(xi, xj)
                h += 1
            elif (yi > y) != (yj > y):
                xi = poly_x[e]
                xj = poly_x[prev]
                xint_buf[m] = (xj - xi) * (y - yi) / (yj - yi) + xi
                m += 1
            prev = e

        # Insertion sort the (typically few) crossings in ascending x.
        for a in range(1, m):
            tmp = xint_buf[a]
            b = a - 1
            while b >= 0 and xint_buf[b] > tmp:
                xint_buf[b + 1] = xint_buf[b]
                b -= 1
            xint_buf[b + 1] = tmp

        # Sweep the row's samples (monotonically increasing in x). A
        # sample is inside when an odd number of crossings lie strictly
        # to its right (the ``x < xint`` parity rule), unless it lies
        # on the boundary. On-boundary samples are excluded (classified
        # as outside), consistent with ``point_in_polygon``: a sample
        # whose x coincides exactly with a crossing lies on a sloped or
        # vertical edge, and a sample within a horizontal edge's closed
        # x-span lies on that edge.
        x = x0 + 0.5 * dx
        ptr = 0
        for _j in range(subpixels):
            while ptr < m and xint_buf[ptr] < x:
                ptr += 1
            inside = (((m - ptr) & 1) == 1
                      and not (ptr < m and xint_buf[ptr] == x))
            if inside:
                for hh in range(h):
                    if hxmin_buf[hh] <= x <= hxmax_buf[hh]:
                        inside = False
                        break
            if inside:
                hits += 1.0
            x += dx
        y += dy

    return hits / (subpixels * subpixels)


cdef inline double _polygon_signed_area(const double *xs, const double *ys,
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


cdef void _ensure_ccw(const double[::1] xs, const double[::1] ys, int n,
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
