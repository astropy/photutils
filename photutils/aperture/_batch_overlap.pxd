# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Shared per-pixel aperture overlap helpers for the batch Cython drivers.

Each helper returns the fraction of a single pixel that overlaps an
aperture shape centered on the origin, using exactly the same arithmetic
as the corresponding `photutils.geometry` grid function (including the
bounding-box and interior/exterior fast paths). The helpers are
``cdef inline`` so that they are inlined into the per-pixel loop of each
importing module (e.g., ``_batch_photometry`` and ``_batch_stats``),
avoiding a function call per pixel.

These functions are pure C math and use no global mutable state, so they
are safe to call without the GIL, including on free-threaded Python
builds.
"""

from photutils.geometry._polygon_overlap cimport (
    convex_polygon_pixel_overlap, polygon_overlap_single_subpixel,
    polygon_pixel_overlap)
from photutils.geometry.circle_overlap cimport circle_frac_from_d2
from photutils.geometry.ellipse_overlap cimport ellipse_frac_from_rpix2
from photutils.geometry.rectangle_overlap cimport (
    rect_vertices, rectangle_overlap_single_subpixel)


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)


cdef inline double _circle_pixel_frac(double pxmin, double pymin,
                                      double dx, double dy,
                                      double pixel_radius, double r,
                                      int use_exact,
                                      int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a circle of radius ``r``
    centered on the origin.

    This replicates the per-pixel logic of ``circular_overlap_grid``,
    including the bounding-box and "well inside/outside" shortcuts, so
    the result is identical to the grid function.
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double pxcen, pycen, d2

    # Bounding-box check
    if not (pxmax > -r - 0.5 * dx and pxmin < r + 0.5 * dx
            and pymax > -r - 0.5 * dy and pymin < r + 0.5 * dy):
        return 0.0

    # Squared distance from circle center to pixel center
    pxcen = pxmin + dx * 0.5
    pycen = pymin + dy * 0.5
    d2 = pxcen * pxcen + pycen * pycen

    return circle_frac_from_d2(pxmin, pymin, pxmax, pymax, dx, dy,
                               pixel_radius, d2, r, use_exact, subpixels)


cdef inline double _circular_annulus_pixel_frac(double pxmin, double pymin,
                                                double dx, double dy,
                                                double pixel_radius,
                                                double r_in, double r_out,
                                                int use_exact,
                                                int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a circular annulus with
    inner radius ``r_in`` and outer radius ``r_out`` centered on the
    origin.

    The squared pixel-center distance and the (outer) bounding-box test
    are computed once and shared by the outer and inner boundary
    evaluations, avoiding the redundant setup of subtracting two
    independent ``_circle_pixel_frac`` calls. The result is clamped at
    zero: an annulus overlap can never be negative, but subtracting two
    overlaps can yield a tiny negative value from floating-point noise.
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double pxcen, pycen, d2, frac

    # Outer bounding-box check
    if not (pxmax > -r_out - 0.5 * dx and pxmin < r_out + 0.5 * dx
            and pymax > -r_out - 0.5 * dy and pymin < r_out + 0.5 * dy):
        return 0.0

    # Squared distance from the annulus center to the pixel center (shared)
    pxcen = pxmin + dx * 0.5
    pycen = pymin + dy * 0.5
    d2 = pxcen * pxcen + pycen * pycen

    frac = (circle_frac_from_d2(pxmin, pymin, pxmax, pymax, dx, dy,
                                pixel_radius, d2, r_out, use_exact, subpixels)
            - circle_frac_from_d2(pxmin, pymin, pxmax, pymax, dx, dy,
                                  pixel_radius, d2, r_in, use_exact,
                                  subpixels))
    return frac if frac > 0.0 else 0.0


cdef inline double _ellipse_frac_core(double pxmin, double pymin,
                                      double pxmax, double pymax,
                                      double dx, double dy, double norm,
                                      double pxcen, double pycen,
                                      double rx, double ry,
                                      double cos_theta, double sin_theta,
                                      int use_exact,
                                      int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps an ellipse with semi-axes
    ``rx``/``ry`` and orientation ``cos_theta``/``sin_theta`` centered
    on the origin, given the precomputed pixel center ``pxcen``/
    ``pycen``.

    No bounding-box check is performed (the caller is responsible for
    it), so this can be shared by the ellipse and elliptical-annulus
    helpers.
    """
    cdef double inv_rx2 = 1.0 / (rx * rx)
    cdef double inv_ry2 = 1.0 / (ry * ry)
    cdef double cxx, cyy, cxy, margin, f_in, f_out, rpix2

    # Quadratic-form coefficients and fast-path thresholds. These are
    # kept inline (rather than calling ``ellipse_quadratic_coeffs``) so
    # they stay in registers in this per-pixel hot path; the shared
    # decision core ``ellipse_frac_from_rpix2`` does the fast path and
    # exact/subpixel dispatch.
    cxx = cos_theta * cos_theta * inv_rx2 + sin_theta * sin_theta * inv_ry2
    cyy = sin_theta * sin_theta * inv_rx2 + cos_theta * cos_theta * inv_ry2
    cxy = 2.0 * cos_theta * sin_theta * (inv_rx2 - inv_ry2)
    margin = 0.5 * sqrt(dx * dx + dy * dy) / fmin(rx, ry)
    f_in = 1.0 - margin
    f_in = f_in * f_in if f_in > 0.0 else 0.0
    f_out = (1.0 + margin) * (1.0 + margin)

    rpix2 = cxx * pxcen * pxcen + cyy * pycen * pycen + cxy * pxcen * pycen
    return ellipse_frac_from_rpix2(pxmin, pymin, pxmax, pymax, norm, rx, ry,
                                   cos_theta, sin_theta, rpix2, f_in, f_out,
                                   use_exact, subpixels)


cdef inline double _ellipse_pixel_frac(double pxmin, double pymin,
                                       double dx, double dy, double norm,
                                       double rx, double ry,
                                       double cos_theta, double sin_theta,
                                       int use_exact,
                                       int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps an ellipse with semimajor
    and semiminor axes ``rx`` and ``ry`` and position angle ``theta``
    centered on the origin.

    ``cos_theta`` and ``sin_theta`` are the cosine and sine of the
    position angle, precomputed once per aperture by the caller.

    This replicates the per-pixel logic of ``elliptical_overlap_grid``,
    including the bounding-circle shortcut and the interior/exterior
    fast path, so the result is identical to the grid function.
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double r = fmax(rx, ry)  # bounding circle radius
    cdef double pxcen, pycen

    # Bounding-box check
    if not (pxmax > -r - 0.5 * dx and pxmin < r + 0.5 * dx
            and pymax > -r - 0.5 * dy and pymin < r + 0.5 * dy):
        return 0.0

    pxcen = pxmin + 0.5 * dx
    pycen = pymin + 0.5 * dy
    return _ellipse_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, norm,
                              pxcen, pycen, rx, ry, cos_theta, sin_theta,
                              use_exact, subpixels)


cdef inline double _elliptical_annulus_pixel_frac(
        double pxmin, double pymin, double dx, double dy, double norm,
        double rx_in, double ry_in, double rx_out, double ry_out,
        double cos_theta, double sin_theta,
        int use_exact, int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps an elliptical annulus with
    inner semi-axes ``rx_in``/``ry_in`` and outer semi-axes ``rx_out``/
    ``ry_out`` (shared orientation ``cos_theta``/``sin_theta``) centered
    on the origin.

    The pixel center and the (outer) bounding-box test are computed once
    and shared by the outer and inner boundary evaluations. The result
    is clamped at zero to remove tiny negative values from
    floating-point noise (an annulus overlap can never be negative).
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double r = fmax(rx_out, ry_out)  # outer bounding circle radius
    cdef double pxcen, pycen, frac

    # Outer bounding-box check
    if not (pxmax > -r - 0.5 * dx and pxmin < r + 0.5 * dx
            and pymax > -r - 0.5 * dy and pymin < r + 0.5 * dy):
        return 0.0

    pxcen = pxmin + 0.5 * dx
    pycen = pymin + 0.5 * dy
    frac = (_ellipse_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, norm,
                               pxcen, pycen, rx_out, ry_out, cos_theta,
                               sin_theta, use_exact, subpixels)
            - _ellipse_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, norm,
                                 pxcen, pycen, rx_in, ry_in, cos_theta,
                                 sin_theta, use_exact, subpixels))
    return frac if frac > 0.0 else 0.0


cdef inline double _rect_frac_core(double pxmin, double pymin,
                                   double pxmax, double pymax,
                                   double dx, double dy, double margin,
                                   double half_width, double half_height,
                                   double cos_theta, double sin_theta,
                                   double bbox_dx, double bbox_dy,
                                   double axrot, double ayrot,
                                   double *poly_x, double *poly_y,
                                   double *buf_a_x, double *buf_a_y,
                                   double *buf_b_x, double *buf_b_y,
                                   int use_exact,
                                   int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a rotated rectangle
    centered on the origin, given the precomputed rotated pixel-center
    distances ``axrot``/``ayrot`` (used only by the exact
    interior/exterior fast path).

    This is shared by the rectangle and rectangular-annulus helpers so
    the rotated pixel center is computed only once per pixel.
    """
    if use_exact:
        # Bounding-box check
        if (pxmax <= -bbox_dx or pxmin >= bbox_dx
                or pymax <= -bbox_dy or pymin >= bbox_dy):
            return 0.0

        if axrot >= half_width + margin or ayrot >= half_height + margin:
            return 0.0  # wholly outside
        if (axrot <= half_width - margin
                and ayrot <= half_height - margin):
            return 1.0  # wholly inside

        return polygon_pixel_overlap(pxmin, pymin, pxmax, pymax,
                                     poly_x, poly_y, 4,
                                     buf_a_x, buf_a_y, buf_b_x, buf_b_y,
                                     32) / (dx * dy)

    return rectangle_overlap_single_subpixel(pxmin, pymin, pxmax, pymax,
                                             half_width, half_height,
                                             cos_theta, sin_theta,
                                             subpixels)


cdef inline double _rect_pixel_frac(double pxmin, double pymin,
                                    double dx, double dy, double margin,
                                    double half_width, double half_height,
                                    double cos_theta, double sin_theta,
                                    double bbox_dx, double bbox_dy,
                                    double *poly_x, double *poly_y,
                                    double *buf_a_x, double *buf_a_y,
                                    double *buf_b_x, double *buf_b_y,
                                    int use_exact,
                                    int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a rectangle of full width
    ``2 * half_width`` and full height ``2 * half_height`` rotated by
    ``theta`` (given as ``cos_theta``/``sin_theta``) centered on the
    origin. ``margin`` is half the pixel diagonal.

    This replicates the per-pixel logic of ``rectangular_overlap_grid``
    (the exact mode uses an interior/exterior fast path and skips pixels
    outside the axis-aligned bounding box of the rotated rectangle), so
    the result is identical to the grid function.
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double pxcen, pycen, axrot = 0.0, ayrot = 0.0

    if use_exact:
        # Rotation into the rectangle frame is an isometry, so every
        # point of the pixel lies within ``margin`` of the rotated pixel
        # center.
        pxcen = pxmin + 0.5 * dx
        pycen = pymin + 0.5 * dy
        axrot = fabs(pxcen * cos_theta + pycen * sin_theta)
        ayrot = fabs(-pxcen * sin_theta + pycen * cos_theta)

    return _rect_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, margin,
                           half_width, half_height, cos_theta, sin_theta,
                           bbox_dx, bbox_dy, axrot, ayrot,
                           poly_x, poly_y, buf_a_x, buf_a_y,
                           buf_b_x, buf_b_y, use_exact, subpixels)


cdef inline double _rectangular_annulus_pixel_frac(
        double pxmin, double pymin, double dx, double dy, double margin,
        double half_width_in, double half_height_in,
        double half_width_out, double half_height_out,
        double cos_theta, double sin_theta,
        double bbox_dx_in, double bbox_dy_in,
        double bbox_dx_out, double bbox_dy_out,
        double *poly_x_in, double *poly_y_in,
        double *poly_x_out, double *poly_y_out,
        double *buf_a_x, double *buf_a_y, double *buf_b_x, double *buf_b_y,
        int use_exact, int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a rectangular annulus
    (shared orientation ``cos_theta``/``sin_theta``) centered on the
    origin.

    The rotated pixel center is computed once and shared by the outer
    and inner boundary evaluations. The result is clamped at zero to
    remove tiny negative values from floating-point noise (an annulus
    overlap can never be negative).
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy
    cdef double pxcen, pycen, axrot = 0.0, ayrot = 0.0, frac

    if use_exact:
        # Rotated pixel center, shared by both boundary evaluations.
        pxcen = pxmin + 0.5 * dx
        pycen = pymin + 0.5 * dy
        axrot = fabs(pxcen * cos_theta + pycen * sin_theta)
        ayrot = fabs(-pxcen * sin_theta + pycen * cos_theta)

    frac = (_rect_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, margin,
                            half_width_out, half_height_out, cos_theta,
                            sin_theta, bbox_dx_out, bbox_dy_out, axrot,
                            ayrot, poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                            buf_b_x, buf_b_y, use_exact, subpixels)
            - _rect_frac_core(pxmin, pymin, pxmax, pymax, dx, dy, margin,
                              half_width_in, half_height_in, cos_theta,
                              sin_theta, bbox_dx_in, bbox_dy_in, axrot,
                              ayrot, poly_x_in, poly_y_in, buf_a_x, buf_a_y,
                              buf_b_x, buf_b_y, use_exact, subpixels))
    return frac if frac > 0.0 else 0.0


cdef inline double _polygon_pixel_frac(double pxmin, double pymin,
                                       double dx, double dy, double margin,
                                       double *poly_x, double *poly_y,
                                       int n_poly,
                                       double *edge_nx, double *edge_ny,
                                       double *edge_c, int is_convex,
                                       double *buf_a_x, double *buf_a_y,
                                       double *buf_b_x, double *buf_b_y,
                                       int buf_size, int use_exact,
                                       int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a simple polygon supplied
    as counter-clockwise vertices centered on the origin. ``margin`` is
    half the pixel diagonal.

    This replicates the per-pixel logic of ``polygon_overlap_grid``: the
    exact mode uses an interior/exterior fast path for convex polygons
    (``is_convex``) and otherwise clips the pixel against the polygon
    (Sutherland-Hodgman), while the subpixel mode samples pixel centers
    with point-in-polygon tests, so the result is identical to the grid
    function.
    """
    cdef double pxmax = pxmin + dx
    cdef double pymax = pymin + dy

    if use_exact:
        return convex_polygon_pixel_overlap(
            pxmin, pymin, pxmax, pymax, poly_x, poly_y, n_poly,
            edge_nx, edge_ny, edge_c, is_convex, margin,
            buf_a_x, buf_a_y, buf_b_x, buf_b_y, buf_size) / (dx * dy)

    return polygon_overlap_single_subpixel(pxmin, pymin, pxmax, pymax,
                                           poly_x, poly_y, n_poly,
                                           subpixels, buf_a_x, buf_a_y,
                                           buf_b_x)
