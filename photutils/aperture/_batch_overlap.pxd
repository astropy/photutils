# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Shared per-pixel aperture overlap helpers for the batch Cython drivers
for aperture photometry.

Each helper returns the fraction of a single pixel that overlaps an
aperture shape centered on the origin, using exactly the same arithmetic
as the corresponding `photutils.geometry` grid function (including the
bounding-box and interior/exterior fast paths). The helpers are
``cdef inline`` so that they are inlined into the per-pixel loop of each
importing module (e.g., ``_batch_photometry`` and ``_batch_stats``),
avoiding a function call per pixel.

This file also declares the aperture shape codes and the once-per-source
helpers shared by the batch drivers (bounding-box/pixel-grid setup and
packed-buffer presizing).

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
    rectangle_overlap_single_subpixel)


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    double floor(double x)
    double ceil(double x)


# Aperture shape codes. These must stay in sync with the private
# ``_shape_params`` attributes defined by the aperture classes in
# ``photutils.aperture`` (see also the Python-level ``SHAPE_*`` aliases
# in ``_batch_photometry``).
cdef enum:
    _CIRCLE = 0
    _CIRCULAR_ANNULUS = 1
    _ELLIPSE = 2
    _ELLIPTICAL_ANNULUS = 3
    _RECTANGLE = 4
    _RECTANGULAR_ANNULUS = 5
    _POLYGON = 6


cdef inline Py_ssize_t _round_half_away(double x) noexcept nogil:
    """
    Round to the nearest integer, rounding halfway values away from
    zero.

    This replicates ``photutils.utils._round.round_half_away``.

    Parameters
    ----------
    x : double
        The value to round.

    Returns
    -------
    result : Py_ssize_t
        The rounded integer value.
    """
    if x >= 0.0:
        return <Py_ssize_t>floor(x + 0.5)
    return <Py_ssize_t>ceil(x - 0.5)


cdef inline bint _source_grid_setup(double cx, double cy,
                                    double ext_x, double ext_y,
                                    Py_ssize_t nx_data, Py_ssize_t ny_data,
                                    double *gxmin, double *gymin,
                                    double *dx, double *dy,
                                    double *pixel_radius, double *norm,
                                    Py_ssize_t *ixmin, Py_ssize_t *iymin,
                                    Py_ssize_t *ix0, Py_ssize_t *ix1,
                                    Py_ssize_t *iy0,
                                    Py_ssize_t *iy1) noexcept nogil:
    """
    Compute the pixel grid and bounding box for a single source
    aperture.

    This reproduces the geometry computed by ``BoundingBox.from_float``,
    ``BoundingBox.get_overlap_slices``, and the
    ``Aperture._centered_edges`` grid setup used by the
    ``photutils.geometry`` routines. Consequently, the batch drivers
    iterate over exactly the same pixels with the same grid geometry as
    the mask-based code path.

    The bounding-box coordinates are stored as integral ``double``
    values rather than integers to avoid overflow for apertures located
    far outside the image.

    Parameters
    ----------
    cx, cy : double
        The aperture center, in data pixel coordinates.

    ext_x, ext_y : double
        Half the width and half the height of the aperture's bounding
        box, in pixels.

    nx_data, ny_data : Py_ssize_t
        The shape of the data array in the x and y direction.

    gxmin, gymin : double *
        Output. The lower grid edges relative to the aperture center.

    dx, dy : double *
        Output. The pixel spacing of the grid. For the current pixel
        grid, ``dx`` and ``dy`` are always equal to 1.0.

    pixel_radius : double *
        Output. Half the pixel diagonal.

    norm : double *
        Output. The inverse pixel area.

    ixmin, iymin : Py_ssize_t *
        Output. The unclipped bounding-box origin.

    ix0, ix1, iy0, iy1 : Py_ssize_t *
        Output. The bounding-box limits, clipped to the image.

    Returns
    -------
    result : bint
        `False` if the aperture bounding box does not overlap the
        image, in which case the output values are unspecified.
        Otherwise, returns `True` and writes the grid and bounding-box
        values described above through the output pointers.
    """
    cdef double ixmin_d = floor(cx - ext_x + 0.5)
    cdef double ixmax_d = ceil(cx + ext_x + 0.5)
    cdef double iymin_d = floor(cy - ext_y + 0.5)
    cdef double iymax_d = ceil(cy + ext_y + 0.5)
    cdef double gxmax, gymax
    cdef Py_ssize_t grid_nx, grid_ny

    if (ixmin_d >= <double>nx_data or ixmax_d <= 0.0
            or iymin_d >= <double>ny_data or iymax_d <= 0.0):
        return False

    gxmin[0] = ixmin_d - 0.5 - cx
    gxmax = ixmax_d - 0.5 - cx
    gymin[0] = iymin_d - 0.5 - cy
    gymax = iymax_d - 0.5 - cy
    grid_nx = <Py_ssize_t>(ixmax_d - ixmin_d)
    grid_ny = <Py_ssize_t>(iymax_d - iymin_d)
    dx[0] = (gxmax - gxmin[0]) / grid_nx
    dy[0] = (gymax - gymin[0]) / grid_ny
    pixel_radius[0] = 0.5 * sqrt(dx[0] * dx[0] + dy[0] * dy[0])
    norm[0] = 1.0 / (dx[0] * dy[0])

    ixmin[0] = <Py_ssize_t>ixmin_d
    iymin[0] = <Py_ssize_t>iymin_d
    ix0[0] = <Py_ssize_t>fmax(ixmin_d, 0.0)
    ix1[0] = <Py_ssize_t>fmin(ixmax_d, <double>nx_data)
    iy0[0] = <Py_ssize_t>fmax(iymin_d, 0.0)
    iy1[0] = <Py_ssize_t>fmin(iymax_d, <double>ny_data)
    return True


cdef inline Py_ssize_t _presize_packed_offsets(
        const double[:, ::1] positions, double ext_x, double ext_y,
        Py_ssize_t nx_data, Py_ssize_t ny_data,
        Py_ssize_t[::1] starts) noexcept nogil:
    """
    Compute the starting offset of each source within packed per-pixel
    buffers.

    For each source, the clipped bounding-box area provides an upper
    bound on the number of pixels that may contribute. The cumulative
    sum of these areas defines both the size of the packed per-pixel
    buffers used by the batch drivers and the starting offset of each
    source's region within those buffers.

    This function performs only bounding-box and offset calculations
    (using the same bounding-box logic as ``_source_grid_setup``). It
    does not iterate over or evaluate individual pixels.

    Parameters
    ----------
    positions : double[:, ::1]
        The (x, y) center of each source aperture, in data pixel
        coordinates, as an (N, 2) array.

    ext_x, ext_y : double
        Half the width and half the height of the aperture's bounding
        box, in pixels. All sources share the same extent.

    nx_data, ny_data : Py_ssize_t
        The shape of the data array in the x and y direction.

    starts : Py_ssize_t[::1]
        Output. The starting offset of each source's region within the
        packed per-pixel buffers.

    Returns
    -------
    total : Py_ssize_t
        The total required length of the packed per-pixel buffers.
    """
    cdef Py_ssize_t k, ix0, ix1, iy0, iy1, area, total = 0
    cdef double cx, cy, ixmin_d, ixmax_d, iymin_d, iymax_d

    for k in range(positions.shape[0]):
        cx = positions[k, 0]
        cy = positions[k, 1]
        ixmin_d = floor(cx - ext_x + 0.5)
        ixmax_d = ceil(cx + ext_x + 0.5)
        iymin_d = floor(cy - ext_y + 0.5)
        iymax_d = ceil(cy + ext_y + 0.5)
        starts[k] = total
        if (ixmin_d >= <double>nx_data or ixmax_d <= 0.0
                or iymin_d >= <double>ny_data or iymax_d <= 0.0):
            continue
        ix0 = <Py_ssize_t>fmax(ixmin_d, 0.0)
        ix1 = <Py_ssize_t>fmin(ixmax_d, <double>nx_data)
        iy0 = <Py_ssize_t>fmax(iymin_d, 0.0)
        iy1 = <Py_ssize_t>fmin(iymax_d, <double>ny_data)
        area = (ix1 - ix0) * (iy1 - iy0)
        if area > 0:
            total += area
    return total


cdef inline double _circle_pixel_frac(double pxmin, double pymin,
                                      double dx, double dy,
                                      double pixel_radius, double r,
                                      int use_exact,
                                      int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a circle of radius ``r``
    centered on the origin.

    This replicates the per-pixel logic of ``circular_overlap_grid``,
    including the bounding-box check and the interior/exterior fast
    path, so the result is identical to the grid function.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the circle center.

    dx, dy : double
        The pixel width and height.

    pixel_radius : double
        Half the pixel diagonal, used by the interior/exterior fast
        path.

    r : double
        The radius of the circle.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        circle.
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

    The squared pixel-center distance and the outer bounding-box check
    are computed once and shared by the outer and inner boundary
    evaluations, avoiding the redundant setup that two independent
    ``_circle_pixel_frac`` calls would incur.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the annulus center.

    dx, dy : double
        The pixel width and height.

    pixel_radius : double
        Half the pixel diagonal, used by the interior/exterior fast
        path.

    r_in, r_out : double
        The inner and outer radii of the annulus.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        annulus. The result is clamped at zero: an annulus overlap can
        never be negative, but subtracting the inner overlap from the
        outer overlap can otherwise yield a tiny negative value from
        floating-point noise.
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
    ``rx``/``ry`` and orientation ``cos_theta``/``sin_theta``, centered
    on the origin.

    This is the shared core used by both ``_ellipse_pixel_frac`` and
    ``_elliptical_annulus_pixel_frac``, given the precomputed pixel
    center ``pxcen``/``pycen``. The caller is responsible for the
    bounding-box check; none is performed here.

    Parameters
    ----------
    pxmin, pymin, pxmax, pymax : double
        The pixel edges, relative to the ellipse center.

    dx, dy : double
        The pixel width and height.

    norm : double
        The inverse pixel area, used to normalize the exact overlap
        calculation.

    pxcen, pycen : double
        The pixel center, relative to the ellipse center.

    rx, ry : double
        The semimajor and semiminor axes of the ellipse.

    cos_theta, sin_theta : double
        The cosine and sine of the ellipse's position angle.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        ellipse.
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
    and semiminor axes ``rx`` and ``ry`` and position angle ``theta``,
    centered on the origin.

    This replicates the per-pixel logic of ``elliptical_overlap_grid``,
    including the bounding-circle check and the interior/exterior fast
    path, so the result is identical to the grid function.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the ellipse center.

    dx, dy : double
        The pixel width and height.

    norm : double
        The inverse pixel area, used to normalize the exact overlap
        calculation.

    rx, ry : double
        The semimajor and semiminor axes of the ellipse.

    cos_theta, sin_theta : double
        The cosine and sine of the ellipse's position angle ``theta``,
        precomputed once per aperture by the caller.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        ellipse.
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
    inner semi-axes ``rx_in``/``ry_in`` and outer semi-axes
    ``rx_out``/``ry_out`` (sharing orientation
    ``cos_theta``/``sin_theta``), centered on the origin.

    The pixel center and the outer bounding-box check are computed once
    and shared by the outer and inner boundary evaluations.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the annulus center.

    dx, dy : double
        The pixel width and height.

    norm : double
        The inverse pixel area, used to normalize the exact overlap
        calculation.

    rx_in, ry_in : double
        The inner semimajor and semiminor axes of the annulus.

    rx_out, ry_out : double
        The outer semimajor and semiminor axes of the annulus.

    cos_theta, sin_theta : double
        The cosine and sine of the shared position angle ``theta``.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        annulus. The result is clamped at zero to remove tiny negative
        values from floating-point noise (an annulus overlap can never
        be negative).
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
    centered on the origin.

    This is the shared core used by both ``_rect_pixel_frac`` and
    ``_rectangular_annulus_pixel_frac``, so the rotated pixel center is
    computed only once per pixel by the caller and passed in as
    ``axrot``/``ayrot`` (used only by the exact interior/exterior fast
    path).

    Parameters
    ----------
    pxmin, pymin, pxmax, pymax : double
        The pixel edges, relative to the rectangle center.

    dx, dy : double
        The pixel width and height.

    margin : double
        Half the pixel diagonal.

    half_width, half_height : double
        Half the width and half the height of the rectangle.

    cos_theta, sin_theta : double
        The cosine and sine of the rectangle's rotation angle.

    bbox_dx, bbox_dy : double
        Half the width and half the height of the rectangle's
        axis-aligned bounding box.

    axrot, ayrot : double
        The rotated pixel-center distances from the rectangle center,
        precomputed by the caller. Used only when ``use_exact`` is 1.

    poly_x, poly_y : double *
        Scratch buffers for the rectangle's polygon vertices, used by
        the general (non-fast-path) exact overlap calculation.

    buf_a_x, buf_a_y, buf_b_x, buf_b_y : double *
        Scratch buffers used by the polygon-clipping routine.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        rectangle.
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
    ``2 * half_width`` and full height ``2 * half_height``, rotated by
    ``theta`` (given as ``cos_theta``/``sin_theta``) and centered on
    the origin.

    This replicates the per-pixel logic of ``rectangular_overlap_grid``:
    the exact mode uses an interior/exterior fast path and skips pixels
    outside the axis-aligned bounding box of the rotated rectangle, so
    the result is identical to the grid function.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the rectangle center.

    dx, dy : double
        The pixel width and height.

    margin : double
        Half the pixel diagonal.

    half_width, half_height : double
        Half the width and half the height of the rectangle.

    cos_theta, sin_theta : double
        The cosine and sine of the rectangle's rotation angle
        ``theta``.

    bbox_dx, bbox_dy : double
        Half the width and half the height of the rectangle's
        axis-aligned bounding box.

    poly_x, poly_y : double *
        Scratch buffers for the rectangle's polygon vertices, used by
        the general (non-fast-path) exact overlap calculation.

    buf_a_x, buf_a_y, buf_b_x, buf_b_y : double *
        Scratch buffers used by the polygon-clipping routine.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        rectangle.
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
    (sharing orientation ``cos_theta``/``sin_theta``), centered on the
    origin.

    The rotated pixel center is computed once and shared by the outer
    and inner boundary evaluations.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the annulus center.

    dx, dy : double
        The pixel width and height.

    margin : double
        Half the pixel diagonal.

    half_width_in, half_height_in : double
        Half the width and half the height of the inner rectangle.

    half_width_out, half_height_out : double
        Half the width and half the height of the outer rectangle.

    cos_theta, sin_theta : double
        The cosine and sine of the shared rotation angle ``theta``.

    bbox_dx_in, bbox_dy_in : double
        Half the width and half the height of the inner rectangle's
        axis-aligned bounding box.

    bbox_dx_out, bbox_dy_out : double
        Half the width and half the height of the outer rectangle's
        axis-aligned bounding box.

    poly_x_in, poly_y_in : double *
        Scratch buffers for the inner rectangle's polygon vertices.

    poly_x_out, poly_y_out : double *
        Scratch buffers for the outer rectangle's polygon vertices.

    buf_a_x, buf_a_y, buf_b_x, buf_b_y : double *
        Scratch buffers used by the polygon-clipping routine, shared
        by the inner and outer boundary evaluations.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        annulus. The result is clamped at zero to remove tiny negative
        values from floating-point noise (an annulus overlap can never
        be negative).
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
    Fraction of a single pixel that overlaps a simple polygon, supplied
    as counter-clockwise vertices centered on the origin.

    This replicates the per-pixel logic of ``polygon_overlap_grid``: the
    exact mode uses an interior/exterior fast path for convex polygons
    (``is_convex``) and otherwise clips the pixel against the polygon
    using the Sutherland-Hodgman algorithm; the subpixel mode instead
    samples pixel centers with point-in-polygon tests. The result is
    identical to the grid function.

    Parameters
    ----------
    pxmin, pymin : double
        The lower edges of the pixel, relative to the polygon center.

    dx, dy : double
        The pixel width and height.

    margin : double
        Half the pixel diagonal.

    poly_x, poly_y : double *
        The x and y coordinates of the polygon vertices, in
        counter-clockwise order.

    n_poly : int
        The number of polygon vertices.

    edge_nx, edge_ny, edge_c : double *
        The precomputed edge normals and offsets used by the convex
        interior/exterior fast path.

    is_convex : int
        Set to 1 if the polygon is convex, which enables the
        interior/exterior fast path. Set to 0 for general (possibly
        non-convex) polygons.

    buf_a_x, buf_a_y, buf_b_x, buf_b_y : double *
        Scratch buffers used by the polygon-clipping routine.

    buf_size : int
        The size of the scratch buffers.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        polygon.
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
