# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to provide a batch driver for aperture photometry.

For each source position, the overlap fraction of the aperture with
each pixel in the aperture bounding box is computed and immediately
accumulated into the aperture sum, without materializing per-source
mask arrays or making per-source Python calls. The per-pixel overlap
fractions are computed with exactly the same arithmetic as the
``photutils.geometry`` grid functions, so the results agree with the
mask-based photometry code path.

The main source loop runs without the GIL and uses no global mutable
state, so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from photutils.geometry._polygon_overlap cimport (
    convex_edge_normals, convex_polygon_pixel_overlap,
    polygon_overlap_single_subpixel, polygon_pixel_overlap)
from photutils.geometry.circle_overlap cimport (circle_overlap_single_exact,
                                                circle_overlap_single_subpixel)
from photutils.geometry.ellipse_overlap cimport (
    ellipse_overlap_single_exact, ellipse_overlap_single_subpixel)
from photutils.geometry.rectangle_overlap cimport (
    rectangle_overlap_single_subpixel)

__all__ = ['batch_aperture_sums']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    double floor(double x)
    double ceil(double x)

# Aperture shape codes. These must stay in sync with the private
# ``_shape_params`` attributes defined by the aperture classes in
# ``photutils.aperture``.
cdef enum:
    _CIRCLE = 0
    _CIRCULAR_ANNULUS = 1
    _ELLIPSE = 2
    _ELLIPTICAL_ANNULUS = 3
    _RECTANGLE = 4
    _RECTANGULAR_ANNULUS = 5
    _POLYGON = 6

# Python-level aliases of the shape codes
SHAPE_CIRCLE = _CIRCLE
SHAPE_CIRCULAR_ANNULUS = _CIRCULAR_ANNULUS
SHAPE_ELLIPSE = _ELLIPSE
SHAPE_ELLIPTICAL_ANNULUS = _ELLIPTICAL_ANNULUS
SHAPE_RECTANGLE = _RECTANGLE
SHAPE_RECTANGULAR_ANNULUS = _RECTANGULAR_ANNULUS
SHAPE_POLYGON = _POLYGON


def batch_aperture_sums(const double[:, ::1] data, const double[:, ::1] error,
                        const unsigned char[:, ::1] mask,
                        const double[:, ::1] positions, int shape_code,
                        const double[::1] params, double ext_x, double ext_y,
                        int use_exact, int subpixels,
                        const Py_ssize_t[:, ::1] segmentation=None,
                        const Py_ssize_t[::1] labels=None, int seg_method=0):
    """
    Compute aperture sums for many source positions in a single call.

    For each position, the aperture bounding box is computed in exactly
    the same way as `photutils.aperture.BoundingBox.from_float`, and
    the per-pixel overlap fractions within the bounding box (clipped
    to the data) are computed with exactly the same arithmetic as the
    `photutils.geometry` grid functions, so the resulting sums match the
    mask-based photometry code path.

    Pixels with a non-positive overlap fraction or that are masked are
    excluded from the sums.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array.

    error : 2D ndarray of float64 (C-contiguous) or `None`
        The pixel-wise 1-sigma errors. Must have the same shape as
        ``data``.

    mask : 2D ndarray of uint8 (C-contiguous) or `None`
        A mask array where nonzero values indicate masked (excluded)
        pixels. Must have the same shape as ``data``.

    positions : 2D ndarray of float64 (C-contiguous)
        The (x, y) source positions with shape ``(n_sources, 2)``.

    shape_code : int
        The aperture shape code: 0=circle, 1=circular annulus,
        2=ellipse, 3=elliptical annulus, 4=rectangle, 5=rectangular
        annulus, 6=polygon (see the module-level ``SHAPE_*`` constants).

    params : 1D ndarray of float64 (C-contiguous)
        The aperture shape parameters:

        * circle: ``(r,)``
        * circular annulus: ``(r_in, r_out)``
        * ellipse: ``(a, b, theta)``
        * elliptical annulus: ``(a_in, b_in, a_out, b_out, theta)``
        * rectangle: ``(w, h, theta)``
        * rectangular annulus: ``(w_in, h_in, w_out, h_out, theta)``
        * polygon: the flattened counter-clockwise vertex offsets
          ``(x0, y0, x1, y1, ...)`` relative to each position (at least
          3 vertices, i.e., 6 values)

        where ``theta`` is in radians.

    ext_x, ext_y : float
        The half-extents of the aperture minimal bounding box in the x
        and y directions (i.e., ``Aperture._xy_extents``).

    use_exact : int
        Whether to compute exact overlap fractions (1) or use subpixel
        sampling (0).

    subpixels : int
        The number of subpixels in each dimension when ``use_exact`` is
        0.

    segmentation : 2D ndarray of intp (C-contiguous) or `None`
        A segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``data``. If `None`, no segmentation masking is applied.

    labels : 1D ndarray of intp (C-contiguous) or `None`
        The target source label for each position with shape
        ``(n_sources,)``. A label of 0 disables segmentation masking for
        that source. Required (not `None`) if ``segmentation`` is input.

    seg_method : int
        The segmentation masking method:

        * 0: disables masking
        * 1: excludes neighbor-source pixels
             (``(seg > 0) & (seg != label)``)
        * 2: excludes all pixels not assigned to the target source
             (``seg != label``).
        * 3: replaces neighbor-source pixels with the values mirrored
             across the (rounded) aperture center (the symmetric
             ``'correct'`` method). For method 3, a neighbor pixel whose
             mirror falls outside the aperture bounding box, is itself a
             neighbor, or is masked is excluded instead of replaced.

    Returns
    -------
    sums : 1D ndarray of float64
        The aperture sums. NaN where the aperture bounding box does not
        overlap the data.

    sum_errs : 1D ndarray of float64
        The quadrature-summed errors. NaN where ``error`` is `None` or
        the aperture bounding box does not overlap the data.

    overlap : 1D ndarray of bool
        Whether the aperture bounding box overlaps with the data.
    """
    cdef Py_ssize_t n_src = positions.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    sums_arr = np.full(n_src, np.nan)
    errs_arr = np.full(n_src, np.nan)
    overlap_arr = np.zeros(n_src, dtype=np.uint8)
    cdef double[::1] sums = sums_arr
    cdef double[::1] errs = errs_arr
    cdef unsigned char[::1] overlap = overlap_arr

    cdef bint has_error = error is not None
    cdef bint has_mask = mask is not None
    cdef bint has_seg = segmentation is not None
    cdef Py_ssize_t lbl = 0, seg_val

    # Aperture shape parameters (constant over all source positions)
    cdef double r_in = 0.0, r_out = 0.0
    cdef double rx_in = 0.0, ry_in = 0.0, rx_out = 0.0, ry_out = 0.0
    cdef double theta = 0.0, cos_theta = 1.0, sin_theta = 0.0
    cdef double half_width_in = 0.0, half_height_in = 0.0
    cdef double half_width_out = 0.0, half_height_out = 0.0
    cdef double bbox_dx_in = 0.0, bbox_dy_in = 0.0
    cdef double bbox_dx_out = 0.0, bbox_dy_out = 0.0
    cdef double poly_x_in[4]
    cdef double poly_y_in[4]
    cdef double poly_x_out[4]
    cdef double poly_y_out[4]

    # Scratch buffers for the polygon clipping (rectangular apertures);
    # these are local to this call, so this function is thread safe.
    cdef double buf_a_x[32]
    cdef double buf_a_y[32]
    cdef double buf_b_x[32]
    cdef double buf_b_y[32]

    # Working buffers for arbitrary-polygon apertures. The vertex count
    # is variable, so these are allocated as a single numpy block (kept
    # alive by ``poly_work``) and accessed through raw pointers. They
    # are local to this call, so this function is thread safe.
    cdef int n_poly = 0, poly_buf_size = 0
    cdef int is_poly_convex = 0
    cdef Py_ssize_t pk
    cdef double[::1] poly_work
    cdef double *poly_x = NULL
    cdef double *poly_y = NULL
    cdef double *pbuf_a_x = NULL
    cdef double *pbuf_a_y = NULL
    cdef double *pbuf_b_x = NULL
    cdef double *pbuf_b_y = NULL
    cdef double *pedge_nx = NULL
    cdef double *pedge_ny = NULL
    cdef double *pedge_c = NULL

    if shape_code == _CIRCLE:
        r_out = params[0]
    elif shape_code == _CIRCULAR_ANNULUS:
        r_in = params[0]
        r_out = params[1]
    elif shape_code == _ELLIPSE:
        rx_out = params[0]
        ry_out = params[1]
        theta = params[2]
        cos_theta = cos(theta)
        sin_theta = sin(theta)
    elif shape_code == _ELLIPTICAL_ANNULUS:
        rx_in = params[0]
        ry_in = params[1]
        rx_out = params[2]
        ry_out = params[3]
        theta = params[4]
        cos_theta = cos(theta)
        sin_theta = sin(theta)
    elif shape_code == _RECTANGLE or shape_code == _RECTANGULAR_ANNULUS:
        if shape_code == _RECTANGLE:
            half_width_out = 0.5 * params[0]
            half_height_out = 0.5 * params[1]
            theta = params[2]
        else:
            half_width_in = 0.5 * params[0]
            half_height_in = 0.5 * params[1]
            half_width_out = 0.5 * params[2]
            half_height_out = 0.5 * params[3]
            theta = params[4]

        cos_theta = cos(theta)
        sin_theta = sin(theta)
        _rect_vertices(half_width_out, half_height_out, cos_theta,
                       sin_theta, poly_x_out, poly_y_out)
        bbox_dx_out = (half_width_out * fabs(cos_theta)
                       + half_height_out * fabs(sin_theta))
        bbox_dy_out = (half_width_out * fabs(sin_theta)
                       + half_height_out * fabs(cos_theta))
        if shape_code == _RECTANGULAR_ANNULUS:
            _rect_vertices(half_width_in, half_height_in, cos_theta,
                           sin_theta, poly_x_in, poly_y_in)
            bbox_dx_in = (half_width_in * fabs(cos_theta)
                          + half_height_in * fabs(sin_theta))
            bbox_dy_in = (half_width_in * fabs(sin_theta)
                          + half_height_in * fabs(cos_theta))
    elif shape_code == _POLYGON:
        # ``params`` holds the flattened counter-clockwise vertex
        # offsets (x0, y0, x1, y1, ...).
        n_poly = params.shape[0] // 2
        if n_poly < 3 or 2 * n_poly != params.shape[0]:
            msg = ('polygon params must be the flattened (x, y) offsets '
                   'of at least 3 vertices')
            raise ValueError(msg)

        # Each Sutherland-Hodgman clip can at most double the vertex
        # count of a non-convex polygon, so 16 * n_poly is a safe bound
        # (see ``polygon_overlap_grid``).
        poly_buf_size = 16 * n_poly
        poly_work = np.empty(2 * n_poly + 4 * poly_buf_size + 3 * n_poly,
                             dtype=np.float64)
        poly_x = &poly_work[0]
        poly_y = &poly_work[n_poly]
        pbuf_a_x = &poly_work[2 * n_poly]
        pbuf_a_y = &poly_work[2 * n_poly + poly_buf_size]
        pbuf_b_x = &poly_work[2 * n_poly + 2 * poly_buf_size]
        pbuf_b_y = &poly_work[2 * n_poly + 3 * poly_buf_size]
        pedge_nx = &poly_work[2 * n_poly + 4 * poly_buf_size]
        pedge_ny = &poly_work[3 * n_poly + 4 * poly_buf_size]
        pedge_c = &poly_work[4 * n_poly + 4 * poly_buf_size]
        for pk in range(n_poly):
            poly_x[pk] = params[2 * pk]
            poly_y[pk] = params[2 * pk + 1]

        # One-time convexity test; convex polygons use an
        # interior/exterior fast path in ``_polygon_pixel_frac``.
        is_poly_convex = convex_edge_normals(poly_x, poly_y, n_poly,
                                             pedge_nx, pedge_ny, pedge_c)
    else:
        msg = f'Invalid shape_code: {shape_code}'
        raise ValueError(msg)

    cdef Py_ssize_t k, ix, iy, ix0, ix1, iy0, iy1
    cdef Py_ssize_t ixmin, iymin, grid_nx, grid_ny
    cdef Py_ssize_t six, siy, xm, ym, ccx = 0, ccy = 0, mseg
    cdef double cx, cy
    cdef double ixmin_d, ixmax_d, iymin_d, iymax_d
    cdef double gxmin, gxmax, gymin, gymax
    cdef double dx, dy, pixel_radius, norm
    cdef double pxmin, pymin, frac, err_val, sum_val, var_val

    with nogil:
        for k in range(n_src):
            cx = positions[k, 0]
            cy = positions[k, 1]
            if has_seg:
                lbl = labels[k]
                if seg_method == 3:
                    # Center pixel for the symmetric 'correct' mirror,
                    # rounded half away from zero (round_half_away)
                    if cx >= 0.0:
                        ccx = <Py_ssize_t>floor(cx + 0.5)
                    else:
                        ccx = <Py_ssize_t>ceil(cx - 0.5)
                    if cy >= 0.0:
                        ccy = <Py_ssize_t>floor(cy + 0.5)
                    else:
                        ccy = <Py_ssize_t>ceil(cy - 0.5)

            # Replicate BoundingBox.from_float; the values are kept as
            # (integral) doubles to avoid integer overflow for apertures
            # far outside the data
            ixmin_d = floor(cx - ext_x + 0.5)
            ixmax_d = ceil(cx + ext_x + 0.5)
            iymin_d = floor(cy - ext_y + 0.5)
            iymax_d = ceil(cy + ext_y + 0.5)

            # No overlap of the aperture bounding box with the data
            # (replicates BoundingBox.get_overlap_slices); the sums stay
            # NaN
            if (ixmin_d >= <double>nx_data or ixmax_d <= 0.0
                    or iymin_d >= <double>ny_data or iymax_d <= 0.0):
                continue
            overlap[k] = 1

            # Pixel-grid edges relative to the aperture center
            # (replicates Aperture._centered_edges and the grid setup of
            # the photutils.geometry grid functions)
            gxmin = ixmin_d - 0.5 - cx
            gxmax = ixmax_d - 0.5 - cx
            gymin = iymin_d - 0.5 - cy
            gymax = iymax_d - 0.5 - cy
            grid_nx = <Py_ssize_t>(ixmax_d - ixmin_d)
            grid_ny = <Py_ssize_t>(iymax_d - iymin_d)
            dx = (gxmax - gxmin) / grid_nx
            dy = (gymax - gymin) / grid_ny
            pixel_radius = 0.5 * sqrt(dx * dx + dy * dy)
            norm = 1.0 / (dx * dy)

            # Pixel index ranges clipped to the data
            ixmin = <Py_ssize_t>ixmin_d
            iymin = <Py_ssize_t>iymin_d
            ix0 = <Py_ssize_t>fmax(ixmin_d, 0.0)
            ix1 = <Py_ssize_t>fmin(ixmax_d, <double>nx_data)
            iy0 = <Py_ssize_t>fmax(iymin_d, 0.0)
            iy1 = <Py_ssize_t>fmin(iymax_d, <double>ny_data)

            sum_val = 0.0
            var_val = 0.0
            for iy in range(iy0, iy1):
                pymin = gymin + (iy - iymin) * dy
                for ix in range(ix0, ix1):
                    if has_mask and mask[iy, ix]:
                        continue
                    six = ix
                    siy = iy
                    if has_seg and lbl != 0:
                        seg_val = segmentation[iy, ix]
                        if seg_method == 1:
                            if seg_val != 0 and seg_val != lbl:
                                continue
                        elif seg_method == 2:
                            if seg_val != lbl:
                                continue
                        elif seg_method == 3:
                            if seg_val != 0 and seg_val != lbl:
                                # Neighbor pixel: replace its value with
                                # the pixel mirrored across the center.
                                xm = 2 * ccx - ix
                                ym = 2 * ccy - iy
                                if (xm < ix0 or xm >= ix1
                                        or ym < iy0 or ym >= iy1):
                                    continue
                                mseg = segmentation[ym, xm]
                                if mseg != 0 and mseg != lbl:
                                    continue
                                if has_mask and mask[ym, xm]:
                                    continue
                                six = xm
                                siy = ym
                    pxmin = gxmin + (ix - ixmin) * dx

                    if shape_code == _CIRCLE:
                        frac = _circle_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, r_out,
                            use_exact, subpixels)
                    elif shape_code == _CIRCULAR_ANNULUS:
                        frac = (_circle_pixel_frac(
                                    pxmin, pymin, dx, dy, pixel_radius,
                                    r_out, use_exact, subpixels)
                                - _circle_pixel_frac(
                                    pxmin, pymin, dx, dy, pixel_radius,
                                    r_in, use_exact, subpixels))
                    elif shape_code == _ELLIPSE:
                        frac = _ellipse_pixel_frac(
                            pxmin, pymin, dx, dy, norm, rx_out, ry_out,
                            cos_theta, sin_theta, use_exact, subpixels)
                    elif shape_code == _ELLIPTICAL_ANNULUS:
                        frac = (_ellipse_pixel_frac(
                                    pxmin, pymin, dx, dy, norm, rx_out,
                                    ry_out, cos_theta, sin_theta,
                                    use_exact, subpixels)
                                - _ellipse_pixel_frac(
                                    pxmin, pymin, dx, dy, norm, rx_in,
                                    ry_in, cos_theta, sin_theta,
                                    use_exact, subpixels))
                    elif shape_code == _RECTANGLE:
                        frac = _rect_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius,
                            half_width_out, half_height_out, cos_theta,
                            sin_theta, bbox_dx_out, bbox_dy_out,
                            poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                            buf_b_x, buf_b_y, use_exact, subpixels)
                    elif shape_code == _RECTANGULAR_ANNULUS:
                        frac = (_rect_pixel_frac(
                                    pxmin, pymin, dx, dy, pixel_radius,
                                    half_width_out, half_height_out,
                                    cos_theta, sin_theta, bbox_dx_out,
                                    bbox_dy_out, poly_x_out, poly_y_out,
                                    buf_a_x, buf_a_y, buf_b_x, buf_b_y,
                                    use_exact, subpixels)
                                - _rect_pixel_frac(
                                    pxmin, pymin, dx, dy, pixel_radius,
                                    half_width_in, half_height_in,
                                    cos_theta, sin_theta, bbox_dx_in,
                                    bbox_dy_in, poly_x_in, poly_y_in,
                                    buf_a_x, buf_a_y, buf_b_x, buf_b_y,
                                    use_exact, subpixels))
                    else:
                        frac = _polygon_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius,
                            poly_x, poly_y, n_poly, pedge_nx, pedge_ny,
                            pedge_c, is_poly_convex, pbuf_a_x, pbuf_a_y,
                            pbuf_b_x, pbuf_b_y, poly_buf_size, use_exact,
                            subpixels)

                    if frac <= 0.0:
                        continue
                    sum_val += data[siy, six] * frac
                    if has_error:
                        err_val = error[siy, six]
                        var_val += err_val * err_val * frac

            sums[k] = sum_val
            if has_error:
                errs[k] = sqrt(var_val)

    return sums_arr, errs_arr, overlap_arr.view(bool)


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
    cdef double pxcen, pycen, d

    # Bounding-box check
    if not (pxmax > -r - 0.5 * dx and pxmin < r + 0.5 * dx
            and pymax > -r - 0.5 * dy and pymin < r + 0.5 * dy):
        return 0.0

    # Distance from circle center to pixel center
    pxcen = pxmin + dx * 0.5
    pycen = pymin + dy * 0.5
    d = sqrt(pxcen * pxcen + pycen * pycen)

    if d < r - pixel_radius:  # pixel is well within the circle
        return 1.0

    if d < r + pixel_radius:  # pixel is close to the circle border
        if use_exact:
            return circle_overlap_single_exact(pxmin, pymin, pxmax, pymax,
                                               r) / (dx * dy)
        return circle_overlap_single_subpixel(pxmin, pymin, pxmax, pymax, r,
                                              subpixels)

    return 0.0  # pixel is fully outside the circle


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
    cdef double pxcen, pycen, rpix2
    cdef double inv_rx2, inv_ry2
    cdef double cxx, cyy, cxy, margin, f_in, f_out

    # Bounding-box check
    if not (pxmax > -r - 0.5 * dx and pxmin < r + 0.5 * dx
            and pymax > -r - 0.5 * dy and pymin < r + 0.5 * dy):
        return 0.0

    # Quadratic-form coefficients of the ellipse, such that a point
    # (x, y) lies inside when ``cxx*x**2 + cyy*y**2 + cxy*x*y < 1``.
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)
    cxx = cos_theta * cos_theta * inv_rx2 + sin_theta * sin_theta * inv_ry2
    cyy = sin_theta * sin_theta * inv_rx2 + cos_theta * cos_theta * inv_ry2
    cxy = 2.0 * cos_theta * sin_theta * (inv_rx2 - inv_ry2)

    # Boundary band for the interior/exterior fast path (see
    # ``elliptical_overlap_grid``).
    margin = 0.5 * sqrt(dx * dx + dy * dy) / fmin(rx, ry)
    f_in = 1.0 - margin
    f_in = f_in * f_in if f_in > 0.0 else 0.0
    f_out = (1.0 + margin) * (1.0 + margin)

    pxcen = pxmin + 0.5 * dx
    pycen = pymin + 0.5 * dy
    rpix2 = cxx * pxcen * pxcen + cyy * pycen * pycen + cxy * pxcen * pycen
    if rpix2 >= f_out:
        return 0.0  # pixel fully outside the ellipse
    if rpix2 <= f_in:
        return 1.0  # pixel fully inside the ellipse

    if use_exact:
        return ellipse_overlap_single_exact(pxmin, pymin, pxmax, pymax, rx, ry,
                                            cos_theta, sin_theta) * norm
    return ellipse_overlap_single_subpixel(pxmin, pymin, pxmax, pymax, rx, ry,
                                           cos_theta, sin_theta, subpixels)


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
    cdef double pxcen, pycen, axrot, ayrot

    if use_exact:
        # Bounding-box check
        if (pxmax <= -bbox_dx or pxmin >= bbox_dx
                or pymax <= -bbox_dy or pymin >= bbox_dy):
            return 0.0

        # Interior/exterior fast path. Rotation into the rectangle frame
        # is an isometry, so every point of the pixel lies within
        # ``margin`` of the rotated pixel center.
        pxcen = pxmin + 0.5 * dx
        pycen = pymin + 0.5 * dy
        axrot = fabs(pxcen * cos_theta + pycen * sin_theta)
        ayrot = fabs(-pxcen * sin_theta + pycen * cos_theta)
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


cdef inline void _rect_vertices(double half_width, double half_height,
                                double cos_theta, double sin_theta,
                                double *poly_x,
                                double *poly_y) noexcept nogil:
    """
    Build the four CCW vertices of a rotated rectangle centered on the
    origin (same arithmetic as ``rectangular_overlap_grid``).
    """
    poly_x[0] = -half_width * cos_theta - (-half_height) * sin_theta
    poly_y[0] = -half_width * sin_theta + (-half_height) * cos_theta
    poly_x[1] = half_width * cos_theta - (-half_height) * sin_theta
    poly_y[1] = half_width * sin_theta + (-half_height) * cos_theta
    poly_x[2] = half_width * cos_theta - half_height * sin_theta
    poly_y[2] = half_width * sin_theta + half_height * cos_theta
    poly_x[3] = -half_width * cos_theta - half_height * sin_theta
    poly_y[3] = -half_width * sin_theta + half_height * cos_theta
