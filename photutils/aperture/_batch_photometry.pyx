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

from photutils.aperture._batch_overlap cimport (
    _circle_pixel_frac, _circular_annulus_pixel_frac, _ellipse_pixel_frac,
    _elliptical_annulus_pixel_frac, _polygon_pixel_frac, _rect_pixel_frac,
    _rectangular_annulus_pixel_frac)
from photutils.geometry._polygon_overlap cimport convex_edge_normals
from photutils.geometry.rectangle_overlap cimport rect_vertices

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
                        const Py_ssize_t[::1] labels=None, int seg_method=0,
                        const double[::1] local_bkg=None, int emit_sum=0):
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

    local_bkg : 1D ndarray of float64 (C-contiguous) or `None`
        The per-source local background to subtract from each pixel
        value of that source. If `None`, no background is subtracted.

    emit_sum : int, optional
        If nonzero, also emit the packed per-pixel member buffers
        (``sum_values``, ``sum_fracs``, ``sum_errsq``, ``sum_counts``)
        needed to recompute the aperture sum, variance, and area after
        per-source sigma clipping. When zero (default), those four
        outputs are empty arrays.

    Returns
    -------
    sums : 1D ndarray of float64
        The aperture sums. NaN where the aperture bounding box does not
        overlap the data.

    sum_vars : 1D ndarray of float64
        The aperture error variances (the quadrature sum of the pixel
        variances weighted by the overlap fractions). NaN where
        ``error`` is `None` or the aperture bounding box does not overlap
        the data. The caller takes the square root to obtain the error.

    areas : 1D ndarray of float64
        The total unmasked overlap area of the aperture (the sum of the
        overlap fractions). NaN where the aperture bounding box does not
        overlap the data.

    overlap : 1D ndarray of bool
        Whether the aperture bounding box overlaps with the data.

    starts : 1D ndarray of intp
        The per-source starting offset into the packed member buffers.
        Zeros unless ``emit_sum`` is nonzero.

    sum_values : 1D ndarray of float64
        The packed per-pixel ``data - local_bkg`` values. Empty unless
        ``emit_sum`` is nonzero.

    sum_fracs : 1D ndarray of float64
        The packed per-pixel overlap fractions. Empty unless
        ``emit_sum`` is nonzero.

    sum_errsq : 1D ndarray of float64
        The packed per-pixel squared errors (zero where ``error`` is
        `None`). Empty unless ``emit_sum`` is nonzero.

    sum_counts : 1D ndarray of intp
        The per-source count of packed contributing pixels. Empty
        unless ``emit_sum`` is nonzero.
    """
    cdef Py_ssize_t n_src = positions.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    sums_arr = np.full(n_src, np.nan)
    vars_arr = np.full(n_src, np.nan)
    areas_arr = np.full(n_src, np.nan)
    overlap_arr = np.zeros(n_src, dtype=np.uint8)
    starts_arr = np.zeros(n_src, dtype=np.intp)
    cdef double[::1] sums = sums_arr
    cdef double[::1] sum_vars = vars_arr
    cdef double[::1] areas = areas_arr
    cdef unsigned char[::1] overlap = overlap_arr
    cdef Py_ssize_t[::1] starts = starts_arr

    cdef bint has_error = error is not None
    cdef bint has_mask = mask is not None
    cdef bint has_seg = segmentation is not None
    cdef bint has_bkg = local_bkg is not None
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
        rect_vertices(half_width_out, half_height_out, cos_theta,
                      sin_theta, poly_x_out, poly_y_out)
        bbox_dx_out = (half_width_out * fabs(cos_theta)
                       + half_height_out * fabs(sin_theta))
        bbox_dy_out = (half_width_out * fabs(sin_theta)
                       + half_height_out * fabs(cos_theta))
        if shape_code == _RECTANGULAR_ANNULUS:
            rect_vertices(half_width_in, half_height_in, cos_theta,
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
    cdef Py_ssize_t ixmin, iymin, grid_nx, grid_ny, area
    cdef Py_ssize_t six, siy, xm, ym, ccx = 0, ccy = 0, mseg
    cdef double cx, cy, lbk = 0.0
    cdef double ixmin_d, ixmax_d, iymin_d, iymax_d
    cdef double gxmin, gxmax, gymin, gymax
    cdef double dx, dy, pixel_radius, norm
    cdef double pxmin, pymin, frac, err_val, sum_val, var_val, area_val, val
    cdef double errsq = 0.0
    cdef Py_ssize_t total = 0, spos = 0

    # Pass 1 (only when emitting the packed member buffers): compute the
    # clipped bounding-box area per source (an upper bound on the number
    # of contributing pixels) to size and offset the packed buffers.
    # This is pure arithmetic; no pixel walk is performed here.
    if emit_sum:
        with nogil:
            for k in range(n_src):
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

    cdef Py_ssize_t sum_cap = total if emit_sum else 0
    sum_values_arr = np.empty(sum_cap, dtype=np.float64)
    sum_fracs_arr = np.empty(sum_cap, dtype=np.float64)
    sum_errsq_arr = np.empty(sum_cap, dtype=np.float64)
    scounts_arr = np.zeros(n_src if emit_sum else 0, dtype=np.intp)
    cdef double[::1] sum_values = sum_values_arr
    cdef double[::1] sum_fracs = sum_fracs_arr
    cdef double[::1] sum_errsq = sum_errsq_arr
    cdef Py_ssize_t[::1] scounts = scounts_arr

    with nogil:
        for k in range(n_src):
            cx = positions[k, 0]
            cy = positions[k, 1]
            if has_bkg:
                lbk = local_bkg[k]
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
            area_val = 0.0
            spos = starts[k]
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
                        frac = _circular_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, r_in,
                            r_out, use_exact, subpixels)
                    elif shape_code == _ELLIPSE:
                        frac = _ellipse_pixel_frac(
                            pxmin, pymin, dx, dy, norm, rx_out, ry_out,
                            cos_theta, sin_theta, use_exact, subpixels)
                    elif shape_code == _ELLIPTICAL_ANNULUS:
                        frac = _elliptical_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, norm, rx_in, ry_in,
                            rx_out, ry_out, cos_theta, sin_theta,
                            use_exact, subpixels)
                    elif shape_code == _RECTANGLE:
                        frac = _rect_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius,
                            half_width_out, half_height_out, cos_theta,
                            sin_theta, bbox_dx_out, bbox_dy_out,
                            poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                            buf_b_x, buf_b_y, use_exact, subpixels)
                    elif shape_code == _RECTANGULAR_ANNULUS:
                        frac = _rectangular_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius,
                            half_width_in, half_height_in,
                            half_width_out, half_height_out,
                            cos_theta, sin_theta, bbox_dx_in, bbox_dy_in,
                            bbox_dx_out, bbox_dy_out, poly_x_in, poly_y_in,
                            poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                            buf_b_x, buf_b_y, use_exact, subpixels)
                    else:
                        frac = _polygon_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius,
                            poly_x, poly_y, n_poly, pedge_nx, pedge_ny,
                            pedge_c, is_poly_convex, pbuf_a_x, pbuf_a_y,
                            pbuf_b_x, pbuf_b_y, poly_buf_size, use_exact,
                            subpixels)

                    # Annulus fractions are a difference of two shapes,
                    # so floating-point noise can leave a boundary
                    # pixel's fraction a tiny (nonnegative) value. The
                    # mask-based path weights every nonzero-fraction
                    # pixel, so match it with ``!= 0`` here.
                    if frac != 0.0:
                        val = data[siy, six] - lbk
                        sum_val += val * frac
                        area_val += frac
                        if has_error:
                            err_val = error[siy, six]
                            errsq = err_val * err_val
                            var_val += errsq * frac
                        if emit_sum:
                            sum_values[spos] = val
                            sum_fracs[spos] = frac
                            sum_errsq[spos] = errsq if has_error else 0.0
                            spos += 1

            sums[k] = sum_val
            areas[k] = area_val
            if has_error:
                sum_vars[k] = var_val
            if emit_sum:
                scounts[k] = spos - starts[k]

    return (sums_arr, vars_arr, areas_arr, overlap_arr.view(bool),
            starts_arr, sum_values_arr, sum_fracs_arr, sum_errsq_arr,
            scounts_arr)
