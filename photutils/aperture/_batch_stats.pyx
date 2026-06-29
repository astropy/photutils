# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to provide a batch driver for aperture statistics.

For each source position, this module walks the aperture bounding box
once and, using exactly the same per-pixel overlap arithmetic as the
`photutils.geometry` grid functions (and the
`~photutils.aperture._batch_photometry` driver), gathers the unmasked
"center"-method pixel values into a single packed buffer and accumulates
the cheap streaming scalars (the ``sum_method`` aperture sum, error
variance, and area). The packed value buffer is then reduced lazily by
the higher tiers (moments, median, and other order statistics) without
creating per-source mask arrays or making per-source Python calls.

The main source loop runs without the GIL and uses no global mutable
state, so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from photutils.aperture._batch_overlap cimport (_circle_pixel_frac,
                                                _ellipse_pixel_frac,
                                                _polygon_pixel_frac,
                                                _rect_pixel_frac,
                                                _rect_vertices)
from photutils.geometry._polygon_overlap cimport convex_edge_normals

__all__ = ['batch_aperture_gather']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)
    double floor(double x)
    double ceil(double x)
    bint isfinite(double x)

# Aperture shape codes. These must stay in sync with the private
# ``_shape_params`` attributes defined by the aperture classes and with
# ``_batch_photometry``.
cdef enum:
    _CIRCLE = 0
    _CIRCULAR_ANNULUS = 1
    _ELLIPSE = 2
    _ELLIPTICAL_ANNULUS = 3
    _RECTANGLE = 4
    _RECTANGULAR_ANNULUS = 5
    _POLYGON = 6


cdef inline double _shape_frac(int shape_code, double pxmin, double pymin,
                               double dx, double dy, double pixel_radius,
                               double norm, double r_in, double r_out,
                               double rx_in, double ry_in, double rx_out,
                               double ry_out, double cos_theta,
                               double sin_theta, double hw_in, double hh_in,
                               double hw_out, double hh_out, double bdx_in,
                               double bdy_in, double bdx_out, double bdy_out,
                               double *poly_x_in, double *poly_y_in,
                               double *poly_x_out, double *poly_y_out,
                               double *buf_a_x, double *buf_a_y,
                               double *buf_b_x, double *buf_b_y,
                               double *poly_x, double *poly_y, int n_poly,
                               double *pedge_nx, double *pedge_ny,
                               double *pedge_c, int is_poly_convex,
                               double *pbuf_a_x, double *pbuf_a_y,
                               double *pbuf_b_x, double *pbuf_b_y,
                               int poly_buf_size, int use_exact,
                               int subpixels) noexcept nogil:
    """
    Per-pixel overlap fraction for any aperture shape, dispatching to the
    shared `_batch_overlap` helpers. Inlined into the per-pixel loop.
    """
    if shape_code == _CIRCLE:
        return _circle_pixel_frac(pxmin, pymin, dx, dy, pixel_radius,
                                  r_out, use_exact, subpixels)
    if shape_code == _CIRCULAR_ANNULUS:
        return (_circle_pixel_frac(pxmin, pymin, dx, dy, pixel_radius,
                                   r_out, use_exact, subpixels)
                - _circle_pixel_frac(pxmin, pymin, dx, dy, pixel_radius,
                                     r_in, use_exact, subpixels))
    if shape_code == _ELLIPSE:
        return _ellipse_pixel_frac(pxmin, pymin, dx, dy, norm, rx_out,
                                   ry_out, cos_theta, sin_theta, use_exact,
                                   subpixels)
    if shape_code == _ELLIPTICAL_ANNULUS:
        return (_ellipse_pixel_frac(pxmin, pymin, dx, dy, norm, rx_out,
                                    ry_out, cos_theta, sin_theta, use_exact,
                                    subpixels)
                - _ellipse_pixel_frac(pxmin, pymin, dx, dy, norm, rx_in,
                                      ry_in, cos_theta, sin_theta, use_exact,
                                      subpixels))
    if shape_code == _RECTANGLE:
        return _rect_pixel_frac(pxmin, pymin, dx, dy, pixel_radius, hw_out,
                                hh_out, cos_theta, sin_theta, bdx_out,
                                bdy_out, poly_x_out, poly_y_out, buf_a_x,
                                buf_a_y, buf_b_x, buf_b_y, use_exact,
                                subpixels)
    if shape_code == _RECTANGULAR_ANNULUS:
        return (_rect_pixel_frac(pxmin, pymin, dx, dy, pixel_radius, hw_out,
                                 hh_out, cos_theta, sin_theta, bdx_out,
                                 bdy_out, poly_x_out, poly_y_out, buf_a_x,
                                 buf_a_y, buf_b_x, buf_b_y, use_exact,
                                 subpixels)
                - _rect_pixel_frac(pxmin, pymin, dx, dy, pixel_radius, hw_in,
                                   hh_in, cos_theta, sin_theta, bdx_in,
                                   bdy_in, poly_x_in, poly_y_in, buf_a_x,
                                   buf_a_y, buf_b_x, buf_b_y, use_exact,
                                   subpixels))
    return _polygon_pixel_frac(pxmin, pymin, dx, dy, pixel_radius, poly_x,
                               poly_y, n_poly, pedge_nx, pedge_ny, pedge_c,
                               is_poly_convex, pbuf_a_x, pbuf_a_y, pbuf_b_x,
                               pbuf_b_y, poly_buf_size, use_exact, subpixels)


def batch_aperture_gather(const double[:, ::1] data,
                          const double[:, ::1] error,
                          const unsigned char[:, ::1] mask,
                          const double[:, ::1] positions, int shape_code,
                          const double[::1] params, double ext_x,
                          double ext_y, int sum_use_exact, int sum_subpixels,
                          const double[::1] local_bkg,
                          const Py_ssize_t[:, ::1] segmentation=None,
                          const Py_ssize_t[::1] labels=None,
                          int seg_method=0):
    """
    Gather aperture statistics inputs for many source positions.

    For each position, the aperture bounding box is computed exactly as
    in `photutils.aperture.BoundingBox.from_float`, and the per-pixel
    overlap fractions are computed with exactly the same arithmetic as
    the `photutils.geometry` grid functions, so the results match the
    mask-based `~photutils.aperture.ApertureStats` code path.

    Two aperture mask methods are evaluated per pixel: the "center"
    method (``use_exact=0``, ``subpixels=1``) selects the unmasked pixel
    values used for the order/moment statistics, and the input
    ``sum_method`` (``sum_use_exact``/``sum_subpixels``) is used for the
    aperture sum, error, and area.

    Pixels that are masked, non-finite, or excluded by the segmentation
    masking are skipped. The local background ``local_bkg[k]`` is
    subtracted from each pixel value of source ``k``.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array (background not yet subtracted).

    error : 2D ndarray of float64 (C-contiguous) or `None`
        The pixel-wise 1-sigma errors, same shape as ``data``.

    mask : 2D ndarray of uint8 (C-contiguous) or `None`
        A mask array where nonzero values indicate masked pixels.

    positions : 2D ndarray of float64 (C-contiguous)
        The ``(x, y)`` source positions with shape ``(n_sources, 2)``.

    shape_code : int
        The aperture shape code (see `_batch_photometry`).

    params : 1D ndarray of float64 (C-contiguous)
        The aperture shape parameters (see `_batch_photometry`).

    ext_x, ext_y : float
        The aperture bounding-box half-extents.

    sum_use_exact, sum_subpixels : int
        The translated ``sum_method`` overlap parameters.

    local_bkg : 1D ndarray of float64 (C-contiguous)
        The per-source local background to subtract.

    segmentation : 2D ndarray of intp (C-contiguous) or `None`
        The segmentation image for segmentation-based masking.

    labels : 1D ndarray of intp (C-contiguous) or `None`
        The per-source segmentation labels.

    seg_method : int
        The segmentation masking method code (see `_batch_photometry`).

    Returns
    -------
    values : 1D ndarray of float64
        The packed buffer of unmasked "center"-method pixel values
        (background subtracted). The values for source ``k`` occupy
        ``values[starts[k]:starts[k] + counts[k]]``.

    local_x, local_y : 1D ndarray of intp
        The packed pixel coordinates (relative to the clipped cutout
        origin) corresponding to ``values``, used for the image moments.

    starts : 1D ndarray of intp
        The start offset into ``values`` for each source.

    counts : 1D ndarray of intp
        The number of unmasked "center"-method pixels for each source
        (i.e., the center aperture area).

    sum_aper : 1D ndarray of float64
        The ``sum_method`` aperture sum. NaN where the bounding box does
        not overlap the data.

    var_aper : 1D ndarray of float64
        The ``sum_method`` aperture error variance (the square of the
        aperture sum error). NaN where ``error`` is `None` or there is
        no overlap.

    sum_area : 1D ndarray of float64
        The ``sum_method`` aperture area. NaN where there is no overlap.

    overlap : 1D ndarray of bool
        Whether the aperture bounding box overlaps with the data.
    """
    cdef Py_ssize_t n_src = positions.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    cdef bint has_error = error is not None
    cdef bint has_mask = mask is not None
    cdef bint has_seg = segmentation is not None
    cdef Py_ssize_t lbl = 0, seg_val

    starts_arr = np.zeros(n_src, dtype=np.intp)
    counts_arr = np.zeros(n_src, dtype=np.intp)
    sum_arr = np.full(n_src, np.nan)
    var_arr = np.full(n_src, np.nan)
    area_arr = np.full(n_src, np.nan)
    overlap_arr = np.zeros(n_src, dtype=np.uint8)
    cdef Py_ssize_t[::1] starts = starts_arr
    cdef Py_ssize_t[::1] counts = counts_arr
    cdef double[::1] sum_aper = sum_arr
    cdef double[::1] var_aper = var_arr
    cdef double[::1] sum_area = area_arr
    cdef unsigned char[::1] overlap = overlap_arr

    # Aperture shape parameters (constant over all source positions)
    cdef double r_in = 0.0, r_out = 0.0
    cdef double rx_in = 0.0, ry_in = 0.0, rx_out = 0.0, ry_out = 0.0
    cdef double theta = 0.0, cos_theta = 1.0, sin_theta = 0.0
    cdef double hw_in = 0.0, hh_in = 0.0, hw_out = 0.0, hh_out = 0.0
    cdef double bdx_in = 0.0, bdy_in = 0.0, bdx_out = 0.0, bdy_out = 0.0
    cdef double poly_x_in[4]
    cdef double poly_y_in[4]
    cdef double poly_x_out[4]
    cdef double poly_y_out[4]
    cdef double buf_a_x[32]
    cdef double buf_a_y[32]
    cdef double buf_b_x[32]
    cdef double buf_b_y[32]

    # Working buffers for arbitrary-polygon apertures (see
    # ``_batch_photometry``).
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
            hw_out = 0.5 * params[0]
            hh_out = 0.5 * params[1]
            theta = params[2]
        else:
            hw_in = 0.5 * params[0]
            hh_in = 0.5 * params[1]
            hw_out = 0.5 * params[2]
            hh_out = 0.5 * params[3]
            theta = params[4]

        cos_theta = cos(theta)
        sin_theta = sin(theta)
        _rect_vertices(hw_out, hh_out, cos_theta, sin_theta, poly_x_out,
                       poly_y_out)
        bdx_out = hw_out * fabs(cos_theta) + hh_out * fabs(sin_theta)
        bdy_out = hw_out * fabs(sin_theta) + hh_out * fabs(cos_theta)
        if shape_code == _RECTANGULAR_ANNULUS:
            _rect_vertices(hw_in, hh_in, cos_theta, sin_theta, poly_x_in,
                           poly_y_in)
            bdx_in = hw_in * fabs(cos_theta) + hh_in * fabs(sin_theta)
            bdy_in = hw_in * fabs(sin_theta) + hh_in * fabs(cos_theta)
    elif shape_code == _POLYGON:
        n_poly = params.shape[0] // 2
        if n_poly < 3 or 2 * n_poly != params.shape[0]:
            msg = ('polygon params must be the flattened (x, y) offsets '
                   'of at least 3 vertices')
            raise ValueError(msg)

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

        is_poly_convex = convex_edge_normals(poly_x, poly_y, n_poly,
                                             pedge_nx, pedge_ny, pedge_c)
    else:
        msg = f'Invalid shape_code: {shape_code}'
        raise ValueError(msg)

    cdef Py_ssize_t k, ix, iy, ix0, ix1, iy0, iy1
    cdef Py_ssize_t ixmin, iymin, grid_nx, grid_ny, area
    cdef Py_ssize_t six, siy, xm, ym, ccx = 0, ccy = 0, mseg
    cdef double cx, cy, lbk
    cdef double ixmin_d, ixmax_d, iymin_d, iymax_d
    cdef double gxmin, gxmax, gymin, gymax
    cdef double dx, dy, pixel_radius, norm
    cdef double pxmin, pymin, cfrac, sfrac, val, err_val
    cdef double s_sum, s_var, s_area
    cdef Py_ssize_t total = 0, pos

    # Pass 1: compute the clipped bounding-box area per source (an upper
    # bound on the number of "center"-method survivors) to size and
    # offset the packed value buffer. This is pure arithmetic; no pixel
    # walk is performed here.
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

    values_arr = np.empty(total, dtype=np.float64)
    lx_arr = np.empty(total, dtype=np.intp)
    ly_arr = np.empty(total, dtype=np.intp)
    cdef double[::1] values = values_arr
    cdef Py_ssize_t[::1] local_x = lx_arr
    cdef Py_ssize_t[::1] local_y = ly_arr

    # Pass 2: walk each bounding box once, gather center-method survivors
    # into the packed buffer, and accumulate the sum-method scalars.
    with nogil:
        for k in range(n_src):
            cx = positions[k, 0]
            cy = positions[k, 1]
            lbk = local_bkg[k]
            if has_seg:
                lbl = labels[k]
                if seg_method == 3:
                    if cx >= 0.0:
                        ccx = <Py_ssize_t>floor(cx + 0.5)
                    else:
                        ccx = <Py_ssize_t>ceil(cx - 0.5)
                    if cy >= 0.0:
                        ccy = <Py_ssize_t>floor(cy + 0.5)
                    else:
                        ccy = <Py_ssize_t>ceil(cy - 0.5)

            ixmin_d = floor(cx - ext_x + 0.5)
            ixmax_d = ceil(cx + ext_x + 0.5)
            iymin_d = floor(cy - ext_y + 0.5)
            iymax_d = ceil(cy + ext_y + 0.5)

            if (ixmin_d >= <double>nx_data or ixmax_d <= 0.0
                    or iymin_d >= <double>ny_data or iymax_d <= 0.0):
                continue
            overlap[k] = 1

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

            ixmin = <Py_ssize_t>ixmin_d
            iymin = <Py_ssize_t>iymin_d
            ix0 = <Py_ssize_t>fmax(ixmin_d, 0.0)
            ix1 = <Py_ssize_t>fmin(ixmax_d, <double>nx_data)
            iy0 = <Py_ssize_t>fmax(iymin_d, 0.0)
            iy1 = <Py_ssize_t>fmin(iymax_d, <double>ny_data)

            pos = starts[k]
            s_sum = 0.0
            s_var = 0.0
            s_area = 0.0
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

                    val = data[siy, six]
                    if not isfinite(val):  # NaN/inf data are masked
                        continue
                    val = val - lbk

                    pxmin = gxmin + (ix - ixmin) * dx

                    cfrac = _shape_frac(
                        shape_code, pxmin, pymin, dx, dy, pixel_radius,
                        norm, r_in, r_out, rx_in, ry_in, rx_out, ry_out,
                        cos_theta, sin_theta, hw_in, hh_in, hw_out, hh_out,
                        bdx_in, bdy_in, bdx_out, bdy_out, poly_x_in,
                        poly_y_in, poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                        buf_b_x, buf_b_y, poly_x, poly_y, n_poly, pedge_nx,
                        pedge_ny, pedge_c, is_poly_convex, pbuf_a_x,
                        pbuf_a_y, pbuf_b_x, pbuf_b_y, poly_buf_size, 0, 1)

                    if cfrac > 0.0:
                        values[pos] = val
                        local_x[pos] = ix - ix0
                        local_y[pos] = iy - iy0
                        pos += 1

                    sfrac = _shape_frac(
                        shape_code, pxmin, pymin, dx, dy, pixel_radius,
                        norm, r_in, r_out, rx_in, ry_in, rx_out, ry_out,
                        cos_theta, sin_theta, hw_in, hh_in, hw_out, hh_out,
                        bdx_in, bdy_in, bdx_out, bdy_out, poly_x_in,
                        poly_y_in, poly_x_out, poly_y_out, buf_a_x, buf_a_y,
                        buf_b_x, buf_b_y, poly_x, poly_y, n_poly, pedge_nx,
                        pedge_ny, pedge_c, is_poly_convex, pbuf_a_x,
                        pbuf_a_y, pbuf_b_x, pbuf_b_y, poly_buf_size,
                        sum_use_exact, sum_subpixels)

                    if sfrac > 0.0:
                        s_sum += val * sfrac
                        s_area += sfrac
                        if has_error:
                            err_val = error[siy, six]
                            s_var += err_val * err_val * sfrac

            counts[k] = pos - starts[k]
            sum_aper[k] = s_sum
            sum_area[k] = s_area
            if has_error:
                var_aper[k] = s_var

    return (values_arr, lx_arr, ly_arr, starts_arr, counts_arr, sum_arr,
            var_arr, area_arr, overlap_arr.view(bool))
