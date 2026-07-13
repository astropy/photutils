# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to provide a batch driver for aperture statistics.

For each source position, this module iterates over the aperture
bounding box once and, using exactly the same per-pixel overlap
arithmetic as the `photutils.geometry` grid functions (and the
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

from photutils.aperture._batch_overlap cimport (
    _CIRCLE, _CIRCULAR_ANNULUS, _ELLIPSE, _ELLIPTICAL_ANNULUS, _POLYGON,
    _RECTANGLE, _RECTANGULAR_ANNULUS, _circle_pixel_frac,
    _circular_annulus_pixel_frac, _ellipse_pixel_frac,
    _elliptical_annulus_pixel_frac, _polygon_pixel_frac,
    _presize_packed_offsets, _rect_pixel_frac, _rectangular_annulus_pixel_frac,
    _round_half_away, _source_grid_setup)
from photutils.geometry._polygon_overlap cimport (convex_edge_normals,
                                                  polygon_work_partition,
                                                  polygon_work_size)
from photutils.geometry.rectangle_overlap cimport rect_vertices

__all__ = ['batch_aperture_gather', 'batch_moments', 'batch_sort_values',
           'batch_order_stats', 'batch_mean_var', 'batch_mad',
           'batch_biweight', 'batch_gini', 'batch_sigma_clip_center',
           'batch_sigma_clip_sum']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double fabs(double x)
    const double NAN

cdef extern from "stdlib.h" nogil:
    void qsort(void *base, size_t nmemb, size_t size,
               int (*compar)(const void *, const void *))

# Scale factor that converts the median absolute deviation to a robust
# estimate of the standard deviation (1 / scipy.stats.norm.ppf(0.75)).
# This must match ``astropy.stats.mad_std``.
cdef double _MAD_STD_SCALE = 1.482602218505602


cdef int _cmp_double(const void *a, const void *b) noexcept nogil:
    cdef double da = (<const double *>a)[0]
    cdef double db = (<const double *>b)[0]
    if da < db:
        return -1
    if da > db:
        return 1
    return 0


cdef inline double _median_sorted(double *s, Py_ssize_t n) noexcept nogil:
    """
    Median of an ascending-sorted buffer.

    This matches ``np.median``.

    Parameters
    ----------
    s : double *
        The ascending-sorted buffer.

    n : Py_ssize_t
        The number of elements in ``s``.

    Returns
    -------
    result : double
        The median value.
    """
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# Sigma-clip center/scale function codes (must match the ``cenfunc`` and
# ``stdfunc`` mapping in ``ApertureStats``).
cdef enum:
    _CEN_MEDIAN = 0
    _CEN_MEAN = 1
    _STD_STD = 0
    _STD_MADSTD = 1


cdef inline void _sigma_clip_bounds(double *s, double *work, Py_ssize_t n,
                                    double sigma_lower, double sigma_upper,
                                    Py_ssize_t maxiters, int cenfunc_code,
                                    int stdfunc_code, double *out_min,
                                    double *out_max) noexcept nogil:
    """
    Compute the converged sigma-clip bounds for one source.

    This reproduces `astropy.stats.SigmaClip` for the no-axis, no-grow
    case: it iteratively narrows the kept range and returns the final
    lower and upper value bounds. A source survives the clip if its
    value ``v`` satisfies ``not (v < out_min) and not (v > out_max)``,
    so NaN bounds keep every value, matching astropy's degenerate
    empty-set behavior.

    Parameters
    ----------
    s : double *
        The ascending-sorted values, as ``s[0:n]``.

    work : double *
        A scratch buffer of at least ``n`` elements.

    n : Py_ssize_t
        The number of values in ``s``.

    sigma_lower, sigma_upper : double
        The lower and upper clipping limits in units of the scale.

    maxiters : Py_ssize_t
        The maximum number of clipping iterations, or a negative value
        to iterate until convergence.

    cenfunc_code : int
        The center function code (``_CEN_MEDIAN`` or ``_CEN_MEAN``).

    stdfunc_code : int
        The scale function code (``_STD_STD`` or ``_STD_MADSTD``).

    out_min, out_max : double *
        Output. The converged lower and upper value bounds.
    """
    cdef Py_ssize_t lo = 0, hi = n, new_lo, new_hi, cnt, i, iteration = 0
    cdef Py_ssize_t nchanged = 1
    cdef double cen, std, mu, ss, med2, minv, maxv

    minv = NAN
    maxv = NAN
    while nchanged != 0 and (maxiters < 0 or iteration < maxiters):
        iteration += 1
        cnt = hi - lo
        if cnt == 0:
            minv = NAN
            maxv = NAN
            break

        # Center.
        if cenfunc_code == _CEN_MEDIAN:
            cen = _median_sorted(&s[lo], cnt)
        else:
            mu = 0.0
            for i in range(lo, hi):
                mu += s[i]
            cen = mu / cnt

        # Scale.
        if stdfunc_code == _STD_STD:
            mu = 0.0
            for i in range(lo, hi):
                mu += s[i]
            mu = mu / cnt
            ss = 0.0
            for i in range(lo, hi):
                ss += (s[i] - mu) * (s[i] - mu)
            std = sqrt(ss / cnt)
        else:
            med2 = _median_sorted(&s[lo], cnt)
            for i in range(lo, hi):
                work[i] = fabs(s[i] - med2)
            qsort(&work[lo], <size_t>cnt, sizeof(double), &_cmp_double)
            std = _median_sorted(&work[lo], cnt) * _MAD_STD_SCALE

        minv = cen - std * sigma_lower
        maxv = cen + std * sigma_upper

        new_lo = lo
        while new_lo < hi and s[new_lo] < minv:
            new_lo += 1
        new_hi = hi
        while new_hi > new_lo and s[new_hi - 1] > maxv:
            new_hi -= 1
        nchanged = (hi - lo) - (new_hi - new_lo)
        lo = new_lo
        hi = new_hi

    out_min[0] = minv
    out_max[0] = maxv


def batch_aperture_gather(const double[:, ::1] data,
                          const unsigned char[:, ::1] mask,
                          const double[:, ::1] positions, int shape_code,
                          const double[::1] params, double ext_x,
                          double ext_y, const double[::1] local_bkg=None,
                          const Py_ssize_t[:, ::1] segmentation=None,
                          const Py_ssize_t[::1] labels=None,
                          int seg_method=0):
    """
    Gather the "center"-method pixel values for many source positions.

    For each position, the aperture bounding box is computed exactly as
    in `photutils.aperture.BoundingBox.from_float`, and the per-pixel
    overlap fractions are computed with exactly the same arithmetic as
    the `photutils.geometry` grid functions, so the results match the
    mask-based `~photutils.aperture.ApertureStats` code path.

    The "center" aperture mask method (``use_exact=0``, ``subpixels=1``)
    selects the unmasked pixel values used for the order and moment
    statistics; each survivor is packed into a contiguous buffer. The
    ``sum_method`` aperture sum, error, and area are computed separately
    by `~photutils.aperture._batch_photometry.batch_aperture_sums`.

    Pixels that are masked or excluded by the segmentation masking are
    skipped. Non-finite ``data`` pixels are handled by the caller, which
    folds them into ``mask`` (matching the mask-based path, which masks
    non-finite data before any segmentation correction). The local
    background ``local_bkg[k]`` is subtracted from each pixel value of
    source ``k``.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array (background not yet subtracted).

    mask : 2D ndarray of uint8 (C-contiguous) or `None`
        A mask array where nonzero values indicate masked pixels. The
        caller must also fold non-finite ``data`` pixels into this mask.

    positions : 2D ndarray of float64 (C-contiguous)
        The ``(x, y)`` source positions with shape ``(n_sources, 2)``.

    shape_code : int
        The aperture shape code (see `_batch_photometry`).

    params : 1D ndarray of float64 (C-contiguous)
        The aperture shape parameters (see `_batch_photometry`).

    ext_x, ext_y : float
        The aperture bounding-box half-extents.

    local_bkg : 1D ndarray of float64 (C-contiguous) or `None`
        The per-source local background to subtract. If `None`, no
        background is subtracted.

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

    local_x, local_y : 1D ndarray of int32
        The packed pixel coordinates (relative to the clipped cutout
        origin) corresponding to ``values``, used for the image moments.
        Cutout coordinates always fit in 32 bits, and the narrower
        dtype reduces the packed-buffer memory footprint.

    starts : 1D ndarray of intp
        The start offset into ``values`` for each source.

    counts : 1D ndarray of intp
        The number of unmasked "center"-method pixels for each source
        (i.e., the center aperture area).

    overlap : 1D ndarray of bool
        Whether the aperture bounding box overlaps with the data.
    """
    cdef Py_ssize_t n_src = positions.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    cdef bint has_mask = mask is not None
    cdef bint has_bkg = local_bkg is not None
    cdef bint has_seg = segmentation is not None
    cdef Py_ssize_t lbl = 0, seg_val

    starts_arr = np.zeros(n_src, dtype=np.intp)
    counts_arr = np.zeros(n_src, dtype=np.intp)
    overlap_arr = np.zeros(n_src, dtype=np.uint8)
    cdef Py_ssize_t[::1] starts = starts_arr
    cdef Py_ssize_t[::1] counts = counts_arr
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
        rect_vertices(hw_out, hh_out, cos_theta, sin_theta, poly_x_out,
                      poly_y_out)
        bdx_out = hw_out * fabs(cos_theta) + hh_out * fabs(sin_theta)
        bdy_out = hw_out * fabs(sin_theta) + hh_out * fabs(cos_theta)
        if shape_code == _RECTANGULAR_ANNULUS:
            rect_vertices(hw_in, hh_in, cos_theta, sin_theta, poly_x_in,
                          poly_y_in)
            bdx_in = hw_in * fabs(cos_theta) + hh_in * fabs(sin_theta)
            bdy_in = hw_in * fabs(sin_theta) + hh_in * fabs(cos_theta)
    elif shape_code == _POLYGON:
        n_poly = params.shape[0] // 2
        if n_poly < 3 or 2 * n_poly != params.shape[0]:
            msg = ('polygon params must be the flattened (x, y) offsets '
                   'of at least 3 vertices')
            raise ValueError(msg)

        # Single numpy block (kept alive by ``poly_work``) whose sizing
        # and layout are shared with ``polygon_overlap_grid`` via
        # ``polygon_work_size``/``polygon_work_partition``.
        poly_work = np.empty(polygon_work_size(n_poly, &poly_buf_size),
                             dtype=np.float64)
        polygon_work_partition(&poly_work[0], n_poly, poly_buf_size,
                               &poly_x, &poly_y, &pbuf_a_x, &pbuf_a_y,
                               &pbuf_b_x, &pbuf_b_y, &pedge_nx, &pedge_ny,
                               &pedge_c)
        for pk in range(n_poly):
            poly_x[pk] = params[2 * pk]
            poly_y[pk] = params[2 * pk + 1]

        is_poly_convex = convex_edge_normals(poly_x, poly_y, n_poly,
                                             pedge_nx, pedge_ny, pedge_c)
    else:
        msg = f'Invalid shape_code: {shape_code}'
        raise ValueError(msg)

    cdef Py_ssize_t k, ix, iy, ix0, ix1, iy0, iy1
    cdef Py_ssize_t ixmin, iymin
    cdef Py_ssize_t six, siy, xm, ym, ccx = 0, ccy = 0, mseg
    cdef double cx, cy, lbk
    cdef double gxmin, gymin
    cdef double dx, dy, pixel_radius, norm
    cdef double pxmin, pymin, cfrac
    cdef Py_ssize_t total = 0, pos

    # Pass 1: size and offset the packed value buffer from the
    # per-source clipped bounding-box areas (an upper bound on the
    # number of "center"-method survivors). This performs only
    # bounding-box arithmetic; it does not iterate over or evaluate
    # individual pixels.
    with nogil:
        total = _presize_packed_offsets(positions, ext_x, ext_y, nx_data,
                                        ny_data, starts)

    values_arr = np.empty(total, dtype=np.float64)
    lx_arr = np.empty(total, dtype=np.int32)
    ly_arr = np.empty(total, dtype=np.int32)
    cdef double[::1] values = values_arr
    cdef int[::1] local_x = lx_arr
    cdef int[::1] local_y = ly_arr

    # Pass 2: iterate over each bounding box once and gather the
    # "center"-method survivors into the packed buffer.
    with nogil:
        for k in range(n_src):
            cx = positions[k, 0]
            cy = positions[k, 1]
            lbk = 0.0
            if has_bkg:
                lbk = local_bkg[k]
            if has_seg:
                lbl = labels[k]
                if seg_method == 3:
                    # Center pixel for the symmetric 'correct' mirror
                    ccx = _round_half_away(cx)
                    ccy = _round_half_away(cy)

            # Bounding box, overlap test, and pixel grid, replicated
            # from the mask-based path (see ``_source_grid_setup``).
            if not _source_grid_setup(cx, cy, ext_x, ext_y, nx_data,
                                      ny_data, &gxmin, &gymin, &dx, &dy,
                                      &pixel_radius, &norm, &ixmin,
                                      &iymin, &ix0, &ix1, &iy0, &iy1):
                continue
            overlap[k] = 1

            # Non-finite ``data`` pixels are folded into ``mask`` by the
            # caller, so pixel values are loaded lazily (only for pixels
            # that actually contribute).
            pos = starts[k]
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

                    pxmin = gxmin + (ix - ixmin) * dx
                    if shape_code == _CIRCLE:
                        cfrac = _circle_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, r_out,
                            0, 1)
                    elif shape_code == _CIRCULAR_ANNULUS:
                        cfrac = _circular_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, r_in,
                            r_out, 0, 1)
                    elif shape_code == _ELLIPSE:
                        cfrac = _ellipse_pixel_frac(
                            pxmin, pymin, dx, dy, norm, rx_out, ry_out,
                            cos_theta, sin_theta, 0, 1)
                    elif shape_code == _ELLIPTICAL_ANNULUS:
                        cfrac = _elliptical_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, norm, rx_in, ry_in,
                            rx_out, ry_out, cos_theta, sin_theta, 0, 1)
                    elif shape_code == _RECTANGLE:
                        cfrac = _rect_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, hw_out,
                            hh_out, cos_theta, sin_theta, bdx_out,
                            bdy_out, poly_x_out, poly_y_out, buf_a_x,
                            buf_a_y, buf_b_x, buf_b_y, 0, 1)
                    elif shape_code == _RECTANGULAR_ANNULUS:
                        cfrac = _rectangular_annulus_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, hw_in,
                            hh_in, hw_out, hh_out, cos_theta, sin_theta,
                            bdx_in, bdy_in, bdx_out, bdy_out, poly_x_in,
                            poly_y_in, poly_x_out, poly_y_out, buf_a_x,
                            buf_a_y, buf_b_x, buf_b_y, 0, 1)
                    else:
                        cfrac = _polygon_pixel_frac(
                            pxmin, pymin, dx, dy, pixel_radius, poly_x,
                            poly_y, n_poly, pedge_nx, pedge_ny, pedge_c,
                            is_poly_convex, pbuf_a_x, pbuf_a_y, pbuf_b_x,
                            pbuf_b_y, poly_buf_size, 0, 1)

                    if cfrac > 0.0:
                        values[pos] = data[siy, six] - lbk
                        local_x[pos] = <int>(ix - ix0)
                        local_y[pos] = <int>(iy - iy0)
                        pos += 1
            counts[k] = pos - starts[k]

    return (values_arr, lx_arr, ly_arr, starts_arr, counts_arr,
            overlap_arr.view(bool))


def batch_moments(const double[::1] values, const int[::1] local_x,
                  const int[::1] local_y,
                  const Py_ssize_t[::1] starts,
                  const Py_ssize_t[::1] counts, const double[::1] cx,
                  const double[::1] cy):
    """
    Compute image moments up to 3rd order from a packed value buffer.

    For each source ``k``, the ``(4, 4)`` moment matrix is

    .. math::

        M_{ij} = \\sum_p (y_p - c_y)^i \\, v_p \\, (x_p - c_x)^j

    where the sum is over the packed pixels ``p`` of source ``k`` (the
    slice ``[starts[k]:starts[k] + counts[k]]``), ``v_p`` is the pixel
    value, ``(x_p, y_p)`` are its cutout coordinates, and ``(c_x, c_y) =
    (cx[k], cy[k])``. This uses exactly the same definition as
    `photutils.utils._moments._image_moments`, so passing ``cx = cy =
    0`` gives the raw moments and passing the cutout centroid gives the
    central moments.

    Parameters
    ----------
    values : 1D ndarray of float64
        The packed pixel values (see ``batch_aperture_gather``).

    local_x, local_y : 1D ndarray of int32
        The packed pixel cutout coordinates.

    starts, counts : 1D ndarray of intp
        The per-source start offset into ``values`` and the per-source
        pixel count.

    cx, cy : 1D ndarray of float64
        The per-source ``(x, y)`` moment center.

    Returns
    -------
    moments : 3D ndarray of float64
        The ``(n_sources, 4, 4)`` image moments. Sources with no pixels
        have all-zero moments.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    moments_arr = np.zeros((n_src, 4, 4), dtype=np.float64)
    cdef double[:, :, ::1] moments = moments_arr

    cdef Py_ssize_t k, p, i, j, start, count
    cdef double cxk, cyk, val, dxp, dyp
    cdef double px[4]
    cdef double py[4]

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            cxk = cx[k]
            cyk = cy[k]
            for p in range(start, start + count):
                val = values[p]
                dxp = <double>local_x[p] - cxk
                dyp = <double>local_y[p] - cyk
                px[0] = 1.0
                px[1] = dxp
                px[2] = dxp * dxp
                px[3] = px[2] * dxp
                py[0] = 1.0
                py[1] = dyp
                py[2] = dyp * dyp
                py[3] = py[2] * dyp
                for i in range(4):
                    for j in range(4):
                        moments[k, i, j] += py[i] * val * px[j]

    return moments_arr


def batch_sort_values(const double[::1] values,
                      const Py_ssize_t[::1] starts,
                      const Py_ssize_t[::1] counts):
    """
    Sort each source's packed pixel values in ascending order.

    For each source ``k`` the slice ``[starts[k]:starts[k] + counts[k]]``
    of ``values`` is copied to the output buffer and sorted ascending.
    The sorted buffer is shared by the order statistics (``min``,
    ``max``, ``median``), ``mad_std``, and the biweight estimators, so
    the per-source sort is performed only once.

    Parameters
    ----------
    values : 1D ndarray of float64
        The packed pixel values (see ``batch_aperture_gather``). All
        values are assumed finite.

    starts, counts : 1D ndarray of intp
        The per-source start offset into ``values`` and the per-source
        pixel count.

    Returns
    -------
    sorted_values : 1D ndarray of float64
        The packed buffer whose per-source slice is sorted ascending.
        The ``starts`` and ``counts`` offsets are unchanged.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    out_arr = np.empty(values.shape[0], dtype=np.float64)
    cdef double[::1] out = out_arr

    cdef Py_ssize_t k, i, start, count

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            for i in range(count):
                out[start + i] = values[start + i]
            qsort(&out[start], <size_t>count, sizeof(double), &_cmp_double)

    return out_arr


def batch_order_stats(const double[::1] sorted_values,
                      const Py_ssize_t[::1] starts,
                      const Py_ssize_t[::1] counts):
    """
    Reduce a sorted packed buffer to the per-source min, max, and median.

    The conventions match `numpy.min`, `numpy.max`, and `numpy.median`.
    Sources with no pixels (``counts[k] == 0``) are set to NaN.

    Parameters
    ----------
    sorted_values : 1D ndarray of float64
        The packed per-source ascending-sorted values (see
        ``batch_sort_values``).

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    Returns
    -------
    vmin, vmax, median : 1D ndarray of float64
        The per-source statistics, each of shape ``(n_sources,)``.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    vmin_arr = np.full(n_src, np.nan, dtype=np.float64)
    vmax_arr = np.full(n_src, np.nan, dtype=np.float64)
    median_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] vmin = vmin_arr
    cdef double[::1] vmax = vmax_arr
    cdef double[::1] median = median_arr

    cdef Py_ssize_t k, start, count

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            vmin[k] = sorted_values[start]
            vmax[k] = sorted_values[start + count - 1]
            median[k] = _median_sorted(&sorted_values[start], count)

    return (vmin_arr, vmax_arr, median_arr)


def batch_mean_var(const double[::1] values,
                   const Py_ssize_t[::1] starts,
                   const Py_ssize_t[::1] counts):
    """
    Reduce a packed value buffer to the per-source mean and variance.

    The conventions match `numpy.mean` and `numpy.var` (``ddof=0``). No
    sort is required. Sources with no pixels (``counts[k] == 0``) are set
    to NaN.

    Parameters
    ----------
    values : 1D ndarray of float64
        The packed pixel values (see ``batch_aperture_gather``).

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    Returns
    -------
    mean, var : 1D ndarray of float64
        The per-source mean and population variance.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    mean_arr = np.full(n_src, np.nan, dtype=np.float64)
    var_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] mean = mean_arr
    cdef double[::1] var = var_arr

    cdef Py_ssize_t k, i, start, count
    cdef double m, ssum, dsum, dval

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            ssum = 0.0
            for i in range(start, start + count):
                ssum += values[i]
            m = ssum / count
            mean[k] = m
            dsum = 0.0
            for i in range(start, start + count):
                dval = values[i] - m
                dsum += dval * dval
            var[k] = dsum / count

    return (mean_arr, var_arr)


def batch_mad(const double[::1] sorted_values,
              const Py_ssize_t[::1] starts,
              const Py_ssize_t[::1] counts):
    """
    Reduce a sorted packed buffer to the per-source median absolute
    deviation.

    The returned value is the (unscaled) median of the absolute
    deviations from the per-source median. Multiply by the ``mad_std``
    scale factor to obtain `astropy.stats.mad_std`. Sources with no
    pixels (``counts[k] == 0``) are set to NaN.

    Parameters
    ----------
    sorted_values : 1D ndarray of float64
        The packed per-source ascending-sorted values (see
        ``batch_sort_values``).

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    Returns
    -------
    mad : 1D ndarray of float64
        The per-source unscaled median absolute deviation.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    mad_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] mad = mad_arr

    cdef Py_ssize_t maxn = 0, k
    for k in range(n_src):
        if counts[k] > maxn:
            maxn = counts[k]
    if maxn == 0:
        return mad_arr

    work_arr = np.empty(maxn, dtype=np.float64)
    cdef double[::1] w = work_arr

    cdef Py_ssize_t i, start, count
    cdef double med

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            med = _median_sorted(&sorted_values[start], count)
            for i in range(count):
                w[i] = fabs(sorted_values[start + i] - med)
            qsort(&w[0], <size_t>count, sizeof(double), &_cmp_double)
            mad[k] = _median_sorted(&w[0], count)

    return mad_arr


def batch_biweight(const double[::1] sorted_values,
                   const Py_ssize_t[::1] starts,
                   const Py_ssize_t[::1] counts,
                   const double[::1] median, const double[::1] mad):
    """
    Reduce a packed buffer to the per-source biweight location and
    midvariance.

    The conventions match `astropy.stats.biweight_location` (``c=6``)
    and `astropy.stats.biweight_midvariance` (``c=9``). The per-source
    ``median`` and (unscaled) ``mad`` are reused from ``batch_order_stats``
    and ``batch_mad``. Sources with no pixels (``counts[k] == 0``) are set
    to NaN.

    Parameters
    ----------
    sorted_values : 1D ndarray of float64
        The packed per-source ascending-sorted values (see
        ``batch_sort_values``). The estimators are order-independent, so
        the sorted buffer is reused directly.

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    median : 1D ndarray of float64
        The per-source median (see ``batch_order_stats``).

    mad : 1D ndarray of float64
        The per-source unscaled median absolute deviation (see
        ``batch_mad``).

    Returns
    -------
    biloc, bivar : 1D ndarray of float64
        The per-source biweight location and midvariance.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    biloc_arr = np.full(n_src, np.nan, dtype=np.float64)
    bivar_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] biloc = biloc_arr
    cdef double[::1] bivar = bivar_arr

    cdef Py_ssize_t k, i, start, count
    cdef double med, madk, dval, uu, weight, num, den, u2, omu, f1, f2

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            med = median[k]
            madk = mad[k]

            if madk == 0.0:
                biloc[k] = med
                bivar[k] = 0.0
                continue

            num = 0.0
            den = 0.0
            f1 = 0.0
            f2 = 0.0
            for i in range(start, start + count):
                dval = sorted_values[i] - med
                uu = dval / (6.0 * madk)
                if fabs(uu) < 1.0:
                    weight = (1.0 - uu * uu)
                    weight = weight * weight
                    num += dval * weight
                    den += weight
                u2 = dval / (9.0 * madk)
                u2 = u2 * u2
                if u2 < 1.0:
                    omu = 1.0 - u2
                    f1 += dval * dval * omu * omu * omu * omu
                    f2 += omu * (1.0 - 5.0 * u2)
            biloc[k] = med + num / den
            bivar[k] = count * f1 / (f2 * f2)

    return (biloc_arr, bivar_arr)


def batch_gini(const double[::1] values,
               const Py_ssize_t[::1] starts,
               const Py_ssize_t[::1] counts):
    """
    Reduce a packed value buffer to the per-source Gini coefficient.

    The convention matches `photutils.morphology.gini`. Sources with no
    pixels (``counts[k] == 0``) are set to NaN.

    Parameters
    ----------
    values : 1D ndarray of float64
        The packed pixel values (see ``batch_aperture_gather``). The
        values need not be sorted; the absolute values are sorted
        internally.

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    Returns
    -------
    gini : 1D ndarray of float64
        The per-source Gini coefficient.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    gini_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] gini = gini_arr

    cdef Py_ssize_t maxn = 0, k
    for k in range(n_src):
        if counts[k] > maxn:
            maxn = counts[k]
    if maxn == 0:
        return gini_arr

    work_arr = np.empty(maxn, dtype=np.float64)
    cdef double[::1] w = work_arr

    cdef Py_ssize_t i, start, count
    cdef double gsum, meanabs, norm

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]

            if count == 1:
                gini[k] = 0.0
                continue

            for i in range(count):
                w[i] = fabs(values[start + i])
            qsort(&w[0], <size_t>count, sizeof(double), &_cmp_double)
            gsum = 0.0
            for i in range(count):
                gsum += w[i]
            meanabs = gsum / count
            norm = meanabs * count * (count - 1)
            if norm == 0.0:
                gini[k] = 0.0
            else:
                gsum = 0.0
                for i in range(count):
                    gsum += (2.0 * i + 1.0 - count) * w[i]
                gini[k] = gsum / norm

    return gini_arr


def batch_sigma_clip_center(const double[::1] values,
                            const int[::1] local_x,
                            const int[::1] local_y,
                            const Py_ssize_t[::1] starts,
                            const Py_ssize_t[::1] counts,
                            double sigma_lower, double sigma_upper,
                            Py_ssize_t maxiters, int cenfunc_code,
                            int stdfunc_code):
    """
    Sigma-clip each source's packed center buffer.

    For each source the packed pixel values (and their cutout
    coordinates) are sigma-clipped following `astropy.stats.SigmaClip`
    (no-axis, no-grow case), and the surviving pixels are written to a
    new packed buffer that reuses the input ``starts`` offsets. The
    output can be fed directly to the packed-buffer reductions (e.g.,
    ``batch_mean_var``, ``batch_sort_values``, and ``batch_moments``).

    Parameters
    ----------
    values : 1D ndarray of float64
        The packed center pixel values (see ``batch_aperture_gather``).

    local_x, local_y : 1D ndarray of int32
        The packed pixel cutout coordinates.

    starts, counts : 1D ndarray of intp
        The per-source start offset and pixel count.

    sigma_lower, sigma_upper : float
        The lower and upper clipping limits in units of the scale.

    maxiters : int
        The maximum number of clipping iterations, or a negative value
        to iterate until convergence.

    cenfunc_code, stdfunc_code : int
        The center and scale function codes (``0`` = median/std,
        ``1`` = mean/mad_std).

    Returns
    -------
    values, local_x, local_y, starts, counts : 1D ndarray
        The clipped packed buffer (``starts`` is the input array,
        unchanged) and the per-source surviving pixel counts.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    out_values_arr = np.empty(values.shape[0], dtype=np.float64)
    out_lx_arr = np.empty(values.shape[0], dtype=np.int32)
    out_ly_arr = np.empty(values.shape[0], dtype=np.int32)
    counts2_arr = np.zeros(n_src, dtype=np.intp)
    cdef double[::1] out_values = out_values_arr
    cdef int[::1] out_lx = out_lx_arr
    cdef int[::1] out_ly = out_ly_arr
    cdef Py_ssize_t[::1] counts2 = counts2_arr

    cdef Py_ssize_t maxn = 0, k
    for k in range(n_src):
        if counts[k] > maxn:
            maxn = counts[k]
    if maxn == 0:
        return (out_values_arr, out_lx_arr, out_ly_arr, np.asarray(starts),
                counts2_arr)

    sort_arr = np.empty(maxn, dtype=np.float64)
    work_arr = np.empty(maxn, dtype=np.float64)
    cdef double[::1] s = sort_arr
    cdef double[::1] w = work_arr

    cdef Py_ssize_t start, count, i, j
    cdef double minv, maxv, v

    with nogil:
        for k in range(n_src):
            count = counts[k]
            if count == 0:
                continue
            start = starts[k]
            for i in range(count):
                s[i] = values[start + i]
            qsort(&s[0], <size_t>count, sizeof(double), &_cmp_double)
            _sigma_clip_bounds(&s[0], &w[0], count, sigma_lower, sigma_upper,
                               maxiters, cenfunc_code, stdfunc_code,
                               &minv, &maxv)
            j = 0
            for i in range(count):
                v = values[start + i]
                if not (v < minv) and not (v > maxv):
                    out_values[start + j] = v
                    out_lx[start + j] = local_x[start + i]
                    out_ly[start + j] = local_y[start + i]
                    j += 1
            counts2[k] = j

    return (out_values_arr, out_lx_arr, out_ly_arr, np.asarray(starts),
            counts2_arr)


def batch_sigma_clip_sum(const double[::1] sum_values,
                         const double[::1] sum_fracs,
                         const double[::1] sum_errsq,
                         const Py_ssize_t[::1] starts,
                         const Py_ssize_t[::1] sum_counts,
                         double sigma_lower, double sigma_upper,
                         Py_ssize_t maxiters, int cenfunc_code,
                         int stdfunc_code, int has_error):
    """
    Sigma-clip each source's packed ``sum_method`` member buffer.

    The sum members are sigma-clipped following
    `astropy.stats.SigmaClip` (using the unweighted, background
    subtracted pixel values), and the aperture sum, error variance, and
    area are recomputed over the surviving members. Sources with no sum
    members have NaN outputs.

    Parameters
    ----------
    sum_values, sum_fracs, sum_errsq : 1D ndarray of float64
        The packed ``sum_method`` member values, overlap fractions, and
        squared total errors (see
        `~photutils.aperture._batch_photometry.batch_aperture_sums`
        with ``emit_sum``).

    starts, sum_counts : 1D ndarray of intp
        The per-source start offset and member count.

    sigma_lower, sigma_upper : float
        The lower and upper clipping limits in units of the scale.

    maxiters : int
        The maximum number of clipping iterations, or a negative value
        to iterate until convergence.

    cenfunc_code, stdfunc_code : int
        The center and scale function codes.

    has_error : int
        Whether the squared errors are available.

    Returns
    -------
    sum_aper, var_aper, sum_area : 1D ndarray of float64
        The clipped aperture sum, error variance, and area. NaN where
        there are no sum members.
    """
    cdef Py_ssize_t n_src = starts.shape[0]
    sum_arr = np.full(n_src, np.nan, dtype=np.float64)
    var_arr = np.full(n_src, np.nan, dtype=np.float64)
    area_arr = np.full(n_src, np.nan, dtype=np.float64)
    cdef double[::1] sum_aper = sum_arr
    cdef double[::1] var_aper = var_arr
    cdef double[::1] sum_area = area_arr

    cdef Py_ssize_t maxn = 0, k
    for k in range(n_src):
        if sum_counts[k] > maxn:
            maxn = sum_counts[k]
    if maxn == 0:
        return (sum_arr, var_arr, area_arr)

    sort_arr = np.empty(maxn, dtype=np.float64)
    work_arr = np.empty(maxn, dtype=np.float64)
    cdef double[::1] s = sort_arr
    cdef double[::1] w = work_arr

    cdef Py_ssize_t start, count, i
    cdef double minv, maxv, v, frac, s_sum, s_area, s_var

    with nogil:
        for k in range(n_src):
            count = sum_counts[k]
            if count == 0:
                continue
            start = starts[k]
            for i in range(count):
                s[i] = sum_values[start + i]
            qsort(&s[0], <size_t>count, sizeof(double), &_cmp_double)
            _sigma_clip_bounds(&s[0], &w[0], count, sigma_lower, sigma_upper,
                               maxiters, cenfunc_code, stdfunc_code,
                               &minv, &maxv)
            s_sum = 0.0
            s_area = 0.0
            s_var = 0.0
            for i in range(count):
                v = sum_values[start + i]
                if not (v < minv) and not (v > maxv):
                    frac = sum_fracs[start + i]
                    s_sum += v * frac
                    s_area += frac
                    if has_error:
                        s_var += sum_errsq[start + i] * frac
            sum_aper[k] = s_sum
            sum_area[k] = s_area
            if has_error:
                var_aper[k] = s_var

    return (sum_arr, var_arr, area_arr)
