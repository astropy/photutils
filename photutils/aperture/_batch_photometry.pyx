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

from photutils.aperture._batch_results import BatchApertureSums

from photutils.aperture._batch_overlap cimport (
    _CIRCLE, _CIRCULAR_ANNULUS, _ELLIPSE, _ELLIPTICAL_ANNULUS, _POLYGON,
    _RECTANGLE, _RECTANGULAR_ANNULUS, _circle_pixel_frac,
    _circular_annulus_pixel_frac, _ellipse_pixel_frac,
    _elliptical_annulus_pixel_frac, _polygon_pixel_frac,
    _presize_packed_offsets, _rect_pixel_frac, _rectangular_annulus_pixel_frac,
    _resolve_seg_pixel, _round_half_away, _source_grid_setup)
from photutils.geometry._polygon_overlap cimport (convex_edge_normals,
                                                  polygon_work_partition,
                                                  polygon_work_size)
from photutils.geometry.rectangle_overlap cimport rect_vertices

__all__ = ['batch_aperture_sums']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double fabs(double x)
    double ceil(double x)
    double sqrt(double x)
    bint isfinite(double x)

# Python-level aliases of the shape codes (the ``cdef enum`` is shared
# with ``_batch_stats`` via ``_batch_overlap.pxd``)
SHAPE_CIRCLE = _CIRCLE
SHAPE_CIRCULAR_ANNULUS = _CIRCULAR_ANNULUS
SHAPE_ELLIPSE = _ELLIPSE
SHAPE_ELLIPTICAL_ANNULUS = _ELLIPTICAL_ANNULUS
SHAPE_RECTANGLE = _RECTANGLE
SHAPE_RECTANGULAR_ANNULUS = _RECTANGULAR_ANNULUS
SHAPE_POLYGON = _POLYGON

# Column indices of the per-source ``flag_counts`` arrays returned by
# the batch drivers (see ``batch_aperture_sums`` and
# ``photutils.aperture._batch_stats.batch_aperture_gather``)
FLAG_COL_N_PIXELS = 0
FLAG_COL_MASKED = 1
FLAG_COL_NONFINITE_DATA = 2
FLAG_COL_NONFINITE_ERROR = 3
FLAG_COL_SEG = 4
FLAG_COL_UNCORRECTED = 5
FLAG_COL_VALID = 6
FLAG_COL_BBOX_CLIPPED = 7
FLAG_COL_SEG_MASKED = 8
FLAG_COL_UNCORRECTED_MASKED = 9

# The total number of ``flag_counts`` columns. Every producer of a
# per-source flag-count array must allocate this many columns
N_FLAG_COLS = 10


cdef int _check_params(int shape_code, const double[::1] params,
                       const double[:, ::1] params_per_source,
                       Py_ssize_t n_src, int emit_sum) except -1:
    """
    Validate the shared and per-source aperture shape parameters.

    Exactly one of ``params`` and ``params_per_source`` must be input.
    This is kept out of ``batch_aperture_sums`` so that its error
    handling stays out of that function's per-pixel loop.

    Parameters
    ----------
    shape_code : int
        The aperture shape code (see the module-level ``SHAPE_*``
        constants).

    params : const double[::1]
        The shared aperture shape parameters, or `None`.

    params_per_source : const double[:, ::1]
        The per-source aperture shape parameters, or `None`.

    n_src : Py_ssize_t
        The number of source positions.

    emit_sum : int
        Whether the packed per-pixel member buffers are emitted.

    Returns
    -------
    result : int
        Zero. A `ValueError` is raised if the parameters are invalid.
    """
    if params_per_source is None:
        if params is None:
            msg = 'params must be given when params_per_source is None'
            raise ValueError(msg)
        return 0

    if params is not None:
        msg = 'give params or params_per_source, not both'
        raise ValueError(msg)
    if shape_code not in (_CIRCLE, _ELLIPSE):
        msg = 'params_per_source supports only circle and ellipse shapes'
        raise ValueError(msg)
    if emit_sum:
        msg = 'params_per_source does not support emit_sum'
        raise ValueError(msg)
    if params_per_source.shape[0] != n_src:
        msg = 'params_per_source must have one row per position'
        raise ValueError(msg)
    if ((shape_code == _CIRCLE and params_per_source.shape[1] != 1)
            or (shape_code == _ELLIPSE
                and params_per_source.shape[1] != 3)):
        msg = 'params_per_source has the wrong column count'
        raise ValueError(msg)
    return 0


def batch_aperture_sums(const double[:, ::1] data, const double[:, ::1] error,
                        const unsigned char[:, ::1] mask,
                        const double[:, ::1] positions, int shape_code,
                        const double[::1] params, double ext_x, double ext_y,
                        double off_x, double off_y,
                        int use_exact, int subpixels,
                        const Py_ssize_t[:, ::1] segmentation=None,
                        const Py_ssize_t[::1] labels=None, int seg_method=0,
                        const double[::1] local_bkg=None, int emit_sum=0,
                        const double[:, ::1] params_per_source=None):
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
        pixels. Must have the same shape as ``data``. For the flag
        counts, bit 1 (value 1) marks input-masked pixels and bit 2
        (value 2) marks non-finite data pixels folded into the mask by
        the caller. Any nonzero value excludes the pixel.

    positions : 2D ndarray of float64 (C-contiguous)
        The (x, y) source positions with shape ``(n_sources, 2)``.

    shape_code : int
        The aperture shape code: 0=circle, 1=circular annulus,
        2=ellipse, 3=elliptical annulus, 4=rectangle, 5=rectangular
        annulus, 6=polygon (see the module-level ``SHAPE_*`` constants).

    params : 1D ndarray of float64 (C-contiguous) or `None`
        The aperture shape parameters, shared by all source positions.
        Must be `None` if ``params_per_source`` is input:

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
        and y directions (i.e., ``Aperture._xy_extents``). These values
        are ignored (and recomputed for each source) if
        ``params_per_source`` is input.

    off_x, off_y : float
        The (x, y) offset of the bounding-box center from each source
        position (i.e., ``Aperture._xy_bbox_offset``). This is zero for
        an aperture whose bounding box is centered on its position.

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

    params_per_source : 2D ndarray of float64 (C-contiguous) or `None`
        The per-source aperture shape parameters with shape
        ``(n_sources, k)``, used instead of the shared ``params``
        vector (which must then be `None`). Only the circle (``k`` =
        1, ``(r,)``) and ellipse (``k`` = 3, ``(a, b, theta)``) shapes
        are supported, and ``emit_sum`` must be zero. The bounding-box
        half-extents are computed from each source's own parameters, so
        the ``ext_x`` and ``ext_y`` inputs are ignored.

    Returns
    -------
    result : `~photutils.aperture._batch_results.BatchApertureSums`
        A named tuple with the following fields (in order).

    sums : 1D ndarray of float64
        The aperture sums. NaN where the aperture bounding box does not
        overlap the data.

    sum_vars : 1D ndarray of float64
        The aperture error variances (the quadrature sum of the pixel
        variances weighted by the squared overlap fractions). NaN where
        ``error`` is `None` or the aperture bounding box does not
        overlap the data. The caller takes the square root to obtain the
        error.

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

    flag_counts : 2D ndarray of intp
        Per-source pixel counts for the quality flags, with one row per
        source and the columns given by the module-level ``FLAG_COL_*``
        constants:

        * ``FLAG_COL_N_PIXELS``: the number of nonzero-fraction pixels
          inside the data
        * ``FLAG_COL_MASKED``: the number of those pixels that are
          input-masked
        * ``FLAG_COL_NONFINITE_DATA``: the number of those pixels that
          are non-finite in the data (masked or unmasked)
        * ``FLAG_COL_NONFINITE_ERROR``: the number of those pixels that
          are non-finite in the error (masked or unmasked)
        * ``FLAG_COL_SEG``: the number of those unmasked pixels that
          were excluded or corrected by the segmentation masking
        * ``FLAG_COL_UNCORRECTED``: the number of those unmasked pixels
          that could not be corrected by the segmentation masking
        * ``FLAG_COL_VALID``: the number of contributing pixels
          (unmasked, finite, and not excluded by segmentation)
        * ``FLAG_COL_BBOX_CLIPPED``: a 0/1 indicator of whether the
          bounding box is clipped by a data edge
        * ``FLAG_COL_SEG_MASKED``: the number of those masked pixels
          that lie on a neighboring source
        * ``FLAG_COL_UNCORRECTED_MASKED``: the number of those masked
          pixels whose mirror pixel was also unavailable

        Masked pixels are excluded before the segmentation masking is
        applied, so they are counted in the ``FLAG_COL_SEG_MASKED``
        and ``FLAG_COL_UNCORRECTED_MASKED`` columns instead of the
        ``FLAG_COL_SEG`` and ``FLAG_COL_UNCORRECTED`` columns. This lets
        callers that treat the mask and neighbor overlays independently
        recover the full neighbor-pixel counts.

        The ``FLAG_COL_NONFINITE_DATA`` and ``FLAG_COL_NONFINITE_ERROR``
        columns are nonzero when non-finite data or error values
        contribute to the aperture. Pixels marked non-finite in the mask
        plane (bit 2) are counted exactly, while unmasked non-finite
        contributions are detected from the accumulated sums as a 0/1
        indicator. Rows are all zero where the aperture bounding box
        does not overlap the data.

    weights_out : 1D ndarray of uint8
        A per-source 0/1 indicator of whether the aperture has one
        or more nonzero-fraction pixels outside the data. This is
        the precise outside-weight test, computed only for the
        sources whose bounding box is clipped by a data edge (the
        ``FLAG_COL_BBOX_CLIPPED`` column). It is exactly zero for
        unclipped sources and for sources whose bounding box does not
        overlap the data at all.
    """
    cdef Py_ssize_t n_src = positions.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    sums_arr = np.full(n_src, np.nan)
    vars_arr = np.full(n_src, np.nan)
    areas_arr = np.full(n_src, np.nan)
    overlap_arr = np.zeros(n_src, dtype=np.uint8)
    starts_arr = np.zeros(n_src, dtype=np.intp)
    fcounts_arr = np.zeros((n_src, N_FLAG_COLS), dtype=np.intp)
    wout_arr = np.zeros(n_src, dtype=np.uint8)
    cdef double[::1] sums = sums_arr
    cdef double[::1] sum_vars = vars_arr
    cdef double[::1] areas = areas_arr
    cdef unsigned char[::1] overlap = overlap_arr
    cdef Py_ssize_t[::1] starts = starts_arr
    cdef Py_ssize_t[:, ::1] fcounts = fcounts_arr
    cdef unsigned char[::1] weights_out = wout_arr

    cdef bint has_error = error is not None
    cdef bint has_mask = mask is not None
    cdef bint has_seg = segmentation is not None
    cdef bint has_bkg = local_bkg is not None
    cdef Py_ssize_t lbl = 0

    # Base pointers for the C-contiguous segmentation and mask planes,
    # used by the shared per-pixel segmentation helper
    cdef const Py_ssize_t *seg_ptr = NULL
    cdef const unsigned char *mask_ptr = NULL
    if has_seg:
        seg_ptr = &segmentation[0, 0]
    if has_mask:
        mask_ptr = &mask[0, 0]

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

    cdef bint has_psrc = params_per_source is not None
    _check_params(shape_code, params, params_per_source, n_src, emit_sum)

    if not has_psrc:
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

            # One-time convexity test; convex polygons use an
            # interior/exterior fast path in ``_polygon_pixel_frac``.
            is_poly_convex = convex_edge_normals(poly_x, poly_y, n_poly,
                                                 pedge_nx, pedge_ny, pedge_c)
        else:
            msg = f'Invalid shape_code: {shape_code}'
            raise ValueError(msg)

    cdef Py_ssize_t k, ix, iy, ix0, ix1, iy0, iy1
    cdef Py_ssize_t sx0, sx1, sy0, sy1
    cdef bint outside, found_out
    cdef Py_ssize_t ixmin, iymin
    cdef Py_ssize_t six, siy, ccx = 0, ccy = 0
    cdef double cx, cy, lbk = 0.0
    cdef double gxmin, gymin
    cdef double dx, dy, pixel_radius, norm
    cdef double pxmin, pymin, frac, err_val, sum_val, var_val, area_val, val
    cdef double errsq = 0.0
    cdef Py_ssize_t total = 0, spos = 0
    cdef Py_ssize_t n_pix, n_masked, n_nonfin, n_nonfin_err
    cdef Py_ssize_t n_seg_px, n_uncorr, n_valid, clipped
    cdef Py_ssize_t n_seg_masked, n_unc_masked
    cdef Py_ssize_t ixmax_full, iymax_full
    cdef unsigned char mbits

    # Pass 1 (only when emitting the packed member buffers): size and
    # offset the packed buffers from the per-source clipped bounding-box
    # areas. This performs only bounding-box arithmetic; it does not
    # iterate over or evaluate individual pixels.
    if emit_sum:
        with nogil:
            total = _presize_packed_offsets(positions, ext_x, ext_y,
                                            off_x, off_y, nx_data, ny_data,
                                            starts)

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

            # Per-source shape parameters and bounding-box half-extents
            if has_psrc:
                if shape_code == _CIRCLE:
                    r_out = params_per_source[k, 0]
                    ext_x = r_out
                    ext_y = r_out
                else:
                    rx_out = params_per_source[k, 0]
                    ry_out = params_per_source[k, 1]
                    theta = params_per_source[k, 2]
                    cos_theta = cos(theta)
                    sin_theta = sin(theta)
                    ext_x = sqrt(rx_out * rx_out * cos_theta
                                 * cos_theta + ry_out * ry_out
                                 * sin_theta * sin_theta)
                    ext_y = sqrt(rx_out * rx_out * sin_theta
                                 * sin_theta + ry_out * ry_out
                                 * cos_theta * cos_theta)

            if has_bkg:
                lbk = local_bkg[k]
            if has_seg:
                lbl = labels[k]
                if seg_method == 3:
                    # Center pixel for the symmetric 'correct' mirror
                    ccx = _round_half_away(cx)
                    ccy = _round_half_away(cy)

            # Bounding box, overlap test, and pixel grid, replicated
            # from the mask-based path (see ``_source_grid_setup``); the
            # sums stay NaN when there is no overlap.
            if not _source_grid_setup(cx, cy, ext_x, ext_y, off_x, off_y,
                                      nx_data, ny_data, &gxmin, &gymin,
                                      &dx, &dy, &pixel_radius, &norm, &ixmin,
                                      &iymin, &ix0, &ix1, &iy0, &iy1):
                continue
            overlap[k] = 1

            # Whether the bounding box is clipped by a data edge. Only
            # these sources can have aperture weights outside the data,
            # so only for them is the pixel loop widened below to the
            # full (unclipped) bounding box.
            ixmax_full = <Py_ssize_t>ceil(cx + off_x + ext_x + 0.5)
            iymax_full = <Py_ssize_t>ceil(cy + off_y + ext_y + 0.5)
            if (ixmin < ix0 or ixmax_full > ix1
                    or iymin < iy0 or iymax_full > iy1):
                clipped = 1  # bounding box clipped by data edge
                sx0 = ixmin
                sx1 = ixmax_full
                sy0 = iymin
                sy1 = iymax_full
            else:
                clipped = 0  # bounding box fully inside data
                sx0 = ix0
                sx1 = ix1
                sy0 = iy0
                sy1 = iy1

            sum_val = 0.0
            var_val = 0.0
            area_val = 0.0
            n_pix = 0
            n_masked = 0
            n_nonfin = 0
            n_nonfin_err = 0
            n_seg_px = 0
            n_uncorr = 0
            n_seg_masked = 0
            n_unc_masked = 0
            n_valid = 0
            spos = starts[k]
            found_out = 0
            for iy in range(sy0, sy1):
                # Once the outside-weight test has been answered, the
                # pixels outside the data can no longer change any
                # result, so the remaining rows are narrowed (or
                # skipped entirely) back to the part of the bounding
                # box that lies inside the data.
                if clipped and found_out:
                    if iy < iy0 or iy >= iy1:
                        continue
                    sx0 = ix0
                    sx1 = ix1

                pymin = gymin + (iy - iymin) * dy
                for ix in range(sx0, sx1):
                    # A pixel outside the data contributes to nothing
                    # but the outside-weight test, so it costs only
                    # this test once that test has been answered.
                    outside = clipped and not (iy0 <= iy < iy1
                                               and ix0 <= ix < ix1)
                    if outside and found_out:
                        continue

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

                    # A nonzero fraction outside the data answers the
                    # precise outside-weight test. A clipped bounding
                    # box does not by itself imply nonzero outside
                    # weights, because the aperture may not reach into
                    # the clipped-away rows or columns. The pixel is
                    # then skipped: its (out-of-range) coordinates must
                    # never reach the data accesses below.
                    if outside:
                        if frac > 0.0:
                            found_out = 1
                        continue

                    # Annulus fractions are a difference of two shapes,
                    # so floating-point noise can leave a boundary
                    # pixel's fraction a tiny negative value. Both the
                    # mask-based path (via ``np.maximum``) and the batch
                    # overlap functions clamp such fractions to zero, so
                    # only strictly positive fractions contribute here.
                    if frac <= 0.0:
                        continue

                    n_pix += 1

                    if has_mask:
                        mbits = mask[iy, ix]
                        if mbits != 0:
                            if mbits & 1:
                                n_masked += 1
                            elif mbits & 2:
                                n_nonfin += 1
                            # Masked pixels never reach the
                            # segmentation branch below, so count
                            # masked neighbor-segment pixels (and their
                            # mirror availability) here for callers
                            # that treat the mask and neighbor overlays
                            # independently. The helper is called only
                            # for its counter side effects. Its return
                            # value and resolved coordinates are unused.
                            if (has_seg and lbl != 0
                                    and seg_method != 0):
                                six = ix
                                siy = iy
                                _resolve_seg_pixel(
                                    seg_ptr, mask_ptr, nx_data,
                                    seg_method, lbl, ix, iy, ix0,
                                    ix1, iy0, iy1, ccx, ccy, &six,
                                    &siy, &n_seg_masked,
                                    &n_unc_masked)
                            continue
                    six = ix
                    siy = iy
                    if (has_seg and lbl != 0
                            and not _resolve_seg_pixel(
                                seg_ptr, mask_ptr, nx_data, seg_method,
                                lbl, ix, iy, ix0, ix1, iy0, iy1, ccx,
                                ccy, &six, &siy, &n_seg_px, &n_uncorr)):
                        continue

                    val = data[siy, six] - lbk
                    n_valid += 1
                    sum_val += val * frac
                    area_val += frac
                    if has_error:
                        err_val = error[siy, six]
                        errsq = err_val * err_val
                        var_val += errsq * frac * frac
                    if emit_sum:
                        sum_values[spos] = val
                        sum_fracs[spos] = frac
                        sum_errsq[spos] = errsq if has_error else 0.0
                        spos += 1

            # Unmasked non-finite data or error values corrupt the
            # accumulated sums, so their presence is detected from the
            # final sums (avoiding per-pixel finiteness tests in the hot
            # loop). Non-finite pixels folded into the mask plane (bit
            # 2) are counted exactly in the masked branch above.
            if not isfinite(sum_val):
                n_nonfin += 1
            if has_error and not isfinite(var_val):
                n_nonfin_err += 1

            weights_out[k] = found_out
            sums[k] = sum_val
            areas[k] = area_val
            if has_error:
                sum_vars[k] = var_val
            if emit_sum:
                scounts[k] = spos - starts[k]
            fcounts[k, 0] = n_pix
            fcounts[k, 1] = n_masked
            fcounts[k, 2] = n_nonfin
            fcounts[k, 3] = n_nonfin_err
            fcounts[k, 4] = n_seg_px
            fcounts[k, 5] = n_uncorr
            fcounts[k, 6] = n_valid
            fcounts[k, 7] = clipped
            fcounts[k, 8] = n_seg_masked
            fcounts[k, 9] = n_unc_masked

    return BatchApertureSums(
        sums=sums_arr, sum_vars=vars_arr, areas=areas_arr,
        overlap=overlap_arr.view(bool), starts=starts_arr,
        sum_values=sum_values_arr, sum_fracs=sum_fracs_arr,
        sum_errsq=sum_errsq_arr, sum_counts=scounts_arr,
        flag_counts=fcounts_arr, weights_out=wout_arr)
