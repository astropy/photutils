# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Batch Cython drivers for `~photutils.segmentation.SourceCatalog`
per-source computations.

Each driver accumulates per-pixel contributions directly into
per-source outputs without generating per-source cutouts or making
per-source Python calls. The per-pixel segmentation masking (mask or
mirror-correct) uses the same helper as the aperture batch drivers, so
the semantics match the previous cutout-based code paths.

The source loops run without the GIL and use no global mutable state,
so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from photutils.aperture._batch_overlap cimport _resolve_seg_pixel

__all__ = ['batch_centroid_win']


cdef extern from "math.h" nogil:
    double exp(double x)
    double sqrt(double x)
    bint isfinite(double x)
    double NAN


cdef void _centroid_win_source(const double *data, const double *error,
                               const unsigned char *mask,
                               const Py_ssize_t *segm,
                               Py_ssize_t nx_data, Py_ssize_t ny_data,
                               Py_ssize_t label, double xcen0,
                               double ycen0, double sigma,
                               int seg_method, bint compute_err,
                               Py_ssize_t max_aper_size,
                               double *out) noexcept nogil:
    """
    Compute the raw windowed-centroid quantities for a single source.

    Replicates ``SourceCatalog._iterate_centroid_win``: an iterative
    Gaussian-weighted centroid within a binary circular window of
    radius ``4 * sigma``, with masked pixels contributing zero
    and neighbor-source pixels excluded or mirror-corrected per
    ``seg_method``. Writes the 10 output columns ``(xcen, ycen,
    weighted_flux, cen_mom_xx, cen_mom_yy, cen_mom_xy, err_sum,
    err_var_x, err_var_y, err_cov_xy)`` into ``out``.
    """
    cdef double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)
    cdef double radius = 4.0 * sigma
    cdef double radius_sq = radius * radius
    # Truncation of (radius + 1.5) replicates the Python int() call
    cdef Py_ssize_t bbox_halfsize = <Py_ssize_t>(radius + 1.5)
    cdef Py_ssize_t full_n = 2 * bbox_halfsize + 1

    out[0] = NAN
    out[1] = NAN
    out[2] = 0.0
    out[3] = 0.0
    out[4] = 0.0
    out[5] = 0.0
    out[6] = NAN
    out[7] = NAN
    out[8] = NAN
    out[9] = NAN
    if full_n * full_n > max_aper_size:
        return

    cdef double xcen = xcen0
    cdef double ycen = ycen0
    cdef double dcen = 1.0
    cdef double weighted_flux = 0.0
    cdef double dx_mom = 0.0
    cdef double dy_mom = 0.0
    cdef Py_ssize_t iter_ = 0
    cdef Py_ssize_t ixmin, ixmax, iymin, iymax, x0, x1, y0, y1
    cdef Py_ssize_t ccx, ccy, ix, iy, six, siy
    cdef Py_ssize_t n_seg = 0, n_unc = 0
    cdef double dx, dy, rr2, w, v, e, wv, wsq
    cdef double sumw, sumwx, sumwy
    # State of the last completed accumulation, for the final
    # moment/error pass (the window may go empty on a later iteration,
    # in which case the previous iteration's window is what the Python
    # implementation reused)
    cdef bint have_window = False
    cdef double xc_last = 0.0, yc_last = 0.0
    cdef double dxm_last = 0.0, dym_last = 0.0
    cdef Py_ssize_t lx0 = 0, lx1 = 0, ly0 = 0, ly1 = 0
    cdef Py_ssize_t lccx = 0, lccy = 0
    cdef double sxx, syy, sxy, esum, evx, evy, ecxy

    while iter_ < 16 and dcen > 0.0001:
        # Truncation of (xcen + 0.5) replicates the Python int() call
        # (including truncation toward zero for negative centers)
        ixmin = <Py_ssize_t>(xcen + 0.5) - bbox_halfsize
        ixmax = ixmin + full_n
        iymin = <Py_ssize_t>(ycen + 0.5) - bbox_halfsize
        iymax = iymin + full_n
        x0 = ixmin if ixmin > 0 else 0
        x1 = ixmax if ixmax < nx_data else nx_data
        y0 = iymin if iymin > 0 else 0
        y1 = iymax if iymax < ny_data else ny_data
        if y0 >= y1 or x0 >= x1:
            xcen = NAN
            ycen = NAN
            break
        ccx = <Py_ssize_t>(xcen + 0.5)
        ccy = <Py_ssize_t>(ycen + 0.5)

        sumw = 0.0
        sumwx = 0.0
        sumwy = 0.0
        for iy in range(y0, y1):
            for ix in range(x0, x1):
                dx = ix - xcen
                dy = iy - ycen
                rr2 = dx * dx + dy * dy
                if rr2 > radius_sq:
                    continue
                if mask[iy * nx_data + ix] != 0:
                    continue
                six = ix
                siy = iy
                if (seg_method != 0
                        and not _resolve_seg_pixel(
                            segm, mask, nx_data, seg_method, label,
                            ix, iy, x0, x1, y0, y1, ccx, ccy,
                            &six, &siy, &n_seg, &n_unc)):
                    continue
                v = data[siy * nx_data + six]
                w = exp(rr2 * inv_2sigma2)
                wv = v * w
                sumw += wv
                sumwx += wv * dx
                sumwy += wv * dy

        weighted_flux = sumw
        # 0/0 yields NaN (cdivision), matching the suppressed NumPy
        # RuntimeWarning path; a NaN dcen ends the loop
        dx_mom = sumwx / sumw
        dy_mom = sumwy / sumw
        dcen = sqrt(dx_mom * dx_mom + dy_mom * dy_mom)

        have_window = True
        xc_last = xcen
        yc_last = ycen
        dxm_last = dx_mom
        dym_last = dy_mom
        lx0 = x0
        lx1 = x1
        ly0 = y0
        ly1 = y1
        lccx = ccx
        lccy = ccy

        xcen += dx_mom * 2.0
        ycen += dy_mom * 2.0
        iter_ += 1

    out[0] = xcen
    out[1] = ycen
    out[2] = weighted_flux
    if not (isfinite(weighted_flux) and weighted_flux > 0
            and have_window):
        return

    # Final pass over the last completed window using the pre-update
    # center: windowed central 2nd-order moments and raw error sums.
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    esum = 0.0
    evx = 0.0
    evy = 0.0
    ecxy = 0.0
    for iy in range(ly0, ly1):
        for ix in range(lx0, lx1):
            dx = ix - xc_last
            dy = iy - yc_last
            rr2 = dx * dx + dy * dy
            if rr2 > radius_sq:
                continue
            if mask[iy * nx_data + ix] != 0:
                continue
            six = ix
            siy = iy
            if (seg_method != 0
                    and not _resolve_seg_pixel(
                        segm, mask, nx_data, seg_method, label,
                        ix, iy, lx0, lx1, ly0, ly1, lccx, lccy,
                        &six, &siy, &n_seg, &n_unc)):
                continue
            v = data[siy * nx_data + six]
            w = exp(rr2 * inv_2sigma2)
            wv = v * w
            sxx += wv * dx * dx
            syy += wv * dy * dy
            sxy += wv * dx * dy
            if compute_err:
                e = error[siy * nx_data + six]
                wsq = w * w
                esum += wsq * e * e
                evx += wsq * e * e * dx * dx
                evy += wsq * e * e * dy * dy
                ecxy += wsq * e * e * dx * dy

    out[3] = sxx / weighted_flux - dxm_last * dxm_last
    out[4] = syy / weighted_flux - dym_last * dym_last
    out[5] = sxy / weighted_flux - dxm_last * dym_last
    if compute_err:
        out[6] = esum
        out[7] = evx
        out[8] = evy
        out[9] = ecxy


def batch_centroid_win(const double[:, ::1] data, *,
                       const double[:, ::1] error,
                       const unsigned char[:, ::1] mask,
                       const Py_ssize_t[:, ::1] segm,
                       const Py_ssize_t[::1] labels,
                       const double[::1] xcen0,
                       const double[::1] ycen0,
                       const double[::1] sigma,
                       const unsigned char[::1] skip,
                       int seg_method, int compute_err,
                       Py_ssize_t max_aper_size):
    """
    Compute windowed-centroid quantities for many sources in one call.

    For each source, an iterative Gaussian-weighted centroid is computed
    within a binary circular window of radius ``4 * sigma`` centered
    on the current centroid estimate. The iteration, the integer
    bounding-box arithmetic, the per-pixel masking, and the final
    windowed second-order moments and raw error sums replicate the
    previous per-source Python implementation exactly.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array.

    error : 2D ndarray of float64 (C-contiguous) or `None`
        The pixel-wise 1-sigma errors. Must have the same shape as
        ``data``. Required (not `None`) if ``compute_err`` is nonzero.

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where nonzero values indicate masked (excluded)
        pixels. Must have the same shape as ``data``. Bit 1 (value 1)
        marks input-masked pixels and bit 2 (value 2) marks non-finite
        data pixels folded into the mask by the caller. Any nonzero
        value excludes the pixel and prevents it from being used as a
        mirror pixel.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``data``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape ``(n_sources,)``.

    xcen0, ycen0 : 1D ndarray of float64 (C-contiguous)
        The initial (isophotal) centroids for each source, with shape
        ``(n_sources,)``.

    sigma : 1D ndarray of float64 (C-contiguous)
        The Gaussian weighting sigma for each source, with shape
        ``(n_sources,)``.

    skip : 1D ndarray of uint8 (C-contiguous)
        Nonzero for sources that cannot have a meaningful windowed
        centroid (e.g., a non-finite half-light radius or a non-finite
        initial centroid), with shape ``(n_sources,)``.

    seg_method : int
        The segmentation masking method:

        * 0: disables masking
        * 1: excludes neighbor-source pixels
             (``(seg > 0) & (seg != label)``)
        * 3: replaces neighbor-source pixels with the values mirrored
             across the (rounded) window center (the symmetric
             ``'correct'`` method). A neighbor pixel whose mirror falls
             outside the clipped window, is itself a neighbor, or is
             masked is excluded instead of replaced.

    compute_err : int
        If nonzero, also accumulate the raw (unnormalized) weighted
        error sums.

    max_aper_size : Py_ssize_t
        The maximum number of pixels in the (unclipped) window. Sources
        with a larger window are skipped (out-of-memory guard).

    Returns
    -------
    result : 2D ndarray of float64
        The raw per-source results with shape ``(n_sources, 10)`` and
        columns ``(xcen, ycen, weighted_flux, cen_mom_xx, cen_mom_yy,
        cen_mom_xy, err_sum, err_var_x, err_var_y, err_cov_xy)``. The
        error columns are NaN unless ``compute_err`` is nonzero.
        Skipped sources (including those rejected by ``max_aper_size``)
        get the row ``(nan, nan, 0, 0, 0, 0, nan, nan, nan, nan)``.

    Raises
    ------
    ValueError
        If ``compute_err`` is nonzero and ``error`` is `None`.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]
    cdef bint has_error = error is not None

    if compute_err and not has_error:
        msg = 'error must be provided when compute_err is set'
        raise ValueError(msg)

    results_arr = np.empty((n_src, 10))
    cdef double[:, ::1] results = results_arr
    cdef const double *err_ptr = NULL
    if has_error:
        err_ptr = &error[0, 0]

    cdef Py_ssize_t i
    with nogil:
        for i in range(n_src):
            if skip[i]:
                results[i, 0] = NAN
                results[i, 1] = NAN
                results[i, 2] = 0.0
                results[i, 3] = 0.0
                results[i, 4] = 0.0
                results[i, 5] = 0.0
                results[i, 6] = NAN
                results[i, 7] = NAN
                results[i, 8] = NAN
                results[i, 9] = NAN
                continue
            _centroid_win_source(&data[0, 0], err_ptr, &mask[0, 0],
                                 &segm[0, 0], nx_data, ny_data,
                                 labels[i], xcen0[i], ycen0[i],
                                 sigma[i], seg_method, compute_err,
                                 max_aper_size, &results[i, 0])
    return results_arr
