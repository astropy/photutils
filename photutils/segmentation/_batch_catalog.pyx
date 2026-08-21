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

The moment kernels accumulate the raw and central spatial moments (and
the raw centroid error sums) over each source's segment bounding box,
using cutout-frame coordinates, so that the moment-based properties
never need per-source cutout arrays.

The fractional-flux radius solver additionally runs its bracketed
root-find entirely in C, using the ``brentq`` implementation exported
by ``scipy.optimize.cython_optimize``, so that no Python call is made
per root-finder iteration.

The source loops run without the GIL and use no global mutable state,
so this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from scipy.optimize.cython_optimize cimport brentq, zeros_full_output

from photutils.aperture._batch_overlap cimport (_circle_pixel_frac,
                                                _resolve_seg_pixel)

__all__ = ['batch_central_moments', 'batch_centroid_win',
           'batch_flux_radius_solve', 'batch_kron_radius',
           'batch_moment_err', 'batch_raw_moments']


cdef extern from "math.h" nogil:
    double exp(double x)
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double floor(double x)
    double fmax(double x, double y)
    bint isfinite(double x)
    double NAN


cdef int _check_length(Py_ssize_t n, Py_ssize_t n_src,
                       str name) except -1:
    """
    Validate the length of a per-source input array.

    This is kept out of the drivers so that their error handling stays
    out of the per-pixel loops. The drivers compile with
    ``boundscheck=False``, so a short per-source array would otherwise
    read out of bounds.

    Parameters
    ----------
    n : Py_ssize_t
        The length of the per-source array.

    n_src : Py_ssize_t
        The number of sources (the length of ``labels``).

    name : str
        The name of the per-source array, used in the error message.

    Returns
    -------
    result : int
        Zero. A `ValueError` is raised if the length is invalid.
    """
    if n != n_src:
        msg = f'{name} must have the same length as labels'
        raise ValueError(msg)
    return 0


cdef int _check_shape(Py_ssize_t ny, Py_ssize_t nx, Py_ssize_t ny_ref,
                      Py_ssize_t nx_ref, str name,
                      str ref_name) except -1:
    """
    Validate the shape of a 2D image-sized input array.

    This is kept out of the drivers so that their error handling stays
    out of the per-pixel loops. The drivers compile with
    ``boundscheck=False``, so a smaller array would otherwise read out
    of bounds.

    Parameters
    ----------
    ny, nx : Py_ssize_t
        The shape of the array being checked.

    ny_ref, nx_ref : Py_ssize_t
        The shape of the reference array.

    name : str
        The name of the array being checked, used in the error
        message.

    ref_name : str
        The name of the reference array, used in the error message.

    Returns
    -------
    result : int
        Zero. A `ValueError` is raised if the shape is invalid.
    """
    if ny != ny_ref or nx != nx_ref:
        msg = f'{name} must have the same shape as {ref_name}'
        raise ValueError(msg)
    return 0


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

    Replicates the previous per-source Python implementation: an
    iterative Gaussian-weighted centroid within a binary circular
    window of radius ``4 * sigma``, with masked pixels contributing
    zero and neighbor-source pixels excluded or mirror-corrected per
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
    # Write-only sinks required by the ``_resolve_seg_pixel``
    # signature; they accumulate across all iterations and the final
    # pass, so they are not a per-window count and are never read
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
        If ``compute_err`` is nonzero and ``error`` is `None`, if a
        per-source array does not have the same length as ``labels``,
        or if a 2D array does not have the same shape as ``data``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]
    cdef bint has_error = error is not None

    if compute_err and not has_error:
        msg = 'error must be provided when compute_err is set'
        raise ValueError(msg)

    _check_length(xcen0.shape[0], n_src, 'xcen0')
    _check_length(ycen0.shape[0], n_src, 'ycen0')
    _check_length(sigma.shape[0], n_src, 'sigma')
    _check_length(skip.shape[0], n_src, 'skip')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'data')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'data')
    if has_error:
        _check_shape(error.shape[0], error.shape[1], ny_data, nx_data,
                     'error', 'data')

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


cdef void _kron_radius_source(const double *data,
                              const unsigned char *mask,
                              const Py_ssize_t *segm,
                              Py_ssize_t nx_data, Py_ssize_t ny_data,
                              Py_ssize_t label, double xc, double yc,
                              double a, double b, double theta,
                              double cxx, double cxy, double cyy,
                              int seg_method, double scale,
                              double min_circ_radius,
                              Py_ssize_t max_aper_size,
                              double *out) noexcept nogil:
    """
    Accumulate the Kron radius numerator and denominator for a single
    source.

    Replicates the previous per-source Python implementation: the
    sums of ``data * r`` and ``data`` over the pixels whose centers
    fall inside the ellipse of elliptical radius ``scale`` (or the
    circle of radius ``min_circ_radius`` when both axes are zero),
    excluding masked pixels and handling neighbor-source pixels per
    ``seg_method``. Writes ``(numerator, denominator)`` into ``out``,
    or NaN for an undefined measurement (no minimum circular radius
    for a degenerate ellipse, an unreasonably large bounding box, or
    no overlap with the data).
    """
    out[0] = NAN
    out[1] = NAN

    cdef bint use_circular = (a == 0 and b == 0)
    cdef double half_w, half_h, cos_theta, sin_theta
    if use_circular:
        if min_circ_radius <= 0:
            return
        half_w = min_circ_radius
        half_h = min_circ_radius
    else:
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        half_w = sqrt(a * a * cos_theta * cos_theta
                      + b * b * sin_theta * sin_theta)
        half_h = sqrt(a * a * sin_theta * sin_theta
                      + b * b * cos_theta * cos_theta)

    # The bounding box is kept as integral doubles until it is
    # clipped, to avoid integer overflow for apertures far outside
    # the image
    cdef double ixmin_d = floor(xc - half_w + 0.5)
    cdef double ixmax_d = floor(xc + half_w + 0.5) + 1.0
    cdef double iymin_d = floor(yc - half_h + 0.5)
    cdef double iymax_d = floor(yc + half_h + 0.5) + 1.0
    if (ixmax_d - ixmin_d) * (iymax_d - iymin_d) > max_aper_size:
        return

    # Clip to the data
    cdef double x0_d = ixmin_d if ixmin_d > 0.0 else 0.0
    cdef double x1_d = ixmax_d if ixmax_d < nx_data else nx_data
    cdef double y0_d = iymin_d if iymin_d > 0.0 else 0.0
    cdef double y1_d = iymax_d if iymax_d < ny_data else ny_data
    if x0_d >= x1_d or y0_d >= y1_d:
        return
    cdef Py_ssize_t x0 = <Py_ssize_t>x0_d
    cdef Py_ssize_t x1 = <Py_ssize_t>x1_d
    cdef Py_ssize_t y0 = <Py_ssize_t>y0_d
    cdef Py_ssize_t y1 = <Py_ssize_t>y1_d

    # The cutout-frame center offsets and the mirror center of the
    # 'correct' method (truncation of the cutout center + 0.5
    # replicates the Python int() call)
    cdef double xoff = xc - <double>x0
    cdef double yoff = yc - <double>y0
    cdef Py_ssize_t ccx = <Py_ssize_t>(xoff + 0.5) + x0
    cdef Py_ssize_t ccy = <Py_ssize_t>(yoff + 0.5) + y0
    cdef double r_circ2 = min_circ_radius * min_circ_radius

    cdef double numerator = 0.0
    cdef double denominator = 0.0
    cdef Py_ssize_t ix, iy, six, siy
    # Write-only sinks required by the ``_resolve_seg_pixel``
    # signature
    cdef Py_ssize_t n_seg = 0, n_unc = 0
    cdef double xx, yy, rr_sq, rr, v
    for iy in range(y0, y1):
        yy = <double>(iy - y0) - yoff
        for ix in range(x0, x1):
            xx = <double>(ix - x0) - xoff
            rr_sq = cxx * xx * xx + cxy * xx * yy + cyy * yy * yy
            rr = sqrt(fmax(rr_sq, 0.0))
            if use_circular:
                if xx * xx + yy * yy > r_circ2:
                    continue
            elif rr > scale:
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
            numerator += v * rr
            denominator += v

    out[0] = numerator
    out[1] = denominator


def batch_kron_radius(const double[:, ::1] data, *,
                      const unsigned char[:, ::1] mask,
                      const Py_ssize_t[:, ::1] segm,
                      const Py_ssize_t[::1] labels,
                      const double[::1] xcen,
                      const double[::1] ycen,
                      const double[::1] semimajor,
                      const double[::1] semiminor,
                      const double[::1] theta,
                      const double[::1] cxx,
                      const double[::1] cxy,
                      const double[::1] cyy,
                      const unsigned char[::1] skip,
                      int seg_method, double scale,
                      double min_circ_radius,
                      Py_ssize_t max_aper_size):
    """
    Compute the Kron radius numerator and denominator for many sources
    in one call.

    For each source, the sums of ``data * r`` and ``data`` are
    accumulated over the pixels whose centers fall inside the ellipse
    of elliptical radius ``scale`` (in units of the isophotal
    ellipse), where ``r`` is the elliptical radius of each pixel, or
    inside the circle of radius ``min_circ_radius`` for a source whose
    elliptical axes are both zero. The bounding-box arithmetic, the
    per-pixel masking, and the neighbor handling replicate the
    previous per-source Python implementation exactly. The caller
    forms the Kron radius from the two sums.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array.

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

    xcen, ycen : 1D ndarray of float64 (C-contiguous)
        The source centroids, with shape ``(n_sources,)``.

    semimajor, semiminor : 1D ndarray of float64 (C-contiguous)
        The isophotal semimajor and semiminor axes (1-sigma), with
        shape ``(n_sources,)``. They are multiplied by ``scale`` to
        form the measurement ellipse.

    theta : 1D ndarray of float64 (C-contiguous)
        The ellipse orientation in radians, with shape
        ``(n_sources,)``.

    cxx, cxy, cyy : 1D ndarray of float64 (C-contiguous)
        The isophotal ellipse coefficients, with shape
        ``(n_sources,)``.

    skip : 1D ndarray of uint8 (C-contiguous)
        Nonzero for sources that cannot have a measured Kron radius
        (e.g., a completely masked source or a non-finite centroid or
        shape), with shape ``(n_sources,)``.

    seg_method : int
        The segmentation masking method:

        * 0: disables masking
        * 1: excludes neighbor-source pixels
             (``(seg > 0) & (seg != label)``)
        * 3: replaces neighbor-source pixels with the values mirrored
             across the (rounded) aperture center (the symmetric
             ``'correct'`` method). A neighbor pixel whose mirror falls
             outside the clipped bounding box, is itself a neighbor,
             or is masked contributes zero instead.

    scale : float
        The elliptical radius of the measurement ellipse, in units of
        the isophotal ellipse.

    min_circ_radius : float
        The radius of the measurement circle for sources whose
        elliptical axes are both zero. Such sources are undefined if
        it is not positive.

    max_aper_size : Py_ssize_t
        The maximum number of pixels in the (unclipped) bounding box.
        Sources with a larger bounding box are undefined
        (out-of-memory guard).

    Returns
    -------
    result : 2D ndarray of float64
        The per-source sums with shape ``(n_sources, 2)`` and columns
        ``(numerator, denominator)``. Skipped and undefined sources
        (including those rejected by ``max_aper_size`` and those whose
        bounding box does not overlap the data) get the row
        ``(nan, nan)``.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if a 2D array does not have the same shape as
        ``data``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = data.shape[0]
    cdef Py_ssize_t nx_data = data.shape[1]

    _check_length(xcen.shape[0], n_src, 'xcen')
    _check_length(ycen.shape[0], n_src, 'ycen')
    _check_length(semimajor.shape[0], n_src, 'semimajor')
    _check_length(semiminor.shape[0], n_src, 'semiminor')
    _check_length(theta.shape[0], n_src, 'theta')
    _check_length(cxx.shape[0], n_src, 'cxx')
    _check_length(cxy.shape[0], n_src, 'cxy')
    _check_length(cyy.shape[0], n_src, 'cyy')
    _check_length(skip.shape[0], n_src, 'skip')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'data')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'data')

    results_arr = np.empty((n_src, 2))
    cdef double[:, ::1] results = results_arr

    cdef Py_ssize_t i
    with nogil:
        for i in range(n_src):
            if skip[i]:
                results[i, 0] = NAN
                results[i, 1] = NAN
                continue
            _kron_radius_source(&data[0, 0], &mask[0, 0], &segm[0, 0],
                                nx_data, ny_data, labels[i], xcen[i],
                                ycen[i], semimajor[i] * scale,
                                semiminor[i] * scale, theta[i], cxx[i],
                                cxy[i], cyy[i], seg_method, scale,
                                min_circ_radius, max_aper_size,
                                &results[i, 0])
    return results_arr


ctypedef struct _FluxRadiusArgs:
    const double *data
    Py_ssize_t nx
    Py_ssize_t ny
    double xmin_e
    double ymin_e
    double dx
    double dy
    double pixel_radius
    int use_exact
    int subpixels
    double normflux


cdef double _flux_radius_objective(double r,
                                   void *args) noexcept nogil:
    """
    Fraction-of-flux residual ``1 - flux(r) / normflux`` for the
    brentq root-find, accumulating ``data * overlap`` with the same
    per-pixel arithmetic as ``circular_overlap_grid`` (grid edges
    relative to the source centroid).
    """
    cdef _FluxRadiusArgs *p = <_FluxRadiusArgs *>args
    cdef Py_ssize_t ix, iy
    cdef double flux = 0.0
    cdef double pymin, pxmin, frac

    for iy in range(p.ny):
        pymin = p.ymin_e + iy * p.dy
        for ix in range(p.nx):
            pxmin = p.xmin_e + ix * p.dx
            frac = _circle_pixel_frac(pxmin, pymin, p.dx, p.dy,
                                      p.pixel_radius, r, p.use_exact,
                                      p.subpixels)
            if frac > 0.0:
                flux += p.data[iy * p.nx + ix] * frac
    return 1.0 - flux / p.normflux


def batch_flux_radius_solve(args_list, *, double fraction):
    """
    Solve the fractional-flux radius for many prepared sources in one
    call.

    For each source, the circular radius enclosing ``fraction`` of the
    Kron flux is the root of ``1 - flux(r) / (kronflux * fraction)``,
    found by a bracketed Brent root-find over ``[0.1, max_radius]``.
    The per-pixel circular overlap, the bracket, and the root-finder
    tolerances replicate the previous per-source
    `scipy.optimize.root_scalar` implementation exactly.

    A bracket whose endpoints have the same sign has no (or multiple)
    solutions. As in the previous implementation, the maximum radius
    is then reduced by 10% of its original value and the root-find is
    retried, until either a root is found or the maximum radius drops
    to the minimum radius (0.1), in which case NaN is returned.

    Parameters
    ----------
    args_list : list
        The prepared per-source arguments, as built by
        ``SourceCatalog._flux_radius_optimizer_args``. Each entry is
        either `None`, for a source that cannot have a solution (NaN
        is returned), or the list ``[clean_data, grid_params,
        kronflux, max_radius]``. ``clean_data`` is the C-contiguous
        float64 source cutout with masked pixels zeroed,
        ``grid_params`` is the tuple ``(xmin_e, xmax_e, ymin_e,
        ymax_e, nx, ny, use_exact, subpixels)`` of
        `~photutils.geometry.circular_overlap_grid` parameters whose
        grid edges are relative to the source centroid, ``kronflux``
        is the source Kron flux, and ``max_radius`` is the initial
        upper bracket radius.

    fraction : float
        The fraction of the Kron flux at which to find the circular
        radius.

    Returns
    -------
    radius : 1D `~numpy.ndarray`
        The circular radius enclosing the specified fraction of the
        Kron flux for each source, with shape ``(n_sources,)``. NaN is
        returned for `None` entries and where no solution was found.
    """
    cdef Py_ssize_t n_src = len(args_list)
    radius_arr = np.full(n_src, np.nan)
    cdef double[::1] radius = radius_arr
    cdef const double[:, ::1] cdata
    cdef _FluxRadiusArgs p
    cdef zeros_full_output full_output
    cdef double xmax_e, ymax_e
    cdef double max_radius, delta, result
    cdef bint found
    cdef Py_ssize_t i

    for i, entry in enumerate(args_list):
        if entry is None:
            continue
        clean_data, grid_params, kronflux, max_radius_py = entry
        cdata = clean_data
        p.data = &cdata[0, 0]
        p.xmin_e = grid_params[0]
        xmax_e = grid_params[1]
        p.ymin_e = grid_params[2]
        ymax_e = grid_params[3]
        p.nx = grid_params[4]
        p.ny = grid_params[5]
        # The pixel size and pixel radius are derived from the grid
        # extent exactly as in ``circular_overlap_grid``
        p.dx = (xmax_e - p.xmin_e) / p.nx
        p.dy = (ymax_e - p.ymin_e) / p.ny
        p.pixel_radius = 0.5 * sqrt(p.dx * p.dx + p.dy * p.dy)
        p.use_exact = grid_params[6]
        p.subpixels = grid_params[7]
        p.normflux = kronflux * fraction
        max_radius = max_radius_py

        with nogil:
            found = False
            result = NAN
            delta = 0.1 * max_radius
            while max_radius > 0.1 and not found:
                # xtol, rtol (4 * float64 eps), and maxiter are the
                # root_scalar(method='brentq') defaults, so the same
                # underlying C algorithm follows the identical
                # iteration sequence
                result = brentq(_flux_radius_objective, 0.1,
                                max_radius, &p, 2e-12,
                                8.881784197001252e-16, 100,
                                &full_output)
                if full_output.error_num == -1:
                    # Sign error (same-sign bracket): shrink and
                    # retry, matching the ValueError path of
                    # root_scalar
                    max_radius -= delta
                else:
                    # Success or non-convergence: root_scalar does
                    # not raise on non-convergence, so accept the
                    # root either way
                    found = True
            if not found:
                result = NAN
        radius[i] = result
    return radius_arr


def batch_raw_moments(const double[:, ::1] convdata, *,
                      const unsigned char[:, ::1] mask,
                      const Py_ssize_t[:, ::1] segm,
                      const Py_ssize_t[::1] labels,
                      const Py_ssize_t[::1] bbox_iymin,
                      const Py_ssize_t[::1] bbox_iymax,
                      const Py_ssize_t[::1] bbox_ixmin,
                      const Py_ssize_t[::1] bbox_ixmax):
    """
    Compute the raw spatial moments for many sources in one call.

    The moments are computed up to 3rd order along each axis over the
    segment bounding box of each source, using coordinates relative
    to the bounding-box origin (i.e., the cutout frame). Pixels
    outside the source segment, input-masked pixels, non-finite
    values, and negative values contribute zero, which replicates the
    zeroed moment cutouts of the previous per-source implementation.

    Parameters
    ----------
    convdata : 2D ndarray of float64 (C-contiguous)
        The convolved data array (or the data array itself if no
        convolved data was input).

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where bit 1 (value 1) marks input-masked pixels
        and bit 2 (value 2) marks non-finite data pixels folded into
        the mask by the caller. Only bit 1 excludes a pixel here;
        non-finite convolved values are excluded by their own test.
        Must have the same shape as ``convdata``.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same
        shape as ``convdata``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape
        ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    Returns
    -------
    result : 3D `~numpy.ndarray`
        The raw moments with shape ``(n_sources, 4, 4)``. The element
        ``[i, p, q]`` is the sum of ``v * cy**p * cx**q`` over the
        included pixels of source ``i``, where ``v`` is the convolved
        data value and ``(cx, cy)`` are the cutout-frame pixel
        coordinates.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if a 2D array does not have the same shape as
        ``convdata``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = convdata.shape[0]
    cdef Py_ssize_t nx_data = convdata.shape[1]

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'convdata')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'convdata')

    result_arr = np.zeros((n_src, 4, 4))
    cdef double[:, :, ::1] result = result_arr
    cdef Py_ssize_t i, ix, iy, p, q, y0, y1, x0, x1, lbl
    cdef double v, cx, cy
    cdef double xpow[4]
    cdef double ypow[4]

    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            for iy in range(y0, y1):
                cy = <double>(iy - y0)
                ypow[0] = 1.0
                ypow[1] = cy
                ypow[2] = cy * cy
                ypow[3] = cy * cy * cy
                for ix in range(x0, x1):
                    if segm[iy, ix] != lbl:
                        continue
                    if mask[iy, ix] & 1:
                        continue
                    v = convdata[iy, ix]
                    if not isfinite(v) or v < 0.0:
                        continue
                    cx = <double>(ix - x0)
                    xpow[0] = 1.0
                    xpow[1] = cx
                    xpow[2] = cx * cx
                    xpow[3] = cx * cx * cx
                    for p in range(4):
                        for q in range(4):
                            result[i, p, q] += (v * ypow[p]
                                                * xpow[q])
    return result_arr


def batch_central_moments(const double[:, ::1] convdata, *,
                          const unsigned char[:, ::1] mask,
                          const Py_ssize_t[:, ::1] segm,
                          const Py_ssize_t[::1] labels,
                          const Py_ssize_t[::1] bbox_iymin,
                          const Py_ssize_t[::1] bbox_iymax,
                          const Py_ssize_t[::1] bbox_ixmin,
                          const Py_ssize_t[::1] bbox_ixmax,
                          const double[::1] xcen,
                          const double[::1] ycen):
    """
    Compute the central spatial moments for many sources in one call.

    These are the translation-invariant moments up to 3rd order along
    each axis, i.e., the raw moments of `batch_raw_moments` computed
    with coordinates measured from the given cutout-frame centroid.
    The pixel-inclusion rules are identical to those of
    `batch_raw_moments`.

    Parameters
    ----------
    convdata : 2D ndarray of float64 (C-contiguous)
        The convolved data array (or the data array itself if no
        convolved data was input).

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where bit 1 (value 1) marks input-masked pixels
        and bit 2 (value 2) marks non-finite data pixels folded into
        the mask by the caller. Only bit 1 excludes a pixel here;
        non-finite convolved values are excluded by their own test.
        Must have the same shape as ``convdata``.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same
        shape as ``convdata``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape
        ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    xcen, ycen : 1D ndarray of float64 (C-contiguous)
        The cutout-frame centroid of each source, with shape
        ``(n_sources,)``.

    Returns
    -------
    result : 3D `~numpy.ndarray`
        The central moments with shape ``(n_sources, 4, 4)``. The
        element ``[i, p, q]`` is the sum of ``v * dy**p * dx**q``
        over the included pixels of source ``i``, where ``v`` is the
        convolved data value and ``(dx, dy)`` are the cutout-frame
        pixel coordinates relative to the source centroid. If a
        source centroid is not finite, every element except
        ``[i, 0, 0]``, which is the plain sum of the included values,
        is NaN.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if a 2D array does not have the same shape as
        ``convdata``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = convdata.shape[0]
    cdef Py_ssize_t nx_data = convdata.shape[1]

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_length(xcen.shape[0], n_src, 'xcen')
    _check_length(ycen.shape[0], n_src, 'ycen')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'convdata')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'convdata')

    result_arr = np.zeros((n_src, 4, 4))
    cdef double[:, :, ::1] result = result_arr
    cdef Py_ssize_t i, ix, iy, p, q, y0, y1, x0, x1, lbl
    cdef double v, cx, cy
    cdef double xpow[4]
    cdef double ypow[4]

    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            for iy in range(y0, y1):
                cy = <double>(iy - y0) - ycen[i]
                ypow[0] = 1.0
                ypow[1] = cy
                ypow[2] = cy * cy
                ypow[3] = cy * cy * cy
                for ix in range(x0, x1):
                    if segm[iy, ix] != lbl:
                        continue
                    if mask[iy, ix] & 1:
                        continue
                    v = convdata[iy, ix]
                    if not isfinite(v) or v < 0.0:
                        continue
                    cx = <double>(ix - x0) - xcen[i]
                    xpow[0] = 1.0
                    xpow[1] = cx
                    xpow[2] = cx * cx
                    xpow[3] = cx * cx * cx
                    for p in range(4):
                        for q in range(4):
                            result[i, p, q] += (v * ypow[p]
                                                * xpow[q])

            if not (isfinite(xcen[i]) and isfinite(ycen[i])):
                # A non-finite centroid poisons every coordinate
                # power, so all elements except [0, 0] (the plain
                # flux sum) are NaN, as in the previous NumPy
                # implementation. The sums above are NaN only where
                # a source has at least one included pixel, so set
                # them explicitly.
                for p in range(4):
                    for q in range(4):
                        if p != 0 or q != 0:
                            result[i, p, q] = NAN
    return result_arr


def batch_moment_err(const double[:, ::1] error, *,
                     const double[:, ::1] convdata,
                     const unsigned char[:, ::1] mask,
                     const Py_ssize_t[:, ::1] segm,
                     const Py_ssize_t[::1] labels,
                     const Py_ssize_t[::1] bbox_iymin,
                     const Py_ssize_t[::1] bbox_iymax,
                     const Py_ssize_t[::1] bbox_ixmin,
                     const Py_ssize_t[::1] bbox_ixmax,
                     const double[::1] xcen,
                     const double[::1] ycen):
    """
    Compute the raw centroid error sums for many sources in one call.

    These are the unnormalized sums that the isophotal centroid error
    covariance is built from. A pixel contributes its error variance
    only if it is inside the source segment, is unmasked (neither
    input-masked nor non-finite data), and contributes nonzero flux
    weight to the moments (a finite, strictly positive convolved
    value). This replicates the zeroing of the previous per-source
    implementation, which set the error variance to zero wherever the
    total mask was set or the moment data were zero.

    Parameters
    ----------
    error : 2D ndarray of float64 (C-contiguous)
        The pixel-wise 1-sigma errors. Must have the same shape as
        ``convdata``.

    convdata : 2D ndarray of float64 (C-contiguous)
        The convolved data array (or the data array itself if no
        convolved data was input).

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where bit 1 (value 1) marks input-masked pixels
        and bit 2 (value 2) marks non-finite data pixels folded into
        the mask by the caller. Any nonzero value excludes the pixel.
        Must have the same shape as ``convdata``.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same
        shape as ``convdata``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape
        ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    xcen, ycen : 1D ndarray of float64 (C-contiguous)
        The cutout-frame centroid of each source, with shape
        ``(n_sources,)``.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The raw error sums with shape ``(n_sources, 4)`` and columns
        ``(sum_e2, sum_e2_dx2, sum_e2_dy2, sum_e2_dxdy)``, where
        ``e2`` is the pixel error variance and ``(dx, dy)`` are the
        cutout-frame pixel coordinates relative to the source
        centroid.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if a 2D array does not have the same shape as
        ``error``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = error.shape[0]
    cdef Py_ssize_t nx_data = error.shape[1]

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_length(xcen.shape[0], n_src, 'xcen')
    _check_length(ycen.shape[0], n_src, 'ycen')
    _check_shape(convdata.shape[0], convdata.shape[1], ny_data,
                 nx_data, 'convdata', 'error')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'error')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'error')

    result_arr = np.zeros((n_src, 4))
    cdef double[:, ::1] result = result_arr
    cdef Py_ssize_t i, ix, iy, y0, y1, x0, x1, lbl
    cdef double v, e, e2, dx, dy

    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            for iy in range(y0, y1):
                dy = <double>(iy - y0) - ycen[i]
                for ix in range(x0, x1):
                    if segm[iy, ix] != lbl:
                        continue
                    if mask[iy, ix] != 0:
                        continue
                    v = convdata[iy, ix]
                    if not isfinite(v) or v <= 0.0:
                        continue
                    dx = <double>(ix - x0) - xcen[i]
                    e = error[iy, ix]
                    e2 = e * e
                    result[i, 0] += e2
                    result[i, 1] += e2 * dx * dx
                    result[i, 2] += e2 * dy * dy
                    result[i, 3] += e2 * dx * dy
    return result_arr
