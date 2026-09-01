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
never need per-source cutout arrays. The segment gather packs the
unmasked segment pixel values of all sources into a single array for
the pixel-statistics properties (e.g., ``segment_flux``).

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
                                                _seg_pixel_contributes)

__all__ = ['batch_central_moments', 'batch_centroid_win',
           'batch_flux_radius_prepare', 'batch_flux_radius_solve',
           'batch_kron_radius', 'batch_moment_err', 'batch_perimeter',
           'batch_quad_boxes', 'batch_raw_moments',
           'batch_segment_gather']


cdef extern from "math.h" nogil:
    double exp(double x)
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double floor(double x)
    double ceil(double x)
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
                        and not _seg_pixel_contributes(
                            segm, mask, nx_data, seg_method, label,
                            ix, iy, x0, x1, y0, y1, ccx, ccy,
                            &six, &siy)):
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
                    and not _seg_pixel_contributes(
                        segm, mask, nx_data, seg_method, label,
                        ix, iy, lx0, lx1, ly0, ly1, lccx, lccy,
                        &six, &siy)):
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
    previous per-source Python implementation to within floating-point
    rounding (the pixel sums are accumulated sequentially rather than
    pairwise).

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
                    and not _seg_pixel_contributes(
                        segm, mask, nx_data, seg_method, label,
                        ix, iy, x0, x1, y0, y1, ccx, ccy,
                        &six, &siy)):
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
    previous per-source Python implementation, with the sums agreeing
    to within floating-point rounding (they are accumulated
    sequentially rather than pairwise). The caller forms the Kron
    radius from the two sums.

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


cdef void _flux_radius_cutout(const double *data,
                              const unsigned char *mask,
                              const Py_ssize_t *segm,
                              Py_ssize_t nx_data, Py_ssize_t label,
                              Py_ssize_t x0, Py_ssize_t x1,
                              Py_ssize_t y0, Py_ssize_t y1,
                              Py_ssize_t ccx, Py_ssize_t ccy,
                              double local_bkg, int seg_method,
                              double *out) noexcept nogil:
    """
    Fill the cleaned, background-subtracted cutout of a single source
    for the flux-radius root-find.

    Replicates the previous per-source Python preparation: masked and
    non-finite pixels are zero, neighbor-source pixels are zeroed or
    mirror-corrected per ``seg_method`` (an uncorrectable neighbor
    pixel is zero), and every other pixel is ``data - local_bkg``.
    Writes the ``(y1 - y0, x1 - x0)`` cutout in row-major order into
    ``out``.
    """
    cdef Py_ssize_t ix, iy, six, siy
    cdef Py_ssize_t nx = x1 - x0
    cdef double value
    for iy in range(y0, y1):
        for ix in range(x0, x1):
            value = 0.0
            if mask[iy * nx_data + ix] == 0:
                six = ix
                siy = iy
                if (seg_method == 0
                        or _seg_pixel_contributes(
                            segm, mask, nx_data, seg_method, label,
                            ix, iy, x0, x1, y0, y1, ccx, ccy,
                            &six, &siy)):
                    value = data[siy * nx_data + six] - local_bkg
            out[(iy - y0) * nx + (ix - x0)] = value


def batch_flux_radius_prepare(const double[:, ::1] data, *,
                              const unsigned char[:, ::1] mask,
                              const Py_ssize_t[:, ::1] segm,
                              const Py_ssize_t[::1] labels,
                              const double[::1] xcen,
                              const double[::1] ycen,
                              const double[::1] local_bkg,
                              const double[::1] kronflux,
                              const double[::1] max_radius,
                              const unsigned char[::1] skip,
                              int seg_method, int use_exact,
                              int subpixels, Py_ssize_t max_aper_size):
    """
    Prepare the per-source inputs of the flux-radius root-find for
    many sources in one call.

    For each source, the cutout of the data within the bounding box
    of the circle of radius ``max_radius`` (clipped to the data) is
    cleaned and background-subtracted (see ``_flux_radius_cutout``)
    and the `~photutils.geometry.circular_overlap_grid` grid
    parameters of that cutout, relative to the source centroid, are
    computed. The bounding-box arithmetic, the per-pixel masking, and
    the neighbor handling replicate the previous per-source Python
    implementation exactly.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array.

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where nonzero values indicate masked (zeroed)
        pixels. Must have the same shape as ``data``. Bit 1 (value 1)
        marks input-masked pixels and bit 2 (value 2) marks non-finite
        data pixels folded into the mask by the caller. Any nonzero
        value zeroes the pixel and prevents it from being used as a
        mirror pixel.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``data``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape ``(n_sources,)``.

    xcen, ycen : 1D ndarray of float64 (C-contiguous)
        The source centroids, with shape ``(n_sources,)``.

    local_bkg : 1D ndarray of float64 (C-contiguous)
        The per-source local background to subtract from each pixel
        value, with shape ``(n_sources,)``.

    kronflux : 1D ndarray of float64 (C-contiguous)
        The Kron flux of each source, with shape ``(n_sources,)``.

    max_radius : 1D ndarray of float64 (C-contiguous)
        The maximum circular radius of each source (the initial upper
        bracket of the root-find), with shape ``(n_sources,)``.

    skip : 1D ndarray of uint8 (C-contiguous)
        Nonzero for sources that cannot have a solution (e.g., a
        non-finite centroid, Kron flux, or maximum radius, or a zero
        Kron flux), with shape ``(n_sources,)``.

    seg_method : int
        The segmentation masking method:

        * 0: disables masking
        * 1: zeroes neighbor-source pixels
             (``(seg > 0) & (seg != label)``)
        * 3: replaces neighbor-source pixels with the values mirrored
             across the (rounded) cutout center (the symmetric
             ``'correct'`` method). A neighbor pixel whose mirror falls
             outside the clipped cutout, is itself a neighbor, or is
             masked is zeroed instead.

    use_exact : int
        Whether the root-find computes exact overlap fractions (1) or
        uses subpixel sampling (0).

    subpixels : int
        The number of subpixels in each dimension when ``use_exact``
        is 0.

    max_aper_size : Py_ssize_t
        The maximum number of pixels in the (unclipped) bounding box.
        Sources with a larger bounding box are skipped (out-of-memory
        guard).

    Returns
    -------
    args : list
        The prepared per-source arguments consumed by
        ``batch_flux_radius_solve``, with one entry per source. Each
        entry is either `None`, for a skipped source (including those
        rejected by ``max_aper_size`` and those whose bounding box
        does not overlap the data), or the list ``[clean_data,
        grid_params, kronflux, max_radius]``, where ``clean_data`` is
        the cleaned C-contiguous float64 cutout and ``grid_params`` is
        the tuple ``(xmin_e, xmax_e, ymin_e, ymax_e, nx, ny,
        use_exact, subpixels)`` of grid parameters whose edges are
        relative to the source centroid.

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
    _check_length(local_bkg.shape[0], n_src, 'local_bkg')
    _check_length(kronflux.shape[0], n_src, 'kronflux')
    _check_length(max_radius.shape[0], n_src, 'max_radius')
    _check_length(skip.shape[0], n_src, 'skip')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'data')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'data')

    cdef Py_ssize_t i, x0, x1, y0, y1, nx, ny, ccx, ccy
    cdef double xc, yc, rmax, ixmin_d, ixmax_d, iymin_d, iymax_d
    cdef double x0_d, x1_d, y0_d, y1_d, cutout_xcen, cutout_ycen
    cdef double[:, ::1] cutout
    args = []
    for i in range(n_src):
        if skip[i]:
            args.append(None)
            continue
        xc = xcen[i]
        yc = ycen[i]
        rmax = max_radius[i]

        # The bounding box of the maximum-radius circle, kept as
        # integral doubles until it is clipped to avoid integer
        # overflow for sources far outside the image
        ixmin_d = floor(xc - rmax + 0.5)
        ixmax_d = ceil(xc + rmax + 0.5)
        iymin_d = floor(yc - rmax + 0.5)
        iymax_d = ceil(yc + rmax + 0.5)
        if (iymax_d - iymin_d) * (ixmax_d - ixmin_d) > max_aper_size:
            args.append(None)
            continue

        # Clip to the data
        x0_d = ixmin_d if ixmin_d > 0.0 else 0.0
        x1_d = ixmax_d if ixmax_d < nx_data else nx_data
        y0_d = iymin_d if iymin_d > 0.0 else 0.0
        y1_d = iymax_d if iymax_d < ny_data else ny_data
        if y0_d >= y1_d or x0_d >= x1_d:
            args.append(None)
            continue
        x0 = <Py_ssize_t>x0_d
        x1 = <Py_ssize_t>x1_d
        y0 = <Py_ssize_t>y0_d
        y1 = <Py_ssize_t>y1_d
        nx = x1 - x0
        ny = y1 - y0

        # The cutout-frame centroid and the mirror center of the
        # 'correct' method (truncation of the cutout centroid + 0.5
        # replicates the Python int() call)
        cutout_xcen = xc - <double>x0
        cutout_ycen = yc - <double>y0
        ccx = <Py_ssize_t>(cutout_xcen + 0.5) + x0
        ccy = <Py_ssize_t>(cutout_ycen + 0.5) + y0

        clean_data = np.empty((ny, nx))
        cutout = clean_data
        with nogil:
            _flux_radius_cutout(&data[0, 0], &mask[0, 0], &segm[0, 0],
                                nx_data, labels[i], x0, x1, y0, y1,
                                ccx, ccy, local_bkg[i], seg_method,
                                &cutout[0, 0])

        grid_params = (-0.5 - cutout_xcen,
                       (<double>nx - 0.5) - cutout_xcen,
                       -0.5 - cutout_ycen,
                       (<double>ny - 0.5) - cutout_ycen,
                       nx, ny, use_exact, subpixels)
        args.append([clean_data, grid_params, kronflux[i], rmax])
    return args


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
    `scipy.optimize.root_scalar` implementation (the same SciPy C
    routine is used), so the roots agree to within floating-point
    rounding of the sequentially accumulated flux sums.

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


cdef inline bint _is_source_pixel(const unsigned char *mask,
                                  const Py_ssize_t *segm,
                                  Py_ssize_t nx_data, Py_ssize_t label,
                                  Py_ssize_t ix, Py_ssize_t iy,
                                  Py_ssize_t x0, Py_ssize_t x1,
                                  Py_ssize_t y0, Py_ssize_t y1) noexcept nogil:
    """
    Whether a pixel is an unmasked pixel of the source segment, with
    pixels outside the clipped bounding box treated as background.
    """
    if ix < x0 or ix >= x1 or iy < y0 or iy >= y1:
        return False
    return (segm[iy * nx_data + ix] == label
            and mask[iy * nx_data + ix] == 0)


cdef inline bint _is_border_pixel(const unsigned char *mask,
                                  const Py_ssize_t *segm,
                                  Py_ssize_t nx_data, Py_ssize_t label,
                                  Py_ssize_t ix, Py_ssize_t iy,
                                  Py_ssize_t x0, Py_ssize_t x1,
                                  Py_ssize_t y0, Py_ssize_t y1) noexcept nogil:
    """
    Whether a pixel is a source pixel with at least one 4-connected
    neighbor that is not a source pixel (i.e., a source pixel removed
    by a binary erosion with a cross footprint).
    """
    if not _is_source_pixel(mask, segm, nx_data, label, ix, iy, x0, x1,
                            y0, y1):
        return False
    return not (_is_source_pixel(mask, segm, nx_data, label, ix - 1, iy,
                                 x0, x1, y0, y1)
                and _is_source_pixel(mask, segm, nx_data, label, ix + 1,
                                     iy, x0, x1, y0, y1)
                and _is_source_pixel(mask, segm, nx_data, label, ix,
                                     iy - 1, x0, x1, y0, y1)
                and _is_source_pixel(mask, segm, nx_data, label, ix,
                                     iy + 1, x0, x1, y0, y1))


def batch_perimeter(const unsigned char[:, ::1] mask, *,
                    const Py_ssize_t[:, ::1] segm,
                    const Py_ssize_t[::1] labels,
                    const Py_ssize_t[::1] bbox_iymin,
                    const Py_ssize_t[::1] bbox_iymax,
                    const Py_ssize_t[::1] bbox_ixmin,
                    const Py_ssize_t[::1] bbox_ixmax):
    """
    Compute the border-pixel pattern histograms of the perimeter
    estimator for many sources in one call.

    For each source, the unmasked segment pixels form a binary image
    (zero outside the segment bounding box). Its border pixels are
    the source pixels removed by a binary erosion with a 4-connected
    cross footprint, and the border image is convolved with the
    ``[[10, 2, 10], [2, 1, 2], [10, 2, 10]]`` kernel at every
    bounding-box pixel. The histogram of the convolved values below 34
    is returned; the caller applies the perimeter weights of the
    estimator of Benkrid et al. (2000) to it. This replicates the
    previous per-source implementation exactly.

    Parameters
    ----------
    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where nonzero values indicate masked (excluded)
        pixels. Bit 1 (value 1) marks input-masked pixels and bit 2
        (value 2) marks non-finite data pixels folded into the mask by
        the caller.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``mask``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    Returns
    -------
    hist : 2D ndarray of intp
        The histogram of the convolved border values of each source,
        with shape ``(n_sources, 34)``.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if ``segm`` does not have the same shape as
        ``mask``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = mask.shape[0]
    cdef Py_ssize_t nx_data = mask.shape[1]

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'mask')

    hist_arr = np.zeros((n_src, 34), dtype=np.intp)
    cdef Py_ssize_t[:, ::1] hist = hist_arr
    cdef const unsigned char *mask_ptr = &mask[0, 0]
    cdef const Py_ssize_t *segm_ptr = &segm[0, 0]
    cdef Py_ssize_t i, ix, iy, y0, y1, x0, x1, lbl, value
    cdef Py_ssize_t dx, dy, weight

    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            for iy in range(y0, y1):
                for ix in range(x0, x1):
                    value = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dy == 0:
                                weight = 1
                            elif dx == 0 or dy == 0:
                                weight = 2
                            else:
                                weight = 10
                            if _is_border_pixel(mask_ptr, segm_ptr,
                                                nx_data, lbl, ix + dx,
                                                iy + dy, x0, x1, y0, y1):
                                value += weight
                    if value < 34:
                        hist[i, value] += 1
    return hist_arr


def batch_quad_boxes(const double[:, ::1] data, *,
                     const double[:, ::1] error,
                     const unsigned char[:, ::1] mask,
                     const Py_ssize_t[:, ::1] segm,
                     const Py_ssize_t[::1] labels,
                     const Py_ssize_t[::1] bbox_iymin,
                     const Py_ssize_t[::1] bbox_iymax,
                     const Py_ssize_t[::1] bbox_ixmin,
                     const Py_ssize_t[::1] bbox_ixmax,
                     int compute_err):
    """
    Gather the 3x3 peak-pixel boxes of the quadratic centroid fit for
    many sources in one call.

    For each source, the data within its segment bounding box is
    treated as a cutout whose pixels outside the source segment,
    input-masked, or non-finite are zero, and the first (row-major)
    maximum of that cutout is the peak pixel. The 3x3 box centered on
    the peak is returned together with the pixel variances of the box
    (zero for masked pixels), replicating the previous per-source
    Python implementation exactly.

    Parameters
    ----------
    data : 2D ndarray of float64 (C-contiguous)
        The data array.

    error : 2D ndarray of float64 (C-contiguous) or `None`
        The pixel-wise 1-sigma errors. Must have the same shape as
        ``data``. Required (not `None`) if ``compute_err`` is nonzero.

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where nonzero values indicate masked (zeroed)
        pixels. Must have the same shape as ``data``. Bit 1 (value 1)
        marks input-masked pixels and bit 2 (value 2) marks non-finite
        data pixels folded into the mask by the caller.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``data``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    compute_err : int
        If nonzero, also gather the pixel variances of the boxes.

    Returns
    -------
    status : 1D ndarray of intp
        The per-source status: 0 if the 3x3 box was gathered, 1 if the
        cutout is smaller than 3x3, 2 if every cutout pixel is masked,
        and 3 if the peak pixel lies on the cutout edge (so no box
        fits). Only status 0 rows have valid boxes.

    peak : 2D ndarray of intp
        The cutout-frame ``(x, y)`` index of the peak pixel, with shape
        ``(n_sources, 2)``. Valid for status 0 and 3; ``(-1, -1)``
        otherwise.

    boxes : 2D ndarray of float64
        The zero-filled cutout values of the 3x3 box around the peak in
        row-major order, with shape ``(n_sources, 9)``. Zero for
        statuses other than 0.

    box_var : 2D ndarray of float64
        The pixel variances of the box, zero for masked pixels, with
        shape ``(n_sources, 9)``. All zero unless ``compute_err`` is
        nonzero.

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

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'data')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'data')
    if has_error:
        _check_shape(error.shape[0], error.shape[1], ny_data, nx_data,
                     'error', 'data')

    status_arr = np.zeros(n_src, dtype=np.intp)
    peak_arr = np.full((n_src, 2), -1, dtype=np.intp)
    boxes_arr = np.zeros((n_src, 9))
    box_var_arr = np.zeros((n_src, 9))
    cdef Py_ssize_t[::1] status = status_arr
    cdef Py_ssize_t[:, ::1] peak = peak_arr
    cdef double[:, ::1] boxes = boxes_arr
    cdef double[:, ::1] box_var = box_var_arr
    cdef const double *err_ptr = NULL
    if has_error:
        err_ptr = &error[0, 0]

    cdef Py_ssize_t i, ix, iy, y0, y1, x0, x1, lbl, k
    cdef Py_ssize_t xpeak, ypeak, bx, by
    cdef double v, vmax, e
    cdef bint found, masked
    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            if y1 - y0 < 3 or x1 - x0 < 3:
                status[i] = 1
                continue

            # First (row-major) maximum of the zero-filled cutout
            found = False
            vmax = 0.0
            xpeak = 0
            ypeak = 0
            for iy in range(y0, y1):
                for ix in range(x0, x1):
                    if segm[iy, ix] != lbl or mask[iy, ix] != 0:
                        v = 0.0
                    else:
                        v = data[iy, ix]
                        found = True
                    if (iy == y0 and ix == x0) or v > vmax:
                        vmax = v
                        xpeak = ix
                        ypeak = iy
            if not found:
                status[i] = 2
                continue

            peak[i, 0] = xpeak - x0
            peak[i, 1] = ypeak - y0
            if (xpeak == x0 or xpeak == x1 - 1 or ypeak == y0
                    or ypeak == y1 - 1):
                status[i] = 3
                continue

            k = 0
            for by in range(ypeak - 1, ypeak + 2):
                for bx in range(xpeak - 1, xpeak + 2):
                    masked = (segm[by, bx] != lbl or mask[by, bx] != 0)
                    if not masked:
                        boxes[i, k] = data[by, bx]
                        if compute_err:
                            e = err_ptr[by * nx_data + bx]
                            box_var[i, k] = e * e
                    k += 1
    return status_arr, peak_arr, boxes_arr, box_var_arr


def batch_segment_gather(const double[:, ::1] values, *,
                         const unsigned char[:, ::1] mask,
                         const Py_ssize_t[:, ::1] segm,
                         const Py_ssize_t[::1] labels,
                         const Py_ssize_t[::1] bbox_iymin,
                         const Py_ssize_t[::1] bbox_iymax,
                         const Py_ssize_t[::1] bbox_ixmin,
                         const Py_ssize_t[::1] bbox_ixmax):
    """
    Pack the unmasked segment pixel values of many sources into one
    array.

    For each source, the values of the pixels within its segment
    bounding box that belong to the source segment and are unmasked
    are copied, in row-major order, into a packed array. A source
    with no such pixels contributes a single NaN, replicating the
    previous per-source masked-array ``compressed()`` values (with a
    single NaN for completely masked sources).

    Parameters
    ----------
    values : 2D ndarray of float64 (C-contiguous)
        The array to gather from (e.g., the data, error, or background
        array).

    mask : 2D ndarray of uint8 (C-contiguous)
        A mask array where nonzero values indicate masked (excluded)
        pixels. Must have the same shape as ``values``. Bit 1 (value
        1) marks input-masked pixels and bit 2 (value 2) marks
        non-finite data pixels folded into the mask by the caller.

    segm : 2D ndarray of intp (C-contiguous)
        The segmentation array where background pixels are zero and
        sources have positive integer labels. Must have the same shape
        as ``values``.

    labels : 1D ndarray of intp (C-contiguous)
        The source label for each source, with shape ``(n_sources,)``.

    bbox_iymin, bbox_iymax, bbox_ixmin, bbox_ixmax : 1D ndarray of intp
        The segment bounding box of each source, with shape
        ``(n_sources,)``. The maxima are exclusive (slice ``stop``
        values).

    Returns
    -------
    packed : 1D ndarray of float64
        The packed pixel values of all sources.

    offsets : 1D ndarray of intp
        The start offset of each source in ``packed``, with shape
        ``(n_sources + 1,)``; the values of source ``i`` are
        ``packed[offsets[i]:offsets[i + 1]]``.

    counts : 1D ndarray of intp
        The number of unmasked segment pixels of each source, with
        shape ``(n_sources,)``. A source with a zero count occupies a
        single NaN in ``packed``.

    Raises
    ------
    ValueError
        If a per-source array does not have the same length as
        ``labels``, or if a 2D array does not have the same shape as
        ``values``.
    """
    cdef Py_ssize_t n_src = labels.shape[0]
    cdef Py_ssize_t ny_data = values.shape[0]
    cdef Py_ssize_t nx_data = values.shape[1]

    _check_length(bbox_iymin.shape[0], n_src, 'bbox_iymin')
    _check_length(bbox_iymax.shape[0], n_src, 'bbox_iymax')
    _check_length(bbox_ixmin.shape[0], n_src, 'bbox_ixmin')
    _check_length(bbox_ixmax.shape[0], n_src, 'bbox_ixmax')
    _check_shape(mask.shape[0], mask.shape[1], ny_data, nx_data,
                 'mask', 'values')
    _check_shape(segm.shape[0], segm.shape[1], ny_data, nx_data,
                 'segm', 'values')

    offsets_arr = np.zeros(n_src + 1, dtype=np.intp)
    counts_arr = np.zeros(n_src, dtype=np.intp)
    cdef Py_ssize_t[::1] offsets = offsets_arr
    cdef Py_ssize_t[::1] counts = counts_arr
    cdef Py_ssize_t i, ix, iy, y0, y1, x0, x1, lbl, n, pos

    # First pass: count the contributing pixels of each source
    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            n = 0
            for iy in range(y0, y1):
                for ix in range(x0, x1):
                    if segm[iy, ix] == lbl and mask[iy, ix] == 0:
                        n += 1
            counts[i] = n
            offsets[i + 1] = offsets[i] + (n if n > 0 else 1)

    packed_arr = np.empty(offsets[n_src])
    cdef double[::1] packed = packed_arr

    # Second pass: copy the values
    with nogil:
        for i in range(n_src):
            lbl = labels[i]
            y0 = bbox_iymin[i]
            y1 = bbox_iymax[i]
            x0 = bbox_ixmin[i]
            x1 = bbox_ixmax[i]
            pos = offsets[i]
            if counts[i] == 0:
                packed[pos] = NAN
                continue
            for iy in range(y0, y1):
                for ix in range(x0, x1):
                    if segm[iy, ix] == lbl and mask[iy, ix] == 0:
                        packed[pos] = values[iy, ix]
                        pos += 1
    return packed_arr, offsets_arr, counts_arr


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
