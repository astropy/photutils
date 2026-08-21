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

__all__ = ['batch_centroid_win', 'batch_flux_radius_solve']


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
