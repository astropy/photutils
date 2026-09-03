# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
The outside-weight scan of the batch aperture photometry driver.

``batch_aperture_sums`` reports, for each aperture whose bounding box
is clipped by a data edge, whether the aperture has a nonzero-fraction
pixel outside the data. The scan lives in its own extension module so
that the per-shape pixel-overlap helpers of ``_batch_overlap.pxd``
keep a single call site in the driver's translation unit. With more
call sites the C compiler stops inlining the larger helpers into the
driver's per-pixel loop, which measured up to 30% slower.

The scan runs without the GIL and uses no global mutable state, so it
is safe to call from multiple threads, including on free-threaded
Python builds.
"""

from photutils.aperture._batch_overlap cimport (
    _CIRCLE, _CIRCULAR_ANNULUS, _ELLIPSE, _ELLIPTICAL_ANNULUS, _RECTANGLE,
    _RECTANGULAR_ANNULUS, _circle_pixel_frac, _circular_annulus_pixel_frac,
    _ellipse_pixel_frac, _elliptical_annulus_pixel_frac, _polygon_pixel_frac,
    _rect_pixel_frac, _rectangular_annulus_pixel_frac)

__all__ = []


cdef inline double _shape_pixel_frac(const _ShapeSpec *sp, double pxmin,
                                     double pymin, double dx, double dy,
                                     double pixel_radius,
                                     double norm) noexcept nogil:
    """
    Return the overlap fraction of one pixel with the aperture shape.

    This is the same dispatch on the shape code that the accumulation
    loop of ``batch_aperture_sums`` carries inline over its local
    variables. Keep the two in step.
    """
    if sp.shape_code == _CIRCLE:
        return _circle_pixel_frac(pxmin, pymin, dx, dy, pixel_radius,
                                  sp.r_out, sp.use_exact, sp.subpixels)
    if sp.shape_code == _CIRCULAR_ANNULUS:
        return _circular_annulus_pixel_frac(
            pxmin, pymin, dx, dy, pixel_radius, sp.r_in, sp.r_out,
            sp.use_exact, sp.subpixels)
    if sp.shape_code == _ELLIPSE:
        return _ellipse_pixel_frac(
            pxmin, pymin, dx, dy, norm, sp.rx_out, sp.ry_out,
            sp.cos_theta, sp.sin_theta, sp.use_exact, sp.subpixels)
    if sp.shape_code == _ELLIPTICAL_ANNULUS:
        return _elliptical_annulus_pixel_frac(
            pxmin, pymin, dx, dy, norm, sp.rx_in, sp.ry_in, sp.rx_out,
            sp.ry_out, sp.cos_theta, sp.sin_theta, sp.use_exact,
            sp.subpixels)
    if sp.shape_code == _RECTANGLE:
        return _rect_pixel_frac(
            pxmin, pymin, dx, dy, pixel_radius, sp.half_width_out,
            sp.half_height_out, sp.cos_theta, sp.sin_theta,
            sp.bbox_dx_out, sp.bbox_dy_out, sp.poly_x_out, sp.poly_y_out,
            sp.buf_a_x, sp.buf_a_y, sp.buf_b_x, sp.buf_b_y, sp.use_exact,
            sp.subpixels)
    if sp.shape_code == _RECTANGULAR_ANNULUS:
        return _rectangular_annulus_pixel_frac(
            pxmin, pymin, dx, dy, pixel_radius, sp.half_width_in,
            sp.half_height_in, sp.half_width_out, sp.half_height_out,
            sp.cos_theta, sp.sin_theta, sp.bbox_dx_in, sp.bbox_dy_in,
            sp.bbox_dx_out, sp.bbox_dy_out, sp.poly_x_in, sp.poly_y_in,
            sp.poly_x_out, sp.poly_y_out, sp.buf_a_x, sp.buf_a_y,
            sp.buf_b_x, sp.buf_b_y, sp.use_exact, sp.subpixels)
    return _polygon_pixel_frac(
        pxmin, pymin, dx, dy, pixel_radius, sp.poly_x, sp.poly_y,
        sp.n_poly, sp.pedge_nx, sp.pedge_ny, sp.pedge_c,
        sp.is_poly_convex, sp.pbuf_a_x, sp.pbuf_a_y, sp.pbuf_b_x,
        sp.pbuf_b_y, sp.poly_buf_size, sp.use_exact, sp.subpixels)


cdef inline bint _ring_row(const _ShapeSpec *sp, Py_ssize_t iy,
                           Py_ssize_t ix_start, Py_ssize_t ix_stop,
                           Py_ssize_t ixc, double gxmin, double gymin,
                           double dx, double dy, double pixel_radius,
                           double norm, Py_ssize_t ixmin,
                           Py_ssize_t iymin) noexcept nogil:
    """
    Whether a pixel of row ``iy`` in ``[ix_start, ix_stop)`` has a
    nonzero fraction, testing outward from the column ``ixc`` nearest
    the aperture center (where a crossing of the data edge is found
    soonest).
    """
    cdef Py_ssize_t lo, hi
    cdef double pymin = gymin + (iy - iymin) * dy
    lo = ixc if ixc > ix_start else ix_start
    lo = lo if lo < ix_stop else ix_stop - 1
    hi = lo + 1
    while lo >= ix_start or hi < ix_stop:
        if lo >= ix_start:
            if _shape_pixel_frac(sp, gxmin + (lo - ixmin) * dx, pymin,
                                 dx, dy, pixel_radius, norm) > 0.0:
                return True
            lo -= 1
        if hi < ix_stop:
            if _shape_pixel_frac(sp, gxmin + (hi - ixmin) * dx, pymin,
                                 dx, dy, pixel_radius, norm) > 0.0:
                return True
            hi += 1
    return False


cdef inline bint _ring_column(const _ShapeSpec *sp, Py_ssize_t ix,
                              Py_ssize_t iy_start, Py_ssize_t iy_stop,
                              Py_ssize_t iyc, double gxmin, double gymin,
                              double dx, double dy, double pixel_radius,
                              double norm, Py_ssize_t ixmin,
                              Py_ssize_t iymin) noexcept nogil:
    """
    The ``_ring_row`` test for the column ``ix`` over the rows
    ``[iy_start, iy_stop)``, testing outward from the row ``iyc``.
    """
    cdef Py_ssize_t lo, hi
    cdef double pxmin = gxmin + (ix - ixmin) * dx
    lo = iyc if iyc > iy_start else iy_start
    lo = lo if lo < iy_stop else iy_stop - 1
    hi = lo + 1
    while lo >= iy_start or hi < iy_stop:
        if lo >= iy_start:
            if _shape_pixel_frac(sp, pxmin, gymin + (lo - iymin) * dy,
                                 dx, dy, pixel_radius, norm) > 0.0:
                return True
            lo -= 1
        if hi < iy_stop:
            if _shape_pixel_frac(sp, pxmin, gymin + (hi - iymin) * dy,
                                 dx, dy, pixel_radius, norm) > 0.0:
                return True
            hi += 1
    return False


cdef bint outside_weight(const _ShapeSpec *sp, bint inside_any,
                         double gxmin, double gymin, double dx, double dy,
                         double pixel_radius, double norm,
                         Py_ssize_t ixmin, Py_ssize_t iymin,
                         Py_ssize_t ixmax_full, Py_ssize_t iymax_full,
                         Py_ssize_t ix0, Py_ssize_t ix1, Py_ssize_t iy0,
                         Py_ssize_t iy1) noexcept nogil:
    """
    Whether an aperture whose bounding box is clipped by a data edge
    has a nonzero-fraction pixel outside the data.

    For the exact overlap method, when the aperture has a
    nonzero-fraction pixel inside the data, only the one-pixel ring
    just outside the data edges (within the unclipped bounding box) is
    tested. Every aperture shape is a connected region, so one with
    positive area both inside and outside the data crosses the data
    boundary at an interior point, and the ring pixel containing that
    point has a positive exact overlap. This makes the cost of the test
    scale with the data perimeter rather than with the (possibly
    enormous) bounding-box area.

    Otherwise every outside pixel is tested until one has a nonzero
    fraction. The center and subpixel methods sample the shape at
    pixel or subpixel centers, for which the ring argument does not
    hold, and an aperture with no weight inside the data may lie
    entirely in the clipped-away part of its bounding box (that scan
    is short, because such a shape reaches the first rows of its own
    bounding box).

    Parameters
    ----------
    sp : const _ShapeSpec *
        The aperture shape parameters.

    inside_any : bint
        Whether the aperture has a nonzero-fraction pixel inside the
        data.

    gxmin, gymin, dx, dy, pixel_radius, norm : double
        The pixel grid of the source (see ``_source_grid_setup``).

    ixmin, iymin, ixmax_full, iymax_full : Py_ssize_t
        The unclipped bounding box (the maxima are exclusive).

    ix0, ix1, iy0, iy1 : Py_ssize_t
        The bounding box clipped to the data (the maxima are exclusive).

    Returns
    -------
    result : bint
        `True` if a pixel outside the data has a nonzero overlap
        fraction.
    """
    cdef Py_ssize_t ix, iy, cx0, cx1, ixc, iyc
    cdef double pxmin, pymin

    if sp.use_exact and inside_any:
        # The columns of the rows just outside the top and bottom data
        # edges include the corner pixels. Each ring row or column is
        # tested outward from the pixel nearest the aperture center,
        # where the shape most likely crosses the data edge.
        cx0 = ixmin if ixmin > ix0 - 1 else ix0 - 1
        cx1 = ixmax_full if ixmax_full < ix1 + 1 else ix1 + 1
        ixc = ixmin + <Py_ssize_t>(-gxmin / dx)
        iyc = iymin + <Py_ssize_t>(-gymin / dy)
        if iymin < iy0 and _ring_row(sp, iy0 - 1, cx0, cx1, ixc, gxmin,
                                     gymin, dx, dy, pixel_radius, norm,
                                     ixmin, iymin):
            return True
        if iy1 < iymax_full and _ring_row(sp, iy1, cx0, cx1, ixc, gxmin,
                                          gymin, dx, dy, pixel_radius,
                                          norm, ixmin, iymin):
            return True
        if ixmin < ix0 and _ring_column(sp, ix0 - 1, iy0, iy1, iyc, gxmin,
                                        gymin, dx, dy, pixel_radius, norm,
                                        ixmin, iymin):
            return True
        if ix1 < ixmax_full and _ring_column(sp, ix1, iy0, iy1, iyc, gxmin,
                                             gymin, dx, dy, pixel_radius,
                                             norm, ixmin, iymin):
            return True
        return False

    # Sampled overlap: test every outside pixel of the unclipped
    # bounding box, skipping the columns inside the data on the rows
    # that overlap it
    for iy in range(iymin, iymax_full):
        pymin = gymin + (iy - iymin) * dy
        if iy0 <= iy < iy1:
            for ix in range(ixmin, ix0):
                pxmin = gxmin + (ix - ixmin) * dx
                if _shape_pixel_frac(sp, pxmin, pymin, dx, dy,
                                     pixel_radius, norm) > 0.0:
                    return True
            for ix in range(ix1, ixmax_full):
                pxmin = gxmin + (ix - ixmin) * dx
                if _shape_pixel_frac(sp, pxmin, pymin, dx, dy,
                                     pixel_radius, norm) > 0.0:
                    return True
        else:
            for ix in range(ixmin, ixmax_full):
                pxmin = gxmin + (ix - ixmin) * dx
                if _shape_pixel_frac(sp, pxmin, pymin, dx, dy,
                                     pixel_radius, norm) > 0.0:
                    return True
    return False
