# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the ellipse overlap functions into other
Cython files. These single-pixel functions are pure C math functions
that are safe to call without the GIL.
"""

cdef double ellipse_overlap_single_subpixel(double x0, double y0,
                                            double x1, double y1,
                                            double rx, double ry,
                                            double cos_theta,
                                            double sin_theta,
                                            int subpixels) noexcept nogil
cdef double ellipse_overlap_single_exact(double xmin, double ymin,
                                         double xmax, double ymax,
                                         double rx, double ry,
                                         double cos_theta,
                                         double sin_theta) noexcept nogil


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double fmin(double x, double y)


cdef inline void ellipse_quadratic_coeffs(double rx, double ry,
                                          double cos_theta,
                                          double sin_theta,
                                          double dx, double dy,
                                          double *cxx, double *cyy,
                                          double *cxy, double *f_in,
                                          double *f_out) noexcept nogil:
    """
    Compute the quadratic-form coefficients and interior/exterior
    fast-path thresholds of an ellipse with semi-axes ``rx``/``ry`` and
    orientation ``cos_theta``/``sin_theta``, for a pixel grid of spacing
    ``dx``/``dy``.

    A point (x, y) lies inside the ellipse when
    ``cxx*x**2 + cyy*y**2 + cxy*x*y < 1``. A pixel is wholly inside
    when its center value is ``<= f_in`` and wholly outside when it is
    ``>= f_out`` (see ``elliptical_overlap_grid``). The results are
    returned through the output pointers so the same setup can be shared
    by the grid kernel (computed once per grid) and the batch ellipse
    helpers.
    """
    cdef double inv_rx2 = 1.0 / (rx * rx)
    cdef double inv_ry2 = 1.0 / (ry * ry)
    cdef double margin, fin

    cxx[0] = cos_theta * cos_theta * inv_rx2 + sin_theta * sin_theta * inv_ry2
    cyy[0] = sin_theta * sin_theta * inv_rx2 + cos_theta * cos_theta * inv_ry2
    cxy[0] = 2.0 * cos_theta * sin_theta * (inv_rx2 - inv_ry2)

    margin = 0.5 * sqrt(dx * dx + dy * dy) / fmin(rx, ry)
    fin = 1.0 - margin
    f_in[0] = fin * fin if fin > 0.0 else 0.0
    f_out[0] = (1.0 + margin) * (1.0 + margin)


cdef inline double ellipse_frac_from_rpix2(double pxmin, double pymin,
                                           double pxmax, double pymax,
                                           double norm, double rx, double ry,
                                           double cos_theta, double sin_theta,
                                           double rpix2, double f_in,
                                           double f_out, int use_exact,
                                           int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps an ellipse, given the
    precomputed quadratic-form value ``rpix2`` at the pixel center and
    the fast-path thresholds ``f_in``/``f_out``.

    This is the shared per-pixel decision core for the elliptical
    overlap (interior/exterior fast path and exact/subpixel dispatch),
    called by both ``elliptical_overlap_grid`` and the batch ellipse
    helpers. No bounding-box check is performed (the caller is
    responsible for it), so it can be shared by the ellipse and
    elliptical-annulus helpers.
    """
    if rpix2 >= f_out:
        return 0.0  # pixel fully outside the ellipse
    if rpix2 <= f_in:
        return 1.0  # pixel fully inside the ellipse

    if use_exact:
        return ellipse_overlap_single_exact(pxmin, pymin, pxmax, pymax, rx, ry,
                                            cos_theta, sin_theta) * norm
    return ellipse_overlap_single_subpixel(pxmin, pymin, pxmax, pymax, rx, ry,
                                           cos_theta, sin_theta, subpixels)
