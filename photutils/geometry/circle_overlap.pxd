# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the circle overlap functions into other
Cython files. These single-pixel functions are pure C math functions
that are safe to call without the GIL.
"""

cdef double circle_overlap_single_subpixel(double x0, double y0,
                                           double x1, double y1,
                                           double r,
                                           int subpixels) noexcept nogil
cdef double circle_overlap_single_exact(double xmin, double ymin,
                                        double xmax, double ymax,
                                        double r) noexcept nogil


cdef inline double circle_frac_from_d2(double pxmin, double pymin,
                                       double pxmax, double pymax,
                                       double dx, double dy,
                                       double pixel_radius, double d2,
                                       double r, int use_exact,
                                       int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a circle of radius ``r``
    centered on the origin, given the precomputed squared distance
    ``d2`` from the origin to the pixel center.

    This is the per-pixel decision core for the circular overlap,
    factored out of ``circular_overlap_grid`` so the interior/exterior
    fast path and the exact/subpixel dispatch live in a single,
    reusable place.

    Using the squared distance avoids a ``sqrt`` per pixel: the
    interior/exterior fast-path thresholds ``r ± pixel_radius`` are
    simply squared. No bounding-box check is performed (the caller is
    responsible for it), so the same core can be shared by the circle
    and circular-annulus helpers to avoid recomputing ``d2`` for each
    boundary.

    Parameters
    ----------
    pxmin, pymin, pxmax, pymax : double
        The pixel edges, relative to the circle center.

    dx, dy : double
        The pixel width and height.

    pixel_radius : double
        Half the pixel diagonal.

    d2 : double
        The squared distance from the circle center to the pixel
        center.

    r : double
        The radius of the circle.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        circle.
    """
    cdef double r_inner = r - pixel_radius
    cdef double r_outer = r + pixel_radius

    # Pixel is well within the circle
    if r_inner > 0.0 and d2 < r_inner * r_inner:
        return 1.0

    if d2 < r_outer * r_outer:  # pixel is close to the circle border
        if use_exact:
            return circle_overlap_single_exact(pxmin, pymin, pxmax, pymax,
                                               r) / (dx * dy)
        return circle_overlap_single_subpixel(pxmin, pymin, pxmax, pymax, r,
                                              subpixels)

    return 0.0  # pixel is fully outside the circle
