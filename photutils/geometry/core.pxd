# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the core geometry functions into other
Cython files. All functions are pure C math functions that are safe to
call without the GIL.

The small helpers used in per-pixel hot loops are defined here as
``cdef inline`` functions so that the C compiler can inline them into
the modules that cimport them.
"""

cdef extern from "math.h" nogil:
    double asin(double x)
    double sin(double x)
    double sqrt(double x)
    double fabs(double x)
    double fmin(double x, double y)


cdef inline double floor_sqrt(double x) noexcept nogil:
    """
    Square root of a value expected to be non-negative, treating small
    negative values (e.g., from floating-point round-off) as zero.

    This does not check whether a negative value is actually close to
    zero, so it should be used only where the true value is expected
    to be non-negative.

    Parameters
    ----------
    x : double
        The value to take the square root of.

    Returns
    -------
    result : double
        ``sqrt(x)`` if ``x > 0``, otherwise 0.
    """
    if x > 0:
        return sqrt(x)
    else:
        return 0


cdef inline double distance(double x1, double y1, double x2,
                            double y2) noexcept nogil:
    """
    Distance between two points in two dimensions.

    Parameters
    ----------
    x1, y1 : float
        The coordinates of the first point.

    x2, y2 : float
        The coordinates of the second point.

    Returns
    -------
    d : float
        The Euclidean distance between the two points.
    """
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


cdef inline double area_arc(double x1, double y1, double x2, double y2,
                            double r) noexcept nogil:
    """
    Area of a circular segment, the region between the chord
    connecting (x1, y1) and (x2, y2) and the circular arc of radius
    ``r`` spanning the same two points.

    Both points are assumed to lie on a circle of radius ``r`` centered
    on the origin.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        The coordinates of the two points defining the chord.

    r : float
        The radius of the circle.

    Returns
    -------
    area : float
        The area of the circular segment.

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """
    cdef double a, theta

    a = distance(x1, y1, x2, y2)
    # Clamp the half-chord ratio to 1 to guard against floating-point
    # round-off pushing a near-diameter chord past the domain of asin,
    # which would return NaN.
    theta = 2.0 * asin(fmin(1.0, 0.5 * a / r))
    return 0.5 * r * r * (theta - sin(theta))


cdef inline double area_triangle(double x1, double y1, double x2, double y2,
                                 double x3, double y3) noexcept nogil:
    """
    Area of a triangle defined by three vertices.

    Parameters
    ----------
    x1, y1, x2, y2, x3, y3 : float
        The coordinates of the three vertices.

    Returns
    -------
    area : float
        The (unsigned) area of the triangle.
    """
    return 0.5 * fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


cdef double area_arc_unit(double x1, double y1, double x2,
                          double y2) noexcept nogil
cdef int in_triangle(double x, double y, double x1, double y1, double x2,
                     double y2, double x3, double y3) noexcept nogil
cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2,
                                              double y2, double x3,
                                              double y3) noexcept nogil
