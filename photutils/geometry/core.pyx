# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Core geometry functions for computing the overlap of an aperture and a
pixel grid.

The cdef functions are not intended to be called from Python code. They
are pure C math functions declared ``noexcept nogil`` so they can be
called without the GIL, including from multiple threads on free-threaded
Python builds. Their signatures are exported via core.pxd.
"""

cdef extern from "math.h" nogil:
    double asin(double x)
    double sin(double x)
    double sqrt(double x)
    double fabs(double x)
    double M_PI
    double NAN

ctypedef struct point:
    double x
    double y

ctypedef struct intersections:
    point p1
    point p2


cdef double floor_sqrt(double x) noexcept nogil:
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


cdef double distance(double x1, double y1, double x2,
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


cdef double area_arc(double x1, double y1, double x2, double y2,
                     double r) noexcept nogil:
    """
    Area of a circular segment: the region between the chord
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
    theta = 2.0 * asin(0.5 * a / r)
    return 0.5 * r * r * (theta - sin(theta))


cdef double area_triangle(double x1, double y1, double x2, double y2,
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
                          double y2) noexcept nogil:
    """
    Area of a circular segment of the unit circle (radius 1): the
    region between the chord connecting (x1, y1) and (x2, y2) and the
    circular arc spanning the same two points.

    Both points are assumed to lie on the unit circle centered on the
    origin. This is equivalent to ``area_arc`` with ``r = 1``.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        The coordinates of the two points defining the chord.

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
    theta = 2.0 * asin(0.5 * a)
    return 0.5 * (theta - sin(theta))


cdef int in_triangle(double x, double y, double x1, double y1, double x2,
                     double y2, double x3, double y3) noexcept nogil:
    """
    Test whether a point lies inside a triangle.

    Parameters
    ----------
    x, y : float
        The coordinates of the point to test.

    x1, y1, x2, y2, x3, y3 : float
        The coordinates of the triangle's three vertices.

    Returns
    -------
    result : int
        1 if the point lies inside the triangle, 0 otherwise.
    """
    cdef int c = 0

    c += ((y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    c += ((y2 > y) != (y3 > y) and x < (x3 - x2) * (y - y2) / (y3 - y2) + x2)
    c += ((y3 > y) != (y1 > y) and x < (x1 - x3) * (y - y3) / (y1 - y3) + x3)

    return c % 2 == 1


cdef intersections circle_line(double x1, double y1, double x2,
                               double y2) noexcept nogil:
    """
    Intersection of the infinite line through two points with the unit
    circle.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        The coordinates of two distinct points defining the line.

    Returns
    -------
    result : intersections
        The two intersection points, ``p1`` and ``p2``. Coordinates
        greater than 1 indicate that there is no intersection.
    """
    cdef double a, b, delta, dx, dy
    cdef double tolerance = 1.e-10
    cdef intersections inter

    dx = x2 - x1
    dy = y2 - y1

    # Initialize to values > 1, indicating no intersection
    inter.p1.x = 2.
    inter.p1.y = 2.
    inter.p2.x = 2.
    inter.p2.y = 2.

    if fabs(dx) < tolerance and fabs(dy) < tolerance:
        return inter

    if fabs(dx) > fabs(dy):
        # Find the slope and intercept of the line
        a = dy / dx
        b = y1 - a * x1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b
        if delta > 0.:  # solutions exist
            delta = sqrt(delta)

            inter.p1.x = (-a * b - delta) / (1. + a * a)
            inter.p1.y = a * inter.p1.x + b
            inter.p2.x = (-a * b + delta) / (1. + a * a)
            inter.p2.y = a * inter.p2.x + b
    else:
        # Find the slope and intercept of the line
        a = dx / dy
        b = x1 - a * y1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b
        if delta > 0.:  # solutions exist
            delta = sqrt(delta)

            inter.p1.y = (-a * b - delta) / (1. + a * a)
            inter.p1.x = a * inter.p1.y + b
            inter.p2.y = (-a * b + delta) / (1. + a * a)
            inter.p2.x = a * inter.p2.y + b

    return inter


cdef point circle_segment_single2(double x1, double y1, double x2,
                                  double y2) noexcept nogil:
    """
    Intersection of the infinite line through two points with the unit
    circle, choosing the intersection closest to the second point.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        The coordinates of two distinct points defining the line.

    Returns
    -------
    result : point
        The intersection point closest to ``(x2, y2)``.
    """
    cdef double dx1, dy1, dx2, dy2
    cdef intersections inter
    cdef point pt1, pt2, pt

    inter = circle_line(x1, y1, x2, y2)

    pt1 = inter.p1
    pt2 = inter.p2

    # Can be optimized, but just checking for correctness right now
    dx1 = fabs(pt1.x - x2)
    dy1 = fabs(pt1.y - y2)
    dx2 = fabs(pt2.x - x2)
    dy2 = fabs(pt2.y - y2)

    if dx1 > dy1:  # compare based on x-axis
        if dx1 > dx2:
            pt = pt2
        else:
            pt = pt1
    else:
        if dy1 > dy2:
            pt = pt2
        else:
            pt = pt1

    return pt


cdef intersections circle_segment(double x1, double y1, double x2,
                                  double y2) noexcept nogil:
    """
    Intersection(s) of a line segment with the unit circle.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        The coordinates of the segment's two endpoints.

    Returns
    -------
    result : intersections
        The two intersection points, ``p1`` and ``p2``. Any solution
        that does not lie on the segment is discarded (coordinates set
        to 2, which is outside the unit circle).
    """
    cdef intersections inter, inter_new
    cdef point pt1, pt2

    inter = circle_line(x1, y1, x2, y2)

    pt1 = inter.p1
    pt2 = inter.p2

    if ((pt1.x > x1 and pt1.x > x2) or (pt1.x < x1 and pt1.x < x2)
            or (pt1.y > y1 and pt1.y > y2)
            or (pt1.y < y1 and pt1.y < y2)):
        pt1.x, pt1.y = 2., 2.
    if ((pt2.x > x1 and pt2.x > x2) or (pt2.x < x1 and pt2.x < x2)
            or (pt2.y > y1 and pt2.y > y2)
            or (pt2.y < y1 and pt2.y < y2)):
        pt2.x, pt2.y = 2., 2.

    if pt1.x > 1. and pt2.x < 2.:
        inter_new.p1 = pt1
        inter_new.p2 = pt2
    else:
        inter_new.p1 = pt2
        inter_new.p2 = pt1

    return inter_new


cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2,
                                              double y2, double x3,
                                              double y3) noexcept nogil:
    """
    Area of overlap between a triangle and the unit circle centered on
    the origin.

    Parameters
    ----------
    x1, y1, x2, y2, x3, y3 : float
        The coordinates of the triangle's three vertices.

    Returns
    -------
    area : float
        The area of overlap between the triangle and the unit circle.
    """
    cdef double d1, d2, d3
    cdef bint in1, in2, in3
    cdef bint on1, on2, on3
    cdef bint intersect13, intersect23
    cdef bint cond1, cond2
    cdef double area, xp, yp
    cdef intersections inter
    cdef point pt1, pt2, pt3, pt4, pt5, pt6

    # Find distance of all vertices to circle center
    d1 = x1 * x1 + y1 * y1
    d2 = x2 * x2 + y2 * y2
    d3 = x3 * x3 + y3 * y3

    # Order vertices by distance from origin
    if d1 < d2:
        if d2 < d3:
            pass
        elif d1 < d3:
            x2, y2, d2, x3, y3, d3 = x3, y3, d3, x2, y2, d2
        else:
            (x1, y1, d1, x2, y2, d2, x3, y3, d3) = (
                x3, y3, d3, x1, y1, d1, x2, y2, d2)

    else:
        if d1 < d3:
            x1, y1, d1, x2, y2, d2 = x2, y2, d2, x1, y1, d1
        elif d2 < d3:
            (x1, y1, d1, x2, y2, d2, x3, y3, d3) = (
                x2, y2, d2, x3, y3, d3, x1, y1, d1)
        else:
            (x1, y1, d1, x2, y2, d2, x3, y3, d3) = (
                x3, y3, d3, x2, y2, d2, x1, y1, d1)

    if d1 > d2 or d2 > d3 or d1 > d3:
        # This should never happen; return NaN to signal an internal
        # logic error without requiring the GIL to raise an exception.
        return NAN

    # Determine number of vertices inside circle
    in1 = d1 < 1
    in2 = d2 < 1
    in3 = d3 < 1

    # Determine which vertices are on the circle
    on1 = fabs(d1 - 1) < 1.e-10
    on2 = fabs(d2 - 1) < 1.e-10
    on3 = fabs(d3 - 1) < 1.e-10

    if on3 or in3:  # triangle is completely in circle
        area = area_triangle(x1, y1, x2, y2, x3, y3)

    elif in2 or on2:
        # If vertex 1 or 2 are on the edge of the circle, then we
        # use the dot product to vertex 3 to determine whether an
        # intersection takes place.
        intersect13 = not on1 or x1 * (x3 - x1) + y1 * (y3 - y1) < 0.
        intersect23 = not on2 or x2 * (x3 - x2) + y2 * (y3 - y2) < 0.
        if intersect13 and intersect23 and not on2:
            pt1 = circle_segment_single2(x1, y1, x3, y3)
            pt2 = circle_segment_single2(x2, y2, x3, y3)
            area = (area_triangle(x1, y1, x2, y2, pt1.x, pt1.y)
                    + area_triangle(x2, y2, pt1.x, pt1.y, pt2.x, pt2.y)
                    + area_arc_unit(pt1.x, pt1.y, pt2.x, pt2.y))
        elif intersect13:
            pt1 = circle_segment_single2(x1, y1, x3, y3)
            area = (area_triangle(x1, y1, x2, y2, pt1.x, pt1.y)
                    + area_arc_unit(x2, y2, pt1.x, pt1.y))
        elif intersect23:
            pt2 = circle_segment_single2(x2, y2, x3, y3)
            area = (area_triangle(x1, y1, x2, y2, pt2.x, pt2.y)
                    + area_arc_unit(x1, y1, pt2.x, pt2.y))
        else:
            area = area_arc_unit(x1, y1, x2, y2)

    elif on1:
        # The triangle is outside the circle
        area = 0.0

    elif in1:
        # Check for intersections of far side with circle
        inter = circle_segment(x2, y2, x3, y3)
        pt1 = inter.p1
        pt2 = inter.p2
        pt3 = circle_segment_single2(x1, y1, x2, y2)
        pt4 = circle_segment_single2(x1, y1, x3, y3)

        if pt1.x > 1.:  # indicates no intersection
            # Code from `sep.h`
            cond1 = ((0. - pt3.y) * (pt4.x - pt3.x)
                     > (pt4.y - pt3.y) * (0. - pt3.x))
            cond2 = ((y1 - pt3.y) * (pt4.x - pt3.x)
                     > (pt4.y - pt3.y) * (x1 - pt3.x))
            if cond1 != cond2:
                area = (area_triangle(x1, y1, pt3.x, pt3.y, pt4.x, pt4.y)
                        + (M_PI - area_arc_unit(pt3.x, pt3.y,
                                                pt4.x, pt4.y)))
            else:
                area = (area_triangle(x1, y1, pt3.x, pt3.y, pt4.x, pt4.y)
                        + area_arc_unit(pt3.x, pt3.y, pt4.x, pt4.y))
        else:
            if ((pt2.x - x2)**2 + (pt2.y - y2)**2
                    < (pt1.x - x2)**2 + (pt1.y - y2)**2):
                pt1, pt2 = pt2, pt1
            area = (area_triangle(x1, y1, pt3.x, pt3.y, pt1.x, pt1.y)
                    + area_triangle(x1, y1, pt1.x, pt1.y, pt2.x, pt2.y)
                    + area_triangle(x1, y1, pt2.x, pt2.y, pt4.x, pt4.y)
                    + area_arc_unit(pt1.x, pt1.y, pt3.x, pt3.y)
                    + area_arc_unit(pt2.x, pt2.y, pt4.x, pt4.y))

    else:
        inter = circle_segment(x1, y1, x2, y2)
        pt1 = inter.p1
        pt2 = inter.p2
        inter = circle_segment(x2, y2, x3, y3)
        pt3 = inter.p1
        pt4 = inter.p2
        inter = circle_segment(x3, y3, x1, y1)
        pt5 = inter.p1
        pt6 = inter.p2

        if pt1.x <= 1.:
            xp, yp = 0.5 * (pt1.x + pt2.x), 0.5 * (pt1.y + pt2.y)
            area = (overlap_area_triangle_unit_circle(x1, y1, x3, y3, xp, yp)
                    + overlap_area_triangle_unit_circle(x2, y2, x3, y3,
                                                        xp, yp))
        elif pt3.x <= 1.:
            xp, yp = 0.5 * (pt3.x + pt4.x), 0.5 * (pt3.y + pt4.y)
            area = (overlap_area_triangle_unit_circle(x3, y3, x1, y1, xp, yp)
                    + overlap_area_triangle_unit_circle(x2, y2, x1, y1,
                                                        xp, yp))
        elif pt5.x <= 1.:
            xp, yp = 0.5 * (pt5.x + pt6.x), 0.5 * (pt5.y + pt6.y)
            area = (overlap_area_triangle_unit_circle(x1, y1, x2, y2, xp, yp)
                    + overlap_area_triangle_unit_circle(x3, y3, x2, y2,
                                                        xp, yp))
        else:  # no intersections
            if in_triangle(0., 0., x1, y1, x2, y2, x3, y3):
                return M_PI
            else:
                return 0.

    return area
