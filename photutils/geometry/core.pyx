# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
The functions here are the core geometry functions.
"""

import numpy as np

cimport numpy as np


cdef extern from "math.h":

    double asin(double x)
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double fabs(double x)

from cpython cimport bool

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

ctypedef struct point:
    double x
    double y


ctypedef struct intersections:
    point p1
    point p2


cdef double floor_sqrt(double x):
    """
    In some of the geometrical functions, we have to take the sqrt of a number
    and we know that the number should be >= 0. However, in some cases the
    value is e.g. -1e-10, but we want to treat it as zero, which is what
    this function does.

    Note that this does **not** check whether negative values are close or not
    to zero, so this should be used only in cases where the value is expected
    to be positive on paper.
    """
    if x > 0:
        return sqrt(x)
    else:
        return 0

# NOTE: The following two functions use cdef because they are not intended to be
# called from the Python code. Using def makes them callable from outside, but
# also slower. Some functions currently return multiple values, and for those we
# still use 'def' for now.


cdef double distance(double x1, double y1, double x2, double y2):
    """
    Distance between two points in two dimensions.

    Parameters
    ----------
    x1, y1 : float
        The coordinates of the first point
    x2, y2 : float
        The coordinates of the second point

    Returns
    -------
    d : float
        The Euclidean distance between the two points
    """

    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


cdef double area_arc(double x1, double y1, double x2, double y2, double r):
    """
    Area of a circle arc with radius r between points (x1, y1) and (x2, y2).

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """

    cdef double a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2.0 * asin(0.5 * a / r)
    return 0.5 * r * r * (theta - sin(theta))


cdef double area_triangle(double x1, double y1, double x2, double y2, double x3,
                  double y3):
    """
    Area of a triangle defined by three vertices.
    """
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


cdef double area_arc_unit(double x1, double y1, double x2, double y2):
    """
    Area of a circle arc with radius R between points (x1, y1) and (x2, y2)

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """
    cdef double a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2.0 * asin(0.5 * a)
    return 0.5 * (theta - sin(theta))


cdef int in_triangle(double x, double y, double x1, double y1, double x2, double y2, double x3, double y3):
    """
    Check if a point (x,y) is inside a triangle
    """
    cdef int c = 0

    c += ((y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    c += ((y2 > y) != (y3 > y) and x < (x3 - x2) * (y - y2) / (y3 - y2) + x2)
    c += ((y3 > y) != (y1 > y) and x < (x1 - x3) * (y - y3) / (y1 - y3) + x3)

    return c % 2 == 1


cdef intersections circle_line(double x1, double y1, double x2, double y2):
    """Intersection of a line defined by two points with a unit circle"""

    cdef double a, b, delta, dx, dy
    cdef double tolerance = 1.e-10
    cdef intersections inter

    dx = x2 - x1
    dy = y2 - y1

    if fabs(dx) < tolerance and fabs(dy) < tolerance:
        inter.p1.x = 2.
        inter.p1.y = 2.
        inter.p2.x = 2.
        inter.p2.y = 2.

    elif fabs(dx) > fabs(dy):

        # Find the slope and intercept of the line
        a = dy / dx
        b = y1 - a * x1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b
        if delta > 0.:  # solutions exist

            delta = sqrt(delta)

            inter.p1.x = (- a * b - delta) / (1. + a * a)
            inter.p1.y = a * inter.p1.x + b
            inter.p2.x = (- a * b + delta) / (1. + a * a)
            inter.p2.y = a * inter.p2.x + b

        else:  # no solution, return values > 1
            inter.p1.x = 2.
            inter.p1.y = 2.
            inter.p2.x = 2.
            inter.p2.y = 2.

    else:

        # Find the slope and intercept of the line
        a = dx / dy
        b = x1 - a * y1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b

        if delta > 0.:  # solutions exist

            delta = sqrt(delta)

            inter.p1.y = (- a * b - delta) / (1. + a * a)
            inter.p1.x = a * inter.p1.y + b
            inter.p2.y = (- a * b + delta) / (1. + a * a)
            inter.p2.x = a * inter.p2.y + b

        else:  # no solution, return values > 1
            inter.p1.x = 2.
            inter.p1.y = 2.
            inter.p2.x = 2.
            inter.p2.y = 2.

    return inter


cdef point circle_segment_single2(double x1, double y1, double x2, double y2):
    """
    The intersection of a line with the unit circle. The intersection the
    closest to (x2, y2) is chosen.
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


cdef intersections circle_segment(double x1, double y1, double x2, double y2):
    """
    Intersection(s) of a segment with the unit circle. Discard any
    solution not on the segment.
    """

    cdef intersections inter, inter_new
    cdef point pt1, pt2

    inter = circle_line(x1, y1, x2, y2)

    pt1 = inter.p1
    pt2 = inter.p2

    if (pt1.x > x1 and pt1.x > x2) or (pt1.x < x1 and pt1.x < x2) or (pt1.y > y1 and pt1.y > y2) or (pt1.y < y1 and pt1.y < y2):
        pt1.x, pt1.y = 2., 2.
    if (pt2.x > x1 and pt2.x > x2) or (pt2.x < x1 and pt2.x < x2) or (pt2.y > y1 and pt2.y > y2) or (pt2.y < y1 and pt2.y < y2):
        pt2.x, pt2.y = 2., 2.

    if pt1.x > 1. and pt2.x < 2.:
        inter_new.p1 = pt1
        inter_new.p2 = pt2
    else:
        inter_new.p1 = pt2
        inter_new.p2 = pt1

    return inter_new


cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2, double y2, double x3, double y3):
    """
    Given a triangle defined by three points (x1, y1), (x2, y2), and
    (x3, y3), find the area of overlap with the unit circle.
    """

    cdef double d1, d2, d3
    cdef bool in1, in2, in3
    cdef bool on1, on2, on3
    cdef double area
    cdef double PI = np.pi
    cdef intersections inter
    cdef point pt1, pt2, pt3, pt4, pt5, pt6, pt_tmp

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
            x1, y1, d1, x2, y2, d2, x3, y3, d3 = x3, y3, d3, x1, y1, d1, x2, y2, d2

    else:
        if d1 < d3:
            x1, y1, d1, x2, y2, d2 = x2, y2, d2, x1, y1, d1
        elif d2 < d3:
            x1, y1, d1, x2, y2, d2, x3, y3, d3 = x2, y2, d2, x3, y3, d3, x1, y1, d1
        else:
            x1, y1, d1, x2, y2, d2, x3, y3, d3 = x3, y3, d3, x2, y2, d2, x1, y1, d1

    if d1 > d2 or d2 > d3 or d1 > d3:
        raise Exception("ERROR: vertices did not sort correctly")

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
        # If vertex 1 or 2 are on the edge of the circle, then we use the dot
        # product to vertex 3 to determine whether an intersection takes place.
        intersect13 = not on1 or x1 * (x3 - x1) + y1 * (y3 - y1) < 0.
        intersect23 = not on2 or x2 * (x3 - x2) + y2 * (y3 - y2) < 0.
        if intersect13 and intersect23 and not on2:
            pt1 = circle_segment_single2(x1, y1, x3, y3)
            pt2 = circle_segment_single2(x2, y2, x3, y3)
            area = area_triangle(x1, y1, x2, y2, pt1.x, pt1.y) \
                 + area_triangle(x2, y2, pt1.x, pt1.y, pt2.x, pt2.y) \
                 + area_arc_unit(pt1.x, pt1.y, pt2.x, pt2.y)
        elif intersect13:
            pt1 = circle_segment_single2(x1, y1, x3, y3)
            area = area_triangle(x1, y1, x2, y2, pt1.x, pt1.y) \
                 + area_arc_unit(x2, y2, pt1.x, pt1.y)
        elif intersect23:
            pt2 = circle_segment_single2(x2, y2, x3, y3)
            area = area_triangle(x1, y1, x2, y2, pt2.x, pt2.y) \
                 + area_arc_unit(x1, y1, pt2.x, pt2.y)
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
            # Code taken from `sep.h`.
            # TODO: use `sep` and get rid of this Cython code.
            if (((0.-pt3.y) * (pt4.x-pt3.x) > (pt4.y-pt3.y) * (0.-pt3.x)) !=
                ((y1-pt3.y) * (pt4.x-pt3.x) > (pt4.y-pt3.y) * (x1-pt3.x))):
                area = area_triangle(x1, y1, pt3.x, pt3.y, pt4.x, pt4.y) \
                     + (PI - area_arc_unit(pt3.x, pt3.y, pt4.x, pt4.y))
            else:
                area = area_triangle(x1, y1, pt3.x, pt3.y, pt4.x, pt4.y) \
                     + area_arc_unit(pt3.x, pt3.y, pt4.x, pt4.y)
        else:
            if (pt2.x - x2)**2 + (pt2.y - y2)**2 < (pt1.x - x2)**2 + (pt1.y - y2)**2:
                pt1, pt2 = pt2, pt1
            area = area_triangle(x1, y1, pt3.x, pt3.y, pt1.x, pt1.y) \
                 + area_triangle(x1, y1, pt1.x, pt1.y, pt2.x, pt2.y) \
                 + area_triangle(x1, y1, pt2.x, pt2.y, pt4.x, pt4.y) \
                 + area_arc_unit(pt1.x, pt1.y, pt3.x, pt3.y) \
                 + area_arc_unit(pt2.x, pt2.y, pt4.x, pt4.y)
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
            area = overlap_area_triangle_unit_circle(x1, y1, x3, y3, xp, yp) \
                 + overlap_area_triangle_unit_circle(x2, y2, x3, y3, xp, yp)
        elif pt3.x <= 1.:
            xp, yp = 0.5 * (pt3.x + pt4.x), 0.5 * (pt3.y + pt4.y)
            area = overlap_area_triangle_unit_circle(x3, y3, x1, y1, xp, yp) \
                 + overlap_area_triangle_unit_circle(x2, y2, x1, y1, xp, yp)
        elif pt5.x <= 1.:
            xp, yp = 0.5 * (pt5.x + pt6.x), 0.5 * (pt5.y + pt6.y)
            area = overlap_area_triangle_unit_circle(x1, y1, x2, y2, xp, yp) \
                 + overlap_area_triangle_unit_circle(x3, y3, x2, y2, xp, yp)
        else:  # no intersections
            if in_triangle(0., 0., x1, y1, x2, y2, x3, y3):
                return PI
            else:
                return 0.

    return area
