# Licensed under a 3-clause BSD style license - see LICENSE.rst

# The functions here are core geomet
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
cimport numpy as np

__all__ = ['elliptical_overlap_grid']

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

# NOTE: The following two functions use cdef because they are not intended to be
# called from the Python code. Using def makes them callable from outside, but
# also slower. Some functions currently return multiple values, and for those we
# still use 'def' for now.


cdef double distance(double x1, double y1, double x2, double y2):
    """
    Distance between two points in two dimensions
    
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


cdef double area_arc(double x1, double y1, double x2, double y2, double R):
    """
    Area of a circle arc with radius R between points (x1, y1) and (x2, y2).

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """

    cdef double a, theta
    a = distance(x1, y1, x2, y2)
    theta = 2. * asin(0.5 * a / R)
    return 0.5 * R * R * (theta - sin(theta))


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
    theta = 2. * asin(0.5 * a)
    return 0.5 * (theta - sin(theta))


cdef double in_triangle(double x, double y, double x1, double y1, double x2, double y2, double x3, double y3):
    """
    Check if a point (x,y) is inside a triangle
    """
    cdef int c = 0

    c += ((y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    c += ((y2 > y) != (y3 > y) and x < (x3 - x2) * (y - y2) / (y3 - y2) + x2)
    c += ((y3 > y) != (y1 > y) and x < (x1 - x3) * (y - y3) / (y1 - y3) + x3)

    return c % 2 == 1


def circle_line(double x1, double y1, double x2, double y2):
    """Intersection of a line defined by two points with a unit circle"""

    cdef double a, b, delta, dx, dy
    cdef double xi1, yi1, xi2, yi2
    cdef double tolerance = 1.e-10

    dx = x2 - x1
    dy = y2 - y1

    if fabs(dx) < tolerance and fabs(dy) < tolerance:

        return 2., 2., 2., 2.

    if fabs(dx) > fabs(dy):

        # Find the slope and intercept of the line
        a = dy / dx
        b = y1 - a * x1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b

        if delta > 0.:  # solutions exist
            delta = sqrt(delta)
            xi1 = (- a * b - delta) / (1. + a * a)
            yi1 = a * xi1 + b
            xi2 = (- a * b + delta) / (1. + a * a)
            yi2 = a * xi2 + b
            return xi1, yi1, xi2, yi2
        else:  # no solution, return values > 1
            return 2., 2., 2., 2.

    else:

        # Find the slope and intercept of the line
        a = dx / dy
        b = x1 - a * y1

        # Find the determinant of the quadratic equation
        delta = 1. + a * a - b * b

        if delta > 0.:  # solutions exist
            delta = sqrt(delta)
            yi1 = (- a * b - delta) / (1. + a * a)
            xi1 = a * yi1 + b
            yi2 = (- a * b + delta) / (1. + a * a)
            xi2 = a * yi2 + b
            return xi1, yi1, xi2, yi2
        else:  # no solution, return values > 1
            return 2., 2., 2., 2.


def circle_segment_single2(double x1, double y1, double x2, double y2):
    """
    The intersection of a line with the unit circle. The intersection the
    closest to (x2, y2) is chosen.
    """

    cdef double xi1, yi1, xi2, yi2
    cdef double dx1, dy1, dx2, dy2

    xi1, yi1, xi2, yi2 = circle_line(x1, y1, x2, y2)

    # Can be optimized, but just checking for correctness right now
    dx1 = abs(xi1 - x2)
    dy1 = abs(yi1 - y2)
    dx2 = abs(xi2 - x2)
    dy2 = abs(yi2 - y2)

    if dx1 > dy1:  # compare based on x-axis
        if dx1 > dx2:
            return xi2, yi2
        else:
            return xi1, yi1
    else:
        if dy1 > dy2:
            return xi2, yi2
        else:
            return xi1, yi1


def circle_segment(double x1, double y1, double x2, double y2):
    """
    Intersection(s) of a segment with the unit circle. Discard any
    solution not on the segment.
    """

    cdef double xi1, yi1, xi2, yi2

    xi1, yi1, xi2, yi2 = circle_line(x1, y1, x2, y2)

    if (xi1 > x1 and xi1 > x2) or (xi1 < x1 and xi1 < x2) or (yi1 > y1 and yi1 > y2) or (yi1 < y1 and yi1 < y2):
        xi1, yi1 = 2., 2.
    if (xi2 > x1 and xi2 > x2) or (xi2 < x1 and xi2 < x2) or (yi2 > y1 and yi2 > y2) or (yi2 < y1 and yi2 < y2):
        xi2, yi2 = 2., 2.

    if xi1 > 1. and xi2 < 2.:
        return xi1, yi1, xi2, yi2
    else:
        return xi2, yi2, xi1, yi1


def overlap_area_triangle_unit_circle(double x1, double y1, double x2, double y2, double x3, double y3):
    """
    Given a triangle defined by three points (x1, y1), (x2, y2), and
    (x3, y3), find the area of overlap with the unit circle.
    """

    cdef double d1, d2, d3
    cdef bool in1, in2, in3
    cdef double xc1, yc1
    cdef double xc2, yc2
    cdef double xc3, yc3
    cdef double xc4, yc4
    cdef double area
    cdef double PI = np.pi

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
    on1 = abs(d1 - 1) < 1.e-10
    on2 = abs(d2 - 1) < 1.e-10
    on3 = abs(d3 - 1) < 1.e-10

    if on3 or in3:  # triangle is completely in circle

        area = area_triangle(x1, y1, x2, y2, x3, y3)

    elif in2 or on2:

        # If vertex 1 or 2 are on the edge of the circle, then we use the dot
        # product to vertex 3 to determine whether an intersection takes place.
        intersect13 = not on1 or x1 * (x3 - x1) + y1 * (y3 - y1) < 0.
        intersect23 = not on2 or x2 * (x3 - x2) + y2 * (y3 - y2) < 0.

        if intersect13 and intersect23:
            xc1, yc1 = circle_segment_single2(x1, y1, x3, y3)
            xc2, yc2 = circle_segment_single2(x2, y2, x3, y3)
            area = area_triangle(x1, y1, x2, y2, xc1, yc1) \
                 + area_triangle(x2, y2, xc1, yc1, xc2, yc2) \
                 + area_arc_unit(xc1, yc1, xc2, yc2)
        elif intersect13:
            xc1, yc1 = circle_segment_single2(x1, y1, x3, y3)
            area = area_triangle(x1, y1, x2, y2, xc1, yc1) \
                 + area_arc_unit(x2, y2, xc1, yc1)
        elif intersect23:
            xc2, yc2 = circle_segment_single2(x2, y2, x3, y3)
            area = area_triangle(x1, y1, x2, y2, xc2, yc2) \
                 + area_arc_unit(x1, y1, xc2, yc2)
        else:
            area = area_arc_unit(x1, y1, x2, y2)

    elif in1:
        # Check for intersections of far side with circle
        xc1, yc1, xc2, yc2 = circle_segment(x2, y2, x3, y3)
        xc3, yc3 = circle_segment_single2(x1, y1, x2, y2)
        xc4, yc4 = circle_segment_single2(x1, y1, x3, y3)
        if xc1 > 1.:  # indicates no intersection
            if in_triangle(0, 0, x1, y1, x2, y2, x3, y3) and not in_triangle(0, 0, x1, y1, xc3, yc3, xc4, yc4):
                area = area_triangle(x1, y1, xc3, yc3, xc4, yc4) \
                     + (PI - area_arc_unit(xc3, yc3, xc4, yc4))
            else:
                area = area_triangle(x1, y1, xc3, yc3, xc4, yc4) \
                     + area_arc_unit(xc3, yc3, xc4, yc4)
        else:
            if abs(xc2 - x2) < abs(xc1 - x2):
                xc1, yc1, xc2, yc2 = xc2, yc2, xc1, yc1
            area = area_triangle(x1, y1, xc3, yc3, xc1, yc1) \
                 + area_triangle(x1, y1, xc1, yc1, xc2, yc2) \
                 + area_triangle(x1, y1, xc2, yc2, xc4, yc4) \
                 + area_arc_unit(xc1, yc1, xc3, yc3) \
                 + area_arc_unit(xc2, yc2, xc4, yc4)
    else:
        xc1, yc1, xc2, yc2 = circle_segment(x1, y1, x2, y2)
        xc3, yc3, xc4, yc4 = circle_segment(x2, y2, x3, y3)
        xc5, yc5, xc6, yc6 = circle_segment(x3, y3, x1, y1)
        if xc1 <= 1.:
            xp, yp = 0.5 * (xc1 + xc2), 0.5 * (yc1 + yc2)
            area = overlap_area_triangle_unit_circle(x1, y1, x3, y3, xp, yp) \
                 + overlap_area_triangle_unit_circle(x2, y2, x3, y3, xp, yp)
        elif xc3 <= 1.:
            xp, yp = 0.5 * (xc3 + xc4), 0.5 * (yc3 + yc4)
            area = overlap_area_triangle_unit_circle(x3, y3, x1, y1, xp, yp) \
                 + overlap_area_triangle_unit_circle(x2, y2, x1, y1, xp, yp)
        elif xc5 <= 1.:
            xp, yp = 0.5 * (xc5 + xc6), 0.5 * (yc5 + yc6)
            area = overlap_area_triangle_unit_circle(x1, y1, x2, y2, xp, yp) \
                 + overlap_area_triangle_unit_circle(x3, y3, x2, y2, xp, yp)
        else:  # no intersections
            if in_triangle(0., 0., x1, y1, x2, y2, x3, y3):
                return PI
            else:
                return 0.

    return area