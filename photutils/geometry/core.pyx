# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Core geometry functions for computing the overlap of an aperture and a
pixel grid.

The cdef functions are not intended to be called from Python code. They
are pure C math functions declared ``noexcept nogil`` so they can be
called without the GIL, including from multiple threads on free-threaded
Python builds. The signatures of the functions used by other modules
are exported via core.pxd.
"""

# The math functions used here are declared in core.pxd.
cdef extern from "math.h" nogil:
    double M_PI
    double NAN

ctypedef struct point:
    double x
    double y

ctypedef struct intersections:
    point p1
    point p2


cdef double area_arc_unit(double x1, double y1, double x2,
                          double y2) noexcept nogil:
    """
    Area of a circular segment of the unit circle (radius 1), the
    region between the chord connecting (x1, y1) and (x2, y2) and the
    circular arc spanning the same two points.

    Both points are assumed to lie on the unit circle centered on the
    origin. This is equivalent to ``area_arc`` with ``r = 1``.

    Parameters
    ----------
    x1, y1, x2, y2 : double
        The coordinates of the two points defining the chord.

    Returns
    -------
    area : double
        The area of the circular segment.

    References
    ----------
    http://mathworld.wolfram.com/CircularSegment.html
    """
    cdef double chord

    # The chord length is twice the sine of half the angle that the
    # chord subtends, but recovering the angle from it with an arcsine
    # loses half of the precision when the chord is nearly a diameter.
    # The sine and cosine of the full angle are the cross and dot
    # products of the two points, which atan2 handles accurately for
    # any chord.
    chord = fabs(x1 * y2 - y1 * x2)
    return 0.5 * (atan2(chord, x1 * x2 + y1 * y2) - chord)


cdef int in_triangle(double x, double y, double x1, double y1, double x2,
                     double y2, double x3, double y3) noexcept nogil:
    """
    Test whether a point lies inside a triangle.

    Parameters
    ----------
    x, y : double
        The coordinates of the point to test.

    x1, y1, x2, y2, x3, y3 : double
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
    x1, y1, x2, y2 : double
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

        # Find the (reduced) discriminant of the quadratic equation
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

        # Find the (reduced) discriminant of the quadratic equation
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
    x1, y1, x2, y2 : double
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
    x1, y1, x2, y2 : double
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

    # Order the output so that p1 is invalid (coordinates > 1) whenever
    # fewer than two intersections lie on the segment. Callers pass
    # segments whose endpoints are both outside the circle and test p1
    # alone to decide whether a full chord exists. For such segments a
    # single on-segment intersection is a degenerate grazing case and
    # is treated as no intersection.
    if pt1.x > 1. and pt2.x < 2.:
        inter_new.p1 = pt1
        inter_new.p2 = pt2
    else:
        inter_new.p1 = pt2
        inter_new.p2 = pt1

    return inter_new


cdef point chord_end(double x1, double y1, double x2,
                     double y2) noexcept nogil:
    """
    End point of the chord of the unit circle that starts at a point
    on the circle and is directed toward a second point.

    Because the first point is on the circle, the other intersection
    of the line with the circle has a closed form that needs no square
    root. It therefore stays well defined when the line is tangent to
    the circle, where the general line intersection can lose its
    solutions to floating-point roundoff.

    Parameters
    ----------
    x1, y1 : double
        The coordinates of the point on the unit circle.

    x2, y2 : double
        The coordinates of the point that defines the direction.

    Returns
    -------
    result : point
        The other end of the chord. This is ``(x1, y1)`` itself if the
        direction is tangent to the circle or points away from it.
    """
    cdef double dx, dy, t
    cdef point pt

    dx = x2 - x1
    dy = y2 - y1
    pt.x = x1
    pt.y = y1

    t = -2.0 * (x1 * dx + y1 * dy)
    if t > 0.0:
        t /= dx * dx + dy * dy
        pt.x += t * dx
        pt.y += t * dy

    return pt


cdef double overlap_area_vertex_on_unit_circle(double x1, double y1,
                                               double x2, double y2,
                                               double x3,
                                               double y3) noexcept nogil:
    """
    Area of overlap between a triangle and the unit circle when the
    first vertex lies on the circle and the other two lie outside it.

    The two edges that meet at the first vertex bound a wedge. Each
    edge that points into the circle is a chord, described by the
    angle ``phi`` between the edge and the tangent to the circle at
    the vertex. That chord cuts off a circular segment of area ``phi
    - sin(phi) * cos(phi)``, so the area of the circle inside the
    wedge is the difference of the two segment areas. An edge that
    points away from the circle is replaced by the tangent direction
    that the wedge sweeps across. The part of the circle beyond the
    opposite edge, if any, is then removed.

    Parameters
    ----------
    x1, y1 : double
        The coordinates of the vertex on the unit circle.

    x2, y2, x3, y3 : double
        The coordinates of the two vertices outside the unit circle.

    Returns
    -------
    area : double
        The area of overlap between the triangle and the unit circle.
    """
    cdef double dx2, dy2, dx3, dy3
    cdef double n2, n3, c2, c3, cross
    cdef double f2, f3, area, cap, side0, side1
    cdef intersections inter

    dx2 = x2 - x1
    dy2 = y2 - y1
    dx3 = x3 - x1
    dy3 = y3 - y1

    # Components of each edge along the inward normal of the circle
    n2 = -(x1 * dx2 + y1 * dy2)
    n3 = -(x1 * dx3 + y1 * dy3)
    if n2 <= 0.0 and n3 <= 0.0:
        # Neither edge enters the circle, so the triangle lies on the
        # outer side of the tangent line.
        return 0.0

    # Components of each edge along the tangent of the circle
    c2 = x1 * dy2 - y1 * dx2
    c3 = x1 * dy3 - y1 * dx3

    # Orientation of the wedge, which determines the tangent direction
    # it sweeps across when only one edge enters the circle
    cross = dx3 * dy2 - dy3 * dx2

    if n2 > 0.0:
        f2 = atan2(n2, c2) - n2 * c2 / (dx2 * dx2 + dy2 * dy2)
    elif cross < 0.0:
        f2 = 0.0
    else:
        f2 = M_PI

    if n3 > 0.0:
        f3 = atan2(n3, c3) - n3 * c3 / (dx3 * dx3 + dy3 * dy3)
    elif cross > 0.0:
        f3 = 0.0
    else:
        f3 = M_PI

    area = fabs(f2 - f3)

    # Remove the circular segment beyond the opposite edge
    inter = circle_segment(x2, y2, x3, y3)
    if inter.p1.x <= 1.:
        cap = area_arc_unit(inter.p1.x, inter.p1.y, inter.p2.x, inter.p2.y)
        # The segment is the larger of the two cut by the chord when
        # the opposite edge separates the first vertex from the circle
        # center.
        side0 = (y3 - y2) * x2 - (x3 - x2) * y2
        side1 = (x3 - x2) * (y1 - y2) - (y3 - y2) * (x1 - x2)
        if (side0 > 0.0) != (side1 > 0.0):
            cap = M_PI - cap
        area -= cap

    return area


cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2,
                                              double y2, double x3,
                                              double y3) noexcept nogil:
    """
    Area of overlap between a triangle and the unit circle centered on
    the origin.

    Parameters
    ----------
    x1, y1, x2, y2, x3, y3 : double
        The coordinates of the triangle's three vertices.

    Returns
    -------
    area : double
        The area of overlap between the triangle and the unit circle.
    """
    cdef double d1, d2, d3
    cdef bint in1, in2, in3
    cdef bint on1, on2, on3
    cdef bint cond1, cond2
    cdef double area, xp, yp
    cdef intersections inter
    cdef point pt1, pt2, pt3, pt4, pt5, pt6

    # Find the squared distance of each vertex from the circle center
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
        # This should never happen. Return NaN to signal an internal
        # logic error without requiring the GIL to raise an exception.
        return NAN

    # Determine which vertices are inside the circle
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
        # Vertex 3 is outside the circle. Find the point where each
        # edge to vertex 3 leaves the circle. For a vertex on the
        # circle this is the end of the chord along that edge, which is
        # the vertex itself if the edge does not enter the circle.
        if on1:
            pt1 = chord_end(x1, y1, x3, y3)
        else:
            pt1 = circle_segment_single2(x1, y1, x3, y3)
        if on2:
            pt2 = chord_end(x2, y2, x3, y3)
        else:
            pt2 = circle_segment_single2(x2, y2, x3, y3)
        area = (area_triangle(x1, y1, x2, y2, pt1.x, pt1.y)
                + area_triangle(x2, y2, pt1.x, pt1.y, pt2.x, pt2.y)
                + area_arc_unit(pt1.x, pt1.y, pt2.x, pt2.y))

    elif on1:
        # Vertices 2 and 3 are outside the circle, but the triangle
        # can still cross it.
        area = overlap_area_vertex_on_unit_circle(x1, y1, x2, y2, x3, y3)

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
