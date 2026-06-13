# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
"""
Declarations needed to cimport the core geometry functions into other
Cython files. All functions are pure C math functions that are safe to
call without the GIL.
"""

cdef double floor_sqrt(double x) noexcept nogil
cdef double distance(double x1, double y1, double x2,
                     double y2) noexcept nogil
cdef double area_arc(double x1, double y1, double x2, double y2,
                     double r) noexcept nogil
cdef double area_triangle(double x1, double y1, double x2, double y2,
                          double x3, double y3) noexcept nogil
cdef double area_arc_unit(double x1, double y1, double x2,
                          double y2) noexcept nogil
cdef int in_triangle(double x, double y, double x1, double y1, double x2,
                     double y2, double x3, double y3) noexcept nogil
cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2,
                                              double y2, double x3,
                                              double y3) noexcept nogil
