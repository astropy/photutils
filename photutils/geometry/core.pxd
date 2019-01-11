# Licensed under a 3-clause BSD style license - see LICENSE.rst
#cython: language_level=3

# This file is needed in order to be able to cimport functions into other Cython files

cdef double distance(double x1, double y1, double x2, double y2)
cdef double area_arc(double x1, double y1, double x2, double y2, double R)
cdef double area_triangle(double x1, double y1, double x2, double y2, double x3, double y3)
cdef double area_arc_unit(double x1, double y1, double x2, double y2)
cdef int in_triangle(double x, double y, double x1, double y1, double x2, double y2, double x3, double y3)
cdef double overlap_area_triangle_unit_circle(double x1, double y1, double x2, double y2, double x3, double y3)
cdef double floor_sqrt(double x)
