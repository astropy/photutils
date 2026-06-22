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
