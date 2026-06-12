# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3

# This file is needed in order to be able to cimport functions into
# other Cython files. This single-pixel function is a pure C math
# function that is safe to call without the GIL.

cdef double rectangular_overlap_single_subpixel(double x0, double y0,
                                                double x1, double y1,
                                                double half_width,
                                                double half_height,
                                                double cos_theta,
                                                double sin_theta,
                                                int subpixels) noexcept nogil
