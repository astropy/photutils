# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import division
import numpy as np
cimport numpy as np

cdef extern from "math.h":

    double sqrt(double x)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def circular_overlap_grid(double xmin, double xmax, int nx,
                          double ymin, double ymax, int ny,
                          double R, int subpixels):
    '''Find area of overlap in each element on grid'''

    cdef unsigned int i, j
    cdef double x, y, dx, dy, d, pixrad, xlim0, xlim1, ylim0, ylim1

    # Output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)

    # Width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Define these here to speed computation below
    pixrad = 0.5 * sqrt(dx ** 2 + dy ** 2)  # Radius of a single pixel
    xlim0 = -R - 0.5 * dx                   # Extent of circle + half pixel
    xlim1 = R + 0.5 * dx                    # ...
    ylim0 = -R - 0.5 * dy                   # ...
    ylim1 = R + 0.5 * dy                    # ...

    x = xmin - 0.5 * dx  # x coordinate of pixel center
    for i in range(nx):
        x += dx
        if x > xlim0 and x < xlim1:
            y = ymin - 0.5 * dy  # y coordinate of pixel center
            for j in range(ny):
                y += dy
                if y > ylim0 and y < ylim1:
                    d = sqrt(x ** 2 + y ** 2)
                    if d < R - pixrad:
                        frac[j, i] = 1.
                    elif d < R + pixrad:
                        frac[j, i] = circular_overlap_single(x - 0.5 * dx, \
                            x + 0.5 * dx, y - 0.5 * dy, y + 0.5 * dy, \
                            R, subpixels)
    return frac
		        
def circular_overlap_single(double x0, double x1, double y0, double y1,
                            double R, int subpixels):
    '''Return the fraction of overlap between a circle and a single pixel.'''

    cdef unsigned int i, j
    cdef double x, y, dx, dy
    cdef double frac = 0.  # Accumulator.

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy
            if x ** 2 + y ** 2 < R ** 2:
                frac += 1.

    return frac / (subpixels * subpixels)		            
