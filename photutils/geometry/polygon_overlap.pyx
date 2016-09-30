# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
cimport numpy as np

__all__ = ['polygon_overlap_grid']

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

def polygon_overlap_grid(double xmin, double xmax, double ymin,
                            double ymax, int nx, int ny, vertices,
                            int use_exact, int subpixels):
    """
    polygon_overlap_grid(xmin, xmax, ymin, ymax, nx, ny,
                            vertices, use_exact, subpixels)

    Area of overlap between a polygon and a pixel grid.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Extent of the grid in the x and y direction
    nx, ny : int
        Grid dimensions
    vertices : list of tuples of length 2
        Vertex co-ordinates for the polygon
    theta : float
        The position angle of the side represented by ``a``
        in radians (counterclockwise).
    use_exact : 0 or 1
        If set to 1, calculates the exact overlap, while if set to 0, uses a
        subpixel sampling method with ``subpixel`` subpixels in each direction
    subpixels : int
        If ``use_exact`` is 0, each pixel is resmapled by this factor in each dimension. Thus, each pixel is divided into ``subpixels ** 2`` subpixels.

    Returns
    -------
    frac : `~numpy.ndarray`
        2-d array giving the fraction of the overlap
    """

    cdef unsigned int i, j 
    cdef double x, y, dx, dy
    cdef double pxmin, pxmax, pymin, pymax

    # Define output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny,nx],dtype=DTYPE)

    if use_exact == 1:
        raise NotImplementedError("Exact mode has not been implemented for "
                                  "rectangular apertures")

    # Find the width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    for i in range(nx):
        pxmin = xmin + i * dx  # lower end of pixel
        pxmax = pxmin + dx  # upper end of pixel
        for j in range(ny):
            pymin = ymin + j * dy
            pymax = pymin + dy
            frac[j, i] = polygon_overlap_single_subpixel(
                pxmin, pymin, pxmax, pymax, vertices, subpixels)

    return frac

cdef double polygon_overlap_single_subpixel(double xi, double yi,
                                       double xf, double yf, vertices,
                                       int subpixels):

    """
    Return the fraction of overlap between a polygon and a single pixel with
    given extent, using a sub-pixel sampling method.
    """

    cdef unsigned int i, j
    cdef double x, y
    cdef double a, b, k
    cdef double frac = 0.  # Accumulator.

    dx = (xf - xi) / subpixels
    dy = (yf - yi) / subpixels

    x = xi - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = yi - 0.5 * dy
        for j in range(subpixels):
            y += dy
            
            if is_inside_polygon(x, y, vertices):
                frac += 1
            
    return frac / (subpixels * subpixels)

cdef int is_inside_polygon(double x, double y,
                    vertices):

    """
    Returns 1 if point is interior of the polygon, 0 otherwise.
    """
    cdef unsigned int i, j, c = 0
    cdef n = len(vertices)
    cdef double xoffset

    i, j = 0, n-1
    while i < n:
        if vertices[j][1]==vertices[i][1]:
            xoffset = vertices[i][0]
        else:
            xoffset = (vertices[j][0]-vertices[i][0])*(y-vertices[i][1])/(vertices[j][1]-vertices[i][1]) + vertices[i][0]
        if (((vertices[i][1] < y and y < vertices[j][1]) or
            (vertices[j][1] < y and y < vertices[i][1])) and
            (x <= xoffset)):

            c = 1-c

        j, i = i, i+1

    return c
