# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to calculate the area of overlap between an ellipse and a pixel
grid.

The method divides the pixel rectangle into two triangles and reprojects
them into a coordinate system where the ellipse becomes the unit circle.
It then computes the intersection area between each triangle and the
unit circle.

The cdef functions are not intended to be called from Python code.
They are pure C math functions declared ``noexcept nogil`` so they can
be called without the GIL (e.g., from the batch aperture photometry
driver), including from multiple threads on free-threaded Python builds.
Their signatures are exported via ellipse_overlap.pxd.

NOTE: The ``elliptical_overlap_grid`` function should be named
``ellipse_overlap_grid``, but it has been public for a long time and
changing the name would break backwards compatibility.
"""

import numpy as np

cimport numpy as np

from .core cimport overlap_area_triangle_unit_circle

__all__ = ['elliptical_overlap_grid']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double fmin(double x, double y)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def elliptical_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                            int nx, int ny, double rx, double ry, double theta,
                            int use_exact, int subpixels):
    """
    ellipse_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, rx, ry,
                         theta, use_exact, subpixels)

    Calculate the fractional overlap between an ellipse and a pixel
    grid.

    The ellipse is centered on the origin.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        The extent of the grid in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax).

    nx, ny : int
        The grid dimensions in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax) and is divided into nx and ny pixels in the x and y
        direction, respectively.

    rx : float
        The semimajor axis of the ellipse.

    ry : float
        The semiminor axis of the ellipse.

    theta : float
        The position angle of the semimajor axis in radians
        (counterclockwise).

    use_exact : 0 or 1
        Set to ``1`` to use an exact method to calculate the overlap
        between the ellipse and each pixel. Set to ``0`` to use a
        sub-pixel sampling method to calculate the overlap, where each
        pixel is divided into ``subpixels ** 2`` subpixels and the
        fraction of subpixels that are within the ellipse is used to
        estimate the overlap.

    subpixels : int
        The number of subpixels to use in each dimension when using
        the sub-pixel sampling method. Each pixel is resampled by this
        factor in each dimension; thus, each pixel is divided into
        ``subpixels ** 2`` subpixels.

        A subpixel is included only if its center lies strictly inside
        the ellipse; subpixel centers lying exactly on the ellipse
        boundary are excluded (weight 0).

    Returns
    -------
    result : `~numpy.ndarray` (float)
        A 2D array of shape (ny, nx) giving the fraction of each
        pixel's area that overlaps with the ellipse, ranging from 0 to
        1. The element at index (j, i) corresponds to the pixel with
        corners at (xmin + i * dx, ymin + j * dy) and (xmin + (i + 1)
        * dx, ymin + (j + 1) * dy), where dx and dy are the width of
        each pixel in the x and y direction, respectively.
    """
    cdef unsigned int i, j
    cdef double dx, dy
    cdef double bx, by
    cdef double bxmin, bxmax, bymin, bymax
    cdef double pxmin, pxmax, pymin, pymax
    cdef double pxcen, pycen, rpix2
    cdef double cos_theta, sin_theta, inv_rx2, inv_ry2
    cdef double cxx, cyy, cxy, margin, f_in, f_out
    cdef double norm

    # Define output array
    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac

    # Find the width of each element in x and y
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    norm = 1.0 / (dx * dy)

    # Quadratic-form coefficients of the ellipse, such that a point
    # (x, y) lies inside the ellipse when
    # ``cxx * x**2 + cyy * y**2 + cxy * x * y < 1``.
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)
    cxx = cos_theta * cos_theta * inv_rx2 + sin_theta * sin_theta * inv_ry2
    cyy = sin_theta * sin_theta * inv_rx2 + cos_theta * cos_theta * inv_ry2
    cxy = 2.0 * cos_theta * sin_theta * (inv_rx2 - inv_ry2)

    # Boundary band for the interior/exterior fast path. A pixel is
    # fully inside (or outside) the ellipse when its center is more than
    # one pixel half-diagonal (measured in the ellipse metric) inside
    # (or outside) the boundary. The metric gradient is bounded by
    # ``1 / min(rx, ry)``, so this margin is conservative.
    margin = 0.5 * sqrt(dx * dx + dy * dy) / fmin(rx, ry)
    f_in = 1.0 - margin
    f_in = f_in * f_in if f_in > 0.0 else 0.0
    f_out = (1.0 + margin) * (1.0 + margin)

    # Tight axis-aligned bounding box of the rotated ellipse. The
    # half-extents are the maxima of |x| and |y| over the ellipse.
    bx = sqrt(rx * rx * cos_theta * cos_theta
              + ry * ry * sin_theta * sin_theta)
    by = sqrt(rx * rx * sin_theta * sin_theta
              + ry * ry * cos_theta * cos_theta)

    # Define bounding box
    bxmin = -bx - 0.5 * dx
    bxmax = +bx + 0.5 * dx
    bymin = -by - 0.5 * dy
    bymax = +by + 0.5 * dy

    with nogil:
        for i in range(nx):
            pxmin = xmin + i * dx  # lower end of pixel
            pxmax = pxmin + dx  # upper end of pixel
            pxcen = pxmin + 0.5 * dx
            if pxmax > bxmin and pxmin < bxmax:
                for j in range(ny):
                    pymin = ymin + j * dy
                    pymax = pymin + dy
                    if pymax > bymin and pymin < bymax:
                        pycen = pymin + 0.5 * dy
                        rpix2 = (cxx * pxcen * pxcen
                                 + cyy * pycen * pycen
                                 + cxy * pxcen * pycen)
                        if rpix2 >= f_out:
                            continue  # pixel fully outside the ellipse
                        if rpix2 <= f_in:
                            frac_view[j, i] = 1.0  # fully inside
                        elif use_exact:
                            frac_view[j, i] = (
                                ellipse_overlap_single_exact(
                                    pxmin, pymin, pxmax, pymax, rx, ry,
                                    cos_theta, sin_theta) * norm)
                        else:
                            frac_view[j, i] = (
                                ellipse_overlap_single_subpixel(
                                    pxmin, pymin, pxmax, pymax, rx, ry,
                                    cos_theta, sin_theta, subpixels))
    return frac


cdef double ellipse_overlap_single_subpixel(double x0, double y0,
                                            double x1, double y1,
                                            double rx, double ry,
                                            double cos_theta,
                                            double sin_theta,
                                            int subpixels) noexcept nogil:
    """
    Return the fraction of overlap between an ellipse and a single
    pixel with given extent, using a sub-pixel sampling method.

    ``cos_theta`` and ``sin_theta`` are the cosine and sine of the
    ellipse position angle, precomputed by the caller.
    """
    cdef unsigned int i, j
    cdef double x, y
    cdef double frac = 0.0  # Accumulator.
    cdef double inv_rx_sq, inv_ry_sq
    cdef double dx, dy
    cdef double x_tr, y_tr

    dx = (x1 - x0) / subpixels
    dy = (y1 - y0) / subpixels

    inv_rx_sq = 1.0 / (rx * rx)
    inv_ry_sq = 1.0 / (ry * ry)

    x = x0 - 0.5 * dx
    for i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for j in range(subpixels):
            y += dy

            # Transform into frame of rotated ellipse
            x_tr = y * sin_theta + x * cos_theta
            y_tr = y * cos_theta - x * sin_theta

            if x_tr * x_tr * inv_rx_sq + y_tr * y_tr * inv_ry_sq < 1.:
                frac += 1.0

    return frac / (subpixels * subpixels)


cdef double ellipse_overlap_single_exact(double xmin, double ymin,
                                         double xmax, double ymax,
                                         double rx, double ry,
                                         double cos_theta,
                                         double sin_theta) noexcept nogil:
    """
    Return the area of overlap between a rectangle and an ellipse.

    The rectangle is defined by (``xmin``, ``ymin``, ``xmax``, ``ymax``)
    and the ellipse has major and minor axes ``rx`` and ``ry``,
    respectively, position angle ``theta``, and is centered at the
    origin. The method divides the pixel rectangle into two triangles
    and reprojects them into a coordinate system where the ellipse
    becomes the unit circle. It then computes the intersection area
    between each triangle and the unit circle, and sums them to get the
    total area of overlap.

    ``cos_theta`` and ``sin_theta`` are the cosine and sine of the
    ellipse position angle, precomputed by the caller.
    """
    # The reprojection uses the -theta rotation, for which
    # cos(-theta) = cos(theta) and sin(-theta) = -sin(theta).
    cdef double cos_m_theta = cos_theta
    cdef double sin_m_theta = -sin_theta
    cdef double scale
    cdef double x1, y1, x2, y2, x3, y3, x4, y4

    # Find scale by which the areas will be shrunk
    scale = rx * ry

    # Reproject rectangle to frame of reference in which ellipse is a
    # unit circle
    x1, y1 = ((xmin * cos_m_theta - ymin * sin_m_theta) / rx,
              (xmin * sin_m_theta + ymin * cos_m_theta) / ry)
    x2, y2 = ((xmax * cos_m_theta - ymin * sin_m_theta) / rx,
              (xmax * sin_m_theta + ymin * cos_m_theta) / ry)
    x3, y3 = ((xmax * cos_m_theta - ymax * sin_m_theta) / rx,
              (xmax * sin_m_theta + ymax * cos_m_theta) / ry)
    x4, y4 = ((xmin * cos_m_theta - ymax * sin_m_theta) / rx,
              (xmin * sin_m_theta + ymax * cos_m_theta) / ry)

    # Divide resulting quadrilateral into two triangles and find
    # intersection with unit circle
    return (overlap_area_triangle_unit_circle(x1, y1, x2, y2, x3, y3) +
            overlap_area_triangle_unit_circle(x1, y1, x4, y4, x3, y3)) * scale
