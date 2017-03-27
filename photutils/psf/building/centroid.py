# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Centroid utilities.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np

from .utils import py2round

__all__ = ['find_peak', 'local_centroid']

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


def find_peak(image_data, xmax=None, ymax=None, peak_fit_box=5,
              peak_search_box=None, mask=None):
    """
    Find location of the peak in an array. This is done by fitting a second
    degree 2D polynomial to the data within a `peak_fit_box` and computing the
    location of its maximum. When `xmax` and `ymax` are both `None`, an initial
    estimate of the position of the maximum will be performed by searching
    for the location of the pixel/array element with the maximum value. This
    kind of initial brute-force search can be performed even when
    `xmax` and `ymax` are provided but when one suspects that these input
    coordinates may not be very accurate by specifying an expanded
    brute-force search box through parameter `peak_search_box`.

    Parameters
    ----------
    image_data : numpy.ndarray
        Image data.

    xmax : float, None, optional
        Initial guess of the x-coordinate of the peak. When both `xmax` and
        `ymax` are `None`, the initial (pre-fit) estimate of the location
        of the peak will be obtained by a brute-force search for the location
        of the maximum-value pixel in the *entire* `image_data` array,
        regardless of the value of ``peak_search_box`` parameter.

    ymax : float, None, optional
        Initial guess of the x-coordinate of the peak. When both `xmax` and
        `ymax` are `None`, the initial (pre-fit) estimate of the location
        of the peak will be obtained by a brute-force search for the location
        of the maximum-value pixel in the *entire* `image_data` array,
        regardless of the value of ``peak_search_box`` parameter.

    peak_fit_box : int, tuple of int, optional
        Size (in pixels) of the box around the input estimate of the maximum
        (given by ``xmax`` and ``ymax``) to be used for quadratic fitting from
        which peak location is computed. If a single integer
        number is provided, then it is assumed that fitting box is a square
        with sides of length given by ``peak_fit_box``. If a tuple of two
        values is provided, then first value indicates the width of the box and
        the second value indicates the height of the box.

    peak_search_box : str {'all', 'off', 'fitbox'}, int, tuple of int, None,\
optional
        Size (in pixels) of the box around the input estimate of the maximum
        (given by ``xmax`` and ``ymax``) to be used for brute-force search of
        the maximum value pixel. This search is performed before quadratic
        fitting in order to improve the original estimate of the peak
        given by input ``xmax`` and ``ymax``. If a single integer
        number is provided, then it is assumed that search box is a square
        with sides of length given by ``peak_fit_box``. If a tuple of two
        values is provided, then first value indicates the width of the box
        and the second value indicates the height of the box. ``'off'`` or
        `None` turns off brute-force search of the maximum. When
        ``peak_search_box`` is ``'all'`` then the entire ``image_data``
        data array is searched for maximum and when it is set to ``'fitbox'``
        then the brute-force search is performed in the same box as
        ``peak_fit_box``.

        .. note::
            This parameter is ignored when both `xmax` and `ymax` are `None`
            since in that case the brute-force search for the maximum is
            performed in the entire input array.

    mask : numpy.ndarray, optional
        A boolean type `~numpy.ndarray` indicating "good" pixels in image data
        (`True`) and "bad" pixels (`False`). If not provided all pixels
        in `image_data` will be used for fitting.

    Returns
    -------
    coord : tuple of float
        A pair of coordinates of the peak.

    """
    # check arguments:
    if ((xmax is None and ymax is not None) or (ymax is None and
                                                xmax is not None)):
        raise ValueError("Both 'xmax' and 'ymax' must be either None or not "
                         "None")

    image_data = np.asarray(image_data, dtype=np.float64)
    ny, nx = image_data.shape

    # process peak search box:
    if peak_search_box == 'fitbox':
        peak_search_box = peak_fit_box

    elif peak_search_box == 'off':
        peak_search_box = None

    elif peak_search_box == 'all':
        peak_search_box = image_data.shape

    if xmax is None:
        # find index of the pixel having maximum value:
        if mask is None:
            jmax, imax = np.unravel_index(np.argmax(image_data),
                                          image_data.shape)
            coord = (float(imax), float(jmax))

        else:
            j, i = np.indices(image_data.shape)
            i = i[mask]
            j = j[mask]
            ind = np.argmax(image_data[mask])
            imax = i[ind]
            jmax = j[ind]
            coord = (float(imax), float(jmax))

        auto_expand_search = False  # we have already searched the data

    else:
        imax = int(py2round(xmax))
        jmax = int(py2round(ymax))
        coord = (xmax, ymax)

        if peak_search_box is not None:
            sbx, sby = _process_box_pars(peak_search_box)

            # choose a box around maxval pixel:
            x1 = max(0, imax - sbx // 2)
            x2 = min(nx, x1 + sbx)
            y1 = max(0, jmax - sby // 2)
            y2 = min(ny, y1 + sby)

            if x1 < x2 and y1 < y2:
                search_cutout = image_data[y1:y2, x1:x2]
                jmax, imax = np.unravel_index(
                    np.argmax(search_cutout),
                    search_cutout.shape
                )
                imax += x1
                jmax += y1
                coord = (float(imax), float(jmax))

        auto_expand_search = (sbx != nx or sby != ny)

    wx, wy = _process_box_pars(peak_fit_box)

    if wx * wy < 6:
        # we need at least 6 points to fit a 2D quadratic polynomial
        return coord

    # choose a box around maxval pixel:
    x1 = max(0, imax - wx // 2)
    x2 = min(nx, x1 + wx)
    y1 = max(0, jmax - wy // 2)
    y2 = min(ny, y1 + wy)

    # if peak is at the edge of the box, return integer indices of the max:
    if imax == x1 or imax == x2 or jmax == y1 or jmax == y2:
        return (float(imax), float(jmax))

    # expand the box if needed:
    if (x2 - x1) < wx:
        if x1 == 0:
            x2 = min(nx, x1 + wx)
        if x2 == nx:
            x1 = max(0, x2 - wx)
    if (y2 - y1) < wy:
        if y1 == 0:
            y2 = min(ny, y1 + wy)
        if y2 == ny:
            y1 = max(0, y2 - wy)

    if (x2 - x1) * (y2 - y1) < 6:
        # we need at least 6 points to fit a 2D quadratic polynomial
        return coord

    # fit a 2D 2nd degree polynomial to data:
    xi = np.arange(x1, x2)
    yi = np.arange(y1, y2)
    x, y = np.meshgrid(xi, yi)
    x = x.ravel()
    y = y.ravel()
    v = np.vstack((np.ones_like(x), x, y, x*y, x*x, y*y)).T
    d = image_data[y1:y2, x1:x2].ravel()
    if mask is not None:
        m = mask[y1:y2, x1:x2].ravel()
        v = v[m]
        d = d[m]
        if d.size < 6:
            # we need at least 6 points to fit a 2D quadratic polynomial
            return coord
    try:
        c = np.linalg.lstsq(v, d)[0]
    except LinAlgError:
        if auto_expand_search:
            return find_peak(image_data, xmax=None, ymax=None,
                             peak_fit_box=(wx, wy), mask=mask)
        else:
            return coord

    # find maximum of the polynomial:
    _, c10, c01, c11, c20, c02 = c
    d = 4 * c02 * c20 - c11**2
    if d <= 0 or ((c20 > 0.0 and c02 >= 0.0) or (c20 >= 0.0 and c02 > 0.0)):
        # polynomial is does not have max. return middle of the window:
        if auto_expand_search:
            return find_peak(image_data, xmax=None, ymax=None,
                             peak_fit_box=(wx, wy), mask=mask)
        else:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    xm = (c01 * c11 - 2.0 * c02 * c10) / d
    ym = (c10 * c11 - 2.0 * c01 * c20) / d

    if xm > 0.0 and xm < (nx - 1.0) and ym > 0.0 and ym < (ny - 1.0):
        coord = (xm, ym)
    elif auto_expand_search:
        coord = find_peak(image_data, xmax=None, ymax=None,
                          peak_fit_box=(wx, wy), mask=mask)

    return coord


def _process_box_pars(par):
    if hasattr(par, '__iter__'):
        if len(par) != 2:
            raise TypeError("Box specification must be either a scalar or "
                            "an iterable with two elements.")
        wx = int(par[0])
        wy = int(par[1])
    else:
        wx = int(par)
        wy = int(par)

    if wx < 1 or wy < 1:
        raise ValueError("Box dimensions must be positive integer numbers.")

    return (wx, wy)


def local_centroid(data, mode='jay', max_position=None):
    """
    Calculate the local centroid of a 2D array using the 1D marginal x
    and y centroids of the central three pixels.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mode : {'jay', '1d', '1dsub'}, optional
        The centroid algorithm to use:

            * ``'jay'``: Jay Anderson's algorithm (default).
            * ``'1d'``: Standard 1D moment.
            * ``'1dsubmin'``: Standard 1D moment after subtraction of
              the minimum pixel value.

    max_position : 2-tuple of float, optional
        The (y, x) position of the maximum pixel in the 2D array.  If
        `None`, then it will be calculated.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    if max_position is None:
        ymax, xmax = np.unravel_index(np.argmax(data), data.shape)
    else:
        ymax, xmax = max_position

    xdata = data[ymax, xmax-1:xmax+2]
    ydata = data[ymax-1:ymax+2, xmax]
    if len(xdata) != 3 or len(ydata) != 3:
        raise ValueError('max_position cannot be on the data edge')

    if mode == 'jay':
        xc = _jay_local_centroid1d(xdata)
        yc = _jay_local_centroid1d(ydata)
    elif mode == '1d':
        xc = _1d_local_centroid1d(xdata, submin=False)
        yc = _1d_local_centroid1d(ydata, submin=False)
    elif mode == '1dsubmin':
        xc = _1d_local_centroid1d(xdata, submin=True)
        yc = _1d_local_centroid1d(ydata, submin=True)

    return (xc + xmax, yc + ymax)


def _jay_local_centroid1d(data):
    data = np.asanyarray(data)
    numer = (data[2] - data[0]) / 2.
    denom = data[1] - np.min([data[0], data[2]])
    if denom == 0:
        return 0.    # the center pixel
    else:
        return (numer / denom)


def _1d_local_centroid1d(data, submin=False):
    data = np.asanyarray(data)
    if submin:
        data = data - data.min()
    return (data[2] - data[0]) / data.sum()
