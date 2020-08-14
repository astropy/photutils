# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module contains tools for centroiding sources.
"""

import inspect
import warnings

from astropy.nddata.utils import overlap_slices
from astropy.utils.exceptions import AstropyUserWarning
from ..psf.epsf import _py2intround
import numpy as np

__all__ = ['centroid_com', 'centroid_quadratic', 'centroid_sources',
           'centroid_epsf']


def centroid_com(data, mask=None, oversampling=1):
    """
    Calculate the centroid of an n-dimensional array as its "center of
    mass" determined from moments.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : array_like
        The input n-dimensional array.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    oversampling : int or tuple of two int, optional
        Oversampling factors of pixel indices. If ``oversampling`` is
        a scalar this is treated as both x and y directions having
        the same oversampling factor; otherwise it is treated as
        ``(x_oversamp, y_oversamp)``.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The coordinates of the centroid in pixel order (e.g., ``(x, y)``
        or ``(x, y, z)``), not numpy axis order.
    """
    data = data.astype(float)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asarray(mask, dtype=bool)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = 0.

    oversampling = np.atleast_1d(oversampling)
    if len(oversampling) == 1:
        oversampling = np.repeat(oversampling, 2)
    oversampling = oversampling[::-1]  # reverse to (y, x) order
    if np.any(oversampling <= 0):
        raise ValueError('Oversampling factors must all be positive numbers.')

    badmask = ~np.isfinite(data)
    if np.any(badmask):
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)
        data[badmask] = 0.

    total = np.sum(data)
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total / oversampling[axis]
                     for axis in range(data.ndim)])[::-1]


def centroid_quadratic(image_data, xmax=None, ymax=None, peak_fit_box=5,
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
    if ((xmax is None and ymax is not None)
            or (ymax is None and xmax is not None)):
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
        imax = _py2intround(xmax)
        jmax = _py2intround(ymax)
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
    v = np.vstack((np.ones_like(x), x, y, x * y, x * x, y * y)).T
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
    except np.linalg.LinAlgError:
        if auto_expand_search:
            return centroid_quadratic(image_data, xmax=None, ymax=None,
                                      peak_fit_box=(wx, wy), mask=mask)
        else:
            return coord

    # find maximum of the polynomial:
    _, c10, c01, c11, c20, c02 = c
    d = 4 * c02 * c20 - c11**2
    if d <= 0 or ((c20 > 0.0 and c02 >= 0.0) or (c20 >= 0.0 and c02 > 0.0)):
        # polynomial is does not have max. return middle of the window:
        if auto_expand_search:
            return centroid_quadratic(image_data, xmax=None, ymax=None,
                                      peak_fit_box=(wx, wy), mask=mask)
        else:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    xm = (c01 * c11 - 2.0 * c02 * c10) / d
    ym = (c10 * c11 - 2.0 * c01 * c20) / d

    if xm > 0.0 and xm < (nx - 1.0) and ym > 0.0 and ym < (ny - 1.0):
        coord = (xm, ym)
    elif auto_expand_search:
        coord = centroid_quadratic(image_data, xmax=None, ymax=None,
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


def centroid_sources(data, xpos, ypos, box_size=11, footprint=None,
                     error=None, mask=None, centroid_func=centroid_com):
    """
    Calculate the centroid of sources at the defined positions.

    A cutout image centered on each input position will be used to
    calculate the centroid position.  The cutout image is defined either
    using the ``box_size`` or ``footprint`` keyword.  The ``footprint``
    keyword can be used to create a non-rectangular cutout image.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    xpos, ypos : float or array-like of float
        The initial ``x`` and ``y`` pixel position(s) of the center
        position.  A cutout image centered on this position be used to
        calculate the centroid.

    box_size : int or array-like of int, optional
        The size of the cutout image along each axis. If ``box_size``
        is a number, then a square cutout of ``box_size`` will be
        created. If ``box_size`` has two elements, they should be in
        ``(ny, nx)`` order. Either ``box_size`` or ``footprint`` must be
        defined. If they are both defined, then ``footprint`` overrides
        ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A 2D boolean array where `True` values describe the local
        footprint region to cutout. ``footprint`` can be used to create
        a non-rectangular cutout image, in which case the input ``xpos``
        and ``ypos`` represent the center of the minimal bounding box
        for the input ``footprint``. ``box_size=(n, m)`` is equivalent
        to ``footprint=np.ones((n, m))``. Either ``box_size`` or
        ``footprint`` must be defined. If they are both defined, then
        ``footprint`` overrides ``box_size``.

    mask : array_like, bool, optional
        A 2D boolean array with the same shape as ``data``, where a
        `True` value indicates the corresponding element of ``data`` is
        masked.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
        ``error`` must have the same shape as ``data``.  ``error`` will
        be used only if supported by the input ``centroid_func``.

    centroid_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword and
        optionally an ``error`` keyword. The callable object must return
        a tuple of two 1D `~numpy.ndarray`, representing the x and y
        centroids. The default is `~photutils.centroids.centroid_com`.

    Returns
    -------
    xcentroid, ycentroid : `~numpy.ndarray`
        The ``x`` and ``y`` pixel position(s) of the centroids.
    """
    xpos = np.atleast_1d(xpos)
    ypos = np.atleast_1d(ypos)
    if xpos.ndim != 1:
        raise ValueError('xpos must be a 1D array.')
    if ypos.ndim != 1:
        raise ValueError('ypos must be a 1D array.')

    if footprint is None:
        if box_size is None:
            raise ValueError('box_size or footprint must be defined.')

        box_size = np.atleast_1d(box_size)
        if len(box_size) == 1:
            box_size = np.repeat(box_size, 2)
        if len(box_size) != 2:
            raise ValueError('box_size must have 1 or 2 elements.')

        footprint = np.ones(box_size, dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)
        if footprint.ndim != 2:
            raise ValueError('footprint must be a 2D array.')

    use_error = False
    spec = inspect.getfullargspec(centroid_func)
    if 'mask' not in spec.args:
        raise ValueError('The input "centroid_func" must have a "mask" '
                         'keyword.')
    if 'error' in spec.args:
        use_error = True

    xcentroids = []
    ycentroids = []
    for xp, yp in zip(xpos, ypos):
        slices_large, slices_small = overlap_slices(data.shape,
                                                    footprint.shape, (yp, xp))
        data_cutout = data[slices_large]

        mask_cutout = None
        if mask is not None:
            mask_cutout = mask[slices_large]

        footprint_mask = ~footprint
        # trim footprint mask if it has only partial overlap on the data
        footprint_mask = footprint_mask[slices_small]

        if mask_cutout is None:
            mask_cutout = footprint_mask
        else:
            # combine the input mask and footprint mask
            mask_cutout = np.logical_or(mask_cutout, footprint_mask)

        if error is not None and use_error:
            error_cutout = error[slices_large]
            xcen, ycen = centroid_func(data_cutout, mask=mask_cutout,
                                       error=error_cutout)
        else:
            xcen, ycen = centroid_func(data_cutout, mask=mask_cutout)

        xcentroids.append(xcen + slices_large[1].start)
        ycentroids.append(ycen + slices_large[0].start)

    return np.array(xcentroids), np.array(ycentroids)


def centroid_epsf(data, mask=None, oversampling=4, shift_val=0.5):
    """
    Calculate centering shift of data using pixel symmetry, as described
    by Anderson and King (2000; PASP 112, 1360) in their ePSF-fitting
    algorithm.

    Calculate the shift of a 2-dimensional symmetric image based on the
    asymmetry between f(x, N) and f(x, -N), along with the differential
    df/dy(x, shift_val) and df/dy(x, -shift_val). Non-finite values
    (e.g., NaN or inf) in the ``data`` array are automatically masked.

    Parameters
    ----------
    data : array_like
        The input n-dimensional array.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
    oversampling : int or tuple of two int, optional
        Oversampling factors of pixel indices. If ``oversampling`` is a
        scalar this is treated as both x and y directions having the
        same oversampling factor.  Otherwise it is treated as
        ``(x_oversamp, y_oversamp)``.
    shift_val : float, optional
        The undersampled value at which to compute the shifts. Default
        is half a pixel. It must be a strictly positive number.

    Returns
    -------
    centroid : tuple of floats
        The (x, y) coordinates of the centroid in pixel order.
    """
    data = data.astype(float)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asarray(mask, dtype=bool)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = 0.

    oversampling = np.atleast_1d(oversampling)
    if len(oversampling) == 1:
        oversampling = np.repeat(oversampling, 2)
    if np.any(oversampling <= 0):
        raise ValueError('Oversampling factors must all be positive numbers.')

    if shift_val <= 0:
        raise ValueError('shift_val must be a positive number.')

    # Assume the center of the ePSF is the middle of an odd-sized grid.
    xidx_0 = int((data.shape[1] - 1) / 2)
    x_0 = np.arange(data.shape[1], dtype=float)[xidx_0] / oversampling[0]
    yidx_0 = int((data.shape[0] - 1) / 2)
    y_0 = np.arange(data.shape[0], dtype=float)[yidx_0] / oversampling[1]

    x_shiftidx = np.around((shift_val * oversampling[0])).astype(int)
    y_shiftidx = np.around((shift_val * oversampling[1])).astype(int)

    badmask = ~np.isfinite([data[y, x]
                            for x in [xidx_0, xidx_0 + x_shiftidx,
                                      xidx_0 + x_shiftidx - 1,
                                      xidx_0 + x_shiftidx + 1]
                            for y in [yidx_0, yidx_0 + y_shiftidx,
                                      yidx_0 + y_shiftidx - 1,
                                      yidx_0 + y_shiftidx + 1]])

    if np.any(badmask):
        raise ValueError('One or more centroiding pixels is set to a '
                         'non-finite value, e.g., NaN or inf.')

    # In Anderson & King (2000) notation this is psi_E(0.5, 0.0) and
    # values used to compute derivatives.
    psi_pos_x = data[yidx_0, xidx_0 + x_shiftidx]
    psi_pos_x_m1 = data[yidx_0, xidx_0 + x_shiftidx - 1]
    psi_pos_x_p1 = data[yidx_0, xidx_0 + x_shiftidx + 1]

    # Our derivatives are simple differences across two data points, but
    # this must be in units of the undersampled grid, so 2 pixels becomes
    # 2/oversampling pixels
    dpsi_pos_x = np.abs(psi_pos_x_p1 - psi_pos_x_m1) / (2. / oversampling[0])

    # psi_E(-0.5, 0.0) and derivative components.
    psi_neg_x = data[yidx_0, xidx_0 - x_shiftidx]
    psi_neg_x_m1 = data[yidx_0, xidx_0 - x_shiftidx - 1]
    psi_neg_x_p1 = data[yidx_0, xidx_0 - x_shiftidx + 1]
    dpsi_neg_x = np.abs(psi_neg_x_p1 - psi_neg_x_m1) / (2. / oversampling[0])

    x_shift = (psi_pos_x - psi_neg_x) / (dpsi_pos_x + dpsi_neg_x)

    # psi_E(0.0, 0.5) and derivatives.
    psi_pos_y = data[yidx_0 + y_shiftidx, xidx_0]
    psi_pos_y_m1 = data[yidx_0 + y_shiftidx - 1, xidx_0]
    psi_pos_y_p1 = data[yidx_0 + y_shiftidx + 1, xidx_0]
    dpsi_pos_y = np.abs(psi_pos_y_p1 - psi_pos_y_m1) / (2. / oversampling[1])

    # psi_E(0.0, -0.5) and derivative components.
    psi_neg_y = data[yidx_0 - y_shiftidx, xidx_0]
    psi_neg_y_m1 = data[yidx_0 - y_shiftidx - 1, xidx_0]
    psi_neg_y_p1 = data[yidx_0 - y_shiftidx + 1, xidx_0]
    dpsi_neg_y = np.abs(psi_neg_y_p1 - psi_neg_y_m1) / (2. / oversampling[1])

    y_shift = (psi_pos_y - psi_neg_y) / (dpsi_pos_y + dpsi_neg_y)

    return x_0 + x_shift, y_0 + y_shift
