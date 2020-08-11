# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module contains tools for centroiding sources.
"""

import inspect
import warnings

from astropy.nddata.utils import overlap_slices
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

__all__ = ['centroid_com', 'centroid_sources', 'centroid_epsf']


def centroid_com(data, mask=None, oversampling=1.):
    """
    Calculate the centroid of an n-dimensional array as its "center of
    mass" determined from moments.

    Invalid values (e.g. NaNs or infs) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : array_like
        The input n-dimensional array.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    oversampling : float or tuple of two floats, optional
        Oversampling factors of pixel indices. If ``oversampling`` is
        a scalar this is treated as both x and y directions having
        the same oversampling factor; otherwise it is treated as
        ``(x_oversamp, y_oversamp)``.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The coordinates of the centroid in pixel order (e.g. ``(x, y)``
        or ``(x, y, z)``), not numpy axis order.
    """
    oversampling = np.atleast_1d(oversampling)
    if len(oversampling) == 1:
        oversampling = np.repeat(oversampling, 2)
    # as these need to match data we reverse the order to (y, x)
    oversampling = oversampling[::-1]
    if np.any(oversampling <= 0):
        raise ValueError('Oversampling factors must all be positive numbers.')
    data = data.astype(float)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asarray(mask, dtype=bool)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = 0.

    badidx = ~np.isfinite(data)
    if np.any(badidx):
        warnings.warn('Input data contains input values (e.g. NaNs or infs), '
                      'which were automatically masked.', AstropyUserWarning)
        data[badidx] = 0.

    total = np.sum(data)
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total / oversampling[axis]
                     for axis in range(data.ndim)])[::-1]


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
        The size of the cutout image along each axis.  If ``box_size``
        is a number, then a square cutout of ``box_size`` will be
        created.  If ``box_size`` has two elements, they should be in
        ``(ny, nx)`` order.

    footprint : `~numpy.ndarray` of bools, optional
        A 2D boolean array where `True` values describe the local
        footprint region to cutout.  ``footprint`` can be used to create
        a non-rectangular cutout image, in which case the input ``xpos``
        and ``ypos`` represent the center of the minimal bounding box
        for the input ``footprint``.  ``box_size=(n, m)`` is equivalent
        to ``footprint=np.ones((n, m))``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
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
        A callable object (e.g. function or class) that is used to
        calculate the centroid of a 2D array.  The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword and
        optionally an ``error`` keyword.  The callable object must
        return a tuple of two 1D `~numpy.ndarray`\\s, representing the x
        and y centroids.  The default is
        `~photutils.centroids.centroid_com`.

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
        # trim footprint mask if partial overlap on the data
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
    Calculates centering shift of data using pixel symmetry, as
    described by Anderson and King (2000; PASP 112, 1360) in their
    ePSF-fitting algorithm.

    Calculate the shift of a 2-dimensional symmetric image based on the
    asymmetry between f(x, N) and f(x, -N), along with the differential
    df/dy(x, shift_val) and df/dy(x, -shift_val). Invalid values (e.g.
    NaNs or infs) in the ``data`` array are automatically masked.

    Parameters
    ----------
    data : array_like
        The input n-dimensional array.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
    oversampling : float or tuple of two floats, optional
        Oversampling factors of pixel indices. If ``oversampling`` is a
        scalar this is treated as both x and y directions having the
        same oversampling factor.  Otherwise it is treated as
        ``(x_oversamp, y_oversamp)``.
    shift_val : float, optional
        The undersampled value at which to compute the shifts. Default
        is half a pixel. If supplied, must be a strictly positive
        number.

    Returns
    -------
    centroid : tuple of floats
        The (x, y) coordinates of the centroid in pixel order.
    """
    oversampling = np.atleast_1d(oversampling)
    if len(oversampling) == 1:
        oversampling = np.repeat(oversampling, 2)
    if np.any(oversampling <= 0):
        raise ValueError('Oversampling factors must all be positive numbers.')

    data = data.astype(float)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asarray(mask, dtype=bool)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = 0.

    if shift_val <= 0:
        raise ValueError('shift_val must be a positive number.')

    # Assume the center of the ePSF is the middle of an odd-sized grid.
    xidx_0 = int((data.shape[1] - 1) / 2)
    x_0 = np.arange(data.shape[1], dtype=float)[xidx_0] / oversampling[0]
    yidx_0 = int((data.shape[0] - 1) / 2)
    y_0 = np.arange(data.shape[0], dtype=float)[yidx_0] / oversampling[1]

    x_shiftidx = np.around((shift_val * oversampling[0])).astype(int)
    y_shiftidx = np.around((shift_val * oversampling[1])).astype(int)

    badidx = ~np.isfinite([data[y, x]
                           for x in [xidx_0, xidx_0+x_shiftidx,
                                     xidx_0+x_shiftidx-1, xidx_0+x_shiftidx+1]
                           for y in [yidx_0, yidx_0+y_shiftidx,
                                     yidx_0+y_shiftidx-1,
                                     yidx_0+y_shiftidx+1]])
    if np.any(badidx):
        raise ValueError('One or more centroiding pixels is set to a bad '
                         'value, e.g., NaN or inf.')

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
