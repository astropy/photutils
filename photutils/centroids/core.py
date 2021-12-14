# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The module contains tools for centroiding sources.
"""

import inspect
import warnings

from astropy.nddata.utils import overlap_slices
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from ..utils._round import _py2intround

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


def centroid_quadratic(data, xpeak=None, ypeak=None, fit_boxsize=5,
                       search_boxsize=None, mask=None):
    """
    Calculate the centroid of an n-dimensional array by fitting a 2D
    quadratic polynomial.

    A second degree 2D polynomial is fit within a small region of the
    data defined by ``fit_boxsize`` to calculate the centroid position.
    The initial center of the fitting box can specified using the
    ``xpeak`` and ``ypeak`` keywords. If both ``xpeak`` and ``ypeak``
    are `None`, then the box will be centered at the position of the
    maximum value in the input ``data``.

    If ``xpeak`` and ``ypeak`` are specified, the ``search_boxsize``
    optional keyword can be used to further refine the initial center of
    the fitting box by searching for the position of the maximum pixel
    within a box of size ``search_boxsize``.

    `Vakili & Hogg (2016) <https://arxiv.org/abs/1610.05873>`_
    demonstrate that 2D quadratic centroiding comes very
    close to saturating the `Cramér-Rao lower bound
    <https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound>`_ in a
    wide range of conditions.

    Parameters
    ----------
    data : numpy.ndarray
        Image data.

    xpeak, ypeak : float or `None`, optional
        The initial guess of the position of the centroid. If either
        ``xpeak`` or ``ypeak`` is `None` then the position of the
        maximum value in the input ``data`` will be used as the initial
        guess.

    fit_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to define the fitting
        region. If ``fit_boxsize`` has two elements, they should be in
        ``(ny, nx)`` order. If ``fit_boxsize`` is a scalar then a square
        box of size ``fit_boxsize`` will be used.

    search_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to search for the maximum
        pixel value if ``xpeak`` and ``ypeak`` are both specified. If
        ``fit_boxsize`` has two elements, they should be in ``(ny,
        nx)`` order. If ``fit_boxsize`` is a scalar then a square box
        of size ``fit_boxsize`` will be used. This parameter is ignored
        if either ``xpeak`` or ``ypeak`` is `None`. In that case, the
        entire array is search for the maximum value.

    mask : bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.

    Notes
    -----
    Use ``fit_boxsize = (3, 3)`` to match the work of `Vakili &
    Hogg (2016) <https://arxiv.org/abs/1610.05873>`_ for their 2D
    second-order polynomial centroiding method.

    References
    ----------
    .. [1] Vakili and Hogg 2016; arXiv:1610.05873
        (https://arxiv.org/abs/1610.05873)
    """
    if ((xpeak is None and ypeak is not None)
            or (xpeak is not None and ypeak is None)):
        raise ValueError('xpeak and ypeak must both be input or "None"')

    if xpeak is not None and ((xpeak < 0) or (xpeak > data.shape[1] - 1)):
        raise ValueError('xpeak is outside of the input data')
    if ypeak is not None and ((ypeak < 0) or (ypeak > data.shape[0] - 1)):
        raise ValueError('ypeak is outside of the input data')

    data = np.asanyarray(data, dtype=float)
    ny, nx = data.shape

    badmask = ~np.isfinite(data)
    if np.any(badmask):
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)
        data[badmask] = np.nan

    if mask is not None:
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data[mask] = np.nan

    fit_boxsize = _process_boxsize(fit_boxsize, data.shape)

    if np.product(fit_boxsize) < 6:
        raise ValueError('fit_boxsize is too small.  6 values are required '
                         'to fit a 2D quadratic polynomial.')

    if xpeak is None or ypeak is None:
        yidx, xidx = np.unravel_index(np.nanargmax(data), data.shape)
    else:
        xidx = _py2intround(xpeak)
        yidx = _py2intround(ypeak)

        if search_boxsize is not None:
            search_boxsize = _process_boxsize(search_boxsize, data.shape)

            slc_data, _ = overlap_slices(data.shape, search_boxsize,
                                         (yidx, xidx), mode='trim')
            cutout = data[slc_data]
            yidx, xidx = np.unravel_index(np.nanargmax(cutout), cutout.shape)
            xidx += slc_data[1].start
            yidx += slc_data[0].start

    # if peak is at the edge of the data, return the position of the maximum
    if xidx == 0 or xidx == nx - 1 or yidx == 0 or yidx == ny - 1:
        warnings.warn('maximum value is at the edge of the data and its '
                      'position was returned; no quadratic fit was '
                      'performed', AstropyUserWarning)
        return np.array((xidx, yidx), dtype=float)

    # extract the fitting region
    slc_data, _ = overlap_slices(data.shape, fit_boxsize, (yidx, xidx),
                                 mode='trim')
    xidx0, xidx1 = (slc_data[1].start, slc_data[1].stop)
    yidx0, yidx1 = (slc_data[0].start, slc_data[0].stop)

    # shift the fitting box if it was clipped by the data edge
    if (xidx1 - xidx0) < fit_boxsize[1]:
        if xidx0 == 0:
            xidx1 = min(nx, xidx0 + fit_boxsize[1])
        if xidx1 == nx:
            xidx0 = max(0, xidx1 - fit_boxsize[1])
    if (yidx1 - yidx0) < fit_boxsize[0]:
        if yidx0 == 0:
            yidx1 = min(ny, yidx0 + fit_boxsize[0])
        if yidx1 == ny:
            yidx0 = max(0, yidx1 - fit_boxsize[0])

    cutout = data[yidx0:yidx1, xidx0:xidx1].ravel()
    if np.count_nonzero(~np.isnan(cutout)) < 6:
        warnings.warn('at least 6 unmasked data points are required to '
                      'perform a 2D quadratic fit',
                      AstropyUserWarning)
        return np.array((np.nan, np.nan))

    # fit a 2D quadratic polynomial to the fitting region
    xi = np.arange(xidx0, xidx1)
    yi = np.arange(yidx0, yidx1)
    x, y = np.meshgrid(xi, yi)
    x = x.ravel()
    y = y.ravel()
    coeff_matrix = np.vstack((np.ones_like(x), x, y, x * y, x * x, y * y)).T

    try:
        c = np.linalg.lstsq(coeff_matrix, cutout, rcond=None)[0]
    except np.linalg.LinAlgError:
        warnings.warn('quadratic fit failed', AstropyUserWarning)
        return np.array((np.nan, np.nan))

    # analytically find the maximum of the polynomial
    _, c10, c01, c11, c20, c02 = c
    det = 4 * c20 * c02 - c11**2
    if det <= 0 or ((c20 > 0.0 and c02 >= 0.0) or (c20 >= 0.0 and c02 > 0.0)):
        warnings.warn('quadratic fit does not have a maximum',
                      AstropyUserWarning)
        return np.array((np.nan, np.nan))

    xm = (c01 * c11 - 2.0 * c02 * c10) / det
    ym = (c10 * c11 - 2.0 * c20 * c01) / det
    if 0.0 < xm < (nx - 1.0) and 0.0 < ym < (ny - 1.0):
        xycen = np.array((xm, ym), dtype=float)
    else:
        warnings.warn('quadratic polynomial maximum value falls outside '
                      'of the image', AstropyUserWarning)
        return np.array((np.nan, np.nan))

    return xycen


def _process_boxsize(box_size, data_shape):
    box_size = np.round(np.atleast_1d(box_size)).astype(int)
    if len(box_size) == 1:
        box_size = np.repeat(box_size, 2)
    if len(box_size) > 2:
        raise ValueError('box size must contain only 1 or 2 values')
    if np.any(box_size < 0):
        raise ValueError('box size must be >= 0')
    # box_size cannot be larger than the data shape
    box_size = (min(box_size[0], data_shape[0]),
                min(box_size[1], data_shape[1]))
    return box_size


def centroid_sources(data, xpos, ypos, box_size=11, footprint=None, mask=None,
                     centroid_func=centroid_com, **kwargs):
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

    **kwargs : `dict`
        Any additional keyword arguments accepted by the
        ``centroid_func``.

    Returns
    -------
    xcentroid, ycentroid : `~numpy.ndarray`
        The ``x`` and ``y`` pixel position(s) of the centroids. NaNs
        will be returned where the centroid failed. This is usually due
        a ``box_size`` that is too small when using a fitting-based
        centroid function (e.g., `centroid_1dg`, `centroid_2dg`, or
        `centroid_quadratic`.
    """
    xpos = np.atleast_1d(xpos)
    ypos = np.atleast_1d(ypos)
    if xpos.ndim != 1:
        raise ValueError('xpos must be a 1D array.')
    if ypos.ndim != 1:
        raise ValueError('ypos must be a 1D array.')

    if (np.any(np.min(xpos) < 0) or np.any(np.min(ypos) < 0)
            or np.any(np.max(xpos) > data.shape[1] - 1)
            or np.any(np.max(ypos) > data.shape[0] - 1)):
        raise ValueError('xpos, ypos values contains point(s) outside of '
                         'input data')

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

    spec = inspect.getfullargspec(centroid_func)
    if 'mask' not in spec.args:
        raise ValueError('The input "centroid_func" must have a "mask" '
                         'keyword.')

    # drop any **kwargs not supported by the centroid_func
    centroid_kwargs = {}
    for key, val in kwargs.items():
        if key in spec.args:
            centroid_kwargs[key] = val

    xcentroids = []
    ycentroids = []
    for xp, yp in zip(xpos, ypos):
        slices_large, slices_small = overlap_slices(data.shape,
                                                    footprint.shape, (yp, xp))
        data_cutout = data[slices_large]

        footprint_mask = np.logical_not(footprint)
        # trim footprint mask if it has only partial overlap on the data
        footprint_mask = footprint_mask[slices_small]

        if mask is not None:
            # combine the input mask cutout and footprint mask
            mask_cutout = np.logical_or(mask[slices_large], footprint_mask)
        else:
            mask_cutout = footprint_mask

        centroid_kwargs.update({'mask': mask_cutout})

        if 'error' in centroid_kwargs:
            error_cutout = centroid_kwargs['error'][slices_large]
            centroid_kwargs['error'] = error_cutout

        if 'xpeak' in centroid_kwargs and 'ypeak' in centroid_kwargs:
            centroid_kwargs['xpeak'] -= slices_large[1].start
            centroid_kwargs['ypeak'] -= slices_large[0].start

        try:
            xcen, ycen = centroid_func(data_cutout, **centroid_kwargs)
        except (ValueError, TypeError):
            xcen, ycen = np.nan, np.nan

        xcentroids.append(xcen + slices_large[1].start)
        ycentroids.append(ycen + slices_large[0].start)

    return np.array(xcentroids), np.array(ycentroids)


@deprecated('1.2')
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
