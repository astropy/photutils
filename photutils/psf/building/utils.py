from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


__all__ = ['py2round', 'interpolate_missing_data']


def py2round(x):
    """
    This function returns a rounded up value of the argument, similar
    to Python 2.

    """
    if hasattr(x, '__iter__'):
        rx = np.empty_like(x)
        m = x >= 0.0
        rx[m] = np.floor(x[m] + 0.5)
        m = np.logical_not(m)
        rx[m] = np.ceil(x[m] - 0.5)
        return rx

    else:
        if x >= 0.0:
            return np.floor(x + 0.5)
        else:
            return np.ceil(x - 0.5)


def interpolate_missing_data(data, method, mask=None, const_fillval=0.0):
    """
    Interpolate over (or fill in) bad data as identified by the ``mask``
    parameter. This function assumes that `True` values in the mask indicate
    "good" data and that `False` values indicate the location of "bad" pixels
    in input ``data``.

    Parameters
    ----------

    data : numpy.ndarray
        Array containing 2D image.

    method : str
        Method of "interpolating" over missing data. Possible values are:

        - **'nearest'**: Data are interpolated using nearest-neighbour
          interpolation.

        - **'cubic'**: Data are interpolated using 2D cubic splines.

        - **'const'**: Missing data are filled in with a constant value defined
          by the ``const_fillval`` parameter.

    mask : numpy.ndarray, optional
        Array containing 2D boolean array of the same shape as ``data``.
        `True` values in the mask show "good" data and `False` values show
        the location of "bad" pixels in input ``data``.

    const_fillval : float, optional
        Constant value used to replace missing (or "bad" data) when ``method``
        is ``'const'``. This parameter is ignored for other methods.

    Returns
    -------
    idata : numpy.ndarray
        A 2D image of the same shape as input ``data`` interpolated over
        missing ("bad") data.

    """
    from scipy import interpolate

    idata = np.array(data, copy=True)
    if len(idata.shape) != 2:
        raise ValueError("Input 'data' must be a 2D array-like object.")

    if mask is None:
        return idata

    imask = np.logical_not(mask)
    if idata.shape != imask.shape:
        raise ValueError("'mask' must have same shape as 'data'.")

    if method == 'const':
        idata[imask] = const_fillval
        return idata

    y, x = np.indices(idata.shape)
    xy = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    z = idata[mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)

    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)

    else:
        raise ValueError("Unsupported interpolation method.")

    xynan = np.dstack((x[imask].ravel(), y[imask].ravel()))[0]
    idata[imask] = interpol(xynan)

    return idata
