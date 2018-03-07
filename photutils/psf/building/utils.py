
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import numpy as np
from scipy import interpolate
from scipy.stats import sigmaclip
from astropy.stats import SigmaClip
from astropy.convolution import convolve, Kernel

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


__all__ = ['py2round', 'interpolate_missing_data']


_kernel_quar = np.array(
    [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
     [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
     [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
     [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
     [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]]
)

_kernel_quad = np.array(
    [[-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311],
     [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
     [+0.03999952, 0.12571449, 0.15428215, 0.12571449, +0.03999952],
     [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
     [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311]]
)


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


def _pixstat(data, stat='mean',
             sigma_clip=SigmaClip(sigma_lower=3., sigma_upper=3.),
             default=np.nan):

    nclip = 0
    #if nclip > 0:
    #    if lsig is None or usig is None:
    #        raise ValueError("When 'nclip' > 0 neither 'lsig' nor 'usig' "
    #                         "may be None")

    data = np.ravel(data)
    nd, = data.shape

    if nd == 0:
        return default

    m = np.mean(data, dtype=np.float64)

    if nd == 1:
        return m

    need_std = (stat != 'mean' or nclip > 0)
    if need_std:
        s = np.std(data, dtype=np.float64)

    i = np.ones(nd, dtype=np.bool)


    low = sigma_clip.sigma_lower
    high = sigma_clip.sigma_upper
    #data_clipped = sigmaclip(data, low=low, high=high)

    for x in range(nclip):
        m_prev = m
        s_prev = s
        nd_prev = nd

        # sigma clipping:
        lval = m - lsig * s
        uval = m + usig * s
        i = ((data >= lval) & (data <= uval))
        d = data[i]
        nd, = d.shape
        if nd < 1:
            # return statistics based on previous iteration
            break

        m = np.mean(d, dtype=np.float64)
        s = np.std(d, dtype=np.float64)

        if nd == nd_prev:
            # NOTE: we could also add m == m_prev and s == s_prev
            # NOTE: a more rigurous check would be needed to see if
            #       index array 'i' did not change but that would be too slow
            #       and the current check is very likely good enough.
            break

    if stat == 'mean':
        return m
        #return np.mean(data_clipped)
    elif stat == 'median':
        return np.median(data[i])
        #return np.median(data_clipped)
    elif stat == 'pmode1':
        return (2.5 * np.median(data[i]) - 1.5 * m)
        #return (2.5 * np.median(data_clipped) - 1.5 * np.mean(data_clipped))
    elif stat == 'pmode2':
        return (3.0 * np.median(data[i]) - 2.0 * m)
        #return (3.0 * np.median(data_clipped) - 2.0 * np.mean(data_clipped))
    else:
        raise ValueError("Unsupported 'stat' value")


def _smoothPSF(psf, kernel):
    if kernel is None:
        return psf
    if kernel == 'quad':
        ker = _kernel_quad
    elif kernel == 'quar':
        ker = _kernel_quar
    elif isinstance(kernel, np.ndarray) or isinstance(kernel, Kernel):
        ker = kernel
    else:
        raise TypeError("Unsupported kernel.")

    spsf = convolve(psf, ker)

    return spsf


def _parse_tuple_pars(par, default=None, name='', dtype=None,
                      check_positive=True):
    if par is None:
        par = default

    if hasattr(par, '__iter__'):
        if len(par) != 2:
            raise TypeError("Parameter '{:s}' must be either a scalar or an "
                            "iterable with two elements.".format(name))
        px = par[0]
        py = par[1]
    elif par is None:
        return (None, None)
    else:
        px = par
        py = par

    if dtype is not None or check_positive:
        try:
            pxf = dtype(px)
            pyf = dtype(px)
        except TypeError:
            raise TypeError("Parameter '{:s}' must be a number or a tuple of "
                            "numbers.".format(name))

        if dtype is not None:
            px = pxf
            py = pyf

        if check_positive and (pxf <= 0 or pyf <= 0):
            raise TypeError("Parameter '{:s}' must be a strictly positive "
                            "number or a tuple of strictly positive numbers."
                            .format(name))

    return (px, py)
