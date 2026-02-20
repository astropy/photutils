# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for centroiding sources.
"""

import inspect
import warnings

import numpy as np
from astropy.nddata import overlap_slices
from astropy.utils.decorators import deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._parameters import as_pair
from photutils.utils._repr import make_repr
from photutils.utils._round import py2intround

__all__ = ['CentroidQuadratic', 'centroid_com', 'centroid_quadratic',
           'centroid_sources']


def _validate_data(data, ndim=2):
    """
    Validate that the input data is a 2D array.
    """
    data = np.asanyarray(data, dtype=float)
    if ndim is not None and data.ndim != ndim:
        msg = f'data must be a {ndim}D array'
        raise ValueError(msg)
    return data


def _validate_mask_shape(data, mask):
    """
    Validate that the data and mask have the same shape.
    """
    if mask is not None and data.shape != mask.shape:
        msg = 'data and mask must have the same shape'
        raise ValueError(msg)


def _process_data_mask(data, mask, ndim=2, fill_value=np.nan):
    """
    Process the input data and mask.

    This function validates the input data and mask, handles non-finite
    values, and returns the processed data and mask.
    """
    data = _validate_data(data, ndim=ndim)
    is_copied = False
    _validate_mask_shape(data, mask)

    badmask = ~np.isfinite(data)
    if mask is not None:
        if np.any(mask):
            data = data.copy()
            is_copied = True
            data[mask] = fill_value
        badmask &= ~mask

    if np.any(badmask):
        warnings.warn('Input data contains non-finite values (e.g., NaN or '
                      'inf) that were automatically masked.',
                      AstropyUserWarning)
        if not is_copied:
            data = data.copy()
        data[badmask] = fill_value

    return data


def centroid_com(data, *, mask=None):
    """
    Calculate the centroid of an n-dimensional array as
    its "center of mass" determined from `image moments
    <https://en.wikipedia.org/wiki/Image_moment>`_.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The input n-dimensional array. The image should be a
        background-subtracted cutout image containing a single
        source. The source should be significantly stronger than the
        background noise. If the data contains nearly equal positive and
        negative values (i.e., the sum is close to zero), the centroid
        calculation will be numerically unstable and may produce
        undefined results that fall outside the array bounds.

    mask : bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The coordinates of the centroid in pixel order (e.g., ``(x,
        y)`` or ``(x, y, z)``), not numpy axis order. If the sum of the
        (unmasked) data is zero, then a `~numpy.ndarray` of NaN values
        will be returned. If the sum is close to zero, the centroid may
        be poorly defined and fall outside the array bounds.

    Notes
    -----
    The centroid is calculated as:

    .. math::
        x_c = \\frac{\\sum x_i I_i}{\\sum I_i}, \\quad
        y_c = \\frac{\\sum y_i I_i}{\\sum I_i}

    where :math:`I_i` is the intensity at pixel :math:`(x_i, y_i)`.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.centroids import centroid_com
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> x1, y1 = centroid_com(data)
    >>> print(np.array((x1, y1)))
    [19.9796724  20.00992593]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_com
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        data = data[40:80, 70:110]
        xycen = centroid_com(data)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(data, origin='lower', interpolation='nearest')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    # preserve input data - which should be a small cutout image
    data = _process_data_mask(data, mask, ndim=None, fill_value=0.0)

    total = np.sum(data)
    if total == 0:
        return np.full(data.ndim, np.nan)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]

    # Output array is reversed to give (x, y) order
    return np.array([np.sum(indices[axis] * data) / total
                     for axis in range(data.ndim)])[::-1]


@deprecated_renamed_argument('xpeak', None, '3.0')
@deprecated_renamed_argument('ypeak', None, '3.0')
@deprecated_renamed_argument('search_boxsize', None, '3.0')
def centroid_quadratic(data, *, mask=None, fit_boxsize=5, xpeak=None,
                       ypeak=None, search_boxsize=None):
    """
    Calculate the centroid of an n-dimensional array by fitting a 2D
    quadratic polynomial.

    A second degree 2D polynomial is fit within a small region of the
    data defined by ``fit_boxsize`` to calculate the centroid position.
    The initial center of the fitting box can be specified using the
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
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be a background-subtracted
        cutout image containing a single source.

    mask : bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    fit_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to define the fitting
        region. If ``fit_boxsize`` has two elements, they must be in
        ``(ny, nx)`` order. If ``fit_boxsize`` is a scalar then a square
        box of size ``fit_boxsize`` will be used. ``fit_boxsize`` must
        have odd values for both axes.

    xpeak, ypeak : float or `None`, optional
        The initial guess of the position of the centroid. If either
        ``xpeak`` or ``ypeak`` is `None` then the position of the
        maximum value in the input ``data`` will be used as the initial
        guess.

        .. deprecated:: 3.0
           The ``xpeak`` and ``ypeak`` keywords are deprecated
           and will be removed in a future version. Use
           `~photutils.centroids.centroid_sources` to centroid sources
           at specific positions.

    search_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to search for the maximum
        pixel value if ``xpeak`` and ``ypeak`` are both specified. If
        ``fit_boxsize`` has two elements, they must be in ``(ny,
        nx)`` order. If ``search_boxsize`` is a scalar then a square
        box of size ``search_boxsize`` will be used. ``search_boxsize``
        must have odd values for both axes. This parameter is ignored
        if either ``xpeak`` or ``ypeak`` is `None`. In that case, the
        entire array is searched for the maximum value.

        .. deprecated:: 3.0
           The ``search_boxsize`` keyword is deprecated
           and will be removed in a future version. Use
           `~photutils.centroids.centroid_sources` to centroid sources
           at specific positions.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.

    Notes
    -----
    Use ``fit_boxsize = (3, 3)`` to match the work of `Vakili &
    Hogg (2016) <https://arxiv.org/abs/1610.05873>`_ for their 2D
    second-order polynomial centroiding method.

    Because this centroid is based on fitting data, it can fail for many
    reasons, returning (np.nan, np.nan):

    * quadratic fit failed
    * quadratic fit does not have a maximum
    * quadratic fit maximum falls outside image
    * not enough unmasked data points (6 are required)

    Also note that a fit is not performed if the maximum data value is
    at the edge of the data. In this case, the position of the maximum
    pixel will be returned.

    References
    ----------
    .. [1] Vakili and Hogg 2016, "Do fast stellar centroiding methods
           saturate the Cramér-Rao lower bound?", `arXiv:1610.05873
           <https://arxiv.org/abs/1610.05873>`_

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.centroids import centroid_quadratic
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> x1, y1 = centroid_quadratic(data)
    >>> print(np.array((x1, y1)))
    [19.94009505 20.06884997]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        data = data[40:80, 70:110]
        xycen = centroid_quadratic(data)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(data, origin='lower', interpolation='nearest')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    data = _process_data_mask(data, mask)
    ny, nx = data.shape

    fit_boxsize = as_pair('fit_boxsize', fit_boxsize, lower_bound=(0, 1),
                          upper_bound=data.shape, check_odd=True)

    if np.prod(fit_boxsize) < 6:
        msg = ('fit_boxsize is too small. 6 values are required to fit a '
               '2D quadratic polynomial.')
        raise ValueError(msg)

    if ((xpeak is None and ypeak is not None)
            or (xpeak is not None and ypeak is None)):
        msg = 'xpeak and ypeak must both be input or "None"'
        raise ValueError(msg)

    if xpeak is not None and ((xpeak < 0) or (xpeak > data.shape[1] - 1)):
        msg = 'xpeak is outside the input data'
        raise ValueError(msg)
    if ypeak is not None and ((ypeak < 0) or (ypeak > data.shape[0] - 1)):
        msg = 'ypeak is outside the input data'
        raise ValueError(msg)

    if xpeak is None or ypeak is None:
        yidx, xidx = np.unravel_index(np.nanargmax(data), data.shape)
    else:
        xidx = py2intround(xpeak)
        yidx = py2intround(ypeak)

        if search_boxsize is not None:
            search_boxsize = as_pair('search_boxsize', search_boxsize,
                                     lower_bound=(0, 1),
                                     upper_bound=data.shape, check_odd=True)

            slc_data, _ = overlap_slices(data.shape, search_boxsize,
                                         (yidx, xidx), mode='trim')
            cutout = data[slc_data]
            yidx, xidx = np.unravel_index(np.nanargmax(cutout), cutout.shape)
            xidx += slc_data[1].start
            yidx += slc_data[0].start

    # Return the position of the maximum if it is at the edge of the
    # data
    if xidx in (0, nx - 1) or yidx in (0, ny - 1):
        warnings.warn('maximum value is at the edge of the data and its '
                      'position was returned; no quadratic fit was '
                      'performed', AstropyUserWarning)
        return np.array((xidx, yidx), dtype=float)

    # Extract the fitting region
    slc_data, _ = overlap_slices(data.shape, fit_boxsize, (yidx, xidx),
                                 mode='trim')
    xidx0, xidx1 = (slc_data[1].start, slc_data[1].stop)
    yidx0, yidx1 = (slc_data[0].start, slc_data[0].stop)

    # Shift the fitting box if it was clipped by the data edge
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

    # Fit a 2D quadratic polynomial to the fitting region
    xi = np.arange(xidx0, xidx1)
    yi = np.arange(yidx0, yidx1)
    x, y = np.meshgrid(xi, yi)
    x = x.ravel()
    y = y.ravel()

    # Pre-allocate coefficient matrix for optimization
    coeff_matrix = np.empty((x.size, 6), dtype=float)
    coeff_matrix[:, 0] = 1
    coeff_matrix[:, 1] = x
    coeff_matrix[:, 2] = y
    coeff_matrix[:, 3] = x * y
    coeff_matrix[:, 4] = x * x
    coeff_matrix[:, 5] = y * y

    # Remove NaNs from data to be fit
    mask = ~np.isnan(cutout)
    if np.any(mask):
        coeff_matrix = coeff_matrix[mask]
        cutout = cutout[mask]

    try:
        c = np.linalg.lstsq(coeff_matrix, cutout, rcond=None)[0]
    except np.linalg.LinAlgError:
        warnings.warn('quadratic fit failed', AstropyUserWarning)
        return np.array((np.nan, np.nan))

    # Analytically find the maximum of the polynomial
    _, c10, c01, c11, c20, c02 = c
    det = 4 * c20 * c02 - c11**2

    # If the determinant is <= 0, the surface has a saddle point. If
    # the determinant is > 0, the surface has a minimum or maximum. The
    # curvature is negative (maximum) if c20 < 0 and c02 < 0. However,
    # if det > 0, then 4 * c20 * c02 > c11**2 >= 0, so c20 and c02 must
    # have the same sign. Therefore, we only need to check if c20 > 0
    # (or c02 > 0) to determine if the surface has a minimum.
    if det <= 0 or c20 > 0:
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


class CentroidQuadratic:
    """
    Class to calculate the centroid of a 2D array by fitting a 2D
    quadratic polynomial.

    This class provides a callable interface to the
    `~photutils.centroids.centroid_quadratic` function, allowing a
    centroid function with specific fit parameters to be defined and
    reused. This is useful, for example, when using a customized
    centroid function with `~photutils.centroids.centroid_sources`.

    Parameters
    ----------
    fit_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to define the fitting
        region. If ``fit_boxsize`` has two elements, they must be in
        ``(ny, nx)`` order. If ``fit_boxsize`` is a scalar then a square
        box of size ``fit_boxsize`` will be used. ``fit_boxsize`` must
        have odd values for both axes.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_4gaussians_image
    >>> from photutils.centroids import CentroidQuadratic
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> centroid_func = CentroidQuadratic(fit_boxsize=5)
    >>> x1, y1 = centroid_func(data)
    >>> print(np.array((x1, y1)))  # doctest: +FLOAT_CMP
    [19.94009505 20.06884997]

    Using with `~photutils.centroids.centroid_sources`::

        >>> from photutils.centroids import centroid_sources
        >>> data = make_4gaussians_image()
        >>> data -= np.median(data[0:30, 0:125])
        >>> x_init = (25, 91, 151, 160)
        >>> y_init = (40, 61, 24, 71)
        >>> centroid_func = CentroidQuadratic(fit_boxsize=3)
        >>> x, y = centroid_sources(data, x_init, y_init, box_size=25,
        ...                         centroid_func=centroid_func)
    """

    def __init__(self, *, fit_boxsize=5):
        self.fit_boxsize = fit_boxsize

    def __repr__(self):
        return make_repr(self, ['fit_boxsize'])

    def __str__(self):
        return make_repr(self, ['fit_boxsize'], long=True)

    def __call__(self, data, *, mask=None):
        """
        Calculate the centroid.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image data. The image should be a
            background-subtracted cutout image containing a single
            source.

        mask : bool `~numpy.ndarray`, optional
            A boolean mask, with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from calculations.

        Returns
        -------
        centroid : `~numpy.ndarray`
            The ``x, y`` coordinates of the centroid.
        """
        kwargs = {'mask': mask,
                  'fit_boxsize': self.fit_boxsize,
                  }
        return centroid_quadratic(data, **kwargs)


def centroid_sources(data, xpos, ypos, *, box_size=11, footprint=None,
                     mask=None, centroid_func=centroid_com, **kwargs):
    """
    Calculate the centroid of sources at the defined positions.

    A cutout image centered on each input position will be used to
    calculate the centroid position. The cutout image is defined either
    using the ``box_size`` or ``footprint`` keyword. The ``footprint``
    keyword can be used to create a non-rectangular cutout image.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. The image should be background-subtracted.

    xpos, ypos : float or array_like of float
        The initial ``x`` and ``y`` pixel position(s) of the center
        position. A cutout image centered on this position will be used
        to calculate the centroid.

    box_size : int or array_like of int, optional
        The size of the cutout image along each axis. If ``box_size`` is
        a number, then a square cutout of ``box_size`` will be created.
        If ``box_size`` has two elements, they must be in ``(ny, nx)``
        order. ``box_size`` must have odd values for both axes. Either
        ``box_size`` or ``footprint`` must be defined. If they are both
        defined, then ``footprint`` overrides ``box_size``.

    footprint : bool `~numpy.ndarray`, optional
        A 2D boolean array where `True` values describe the local
        footprint region to cutout. ``footprint`` can be used to create
        a non-rectangular cutout image, in which case the input ``xpos``
        and ``ypos`` represent the center of the minimal bounding box
        for the input ``footprint``. ``box_size=(n, m)`` is equivalent
        to ``footprint=np.ones((n, m))``. Either ``box_size`` or
        ``footprint`` must be defined. If they are both defined, then
        ``footprint`` overrides ``box_size``. The same ``footprint`` is
        used for all sources.

    mask : 2D bool `~numpy.ndarray`, optional
        A 2D boolean array with the same shape as ``data``, where a
        `True` value indicates the corresponding element of ``data`` is
        masked.

    centroid_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword and
        optionally an ``error`` keyword. The callable object must return
        a tuple of two 1D `~numpy.ndarray`, representing the x and y
        centroids. The default is `~photutils.centroids.centroid_com`.

    **kwargs : dict, optional
        Any additional keyword arguments accepted by the
        ``centroid_func``.

    Returns
    -------
    xcentroid, ycentroid : `~numpy.ndarray`
        The ``x`` and ``y`` pixel position(s) of the centroids. NaNs
        will be returned where the centroid failed. This is usually due
        a ``box_size`` that is too small when using a fitting-based
        centroid function (e.g., `centroid_1dg`, `centroid_2dg`, or
        `centroid_quadratic`).

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.centroids import centroid_2dg, centroid_sources
    >>> from photutils.datasets import make_4gaussians_image

    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> x_init = (25, 91, 151, 160)
    >>> y_init = (40, 61, 24, 71)
    >>> x, y = centroid_sources(data, x_init, y_init, box_size=25,
    ...                         centroid_func=centroid_2dg)
    >>> print(x)  # doctest: +FLOAT_CMP
    [ 24.96807828  89.98684636 149.96545721 160.18810915]
    >>> print(y)  # doctest: +FLOAT_CMP
    [40.03657613 60.01836631 24.96777946 69.80208702]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_2dg, centroid_sources
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        x_init = (25, 91, 151, 160)
        y_init = (40, 61, 24, 71)
        x, y = centroid_sources(data, x_init, y_init, box_size=25,
                                centroid_func=centroid_2dg)
        plt.figure(figsize=(8, 4))
        plt.imshow(data, origin='lower', interpolation='nearest')
        plt.scatter(x, y, marker='+', s=80, color='red', label='Centroids')
        plt.legend()
        plt.tight_layout()
    """
    xpos = np.atleast_1d(xpos)
    ypos = np.atleast_1d(ypos)
    if xpos.ndim != 1:
        msg = 'xpos must be a 1D array'
        raise ValueError(msg)
    if ypos.ndim != 1:
        msg = 'ypos must be a 1D array'
        raise ValueError(msg)

    if (np.any(np.min(xpos) < 0) or np.any(np.min(ypos) < 0)
            or np.any(np.max(xpos) > data.shape[1] - 1)
            or np.any(np.max(ypos) > data.shape[0] - 1)):
        msg = 'xpos, ypos values contain points outside the input data'
        raise ValueError(msg)

    if footprint is None:
        if box_size is None:
            msg = 'box_size or footprint must be defined'
            raise ValueError(msg)
        box_size = as_pair('box_size', box_size, lower_bound=(0, 1),
                           check_odd=True)
        footprint = np.ones(box_size, dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)
        if footprint.ndim != 2:
            msg = 'footprint must be a 2D array'
            raise ValueError(msg)

    spec = inspect.signature(centroid_func)
    if 'mask' not in spec.parameters:
        msg = 'The input "centroid_func" must have a "mask" keyword.'
        raise ValueError(msg)

    # Drop any **kwargs not supported by the centroid_func
    centroid_kwargs = {key: val for key, val in kwargs.items()
                       if key in spec.parameters}

    n_sources = len(xpos)
    xcentroids = np.zeros(n_sources, dtype=float)
    ycentroids = np.zeros(n_sources, dtype=float)

    for i, (xp, yp) in enumerate(zip(xpos, ypos, strict=True)):
        slices_large, slices_small = overlap_slices(data.shape,
                                                    footprint.shape, (yp, xp))
        data_cutout = data[slices_large]

        footprint_mask = np.logical_not(footprint)
        # Trim footprint mask if it has only partial overlap on the data
        footprint_mask = footprint_mask[slices_small]

        if mask is not None:
            # Combine the input mask cutout and footprint mask
            mask_cutout = np.logical_or(mask[slices_large], footprint_mask)
        else:
            mask_cutout = footprint_mask

        if np.all(mask_cutout):
            msg = (f'The cutout for the source at ({xp, yp}) is completely '
                   'masked. Please check your input mask and footprint. '
                   'Also note that footprint must be a small, local '
                   'footprint.')
            raise ValueError(msg)

        centroid_kwargs.update({'mask': mask_cutout})

        error = centroid_kwargs.get('error')
        if error is not None:
            centroid_kwargs['error'] = error[slices_large]

        # Remove this block once xpeak and ypeak are fully deprecated
        # remove xpeak and ypeak from the dict and add back only if both
        # are specified and not None
        xpeak = centroid_kwargs.pop('xpeak', None)
        ypeak = centroid_kwargs.pop('ypeak', None)
        if xpeak is not None and ypeak is not None:
            centroid_kwargs['xpeak'] = xpeak - slices_large[1].start
            centroid_kwargs['ypeak'] = ypeak - slices_large[0].start

        try:
            xcen, ycen = centroid_func(data_cutout, **centroid_kwargs)
        except (ValueError, TypeError):
            xcen, ycen = np.nan, np.nan

        xcentroids[i] = xcen + slices_large[1].start
        ycentroids[i] = ycen + slices_large[0].start

    return xcentroids, ycentroids
