# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for centroiding sources.
"""

import inspect
import numbers
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize

from photutils.centroids._utils import _process_data_mask
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._overlap import overlap_slices
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr
from photutils.utils._round import round_half_away

__all__ = ['CentroidQuadratic', 'centroid_com', 'centroid_quadratic',
           'centroid_sources', 'centroid_symmetry']


@deprecated_positional_kwargs(since='3.0', until='4.0')
def centroid_com(data, mask=None):
    """
    Calculate the centroid of an array as the flux-weighted
    center of mass derived from `image moments
    <https://en.wikipedia.org/wiki/Image_moment>`_.

    Non-finite values (e.g., NaN or inf) in the ``data`` array are
    automatically masked. An
    `~astropy.utils.exceptions.AstropyUserWarning` is emitted if any
    of them is not already masked. The final mask is a logical OR
    combination of the input ``mask``, the automatically generated mask
    for non-finite values, and the mask of the input ``data`` if it is a
    `~numpy.ma.MaskedArray`. The centroid is calculated using only the
    unmasked data values.

    Parameters
    ----------
    data : array_like
        The input n-dimensional array. ``data`` can be a
        `~numpy.ma.MaskedArray` or a `~astropy.units.Quantity`. The
        image should be a background-subtracted cutout image containing
        a single source. The source should be significantly stronger
        than the background noise. If the data contains nearly equal
        positive and negative values (i.e., the sum is close to zero),
        the centroid calculation will be numerically unstable and may
        produce undefined results that fall outside the array bounds.

    mask : bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        If ``data`` is a `~numpy.ma.MaskedArray`, its mask will be
        combined (using bitwise OR) with the input ``mask``.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The coordinates of the centroid in pixel order (e.g., ``(x, y)``
        or ``(x, y, z)``), not numpy axis order. If the absolute value
        of the sum of the (unmasked) data is smaller than 1e-30 (i.e.,
        consistent with zero), then a `~numpy.ndarray` of NaN values
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
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, origin='lower')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    (data,), _ = process_quantities((data,), ('data',))
    data = _process_data_mask(data, mask, ndim=None, fill_value=0.0)

    total = np.sum(data)
    if abs(total) < 1.e-30:
        return np.full(data.ndim, np.nan)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]

    # Output array is reversed to give (x, y) order (e.g., for 2D data)
    return np.array([np.sum(indices[axis] * data) / total
                     for axis in range(data.ndim)])[::-1]


def centroid_symmetry(data, *, mask=None, radius=None):
    """
    Calculate the center of a 2D array as its point of maximal point
    symmetry.

    The center is the position about which the data within ``radius``
    are most symmetric under a rotation by 180 degrees. It is found by
    minimizing the mean of the absolute differences between the data
    values at opposite offsets from the center.

    This is the definition of the center of an effective
    PSF (ePSF) of `Anderson 2016 (WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_.
    Unlike the center of mass (`centroid_com`), it depends only on the
    core of the source and is insensitive to asymmetric structure beyond
    ``radius``.

    Non-finite values (e.g., NaN or inf) in the
    ``data`` array are automatically masked. An
    `~astropy.utils.exceptions.AstropyUserWarning` is emitted if any
    of them is not already masked. The final mask is a logical OR
    combination of the input ``mask``, the automatically generated mask
    for non-finite values, and the mask of the input ``data`` if it is
    a `~numpy.ma.MaskedArray`. The pairs of opposite offsets with a
    position near a masked value are given a lower weight or are not
    used.

    .. warning::

        The result is less accurate if masked or non-finite values lie
        within ``radius`` of the center. The data are interpolated
        between the pixels, so a masked value is first replaced by a
        weighted mean of the unmasked values around it, which is only an
        estimate. In tests with a noiseless Gaussian source in a 21x21
        array, the default ``radius``, and a single masked pixel
        adjacent to its peak pixel, the result changed by up to 0.002
        pixels for a standard deviation of 2 pixels, 0.03 pixels for
        1.2 pixels, and 0.1 pixels for 0.9 pixels. The change is larger
        for a smaller ``radius``. It was up to 0.01 pixels for a radius
        of 3 pixels and a standard deviation of 2 pixels, and up to 0.2
        pixels for a radius of 1.5 pixels at any of these standard
        deviations.

        The search starts at the maximum value near the center of the
        array. A bright outlier such as a hot pixel or a cosmic ray
        near the source should be masked, because a single bright pixel
        is symmetric about itself.

    Parameters
    ----------
    data : 2D array_like
        The 2D image data. ``data`` can be a `~numpy.ma.MaskedArray`
        or a `~astropy.units.Quantity`. The image should be a
        background-subtracted cutout image containing a single positive
        source near its center. It must have at least 4 pixels along
        each axis.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        If ``data`` is a `~numpy.ma.MaskedArray`, its mask will be
        combined (using bitwise OR) with the input ``mask``.

    radius : float or `None`, optional
        The radius (in pixels) of the circular region around the center
        in which the symmetry is measured. If `None`, the radius is 0.3
        times the smaller dimension of ``data`` (e.g., 1.5 pixels for a
        5x5 array). The radius must be at least 1 pixel and smaller than
        half of the smaller dimension of ``data`` minus one half. The
        default is intended for a small cutout of the core of a source.
        For a cutout that is much larger than the core, set ``radius``
        to about the size of the core. A larger radius includes pairs of
        background values, which add noise to the result.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``(x, y)`` coordinates of the center. An array of NaN values
        is returned if no pair of opposite offsets is unmasked, if the
        unmasked data values are all equal, or if all of the values in
        the area that is searched for the center are masked.

    See Also
    --------
    centroid_com, centroid_quadratic

    Notes
    -----
    The asymmetry that is minimized is

    .. math::
        A(x_c, y_c) = \\frac{1}{N} \\sum_{|u| \\le r}
            \\left| I(x_c + u_x, y_c + u_y)
            - I(x_c - u_x, y_c - u_y) \\right|

    where the sum is over the :math:`N` pairs of opposite offsets
    :math:`\\pm u` on a grid within the radius :math:`r`, and :math:`I`
    is the data interpolated with a bicubic spline. The spacing of the
    grid is one pixel for a radius of 6 pixels or more and half a pixel
    for a smaller radius, so that a small region is still sampled by
    enough pairs. If there are masked values, the mean is a weighted
    mean. The weight of a position decreases linearly from 1 to 0 as
    it moves from an unmasked pixel halfway to a masked pixel, and the
    weight of a pair is the product of the weights of its two positions.

    The interpolation is less accurate for an undersampled source. For
    a noiseless Gaussian source integrated over the pixels, the error
    of the result is up to about 0.02 pixels for a standard deviation
    of 0.6 to 0.8 pixels (a FWHM of 1.4 to 1.9 pixels), 0.01 pixels for
    1 pixel, 0.005 pixels for 1.2 pixels, and less than 0.002 pixels
    for 1.5 pixels or more. The error increases quickly for a narrower
    source (0.05 pixels for a standard deviation of 0.5 pixels), and
    it is larger for a radius of less than 2 pixels, which is the
    default for an array with fewer than 7 pixels along an axis.

    The region must stay within the array, so the center is searched
    only within ``(n - 1) / 2 - radius`` pixels of the center of the
    array along each axis, where ``n`` is the size of the array along
    that axis. The source should therefore be roughly centered in
    ``data``. If the center of the source lies outside of that area, the
    returned position is on its edge. A region without a source is also
    symmetric, so the search starts at the maximum value near that area
    and finds the minimum of the asymmetry nearest to it. The source
    must therefore be positive.

    If the source lies far outside of that area, the asymmetry decreases
    away from the source and the search moves away from it. This case
    is recognized by the mean of the data values at the center and at
    the offsets within ``radius``, which is then less than half of its
    value at the starting position. The returned position is then the
    nearby position in that area where this mean is largest, which for
    a single source is the position nearest to it. The mean uses the
    estimates that replace the masked values. The two means are compared
    as a ratio, so the data must be background subtracted, and this
    case is not recognized if the mean at the starting position is not
    positive. If the noise is comparable to the data values near that
    area, the source cannot be recognized and the returned position is
    not meaningful.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.centroids import centroid_symmetry
    >>> from photutils.datasets import make_4gaussians_image
    >>> data = make_4gaussians_image()
    >>> data -= np.median(data[0:30, 0:125])
    >>> data = data[40:80, 70:110]
    >>> x1, y1 = centroid_symmetry(data)
    >>> print(np.array((x1, y1)))
    [19.98462513 20.0077986 ]

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.centroids import centroid_symmetry
        from photutils.datasets import make_4gaussians_image

        data = make_4gaussians_image()
        data -= np.median(data[0:30, 0:125])
        data = data[40:80, 70:110]
        xycen = centroid_symmetry(data)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, origin='lower')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    (data,), _ = process_quantities((data,), ('data',))
    data = _process_data_mask(data, mask, ndim=2, fill_value=np.nan)
    ny, nx = data.shape
    if min(ny, nx) < 4:
        msg = 'data must have at least 4 pixels along each axis'
        raise ValueError(msg)
    half_size = (min(ny, nx) - 1) / 2

    if radius is None:
        radius = 0.3 * min(ny, nx)
    if not isinstance(radius, numbers.Real):
        msg = 'radius must be a real number or None'
        raise TypeError(msg)
    if not 1 <= radius < half_size:
        msg = ('radius must be at least 1 and less than half of the '
               'smaller dimension of data minus one half')
        raise ValueError(msg)

    # Constant data are symmetric about every position
    bad = ~np.isfinite(data)
    if np.all(bad) or np.ptp(data[~bad]) == 0:
        return np.full(2, np.nan)

    # The spline needs finite values, and a masked value also changes
    # the spline at the adjacent positions. The masked values are
    # therefore replaced by a Gaussian-weighted mean of the unmasked
    # values around them. The pairs near a masked value are given a
    # lower weight in _asymmetry.
    if np.any(bad):
        total = gaussian_filter(np.where(bad, 0.0, data), 1.0,
                                mode='constant')
        weight = gaussian_filter((~bad).astype(float), 1.0,
                                 mode='constant')
        fill = np.divide(total, weight, out=np.zeros_like(total),
                         where=weight > 0)
        data = np.where(bad, fill, data)
        good = (~bad).astype(float)
    else:
        bad = None
        good = None
    spline = RectBivariateSpline(np.arange(ny), np.arange(nx), data)

    # One offset of each opposite pair within the radius. A small
    # radius has few pairs at a spacing of one pixel, so the spacing is
    # then half a pixel. Other fractions of a pixel are much less
    # accurate for an undersampled source.
    spacing = 0.5 if radius < 6 else 1.0
    n_max = int(radius / spacing)
    y_off, x_off = np.mgrid[-n_max:n_max + 1, -n_max:n_max + 1] * spacing
    keep = ((np.hypot(x_off, y_off) <= radius)
            & ((y_off > 0) | ((y_off == 0) & (x_off > 0))))
    x_off = x_off[keep]
    y_off = y_off[keep]
    args = (spline, x_off, y_off, good)

    # The region must stay within the array
    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2
    x_margin = x_center - radius
    y_margin = y_center - radius
    x_bounds = (x_center - x_margin, x_center + x_margin)
    y_bounds = (y_center - y_margin, y_center + y_margin)

    # A featureless region is also symmetric, so the search starts at
    # the maximum value and a coarse search is made only within one
    # pixel of it. The pixels that partly overlap the allowed region
    # are included, because the allowed region of an array with an even
    # size can lie between the pixel centers.
    yslc = slice(int(np.floor(y_bounds[0])), int(np.ceil(y_bounds[1])) + 1)
    xslc = slice(int(np.floor(x_bounds[0])), int(np.ceil(x_bounds[1])) + 1)
    region = data[yslc, xslc]
    if bad is not None:
        region = np.where(bad[yslc, xslc], -np.inf, region)
        if not np.any(np.isfinite(region)):
            return np.full(2, np.nan)
    ypeak, xpeak = np.unravel_index(np.argmax(region), region.shape)
    xy_peak = np.array((np.clip(xpeak + xslc.start, *x_bounds),
                        np.clip(ypeak + yslc.start, *y_bounds)))
    steps = np.linspace(-1.0, 1.0, 5)
    x_grid = np.unique(np.clip(xpeak + xslc.start + steps, *x_bounds))
    y_grid = np.unique(np.clip(ypeak + yslc.start + steps, *y_bounds))
    values = np.array([[_asymmetry((x, y), *args) for x in x_grid]
                       for y in y_grid])
    if not np.any(np.isfinite(values)):
        return np.full(2, np.nan)
    yidx, xidx = np.unravel_index(np.argmin(values), values.shape)
    x_init = x_grid[xidx]
    y_init = y_grid[yidx]

    # The default initial simplex of Nelder-Mead has a size that is
    # proportional to the starting coordinates. For a large array it
    # would reach the featureless region around the source. The steps
    # here are a fraction of a pixel and they point toward the center
    # of the allowed region to stay within the bounds.
    #
    # The scale of the asymmetry depends on the data, so only the size
    # of the simplex is used to stop the search.
    #
    # The positions outside of the bounds are moved onto them, so a
    # simplex next to an edge can collapse onto it and then never leave
    # it, even if the minimum is a fraction of the step inside. If the
    # asymmetry is lower just inside of a result on an edge, the search
    # is therefore repeated from that result with a smaller step.
    bounds = (x_bounds, y_bounds)
    xy_center = np.array((x_center, y_center))
    xy_margin = np.array((x_margin, y_margin))
    xatol = 1.0e-5
    step = 0.25
    while True:
        x_step, y_step = (np.minimum(step, xy_margin)
                          * np.where((x_init, y_init) <= xy_center, 1, -1))
        initial_simplex = [(x_init, y_init), (x_init + x_step, y_init),
                           (x_init, y_init + y_step)]
        options = {'initial_simplex': initial_simplex, 'xatol': xatol,
                   'fatol': np.inf}
        result = minimize(_asymmetry, (x_init, y_init), args=args,
                          method='Nelder-Mead', bounds=bounds,
                          options=options)
        xycen = np.array(result.x)
        step /= 4
        if step < xatol or not _is_lower_inside(xycen, result.fun, args,
                                                bounds, 10 * xatol):
            break
        x_init, y_init = xycen

    # If the source lies far outside of the allowed region, the region
    # around the starting position is beyond the wings of the source
    # where the asymmetry decreases away from the source, and the search
    # moves away from the source. The mean of the data values within
    # the radius is then much lower at the result than at the starting
    # position. For a source in the allowed region it is about the same
    # or larger, because the brightest pixel is near the source. The
    # position in the allowed region that is nearest to the source is
    # the one where the mean is largest. The mean is a smooth function
    # of the position, so a gradient-based search is used. It starts at
    # the brightest pixel so that it is not drawn to another source.
    #
    # The factor of 0.5 separates the two cases in tests with Gaussian
    # sources. The ratio of the two means was at least 0.79 for sources
    # in the allowed region with noise of up to 30% of the peak value,
    # and at most 0.40 for noiseless sources at 1.8 to 3 times the size
    # of the allowed region from its center. The ratio has no meaning
    # if the mean at the starting position is not positive.
    mean_args = (spline, x_off, y_off, np.ptp(data))
    peak_mean = -_negative_mean(xy_peak, *mean_args)[0]
    result_mean = -_negative_mean(xycen, *mean_args)[0]
    if peak_mean > 0 and result_mean < 0.5 * peak_mean:
        result = minimize(_negative_mean, xy_peak, args=mean_args,
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'ftol': 0.0, 'gtol': 1.0e-10})
        xycen = np.array(result.x)

    return xycen


def _is_lower_inside(xy, value, args, bounds, step):
    """
    Determine whether the asymmetry is lower just inside of a position
    on an edge of the allowed region.

    Parameters
    ----------
    xy : 1D `~numpy.ndarray`
        The ``(x, y)`` position.

    value : float
        The asymmetry at ``xy``.

    args : tuple
        The arguments of `_asymmetry` after the trial center.

    bounds : tuple of 2 tuples of 2 floats
        The ``(lower, upper)`` bounds of the allowed region along the x
        and y axes.

    step : float
        The distance (in pixels) from the edge at which the asymmetry
        is calculated.

    Returns
    -------
    result : bool
        `True` if ``xy`` is on an edge and the asymmetry is lower at a
        position that is ``step`` inside of it along any of the axes
        with an edge or along all of them.
    """
    lower, upper = np.transpose(bounds)
    inward = (np.isclose(xy, lower, rtol=0.0, atol=0.1 * step).astype(float)
              - np.isclose(xy, upper, rtol=0.0, atol=0.1 * step))
    axes = np.flatnonzero(inward)
    trials = [inward * (np.arange(2) == axis) for axis in axes]
    if axes.size == 2:
        trials.append(inward)
    return any(_asymmetry(xy + step * trial, *args) < value
               for trial in trials)


def _negative_mean(xy, spline, x_off, y_off, scale):
    """
    Calculate the negative of the mean of the data values at a trial
    center and at the pairs of opposite offsets from it, and its
    gradient.

    Parameters
    ----------
    xy : tuple of 2 floats
        The ``(x, y)`` trial center.

    spline : `~scipy.interpolate.RectBivariateSpline`
        The spline that interpolates the data.

    x_off, y_off : 1D `~numpy.ndarray`
        The x and y offsets of one position of each opposite pair.

    scale : float
        The value by which the mean is divided, which makes the gradient
        independent of the scale of the data.

    Returns
    -------
    result : float
        The negative of the scaled mean of the data values.

    gradient : 1D `~numpy.ndarray`
        The derivatives of ``result`` with respect to ``x`` and ``y``.
    """
    x_pos = xy[0] + np.concatenate((x_off, -x_off, (0.0,)))
    y_pos = xy[1] + np.concatenate((y_off, -y_off, (0.0,)))
    result = -np.mean(spline.ev(y_pos, x_pos)) / scale
    gradient = -np.array((np.mean(spline.ev(y_pos, x_pos, dy=1)),
                          np.mean(spline.ev(y_pos, x_pos, dx=1)))) / scale
    return result, gradient


def _asymmetry(xy, spline, x_off, y_off, good):
    """
    Calculate the weighted mean absolute difference of the data values
    at opposite offsets from a trial center.

    Parameters
    ----------
    xy : tuple of 2 floats
        The ``(x, y)`` trial center.

    spline : `~scipy.interpolate.RectBivariateSpline`
        The spline that interpolates the data.

    x_off, y_off : 1D `~numpy.ndarray`
        The x and y offsets of one position of each opposite pair.

    good : 2D float `~numpy.ndarray` or `None`
        An array that is 1 where the data values are unmasked and 0
        where they are masked. The weight of a position decreases
        linearly from 1 to 0 as it moves from an unmasked pixel halfway
        to a masked pixel, and the weight of a pair is the product of
        the weights of its two positions. The weights are a continuous
        function of the trial center, which keeps the asymmetry
        continuous. If `None`, all of the pairs have equal weight.

    Returns
    -------
    result : float
        The asymmetry about the trial center. It is infinite if every
        pair has zero weight.
    """
    x1 = xy[0] + x_off
    y1 = xy[1] + y_off
    x2 = xy[0] - x_off
    y2 = xy[1] - y_off
    diff = np.abs(spline.ev(y1, x1) - spline.ev(y2, x2))

    if good is None:
        return np.mean(diff)

    weights = 1.0
    for x, y in ((x1, y1), (x2, y2)):
        frac = map_coordinates(good, (y, x), order=1, mode='nearest')
        weights = weights * np.clip(2.0 * frac - 1.0, 0.0, 1.0)
    total = np.sum(weights)
    if total <= 0:
        return np.inf
    return np.sum(weights * diff) / total


@deprecated_positional_kwargs(since='3.0', until='4.0')
@deprecated_renamed_argument('xpeak', None, '3.0', until='4.0')
@deprecated_renamed_argument('ypeak', None, '3.0', until='4.0')
@deprecated_renamed_argument('search_boxsize', None, '3.0', until='4.0')
def centroid_quadratic(data, mask=None, fit_boxsize=5, xpeak=None,
                       ypeak=None, search_boxsize=None):
    """
    Calculate the centroid of a 2D array by fitting a 2D quadratic
    polynomial.

    Non-finite values (e.g., NaN or inf) in the
    ``data`` array are automatically masked. An
    `~astropy.utils.exceptions.AstropyUserWarning` is emitted if any
    of them is not already masked. The final mask is a logical OR
    combination of the input ``mask``, the automatically generated mask
    for non-finite values, and the mask of the input ``data`` if it is a
    `~numpy.ma.MaskedArray`. The centroid is calculated using only the
    unmasked data values.

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

    Vakili & Hogg (2016) [1]_ demonstrate that 2D quadratic centroiding
    comes very close to saturating the `Cramér-Rao lower bound
    <https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound>`_ in a
    wide range of conditions.

    Parameters
    ----------
    data : 2D array_like
        The 2D image data. ``data`` can be a `~numpy.ma.MaskedArray`
        or a `~astropy.units.Quantity`. The image should be a
        background-subtracted cutout image containing a single source.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations. If ``data`` is
        a `~numpy.ma.MaskedArray`, its mask will be combined (using
        bitwise OR) with the input ``mask``.

    fit_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to define the fitting
        region. If ``fit_boxsize`` has two elements, they must be in
        ``(ny, nx)`` order. If ``fit_boxsize`` is a scalar then a square
        box of size ``fit_boxsize`` will be used. ``fit_boxsize`` must
        have odd values for both axes and the box must contain at least
        6 pixels. A size larger than ``data`` along an axis is reduced
        to the largest odd size that fits, with a warning.

    xpeak, ypeak : float or `None`, optional
        The initial guess of the position of the centroid. If both are
        `None`, the position of the maximum value in the input ``data``
        is used as the initial guess. A `ValueError` is raised if only
        one of them is `None` or if the position is outside of
        ``data``.

        .. deprecated:: 3.0
           The ``xpeak`` and ``ypeak`` keywords are deprecated
           and will be removed in version 4.0. Use
           `~photutils.centroids.centroid_sources` to centroid sources
           at specific positions.

    search_boxsize : int or tuple of int, optional
        The size (in pixels) of the box used to search for the maximum
        pixel value if ``xpeak`` and ``ypeak`` are both specified. If
        ``search_boxsize`` has two elements, they must be in ``(ny,
        nx)`` order. If ``search_boxsize`` is a scalar then a square
        box of size ``search_boxsize`` will be used. ``search_boxsize``
        must have odd values for both axes. This parameter is ignored
        if ``xpeak`` and ``ypeak`` are `None`. In that case, the entire
        array is searched for the maximum value.

        .. deprecated:: 3.0
           The ``search_boxsize`` keyword is deprecated
           and will be removed in version 4.0. Use
           `~photutils.centroids.centroid_sources` to centroid sources
           at specific positions.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``(x, y)`` coordinates of the centroid.

    Notes
    -----
    Use ``fit_boxsize = (3, 3)`` to match the work of Vakili & Hogg
    (2016) [1]_ for their 2D second-order polynomial centroiding method.

    Because this centroid is based on fitting data, it can fail for
    several reasons. In each of these cases, an
    `~astropy.utils.exceptions.AstropyUserWarning` is emitted and
    ``(np.nan, np.nan)`` is returned:

    * quadratic fit failed
    * quadratic fit does not have a maximum
    * quadratic fit maximum falls outside image
    * not enough unmasked data points (6 are required)

    A `ValueError` is raised if all data values are masked or
    non-finite.

    Also note that a fit is not performed if the maximum data value is
    at the edge of the data. In this case, a warning is emitted and the
    position of the maximum pixel is returned.

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
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, origin='lower')
        ax.scatter(*xycen, color='red', marker='+', s=100, label='Centroid')
        ax.legend()
    """
    (data,), _ = process_quantities((data,), ('data',))

    if ((xpeak is None and ypeak is not None)
            or (xpeak is not None and ypeak is None)):
        msg = 'xpeak and ypeak must both be input or "None"'
        raise ValueError(msg)

    data = _process_data_mask(data, mask)
    ny, nx = data.shape

    if not np.any(np.isfinite(data)):
        msg = 'All data values are masked or non-finite'
        raise ValueError(msg)

    fit_boxsize = as_pair('fit_boxsize', fit_boxsize, lower_bound=(0, 0),
                          upper_bound=data.shape, check_odd=True)

    if np.prod(fit_boxsize) < 6:
        msg = ('fit_boxsize is too small. 6 values are required to fit a '
               '2D quadratic polynomial.')
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
        xidx = round_half_away(xpeak)
        yidx = round_half_away(ypeak)

        if search_boxsize is not None:
            search_boxsize = as_pair('search_boxsize', search_boxsize,
                                     lower_bound=(0, 0),
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
        msg = ('maximum value is at the edge of the data and its '
               'position was returned. No quadratic fit was performed')
        warnings.warn(msg, AstropyUserWarning)
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
        msg = ('at least 6 unmasked data points are required to '
               'perform a 2D quadratic fit')
        warnings.warn(msg, AstropyUserWarning)
        return np.array((np.nan, np.nan))

    # Fit a 2D quadratic polynomial to the fitting region. The fit
    # coordinates are centered on the peak pixel to keep the design
    # matrix well conditioned. With absolute coordinates the condition
    # number grows as ~coordinate**4 and the fit fails for sources at
    # large pixel coordinates (e.g., in large mosaic images).
    xi = np.arange(xidx0, xidx1) - xidx
    yi = np.arange(yidx0, yidx1) - yidx
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

    # Include only finite values in the fit.
    finite_mask = np.isfinite(cutout)
    if not np.all(finite_mask):
        coeff_matrix = coeff_matrix[finite_mask]
        cutout = cutout[finite_mask]

    try:
        c = np.linalg.lstsq(coeff_matrix, cutout, rcond=None)[0]
    except np.linalg.LinAlgError:
        msg = 'quadratic fit failed'
        warnings.warn(msg, AstropyUserWarning)
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
        msg = 'quadratic fit does not have a maximum'
        warnings.warn(msg, AstropyUserWarning)
        return np.array((np.nan, np.nan))

    # Add back the peak-pixel offset to convert the analytic maximum
    # from fit coordinates to data coordinates
    xm = (c01 * c11 - 2.0 * c02 * c10) / det + xidx
    ym = (c10 * c11 - 2.0 * c20 * c01) / det + yidx
    if 0.0 < xm < (nx - 1.0) and 0.0 < ym < (ny - 1.0):
        xycen = np.array((xm, ym), dtype=float)
    else:
        msg = 'quadratic polynomial maximum value falls outside of the image'
        warnings.warn(msg, AstropyUserWarning)
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
        have odd values for both axes and the box must contain at least
        6 pixels. A size larger than ``data`` along an axis is reduced
        to the largest odd size that fits, with a warning. The value is
        validated when the instance is called.

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
    >>> print(np.array((x1, y1)))
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

        Non-finite values (e.g., NaN or inf) in the ``data`` array
        are automatically masked. An
        `~astropy.utils.exceptions.AstropyUserWarning` is emitted if
        any of them is not already masked. The automatically masked
        values are combined (using bitwise OR) with the input ``mask``.
        If ``data`` is a `~numpy.ma.MaskedArray`, its mask will also be
        combined (using bitwise OR) with the input ``mask``.

        Parameters
        ----------
        data : 2D array_like
            The 2D image data. ``data`` can be a
            `~numpy.ma.MaskedArray` or a `~astropy.units.Quantity`.
            The image should be a background-subtracted cutout image
            containing a single source.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask, with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. If ``data`` is a `~numpy.ma.MaskedArray`, its
            mask will be combined (using bitwise OR) with the input
            ``mask``. Masked data are excluded from calculations.

        Returns
        -------
        centroid : `~numpy.ndarray`
            The ``(x, y)`` coordinates of the centroid.

        Notes
        -----
        Unlike `~photutils.centroids.centroid_1dg` and
        `~photutils.centroids.centroid_2dg`, this method does not
        support an error array.

        See `~photutils.centroids.centroid_quadratic` for the cases
        in which a warning is emitted and NaN values or the position of
        the maximum pixel are returned.
        """
        kwargs = {'mask': mask,
                  'fit_boxsize': self.fit_boxsize,
                  }
        return centroid_quadratic(data, **kwargs)


@deprecated_positional_kwargs(since='3.0', until='4.0')
def centroid_sources(data, xpos, ypos, box_size=11, footprint=None,
                     mask=None, centroid_func=centroid_com, **kwargs):
    """
    Calculate the centroid of sources at the defined positions in a 2D
    array using a specified centroid function.

    A cutout image centered on each input position will be used to
    calculate the centroid position. The cutout image is defined either
    using the ``box_size`` or ``footprint`` keyword. The ``footprint``
    keyword can be used to create a non-rectangular cutout image.

    Masks and non-finite values are handled by the input
    ``centroid_func``. When using a centroid function provided by
    Photutils, non-finite values (e.g., NaN or inf) in the ``data``
    array are automatically masked. The `centroid_1dg` and
    `centroid_2dg` functions also automatically mask any pixels with
    non-finite ``error`` array values. The final mask is a logical OR
    combination of the input ``mask``, the automatically generated
    mask(s) for non-finite values, and the mask of the input ``data`` if
    it is a `~numpy.ma.MaskedArray`. The centroid is calculated using
    only the unmasked data values.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image data. ``data`` can be a `~numpy.ma.MaskedArray`.
        The image should be background-subtracted.

    xpos, ypos : float or array_like of float
        The initial ``x`` and ``y`` pixel position(s) of the sources. A
        cutout image centered on each position will be used to calculate
        the centroid. The positions must be finite and within ``data``.

    box_size : int or array_like of int, optional
        The size of the cutout image along each axis. If ``box_size`` is
        a number, then a square cutout of ``box_size`` will be created.
        If ``box_size`` has two elements, they must be in ``(ny, nx)``
        order. ``box_size`` must have odd values for both axes. Either
        ``box_size`` or ``footprint`` must be defined. If they are both
        defined, then ``footprint`` overrides ``box_size``.

    footprint : bool `~numpy.ndarray`, optional
        A 2D boolean array where `True` values describe the local
        footprint region to cut out. ``footprint`` can be used to create
        a non-rectangular cutout image, in which case the input ``xpos``
        and ``ypos`` represent the center of the ``footprint`` array.
        ``box_size=(n, m)`` is equivalent to ``footprint=np.ones((n,
        m))``. Either ``box_size`` or ``footprint`` must be defined.
        If they are both defined, then ``footprint`` overrides
        ``box_size``. The same ``footprint`` is used for all sources.

    mask : 2D bool `~numpy.ndarray`, optional
        A 2D boolean array with the same shape as ``data``, where a
        `True` value indicates the corresponding element of ``data`` is
        masked. If ``data`` is a `~numpy.ma.MaskedArray`, its mask will
        be combined (using bitwise OR) with the input ``mask``.

    centroid_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The ``centroid_func``
        must accept a 2D `~numpy.ndarray`, have a ``mask`` keyword and
        optionally an ``error`` keyword. A callable whose signature
        accepts arbitrary keyword arguments (``**kwargs``) is assumed to
        handle a ``mask`` keyword. The callable object must return two
        scalar values representing the (x, y) centroid. The default is
        `~photutils.centroids.centroid_com`.

    **kwargs : dict, optional
        Any additional keyword arguments accepted by the
        ``centroid_func``. A `TypeError` is raised for keyword arguments
        not accepted by the ``centroid_func``. An ``error`` keyword
        must be an array with the same shape as ``data``. The cutout
        of it for each source is passed to the ``centroid_func``.
        ``error=None`` is ignored.

    Returns
    -------
    xcentroid, ycentroid : `~numpy.ndarray`
        The 1D arrays of the ``x`` and ``y`` pixel positions
        of the centroids. If the ``centroid_func`` raises
        a `ValueError` or `TypeError` for a source, an
        `~astropy.utils.exceptions.AstropyUserWarning` is emitted
        and NaN is returned for that source. This is usually due to
        a ``box_size`` that is too small when using a fitting-based
        centroid function (e.g., `centroid_1dg`, `centroid_2dg`, or
        `centroid_quadratic`). A `ValueError` is raised if the cutout of
        a source is completely masked.

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
    >>> print(x)
    [ 24.96807828  89.98684636 149.96545721 160.18810915]
    >>> print(y)
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
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(data, origin='lower')
        ax.scatter(x, y, marker='+', s=80, color='red', label='Centroids')
        ax.legend()
        fig.tight_layout()
    """
    if np.ndim(data) != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    xpos = np.atleast_1d(xpos)
    ypos = np.atleast_1d(ypos)
    if xpos.ndim != 1:
        msg = 'xpos must be a 1D array'
        raise ValueError(msg)
    if ypos.ndim != 1:
        msg = 'ypos must be a 1D array'
        raise ValueError(msg)
    if len(xpos) != len(ypos):
        msg = 'xpos and ypos must have the same length'
        raise ValueError(msg)

    if not (np.all(np.isfinite(xpos)) and np.all(np.isfinite(ypos))):
        msg = 'xpos and ypos must contain only finite values'
        raise ValueError(msg)

    if (xpos.min() < 0 or ypos.min() < 0
            or xpos.max() > data.shape[1] - 1
            or ypos.max() > data.shape[0] - 1):
        msg = 'xpos, ypos values contain points outside the input data'
        raise ValueError(msg)

    if footprint is None:
        if box_size is None:
            msg = 'box_size or footprint must be defined'
            raise ValueError(msg)
        box_size = as_pair('box_size', box_size, lower_bound=(0, 0),
                           check_odd=True)
        footprint = np.ones(box_size, dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)
        if footprint.ndim != 2:
            msg = 'footprint must be a 2D array'
            raise ValueError(msg)
        if not np.any(footprint):
            msg = 'footprint must contain at least one True value'
            raise ValueError(msg)

    if mask is not None and mask.shape != data.shape:
        msg = 'mask and data must have the same shape'
        raise ValueError(msg)

    # Setting error to None is equivalent to no error array, so allow it
    # even for centroid functions that do not accept an error keyword
    if kwargs.get('error') is None:
        kwargs.pop('error', None)

    # Allow arbitrary keyword arguments (**kwargs)
    spec = inspect.signature(centroid_func)
    accepts_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD
                              for param in spec.parameters.values())
    if 'mask' not in spec.parameters and not accepts_var_keyword:
        msg = "The input 'centroid_func' must have a 'mask' keyword."
        raise ValueError(msg)

    if not accepts_var_keyword:
        unknown_keys = set(kwargs) - set(spec.parameters)
        if unknown_keys:
            msg = ('Unrecognized keyword argument(s) for the input '
                   f"'centroid_func': {sorted(unknown_keys)}")
            raise TypeError(msg)
    centroid_kwargs = dict(kwargs)

    # Save the original error array so that each source independently
    # slices the full-image array
    error_array = centroid_kwargs.pop('error', None)
    if error_array is not None and np.shape(error_array) != data.shape:
        msg = 'error and data must have the same shape'
        raise ValueError(msg)

    # Extract xpeak/ypeak so the original absolute coordinates are
    # available for every source. The per-source function below re-adds
    # them with the correct cutout offset.
    # Remove this block once xpeak and ypeak are fully deprecated.
    xpeak_orig = centroid_kwargs.pop('xpeak', None)
    ypeak_orig = centroid_kwargs.pop('ypeak', None)

    inverted_footprint = np.logical_not(footprint)

    def _centroid_source(xypos):
        """
        Compute the centroid of the source at the given (x, y)
        position.
        """
        xp, yp = xypos
        slices_large, slices_small = overlap_slices(data.shape,
                                                    footprint.shape, (yp, xp))
        data_cutout = data[slices_large]

        # Trim footprint mask if it has only partial overlap on the data
        footprint_mask = inverted_footprint[slices_small]

        if mask is not None:
            # Combine the input mask cutout and footprint mask
            mask_cutout = np.logical_or(mask[slices_large], footprint_mask)
        else:
            mask_cutout = footprint_mask

        if np.all(mask_cutout):
            msg = (f'The cutout for the source at ({xp}, {yp}) is completely '
                   'masked. Please check your input mask and footprint. '
                   'Also note that footprint must be a small, local '
                   'footprint.')
            raise ValueError(msg)

        # Build the per-source keyword arguments from a local copy so
        # that no shared state is mutated across sources
        src_kwargs = dict(centroid_kwargs)
        src_kwargs['mask'] = mask_cutout

        if error_array is not None:
            src_kwargs['error'] = error_array[slices_large]

        # Add xpeak/ypeak with the offset relative to this source's
        # cutout.
        # Remove this block once xpeak and ypeak are fully deprecated.
        if xpeak_orig is not None and ypeak_orig is not None:
            src_kwargs['xpeak'] = xpeak_orig - slices_large[1].start
            src_kwargs['ypeak'] = ypeak_orig - slices_large[0].start

        try:
            xcen, ycen = centroid_func(data_cutout, **src_kwargs)
        except (ValueError, TypeError) as exc:
            msg = f'Centroid failed for source at ({xp}, {yp}): {exc}'
            warnings.warn(msg, AstropyUserWarning)
            xcen, ycen = np.nan, np.nan

        return (xcen + slices_large[1].start,
                ycen + slices_large[0].start)

    results = [_centroid_source(xypos)
               for xypos in zip(xpos, ypos, strict=True)]
    results = np.array(results, dtype=float)
    return results[:, 0], results[:, 1]
