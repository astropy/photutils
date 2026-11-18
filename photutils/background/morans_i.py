# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making a source mask using the Moran's I spatial
autocorrelation statistic.
"""

import math
from functools import partial

import numpy as np
from astropy.units import Quantity
from scipy.ndimage import uniform_filter1d

from photutils.utils._misc import split_masked_array

__all__ = ['make_morans_i_map']

# The minimum fraction of a full patch that must be usable pixels
_MIN_PATCH_FRACTION = 0.25

# The approximate number of pixels sampled to estimate image statistics
_N_SAMPLES = 1_000_000

# Patches whose variance is below this fraction of their mean squared
# value are treated as constant. This rejects rounding residuals in
# regions of constant data.
_VARIANCE_FLOOR = 1.0e-10


def _subsample(array):
    """
    Return a regularly-spaced subsample of a 2D array as a view.
    """
    step = max(1, math.ceil(math.sqrt(array.size / _N_SAMPLES)))
    return array[::step, ::step]


def _box_sum(array, size):
    """
    Return the sum of a 2D array in a square box centered on each
    pixel, with zeros outside of the array.
    """
    array = uniform_filter1d(array, size, axis=0, mode='constant')
    array *= size
    uniform_filter1d(array, size, axis=1, mode='constant', output=array)
    array *= size
    return array


def _neighbor_sum(array):
    """
    Return the sum of the eight nearest neighbors of each pixel, with
    zeros outside of the array.

    The center pixel is left out of the sum instead of being subtracted
    from it, so the sum of nonnegative values is exactly zero only if
    all of the neighbors are zero.
    """
    ny, nx = array.shape
    padded = np.pad(array, 1)
    result = np.zeros(array.shape)
    for dy in range(3):
        for dx in range(3):
            if (dy, dx) != (1, 1):
                result += padded[dy:dy + ny, dx:dx + nx]
    return result


def make_morans_i_map(data, *, patch_size=21, mask=None, weights=None):
    """
    Calculate a map of the Moran's I spatial autocorrelation statistic.

    The value at each pixel is the global `Moran's I
    <https://en.wikipedia.org/wiki/Moran%27s_I>`_ statistic of the
    square patch of pixels centered on that pixel. The statistic
    measures how strongly each pixel value is correlated with the
    values of its eight nearest neighbors. Patches that contain only
    uncorrelated background noise have values close to zero, while
    patches that contain sources have large positive values.

    The statistic does not change if a constant is added to the data or
    if the data are multiplied by a positive constant. It therefore
    does not require the background level or the background noise to be
    known.

    Parameters
    ----------
    data : 2D array_like or `~astropy.units.Quantity`
        The 2D image. Nonfinite values (NaN and inf) are excluded. If
        ``data`` is a `~numpy.ma.MaskedArray`, its mask is combined
        with the input ``mask``.

    patch_size : int, optional
        The side length in pixels of the square patch. It must be an
        odd integer larger than 3. Small patches give a noisier
        statistic, while large patches lose spatial resolution. The
        standard deviation of the statistic for uncorrelated noise is
        approximately ``0.5 / patch_size``.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded (e.g., bad pixels).

    weights : 2D array_like, optional
        The weights of the pixels, with the same shape as ``data``
        (e.g., an inverse-variance map). The patch means, the patch
        variance, and the neighbor means are then weighted. Pixels with
        a nonfinite or nonpositive weight are excluded. The statistic
        does not change if the weights are multiplied by a positive
        constant. If `None`, all of the pixels have the same weight.

    Returns
    -------
    result : 2D float32 `~numpy.ndarray`
        The Moran's I map. A pixel is NaN if it is excluded, if fewer
        than 25% of the pixels of a full patch are usable, or if its
        patch has zero variance.

    Notes
    -----
    For a patch with pixel values :math:`x_i` and pixel weights
    :math:`w_i`, the statistic is

    .. math::

        I = \\frac{\\sum_i w_i (x_i - \\bar{x}) (n_i - \\bar{n})}
            {\\sum_i w_i (x_i - \\bar{x})^2}

    where :math:`n_i` is the weighted mean of the usable pixels among
    the eight nearest neighbors of pixel :math:`i`, and :math:`\\bar{x}`
    and :math:`\\bar{n}` are the means of :math:`x_i` and :math:`n_i`
    over the patch, weighted by :math:`w_i`. This is the global Moran's
    I statistic with row-standardized weights. The neighbor means use
    all of the image pixels, including those just outside of the patch.
    Patches are truncated at the image edges.

    Noise that is correlated between adjacent pixels (e.g., in
    resampled or drizzled images) has a positive Moran's I even without
    sources. The statistic should then be compared to its typical value
    over the image instead of to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.background import make_morans_i_map
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(5.0, 1.0, (101, 101))
    >>> data[45:56, 45:56] += 5.0
    >>> result = make_morans_i_map(data)
    >>> print(f'{result[50, 50]:.2f}')
    0.68
    >>> print(f'{result[10, 10]:.2f}')
    -0.01
    """
    data, mask = split_masked_array(data, mask)
    if isinstance(data, Quantity):
        data = data.value
    data = np.asanyarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(float)

    if (isinstance(patch_size, bool)
            or not isinstance(patch_size, (int, np.integer))
            or patch_size <= 3 or patch_size % 2 == 0):
        msg = 'patch_size must be an odd integer larger than 3'
        raise ValueError(msg)

    valid = np.isfinite(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)
        valid &= ~mask

    if weights is None:
        wts = valid.astype(float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != data.shape:
            msg = 'weights must have the same shape as data'
            raise ValueError(msg)
        with np.errstate(invalid='ignore'):
            valid &= np.isfinite(weights) & (weights > 0)
        wts = np.where(valid, weights, 0.0)

    # Subtract an approximate background level to limit the rounding
    # error of the box sums. The statistic does not depend on it.
    sample = _subsample(data)[_subsample(valid)]
    pedestal = float(np.median(sample)) if sample.size > 0 else 0.0
    values = np.zeros(data.shape)
    values[valid] = data[valid] - pedestal

    # The weighted mean of the valid pixels among the eight nearest
    # neighbors
    neighbor_wts = _neighbor_sum(wts)
    usable = valid & (neighbor_wts > 0)
    wtd_values = wts * values
    with np.errstate(invalid='ignore', divide='ignore'):
        neighbors = _neighbor_sum(wtd_values) / neighbor_wts
    del neighbor_wts

    # Only the usable pixels contribute to the patch sums. The arrays
    # are modified in place and deleted as soon as they are no longer
    # needed to limit the memory use.
    not_usable = ~usable
    neighbors[not_usable] = 0.0
    wts[not_usable] = 0.0
    wtd_values[not_usable] = 0.0
    del not_usable
    wtd_sq_values = np.multiply(values, wtd_values, out=values)
    del values

    patch_sum = partial(_box_sum, size=patch_size)
    min_count = _MIN_PATCH_FRACTION * patch_size**2
    sum_wts = patch_sum(wts)
    # Without input weights, the weights are 1 for the usable pixels
    # and 0 otherwise, so their patch sum is the number of usable pixels
    count = sum_wts if weights is None else patch_sum(usable.astype(float))
    good = usable & (np.rint(count) >= min_count)
    del usable, count

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = patch_sum(wtd_values)
        mean /= sum_wts
        mean_sq = patch_sum(wtd_sq_values)
        del wtd_sq_values
        mean_sq /= sum_wts
        var = mean_sq - mean**2
        good &= var > _VARIANCE_FLOOR * mean_sq
        del mean_sq

        result = patch_sum(wtd_values * neighbors)
        del wtd_values
        result /= sum_wts
        mean_neighbors = patch_sum(wts * neighbors)
        del wts, neighbors
        mean_neighbors /= sum_wts
        mean_neighbors *= mean
        result -= mean_neighbors
        result /= var

    result[~good] = np.nan
    return result.astype(np.float32)
