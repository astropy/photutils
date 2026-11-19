# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making a source mask using the Moran's I spatial
autocorrelation statistic.
"""

import math
import warnings
from functools import partial

import numpy as np
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import label, uniform_filter1d

from photutils.utils._misc import split_masked_array

__all__ = ['make_morans_i_map', 'make_morans_i_mask']

# The minimum fraction of a full patch that must be usable pixels
_MIN_PATCH_FRACTION = 0.25

# The approximate number of pixels sampled to estimate image statistics
_N_SAMPLES = 1_000_000

# Patches whose variance is below this fraction of the sum of their
# mean squared value and the mean squared value of the image are
# treated as constant. This rejects the rounding residuals of the patch
# sums in regions of constant data. The image term is needed where the
# constant is equal or close to the value subtracted from the data,
# because the mean squared value of the patch is then itself only a
# rounding residual.
_VARIANCE_FLOOR = 1.0e-10

# The mean squared value of the image is limited to this multiple of
# its robust variance. A few extreme unmasked values (e.g., hot pixels)
# would otherwise make every patch look constant.
_MAX_MEAN_SQ_RATIO = 1.0e6

# The standard deviation of the statistic for uncorrelated noise is
# about 0.5 / patch_size. Noise that is strongly correlated between
# adjacent pixels increases it by up to about a factor of 2. A larger
# ratio than this means that the background could not be measured.
_MAX_STD_RATIO = 4.0

# The ratio of the standard deviation to the median absolute deviation
# of a Gaussian distribution
_MAD_TO_STD = 1.482602218505602

# The minimum image width for which the running sum over whole rows is
# faster than filtering along the columns
_MIN_ROW_SUM_WIDTH = 256

# The approximate number of pixels in each strip of rows used to
# calculate the Moran's I map
_STRIP_PIXELS = 2**20


def _subsample(array):
    """
    Return a regularly-spaced subsample of a 2D array as a view.
    """
    step = max(1, math.ceil(math.sqrt(array.size / _N_SAMPLES)))
    return array[::step, ::step]


def _as_real_array(array):
    """
    Return the input as an array with a real numeric dtype.

    Inputs with any other dtype are converted to float64. Real numeric
    dtypes are kept, so that a large image is not copied.
    """
    array = np.asarray(array)
    if array.dtype.kind not in 'biuf':
        array = array.astype(float)
    return array


def _image_mean_sq(values):
    """
    Return the mean squared value of a sample of the image, limited to
    a multiple of its robust variance.

    The robust variance is calculated from the median absolute value
    of the nonzero values. The zero values are left out because more
    than half of the values are zero if most of the image is a constant
    region.
    """
    mean_sq = float(np.mean(values**2))
    nonzero = np.abs(values[values != 0])
    if nonzero.size == 0:
        return mean_sq
    robust_std = _MAD_TO_STD * float(np.median(nonzero))
    return min(mean_sq, _MAX_MEAN_SQ_RATIO * robust_std**2)


def _box_sum(array, size):
    """
    Return the sum of a 2D array in a square box centered on each
    pixel, with zeros outside of the array.
    """
    if array.shape[1] >= _MIN_ROW_SUM_WIDTH:
        array = _row_sum(array, size)
    else:
        array = uniform_filter1d(array, size, axis=0, mode='constant')
        array *= size
    uniform_filter1d(array, size, axis=1, mode='constant', output=array)
    array *= size
    return array


def _row_sum(array, size):
    """
    Return the sum of the ``size`` rows centered on each row of a 2D
    array, with zeros outside of the array.

    The running sum adds and subtracts whole rows, which is much faster
    than filtering along the columns of a wide C-contiguous array.
    """
    half = size // 2
    n_rows = array.shape[0]
    result = np.empty_like(array)
    running = array[:half + 1].sum(axis=0)
    result[0] = running
    for row in range(1, n_rows):
        if row + half < n_rows:
            running += array[row + half]
        if row > half:
            running -= array[row - half - 1]
        result[row] = running
    return result


def _neighbor_sum(array):
    """
    Return the sum of the eight nearest neighbors of each pixel, with
    zeros outside of the array.

    The center pixel is left out of the sum instead of being subtracted
    from it, so the sum of nonnegative values is exactly zero only if
    all of the neighbors are zero. The sums of three adjacent pixels in
    each row are shared by the rows above and below.
    """
    padded = np.pad(array, 1)
    row_sums = padded[:, :-2] + padded[:, 1:-1]
    row_sums += padded[:, 2:]
    result = row_sums[:-2] + row_sums[2:]
    result += padded[1:-1, :-2]
    result += padded[1:-1, 2:]
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
        a nonfinite or nonpositive weight are excluded, as are the
        masked pixels if ``weights`` is a `~numpy.ma.MaskedArray`. The
        statistic does not change if the weights are multiplied by a
        positive constant. If `None`, all of the pixels have the same
        weight.

    Returns
    -------
    result : 2D float32 `~numpy.ndarray`
        The Moran's I map. A pixel is NaN if it is excluded, if fewer
        than 25% of the pixels of a full patch are usable, or if the
        variance of its patch is too small to be measured (see Notes).

    See Also
    --------
    make_morans_i_mask

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

    The variance of a patch is too small to be measured if it is
    smaller than about ``1e-10`` times the mean squared deviation from
    the median of the image, of either the patch or the whole image.
    Regions of constant data (e.g., blank regions that are filled with
    zeros) therefore have no value of the statistic. Neither do regions
    whose values differ from the median of the image by more than
    about ``1e5`` times their standard deviation.

    Noise that is correlated between adjacent pixels (e.g., in
    resampled or drizzled images) has a positive Moran's I even without
    sources. The statistic should then be compared to its typical value
    over the image instead of to zero, as is done by
    `make_morans_i_mask`.

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
    data = _as_real_array(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

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

    if weights is not None:
        weights, weights_mask = split_masked_array(weights, None)
        weights = _as_real_array(weights)
        if weights.shape != data.shape:
            msg = 'weights must have the same shape as data'
            raise ValueError(msg)
        with np.errstate(invalid='ignore'):
            valid &= np.isfinite(weights) & (weights > 0)
        if weights_mask is not None:
            valid &= ~weights_mask

    result = np.empty(data.shape, dtype=np.float32)
    if data.size == 0:
        return result

    # Subtract an approximate background level to limit the rounding
    # error of the box sums. The statistic does not depend on it. The
    # mean squared value of the image sets the scale of that rounding
    # error.
    sample = _subsample(data)[_subsample(valid)].astype(float)
    pedestal = 0.0
    image_mean_sq = 0.0
    if sample.size > 0:
        pedestal = float(np.median(sample))
        image_mean_sq = _image_mean_sq(sample - pedestal)

    # The statistic of each pixel depends only on the pixels within
    # this many rows of it, so the map is calculated in strips of rows
    # that overlap by this many rows on each side. The strips limit the
    # memory used by the intermediate arrays, including the float64
    # copies of data and weights with other dtypes.
    overlap = patch_size // 2 + 1
    n_rows, n_cols = data.shape
    strip_rows = max(overlap, _STRIP_PIXELS // n_cols)
    for start in range(0, n_rows, strip_rows):
        stop = min(start + strip_rows, n_rows)
        lower = max(start - overlap, 0)
        rows = slice(lower, min(stop + overlap, n_rows))
        strip_weights = None if weights is None else weights[rows]
        strip = _morans_i_strip(data[rows], valid[rows], strip_weights,
                                pedestal=pedestal,
                                image_mean_sq=image_mean_sq,
                                patch_size=patch_size)
        result[start:stop] = strip[start - lower:stop - lower]
    return result


def _morans_i_strip(data, valid, weights, *, pedestal, image_mean_sq,
                    patch_size):
    """
    Calculate the Moran's I map of a strip of rows.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The strip of the image, with any real numeric dtype.

    valid : 2D bool `~numpy.ndarray`
        The valid pixels of the strip.

    weights : 2D `~numpy.ndarray` or `None`
        The weights of the strip, with any real numeric dtype, or
        `None` for equal weights.

    pedestal : float
        The value subtracted from the data.

    image_mean_sq : float
        The mean squared value of the whole image after ``pedestal`` is
        subtracted.

    patch_size : int
        The side length in pixels of the square patch.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The float64 Moran's I map of the strip. The rows near the
        edges of the strip are incorrect unless they are also edges of
        the image.
    """
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(float)
    if weights is None:
        wts = valid.astype(float)
    else:
        wts = np.where(valid, weights, 0.0).astype(float, copy=False)
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
        mean_sq += image_mean_sq
        mean_sq *= _VARIANCE_FLOOR
        good &= var > mean_sq
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
    return result


def _half_sample_mode(values):
    """
    Calculate the half-sample mode of a 1D array.

    The mode is found by repeatedly keeping the shortest interval that
    contains half of the values (Bickel & Fruhwirth 2006).

    Parameters
    ----------
    values : 1D `~numpy.ndarray`
        The values. Must not be empty.

    Returns
    -------
    mode : float
        The half-sample mode.
    """
    values = np.sort(values)
    while values.size > 3:
        n_half = (values.size + 1) // 2
        widths = values[n_half - 1:] - values[:values.size - n_half + 1]
        start = int(np.argmin(widths))
        values = values[start:start + n_half]
    return float(np.median(values))


def _lower_std(values, center):
    """
    Estimate the standard deviation of the values from the values that
    are not above ``center``.

    The estimate is the median deviation of those values from
    ``center``, scaled to the standard deviation of a Gaussian
    distribution. Unlike an RMS deviation, it is not affected by
    outliers (e.g., the extreme negative values of the statistic next
    to unmasked hot pixels) unless they are half of the values.

    Parameters
    ----------
    values : 1D `~numpy.ndarray`
        The values. At least one value must not be above ``center``.

    center : float
        The center value.

    Returns
    -------
    std : float
        The estimated standard deviation.
    """
    lower = values[values <= center]
    return _MAD_TO_STD * float(np.median(center - lower))


def _calc_threshold(stat, *, n_sigma, percent, patch_size):
    """
    Calculate the Moran's I threshold above which pixels are sources.

    Parameters
    ----------
    stat : 2D `~numpy.ndarray`
        The Moran's I map.

    n_sigma : float
        The number of standard deviations above the background value.

    percent : float or `None`
        The percentage of pixels to flag. If not `None`, it is used
        instead of ``n_sigma``.

    patch_size : int
        The side length in pixels of the patch used for ``stat``.

    Returns
    -------
    threshold : float or `None`
        The threshold. `None` if ``stat`` has no finite values.
    """
    sample = _subsample(stat)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None

    if percent is not None:
        return np.percentile(sample, 100.0 - percent)

    # Sources only increase the statistic, so the background values
    # form the lowest peak of its distribution. The median is a source
    # value if most of the patches contain sources, but the values
    # below the median still peak at the background value unless
    # nearly all of the patches contain sources. The mode of those
    # values is then refined with the median of the values near it,
    # which removes the bias of the mode for pure noise. The scatter
    # of the background values is measured robustly from the values
    # below the background value, because those values also include
    # the extreme negative values next to any unmasked hot pixels. The
    # center is a median of some of the values, so at least one value
    # is not above it.
    center = _half_sample_mode(sample[sample <= np.median(sample)])
    std = _lower_std(sample, center)
    for _ in range(2):
        center = float(np.median(sample[np.abs(sample - center)
                                        <= 3.0 * std]))
        std = _lower_std(sample, center)

    if std > _MAX_STD_RATIO * 0.5 / patch_size:
        msg = ("The background value of the Moran's I statistic could "
               'not be estimated, probably because nearly all of the '
               'patches contain sources or the background changes '
               'strongly across each patch. The source mask may be '
               'incomplete. Use the percent keyword or a smaller '
               'patch_size instead.')
        warnings.warn(msg, AstropyUserWarning)

    return center + n_sigma * std


def _diamond_morphology(mask, radius, *, erode):
    """
    Erode or dilate a 2D boolean mask with a diamond of the given
    radius.

    The result is the same as ``radius`` iterations of
    `scipy.ndimage.binary_erosion` (with ``border_value=1``) or
    `scipy.ndimage.binary_dilation` with their default 3x3 cross
    structuring element. The diamond is made of the horizontal segments
    of half-width ``radius - abs(offset)`` at each row offset, and the
    segments are grown one pixel at a time, which is much faster than
    the iterations.

    Parameters
    ----------
    mask : 2D bool `~numpy.ndarray`
        The input mask.

    radius : int
        The radius of the diamond.

    erode : bool
        If `True`, erode the mask, treating the pixels outside of the
        array as `True`. If `False`, dilate it, treating those pixels as
        `False`.

    Returns
    -------
    result : 2D bool `~numpy.ndarray`
        The eroded or dilated mask.
    """
    n_rows, n_cols = mask.shape
    func = np.logical_and if erode else np.logical_or
    segments = np.pad(mask, radius, constant_values=erode)
    cols = slice(radius, radius + n_cols)
    result = None
    for half_width in range(radius + 1):
        if half_width > 0:
            grown = segments.copy()
            func(grown[:, 1:], segments[:, :-1], out=grown[:, 1:])
            func(grown[:, :-1], segments[:, 1:], out=grown[:, :-1])
            segments = grown

        # Combine the segments in the rows at this offset above and
        # below each row
        offset = radius - half_width
        for start in {radius - offset, radius + offset}:
            rows = segments[start:start + n_rows, cols]
            if result is None:
                result = rows.copy()
            else:
                func(result, rows, out=result)
    return result


def _fill_holes(mask):
    """
    Fill the holes in a 2D boolean mask.

    The holes are the connected regions of `False` values that do not
    touch the edges of the array, with the same connectivity as
    `scipy.ndimage.binary_fill_holes`. Labeling the regions is much
    faster than the iterative dilation of that function.
    """
    labels, n_labels = label(~mask)
    keep = np.zeros(n_labels + 1, dtype=bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        keep[edge] = True

    # The True values of the input mask have the label 0
    keep[0] = False
    return ~keep[labels]


def make_morans_i_mask(data, *, patch_size=21, n_sigma=3.0, percent=None,
                       mask=None, weights=None, opening_iterations=None):
    """
    Make a source mask using the Moran's I spatial autocorrelation
    statistic.

    Pixels are flagged as sources where the Moran's I statistic of the
    surrounding patch of pixels (see `make_morans_i_map`) is above a
    threshold. Holes in the flagged regions are then filled and small
    isolated regions are removed.

    The mask can be made before the background is known because the
    statistic does not depend on the background level or on the
    amplitude of the background noise. It is intended to be used as the
    ``mask`` input to `~photutils.background.Background2D`.

    Parameters
    ----------
    data : 2D array_like or `~astropy.units.Quantity`
        The 2D image. Nonfinite values (NaN and inf) are excluded. If
        ``data`` is a `~numpy.ma.MaskedArray`, its mask is combined
        with the input ``mask``.

    patch_size : int, optional
        The side length in pixels of the square patch used to calculate
        the statistic. It must be an odd integer larger than 3. Sources
        are masked out to a distance of approximately half the patch
        size beyond where they are detected.

    n_sigma : float, optional
        The number of standard deviations above the background value of
        the statistic at which pixels are flagged as sources. The
        background value and its standard deviation are estimated from
        the statistic map itself, so the threshold adapts to noise that
        is correlated between pixels. The background value is taken
        from the lowest peak of the distribution of the statistic, so
        it can be estimated even if most of the patches contain
        sources. A warning is raised if nearly all of the patches
        contain sources, because the background value then cannot be
        estimated. Ignored if ``percent`` is input.

    percent : float or `None`, optional
        The percentage of the pixels with a value of the statistic to
        flag as sources, selecting the pixels with the largest values.
        It must be larger than 0 and smaller than 100. If input, it is
        used instead of ``n_sigma``. This percentage of the pixels is
        flagged even if the image contains no sources. The final
        masked fraction differs somewhat because holes in the flagged
        regions are then filled and the binary opening removes small
        regions.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded from the statistic
        (e.g., bad pixels). These pixels are not flagged as sources
        unless they are enclosed by flagged pixels, so this mask
        should be combined with the returned source mask (e.g.,
        ``source_mask | mask``) when it is input to
        `~photutils.background.Background2D`. Bad pixels with
        extreme values (e.g., hot pixels) should be excluded, because
        such a pixel dominates the statistic of every patch that
        contains it, which hides the sources near it.

    weights : 2D array_like, optional
        The weights of the pixels used to calculate the statistic,
        with the same shape as ``data`` (e.g., an inverse-variance
        map). Pixels with a nonfinite or nonpositive weight are
        excluded, as are the masked pixels if ``weights`` is a
        `~numpy.ma.MaskedArray`. If `None`, all of the pixels have the
        same weight.

    opening_iterations : int or `None`, optional
        The number of iterations of the binary opening that is applied
        to the mask. The opening removes flagged regions that are
        narrower than about twice this number of pixels and rounds the
        corners of the remaining regions, without enlarging them.
        Regions are not eroded from the image edges or from pixels
        where the statistic is undefined (e.g., masked pixels or
        regions with too few usable pixels). If `None`,
        ``patch_size // 2 - 2`` iterations are used. A detected source
        is flagged in a region at least as wide as the patch, so the
        default keeps the sources while rounding the square shape that
        the patch gives their regions. Larger values remove compact
        sources. Set to 0 to skip this step.

    Returns
    -------
    source_mask : 2D bool `~numpy.ndarray`
        The source mask, where `True` values indicate source pixels.

    See Also
    --------
    make_morans_i_map
    photutils.segmentation.SegmentationImage.make_source_mask

    Examples
    --------
    >>> from photutils.background import Background2D, make_morans_i_mask
    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()
    >>> mask = make_morans_i_mask(data)
    >>> bkg = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3),
    ...                    exclude_percentile=50.0)

    The source mask covers a large fraction of the image, so
    ``exclude_percentile`` is increased to keep the boxes that are
    partly covered by the mask.

    Excluded pixels are not included in the returned source mask, so a
    bad-pixel mask must be combined with the source mask when it is
    input to `~photutils.background.Background2D`.

    >>> import numpy as np
    >>> bad_mask = np.zeros(data.shape, dtype=bool)
    >>> bad_mask[:, 300] = True
    >>> source_mask = make_morans_i_mask(data, mask=bad_mask)
    >>> bkg = Background2D(data, (50, 50), mask=source_mask | bad_mask,
    ...                    filter_size=(3, 3), exclude_percentile=50.0)
    """
    if not n_sigma > 0:
        msg = 'n_sigma must be a positive number'
        raise ValueError(msg)
    if percent is not None and not 0 < percent < 100:
        msg = 'percent must be between 0 and 100 (exclusive)'
        raise ValueError(msg)
    if (opening_iterations is not None
            and (isinstance(opening_iterations, bool)
                 or not isinstance(opening_iterations, (int, np.integer))
                 or opening_iterations < 0)):
        msg = 'opening_iterations must be a nonnegative integer'
        raise ValueError(msg)

    stat = make_morans_i_map(data, patch_size=patch_size, mask=mask,
                             weights=weights)
    threshold = _calc_threshold(stat, n_sigma=n_sigma, percent=percent,
                                patch_size=patch_size)
    if threshold is None:
        return np.zeros(stat.shape, dtype=bool)
    if opening_iterations is None:
        opening_iterations = patch_size // 2 - 2

    source_mask = _fill_holes(stat > threshold)
    if opening_iterations > 0:
        # Pixels outside of the image and pixels without a value of the
        # statistic (e.g., masked regions) are taken as flagged for the
        # erosion, so that the regions of sources next to them are not
        # eroded from those sides. Those pixels are then removed again,
        # and the result is limited to the flagged pixels.
        excluded = np.isnan(stat)
        eroded = _diamond_morphology(source_mask | excluded,
                                     opening_iterations, erode=True)
        eroded &= ~excluded
        source_mask &= _diamond_morphology(eroded, opening_iterations,
                                           erode=False)
    return source_mask
