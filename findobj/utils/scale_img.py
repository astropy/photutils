# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.stats import sigma_clip

# from scikit-image
DTYPE_RANGE = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
DTYPE_RANGE.update((d.__name__, limits) for d, limits in list(DTYPE_RANGE.items()))
DTYPE_RANGE.update({'uint10': (0, 2**10 - 1), 'uint12': (0, 2**12 - 1),
                    'uint14': (0, 2**14 - 1), 'bool': DTYPE_RANGE[np.bool_],
                    'float': DTYPE_RANGE[np.float64]})


def find_imgcuts(image, min_cut=None, max_cut=None, min_percent=None,
                 max_percent=None, percent=None):
    """
    Find minimum and maximum image cut levels from percentiles of the
    image values.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    min_cut : float, optional
        The minimum cut level.  Data values less than `min_cut` will set to
        `min_cut` before scaling the image.

    max_cut : float, optional
        The maximum cut level.  Data values greater than `max_cut` will set
        to `max_cut` before scaling the image.

    min_percent : float, optional
        The minimum cut level as a percentile of the values in the image.
        If `min_cut` is input, then `min_percent` will be ignored.

    max_percent : float, optional
        The maximum cut level as a percentile of the values in the image.
        If `max_cut` is input, then `max_percent` will be ignored.

    percent : float, optional
        The percentage of the image values to scale.  The lower image cut
        level will set at the `(100 - percent) / 2` percentile, while the
        upper cut level will be set at the `(100 + percent) / 2` percentile.
        This value overrides the values of  `min_percent` and `max_percent`.
    """

    if min_cut is not None and max_cut is not None:
        return min_cut, max_cut
    if percent:
        assert (percent >= 0) and (percent <= 100.0), 'percent must be >= 0 and <= 100.0'
        if not min_percent and not max_percent:
            min_percent = (100.0 - float(percent)) / 2.0
            max_percent = 100.0 - min_percent
    if min_cut is None:
        if min_percent is None:
            min_percent = 0.0
        assert min_percent >= 0, 'min_percent must be >= 0'
        min_cut = np.percentile(image, min_percent)
    if max_cut is None:
        if max_percent is None:
            max_percent = 100.0
        assert max_percent <= 100.0, 'max_percent must be <= 100.0'
        max_cut = np.percentile(image, max_percent)
    assert min_cut <= max_cut, 'min_cut must be <= max_cut'
    return min_cut, max_cut


def _imgstats(image, image_mask=None, mask_val=None, sig=3.0, iters=None):
    """
    Perform sigma-clipped statistics on an image.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    image_mask : boolean, array_like, optional
        The 2D array of image mask values.  Bad pixels have a value of
        `True` and bad pixels have a value of `False.`

    mask_val : float, optional
        An image data value (e.g. 0.0) that is ignored when computing
        the image statistics (crude!).  `mask_val` is ignored if
        `image_mask` is input.

    sig : float, optional
        The number of standard deviations to use as the clipping limit.

    iters : float, optional
       The number of iterations to perform clipping, or `None` to clip
       until convergence is achieved (i.e. continue until the last
       iteration clips nothing).
    """

    if image_mask:
        image = image[~image_mask]
    if mask_val and not image_mask:
        idx = (image != mask_val).nonzero()
        image = image[idx]
    image_clip = sigma_clip(image, sig=sig, iters=iters)
    goodvals = image_clip.data[~image_clip.mask]
    return np.mean(goodvals), np.median(goodvals), np.std(goodvals)


def rescale_img(image, min_cut=None, max_cut=None, min_percent=None,
                max_percent=None, percent=None):
    """
    Rescale image values between minimum and maximum cut levels to
    values between 0 and 1, inclusive.
    """

    image = image.astype(np.float64)
    min_cut, max_cut = find_imgcuts(image, min_cut=min_cut, max_cut=max_cut,
                                    min_percent=min_percent,
                                    max_percent=max_percent, percent=percent)
    outimg = rescale_intensity(image, in_range=(min_cut, max_cut),
                               out_range=(0, 1))
    return outimg, min_cut, max_cut


def scale_linear(image, min_cut=None, max_cut=None, min_percent=None,
                 max_percent=None, percent=None):
    """
    Perform linear scaling of image between minimum and maximum cut levels.
    """

    result = rescale_img(image, min_cut=min_cut, max_cut=max_cut,
                         min_percent=min_percent, max_percent=max_percent,
                         percent=percent)
    return result[0]


def scale_sqrt(image, min_cut=None, max_cut=None, min_percent=None,
               max_percent=None, percent=None):
    """
    Perform square-root scaling of image between minimum and maximum
    cut levels.
    """

    result = rescale_img(image, min_cut=min_cut, max_cut=max_cut,
                         min_percent=min_percent, max_percent=max_percent,
                         percent=percent)
    return np.sqrt(result[0])


def scale_power(image, power, min_cut=None, max_cut=None, min_percent=None,
                max_percent=None, percent=None):
    """
    Perform power scaling of image between minimum and maximum cut levels.
    """

    result = rescale_img(image, min_cut=min_cut, max_cut=max_cut,
                         min_percent=min_percent, max_percent=max_percent,
                         percent=percent)
    return (result[0])**power


def scale_log(image, min_cut=None, max_cut=None, min_percent=None,
              max_percent=None, percent=None):
    """
    Perform logarithmic (log10) scaling of image between minimum and
    maximum cut levels.
    """

    result = rescale_img(image, min_cut=min_cut, max_cut=max_cut,
                         min_percent=min_percent, max_percent=max_percent,
                         percent=percent)
    outimg = np.log10(result[0] + 1.0) / np.log10(2.0)
    return outimg


def scale_asinh(image, noise_level=None, sigma=2.0, min_cut=None,
                max_cut=None, min_percent=None, max_percent=None,
                percent=None):
    """
    Perform inverse hyperbolic sin (asinh) scaling of image between minimum
    and maximum cut levels.
    """

    result = rescale_img(image, min_cut=min_cut, max_cut=max_cut,
                         min_percent=min_percent, max_percent=max_percent,
                         percent=percent)
    outimg, min_cut, max_cut = result
    if not noise_level:
        mean, median, stddev = _imgstats(outimg)
        noise_level = mean + (sigma * stddev)
    z = (noise_level - min_cut) / (max_cut - min_cut)
    outimg = np.arcsinh(outimg / z) / np.arcsinh(1.0 / z)
    return outimg


# from scikit-image
def rescale_intensity(image, in_range=None, out_range=None):
    """Return image after stretching or shrinking its intensity levels.

    The image intensities are uniformly rescaled such that the minimum and
    maximum values given by `in_range` match those given by `out_range`.

    Parameters
    ----------
    image : array
        Image array.
    in_range : 2-tuple (float, float) or str
        Min and max *allowed* intensity values of input image. If None, the
        *allowed* min/max values are set to the *actual* min/max values in the
        input image. Intensity values outside this range are clipped.
        If string, use data limits of dtype specified by the string.
    out_range : 2-tuple (float, float) or str
        Min and max intensity values of output image. If None, use the min/max
        intensities of the image data type. See `skimage.util.dtype` for
        details. If string, use data limits of dtype specified by the string.

    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.

    Examples
    --------
    By default, intensities are stretched to the limits allowed by the dtype:

    >>> image = np.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)

    It's easy to accidentally convert an image dtype from uint8 to float:

    >>> 1.0 * image
    array([  51.,  102.,  153.])

    Use `rescale_intensity` to rescale to the proper range for float dtypes:

    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([ 0. ,  0.5,  1. ])

    To maintain the low contrast of the original, use the `in_range` parameter:

    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([ 0.2,  0.4,  0.6])

    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:

    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([ 0.5,  1. ,  1. ])

    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter:

    >>> image = np.array([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0,  63, 127], dtype=int8)

    """
    dtype = image.dtype.type

    if in_range is None:
        imin = np.min(image)
        imax = np.max(image)
    elif in_range in DTYPE_RANGE:
        imin, imax = DTYPE_RANGE[in_range]
    else:
        imin, imax = in_range

    if out_range is None or out_range in DTYPE_RANGE:
        out_range = dtype if out_range is None else out_range
        omin, omax = DTYPE_RANGE[out_range]
        if imin >= 0:
            omin = 0
    else:
        omin, omax = out_range

    image = np.clip(image, imin, imax)

    image = (image - imin) / float(imax - imin)
    return dtype(image * (omax - omin) + omin)

