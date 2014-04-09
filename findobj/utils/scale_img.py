
import numpy as np
from astropy.stats import sigma_clip

DTYPE_RANGE = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
DTYPE_RANGE.update((d.__name__, limits) for d, limits in DTYPE_RANGE.items())
DTYPE_RANGE.update({'uint10': (0, 2**10 - 1), 'uint12': (0, 2**12 - 1),
                    'uint14': (0, 2**14 - 1), 'bool': DTYPE_RANGE[np.bool_],
                    'float': DTYPE_RANGE[np.float64]})


def find_imgcuts(img, mincut=None, maxcut=None, minper=None, maxper=None,
                 per=None):
    """
    Find image minimum and maximum cut levels from percentiles.

    Parameters
    ----------
    img :
    mincut :
    maxcut :
    minper :
    maxper :
    per :
    """

    if mincut is not None and maxcut is not None:
        return mincut, maxcut
    if per:
        assert (per >= 0) and (per <= 100.0), 'per must be >=0 and <= 100.0'
        if not minper and not maxper:
            minper = (100.0 - float(per)) / 2.0
            maxper = 100.0 - minper
    if mincut is None:
        if minper is None:
            minper = 0.0
        assert minper >= 0, 'minper must be >= 0'
        mincut = np.percentile(img, minper)
    if maxcut is None:
        if maxper is None:
            maxper = 100.0
        assert maxper <= 100.0, 'maxper must be <= 100.0'
        maxcut = np.percentile(img, maxper)
    assert mincut <= maxcut, 'mincut must be <= maxcut'
    return mincut, maxcut


def imgstats(img, sig=3.0, iters=None):
    """
    Perform sigma-clipped statistics on the image.
    """

    idx = (img != 0.0).nonzero()    # remove padding zeros (crude)
    img_clip = sigma_clip(img[idx], sig=sig, iters=iters)
    vals = img_clip.data[~img_clip.mask]    # only good values
    return np.mean(vals), np.median(vals), np.std(vals)


def rescale_img(img, mincut=None, maxcut=None, minper=None, maxper=None,
                per=None):
    """
    Rescale image to values between 0 and 1, inclusive.
    """

    img = img.astype(np.float64)
    mincut, maxcut = find_imgcuts(img, mincut=mincut, maxcut=maxcut,
                                  minper=minper, maxper=maxper, per=per)
    outimg = rescale_intensity(img, in_range=(mincut, maxcut),
                               out_range=(0, 1))
    return outimg, mincut, maxcut


def scale_linear(img, mincut=None, maxcut=None, minper=None, maxper=None,
                 per=None):
    """
    Perform linear scaling of image.
    """

    outimg, mincut, maxcut = rescale_img(img, mincut=mincut, maxcut=maxcut,
                                         minper=minper, maxper=maxper,
                                         per=per)
    return outimg


def scale_sqrt(img, mincut=None, maxcut=None, minper=None, maxper=None,
               per=None):
    """
    Perform sqrt scaling of image.
    """

    outimg, mincut, maxcut = rescale_img(img, mincut=mincut, maxcut=maxcut,
                                         minper=minper, maxper=maxper,
                                         per=per)
    return np.sqrt(outimg)


def scale_power(img, power, mincut=None, maxcut=None, minper=None,
                maxper=None, per=None):
    """
    Perform power scaling of image.
    """

    outimg, mincut, maxcut = rescale_img(img, mincut=mincut, maxcut=maxcut,
                                         minper=minper, maxper=maxper,
                                         per=per)
    return (outimg)**power


def scale_log(img, mincut=None, maxcut=None, minper=None, maxper=None,
              per=None):
    """
    Perform log scaling of image.
    """

    outimg, mincut, maxcut = rescale_img(img, mincut=mincut, maxcut=maxcut,
                                         minper=minper, maxper=maxper,
                                         per=per)
    outimg = np.log10(outimg + 1.0) / np.log10(2.0)
    return outimg


def scale_asinh(img, noiselevel=None, sigma=2.0, mincut=None, maxcut=None,
                minper=None, maxper=None, per=None):
    """
    Perform asinh scaling of image.
    """
    outimg, mincut, maxcut = rescale_img(img, mincut=mincut, maxcut=maxcut,
                                         minper=minper, maxper=maxper,
                                         per=per)
    if not noiselevel:
        mean, median, std = imgstats(img)
        noiselevel = mean + (sigma * std)
        print noiselevel
    z = (noiselevel - mincut) / (maxcut - mincut)
    outimg = np.arcsinh(outimg / z) / np.arcsinh(1.0 / z)
    # identical, alternate derivation:
    # outimg2= np.arcsinh((img - mincut) / (noiselevel - mincut))
    # in_range = (np.min(outimg2), np.max(outimg2))
    # outimg2 = rescale_intensity(outimg2, in_range=in_range,
    #                             out_range=(0, 1))
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

