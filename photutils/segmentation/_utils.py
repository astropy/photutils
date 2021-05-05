# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides miscellaneous segmentation utilities.
"""

import numpy as np


def mask_to_mirrored_value(data, replace_mask, xycenter, mask=None):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given center position.

    If the mirror pixel is unavailable (i.e., it is outside of the image
    or masked), then the masked pixel value is set to zero.

    Parameters
    ----------
    data : `numpy.ndarray`, 2D
        A 2D array.

    replace_mask : array-like, bool
        A boolean mask where `True` values indicate the pixels that
        should be replaced, if possible, by mirrored pixel values. It
        must have the same shape as ``data``.

    xycenter : tuple of two int
        The (x, y) center coordinates around which masked pixels will be
        mirrored.

    mask : array-like, bool
        A boolean mask where `True` values indicate ``replace_mask``
        *mirrored* pixels that should never be used to fix
        ``replace_mask`` pixels. In other words, if a pixel in
        ``replace_mask`` has a mirror pixel in this ``mask``, then the
        mirrored value is set to zero. Using this keyword prevents
        potential spreading of known non-finite or bad pixel values.

    Returns
    -------
    result : `numpy.ndarray`, 2D
        A 2D array with replaced masked pixels.
    """
    outdata = np.copy(data)

    ymasked, xmasked = np.nonzero(replace_mask)
    xmirror = 2 * int(xycenter[0] + 0.5) - xmasked
    ymirror = 2 * int(xycenter[1] + 0.5) - ymasked

    # Find mirrored pixels that are outside of the image
    badmask = ((xmirror < 0) | (ymirror < 0) | (xmirror >= data.shape[1])
               | (ymirror >= data.shape[0]))

    # remove them from the set of replace_mask pixels and set them to
    # zero
    if np.any(badmask):
        outdata[ymasked[badmask], xmasked[badmask]] = 0.
        # remove the badmask pixels from pixels to be replaced
        goodmask = ~badmask
        ymasked = ymasked[goodmask]
        xmasked = xmasked[goodmask]
        xmirror = xmirror[goodmask]
        ymirror = ymirror[goodmask]

    outdata[ymasked, xmasked] = outdata[ymirror, xmirror]

    # Find mirrored pixels that are masked and replace_mask pixels that are
    # mirrored to other replace_mask pixels. Set them both to zero.
    mirror_mask = replace_mask[ymirror, xmirror]
    if mask is not None:
        mirror_mask |= mask[ymirror, xmirror]
    xbad = xmasked[mirror_mask]
    ybad = ymasked[mirror_mask]
    outdata[ybad, xbad] = 0.0

    return outdata
