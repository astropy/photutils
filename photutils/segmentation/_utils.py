# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides miscellaneous segmentation utilities.
"""

import numpy as np


def mask_to_mirrored_value(data, mask, xycenter):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given center position.

    If the mirror pixel is unavailable (i.e., it is outside of the image
    or masked), then the masked pixel value is set to zero.

    Parameters
    ----------
    data : `numpy.ndarray`, 2D
        A 2D array.

    mask : array-like, bool
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is
        considered bad.

    xycenter : tuple of two int
        The (x, y) center coordinates around which masked pixels will be
        mirrored.

    Returns
    -------
    result : `numpy.ndarray`, 2D
        A 2D array with replaced masked pixels.
    """
    outdata = np.copy(data)

    ymasked, xmasked = np.nonzero(mask)
    xmirror = 2 * int(xycenter[0] + 0.5) - xmasked
    ymirror = 2 * int(xycenter[1] + 0.5) - ymasked

    # Set mirrored pixels that go out of the image to zero.
    badmask = ((xmirror < 0) | (ymirror < 0) | (xmirror >= data.shape[1])
               | (ymirror >= data.shape[0]))
    if np.any(badmask):
        outdata[ymasked[badmask], xmasked[badmask]] = 0.
        goodmask = ~badmask
        ymasked = ymasked[goodmask]
        xmasked = xmasked[goodmask]
        xmirror = xmirror[goodmask]
        ymirror = ymirror[goodmask]

    outdata[ymasked, xmasked] = outdata[ymirror, xmirror]

    # Set pixels that are mirrored to another masked pixel to zero.
    mirror_mask = mask[ymirror, xmirror]
    xbad = xmasked[mirror_mask]
    ybad = ymasked[mirror_mask]
    outdata[ybad, xbad] = 0.0

    return outdata
