# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['interpolate_masked_data', 'mask_to_mirrored_num']


def interpolate_masked_data(data, mask, error=None, background=None):
    """
    Interpolate over masked pixels in data and optional error or
    background images.

    The value of masked pixels are replaced by the mean value of the
    8-connected neighboring non-masked pixels.  This function is intended
    for single, isolated masked pixels (e.g. hot/warm pixels).

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D data array.

    mask : array_like (bool)
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` must have the same shape as ``data``.

    background : array_like, or `~astropy.units.Quantity`, optional
        The pixel-wise background level of the input ``data``.
        ``background`` must have the same shape as ``data``.

    Returns
    -------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``data`` with interpolated masked pixels.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``error`` with interpolated masked pixels.  `None` if
        input ``error`` is not input.

    background : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``background`` with interpolated masked pixels.  `None` if
        input ``background`` is not input.
    """

    if data.shape != mask.shape:
        raise ValueError('data and mask must have the same shape')

    data_out = np.copy(data)    # do not alter input data
    mask_idx = mask.nonzero()
    for j, i in zip(*mask_idx):
        y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
        x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
        goodpix = ~mask[y0:y1, x0:x1]
        if not np.any(goodpix):
            warnings.warn('The masked pixel at "({0}, {1})" is completely '
                          'surrounded by (8-connected) masked pixels, '
                          'thus unable to interpolate'.format(i, j),
                          AstropyUserWarning)
            continue
        data_out[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])

        if background is not None:
            if background.shape != data.shape:
                raise ValueError('background and data must have the same '
                                 'shape')
            background_out = np.copy(background)
            background_out[j, i] = np.mean(background[y0:y1, x0:x1][goodpix])
        else:
            background_out = None

        if error is not None:
            if error.shape != data.shape:
                raise ValueError('error and data must have the same '
                                 'shape')
            error_out = np.copy(error)
            error_out[j, i] = np.sqrt(
                np.mean(error[y0:y1, x0:x1][goodpix]**2))
        else:
            error_out = None

    return data_out, error_out, background_out


def mask_to_mirrored_num(image, mask_image, center_position, bbox=None):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given ``center_position``.  If the mirror pixel is unavailable (i.e.
    itself masked or outside of the image), then the masked pixel value
    is set to zero.

    Parameters
    ----------
    image : `numpy.ndarray`, 2D
        The 2D array of the image.

    mask_image : array-like, bool
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        considered bad.

    center_position : 2-tuple
        (x, y) center coordinates around which masked pixels will be
        mirrored.

    bbox : list, tuple, `numpy.ndarray`, optional
        The bounding box (x_min, x_max, y_min, y_max) over which to
        replace masked pixels.

    Returns
    -------
    result : `numpy.ndarray`, 2D
        A 2D array with replaced masked pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.utils import mask_to_mirrored_num
    >>> image = np.arange(16).reshape(4, 4)
    >>> mask = np.zeros_like(image, dtype=bool)
    >>> mask[0, 0] = True
    >>> mask[1, 1] = True
    >>> mask_to_mirrored_num(image, mask, (1.5, 1.5))
    array([[15,  1,  2,  3],
           [ 4, 10,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    """

    if bbox is None:
        ny, nx = image.shape
        bbox = [0, nx, 0, ny]
    subdata = np.copy(image[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1])
    submask = mask_image[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1]
    y_masked, x_masked = np.nonzero(submask)
    x_mirror = (2 * (center_position[0] - bbox[0])
                - x_masked + 0.5).astype('int32')
    y_mirror = (2 * (center_position[1] - bbox[2])
                - y_masked + 0.5).astype('int32')

    # Reset mirrored pixels that go out of the image.
    outofimage = ((x_mirror < 0) | (y_mirror < 0) |
                  (x_mirror >= subdata.shape[1]) |
                  (y_mirror >= subdata.shape[0]))
    if outofimage.any():
        x_mirror[outofimage] = x_masked[outofimage].astype('int32')
        y_mirror[outofimage] = y_masked[outofimage].astype('int32')

    subdata[y_masked, x_masked] = subdata[y_mirror, x_mirror]

    # Set pixels that mirrored to another masked pixel to zero.
    # This will also set to zero any pixels that mirrored out of
    # the image.
    mirror_is_masked = submask[y_mirror, x_mirror]
    x_bad = x_masked[mirror_is_masked]
    y_bad = y_masked[mirror_is_masked]
    subdata[y_bad, x_bad] = 0.0

    outimage = np.copy(image)
    outimage[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1] = subdata
    return outimage
