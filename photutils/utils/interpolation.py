# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['interpolate_masked_data']


def interpolate_masked_data(data, mask, error=None, background=None):
    """
    Interpolate over masked pixels in data and optional error or
    background images.

    The value of masked pixels are replaced by the mean value of the
    8-connected neighboring non-masked pixels.

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
