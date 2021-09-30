# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the base classes for detecting stars in an
astronomical image. Each star-finding class should define a method
called ``find_stars`` that finds stars in an image.
"""

import abc
import warnings

import numpy as np

from .peakfinder import find_peaks
from ..utils.exceptions import NoDetectionsWarning


__all__ = ['StarFinderBase']


class StarFinderBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for star finders.
    """

    def __call__(self, data, mask=None):
        return self.find_stars(data, mask=mask)

    @staticmethod
    def _find_stars(convolved_data, kernel, threshold, *, min_separation=0.0,
                    mask=None, exclude_border=False):
        """
        Find stars in an image.

        Parameters
        ----------
        convolved_data : 2D array_like
            The convolved 2D array.

        kernel : `_StarFinderKernel`
            The convolution kernel.

        threshold : float
            The absolute image value above which to select sources.  This
            threshold should be the threshold input to the star finder class
            multiplied by the kernel relerr.

        min_separation : float, optional
            The minimum separation for detected objects in pixels.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a `True`
            value indicates the corresponding element of ``data`` is masked.
            Masked pixels are ignored when searching for stars.

        exclude_border : bool, optional
            Set to `True` to exclude sources found within half the size of
            the convolution kernel from the image borders.  The default is
            `False`, which is the mode used by IRAF's `DAOFIND`_ and
            `starfind`_ tasks.

        Returns
        -------
        result : Nx2 `~numpy.ndarray`
            A Nx2 array containing the (x, y) pixel coordinates.

        .. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind

        .. _starfind: https://iraf.net/irafhelp.php?val=starfind
        """
        # define a local footprint for the peak finder
        if min_separation == 0:  # daofind
            if isinstance(kernel, np.ndarray):
                footprint = np.ones(kernel.shape)
            else:
                footprint = kernel.mask.astype(bool)
        else:
            # define a local circular footprint for the peak finder
            idx = np.arange(-min_separation, min_separation + 1)
            xx, yy = np.meshgrid(idx, idx)
            footprint = np.array((xx**2 + yy**2) <= min_separation**2,
                                 dtype=int)

        # pad the convolved data and mask by half the kernel size (or
        # x/y radius) to allow for detections near the edges
        if isinstance(kernel, np.ndarray):
            ypad = (kernel.shape[0] - 1) // 2
            xpad = (kernel.shape[1] - 1) // 2
        else:
            ypad = kernel.yradius
            xpad = kernel.xradius

        if not exclude_border:
            pad = ((ypad, ypad), (xpad, xpad))
            pad_mode = 'constant'
            convolved_data = np.pad(convolved_data, pad, mode=pad_mode,
                                    constant_values=0.0)
            if mask is not None:
                mask = np.pad(mask, pad, mode=pad_mode, constant_values=False)

        # find local peaks in the convolved data
        # suppress any NoDetectionsWarning from find_peaks
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NoDetectionsWarning)
            tbl = find_peaks(convolved_data, threshold, footprint=footprint,
                             mask=mask)

        if exclude_border:
            xmax = convolved_data.shape[1] - xpad
            ymax = convolved_data.shape[0] - ypad
            mask = ((tbl['x_peak'] > xpad) & (tbl['y_peak'] > ypad)
                    & (tbl['x_peak'] < xmax) & (tbl['y_peak'] < ymax))
            tbl = tbl[mask]

        if tbl is None:
            return None

        xpos, ypos = tbl['x_peak'], tbl['y_peak']
        if not exclude_border:
            xpos -= xpad
            ypos -= ypad

        return np.transpose((xpos, ypos))

    @abc.abstractmethod
    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """
        raise NotImplementedError('Needs to be implemented in a subclass.')
