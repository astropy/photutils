# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the base class and star finder kernel for
detecting stars in an astronomical image. Each star-finding class should
define a method called ``find_stars`` that finds stars in an image.
"""

import abc
import math
import warnings

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma

from photutils.detection.peakfinder import find_peaks
from photutils.utils.exceptions import NoDetectionsWarning

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
        if min_separation == 0.0:  # DAOStarFinder
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

        if tbl is None:
            return None

        if exclude_border:
            xmax = convolved_data.shape[1] - xpad
            ymax = convolved_data.shape[0] - ypad
            mask = ((tbl['x_peak'] > xpad) & (tbl['y_peak'] > ypad)
                    & (tbl['x_peak'] < xmax) & (tbl['y_peak'] < ymax))
            tbl = tbl[mask]

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


class _StarFinderKernel:
    """
    Container class for a 2D Gaussian density enhancement kernel.

    The kernel has negative wings and sums to zero.  It is used by both
    `DAOStarFinder` and `IRAFStarFinder`.

    Parameters
    ----------
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor and major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel, measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].  The default is 1.5.

    normalize_zerosum : bool, optional
        Whether to normalize the Gaussian kernel to have zero sum, The
        default is `True`, which generates a density-enhancement kernel.

    Notes
    -----
    The class attributes include the dimensions of the elliptical kernel
    and the coefficients of a 2D elliptical Gaussian function expressed
    as:

        ``f(x,y) = A * exp(-g(x,y))``

        where

        ``g(x,y) = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2``

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function
    """

    def __init__(self, fwhm, ratio=1.0, theta=0.0, sigma_radius=1.5,
                 normalize_zerosum=True):

        if fwhm < 0:
            raise ValueError('fwhm must be positive.')

        if ratio <= 0 or ratio > 1:
            raise ValueError('ratio must be positive and less or equal '
                             'than 1.')

        if sigma_radius <= 0:
            raise ValueError('sigma_radius must be positive.')

        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius
        self.xsigma = self.fwhm * gaussian_fwhm_to_sigma
        self.ysigma = self.xsigma * self.ratio

        theta_radians = np.deg2rad(self.theta)
        cost = np.cos(theta_radians)
        sint = np.sin(theta_radians)
        xsigma2 = self.xsigma**2
        ysigma2 = self.ysigma**2

        self.a = (cost**2 / (2.0 * xsigma2)) + (sint**2 / (2.0 * ysigma2))
        # CCW
        self.b = 0.5 * cost * sint * ((1.0 / xsigma2) - (1.0 / ysigma2))
        self.c = (sint**2 / (2.0 * xsigma2)) + (cost**2 / (2.0 * ysigma2))

        # find the extent of an ellipse with radius = sigma_radius*sigma;
        # solve for the horizontal and vertical tangents of an ellipse
        # defined by g(x,y) = f
        self.f = self.sigma_radius**2 / 2.0
        denom = (self.a * self.c) - self.b**2

        # nx and ny are always odd
        self.nx = 2 * int(max(2, math.sqrt(self.c * self.f / denom))) + 1
        self.ny = 2 * int(max(2, math.sqrt(self.a * self.f / denom))) + 1

        self.xc = self.xradius = self.nx // 2
        self.yc = self.yradius = self.ny // 2

        # define the kernel on a 2D grid
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        self.circular_radius = np.sqrt((xx - self.xc)**2 + (yy - self.yc)**2)
        self.elliptical_radius = (self.a * (xx - self.xc)**2
                                  + 2.0 * self.b * (xx - self.xc)
                                  * (yy - self.yc)
                                  + self.c * (yy - self.yc)**2)

        self.mask = np.where(
            (self.elliptical_radius <= self.f)
            | (self.circular_radius <= 2.0), 1, 0).astype(int)
        self.npixels = self.mask.sum()

        # NOTE: the central (peak) pixel of gaussian_kernel has a value of 1.0
        self.gaussian_kernel_unmasked = np.exp(-self.elliptical_radius)
        self.gaussian_kernel = self.gaussian_kernel_unmasked * self.mask

        # denom = variance * npixels
        denom = ((self.gaussian_kernel**2).sum()
                 - (self.gaussian_kernel.sum()**2 / self.npixels))
        self.relerr = 1.0 / np.sqrt(denom)

        # normalize the kernel to zero sum
        if normalize_zerosum:
            self.data = ((self.gaussian_kernel
                          - (self.gaussian_kernel.sum() / self.npixels))
                         / denom) * self.mask
        else:
            self.data = self.gaussian_kernel

        self.shape = self.data.shape
