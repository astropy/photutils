# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements common utility functions and classes for the star
finders.
"""

import math
import warnings

from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np

from .peakfinder import find_peaks
from ..utils._convolution import _filter_data
from ..utils.exceptions import NoDetectionsWarning


class _StarFinderKernel:
    """
    Class to calculate a 2D Gaussian density enhancement kernel.

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

        # NOTE: the central (peak) pixel of gaussian_kernel has a value of 1.
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


class _StarCutout:
    """
    Class to hold a 2D image cutout of a single star for the star finder
    classes.

    Parameters
    ----------
    data : 2D array_like
        The cutout 2D image from the input unconvolved 2D image.

    convdata : 2D array_like
        The cutout 2D image from the convolved 2D image.

    slices : tuple of two slices
        A tuple of two slices representing the minimal box of the cutout
        from the original image.

    xpeak, ypeak : float
        The (x, y) pixel coordinates of the peak pixel.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The shape of the kernel must match that
        of the input ``data``.

    threshold_eff : float
        The absolute image value above which to select sources.  This
        threshold should be the threshold value input to the star finder
        class multiplied by the kernel relerr.
    """

    def __init__(self, data, convdata, slices, xpeak, ypeak, kernel,
                 threshold_eff):

        self.data = data
        self.convdata = convdata
        self.slices = slices
        self.xpeak = xpeak
        self.ypeak = ypeak
        self.kernel = kernel
        self.threshold_eff = threshold_eff

        self.shape = data.shape
        self.nx = self.shape[1]  # always odd
        self.ny = self.shape[0]  # always odd
        self.cutout_xcenter = int(self.nx // 2)
        self.cutout_ycenter = int(self.ny // 2)

        self.xorigin = self.slices[1].start  # in original image
        self.yorigin = self.slices[0].start  # in original image

        self.mask = kernel.mask  # kernel mask
        self.npixels = kernel.npixels  # unmasked pixels
        self.data_masked = self.data * self.mask


def _find_stars(data, kernel, threshold_eff, min_separation=None,
                mask=None, exclude_border=False):
    """
    Find stars in an image.

    Parameters
    ----------
    data : 2D array_like
        The 2D array of the image.

    kernel : `_StarFinderKernel`
        The convolution kernel.

    threshold_eff : float
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
    objects : list of `_StarCutout`
        A list of `_StarCutout` objects containing the image cutout for
        each source.

    .. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind

    .. _starfind: https://iraf.net/irafhelp.php?val=starfind
    """

    convolved_data = _filter_data(data, kernel.data, mode='constant',
                                  fill_value=0.0, check_normalization=False)

    # define a local footprint for the peak finder
    if min_separation is None:  # daofind
        footprint = kernel.mask.astype(bool)
    else:
        # define a circular footprint
        idx = np.arange(-min_separation, min_separation + 1)
        xx, yy = np.meshgrid(idx, idx)
        footprint = np.array((xx**2 + yy**2) <= min_separation**2, dtype=int)

    # pad the data and convolved image by the kernel x/y radius to allow
    # for detections near the edges
    if not exclude_border:
        ypad = kernel.yradius
        xpad = kernel.xradius
        pad = ((ypad, ypad), (xpad, xpad))
        pad_mode = 'constant'
        data = np.pad(data, pad, mode=pad_mode, constant_values=[0.])
        if mask is not None:
            mask = np.pad(mask, pad, mode=pad_mode, constant_values=[0.])
        convolved_data = np.pad(convolved_data, pad, mode=pad_mode,
                                constant_values=[0.])

    # find local peaks in the convolved data
    with warnings.catch_warnings():
        # suppress any NoDetectionsWarning from find_peaks
        warnings.filterwarnings('ignore', category=NoDetectionsWarning)
        tbl = find_peaks(convolved_data, threshold_eff, footprint=footprint,
                         mask=mask)

    if tbl is None:
        return None

    coords = np.transpose([tbl['y_peak'], tbl['x_peak']])

    star_cutouts = []
    for (ypeak, xpeak) in coords:
        # now extract the object from the data, centered on the peak
        # pixel in the convolved image, with the same size as the kernel
        x0 = xpeak - kernel.xradius
        x1 = xpeak + kernel.xradius + 1
        y0 = ypeak - kernel.yradius
        y1 = ypeak + kernel.yradius + 1

        if x0 < 0 or x1 > data.shape[1]:
            continue  # pragma: no cover
        if y0 < 0 or y1 > data.shape[0]:
            continue  # pragma: no cover

        slices = (slice(y0, y1), slice(x0, x1))
        data_cutout = data[slices]
        convdata_cutout = convolved_data[slices]

        # correct pixel values for the previous image padding
        if not exclude_border:
            x0 -= kernel.xradius
            x1 -= kernel.xradius
            y0 -= kernel.yradius
            y1 -= kernel.yradius
            xpeak -= kernel.xradius
            ypeak -= kernel.yradius
            slices = (slice(y0, y1), slice(x0, x1))

        star_cutouts.append(_StarCutout(data_cutout, convdata_cutout, slices,
                                        xpeak, ypeak, kernel, threshold_eff))

    return star_cutouts
