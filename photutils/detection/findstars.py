# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements classes, called Finders, for detecting stars in
an astronomical image. The convention is that all Finders are subclasses
of an abstract class called ``StarFinderBase``.  Each Finder class
should define a method called ``find_stars`` that finds stars in an
image.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import warnings
import math
import abc

import six
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Column, Table
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.misc import InheritDocstrings
from astropy.utils import lazyproperty

from .core import find_peaks
from ..utils.convolution import filter_data


__all__ = ['DAOStarFinder', 'IRAFStarFinder', 'StarFinderBase']


class _ABCMetaAndInheritDocstrings(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class StarFinderBase(object):
    """
    Abstract base class for Star Finders.
    """

    def __call__(self, data):
        return self.find_stars(data)

    @abc.abstractmethod
    def find_stars(self, data):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : array_like
            The 2D image array.

        Returns
        -------
        table : `~astropy.table.Table`
            A table of found objects with the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``npix``: number of pixels in the Gaussian kernel.
            * ``sky``: the input ``sky`` parameter.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object flux calculated as the peak density in
              the convolved image divided by the detection threshold.  This
              derivation matches that of `DAOFIND`_ if ``sky`` is 0.0.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.  The derivation matches that of
              `DAOFIND`_ if ``sky`` is 0.0.

        Notes
        -----
        For the convolution step, this routine sets pixels beyond the
        image borders to 0.0.  The equivalent parameters in IRAF's
        `starfind`_ are ``boundary='constant'`` and ``constant=0.0``.

        IRAF's `starfind`_ uses ``hwhmpsf``, ``fradius``, and ``sepmin``
        as input parameters.  The equivalent input values for
        `~photutils.detection.IRAFStarFinder` are:

        * ``fwhm = hwhmpsf * 2``
        * ``sigma_radius = fradius * sqrt(2.0*log(2.0))``
        * ``minsep_fwhm = 0.5 * sepmin``

        The main differences between
        `~photutils.detection.DAOStarFinder` and
        `~photutils.detection.IRAFStarFinder` are:

        * `~photutils.detection.IRAFStarFinder` always uses a 2D
          circular Gaussian kernel, while
          `~photutils.detection.DAOStarFinder` can use an elliptical
          Gaussian kernel.

        * `~photutils.detection.IRAFStarFinder` calculates the objects'
          centroid, roundness, and sharpness using image moments.

        .. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
        .. _starfind: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
        """

        raise NotImplementedError


class DAOStarFinder(StarFinderBase):
    """
    Detect stars in an image using the DAOFIND (`Stetson 1987
    <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_) algorithm.

    DAOFIND (`Stetson 1987; PASP 99, 191
    <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_) searches
    images for local density maxima that have a peak amplitude greater
    than ``threshold`` (approximately; ``threshold`` is applied to a
    convolved image) and have a size and shape similar to the defined 2D
    Gaussian kernel.  The Gaussian kernel is defined by the ``fwhm``,
    ``ratio``, ``theta``, and ``sigma_radius`` input parameters.

    ``DAOStarFinder`` finds the object centroid by fitting the marginal x
    and y 1D distributions of the Gaussian kernel to the marginal x and
    y distributions of the input (unconvolved) ``data`` image.

    ``DAOStarFinder`` calculates the object roundness using two methods. The
    ``roundlo`` and ``roundhi`` bounds are applied to both measures of
    roundness.  The first method (``roundness1``; called ``SROUND`` in
    `DAOFIND`_) is based on the source symmetry and is the ratio of a
    measure of the object's bilateral (2-fold) to four-fold symmetry.
    The second roundness statistic (``roundness2``; called ``GROUND`` in
    `DAOFIND`_) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting
    Gaussian function in y, divided by the average of the best fitting
    Gaussian functions in x and y.  A circular source will have a zero
    roundness.  An source extended in x or y will have a negative or
    positive roundness, respectively.

    The sharpness statistic measures the ratio of the difference between
    the height of the central pixel and the mean of the surrounding
    non-bad pixels in the convolved image, to the height of the best
    fitting Gaussian function at that point.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.
    ratio : float, optional
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).
    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.
    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        (2.0*sqrt(2.0*log(2.0)))``].
    sharplo : float, optional
        The lower bound on sharpness for object detection.
    sharphi : float, optional
        The upper bound on sharpness for object detection.
    roundlo : float, optional
        The lower bound on roundess for object detection.
    roundhi : float, optional
        The upper bound on roundess for object detection.
    sky : float, optional
        The background sky level of the image.  Setting ``sky`` affects
        only the output values of the object ``peak``, ``flux``, and
        ``mag`` values.  The default is 0.0, which should be used to
        replicate the results from `DAOFIND`_.
    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_.

    See Also
    --------
    IRAFStarFinder

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in `DAOFIND`_ are
    ``boundary='constant'`` and ``constant=0.0``.

    References
    ----------
    .. [1] Stetson, P. 1987; PASP 99, 191 (http://adsabs.harvard.edu/abs/1987PASP...99..191S)
    .. [2] http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind

    .. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
    """

    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=0.2, sharphi=1.0, roundlo=-1.0,
                 roundhi=1.0, sky=0.0, exclude_border=False):
        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.sky = sky
        self.exclude_border = exclude_border

        self.kernel = _StarFinderKernel(self.fwhm, self.ratio, self.theta,
                                        self.sigma_radius)
        self.threshold_eff = self.threshold * self.kernel.relerr

    def find_stars(self, data):
        star_cutouts = _find_stars(data, self.threshold_eff, self.kernel,
                                   exclude_border=self.exclude_border)

        if len(star_cutouts) == 0:
            warnings.warn('No sources were found.', AstropyUserWarning)
            return Table()

        star_props = []
        for star_cutout in star_cutouts:
            props = _StarProperties(star_cutout, self.kernel, self.sky)

            if (props.sharpness <= self.sharplo or
                    props.sharpness >= self.sharphi):
                continue

            if (props.roundness1 <= self.roundlo or
                    props.roundness1 >= self.roundhi):
                continue

            if (props.roundness2 <= self.roundlo or
                    props.roundness2 >= self.roundhi):
                continue

            star_props.append(props)

        nstars = len(star_props)
        if nstars == 0:
            warnings.warn('Sources were found, but none pass the sharpness '
                          'and roundness criteria.', AstropyUserWarning)
            return Table()

        table = Table()
        table['id'] = np.arange(nstars) + 1

        columns = ['xcentroid', 'ycentroid', 'sharpness', 'roundness1',
                   'roundness2', 'npix', 'sky', 'peak', 'flux', 'mag']
        for column in columns:
            table[column] = [getattr(props, column) for props in star_props]

        return table


class IRAFStarFinder(StarFinderBase):
    """
    Detect stars in an image using IRAF's "starfind" algorithm.

    `starfind`_ searches images for local density maxima that have a
    peak amplitude greater than ``threshold`` above the local background
    and have a PSF full-width half-maximum similar to the input
    ``fwhm``.  The objects' centroid, roundness (ellipticity), and
    sharpness are calculated using image moments.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.
    fwhm : float
        The full-width half-maximum (FWHM) of the 2D circular Gaussian
        kernel in units of pixels.
    minsep_fwhm : float, optional
        The minimum separation for detected objects in units of
        ``fwhm``.
    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].
    sharplo : float, optional
        The lower bound on sharpness for object detection.
    sharphi : float, optional
        The upper bound on sharpness for object detection.
    roundlo : float, optional
        The lower bound on roundess for object detection.
    roundhi : float, optional
        The upper bound on roundess for object detection.
    sky : float, optional
        The background sky level of the image.  Inputing a ``sky`` value
        will override the background sky estimate.  Setting ``sky``
        affects only the output values of the object ``peak``, ``flux``,
        and ``mag`` values.  The default is ``None``, which means the
        sky value will be estimated using the `starfind`_ method.
    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `starfind`_.

    See Also
    --------
    DAOStarFinder

    References
    ----------
    .. [1] http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind

    .. _starfind: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
    """

    def __init__(self, threshold, fwhm, sigma_radius=1.5, minsep_fwhm=2.5,
                 sharplo=0.5, sharphi=2.0, roundlo=0.0, roundhi=0.2, sky=None,
                 exclude_border=False):
        self.threshold = threshold
        self.fwhm = fwhm
        self.sigma_radius = sigma_radius
        self.minsep_fwhm = minsep_fwhm
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.sky = sky
        self.exclude_border = exclude_border

    def find_stars(self, data):
        starfind_kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                            sigma_radius=self.sigma_radius)
        min_separation = max(2, int((self.fwhm * self.minsep_fwhm) + 0.5))
        objs = _find_stars(data, self.threshold, starfind_kernel,
                           min_separation=min_separation,
                           exclude_border=self.exclude_border)
        tbl = _irafstarfind_properties(objs, starfind_kernel, self.sky)
        if len(objs) == 0:
            warnings.warn('No sources were found.', AstropyUserWarning)
            return tbl     # empty table
        table_mask = ((tbl['sharpness'] > self.sharplo) &
                      (tbl['sharpness'] < self.sharphi) &
                      (tbl['roundness'] > self.roundlo) &
                      (tbl['roundness'] < self.roundhi))
        tbl = tbl[table_mask]
        idcol = Column(name='id', data=np.arange(len(tbl)) + 1)
        tbl.add_column(idcol, 0)
        if len(tbl) == 0:
            warnings.warn('Sources were found, but none pass the sharpness '
                          'and roundness criteria.', AstropyUserWarning)

        return tbl


def _find_stars(data, threshold, kernel, min_separation=None,
                exclude_border=False, local_peaks=True):
    """
    Find sources in an image by convolving the image with the input
    kernel and selecting connected pixels above a given threshold.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float
        The absolute image value above which to select sources.  Note
        that this threshold is not the same threshold input to
        ``daofind`` or ``irafstarfind``.  It should be multiplied by the
        kernel relerr.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        the cutouts.  The kernel should be normalized to zero sum.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_ and `starfind`_.

    local_peaks : bool, optional
        Set to `True` to exactly match the `DAOFIND`_ method of finding
        local peaks.  If `False`, then only one peak per thresholded
        segment will be used.

    Returns
    -------
    objects : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.


    .. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
    .. _starfind: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
    """

    from scipy import ndimage

    if not exclude_border:
        # create a larger image padded by zeros
        ysize = int(data.shape[0] + (2. * kernel.yradius))
        xsize = int(data.shape[1] + (2. * kernel.xradius))
        data_padded = np.zeros((ysize, xsize))
        data_padded[kernel.yradius:kernel.yradius + data.shape[0],
                    kernel.xradius:kernel.xradius + data.shape[1]] = data
        data = data_padded

    convolved_data = filter_data(data, kernel.data, mode='constant',
                                 fill_value=0.0, check_normalization=False)

    if not exclude_border:
        # keep border=0 in convolved data
        convolved_data[:kernel.yradius, :] = 0.
        convolved_data[-kernel.yradius:, :] = 0.
        convolved_data[:, :kernel.xradius] = 0.
        convolved_data[:, -kernel.xradius:] = 0.

    selem = ndimage.generate_binary_structure(2, 2)
    object_labels, nobjects = ndimage.label(convolved_data > threshold,
                                            structure=selem)
    objects = []
    if nobjects == 0:
        return objects

    # find object peaks in the convolved data
    if local_peaks:
        # footprint overrides min_separation in find_peaks
        if min_separation is None:   # daofind
            footprint = kernel.mask.astype(np.bool)
        else:
            from skimage.morphology import disk
            footprint = disk(min_separation)
        tbl = find_peaks(convolved_data, threshold, footprint=footprint)
        coords = np.transpose([tbl['y_peak'], tbl['x_peak']])
    else:
        object_slices = ndimage.find_objects(object_labels)
        coords = []
        for object_slice in object_slices:
            # thresholded_object is not the same size as the kernel
            thresholded_object = convolved_data[object_slice]
            ypeak, xpeak = np.unravel_index(thresholded_object.argmax(),
                                            thresholded_object.shape)
            xpeak += object_slice[1].start
            ypeak += object_slice[0].start
            coords.append((ypeak, xpeak))

    for (ypeak, xpeak) in coords:
        # now extract the object from the data, centered on the peak
        # pixel in the convolved image, with the same size as the kernel
        x0 = xpeak - kernel.xradius
        x1 = xpeak + kernel.xradius + 1
        y0 = ypeak - kernel.yradius
        y1 = ypeak + kernel.yradius + 1
        if x0 < 0 or x1 > data.shape[1]:
            continue    # pragma: no cover (isolated continue is never tested)
        if y0 < 0 or y1 > data.shape[0]:
            continue    # pragma: no cover (isolated continue is never tested)
        object_data = data[y0:y1, x0:x1]
        object_convolved_data = convolved_data[y0:y1, x0:x1].copy()
        if not exclude_border:
            # correct for image padding
            x0 -= kernel.xradius
            y0 -= kernel.yradius
        imgcutout = _ImgCutout(object_data, object_convolved_data, kernel,
                               x0, y0, xpeak, ypeak, threshold)
        objects.append(imgcutout)

    return objects


def _star_properties(star_cutouts, kernel, sky=None):
    props = []
    for star_cutout in star_cutouts:
        props.append(_StarProperties(star_cutout, kernel, sky))

    return props


class _StarProperties(object):
    """
    data : _ImgCutout
    """

    def __init__(self, star_cutout, kernel, sky=None):
        if not isinstance(star_cutout, _ImgCutout):
            raise ValueError('data must be an _ImgCutout object')

        if star_cutout.data.shape != kernel.shape:
            raise ValueError('cutout and kernel must have the same shape')

        self.cutout = star_cutout
        self.kernel = kernel
        self.sky = sky    # DAOFIND uses sky=0

        self.data = star_cutout.data
        self.data_masked = star_cutout.data_masked
        self.npixels = star_cutout.npixels    # unmasked pixels
        self.nx = star_cutout.nx
        self.ny = star_cutout.ny
        self.xcenter = int((self.nx - 1) / 2)
        self.ycenter = int((self.ny - 1) / 2)

    @lazyproperty
    def data_peak(self):
        return self.data[self.ycenter, self.xcenter]

    @lazyproperty
    def conv_peak(self):
        return self.cutout.convdata[self.ycenter, self.xcenter]

    @lazyproperty
    def roundness1(self):
        # set the central (peak) pixel to zero
        cutout_conv = self.cutout.convdata.copy()
        cutout_conv[self.ycenter, self.xcenter] = 0.0

        # calculate the four roundness quadrants
        quad1 = cutout_conv[0:self.ycenter + 1, self.xcenter + 1:]
        quad2 = cutout_conv[0:self.ycenter, 0:self.xcenter + 1]
        quad3 = cutout_conv[self.ycenter:, 0:self.xcenter]
        quad4 = cutout_conv[self.ycenter + 1:, self.xcenter:]

        sum2 = -quad1.sum() + quad2.sum() - quad3.sum() + quad4.sum()
        if sum2 == 0:
            return 0.

        sum4 = np.abs(cutout_conv).sum()
        if sum4 <= 0:
            return None

        return 2.0 * sum2 / sum4

    @lazyproperty
    def sharpness(self):
        npixels = self.npixels - 1    # exclude the peak pixel
        data_mean = (np.sum(self.data_masked) - self.data_peak) / npixels

        return (self.data_peak - data_mean) / self.conv_peak

    def daofind_marginal_fit(self, axis=0):
        # define triangular weighting functions along each axis, peaked
        # in the middle and equal to one at the edge
        x = self.xcenter - np.abs(np.arange(self.nx) - self.xcenter) + 1
        y = self.ycenter - np.abs(np.arange(self.ny) - self.ycenter) + 1
        xwt, ywt = np.meshgrid(x, y)

        if axis == 0:    # marginal distributions along x axis
            wt = xwt[0]    # 1D
            wts = ywt    # 2D
            size = self.nx
            center = self.xcenter
            sigma = self.kernel.xsigma
        elif axis == 1:    # marginal distributions along y axis
            wt = np.transpose(ywt)[0]    # 1D
            wts = xwt    # 2D
            size = self.ny
            center = self.ycenter
            sigma = self.kernel.ysigma

        # compute marginal sums for given axis
        wt_sum = np.sum(wt)
        dx = center - np.arange(size)

        # weighted marginal sums
        kern_sum_1d = np.sum(self.kernel.data * wts, axis=axis)
        kern_sum = np.sum(kern_sum_1d * wt)
        kern2_sum = np.sum(kern_sum_1d**2 * wt)

        dkern_dx = kern_sum_1d * dx
        dkern_dx_sum = np.sum(dkern_dx * wt)
        dkern_dx2_sum = np.sum(dkern_dx**2 * wt)
        kern_dkern_dx_sum = np.sum(kern_sum_1d * dkern_dx * wt)

        data_sum_1d = np.sum(self.data * wts, axis=axis)
        data_sum = np.sum(data_sum_1d * wt)
        data_kern_sum = np.sum(data_sum_1d * kern_sum_1d * wt)
        data_dkern_dx_sum = np.sum(data_sum_1d * dkern_dx * wt)
        data_dx_sum = np.sum(data_sum_1d * dx * wt)

        # perform linear least-squares fit (where data = sky + hx*kernel)
        # to find the amplitude (hx)
        # reject the star if the fit amplitude is not positive
        hx_numer = data_kern_sum - (data_sum * kern_sum) / wt_sum
        #if hx_numer <= 0.:
        #    return np.nan, np.nan

        hx_denom = kern2_sum - (kern_sum**2 / wt_sum)
        #if hx_denom <= 0.:
        #    return np.nan, np.nan

        # compute fit amplitude
        hx = hx_numer / hx_denom
        # sky = (data_sum - (hx * kern_sum)) / wt_sum

        # compute centroid shift
        dx = ((kern_dkern_dx_sum -
               (data_dkern_dx_sum - dkern_dx_sum*data_sum)) /
              (hx * dkern_dx2_sum / sigma**2))

        hsize = size / 2.
        if abs(dx) > hsize:
            if data_sum == 0.:
                dx = 0.0
            else:
                dx = data_dx_sum / data_sum
                if abs(dx) > hsize:
                    dx = 0.0

        return dx, hx

    @lazyproperty
    def dx_hx(self):
        dx, hx = self.daofind_marginal_fit(axis=0)
        return dx, hx

    @lazyproperty
    def dy_hy(self):
        dy, hy = self.daofind_marginal_fit(axis=1)
        return dy, hy

    @lazyproperty
    def dx(self):
        return self.dx_hx[0]

    @lazyproperty
    def dy(self):
        return self.dy_hy[0]

    @lazyproperty
    def xcentroid(self):
        return self.xcenter + self.dx

    @lazyproperty
    def ycentroid(self):
        return self.ycenter + self.dy

    @lazyproperty
    def hx(self):
        return self.dx_hx[1]

    @lazyproperty
    def hy(self):
        return self.dy_hy[1]

    @lazyproperty
    def roundness2(self):
        if np.isnan(self.hx) or np.isnan(self.hy):
            return np.nan
        else:
            return 2.0 * (self.hx - self.hy) / (self.hx + self.hy)

    @lazyproperty
    def peak(self):
        return self.data_peak - self.sky

    @lazyproperty
    def npix(self):
        """
        The total number of pixels in the rectangular cutout image.
        """

        return self.data.size

    @lazyproperty
    def flux(self):
        return ((self.conv_peak / self.cutout.threshold) -
                (self.sky * self.npix))

    @lazyproperty
    def mag(self):
        if self.flux <= 0:
            return np.nan
        else:
            return -2.5 * np.log10(self.flux)


def _irafstarfind_properties(imgcutouts, kernel, sky=None):
    """
    Find the properties of each detected source, as defined by IRAF's
    ``starfind``.

    Parameters
    ----------
    imgcutouts : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        the cutouts.

    sky : float, optional
        The absolute sky level.  If sky is ``None``, then a local sky
        level will be estimated (in a crude fashion).

    Returns
    -------
    table : `~astropy.table.Table`
        A table of the objects' properties.
    """

    result = defaultdict(list)
    for imgcutout in imgcutouts:
        if sky is None:
            skymask = ~kernel.mask.astype(np.bool)   # 1=sky, 0=obj
            nsky = np.count_nonzero(skymask)
            if nsky == 0:
                meansky = imgcutout.data.max() - imgcutout.convdata.max()
            else:
                meansky = (imgcutout.data * skymask).sum() / nsky
        else:
            meansky = sky
        objvals = _irafstarfind_moments(imgcutout, kernel, meansky)
        for key, val in objvals.items():
            result[key].append(val)
    names = ['xcentroid', 'ycentroid', 'fwhm', 'sharpness', 'roundness',
             'pa', 'npix', 'sky', 'peak', 'flux', 'mag']
    if len(result) == 0:
        for name in names:
            result[name] = []
    table = Table(result, names=names)

    return table


def _irafstarfind_moments(imgcutout, kernel, sky):
    """
    Find the properties of each detected source, as defined by IRAF's
    ``starfind``.

    Parameters
    ----------
    imgcutout : `_ImgCutout`
        The image cutout for a single detected source.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        ``imgcutout``.

    sky : float
        The local sky level around the source.

    Returns
    -------
    result : dict
        A dictionary of the object parameters.
    """

    from skimage.measure import moments, moments_central

    result = defaultdict(list)
    img = np.array((imgcutout.data - sky) * kernel.mask)
    img = np.where(img > 0, img, 0)    # starfind discards negative pixels
    if np.count_nonzero(img) <= 1:
        return {}
    m = moments(img, 1)
    result['xcentroid'] = m[1, 0] / m[0, 0]
    result['ycentroid'] = m[0, 1] / m[0, 0]
    result['npix'] = float(np.count_nonzero(img))   # float for easier testing
    result['sky'] = sky
    result['peak'] = np.max(img)
    flux = img.sum()
    result['flux'] = flux
    result['mag'] = -2.5 * np.log10(flux)
    mu = moments_central(
        img, result['ycentroid'], result['xcentroid'], 2) / m[0, 0]
    musum = mu[2, 0] + mu[0, 2]
    mudiff = mu[2, 0] - mu[0, 2]
    result['fwhm'] = 2.0 * np.sqrt(np.log(2.0) * musum)
    result['sharpness'] = result['fwhm'] / kernel.fwhm
    result['roundness'] = np.sqrt(mudiff**2 + 4.0*mu[1, 1]**2) / musum
    pa = 0.5 * np.arctan2(2.0 * mu[1, 1], mudiff) * (180.0 / np.pi)
    if pa < 0.0:
        pa += 180.0
    result['pa'] = pa
    result['xcentroid'] += imgcutout.x0
    result['ycentroid'] += imgcutout.y0

    return result


def _daofind_properties(imgcutouts, threshold, kernel, sky=0.0):
    """
    Find the properties of each detected source, as defined by
    `DAOFIND`_.

    Parameters
    ----------
    imgcutouts : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.

    threshold : float
        The absolute image value above which to select sources.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        the objects in ``imgcutouts``.

    sky : float, optional
        The local sky level around the source.  ``sky`` is used only to
        calculate the source peak value and flux.  The default is 0.0.

    Returns
    -------
    table : `~astropy.table.Table`
        A table of the object parameters.

    .. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
    """

    result = defaultdict(list)

    for imgcutout in imgcutouts:
        convobj = imgcutout.convdata.copy()
        convobj[kernel.yc, kernel.xc] = 0.0
        q1 = convobj[0:kernel.yc+1, kernel.xc+1:]
        q2 = convobj[0:kernel.yc, 0:kernel.xc+1]
        q3 = convobj[kernel.yc:, 0:kernel.xc]
        q4 = convobj[kernel.yc+1:, kernel.xc:]
        sum2 = -q1.sum() + q2.sum() - q3.sum() + q4.sum()
        sum4 = np.abs(convobj).sum()
        result['roundness1'].append(2.0 * sum2 / sum4)

        obj = imgcutout.data
        objpeak = obj[kernel.yc, kernel.xc]
        convpeak = imgcutout.convdata[kernel.yc, kernel.xc]
        npts = kernel.mask.sum()
        obj_masked = obj * kernel.mask
        objmean = (obj_masked.sum() - objpeak) / (npts - 1)   # exclude peak
        sharp = (objpeak - objmean) / convpeak
        result['sharpness'].append(sharp)

        dx, dy, g_roundness = _daofind_centroid_roundness(obj, kernel)
        yc, xc = imgcutout.center
        result['xcentroid'].append(xc + dx)
        result['ycentroid'].append(yc + dy)
        result['roundness2'].append(g_roundness)
        result['sky'].append(sky)      # DAOFIND uses sky=0
        result['npix'].append(float(obj.size))
        result['peak'].append(objpeak - sky)
        flux = (convpeak / threshold) - (sky * obj.size)
        result['flux'].append(flux)
        if flux <= 0:
            mag = np.nan
        else:
            mag = -2.5 * np.log10(flux)
        result['mag'].append(mag)

    names = ['xcentroid', 'ycentroid', 'sharpness', 'roundness1', 'roundness2',
             'npix', 'sky', 'peak', 'flux', 'mag']
    if len(result) == 0:
        for name in names:
            result[name] = []
    table = Table(result, names=names)

    return table


def _daofind_centroid_roundness(obj, kernel):
    """
    Calculate the source (x, y) centroid and `DAOFIND`_ "GROUND"
    roundness statistic.

    `DAOFIND`_ finds the centroid by fitting 1D Gaussians (marginal x/y
    distributions of the kernel) to the marginal x/y distributions of
    the original (unconvolved) image.

    The roundness statistic measures the ratio of the difference in the
    height of the best fitting Gaussian function in x minus the best
    fitting Gaussian function in y, divided by the average of the best
    fitting Gaussian functions in x and y.  A circular source will have
    a zero roundness.  An source extended in x (y) will have a negative
    (positive) roundness.

    Parameters
    ----------
    obj : array_like
        The 2D array of the source cutout.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        ``obj``.

    Returns
    -------
    dx, dy : float
        Fractional shift in x and y of the image centroid relative to
        the maximum pixel.

    g_roundness : float
        `DAOFIND`_ roundness (GROUND) statistic.

    .. _DAOFIND: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
    """

    dx, hx = _daofind_centroidfit(obj, kernel, axis=0)
    dy, hy = _daofind_centroidfit(obj, kernel, axis=1)
    g_roundness = 2.0 * (hx - hy) / (hx + hy)

    return dx, dy, g_roundness


def _daofind_centroidfit(obj, kernel, axis):
    """
    Find the source centroid along one axis by fitting a 1D Gaussian to
    the marginal x or y distribution of the unconvolved source data.

    Parameters
    ----------
    obj : array_like
        The 2D array of the source cutout.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The dimensions should match those of
        ``obj``.

    axis : {0, 1}
        The axis for which the centroid is computed:

        * 0: for the x axis
        * 1: for the y axis

    Returns
    -------
    dx : float
        Fractional shift in x or y (depending on ``axis`` value) of the
        image centroid relative to the maximum pixel.

    hx : float
        Height of the best-fitting Gaussian to the marginal x or y
        (depending on ``axis`` value) distribution of the unconvolved
        source data.
    """

    # define a triangular weighting function, peaked in the middle
    # and equal to one at the edge
    ywtd, xwtd = np.mgrid[0:kernel.ny, 0:kernel.nx]
    xwt = kernel.xradius - abs(xwtd - kernel.xradius) + 1.0
    ywt = kernel.yradius - abs(ywtd - kernel.yradius) + 1.0
    if axis == 0:
        wt = xwt[0]
        wts = ywt
        ksize = kernel.nx
        kernel_sigma = kernel.xsigma
        krad = ksize // 2
        sumdx_vec = krad - np.arange(ksize)
    elif axis == 1:
        wt = ywt.T[0]
        wts = xwt
        ksize = kernel.ny
        kernel_sigma = kernel.ysigma
        krad = ksize // 2
        sumdx_vec = np.arange(ksize) - krad
    n = wt.sum()

    sg = (kernel.gaussian_kernel_unmasked * wts).sum(axis)
    sumg = (wt * sg).sum()
    sumg2 = (wt * sg**2).sum()
    vec = krad - np.arange(ksize)
    dgdx = sg * vec
    sdgdx = (wt * dgdx).sum()
    sdgdx2 = (wt * dgdx**2).sum()
    sgdgdx = (wt * sg * dgdx).sum()
    sd = (obj * wts).sum(axis)
    sumd = (wt * sd).sum()
    sumgd = (wt * sg * sd).sum()
    sddgdx = (wt * sd * dgdx).sum()
    sumdx = (wt * sd * sumdx_vec).sum()
    # linear least-squares fit (data = sky + hx*gaussian_kernel_nomask)
    # to find amplitudes
    denom = (n*sumg2 - sumg**2)
    hx = (n*sumgd - sumg*sumd) / denom
    # sky = (sumg2*sumd - sumg*sumgd) / denom
    dx = (sgdgdx - (sddgdx - sdgdx*sumd)) / (hx * sdgdx2 / kernel_sigma**2)

    hsize = (ksize / 2.)
    if abs(dx) > hsize:
        dx = 0
        if sumd == 0:
            dx = 0.0
        else:
            dx = float(sumdx / sumd)
            if abs(dx) > hsize:
                dx = 0.0

    return dx, hx


class _ImgCutout(object):
    """
    Class to hold image cutouts.

    Parameters
    ----------
    data : array_like
        The cutout 2D image from the input unconvolved 2D image.

    convdata : array_like
        The cutout 2D image from the convolved 2D image.

    x0, y0 : float
        The (x, y) pixel coordinates of the lower-left pixel of the
        cutout region.

    xpeak, ypeak : float
        The (x, y) pixel coordinates of the peak pixel.
    """

    def __init__(self, data, convdata, kernel, x0, y0, xpeak, ypeak,
                 threshold):
        self.shape = data.shape
        self.nx = self.shape[1]
        self.ny = self.shape[0]
        self.data = data
        self.convdata = convdata
        self.x0 = x0
        self.y0 = y0
        self.xpeak = xpeak
        self.ypeak = ypeak
        self.xcenter = xpeak
        self.ycenter = ypeak
        self.mask = kernel.mask   # kernel mask
        self.npixels = kernel.npixels
        self.threshold = threshold

    @lazyproperty
    def data_masked(self):
        return self.data * self.mask

    @property
    def radius(self):
        return [size // 2 for size in self.data.shape]

    @property
    def center(self):
        yr, xr = self.radius
        return yr + self.y0, xr + self.x0


class _StarFinderKernel(object):
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
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].  The default is 1.5.

    normalize_zerosum : bool, optional
        Whether to normalize the Gaussian kernel to have zero sum,
        constructing construct a density-enhancement kernel.  The
        default is `True`.

    Notes
    -----
    The attributes include the dimensions of the elliptical kernel and
    the coefficients of a 2D elliptical Gaussian function expressed as:

        ``f(x,y) = A * exp(-g(x,y))``

        where

        ``g(x,y) = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2``

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gaussian_function
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
        self.nx = 2 * int(max(2, math.sqrt(self.c * self.f / denom))) + 1
        self.ny = 2 * int(max(2, math.sqrt(self.a * self.f / denom))) + 1

        self.xradius = self.nx // 2
        self.yradius = self.ny // 2

        # define the kernel on a 2D grid
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        self.xc = self.nx // 2
        self.yc = self.ny // 2

        self.circular_radius = np.sqrt((xx - self.xc)**2 + (yy - self.yc)**2)
        self.elliptical_radius = (self.a * (xx - self.xc)**2 +
                                  2.0 * self.b * (xx - self.xc) *
                                  (yy - self.yc) +
                                  self.c * (yy - self.yc)**2)

        self.mask = np.where(
            (self.elliptical_radius <= self.f) |
            (self.circular_radius <= 2.0), 1, 0).astype(np.int)
        self.npixels = self.mask.sum()

        # NOTE: the central (peak) pixel of gaussian_kernel has a value of 1.
        self.gaussian_kernel_unmasked = np.exp(-self.elliptical_radius)
        self.gaussian_kernel = self.gaussian_kernel_unmasked * self.mask

        # denom = variance * npixels
        denom = ((self.gaussian_kernel**2).sum() -
                 (self.gaussian_kernel.sum()**2 / self.npixels))
        self.relerr = 1.0 / np.sqrt(denom)

        # normalize the kernel to zero sum
        if normalize_zerosum:
            self.data = ((self.gaussian_kernel -
                          (self.gaussian_kernel.sum() / self.npixels)) /
                         denom) * self.mask
        else:
            self.data = self.gaussian_kernel

        return

    @lazyproperty
    def shape(self):
        """The shape of the 2D kernel array."""

        return self.data.shape
