# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the DAOStarFinder class.
"""

import inspect
import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import extract_array
from astropy.table import QTable
from astropy.utils import lazyproperty

from photutils.detection.core import (StarFinderBase, _StarFinderKernel,
                                      _validate_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import isscalar, process_quantities
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['DAOStarFinder']


class DAOStarFinder(StarFinderBase):
    """
    Detect stars in an image using the DAOFIND (`Stetson 1987
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
    algorithm.

    DAOFIND searches images for local density maxima that have a peak
    amplitude greater than ``threshold`` (approximately; ``threshold``
    is applied to a convolved image) and have a size and shape similar
    to the defined 2D Gaussian kernel. The Gaussian kernel is defined
    by the ``fwhm``, ``ratio``, ``theta``, and ``sigma_radius`` input
    parameters.

    ``DAOStarFinder`` finds the object centroid by fitting the marginal
    x and y 1D distributions of the Gaussian kernel to the marginal x
    and y distributions of the input (unconvolved) ``data`` image.

    ``DAOStarFinder`` calculates the object roundness using two methods.
    The ``roundlo`` and ``roundhi`` bounds are applied to both measures
    of roundness. The first method (``roundness1``; called ``SROUND``
    in DAOFIND) is based on the source symmetry and is the ratio of a
    measure of the object's bilateral (2-fold) to four-fold symmetry.
    The second roundness statistic (``roundness2``; called ``GROUND``
    in DAOFIND) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting
    Gaussian function in y, divided by the average of the best fitting
    Gaussian functions in x and y. A circular source will have a zero
    roundness. A source extended in x or y will have a negative or
    positive roundness, respectively.

    The sharpness statistic measures the ratio of the difference between
    the height of the central pixel and the mean of the surrounding
    non-bad pixels in the convolved image, to the height of the best
    fitting Gaussian function at that point.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.
        If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor to major axis standard deviations of
        the Gaussian kernel. ``ratio`` must be strictly positive and
        less than or equal to 1.0. The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units
        of sigma (standard deviation) [``1 sigma = FWHM /
        (2.0*sqrt(2.0*log(2.0)))``].

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundness for object detection.

    roundhi : float, optional
        The upper bound on roundness for object detection.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders. The default is
        `False`, which is the mode used by DAOFIND.

    brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``brightest`` is set to `None`, all objects
        will be selected.

    peakmax : float, None, optional
        The maximum allowed peak pixel value in an object. Only objects
        whose peak pixel values are strictly smaller than ``peakmax``
        will be selected. This may be used, for example, to exclude
        saturated sources. If the star finder is run on an image that is
        a `~astropy.units.Quantity` array, then ``peakmax`` must have
        the same units. If ``peakmax`` is set to `None`, then no peak
        pixel value filtering will be performed.

    xycoords : `None` or Nx2 `~numpy.ndarray`, optional
        The (x, y) pixel coordinates of the approximate centroid
        positions of identified sources. If ``xycoords`` are input, the
        algorithm will skip the source-finding step.

    min_separation : float, optional
        The minimum separation (in pixels) for detected objects. Note
        that large values may result in long run times.

    See Also
    --------
    IRAFStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peakmax``
    must have the same units as the image.

    For the convolution step, this routine sets pixels beyond the
    image borders to 0.0. The equivalent parameters in DAOFIND are
    ``boundary='constant'`` and ``constant=0.0``.

    The main differences between `~photutils.detection.DAOStarFinder`
    and `~photutils.detection.IRAFStarFinder` are:

    * `~photutils.detection.IRAFStarFinder` always uses a 2D
      circular Gaussian kernel, while
      `~photutils.detection.DAOStarFinder` can use an elliptical
      Gaussian kernel.

    * `~photutils.detection.IRAFStarFinder` calculates the objects'
      centroid, roundness, and sharpness using image moments.

    References
    ----------
    .. [1] Stetson, P. 1987; PASP 99, 191
           (https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract)
    """

    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=0.2, sharphi=1.0, roundlo=-1.0,
                 roundhi=1.0, exclude_border=False, brightest=None,
                 peakmax=None, xycoords=None, min_separation=0.0):

        # here we validate the units, but do not strip them
        inputs = (threshold, peakmax)
        names = ('threshold', 'peakmax')
        _ = process_quantities(inputs, names)

        if not isscalar(threshold):
            raise ValueError('threshold must be a scalar value.')

        if not np.isscalar(fwhm):
            raise TypeError('fwhm must be a scalar value.')

        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.exclude_border = exclude_border
        self.brightest = _validate_brightest(brightest)
        self.peakmax = peakmax

        if min_separation < 0:
            raise ValueError('min_separation must be >= 0')
        self.min_separation = min_separation

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                raise ValueError('xycoords must be shaped as a Nx2 array')
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm,
                                        ratio=self.ratio,
                                        theta=self.theta,
                                        sigma_radius=self.sigma_radius)
        self.threshold_eff = self.threshold * self.kernel.relerr

    def _get_raw_catalog(self, data, *, mask=None):
        convolved_data = _filter_data(data, self.kernel.data, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        if self.xycoords is None:
            xypos = self._find_stars(convolved_data, self.kernel,
                                     self.threshold_eff, mask=mask,
                                     min_separation=self.min_separation,
                                     exclude_border=self.exclude_border)
        else:
            xypos = self.xycoords

        if xypos is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        return _DAOStarFinderCatalog(data, convolved_data, xypos,
                                     self.threshold,
                                     self.kernel,
                                     sharplo=self.sharplo,
                                     sharphi=self.sharphi,
                                     roundlo=self.roundlo,
                                     roundhi=self.roundhi,
                                     brightest=self.brightest,
                                     peakmax=self.peakmax)

    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array. The image should be
            background-subtracted.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.QTable` or `None`
            A table of found stars. `None` is returned if no stars are
            found. The table contains the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``npix``: the total number of pixels in the Gaussian kernel
              array.
            * ``peak``: the peak pixel value of the object.
            * ``flux``: the object instrumental flux calculated as the
              sum of data values within the kernel footprint.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.
            * ``daofind_mag``: the "mag" parameter returned by the DAOFIND
              algorithm. It is a measure of the intensity ratio of the
              amplitude of the best fitting Gaussian function at the
              object position to the detection threshold. This parameter
              is reported only for comparison to the IRAF DAOFIND
              output. It should not be interpreted as a magnitude
              derived from an integrated flux.
        """
        # here we validate the units, but do not strip them
        inputs = (data, self.threshold, self.peakmax)
        names = ('data', 'threshold', 'peakmax')
        _ = process_quantities(inputs, names)

        cat = self._get_raw_catalog(data, mask=mask)
        if cat is None:
            return None

        # apply all selection filters
        cat = cat.apply_all_filters()
        if cat is None:
            return None

        # create the output table
        return cat.to_table()


class _DAOStarFinderCatalog:
    """
    Class to create a catalog of the properties of each detected star,
    as defined by DAOFIND.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image. The image should be background-subtracted.

    convolved_data : 2D `~numpy.ndarray`
        The convolved 2D image. If ``data`` is a
        `~astropy.units.Quantity` array, then ``convolved_data`` must
        have the same units.

    xypos : Nx2 `~numpy.ndarray`
        A Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    threshold : float
        The absolute image value above which sources were selected.
        If ``data`` is a `~astropy.units.Quantity` array, then
        ``threshold`` must have the same units.

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundness for object detection.

    roundhi : float, optional
        The upper bound on roundness for object detection.

    brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``brightest`` is set to `None`, all objects
        will be selected.

    peakmax : float, None, optional
        The maximum allowed peak pixel value in an object. Only objects
        whose peak pixel values are strictly smaller than ``peakmax``
        will be selected. This may be used, for example, to exclude
        saturated sources. If the star finder is run on an image that is
        a `~astropy.units.Quantity` array, then ``peakmax`` must have
        the same units. If ``peakmax`` is set to `None`, then no peak
        pixel value filtering will be performed.
    """

    def __init__(self, data, convolved_data, xypos, threshold, kernel, *,
                 sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                 brightest=None, peakmax=None):

        # here we validate the units, but do not strip them
        inputs = (data, convolved_data, threshold, peakmax)
        names = ('data', 'convolved_data', 'threshold', 'peakmax')
        _ = process_quantities(inputs, names)

        self.data = data
        unit = data.unit if isinstance(data, u.Quantity) else None
        self.unit = unit

        self.convolved_data = convolved_data
        self.xypos = np.atleast_2d(xypos)
        self.kernel = kernel
        self.threshold = threshold
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.brightest = brightest
        self.peakmax = peakmax

        self.id = np.arange(len(self)) + 1
        self.threshold_eff = threshold * kernel.relerr
        self.cutout_shape = kernel.shape
        self.cutout_center = tuple((size - 1) // 2 for size in kernel.shape)
        self.default_columns = ('id', 'xcentroid', 'ycentroid', 'sharpness',
                                'roundness1', 'roundness2', 'npix', 'peak',
                                'flux', 'mag', 'daofind_mag')

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        # NOTE: we allow indexing/slicing of scalar (self.isscalar = True)
        #       instances in order to perform catalog filtering even for
        #       a single source

        newcls = object.__new__(self.__class__)

        # copy these attributes to the new instance
        init_attr = ('data', 'unit', 'convolved_data', 'kernel', 'threshold',
                     'sharplo', 'sharphi', 'roundlo', 'roundhi', 'brightest',
                     'peakmax', 'threshold_eff', 'cutout_shape',
                     'cutout_center', 'default_columns')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as a 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

        # slice the other attributes
        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('id')
        for key in keys:
            value = self.__dict__[key]

            # do not insert lazy attributes that are always scalar (e.g.,
            # isscalar), i.e., not an array/list for each source
            if np.isscalar(value):
                continue

            # value is always at least a 1D array, even for a single source
            value = np.atleast_1d(value[index])

            newcls.__dict__[key] = value
        return newcls

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self.xypos.shape == (1, 2)

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).
        """

        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def reset_ids(self):
        """
        Reset the ID column to be consecutive integers.
        """
        self.id = np.arange(len(self)) + 1

    def make_cutouts(self, data):
        cutouts = []
        for xpos, ypos in self.xypos:
            cutouts.append(extract_array(data, self.cutout_shape, (ypos, xpos),
                                         fill_value=0.0))
        value = np.array(cutouts)
        if self.unit is not None:
            value <<= self.unit

        return value

    @lazyproperty
    def cutout_data(self):
        return self.make_cutouts(self.data)

    @lazyproperty
    def cutout_convdata(self):
        return self.make_cutouts(self.convolved_data)

    @lazyproperty
    def data_peak(self):
        return self.cutout_data[:, self.cutout_center[0],
                                self.cutout_center[1]]

    @lazyproperty
    def convdata_peak(self):
        return self.cutout_convdata[:, self.cutout_center[0],
                                    self.cutout_center[1]]

    @lazyproperty
    def roundness1(self):
        # set the central (peak) pixel to zero for the sum4 calculation
        cutout_conv = self.cutout_convdata.copy()
        cutout_conv[:, self.cutout_center[0], self.cutout_center[1]] = 0.0

        # calculate the four roundness quadrants.
        # the cutout size always matches the kernel size, which has odd
        # dimensions.
        # quad1 = bottom right
        # quad2 = bottom left
        # quad3 = top left
        # quad4 = top right
        # 3 3 4 4 4
        # 3 3 4 4 4
        # 3 3 x 1 1
        # 2 2 2 1 1
        # 2 2 2 1 1
        quad1 = cutout_conv[:, 0:self.cutout_center[0] + 1,
                            self.cutout_center[1] + 1:]
        quad2 = cutout_conv[:, 0:self.cutout_center[0],
                            0:self.cutout_center[1] + 1]
        quad3 = cutout_conv[:, self.cutout_center[0]:,
                            0:self.cutout_center[1]]
        quad4 = cutout_conv[:, self.cutout_center[0] + 1:,
                            self.cutout_center[1]:]

        axis = (1, 2)
        sum2 = (-quad1.sum(axis=axis) + quad2.sum(axis=axis)
                - quad3.sum(axis=axis) + quad4.sum(axis=axis))
        sum4 = np.abs(cutout_conv).sum(axis=axis)

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return 2.0 * sum2 / sum4

    @lazyproperty
    def sharpness(self):
        # mean value of the unconvolved data (excluding the peak)
        cutout_data_masked = self.cutout_data * self.kernel.mask
        data_mean = ((np.sum(cutout_data_masked, axis=(1, 2)) - self.data_peak)
                     / (self.kernel.npixels - 1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (self.data_peak - data_mean) / self.convdata_peak

    def daofind_marginal_fit(self, axis=0):
        """
        Fit 1D Gaussians, defined from the marginal x/y kernel
        distributions, to the marginal x/y distributions of the original
        (unconvolved) image.

        These fits are used calculate the star centroid and roundness2
        ("GROUND") properties.

        Parameters
        ----------
        axis : {0, 1}, optional
            The axis for which the marginal fit is performed:

            * 0: for the x axis
            * 1: for the y axis

        Returns
        -------
        dx : float
            The fractional shift in x or y (depending on ``axis`` value)
            of the image centroid relative to the maximum pixel.

        hx : float
            The height of the best-fitting Gaussian to the marginal
            x or y (depending on ``axis`` value) distribution of the
            unconvolved source data.
        """
        # define triangular weighting functions along each axis, peaked
        # in the middle and equal to one at the edge
        ycen, xcen = self.cutout_center
        xx = xcen - np.abs(np.arange(self.cutout_shape[1]) - xcen) + 1
        yy = ycen - np.abs(np.arange(self.cutout_shape[0]) - ycen) + 1
        xwt, ywt = np.meshgrid(xx, yy)

        if axis == 0:  # marginal distributions along x axis
            wt = xwt[0]  # 1D
            wts = ywt  # 2D
            size = self.cutout_shape[1]
            center = xcen
            sigma = self.kernel.xsigma
            dxx = center - np.arange(size)
        elif axis == 1:  # marginal distributions along y axis
            wt = np.transpose(ywt)[0]  # 1D
            wts = xwt  # 2D
            size = self.cutout_shape[0]
            center = ycen
            sigma = self.kernel.ysigma
            dxx = np.arange(size) - center

        # compute marginal sums for given axis
        wt_sum = np.sum(wt)
        dx = center - np.arange(size)

        # weighted marginal sums
        kern_sum_1d = np.sum(self.kernel.gaussian_kernel_unmasked * wts,
                             axis=axis)
        kern_sum = np.sum(kern_sum_1d * wt)
        kern2_sum = np.sum(kern_sum_1d**2 * wt)

        dkern_dx = kern_sum_1d * dx
        dkern_dx_sum = np.sum(dkern_dx * wt)
        dkern_dx2_sum = np.sum(dkern_dx**2 * wt)
        kern_dkern_dx_sum = np.sum(kern_sum_1d * dkern_dx * wt)

        cutout_data = self.cutout_data
        if isinstance(cutout_data, u.Quantity):
            cutout_data = cutout_data.value

        data_sum_1d = np.sum(cutout_data * wts, axis=axis + 1)
        data_sum = np.sum(data_sum_1d * wt, axis=1)
        data_kern_sum = np.sum(data_sum_1d * kern_sum_1d * wt, axis=1)
        data_dkern_dx_sum = np.sum(data_sum_1d * dkern_dx * wt, axis=1)
        data_dx_sum = np.sum(data_sum_1d * dxx * wt, axis=1)

        # perform linear least-squares fit (where data = hx*kernel)
        # to find the amplitude (hx)
        hx_numer = data_kern_sum - (data_sum * kern_sum) / wt_sum
        hx_denom = kern2_sum - (kern_sum**2 / wt_sum)

        # reject the star if the fit amplitude is not positive
        mask1 = (hx_numer <= 0.0) | (hx_denom <= 0.0)

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # compute fit amplitude
            hx = hx_numer / hx_denom

            # compute centroid shift
            dx = ((kern_dkern_dx_sum
                   - (data_dkern_dx_sum - dkern_dx_sum * data_sum))
                  / (hx * dkern_dx2_sum / sigma**2))

            dx2 = data_dx_sum / data_sum

        hsize = size / 2.0
        mask2 = (np.abs(dx) > hsize)
        mask3 = (data_sum == 0.0)
        mask4 = (mask2 & mask3)
        mask5 = (mask2 & ~mask3)

        dx[mask4] = 0.0
        dx[mask5] = dx2[mask5]
        mask6 = (np.abs(dx) > hsize)
        dx[mask6] = 0.0

        hx[mask1] = np.nan
        dx[mask1] = np.nan

        return np.transpose((dx, hx))

    @lazyproperty
    def dx_hx(self):
        return self.daofind_marginal_fit(axis=0)

    @lazyproperty
    def dy_hy(self):
        return self.daofind_marginal_fit(axis=1)

    @lazyproperty
    def dx(self):
        return np.transpose(self.dx_hx)[0]

    @lazyproperty
    def dy(self):
        return np.transpose(self.dy_hy)[0]

    @lazyproperty
    def hx(self):
        return np.transpose(self.dx_hx)[1]

    @lazyproperty
    def hy(self):
        return np.transpose(self.dy_hy)[1]

    @lazyproperty
    def xcentroid(self):
        return np.transpose(self.xypos)[0] + self.dx

    @lazyproperty
    def ycentroid(self):
        return np.transpose(self.xypos)[1] + self.dy

    @lazyproperty
    def roundness2(self):
        """
        The star roundness.

        This roundness parameter represents the ratio of the difference
        in the height of the best fitting Gaussian function in x minus
        the best fitting Gaussian function in y, divided by the average
        of the best fitting Gaussian functions in x and y. A circular
        source will have a zero roundness. A source extended in x or y
        will have a negative or positive roundness, respectively.
        """
        return 2.0 * (self.hx - self.hy) / (self.hx + self.hy)

    @lazyproperty
    def peak(self):
        return self.data_peak

    @lazyproperty
    def flux(self):
        fluxes = [np.sum(arr) for arr in self.cutout_data]
        if self.unit is not None:
            fluxes = u.Quantity(fluxes)
        else:
            fluxes = np.array(fluxes)
        return fluxes

    @lazyproperty
    def mag(self):
        # ignore RunTimeWarning if flux is <= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            flux = self.flux
            if isinstance(flux, u.Quantity):
                flux = flux.value
            return -2.5 * np.log10(flux)

    @lazyproperty
    def daofind_mag(self):
        """
        The "mag" parameter returned by the original DAOFIND algorithm.

        It is a measure of the intensity ratio of the amplitude of the
        best fitting Gaussian function at the object position to the
        detection threshold.
        """
        # ignore RunTimeWarning if flux is <= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return -2.5 * np.log10(self.convdata_peak / self.threshold_eff)

    @lazyproperty
    def npix(self):
        return np.full(len(self), fill_value=self.kernel.data.size)

    def apply_filters(self):
        """
        Filter the catalog.
        """
        # remove all non-finite values - consider these non-detections
        attrs = ('xcentroid', 'ycentroid', 'hx', 'hy', 'sharpness',
                 'roundness1', 'roundness2', 'peak', 'flux')
        mask = np.ones(len(self), dtype=bool)
        for attr in attrs:
            # if threshold_eff == 0, flux will be np.inf, but
            # coordinates can still be used
            if self.threshold_eff == 0 and attr == 'flux':
                continue
            mask &= np.isfinite(getattr(self, attr))
        newcat = self[mask]

        if len(newcat) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        # filter based on sharpness, roundness, and peakmax
        mask = ((newcat.sharpness > newcat.sharplo)
                & (newcat.sharpness < newcat.sharphi)
                & (newcat.roundness1 > newcat.roundlo)
                & (newcat.roundness1 < newcat.roundhi)
                & (newcat.roundness2 > newcat.roundlo)
                & (newcat.roundness2 < newcat.roundhi))
        if newcat.peakmax is not None:
            mask &= (newcat.peak < newcat.peakmax)
        newcat = newcat[mask]

        if len(newcat) == 0:
            warnings.warn('Sources were found, but none pass the sharpness, '
                          'roundness, or peakmax criteria',
                          NoDetectionsWarning)
            return None

        return newcat

    def select_brightest(self):
        """
        Sort the catalog by the brightest fluxes and select the top
        brightest sources.
        """
        newcat = self
        if self.brightest is not None:
            idx = np.argsort(self.flux)[::-1][:self.brightest]
            newcat = self[idx]
        return newcat

    def apply_all_filters(self):
        """
        Apply all filters, select the brightest, and reset the source
        IDs.
        """
        cat = self.apply_filters()
        if cat is None:
            return None
        cat = cat.select_brightest()
        cat.reset_ids()
        return cat

    def to_table(self, columns=None):
        table = QTable()
        table.meta.update(_get_meta())  # keep table.meta type
        if columns is None:
            columns = self.default_columns
        for column in columns:
            table[column] = getattr(self, column)
        return table
