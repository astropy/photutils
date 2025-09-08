# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define the IRAFStarFinder class.
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
from photutils.utils._moments import _moments, _moments_central
from photutils.utils._quantity_helpers import isscalar, process_quantities
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['IRAFStarFinder']


class IRAFStarFinder(StarFinderBase):
    """
    Detect stars in an image using IRAF's "starfind" algorithm.

    `IRAFStarFinder` searches images for local density maxima that
    have a peak amplitude greater than ``threshold`` above the local
    background and have a PSF full-width at half-maximum similar to the
    input ``fwhm``. The objects' centroid, roundness (ellipticity), and
    sharpness are calculated using image moments.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.
        If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

    fwhm : float
        The full-width half-maximum (FWHM) of the 2D circular Gaussian
        kernel in units of pixels.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units
        of sigma (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].

    minsep_fwhm : float, optional
        The separation (in units of ``fwhm``) for detected objects. The
        minimum separation is calculated as ``int((fwhm * minsep_fwhm) +
        0.5)`` and is clipped to a minimum value of 2. Note that large
        values may result in long run times.

    sharplo : float, optional
        The lower bound on sharpness for object detection. Objects
        with sharpness less than ``sharplo`` will be rejected.

    sharphi : float, optional
        The upper bound on sharpness for object detection. Objects
        with sharpness greater than ``sharphi`` will be rejected.

    roundlo : float, optional
        The lower bound on roundness for object detection. Objects
        with roundness less than ``roundlo`` will be rejected.

    roundhi : float, optional
        The upper bound on roundness for object detection. Objects
        with roundness greater than ``roundhi`` will be rejected.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders. The default is
        `False`, which is the mode used by starfind.

    brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``brightest`` is set to `None`, all objects
        will be selected.

    peakmax : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peakmax`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peakmax`` must have the
        same units. If ``peakmax`` is set to `None`, then no peak pixel
        value filtering will be performed.

    xycoords : `None` or Nx2 `~numpy.ndarray`, optional
        The (x, y) pixel coordinates of the approximate centroid
        positions of identified sources. If ``xycoords`` are input, the
        algorithm will skip the source-finding step.

    min_separation : `None` or float, optional
        The minimum separation (in pixels) for detected objects. If
        `None` then ``minsep_fwhm`` will be used, otherwise this keyword
        overrides ``minsep_fwhm``. Note that large values may result in
        long run times.

    See Also
    --------
    DAOStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peakmax``
    must have the same units as the image.

    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0. The equivalent parameters in IRAF's starfind are
    ``boundary='constant'`` and ``constant=0.0``.

    IRAF's starfind uses ``hwhmpsf``, ``fradius``, and ``sepmin`` as
    input parameters. The equivalent input values for `IRAFStarFinder`
    are:

    * ``fwhm = hwhmpsf * 2``
    * ``sigma_radius = fradius * sqrt(2.0*log(2.0))``
    * ``minsep_fwhm = 0.5 * sepmin``

    The main differences between `~photutils.detection.DAOStarFinder`
    and `~photutils.detection.IRAFStarFinder` are:

    * `~photutils.detection.IRAFStarFinder` always uses a 2D circular
      Gaussian kernel, while `~photutils.detection.DAOStarFinder` can use
      an elliptical Gaussian kernel.

    * `IRAFStarFinder` internally calculates a "sky" background level
      based on unmasked pixels within the kernel footprint.

    * `~photutils.detection.IRAFStarFinder` calculates the objects'
      centroid, roundness, and sharpness using image moments.
    """

    def __init__(self, threshold, fwhm, sigma_radius=1.5, minsep_fwhm=2.5,
                 sharplo=0.5, sharphi=2.0, roundlo=0.0, roundhi=0.2,
                 exclude_border=False, brightest=None, peakmax=None,
                 xycoords=None, min_separation=None):

        # here we validate the units, but do not strip them
        inputs = (threshold, peakmax)
        names = ('threshold', 'peakmax')
        _ = process_quantities(inputs, names)

        if not isscalar(threshold):
            msg = 'threshold must be a scalar value'
            raise TypeError(msg)

        if not np.isscalar(fwhm):
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        self.threshold = threshold
        self.fwhm = fwhm
        self.sigma_radius = sigma_radius
        self.minsep_fwhm = minsep_fwhm
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.exclude_border = exclude_border
        self.brightest = _validate_brightest(brightest)
        self.peakmax = peakmax

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                msg = 'xycoords must be shaped as a Nx2 array'
                raise ValueError(msg)
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                        sigma_radius=self.sigma_radius)

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = max(2, int((self.fwhm * self.minsep_fwhm)
                                             + 0.5))

    def _get_raw_catalog(self, data, *, mask=None):
        convolved_data = _filter_data(data, self.kernel.data, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        if self.xycoords is None:
            xypos = self._find_stars(convolved_data, self.kernel,
                                     self.threshold,
                                     min_separation=self.min_separation,
                                     mask=mask,
                                     exclude_border=self.exclude_border)
        else:
            xypos = self.xycoords

        if xypos is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        return _IRAFStarFinderCatalog(data, convolved_data, xypos, self.kernel,
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
            * ``fwhm``: object FWHM.
            * ``sharpness``: object sharpness.
            * ``roundness``: object roundness.
            * ``pa``: object position angle (degrees counter clockwise from
              the positive x axis).
            * ``npix``: the total number of (positive) unmasked pixels.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object instrumental flux calculated as the
              sum of sky-subtracted data values within the kernel
              footprint.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.
        """
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


class _IRAFStarFinderCatalog:
    """
    Class to create a catalog of the properties of each detected star,
    as defined by IRAF's ``starfind`` task.

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

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    sharplo : float, optional
        The lower bound on sharpness for object detection. Objects
        with sharpness less than ``sharplo`` will be rejected.

    sharphi : float, optional
        The upper bound on sharpness for object detection. Objects
        with sharpness greater than ``sharphi`` will be rejected.

    roundlo : float, optional
        The lower bound on roundness for object detection. Objects
        with roundness less than ``roundlo`` will be rejected.

    roundhi : float, optional
        The upper bound on roundness for object detection. Objects
        with roundness greater than ``roundhi`` will be rejected.

    brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``brightest`` is set to `None`, all objects
        will be selected.

    peakmax : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peakmax`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peakmax`` must have the
        same units. If ``peakmax`` is set to `None`, then no peak pixel
        value filtering will be performed.
    """

    def __init__(self, data, convolved_data, xypos, kernel, *, sharplo=0.2,
                 sharphi=1.0, roundlo=-1.0, roundhi=1.0, brightest=None,
                 peakmax=None):

        # here we validate the units, but do not strip them
        inputs = (data, convolved_data, peakmax)
        names = ('data', 'convolved_data', 'peakmax')
        _ = process_quantities(inputs, names)

        self.data = data
        unit = data.unit if isinstance(data, u.Quantity) else None
        self.unit = unit

        self.convolved_data = convolved_data
        self.xypos = xypos
        self.kernel = kernel
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.brightest = brightest
        self.peakmax = peakmax

        self.id = np.arange(len(self)) + 1
        self.cutout_shape = kernel.shape
        self.default_columns = ('id', 'xcentroid', 'ycentroid', 'fwhm',
                                'sharpness', 'roundness', 'pa', 'npix',
                                'peak', 'flux', 'mag')

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        # NOTE: we allow indexing/slicing of scalar (self.isscalar = True)
        #       instances in order to perform catalog filtering even for
        #       a single source

        newcls = object.__new__(self.__class__)

        # copy these attributes to the new instance
        init_attr = ('data', 'unit', 'convolved_data', 'kernel',
                     'sharplo', 'sharphi', 'roundlo', 'roundhi', 'brightest',
                     'peakmax', 'cutout_shape', 'default_columns')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as a 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

        # index/slice the remaining attributes
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

    @lazyproperty
    def sky(self):
        """
        Calculate the sky background level.

        The local sky level is roughly estimated using the IRAF starfind
        calculation as the average value in the non-masked regions
        within the kernel footprint.
        """
        skymask = ~self.kernel.mask.astype(bool)  # 1=sky, 0=obj
        nsky = np.count_nonzero(skymask)
        axis = (1, 2)
        if nsky == 0.0:  # pragma: no cover
            sky = (np.max(self.cutout_data_nosub, axis=axis)
                   - np.max(self.cutout_convdata, axis=axis))
        else:
            sky = (np.sum(self.cutout_data_nosub * skymask, axis=axis)
                   / nsky)

        if self.unit is not None:
            sky <<= self.unit

        return sky

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
    def cutout_data_nosub(self):
        return self.make_cutouts(self.data)

    @lazyproperty
    def cutout_data(self):
        data = ((self.cutout_data_nosub - self.sky[:, np.newaxis, np.newaxis])
                * self.kernel.mask)
        # IRAF starfind discards negative pixels
        data[data < 0] = 0.0
        return data

    @lazyproperty
    def cutout_convdata(self):  # pragma: no cover
        return self.make_cutouts(self.convolved_data)

    @lazyproperty
    def npix(self):
        return np.count_nonzero(self.cutout_data, axis=(1, 2))

    @lazyproperty
    def moments(self):
        return np.array([_moments(arr, order=1) for arr in self.cutout_data])

    @lazyproperty
    def cutout_centroid(self):
        moments = self.moments

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ycentroid = moments[:, 1, 0] / moments[:, 0, 0]
            xcentroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((ycentroid, xcentroid))

    @lazyproperty
    def cutout_xcentroid(self):
        return np.transpose(self.cutout_centroid)[1]

    @lazyproperty
    def cutout_ycentroid(self):
        return np.transpose(self.cutout_centroid)[0]

    @lazyproperty
    def cutout_xorigin(self):
        return np.transpose(self.xypos)[0] - self.kernel.xradius

    @lazyproperty
    def cutout_yorigin(self):
        return np.transpose(self.xypos)[1] - self.kernel.yradius

    @lazyproperty
    def xcentroid(self):
        return self.cutout_xcentroid + self.cutout_xorigin

    @lazyproperty
    def ycentroid(self):
        return self.cutout_ycentroid + self.cutout_yorigin

    @lazyproperty
    def peak(self):
        peaks = [np.max(arr) for arr in self.cutout_data]
        return u.Quantity(peaks) if self.unit is not None else np.array(peaks)

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
        flux = self.flux
        if isinstance(flux, u.Quantity):
            flux = flux.value
        return -2.5 * np.log10(flux)

    @lazyproperty
    def moments_central(self):
        moments = np.array([_moments_central(arr, center=(xcen_, ycen_),
                                             order=2)
                            for arr, xcen_, ycen_ in
                            zip(self.cutout_data, self.cutout_xcentroid,
                                self.cutout_ycentroid, strict=True)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return moments / self.moments[:, 0, 0][:, np.newaxis, np.newaxis]

    @lazyproperty
    def mu_sum(self):
        return self.moments_central[:, 0, 2] + self.moments_central[:, 2, 0]

    @lazyproperty
    def mu_diff(self):
        return self.moments_central[:, 0, 2] - self.moments_central[:, 2, 0]

    @lazyproperty
    def fwhm(self):
        return 2.0 * np.sqrt(np.log(2.0) * self.mu_sum)

    @lazyproperty
    def roundness(self):
        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (np.sqrt(self.mu_diff**2
                            + 4.0 * self.moments_central[:, 1, 1]**2)
                    / self.mu_sum)

    @lazyproperty
    def sharpness(self):
        return self.fwhm / self.kernel.fwhm

    @lazyproperty
    def pa(self):
        pa = np.rad2deg(0.5 * np.arctan2(2.0 * self.moments_central[:, 1, 1],
                                         self.mu_diff))
        return np.where(pa < 0, pa + 180, pa)

    def apply_filters(self):
        """
        Filter the catalog.
        """
        # remove all non-finite values - consider these non-detections
        attrs = ('xcentroid', 'ycentroid', 'sharpness', 'roundness', 'pa',
                 'sky', 'peak', 'flux')
        mask = np.count_nonzero(self.cutout_data, axis=(1, 2)) > 1
        for attr in attrs:
            mask &= np.isfinite(getattr(self, attr))
        newcat = self[mask]

        if len(newcat) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        # keep sources that are within the sharpness, roundness, and
        # peakmax (inclusive) bounds
        mask = ((newcat.sharpness >= newcat.sharplo)
                & (newcat.sharpness <= newcat.sharphi)
                & (newcat.roundness >= newcat.roundlo)
                & (newcat.roundness <= newcat.roundhi))
        if newcat.peakmax is not None:
            mask &= (newcat.peak <= newcat.peakmax)
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
