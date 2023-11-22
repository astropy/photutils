# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the IRAFStarFinder class.
"""

import inspect
import warnings

import numpy as np
from astropy.nddata import extract_array
from astropy.table import QTable
from astropy.utils import lazyproperty

from photutils.detection.core import StarFinderBase, _StarFinderKernel
from photutils.utils._convolution import _filter_data
from photutils.utils._misc import _get_meta
from photutils.utils._moments import _moments, _moments_central
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['IRAFStarFinder']


class IRAFStarFinder(StarFinderBase):
    """
    Detect stars in an image using IRAF's "starfind" algorithm.

    `IRAFStarFinder` searches images for local density maxima that have
    a peak amplitude greater than ``threshold`` above the local
    background and have a PSF full-width at half-maximum similar to the
    input ``fwhm``.  The objects' centroid, roundness (ellipticity), and
    sharpness are calculated using image moments.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.

    fwhm : float
        The full-width half-maximum (FWHM) of the 2D circular Gaussian
        kernel in units of pixels.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].

    minsep_fwhm : float, optional
        The separation (in units of ``fwhm``) for detected objects. The
        minimum separation is calculated as ``int((fwhm * minsep_fwhm) +
        0.5)`` and is clipped to a minimum value of 2. Note that large
        values may result in long run times.

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundness for object detection.

    roundhi : float, optional
        The upper bound on roundness for object detection.

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

    brightest : int, None, optional
        Number of brightest objects to keep after sorting the full object list.
        If ``brightest`` is set to `None`, all objects will be selected.

    peakmax : float, None, optional
        Maximum peak pixel value in an object. Only objects whose peak pixel
        values are *strictly smaller* than ``peakmax`` will be selected.
        This may be used to exclude saturated sources. By default, when
        ``peakmax`` is set to `None`, all objects will be selected.

        .. warning::
            `IRAFStarFinder` automatically excludes objects whose peak
            pixel values are negative. Therefore, setting ``peakmax`` to a
            non-positive value would result in exclusion of all objects.

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
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in IRAF's `starfind`_ are
    ``boundary='constant'`` and ``constant=0.0``.

    IRAF's `starfind`_ uses ``hwhmpsf``, ``fradius``, and ``sepmin`` as
    input parameters.  The equivalent input values for
    `IRAFStarFinder` are:

    * ``fwhm = hwhmpsf * 2``
    * ``sigma_radius = fradius * sqrt(2.0*log(2.0))``
    * ``minsep_fwhm = 0.5 * sepmin``

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
    .. [1] https://iraf.net/irafhelp.php?val=starfind

    .. _starfind: https://iraf.net/irafhelp.php?val=starfind
    """

    def __init__(self, threshold, fwhm, sigma_radius=1.5, minsep_fwhm=2.5,
                 sharplo=0.5, sharphi=2.0, roundlo=0.0, roundhi=0.2, sky=None,
                 exclude_border=False, brightest=None, peakmax=None,
                 xycoords=None, min_separation=None):

        if not np.isscalar(threshold):
            raise TypeError('threshold must be a scalar value.')

        if not np.isscalar(fwhm):
            raise TypeError('fwhm must be a scalar value.')

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
        self.brightest = self._validate_brightest(brightest)
        self.peakmax = peakmax

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                raise ValueError('xycoords must be shaped as a Nx2 array')
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                        sigma_radius=self.sigma_radius)

        if min_separation is not None:
            if min_separation < 0:
                raise ValueError('min_separation must be >= 0')
            self.min_separation = min_separation
        else:
            self.min_separation = max(2, int((self.fwhm * self.minsep_fwhm)
                                             + 0.5))

    @staticmethod
    def _validate_brightest(brightest):
        if brightest is not None:
            if brightest <= 0:
                raise ValueError('brightest must be >= 0')
            bright_int = int(brightest)
            if bright_int != brightest:
                raise ValueError('brightest must be an integer')
            brightest = bright_int
        return brightest

    def _get_raw_catalog(self, data, mask=None):
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

        cat = _IRAFStarFinderCatalog(data, convolved_data, xypos, self.kernel,
                                     sky=self.sky, sharplo=self.sharplo,
                                     sharphi=self.sharphi,
                                     roundlo=self.roundlo,
                                     roundhi=self.roundhi,
                                     brightest=self.brightest,
                                     peakmax=self.peakmax)
        return cat

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
            is masked.  Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.QTable` or `None`
            A table of found objects with the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``fwhm``: object FWHM.
            * ``sharpness``: object sharpness.
            * ``roundness``: object roundness.
            * ``pa``: object position angle (degrees counter clockwise from
              the positive x axis).
            * ``npix``: the total number of (positive) unmasked pixels.
            * ``sky``: the local ``sky`` value.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object instrumental flux.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.

            `None` is returned if no stars are found.
        """
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
        The 2D image.

    convolved_data : 2D `~numpy.ndarray`
        The convolved 2D image.

    xypos : Nx2 `numpy.ndarray`
        A Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    sky : `None` or float, optional
        The local sky level around the source. If sky is ``None``, then
        a local sky level will be (crudely) estimated using the IRAF
        ``starfind`` calculation.
    """

    def __init__(self, data, convolved_data, xypos, kernel, sky=None,
                 sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                 brightest=None, peakmax=None):

        self.data = data
        self.convolved_data = convolved_data
        self.xypos = xypos
        self.kernel = kernel
        self._sky = sky
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
                                'sky', 'peak', 'flux', 'mag')

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        newcls = object.__new__(self.__class__)
        init_attr = ('data', 'convolved_data', 'kernel', '_sky', 'sharplo',
                     'sharphi', 'roundlo', 'roundhi', 'brightest', 'peakmax',
                     'cutout_shape', 'default_columns')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as a 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

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
        """Reset the ID column to be consecutive integers."""
        self.id = np.arange(len(self)) + 1

    @lazyproperty
    def sky(self):
        if self._sky is None:
            skymask = ~self.kernel.mask.astype(bool)  # 1=sky, 0=obj
            nsky = np.count_nonzero(skymask)
            axis = (1, 2)
            if nsky == 0.0:
                sky = (np.max(self.cutout_data_nosub, axis=axis)
                       - np.max(self.cutout_convdata, axis=axis))
            else:
                sky = (np.sum(self.cutout_data_nosub * skymask, axis=axis)
                       / nsky)
        else:
            sky = np.full(len(self), fill_value=self._sky)

        return sky

    def make_cutouts(self, data):
        cutouts = []
        for xpos, ypos in self.xypos:
            cutouts.append(extract_array(data, self.cutout_shape, (ypos, xpos),
                                         fill_value=0.0))
        return np.array(cutouts)

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
    def cutout_convdata(self):
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
        return np.array([np.max(arr) for arr in self.cutout_data])

    @lazyproperty
    def flux(self):
        return np.array([np.sum(arr) for arr in self.cutout_data])

    @lazyproperty
    def mag(self):
        return -2.5 * np.log10(self.flux)

    @lazyproperty
    def moments_central(self):
        moments = np.array([_moments_central(arr, center=(xcen_, ycen_),
                                             order=2)
                            for arr, xcen_, ycen_ in
                            zip(self.cutout_data, self.cutout_xcentroid,
                                self.cutout_ycentroid)])
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
        pa = np.where(pa < 0, pa + 180, pa)
        return pa

    def apply_filters(self):
        """Filter the catalog."""
        mask = np.count_nonzero(self.cutout_data, axis=(1, 2)) > 1
        mask &= ((self.sharpness > self.sharplo)
                 & (self.sharpness < self.sharphi)
                 & (self.roundness > self.roundlo)
                 & (self.roundness < self.roundhi))
        if self.peakmax is not None:
            mask &= (self.peak < self.peakmax)
        newcat = self[mask]

        if len(newcat) == 0:
            warnings.warn('Sources were found, but none pass the sharpness, '
                          'roundness, or peakmax criteria',
                          NoDetectionsWarning)
            return None

        return newcat

    def select_brightest(self):
        """
        Sort the catalog by the brightest fluxes and select the
        top brightest sources.
        """
        newcat = self
        if self.brightest is not None:
            idx = np.argsort(self.flux)[::-1][:self.brightest]
            newcat = self[idx]
        return newcat

    def apply_all_filters(self):
        """
        Apply all filters, select the brightest, and reset the source
        ids.
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
