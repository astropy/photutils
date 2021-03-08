# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the IRAFStarFinder class.
"""

import warnings

from astropy.table import Table
from astropy.utils import lazyproperty
import numpy as np

from .base import StarFinderBase
from ._utils import _StarCutout, _StarFinderKernel, _find_stars
from ..utils._moments import _moments, _moments_central
from ..utils.exceptions import NoDetectionsWarning

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

    See Also
    --------
    DAOStarFinder

    References
    ----------
    .. [1] https://iraf.net/irafhelp.php?val=starfind

    .. _starfind: https://iraf.net/irafhelp.php?val=starfind
    """

    def __init__(self, threshold, fwhm, sigma_radius=1.5, minsep_fwhm=2.5,
                 sharplo=0.5, sharphi=2.0, roundlo=0.0, roundhi=0.2, sky=None,
                 exclude_border=False, brightest=None, peakmax=None):

        if not np.isscalar(threshold):
            raise TypeError('threshold must be a scalar value.')
        self.threshold = threshold

        if not np.isscalar(fwhm):
            raise TypeError('fwhm must be a scalar value.')
        self.fwhm = fwhm

        self.sigma_radius = sigma_radius
        self.minsep_fwhm = minsep_fwhm
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.sky = sky
        self.exclude_border = exclude_border

        self.min_separation = max(2, int((self.fwhm * self.minsep_fwhm) + 0.5))
        self.kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                        sigma_radius=self.sigma_radius)
        self.brightest = brightest
        self.peakmax = peakmax
        self._star_cutouts = None

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
        table : `~astropy.table.Table` or `None`
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
        star_cutouts = _find_stars(data, self.kernel, self.threshold,
                                   min_separation=self.min_separation,
                                   mask=mask,
                                   exclude_border=self.exclude_border)

        if star_cutouts is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        self._star_cutouts = star_cutouts

        star_props = []
        for star_cutout in star_cutouts:
            props = _IRAFStarFindProperties(star_cutout, self.kernel,
                                            self.sky)

            # star cutout needs more than one non-zero value
            if np.count_nonzero(props.data) <= 1:
                continue

            if (props.sharpness <= self.sharplo
                    or props.sharpness >= self.sharphi):
                continue

            if (props.roundness <= self.roundlo
                    or props.roundness >= self.roundhi):
                continue

            if self.peakmax is not None and props.peak >= self.peakmax:
                continue

            star_props.append(props)

        nstars = len(star_props)
        if nstars == 0:
            warnings.warn('Sources were found, but none pass the sharpness '
                          'and roundness criteria.', NoDetectionsWarning)
            return None

        if self.brightest is not None:
            fluxes = [props.flux for props in star_props]
            idx = sorted(np.argsort(fluxes)[-self.brightest:].tolist())
            star_props = [star_props[k] for k in idx]
            nstars = len(star_props)

        table = Table()
        table['id'] = np.arange(nstars) + 1
        columns = ('xcentroid', 'ycentroid', 'fwhm', 'sharpness', 'roundness',
                   'pa', 'npix', 'sky', 'peak', 'flux', 'mag')
        for column in columns:
            table[column] = [getattr(props, column) for props in star_props]

        return table


class _IRAFStarFindProperties:
    """
    Class to calculate the properties of each detected star, as defined
    by IRAF's ``starfind`` task.

    Parameters
    ----------
    star_cutout : `_StarCutout`
        A `_StarCutout` object containing the image cutout for the star.

    kernel : `_StarFinderKernel`
        The convolution kernel.  The shape of the kernel must match that
        of the input ``star_cutout``.

    sky : `None` or float, optional
        The local sky level around the source.  If sky is ``None``, then
        a local sky level will be (crudely) estimated using the IRAF
        ``starfind`` calculation.
    """

    def __init__(self, star_cutout, kernel, sky=None):
        if not isinstance(star_cutout, _StarCutout):
            raise ValueError('data must be an _StarCutout object')

        if star_cutout.data.shape != kernel.shape:
            raise ValueError('cutout and kernel must have the same shape')

        self.cutout = star_cutout
        self.kernel = kernel

        if sky is None:
            skymask = ~self.kernel.mask.astype(bool)  # 1=sky, 0=obj
            nsky = np.count_nonzero(skymask)
            if nsky == 0:
                mean_sky = (np.max(self.cutout.data)
                            - np.max(self.cutout.convdata))
            else:
                mean_sky = np.sum(self.cutout.data * skymask) / nsky
            self.sky = mean_sky
        else:
            self.sky = sky

    @lazyproperty
    def data(self):
        cutout = np.array((self.cutout.data - self.sky) * self.cutout.mask)
        # IRAF starfind discards negative pixels
        cutout = np.where(cutout > 0, cutout, 0)

        return cutout

    @lazyproperty
    def moments(self):
        return _moments(self.data, order=1)

    @lazyproperty
    def cutout_xcentroid(self):
        return self.moments[0, 1] / self.moments[0, 0]

    @lazyproperty
    def cutout_ycentroid(self):
        return self.moments[1, 0] / self.moments[0, 0]

    @lazyproperty
    def xcentroid(self):
        return self.cutout_xcentroid + self.cutout.xorigin

    @lazyproperty
    def ycentroid(self):
        return self.cutout_ycentroid + self.cutout.yorigin

    @lazyproperty
    def npix(self):
        return np.count_nonzero(self.data)

    @lazyproperty
    def sky(self):
        return self.sky

    @lazyproperty
    def peak(self):
        return np.max(self.data)

    @lazyproperty
    def flux(self):
        return np.sum(self.data)

    @lazyproperty
    def mag(self):
        return -2.5 * np.log10(self.flux)

    @lazyproperty
    def moments_central(self):
        return _moments_central(
            self.data, (self.cutout_xcentroid, self.cutout_ycentroid),
            order=2) / self.moments[0, 0]

    @lazyproperty
    def mu_sum(self):
        return self.moments_central[0, 2] + self.moments_central[2, 0]

    @lazyproperty
    def mu_diff(self):
        return self.moments_central[0, 2] - self.moments_central[2, 0]

    @lazyproperty
    def fwhm(self):
        return 2.0 * np.sqrt(np.log(2.0) * self.mu_sum)

    @lazyproperty
    def sharpness(self):
        return self.fwhm / self.kernel.fwhm

    @lazyproperty
    def roundness(self):
        return np.sqrt(self.mu_diff**2
                       + 4.0 * self.moments_central[1, 1]**2) / self.mu_sum

    @lazyproperty
    def pa(self):
        pa = np.rad2deg(0.5 * np.arctan2(2.0 * self.moments_central[1, 1],
                                         self.mu_diff))
        if pa < 0.:
            pa += 180.

        return pa
