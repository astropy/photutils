# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the IRAFStarFinder class.
"""

import warnings

from astropy.table import Table
import numpy as np

from .base import StarFinderBase
from ._utils import _StarCutout, _StarFinderKernel, _find_stars
from .daofinder import _DAOFindProperties
from ..utils.exceptions import NoDetectionsWarning

__all__ = ['UserStarFinder']



class UserStarFinder(StarFinderBase):
    """
    Measure stars in an image using the DAOFIND (`Stetson 1987
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
    algorithm.  Stars measured using DAOFIND can be identified using
    any algorithm defined by the user, with the results passed in as a
    simple list of coords.

    DAOFIND (`Stetson 1987; PASP 99, 191
    <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_)
    searches images for local density maxima that have a peak amplitude
    greater than ``threshold`` (approximately; ``threshold`` is applied
    to a convolved image) and have a size and shape similar to the
    defined 2D Gaussian kernel.  The Gaussian kernel is defined by the
    ``fwhm``, ``ratio``, ``theta``, and ``sigma_radius`` input
    parameters.

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
    roundness.  A source extended in x or y will have a negative or
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
        The lower bound on roundness for object detection.

    roundhi : float, optional
        The upper bound on roundness for object detection.

    sky : float, optional
        The background sky level of the image.  Setting ``sky`` affects
        only the output values of the object ``peak``, ``flux``, and
        ``mag`` values.  The default is 0.0, which should be used to
        replicate the results from `DAOFIND`_.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_.

    brightest : int, None, optional
        Number of brightest objects to keep after sorting the full object list.
        If ``brightest`` is set to `None`, all objects will be selected.

    peakmax : float, None, optional
        Maximum peak pixel value in an object. Only objects whose peak pixel
        values are *strictly smaller* than ``peakmax`` will be selected.
        This may be used to exclude saturated sources. By default, when
        ``peakmax`` is set to `None`, all objects will be selected.

        .. warning::
            `DAOStarFinder` automatically excludes objects whose peak
            pixel values are negative. Therefore, setting ``peakmax`` to a
            non-positive value would result in exclusion of all objects.

    coords : `~astropy.table.Table` or `None`
        A table, such as returned by `find_peaks`, with approximate X,Y positions
        of identified sources.
        If not provided, the DAOFind algorithm will be used to find sources.

    See Also
    --------
    IRAFStarFinder

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in `DAOFIND`_ are
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
    .. [2] https://iraf.net/irafhelp.php?val=daofind

    .. _DAOFIND: https://iraf.net/irafhelp.php?val=daofind
    """

    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=0.2, sharphi=1.0, roundlo=-1.0,
                 roundhi=1.0, sky=0.0, exclude_border=False,
                 coords=None,
                 brightest=None, peakmax=None):

        if not np.isscalar(threshold):
            raise TypeError('threshold must be a scalar value.')
        self.threshold = threshold

        if not np.isscalar(fwhm):
            raise TypeError('fwhm must be a scalar value.')
        self.fwhm = fwhm

        self.coords = coords
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
            A table of found stars with the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``npix``: the total number of pixels in the Gaussian kernel
              array.
            * ``sky``: the input ``sky`` parameter.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object flux calculated as the peak density in
              the convolved image divided by the detection threshold.  This
              derivation matches that of `DAOFIND`_ if ``sky`` is 0.0.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.  The derivation matches that of
              `DAOFIND`_ if ``sky`` is 0.0.

            `None` is returned if no stars are found.

        """
        if self.coords:
            star_cutouts = get_cutouts(data, self.coords,
                                        self.kernel,
                                        self.threshold_eff,
                                        exclude_border=self.exclude_border)
        else:
            star_cutouts = _find_stars(data, self.kernel, self.threshold_eff,
                                       mask=mask,
                                       exclude_border=self.exclude_border)

        if star_cutouts is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        self._star_cutouts = star_cutouts

        star_props = []
        for star_cutout in star_cutouts:
            props = _DAOFindProperties(star_cutout, self.kernel, self.sky)

            if np.isnan(props.dx_hx).any() or np.isnan(props.dy_hy).any():
                continue

            if (props.sharpness <= self.sharplo or
                    props.sharpness >= self.sharphi):
                continue

            if (props.roundness1 <= self.roundlo or
                    props.roundness1 >= self.roundhi):
                continue

            if (props.roundness2 <= self.roundlo or
                    props.roundness2 >= self.roundhi):
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
        columns = ('xcentroid', 'ycentroid', 'sharpness', 'roundness1',
                   'roundness2', 'npix', 'sky', 'peak', 'flux', 'mag')
        for column in columns:
            table[column] = [getattr(props, column) for props in star_props]

        return table


def get_cutouts(data, star_list, kernel, threshold_eff, exclude_border=False):

    coords = [(row[1], row[0]) for row in star_list]
    convolved_data = data

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
        # Skip slices which include pixels with a value of NaN
        if np.isnan(data_cutout).any():
            continue
        convdata_cutout = convolved_data[slices]

        # correct pixel values for the previous image padding
        if exclude_border:
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
