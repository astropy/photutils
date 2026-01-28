# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define the StarFinder class.
"""

import inspect
import warnings

import astropy.units as u
import numpy as np
from astropy.table import QTable
from astropy.utils import lazyproperty

from photutils.detection.core import (StarFinderBase, StarFinderCatalogBase,
                                      _validate_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._misc import _get_meta
from photutils.utils._moments import _moments, _moments_central
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils.cutouts import _overlap_slices as overlap_slices
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinder']


class StarFinder(StarFinderBase):
    """
    Detect stars in an image using a user-defined kernel.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.
        If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

    kernel : `~numpy.ndarray`
        A 2D array of the PSF kernel.

    min_separation : float, optional
        The minimum separation (in pixels) for detected objects. Note
        that large values may result in long run times.

    exclude_border : bool, optional
        Whether to exclude sources found within half the size of the
        convolution kernel from the image borders.

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

    See Also
    --------
    DAOStarFinder, IRAFStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peakmax``
    must all have the same units as the image.

    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.

    The source properties are calculated using image moments.
    """

    def __init__(self, threshold, kernel, min_separation=5.0,
                 exclude_border=False, brightest=None, peakmax=None):

        # here we validate the units, but do not strip them
        inputs = (threshold, peakmax)
        names = ('threshold', 'peakmax')
        _ = process_quantities(inputs, names)

        self.threshold = threshold
        self.kernel = kernel
        if min_separation < 0:
            msg = 'min_separation must be >= 0'
            raise ValueError(msg)
        self.min_separation = min_separation
        self.exclude_border = exclude_border
        self.brightest = _validate_brightest(brightest)
        self.peakmax = peakmax

    def _get_raw_catalog(self, data, *, mask=None):
        kernel = self.kernel
        kernel /= np.max(kernel)  # normalize max value to 1.0
        denom = np.sum(kernel**2) - (np.sum(kernel)**2 / kernel.size)
        if denom > 0:
            kernel = (kernel - np.sum(kernel) / kernel.size) / denom

        convolved_data = _filter_data(data, kernel, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        xypos = self._find_stars(convolved_data, kernel, self.threshold,
                                 min_separation=self.min_separation,
                                 mask=mask, exclude_border=self.exclude_border)

        if xypos is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        return _StarFinderCatalog(data, xypos, self.kernel,
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
            A table of found objects with the following parameters:

            * ``id``: unique object identification number.
            * ``xcentroid, ycentroid``: object centroid.
            * ``fwhm``: object FWHM.
            * ``roundness``: object roundness.
            * ``pa``: object position angle (degrees counter clockwise from
              the positive x axis).
            * ``max_value``: the maximum pixel value in the source
            * ``flux``: the source instrumental flux.
            * ``mag``: the source instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.

            `None` is returned if no stars are found or no stars meet
            the roundness and peakmax criteria.
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


class _StarFinderCatalog(StarFinderCatalogBase):
    """
    Class to calculate the properties of each detected star.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image. The image should be background-subtracted.

    xypos : Nx2 `~numpy.ndarray`
        An Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    kernel: 2D `~numpy.ndarray`
        A 2D array of the PSF kernel.

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

    def __init__(self, data, xypos, kernel, *, brightest=None, peakmax=None):
        super().__init__(data, xypos, kernel, brightest=brightest, peakmax=peakmax)
        self.default_columns = ('id', 'xcentroid', 'ycentroid', 'fwhm',
                                'roundness', 'pa', 'max_value', 'flux', 'mag')

    def _get_init_attributes(self) -> tuple:
        """
        Return a tuple of attribute names to copy during slicing.
        """
        return ('data', 'unit', 'shape', 'brightest', 'peakmax',
                'default_columns')

    def _get_list_attributes(self) -> tuple:
        """
        Return a tuple of attribute names that are lists instead of arrays.
        """
        return ('slices', 'cutout_data')

    @lazyproperty
    def slices(self):
        slices = []
        for xpos, ypos in self.xypos:
            slc, _ = overlap_slices(self.data.shape, self.shape, (ypos, xpos),
                                    mode='trim')
            slices.append(slc)
        return slices

    @lazyproperty
    def bbox_xmin(self):
        return np.array([slc[1].start for slc in self.slices])

    @lazyproperty
    def bbox_ymin(self):
        return np.array([slc[0].start for slc in self.slices])

    def make_cutouts(self, data):
        cutout = []
        for slc in self.slices:
            cdata = data[slc]
            cdata[cdata < 0] = 0.0  # exclude negative pixels
            cutout.append(cdata)
        return cutout

    @lazyproperty
    def max_value(self):
        return self.peak

    @lazyproperty
    def xcentroid(self):
        return self.cutout_xcentroid + self.bbox_xmin

    @lazyproperty
    def ycentroid(self):
        return self.cutout_ycentroid + self.bbox_ymin

    @lazyproperty
    def roundness(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            factor = self.mu_diff**2 + 4.0 * self.moments_central[:, 1, 1]**2
            return np.sqrt(factor) / self.mu_sum

    def apply_filters(self):
        """
        Filter the catalog.
        """
        # remove all non-finite values - consider these non-detections
        attrs = ('xcentroid', 'ycentroid', 'fwhm', 'roundness', 'pa',
                 'max_value', 'flux')
        mask = np.ones(len(self), dtype=bool)
        for attr in attrs:
            mask &= np.isfinite(getattr(self, attr))
        newcat = self[mask]

        if len(newcat) == 0:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        # keep sources with peak pixel values less than or equal to peakmax
        if newcat.peakmax is not None:
            mask = (newcat.max_value <= newcat.peakmax)
            newcat = newcat[mask]

        if len(newcat) == 0:
            warnings.warn('Sources were found, but none pass the peakmax '
                          'criterion.', NoDetectionsWarning)
            return None

        return newcat
