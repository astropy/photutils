# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the StarFinder class.
"""

import inspect
import warnings

from astropy.nddata import overlap_slices
from astropy.table import QTable
from astropy.utils import lazyproperty
import numpy as np

from .base import StarFinderBase
from .peakfinder import find_peaks
from ..utils._convolution import _filter_data
from ..utils._moments import _moments, _moments_central
from ..utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinder']


class StarFinder(StarFinderBase):
    """
    Detect stars in an image using a user-defined kernel.

    Parameters
    ----------
    threshold : float
        The absolute image value above which to select sources.

    kernel : `numpy.ndarray`
        A 2D array of the PSF kernel.

    min_separation : float, optional
        The minimum separation for detected objects in pixels.

    exclude_border : bool, optional
        Whether to exclude sources found within half the size of the
        convolution kernel from the image borders.

    brightest : `None` or int, optional
        The number of brightest objects to return in the output table.
        If ``brightest`` is set to `None`, all objects will be returned.

    peakmax : `None` or float, optional
        The maximum allowed peak pixel value in an object. Only objects
        whose maximum pixel values are strictly smaller than ``peakmax``
        will be selected. This may be used to exclude saturated sources.
        If set to `None`, all objects will be selected.

        .. warning::
            `StarFinder` automatically excludes objects whose maximum
            pixel values are negative. Therefore, setting ``peakmax`` to
            a non-positive value would result in excluding all objects.

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.

    The source properties are calculated using image moments.

    See Also
    --------
    DAOStarFinder, IRAFStarFinder
    """

    def __init__(self, threshold, kernel, min_separation=5.0,
                 exclude_border=False, brightest=None, peakmax=None):

        self.threshold = threshold
        self.kernel = kernel
        if min_separation < 0:
            raise ValueError('min_separation must be >= 0')
        self.min_separation = min_separation
        self.exclude_border = exclude_border
        self.brightest = self._validate_brightest(brightest)
        self.peakmax = peakmax

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

            * ``label``: unique object label number.
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
        kernel = self.kernel
        kernel /= np.max(kernel)  # normalize max value to 1.0
        denom = np.sum(kernel**2) - (np.sum(kernel)**2 / kernel.size)
        kernel = (kernel - np.sum(kernel) / kernel.size) / denom

        xypos = _find_stars(data, kernel, self.threshold,
                            min_separation=self.min_separation,
                            mask=mask, exclude_border=self.exclude_border)

        if xypos is None:
            warnings.warn('No sources were found.', NoDetectionsWarning)
            return None

        cat = _StarFinderCatalog(data, xypos, self.kernel.shape)

        # filter the catalog
        if self.peakmax is not None:
            mask = (cat.max_value < self.peakmax)
            cat = cat[mask]

        if len(cat) == 0:
            warnings.warn('Sources were found, but none pass the peakmax '
                          'criterion.', NoDetectionsWarning)
            return None

        # sort the catalog by the brightest fluxes
        if self.brightest is not None:
            idx = np.argsort(cat.flux)[::-1][:self.brightest]
            cat = cat[idx]

        # create the output table
        columns = ('xcentroid', 'ycentroid', 'fwhm', 'roundness', 'pa',
                   'max_value', 'flux', 'mag')
        table = cat.to_table(columns=columns)
        table.add_column(np.arange(len(cat)) + 1, name='label', index=0)

        return table


def _find_stars(data, kernel, threshold, min_separation=0.0, mask=None,
                exclude_border=False):

    convolved_data = _filter_data(data, kernel, mode='constant',
                                  fill_value=0.0, check_normalization=False)

    if min_separation == 0:
        footprint = np.ones(kernel.shape)
    else:
        # define a local circular footprint for the peak finder
        idx = np.arange(-min_separation, min_separation + 1)
        xx, yy = np.meshgrid(idx, idx)
        footprint = np.array((xx**2 + yy**2) <= min_separation**2, dtype=int)

    # pad the data, convolved data, and mask by half the kernel size to
    # allow for detections near the edges
    if not exclude_border:
        ypad = (kernel.shape[0] - 1) // 2
        xpad = (kernel.shape[1] - 1) // 2
        pad = ((ypad, ypad), (xpad, xpad))
        pad_mode = 'constant'
        const_val = 0.
        data = np.pad(data, pad, mode=pad_mode, constant_values=const_val)
        convolved_data = np.pad(convolved_data, pad, mode=pad_mode,
                                constant_values=const_val)
        if mask is not None:
            mask = np.pad(mask, pad, mode=pad_mode, constant_values=const_val)

    # find local peaks in the convolved data
    # suppress any NoDetectionsWarning from find_peaks
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NoDetectionsWarning)
        tbl = find_peaks(convolved_data, threshold, footprint=footprint,
                         mask=mask)

    if tbl is None:
        return None

    xpos, ypos = tbl['x_peak'], tbl['y_peak']
    if not exclude_border:
        xpos -= xpad
        ypos -= ypad

    return np.transpose((xpos, ypos))


class _StarFinderCatalog:
    """
    Class to calculate the properties of each detected star.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image.

    xypos:  list of 2-tuples
        A list of (x, y) tuples denoting the central positions of the
        stars.

    shape:  tuple of int
        The shape of the stars cutouts. The shape in both dimensions
        must be odd and match the shape of the smoothing kernel.
    """

    def __init__(self, data, xypos, shape):
        self.data = data
        self.xypos = np.atleast_2d(xypos)
        self.shape = shape

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        newcls = object.__new__(self.__class__)
        init_attr = ('data', 'shape')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        isscalar = value.shape == (2,)
        setattr(newcls, attr, np.atleast_2d(value))

        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        for key in keys:
            value = self.__dict__[key]
            if key in ('slices', 'cutout_data'):
                # apply fancy indices to list properties
                value = np.array(value + [None], dtype=object)[:-1][index]
                if isscalar:
                    value = [value]  # noqa
                else:
                    value = value.tolist()
            else:
                # always keep as 1D array, even for a single source
                value = np.atleast_1d(value[index])

            newcls.__dict__[key] = value
        return newcls

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).
        """
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)
        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

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

    @lazyproperty
    def cutout_data(self):
        cutout = []
        for slc in self.slices:
            cdata = self.data[slc]
            cdata[cdata < 0] = 0.0  # exclude negative pixels
            cutout.append(cdata)
        return cutout

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
    def xcentroid(self):
        return self.cutout_xcentroid + self.bbox_xmin

    @lazyproperty
    def ycentroid(self):
        return self.cutout_ycentroid + self.bbox_ymin

    @lazyproperty
    def max_value(self):
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
        return np.sqrt(self.mu_diff**2
                       + 4.0 * self.moments_central[:, 1, 1]**2) / self.mu_sum

    @lazyproperty
    def pa(self):
        pa = np.rad2deg(0.5 * np.arctan2(2.0 * self.moments_central[:, 1, 1],
                                         self.mu_diff))
        pa = np.where(pa < 0, pa + 180, pa)
        return pa

    def to_table(self, columns):
        table = QTable()
        for column in columns:
            table[column] = getattr(self, column)
        return table
