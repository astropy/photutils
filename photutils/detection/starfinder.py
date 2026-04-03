# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
StarFinder class.
"""

import warnings

import numpy as np
from astropy.utils import lazyproperty

from photutils.detection.core import (StarFinderBase, StarFinderCatalogBase,
                                      _validate_n_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._quantity_helpers import check_units
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinder']


class StarFinder(StarFinderBase):
    """
    Detect stars in an image using a user-defined kernel.

    Parameters
    ----------
    threshold : float or 2D `~numpy.ndarray`
        The absolute image value above which to select sources. If
        ``threshold`` is a 2D array, it must have the same shape as the
        input ``data``. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

    kernel : `~numpy.ndarray`
        A 2D array of the PSF kernel.

    min_separation : `None` or float, optional
        The minimum separation (in pixels) for detected objects. If
        `None` (default) then the minimum separation is set to ``2.5 *
        (min(kernel.shape) // 2)``. Note that large values may result in
        long run times.

        .. versionchanged:: 3.0
            The default ``min_separation`` changed from 5 to ``2.5
            * (min(kernel.shape) // 2)``. To recover the previous
            behavior, set ``min_separation=5``.

    exclude_border : bool, optional
        Whether to exclude sources found within half the size of the
        convolution kernel from the image borders.

    n_brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``n_brightest`` is set to `None`, all objects
        will be selected.

    peak_max : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peak_max`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peak_max`` must have the
        same units. If ``peak_max`` is set to `None`, then no peak pixel
        value filtering will be performed.

    See Also
    --------
    DAOStarFinder, IRAFStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peak_max``
    must all have the same units as the image.

    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.

    The source properties are calculated using image moments.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, kernel, min_separation=None,
                 exclude_border=False, n_brightest=None, peak_max=None):

        # Validate the units
        check_units((threshold, peak_max), ('threshold', 'peak_max'))

        self.threshold = threshold

        kernel = np.asarray(kernel)
        if kernel.ndim != 2:
            msg = 'kernel must be a 2D array'
            raise ValueError(msg)
        self.kernel = kernel

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * (min(self.kernel.shape) // 2)
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

    def _repr_str_params(self):
        params = ('threshold', 'kernel', 'min_separation',
                  'exclude_border', 'n_brightest', 'peak_max')
        overrides = {'kernel': f'<array; shape={self.kernel.shape}>'}
        if not np.isscalar(self.threshold):
            overrides['threshold'] = (
                f'<array; shape={np.shape(self.threshold)}>')
        return params, overrides

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def _get_raw_catalog(self, data, *, mask=None):
        """
        Get the raw catalog of sources from the input data.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array. The image should be
            background-subtracted.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        cat : `_StarFinderCatalog` or `None`
            A catalog of sources found in the input data. `None` is
            returned if no sources are found.
        """
        kernel = self.kernel / np.max(self.kernel)  # normalize max to 1.0
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
            msg = 'No sources were found.'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        return _StarFinderCatalog(data, xypos, self.kernel,
                                  n_brightest=self.n_brightest,
                                  peak_max=self.peak_max)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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
            * ``x_centroid, y_centroid``: object centroid.
            * ``fwhm``: object FWHM.
            * ``roundness``: object roundness.
            * ``orientation``: the angle between the ``x`` axis and the
              major axis source measured counter-clockwise in the range
              [0, 360) degrees.
            * ``max_value``: the maximum pixel value in the source
            * ``flux``: the source instrumental flux.
            * ``mag``: the source instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.

            `None` is returned if no stars are found or no stars meet
            the peak_max criteria.
        """
        # Validate the units
        check_units((data, self.threshold, self.peak_max),
                    ('data', 'threshold', 'peak_max'))

        cat = self._get_raw_catalog(data, mask=mask)
        if cat is None:
            return None

        # Apply all selection filters
        cat = cat.apply_all_filters()
        if cat is None:
            return None

        # Create the output table
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

    n_brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``n_brightest`` is set to `None`, all objects
        will be selected.

    peak_max : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peak_max`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peak_max`` must have the
        same units. If ``peak_max`` is set to `None`, then no peak pixel
        value filtering will be performed.
    """

    def __init__(self, data, xypos, kernel, *, n_brightest=None,
                 peak_max=None):
        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)
        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'fwhm',
                                'roundness', 'orientation', 'max_value',
                                'flux', 'mag')

    def _get_init_attributes(self):
        """
        Return a tuple of attribute names to copy during slicing.
        """
        return ('data', 'unit', 'kernel', 'n_brightest', 'peak_max',
                'cutout_shape', 'default_columns')

    @lazyproperty
    def cutout_data(self):
        """
        The cutout data arrays with negative values set to zero.
        """
        cutouts = self.make_cutouts(self.data)
        cutouts[cutouts < 0] = 0.0  # exclude negative pixels
        return cutouts

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value in the cutout data.
        """
        return self.peak

    @lazyproperty
    def x_centroid(self):
        """
        The x centroid of the source.
        """
        xoff = self.cutout_shape[1] // 2
        return self.cutout_x_centroid + self.xypos[:, 0] - xoff

    @lazyproperty
    def y_centroid(self):
        """
        The y centroid of the source.
        """
        yoff = self.cutout_shape[0] // 2
        return self.cutout_y_centroid + self.xypos[:, 1] - yoff

    def apply_filters(self):
        """
        Filter the catalog.
        """
        attrs = ('x_centroid', 'y_centroid', 'fwhm', 'roundness',
                 'orientation', 'max_value', 'flux')
        newcat = self._filter_finite(attrs)
        if newcat is None:
            return None

        return newcat._filter_bounds([], peakattr='max_value')
