# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Base class and star finder kernel for detecting stars in an astronomical
image.

Each star-finding class should define a method called ``find_stars``
that finds stars in an image.
"""

import abc
import inspect
import math
import warnings

import astropy.units as u
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.detection.peakfinder import find_peaks
from photutils.utils._deprecation import (create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import check_units
from photutils.utils._repr import make_repr
from photutils.utils.cutouts import _make_cutouts
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['StarFinderBase', 'StarFinderCatalogBase']

# Remove in 4.0
_DEPRECATED_ATTRIBUTES: dict = {
    'xcentroid': 'x_centroid',
    'ycentroid': 'y_centroid',
    'cutout_xcentroid': 'cutout_x_centroid',
    'cutout_ycentroid': 'cutout_y_centroid',
    'pa': 'orientation',
    'npix': 'n_pixels',
}


class StarFinderBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for star finders.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """
        return self.find_stars(data, mask=mask)

    @staticmethod
    def _find_stars(convolved_data, kernel, threshold, *, min_separation=0.0,
                    mask=None, exclude_border=False):
        """
        Find stars in an image.

        Parameters
        ----------
        convolved_data : 2D array_like
            The convolved 2D array. Should be NaN-free; any NaN values
            should be handled before calling this method.

        kernel : `_StarFinderKernel` or 2D `~numpy.ndarray`
            The convolution kernel. ``StarFinder`` inputs the kernel
            as a 2D array.

        threshold : float or 2D array_like
            The absolute image value above which to select sources. The
            exact value depends on the calling star finder class (e.g.,
            `DAOStarFinder` multiplies the ``threshold`` by the kernel
            relative error, whereas `IRAFStarFinder` and `StarFinder`
            directly use the input ``threshold``). A 2D ``threshold``
            must have the same shape as ``convolved_data``. If
            ``convolved_data`` is a `~astropy.units.Quantity` array,
            then ``threshold`` must have the same units.

        min_separation : float, optional
            The minimum separation for detected objects in pixels.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        exclude_border : bool, optional
            Set to `True` to exclude sources found within half the
            size of the convolution kernel from the image borders.
            The default is `False`, which is the mode used by IRAF's
            `DAOFIND
            <https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/daofind.html>`_
            and `STARFIND
            <https://iraf.readthedocs.io/en/latest/tasks/images/imcoords/starfind.html>`_.

        Returns
        -------
        result : Nx2 `~numpy.ndarray` or `None`
            An Nx2 array containing the (x, y) pixel coordinates. `None`
            is returned if no sources are found.
        """
        # Define a local footprint for the peak finder
        find_peaks_kwargs = {}
        if min_separation == 0:  # use kernel-shape footprint
            if isinstance(kernel, np.ndarray):
                footprint = np.ones(kernel.shape)
            else:
                footprint = kernel.mask.astype(bool)
            find_peaks_kwargs['footprint'] = footprint
        else:
            find_peaks_kwargs['min_separation'] = min_separation

        # Define the border exclusion region
        if exclude_border:
            if isinstance(kernel, np.ndarray):
                yborder = (kernel.shape[0] - 1) // 2
                xborder = (kernel.shape[1] - 1) // 2
            else:
                yborder = kernel.y_radius
                xborder = kernel.x_radius
            border_width = (yborder, xborder)
        else:
            border_width = None

        # Find local peaks in the convolved data.
        # Suppress any NoDetectionsWarning from find_peaks.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NoDetectionsWarning)
            tbl = find_peaks(convolved_data, threshold, mask=mask,
                             border_width=border_width, **find_peaks_kwargs)

        if tbl is None:
            return None

        return np.transpose((tbl['x_peak'], tbl['y_peak']))

    @abc.abstractmethod
    def find_stars(self, data, *, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """


class StarFinderCatalogBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for star finder catalogs.

    This class provides common functionality for catalog classes that
    store and compute properties of detected sources. External packages
    may subclass it to create custom star finder catalogs.

    Subclasses **must** implement:

    * :attr:`x_centroid` property -- Object centroid in the x direction.
    * :attr:`y_centroid` property -- Object centroid in the y direction.
    * `apply_filters` method -- Filter the catalog using
      algorithm-specific criteria.
    * ``default_columns`` attribute -- A tuple of column names used
      by `to_table` when no explicit columns are given. This should
      be set in the subclass ``__init__``.

    Subclasses **may** override:

    * `_get_init_attributes` -- Return attribute names to copy
      during slicing. The override should include
      ``'default_columns'`` in the returned tuple.
    * `make_cutouts` -- Customize how cutout arrays are extracted.
    * `cutout_data` -- Customize the cutouts used for photometry
      (e.g., zeroing negative pixels).

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image. The image should be background-subtracted.

    xypos : Nx2 `~numpy.ndarray`
        An Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    kernel : 2D `~numpy.ndarray`
        A 2D array of the PSF kernel. Internally, the star finder
        classes may also pass a kernel object.

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

    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, data, xypos, kernel, *, n_brightest=None,
                 peak_max=None):
        # Validate the units
        check_units((data, peak_max), ('data', 'peak_max'))

        self.data = data
        unit = data.unit if isinstance(data, u.Quantity) else None
        self.unit = unit
        self.kernel = kernel
        self.cutout_shape = kernel.shape

        self.xypos = np.atleast_2d(xypos)
        self.n_brightest = n_brightest
        self.peak_max = peak_max
        self.default_columns = ()

        self.id = np.arange(len(self)) + 1

    def __repr__(self):
        params = ('nsources',)
        overrides = {'nsources': len(self)}
        return make_repr(self, params, brackets=True, overrides=overrides)

    def __str__(self):
        params = ('nsources',)
        overrides = {'nsources': len(self)}
        return make_repr(self, params, overrides=overrides, long=True)

    def __len__(self):
        return len(self.xypos)

    def __getitem__(self, index):
        """
        Index or slice the catalog.

        This method should be overridden in subclasses to handle
        class-specific attributes.
        """
        # NOTE: we allow indexing/slicing of scalar (self.isscalar = True)
        #       instances in order to perform catalog filtering even for
        #       a single source

        newcls = object.__new__(self.__class__)

        # Get attributes to copy from subclass
        init_attr = self._get_init_attributes()
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # xypos determines ordering and isscalar
        # NOTE: always keep as 2D array, even for a single source
        attr = 'xypos'
        value = getattr(self, attr)[index]
        setattr(newcls, attr, np.atleast_2d(value))

        # Index/slice the remaining attributes
        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        keys.add('id')
        for key in keys:
            value = self.__dict__[key]

            # Do not insert lazy attributes that are always scalar (e.g.,
            # isscalar), i.e., not an array/list for each source
            if np.isscalar(value):
                continue

            # Ensure value is always at least a 1D array, even for a
            # single source
            value = np.atleast_1d(value[index])

            newcls.__dict__[key] = value

        return newcls

    def _get_init_attributes(self):
        """
        Return a tuple of attribute names to copy during slicing.

        This method should be overridden in subclasses.
        """
        return ('data', 'unit', 'kernel', 'n_brightest', 'peak_max',
                'cutout_shape', 'default_columns')

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).

        The result is cached on the class to avoid repeated
        introspection via `inspect.getmembers`.
        """
        cls = self.__class__
        attr = '_cached_lazyproperties'
        # Subclasses get their own lazyproperty list
        if attr not in cls.__dict__:
            def islazyproperty(obj):
                return isinstance(obj, lazyproperty)

            setattr(cls, attr,
                    [i[0] for i in inspect.getmembers(
                        cls, predicate=islazyproperty)])
        return getattr(cls, attr)

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self.xypos.shape == (1, 2)

    def reset_ids(self):
        """
        Reset the ID column to be consecutive integers.
        """
        self.id = np.arange(len(self)) + 1

    def make_cutouts(self, data):
        """
        Make cutouts from the data array.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array.

        Returns
        -------
        cutouts : 3D `~numpy.ndarray`
            The cutout arrays.
        """
        data_arr = data.value if isinstance(data, u.Quantity) else data

        cutouts, _ = _make_cutouts(data_arr, self.xypos[:, 0],
                                   self.xypos[:, 1], self.cutout_shape)

        if self.unit is not None:
            cutouts <<= self.unit

        return cutouts

    @lazyproperty
    def cutout_data(self):
        """
        The cutout data arrays.

        Subclasses may override this property to customize the cutouts
        used for moment-based photometry calculations (e.g., zeroing
        negative pixels or subtracting a local sky background).
        """
        return self.make_cutouts(self.data)

    @lazyproperty
    def moments(self):
        """
        The raw image moments.
        """
        data = self.cutout_data
        if isinstance(data, u.Quantity):
            data = data.value
        ky, kx = data.shape[1], data.shape[2]
        y = np.arange(ky, dtype=float)
        x = np.arange(kx, dtype=float)
        ypowers = np.column_stack([np.ones(ky), y])  # (ky, 2)
        xpowers = np.column_stack([np.ones(kx), x])  # (kx, 2)
        # M[n, p, q] = sum_jk data[n,j,k] * y[j]^p * x[k]^q
        return ypowers.T @ data @ xpowers

    @lazyproperty
    def moments_central(self):
        """
        The central image moments.
        """
        data = self.cutout_data
        if isinstance(data, u.Quantity):
            data = data.value
        ky, kx = data.shape[1], data.shape[2]
        y = np.arange(ky, dtype=float)
        x = np.arange(kx, dtype=float)
        # Per-source shifted coordinates
        dy = y[np.newaxis, :] - self.cutout_y_centroid[:, np.newaxis]
        dx = x[np.newaxis, :] - self.cutout_x_centroid[:, np.newaxis]
        # Per-source power arrays: (n, ky, 3) and (n, kx, 3)
        ypowers = np.stack([np.ones_like(dy), dy, dy**2], axis=-1)
        xpowers = np.stack([np.ones_like(dx), dx, dx**2], axis=-1)
        # Batched matmul: ypowers^T @ data @ xpowers per source
        moments = (np.transpose(ypowers, (0, 2, 1)) @ data @ xpowers)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return moments / self.moments[:, 0, 0][:, np.newaxis, np.newaxis]

    @lazyproperty
    def cutout_centroid(self):
        """
        The cutout centroids.
        """
        moments = self.moments

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            y_centroid = moments[:, 1, 0] / moments[:, 0, 0]
            x_centroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((y_centroid, x_centroid))

    @lazyproperty
    def cutout_x_centroid(self):
        """
        The cutout x centroids.
        """
        return np.transpose(self.cutout_centroid)[1]

    @lazyproperty
    def cutout_y_centroid(self):
        """
        The cutout y centroids.
        """
        return np.transpose(self.cutout_centroid)[0]

    @property
    @abc.abstractmethod
    def x_centroid(self):
        """
        Object centroid in the x direction.

        This property must be implemented in subclasses.
        """

    @property
    @abc.abstractmethod
    def y_centroid(self):
        """
        Object centroid in the y direction.

        This property must be implemented in subclasses.
        """

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @lazyproperty
    def mu_sum(self):
        """
        The sum of the central moments.
        """
        return (self.moments_central[:, 0, 2]
                + self.moments_central[:, 2, 0])

    @lazyproperty
    def mu_diff(self):
        """
        The difference of the central moments.
        """
        return (self.moments_central[:, 0, 2]
                - self.moments_central[:, 2, 0])

    @lazyproperty
    def fwhm(self):
        """
        The FWHM of the sources.
        """
        return 2.0 * np.sqrt(np.log(2.0) * self.mu_sum)

    @lazyproperty
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction and
        will be in the range [0, 360) degrees.
        """
        angle = 0.5 * np.arctan2(2.0 * self.moments_central[:, 1, 1],
                                 self.mu_diff)
        return (np.rad2deg(angle) % 360) << u.deg

    @lazyproperty
    def roundness(self):
        """
        The roundness of the sources.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (np.sqrt(self.mu_diff**2
                            + 4.0 * self.moments_central[:, 1, 1]**2)
                    / self.mu_sum)

    @lazyproperty
    def peak(self):
        """
        The peak pixel values.
        """
        return np.max(self.cutout_data, axis=(1, 2))

    @lazyproperty
    def flux(self):
        """
        The instrumental fluxes.
        """
        return np.sum(self.cutout_data, axis=(1, 2))

    @lazyproperty
    def mag(self):
        """
        The instrumental magnitudes.
        """
        # Ignore RuntimeWarning if flux is <= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flux = self.flux
            if isinstance(flux, u.Quantity):
                flux = flux.value
            return -2.5 * np.log10(flux)

    def select_brightest(self):
        """
        Sort the catalog by the brightest fluxes and select the top
        brightest sources.
        """
        newcat = self
        if self.n_brightest is not None:
            idx = np.argsort(self.flux)[::-1][:self.n_brightest]
            newcat = self[idx]
        return newcat

    def _filter_finite(self, attrs, *, initial_mask=None,
                       skip_attrs=()):
        """
        Filter the catalog by removing sources with non-finite values.

        Parameters
        ----------
        attrs : tuple of str
            Attribute names to check for finiteness.

        initial_mask : 1D `~numpy.ndarray` of bool or `None`, optional
            A pre-existing boolean mask to combine with. If `None`,
            starts with all `True`.

        skip_attrs : tuple of str, optional
            Attribute names to skip during finiteness checking.

        Returns
        -------
        catalog : ``self.__class__`` or `None`
            The filtered catalog, or `None` if no sources remain.
        """
        if initial_mask is None:
            mask = np.ones(len(self), dtype=bool)
        else:
            mask = initial_mask.copy()

        for attr in attrs:
            if attr in skip_attrs:
                continue
            mask &= np.isfinite(getattr(self, attr))
        newcat = self[mask]

        if len(newcat) == 0:
            msg = 'No sources were found.'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        return newcat

    def _filter_bounds(self, bounds, *, initial_mask=None, peakattr='peak'):
        """
        Filter the catalog by sharpness, roundness, and peak_max bounds.

        Parameters
        ----------
        bounds : list of tuple
            Each tuple is ``(attr_name, range)`` giving the attribute to
            check and the range of allowed values. The range is a tuple
            of the form ``(lower_bound, upper_bound)``, or `None` to
            skip filtering for that attribute.

        initial_mask : 1D `~numpy.ndarray` of bool or `None`, optional
            A pre-existing boolean mask to combine with. If `None`,
            starts with all `True`.

        peakattr : str, optional
            The attribute name for the peak value used for peak_max
            filtering. The default is ``'peak'``.

        Returns
        -------
        catalog : ``self.__class__`` or `None`
            The filtered catalog, or `None` if no sources remain.
        """
        if initial_mask is None:
            mask = np.ones(len(self), dtype=bool)
        else:
            mask = initial_mask.copy()

        for attr, range_val in bounds:
            if range_val is None:
                continue
            min_val, max_val = range_val
            values = getattr(self, attr)
            mask &= (values >= min_val)
            mask &= (values <= max_val)

        # peak_max filtering is applied separately from the bounds list
        # because it uses a different attribute (peakattr) and is always
        # a single upper bound, not a range.
        if self.peak_max is not None:
            mask &= (getattr(self, peakattr) <= self.peak_max)

        newcat = self[mask]

        if len(newcat) == 0:
            msg = 'Sources were found, but none pass the filtering criteria'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        return newcat

    @abc.abstractmethod
    def apply_filters(self):
        """
        Filter the catalog.

        This method must be implemented in subclasses to apply
        algorithm-specific filtering criteria.
        """

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

    def to_table(self, *, columns=None):
        """
        Create a QTable of catalog properties.

        Parameters
        ----------
        columns : list of str, optional
            List of column names to include in the table. If `None`,
            uses ``self.default_columns``.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of the catalog properties.
        """
        # Replace with QTable in 4.0
        table = create_empty_deprecated_qtable(
            _DEPRECATED_ATTRIBUTES, since='3.0', until='4.0')

        table.meta.update(_get_meta())  # keep table.meta type
        if columns is None:
            if not self.default_columns:
                msg = ('default_columns attribute is not set; either '
                       'pass explicit column names or set '
                       'default_columns in the subclass __init__')
                raise AttributeError(msg)
            columns = self.default_columns
        for column in columns:
            table[column] = getattr(self, column)
        return table


class _StarFinderKernel:
    """
    Container class for a 2D Gaussian density enhancement kernel.

    The kernel has negative wings and sums to zero. It is used by both
    `DAOStarFinder` and `IRAFStarFinder`.

    Parameters
    ----------
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor and major axis standard deviations of
        the Gaussian kernel. ``ratio`` must be strictly positive and
        less than or equal to 1.0. The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel, measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units
        of sigma (standard deviation) [``1 sigma = FWHM /
        (2.0 * sqrt(2.0 * log(2.0)))``]. The default is 1.5.

    normalize_zerosum : bool, optional
        Whether to normalize the Gaussian kernel to have zero sum. The
        default is `True`, which generates a density-enhancement kernel.

    Notes
    -----
    The class attributes include the dimensions of the elliptical kernel
    and the coefficients of a 2D elliptical Gaussian function expressed
    as:

    .. math::
        f(x, y) = A \\exp\\bigl(-g(x, y)\\bigr)

    where

    .. math::
        g(x, y) = a (x - x_0)^{2} + 2 b (x - x_0)(y - y_0)
                   + c (y - y_0)^{2}

    Attributes
    ----------
    data : 2D `~numpy.ndarray`
        The kernel data array, used for convolution.

    shape : tuple of int
        The ``(ny, nx)`` shape of ``data``.

    mask : 2D `~numpy.ndarray` of int
        Binary mask (1 inside the kernel footprint, 0 outside).
        Used for the peak-finding footprint, sharpness computation
        in `_DAOStarFinderCatalog`, and sky estimation in
        `_IRAFStarFinderCatalog`.

    rel_err : float
        The kernel relative error, used by `DAOStarFinder` to scale the
        detection threshold.

    gaussian_kernel_unmasked : 2D `~numpy.ndarray`
        The unmasked Gaussian kernel (peak normalized to 1), used by
        `_DAOStarFinderCatalog` for marginal fitting.

    x_sigma, y_sigma : float
        Standard deviations along the major and minor axes.

    x_radius, y_radius : int
        Half-widths of the kernel array in pixels.

    n_pixels : int
        Total number of pixels within the kernel ``mask``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function
    """

    def __init__(self, fwhm, *, ratio=1.0, theta=0.0, sigma_radius=1.5,
                 normalize_zerosum=True):

        if np.ndim(fwhm) != 0:
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        if fwhm <= 0:
            msg = 'fwhm must be positive'
            raise ValueError(msg)

        if ratio <= 0 or ratio > 1:
            msg = 'ratio must be > 0 and <= 1.0'
            raise ValueError(msg)

        if sigma_radius <= 0:
            msg = 'sigma_radius must be positive'
            raise ValueError(msg)

        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta % 360.0
        self.sigma_radius = sigma_radius
        self.x_sigma = self.fwhm * gaussian_fwhm_to_sigma
        self.y_sigma = self.x_sigma * self.ratio

        theta_radians = np.deg2rad(self.theta)
        cost = np.cos(theta_radians)
        sint = np.sin(theta_radians)
        x_sigma2 = self.x_sigma**2
        y_sigma2 = self.y_sigma**2

        a = (cost**2 / (2.0 * x_sigma2)) + (sint**2 / (2.0 * y_sigma2))
        # Counterclockwise rotation
        b = 0.5 * cost * sint * ((1.0 / x_sigma2) - (1.0 / y_sigma2))
        c = (sint**2 / (2.0 * x_sigma2)) + (cost**2 / (2.0 * y_sigma2))

        # Find the extent of an ellipse with radius = sigma_radius*sigma.
        # Solve for the horizontal and vertical tangents of an ellipse
        # defined by g(x,y) = f.
        f = self.sigma_radius**2 / 2.0
        denom = (a * c) - b**2

        # Ensure nx and ny are always odd.
        # The minimum kernel size is 5x5.
        nx = 2 * int(max(2, math.sqrt(c * f / denom))) + 1
        ny = 2 * int(max(2, math.sqrt(a * f / denom))) + 1

        self.x_radius = nx // 2
        self.y_radius = ny // 2

        # Define the kernel on a 2D grid
        xc = self.x_radius
        yc = self.y_radius
        yy, xx = np.mgrid[0:ny, 0:nx]
        circular_radius = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        elliptical_radius = (a * (xx - xc)**2
                             + 2.0 * b * (xx - xc) * (yy - yc)
                             + c * (yy - yc)**2)

        self.mask = np.where(
            (elliptical_radius <= f)
            | (circular_radius <= 2.0), 1, 0).astype(int)
        self.n_pixels = self.mask.sum()

        # Central (peak) pixel of gaussian_kernel has a value of 1.0
        self.gaussian_kernel_unmasked = np.exp(-elliptical_radius)
        gaussian_kernel = self.gaussian_kernel_unmasked * self.mask

        # The denom represents (variance * n_pixels)
        denom = ((gaussian_kernel**2).sum()
                 - (gaussian_kernel.sum()**2 / self.n_pixels))
        self.rel_err = 1.0 / np.sqrt(denom)

        # Normalize the kernel to zero sum
        if normalize_zerosum:
            self.data = ((gaussian_kernel
                          - (gaussian_kernel.sum() / self.n_pixels))
                         / denom) * self.mask
        else:
            self.data = gaussian_kernel

        self.shape = self.data.shape

    def __repr__(self):
        params = ('fwhm', 'ratio', 'theta', 'sigma_radius')
        return make_repr(self, params)

    def __str__(self):
        params = ('fwhm', 'ratio', 'theta', 'sigma_radius')
        return make_repr(self, params, long=True)


def _validate_n_brightest(n_brightest):
    """
    Validate the ``n_brightest`` parameter.

    It must be >0 and an integer.

    Parameters
    ----------
    n_brightest : int, None, or bool
        The number of brightest sources to select. If `None`, all
        sources are selected. If a boolean is passed, a `TypeError` is
        raised.
    """
    if n_brightest is not None:
        if isinstance(n_brightest, bool):
            msg = 'n_brightest must be an integer'
            raise TypeError(msg)
        if n_brightest <= 0:
            msg = 'n_brightest must be > 0'
            raise ValueError(msg)
        bright_int = int(n_brightest)
        if bright_int != n_brightest:
            msg = 'n_brightest must be an integer'
            raise ValueError(msg)
        n_brightest = bright_int
    return n_brightest


def _handle_deprecated_range(old_lower, old_upper, new_range,
                             old_name, new_name, default_range):
    """
    Handle deprecated lower/upper bound parameters replaced by a single
    range parameter.

    Parameters
    ----------
    old_lower : float or `_DeprecatedDefault`
        The deprecated lower-bound parameter value.

    old_upper : float or `_DeprecatedDefault`
        The deprecated upper-bound parameter value.

    new_range : tuple of 2 floats or `None`
        The new range parameter value.

    old_name : str
        The base name of the deprecated parameters (e.g., ``'sharp'``
        for ``'sharplo'`` / ``'sharphi'``).

    new_name : str
        The name of the new range parameter (e.g.,
        ``'sharpness_range'``).

    default_range : tuple of 2 floats
        The default range values when ``new_range`` is `None`.

    Returns
    -------
    result : tuple of 2 floats or `None`
        The resolved range.
    """
    if old_lower is not _DEPR_DEFAULT or old_upper is not _DEPR_DEFAULT:
        msg = (f"The '{old_name}lo' and '{old_name}hi' parameters are "
               'deprecated and will be removed in a future version. '
               f"Use '{new_name}=(lower, upper)' instead.")
        warnings.warn(msg, AstropyDeprecationWarning)
        _default = new_range if new_range is not None else default_range
        lower = (old_lower if old_lower is not _DEPR_DEFAULT
                 else _default[0])
        upper = (old_upper if old_upper is not _DEPR_DEFAULT
                 else _default[1])
        return (lower, upper)
    return new_range


class _DeprecatedDefault:
    """
    Sentinel default value for a deprecated parameter.
    """

    def __repr__(self):
        return '<deprecated>'


_DEPR_DEFAULT = _DeprecatedDefault()
