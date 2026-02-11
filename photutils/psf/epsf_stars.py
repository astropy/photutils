# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for extracting cutouts of stars and data structures to hold the
cutouts for fitting and building ePSFs.
"""

import warnings

import numpy as np
from astropy.nddata import (NDData, NoOverlapError, PartialOverlapError,
                            StdDevUncertainty, overlap_slices)
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import BoundingBox
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import as_pair

__all__ = ['EPSFStar', 'EPSFStars', 'LinkedEPSFStar', 'extract_stars']


class EPSFStar:
    """
    A class to hold a 2D cutout image and associated metadata of a star
    used to build an ePSF.

    Parameters
    ----------
    data : `~numpy.ndarray`
        A 2D cutout image of a single star.

    weights : `~numpy.ndarray` or `None`, optional
        A 2D array of the weights associated with the input ``data``.

    cutout_center : tuple of two floats or `None`, optional
        The ``(x, y)`` position of the star's center with respect to the
        input cutout ``data`` array. If `None`, then the center of the
        input cutout ``data`` array will be used.

    flux : float or `None`, optional
        The flux of the star. If `None`, then the flux will be estimated
        from the input ``data``.

    origin : tuple of two int, optional
        The ``(x, y)`` index of the origin (bottom-left corner) pixel
        of the input cutout array with respect to the original array
        from which the cutout was extracted. This can be used to convert
        positions within the cutout image to positions in the original
        image. ``origin`` and ``wcs_large`` must both be input for a
        linked star (a single star extracted from different images).

    wcs_large : `None` or WCS object, optional
        A WCS object associated with the large image from which
        the cutout array was extracted. It should not be the
        WCS object of the input cutout ``data`` array. The WCS
        object must support the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). ``origin`` and ``wcs_large``
        must both be input for a linked star (a single star extracted
        from different images).

    id_label : int, str, or `None`, optional
        An optional identification number or label for the star.
    """

    def __init__(self, data, *, weights=None, cutout_center=None, flux=None,
                 origin=(0, 0), wcs_large=None, id_label=None):

        self._data = np.asanyarray(data)

        # Validate data dimensionality and shape
        if self._data.ndim != 2:
            msg = f'Input data must be 2-dimensional, got {self._data.ndim}D'
            raise ValueError(msg)
        if self._data.size == 0:
            msg = 'Input data cannot be empty'
            raise ValueError(msg)

        self.shape = self._data.shape

        # Validate and process weights
        if weights is not None:
            weights = np.asanyarray(weights)
            if weights.shape != self._data.shape:
                msg = (f'Weights shape {weights.shape} must match data shape '
                       f'{self._data.shape}')
                raise ValueError(msg)

            # Check for valid weight values
            if not np.all(np.isfinite(weights)):
                warnings.warn('Non-finite weight values detected. These will '
                              'be set to zero.', AstropyUserWarning)
                weights = np.where(np.isfinite(weights), weights, 0.0)

            # Copy to avoid modifying the input weights
            self.weights = weights.astype(float, copy=True)
        else:
            self.weights = np.ones_like(self._data, dtype=float)

        # Create initial mask from weights
        self.mask = (self.weights <= 0.0)

        # Mask out invalid image data and provide informative warning
        invalid_data = ~np.isfinite(self._data)
        if np.any(invalid_data):
            self.weights[invalid_data] = 0.0
            self.mask[invalid_data] = True
            warnings.warn('Input data array contains invalid data that '
                          'will be masked.', AstropyUserWarning)

        # Validate origin
        origin = np.asarray(origin)
        if origin.shape != (2,):
            msg = f'Origin must have exactly 2 elements, got {len(origin)}'
            raise ValueError(msg)
        if not np.all(np.isfinite(origin)):
            msg = 'Origin coordinates must be finite'
            raise ValueError(msg)
        self.origin = origin.astype(int)

        self.wcs_large = wcs_large
        self.id_label = id_label

        if cutout_center is None:
            cutout_center = ((self.shape[1] - 1) / 2.0,
                             (self.shape[0] - 1) / 2.0)

        # Set cutout_center (triggers validation via setter)
        self.cutout_center = cutout_center

        # Keep track of the original center position (before fitting)
        # for reference
        self._center_original = cutout_center + self.origin

        if flux is not None:
            self.flux = float(flux)
            self._has_all_zero_data = False  # Unknown for explicit flux
        else:
            # Check if completely masked before attempting flux estimation
            if np.all(self.mask):
                msg = ('Star cutout is completely masked; no valid data '
                       'available')
                raise ValueError(msg)

            # Check if all unmasked data values are exactly zero
            # Store flag for later warning (to avoid duplicate warnings)
            unmasked_data = self._data[~self.mask]
            self._has_all_zero_data = bool(np.all(unmasked_data == 0.0))

            # Warn if all data is zero
            if self._has_all_zero_data:
                warnings.warn('All unmasked data values in star cutout '
                              'are zero', AstropyUserWarning)

            # Estimate flux
            self.flux = self.estimate_flux()

            # Note: We allow flux <= 0 for real sources that may have
            # negative net flux due to background subtraction or similar
            # effects

        self._excluded_from_fit = False
        self._fit_error_status = 0  # 0: no error, >0: error during fitting
        self._fitinfo = None

    def __array__(self):
        """
        Array representation of the data array (e.g., for matplotlib).
        """
        return self._data

    @property
    def data(self):
        """
        The 2D cutout image.
        """
        return self._data

    @property
    def cutout_center(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of the star's
        center with respect to the input cutout ``data`` array.

        Initially set to the geometric center of the cutout, this value
        is updated during ePSF building iterations to reflect the fitted
        center position as the star is aligned with the ePSF model.
        """
        return self._cutout_center

    @cutout_center.setter
    def cutout_center(self, value):
        # Convert to array-like for validation
        value = np.asarray(value)

        # Validate shape
        if value.shape != (2,):
            msg = ('cutout_center must have exactly two elements in '
                   f'(x, y) form, got shape {value.shape}')
            raise ValueError(msg)

        # Validate finite values
        if not np.all(np.isfinite(value)):
            msg = 'All cutout_center coordinates must be finite'
            raise ValueError(msg)

        # Validate bounds (should be within the cutout image)
        x, y = value
        if not (0 <= x < self.shape[1]):
            warnings.warn(f'cutout_center x-coordinate {x} is outside '
                          f'the cutout bounds [0, {self.shape[1]})',
                          AstropyUserWarning)
        if not (0 <= y < self.shape[0]):
            warnings.warn(f'cutout_center y-coordinate {y} is outside '
                          f'the cutout bounds [0, {self.shape[0]})',
                          AstropyUserWarning)

        self._cutout_center = np.asarray(value)

    @property
    def center(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of the star's
        center in the original (large) image (not the cutout image).
        """
        return self.cutout_center + self.origin

    @lazyproperty
    def slices(self):
        """
        A tuple of two slices representing the cutout region with
        respect to the original (large) image.
        """
        return (slice(self.origin[1], self.origin[1] + self.shape[1]),
                slice(self.origin[0], self.origin[0] + self.shape[0]))

    @lazyproperty
    def bbox(self):
        """
        The minimal `~photutils.aperture.BoundingBox` for the cutout
        region with respect to the original (large) image.
        """
        return BoundingBox(self.slices[1].start, self.slices[1].stop,
                           self.slices[0].start, self.slices[0].stop)

    def estimate_flux(self):
        """
        Estimate the star's flux by summing values in the input cutout
        array.

        Missing data is filled in by interpolation to better estimate
        the total flux.

        Returns
        -------
        flux : float
            The estimated star's flux. If there is no valid data in the
            cutout, `numpy.nan` will be returned.
        """
        if not np.any(self.mask):
            return float(np.sum(self.data))

        # Interpolate missing data to estimate total flux
        data_interp = _interpolate_missing_data(self.data, mask=self.mask,
                                                method='cubic')
        return float(np.sum(data_interp))

    def register_epsf(self, epsf):
        """
        Register and scale (in flux) the input ``epsf`` to the star.

        Parameters
        ----------
        epsf : `ImagePSF`
            The ePSF to register.

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the registered/scaled ePSF.
        """
        # evaluate the input ePSF on the star cutout grid
        yy, xx = np.indices(self.shape, dtype=float)
        return epsf.evaluate(xx, yy, flux=self.flux,
                             x_0=self.cutout_center[0],
                             y_0=self.cutout_center[1])

    def compute_residual_image(self, epsf):
        """
        Compute the residual image of the star data minus the
        registered/scaled ePSF.

        Parameters
        ----------
        epsf : `ImagePSF`
            The ePSF to subtract.

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the residual image.
        """
        return self.data - self.register_epsf(epsf)

    @property
    def _xyidx_centered(self):
        """
        1D arrays of x and y indices of unmasked pixels, with respect
        to the star center, in the cutout reference frame.

        Returns
        -------
        x_centered, y_centered : tuple of `~numpy.ndarray`
            The x and y indices centered on the star position.
        """
        yidx, xidx = np.indices(self._data.shape)
        x_centered = xidx[~self.mask].ravel() - self.cutout_center[0]
        y_centered = yidx[~self.mask].ravel() - self.cutout_center[1]
        return x_centered, y_centered

    @lazyproperty
    def _data_values_normalized(self):
        """
        1D array of unmasked cutout data values, normalized by the
        star's total flux.
        """
        return self.data[~self.mask].ravel() / self.flux


class EPSFStars:
    """
    Class to hold a list of `EPSFStar` and/or `LinkedEPSFStar` objects.

    Parameters
    ----------
    stars_list : list of `EPSFStar` or `LinkedEPSFStar` objects
        A list of `EPSFStar` and/or `LinkedEPSFStar` objects.
    """

    def __init__(self, stars_list):
        if isinstance(stars_list, (EPSFStar, LinkedEPSFStar)):
            self._data = [stars_list]
        elif isinstance(stars_list, list):
            self._data = stars_list
        else:
            msg = ('stars_list must be a list of EPSFStar and/or '
                   'LinkedEPSFStar objects')
            raise TypeError(msg)

    def __len__(self):
        """
        Return the number of stars in this container.
        """
        return len(self._data)

    def __getitem__(self, index):
        """
        Return a new EPSFStars instance containing the indexed star(s).
        """
        return self.__class__(self._data[index])

    def __delitem__(self, index):
        """
        Delete the star at the given index.
        """
        del self._data[index]

    def __iter__(self):
        """
        Iterate over the stars in this container.
        """
        yield from self._data

    def __getstate__(self):
        """
        Return state for pickling (avoids __getattr__ recursion).
        """
        return self.__dict__

    def __setstate__(self, d):
        """
        Restore state from pickling.
        """
        self.__dict__ = d

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying star list.

        This allows accessing star attributes (like ``cutout_center``,
        ``center``, ``flux``) directly on the EPSFStars container,
        returning an array of values from all contained stars.
        """
        result = [getattr(star, attr) for star in self._data]
        if attr in ['cutout_center', 'center', 'flux', '_excluded_from_fit']:
            result = np.array(result)
        if len(self._data) == 1:
            result = result[0]
        return result

    @property
    def cutout_center_flat(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of all the stars'
        centers (including linked stars) with respect to the input
        cutout ``data`` array, as a 2D array (``n_all_stars`` x 2).

        Note that when `EPSFStars` contains any `LinkedEPSFStar`, the
        ``cutout_center`` attribute will be a nested 3D array.
        """
        return np.array([star.cutout_center for star in self.all_stars])

    @property
    def center_flat(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of all the stars'
        centers (including linked stars) with respect to the original
        (large) image (not the cutout image) as a 2D array
        (``n_all_stars`` x 2).

        Note that when `EPSFStars` contains any `LinkedEPSFStar`, the
        ``center`` attribute will be a nested 3D array.
        """
        return np.array([star.center for star in self.all_stars])

    @lazyproperty
    def all_stars(self):
        """
        A list of all `EPSFStar` objects stored in this object,
        including those that comprise linked stars (i.e.,
        `LinkedEPSFStar`), as a flat list.
        """
        stars = []
        for item in self._data:
            if isinstance(item, LinkedEPSFStar):
                stars.extend(item.all_stars)
            else:
                stars.append(item)
        return stars

    @property
    def all_good_stars(self):
        """
        A list of all `EPSFStar` objects stored in this object that have
        not been excluded from fitting, including those that comprise
        linked stars (i.e., `LinkedEPSFStar`), as a flat list.
        """
        stars = []
        for star in self.all_stars:
            if star._excluded_from_fit:
                continue
            stars.append(star)
        return stars

    @lazyproperty
    def n_stars(self):
        """
        The total number of stars.

        A linked star is counted only once.
        """
        return len(self._data)

    @lazyproperty
    def n_all_stars(self):
        """
        The total number of `EPSFStar` objects, including all the linked
        stars within `LinkedEPSFStar`.

        Each linked star is included in the count.
        """
        return len(self.all_stars)

    @property
    def n_good_stars(self):
        """
        The total number of `EPSFStar` objects, including all the linked
        stars within `LinkedEPSFStar`, that have not been excluded from
        fitting.

        Each non-excluded linked star is included in the count.
        """
        return len(self.all_good_stars)


class LinkedEPSFStar:
    """
    A class to hold a list of `EPSFStar` objects for linked stars.

    Linked stars are `EPSFStar` cutouts from different images that
    represent the same physical star. When building the ePSF, linked
    stars are constrained to have the same sky coordinates.

    Note that unlike `EPSFStars` (which is a collection of potentially
    unrelated stars), `LinkedEPSFStar` represents a single logical star
    observed in multiple images.

    Parameters
    ----------
    stars_list : list of `EPSFStar` objects
        A list of `EPSFStar` objects for the same physical star. Each
        `EPSFStar` object must have a valid ``wcs_large`` attribute to
        convert between pixel and sky coordinates.
    """

    def __init__(self, stars_list):
        for star in stars_list:
            if not isinstance(star, EPSFStar):
                msg = 'stars_list must contain only EPSFStar objects'
                raise TypeError(msg)
            if star.wcs_large is None:
                msg = ('Each EPSFStar object must have a valid wcs_large '
                       'attribute')
                raise ValueError(msg)

        self._data = list(stars_list)

    def __len__(self):
        """
        Return the number of EPSFStar objects in this linked star.
        """
        return len(self._data)

    def __getitem__(self, index):
        """
        Return the EPSFStar at the given index.
        """
        return self._data[index]

    def __iter__(self):
        """
        Iterate over the EPSFStar objects in this linked star.
        """
        yield from self._data

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying star list.

        This provides access to common star attributes like cutout_center,
        center, flux, etc. as arrays when accessed on the LinkedEPSFStar.
        """
        if attr.startswith('_'):
            msg = f"'{type(self).__name__}' object has no attribute '{attr}'"
            raise AttributeError(msg)
        result = [getattr(star, attr) for star in self._data]
        if attr in ('cutout_center', 'center', 'flux', '_excluded_from_fit'):
            result = np.array(result)
        if len(self._data) == 1:
            result = result[0]
        return result

    def __getstate__(self):
        """
        Return state for pickling (avoids __getattr__ recursion).
        """
        return self.__dict__

    def __setstate__(self, d):
        """
        Restore state from pickling.
        """
        self.__dict__ = d

    @property
    def all_stars(self):
        """
        A flat list of all `EPSFStar` objects in this linked star.

        Since LinkedEPSFStar only contains EPSFStar objects (not nested
        LinkedEPSFStar), this is simply the internal list.
        """
        return self._data

    @property
    def cutout_center_flat(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of all the stars'
        centers with respect to the input cutout ``data`` array, as a
        2D array (``n_all_stars`` x 2).
        """
        return np.array([star.cutout_center for star in self._data])

    @property
    def center_flat(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of all the stars'
        centers with respect to the original (large) image (not the
        cutout image) as a 2D array (``n_all_stars`` x 2).
        """
        return np.array([star.center for star in self._data])

    @property
    def n_stars(self):
        """
        The number of `EPSFStar` objects in this linked star.

        For LinkedEPSFStar this is the same as n_all_stars since there
        is no nesting.
        """
        return len(self._data)

    @property
    def n_all_stars(self):
        """
        The total number of `EPSFStar` objects in this linked star.

        For LinkedEPSFStar this is the same as n_stars since there
        is no nesting.
        """
        return len(self._data)

    @property
    def n_good_stars(self):
        """
        The number of `EPSFStar` objects that have not been excluded
        from fitting.
        """
        return len(self.all_good_stars)

    @property
    def all_good_stars(self):
        """
        A list of all `EPSFStar` objects that have not been excluded
        from fitting.
        """
        return [star for star in self._data if not star._excluded_from_fit]

    @property
    def all_excluded(self):
        """
        Whether all `EPSFStar` objects in this linked star have been
        excluded from fitting during the ePSF build process.
        """
        return all(star._excluded_from_fit for star in self._data)

    def constrain_centers(self):
        """
        Constrain the centers of linked `EPSFStar` objects (i.e., the
        same physical star) to have the same sky coordinate.

        Only `EPSFStar` objects that have not been excluded during the
        ePSF build process will be used to constrain the centers.

        The single sky coordinate is calculated as the mean of sky
        coordinates of the linked stars.
        """
        if len(self._data) < 2:  # no linked stars
            return

        if self.all_excluded:
            warnings.warn('Cannot constrain centers of linked stars because '
                          'they have all been excluded during the ePSF '
                          'build process.', AstropyUserWarning)
            return

        # Convert pixel coordinates to sky coordinates
        # Note: each star may have a different WCS, so we cannot
        # vectorize
        good_stars = self.all_good_stars
        sky_coords = np.array([
            star.wcs_large.pixel_to_world_values(*star.center)
            for star in good_stars])

        # Compute mean sky coordinate using spherical averaging
        mean_lon, mean_lat = _compute_mean_sky_coordinate(sky_coords)

        # Convert mean sky coordinate back to pixel coordinates for each
        # star
        for star in good_stars:
            pixel_center = star.wcs_large.world_to_pixel_values(
                mean_lon, mean_lat)
            star.cutout_center = np.asarray(pixel_center) - star.origin


def _compute_mean_sky_coordinate(sky_coords):
    """
    Compute the mean sky coordinate using spherical trigonometry.

    This method properly handles coordinate system singularities by
    converting to Cartesian coordinates for averaging, then converting
    back to spherical coordinates.

    Parameters
    ----------
    sky_coords : array-like, shape (N, 2)
        Array of sky coordinates in degrees, where each row contains
        (longitude, latitude).

    Returns
    -------
    mean_lon, mean_lat : float
        Mean longitude and latitude in degrees.
    """
    lon, lat = sky_coords.T
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    # Convert to Cartesian coordinates for averaging
    x_cart = np.cos(lat_rad) * np.cos(lon_rad)
    y_cart = np.cos(lat_rad) * np.sin(lon_rad)
    z_cart = np.sin(lat_rad)

    # Compute mean Cartesian coordinates
    mean_x = np.mean(x_cart)
    mean_y = np.mean(y_cart)
    mean_z = np.mean(z_cart)

    # Convert mean Cartesian coordinates back to spherical
    hypot = np.hypot(mean_x, mean_y)
    mean_lon = np.rad2deg(np.arctan2(mean_y, mean_x))
    mean_lat = np.rad2deg(np.arctan2(mean_z, hypot))

    return mean_lon, mean_lat


def _normalize_data_input(data):
    """
    Normalize the input data to a list of NDData objects.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        The input data to normalize.

    Returns
    -------
    data : list of `~astropy.nddata.NDData`
        The normalized list of NDData objects.

    Raises
    ------
    TypeError
        If the input data is not an NDData object or list of NDData
        objects.
    """
    if isinstance(data, NDData):
        return [data]
    if isinstance(data, list):
        return data
    msg = 'data must be a single NDData object or list of NDData objects'
    raise TypeError(msg)


def _normalize_catalog_input(catalogs):
    """
    Normalize the input catalogs to a list of Table objects.

    Parameters
    ----------
    catalogs : `~astropy.table.Table` or list of `~astropy.table.Table`
        The input catalogs to normalize.

    Returns
    -------
    catalogs : list of `~astropy.table.Table`
        The normalized list of Table objects.

    Raises
    ------
    TypeError
        If the input catalogs is not a Table object or list of Table
        objects.
    """
    if isinstance(catalogs, Table):
        return [catalogs]
    if isinstance(catalogs, list):
        return catalogs
    msg = 'catalogs must be a single Table object or list of Table objects'
    raise TypeError(msg)


def _validate_nddata_list(data):
    """
    Validate that a list contains only valid NDData objects.

    Parameters
    ----------
    data : list of `~astropy.nddata.NDData`
        The list of NDData objects to validate.

    Raises
    ------
    TypeError
        If any element is not an NDData object.
    ValueError
        If any NDData object has no data array or non-2D data.
    """
    for i, img in enumerate(data):
        if not isinstance(img, NDData):
            msg = (f'All data elements must be NDData objects. '
                   f'Element {i} is {type(img)}')
            raise TypeError(msg)
        if img.data.ndim != 2:
            msg = (f'All NDData objects must contain 2D data. '
                   f'Object at index {i} has {img.data.ndim}D data')
            raise ValueError(msg)


def _validate_catalog_list(catalogs):
    """
    Validate that a list contains only valid Table objects.

    Parameters
    ----------
    catalogs : list of `~astropy.table.Table`
        The list of Table objects to validate.

    Raises
    ------
    TypeError
        If any element is not a Table object.
    """
    for i, cat in enumerate(catalogs):
        if not isinstance(cat, Table):
            msg = (f'All catalog elements must be Table objects. '
                   f'Element {i} is {type(cat)}')
            raise TypeError(msg)
        if len(cat) == 0:
            warnings.warn(f'Catalog at index {i} is empty. No stars will '
                          'be extracted from this catalog.',
                          AstropyUserWarning)


def _validate_coordinate_consistency(data, catalogs):
    """
    Validate coordinate system consistency between data and catalogs.

    This function ensures that the necessary coordinate information
    (either pixel coordinates or WCS for sky coordinates) is available
    to extract stars.

    Parameters
    ----------
    data : list of `~astropy.nddata.NDData`
        The list of NDData objects.

    catalogs : list of `~astropy.table.Table`
        The list of Table catalogs.

    Raises
    ------
    ValueError
        If the coordinate information is inconsistent or missing.
    """
    if len(catalogs) == 1 and len(data) > 1:
        # Single catalog with multiple images requires skycoord and WCS
        if 'skycoord' not in catalogs[0].colnames:
            msg = ('When inputting a single catalog with multiple NDData '
                   'objects, the catalog must have a "skycoord" column.')
            raise ValueError(msg)

        if any(img.wcs is None for img in data):
            msg = ('When inputting a single catalog with multiple NDData '
                   'objects, each NDData object must have a wcs attribute.')
            raise ValueError(msg)
    else:
        # Multiple catalogs (or single catalog with single image)
        for i, cat in enumerate(catalogs):
            has_xy = 'x' in cat.colnames and 'y' in cat.colnames
            has_skycoord = 'skycoord' in cat.colnames

            if not has_xy and not has_skycoord:
                msg = (f'Catalog at index {i} must have either '
                       '"x" and "y" columns or a "skycoord" column.')
                raise ValueError(msg)

            # If only skycoord is available, ensure WCS is present
            if has_skycoord and not has_xy:
                data_idx = i if len(data) == len(catalogs) else 0
                if (data_idx < len(data)
                        and data[data_idx].wcs is None):
                    msg = (f'When catalog at index {i} contains only skycoord '
                           f'positions, the corresponding NDData object must '
                           'have a wcs attribute.')
                    raise ValueError(msg)

                if any(img.wcs is None for img in data):
                    msg = ('When inputting catalog(s) with only skycoord '
                           'positions, each NDData object must have a '
                           'wcs attribute.')
                    raise ValueError(msg)

        if len(data) != len(catalogs):
            msg = ('When inputting multiple catalogs, the number of '
                   'catalogs must match the number of input images.')
            raise ValueError(msg)


def extract_stars(data, catalogs, *, size=(11, 11)):
    """
    Extract cutout images centered on stars defined in the input
    catalog(s).

    Stars where the cutout array bounds partially or completely lie
    outside the input ``data`` image will not be extracted.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or list of `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object or a list of
        `~astropy.nddata.NDData` objects containing the 2D image(s) from
        which to extract the stars. If the input ``catalogs`` contain
        only the sky coordinates (i.e., not the pixel coordinates) of
        the stars then each of the `~astropy.nddata.NDData` objects must
        have a valid ``wcs`` attribute.

    catalogs : `~astropy.table.Table`, list of `~astropy.table.Table`
        A catalog or list of catalogs of sources to be extracted from
        the input ``data``. To link stars in multiple images as a single
        source, you must use a single source catalog where the positions
        defined in sky coordinates.

        If a list of catalogs is input (or a single catalog with a
        single `~astropy.nddata.NDData` object), they are assumed to
        correspond to the list of `~astropy.nddata.NDData` objects
        input in ``data`` (i.e., a separate source catalog for each
        2D image). For this case, the center of each source can be
        defined either in pixel coordinates (in ``x`` and ``y`` columns)
        or sky coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object). If both are specified,
        then the pixel coordinates will be used.

        If a single source catalog is input with multiple
        `~astropy.nddata.NDData` objects, then these sources will be
        extracted from every 2D image in the input ``data``. In this
        case, the sky coordinates for each source must be specified as
        a `~astropy.coordinates.SkyCoord` object contained in a column
        called ``skycoord``. Each `~astropy.nddata.NDData` object in the
        input ``data`` must also have a valid ``wcs`` attribute. Pixel
        coordinates (in ``x`` and ``y`` columns) will be ignored.

        Optionally, each catalog may also contain an ``id`` column
        representing the ID/name of stars. If this column is not
        present then the extracted stars will be given an ``id`` number
        corresponding the table row number (starting at 1). Any other
        columns present in the input ``catalogs`` will be ignored.

    size : int or array_like (int), optional
        The extraction box size along each axis. If ``size`` is a scalar
        then a square box of size ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` order. ``size`` must
        have odd values and be greater than or equal to 3 for both axes.

    Returns
    -------
    stars : `EPSFStars` instance
        A `EPSFStars` instance containing the extracted stars.
    """
    data = _normalize_data_input(data)
    catalogs = _normalize_catalog_input(catalogs)
    _validate_nddata_list(data)
    _validate_catalog_list(catalogs)
    _validate_coordinate_consistency(data, catalogs)
    size = as_pair('size', size, lower_bound=(3, 0), check_odd=True)

    if len(catalogs) == 1:  # may include linked stars
        stars_out, overlap_fail_count = _extract_linked_stars(
            data, catalogs[0], size)
    else:  # no linked stars
        stars_out, overlap_fail_count = _extract_unlinked_stars(
            data, catalogs, size)

    if overlap_fail_count > 0:
        warnings.warn(f'{overlap_fail_count} star(s) were not extracted '
                      'because their cutout region extended beyond the '
                      'input image.', AstropyUserWarning)

    return EPSFStars(stars_out)


def _extract_linked_stars(data, catalog, size):
    """
    Extract stars that may be linked across multiple images.

    Parameters
    ----------
    data : list of `~astropy.nddata.NDData`
        A list of `~astropy.nddata.NDData` objects containing
        the 2D images from which to extract the stars. Each
        `~astropy.nddata.NDData` object must have a valid ``wcs``
        attribute.

    catalog : `~astropy.table.Table`
        A single catalog of sources to be extracted from the input
        ``data``. The center of each source must be defined in
        sky coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object).

    size : int or array_like (int)
        The extraction box size along each axis. If ``size`` is a scalar
        then a square box of size ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` order.

    Returns
    -------
    stars : list of `EPSFStar` or `LinkedEPSFStar` objects
        A list of `EPSFStar` and/or `LinkedEPSFStar` instances
        containing the extracted stars. Stars that are linked across
        multiple images will be represented as a single `LinkedEPSFStar`
        instance containing the corresponding `EPSFStar` instances from
        each image. Failed extractions are represented as `None`.

    overlap_fail_count : int
        The number of stars that failed extraction because their cutout
        region extended beyond the input image.
    """
    # Use pixel coords only for single image
    use_xy = len(data) == 1

    # Extract stars from each image
    results = [_extract_stars(img, catalog, size=size, use_xy=use_xy)
               for img in data]
    stars = [r[0] for r in results]
    overlap_fail_count = sum(r[1] for r in results)

    # Transpose to associate linked stars across images
    stars = list(map(list, zip(*stars, strict=True)))

    # Process each potential linked star group
    stars_out = []
    for star_group in stars:
        good_stars = [star for star in star_group if star is not None]

        if not good_stars:
            continue  # No valid stars in any image

        if len(good_stars) == 1:
            # Single star, not linked
            stars_out.append(good_stars[0])
        else:
            # Multiple stars - create linked star
            stars_out.append(LinkedEPSFStar(good_stars))

    return stars_out, overlap_fail_count


def _extract_unlinked_stars(data, catalogs, size):
    """
    Extract stars from individual catalogs (no linking).

    Parameters
    ----------
    data : list of `~astropy.nddata.NDData`
        A list of `~astropy.nddata.NDData` objects containing
        the 2D images from which to extract the stars.

    catalogs : list of `~astropy.table.Table`
        A list of catalogs of sources to be extracted from the
        input ``data``. Each catalog corresponds to the list of
        `~astropy.nddata.NDData` objects input in ``data`` (i.e., a
        separate source catalog for each 2D image). The center of each
        source can be defined either in pixel coordinates (in ``x`` and
        ``y`` columns) or sky coordinates (in a ``skycoord`` column
        containing a `~astropy.coordinates.SkyCoord`.

    size : int or array_like (int)
        The extraction box size along each axis. If ``size`` is a scalar
        then a square box of size ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` order.

    Returns
    -------
    stars : list of `EPSFStar` objects
        A list of `EPSFStar` instances containing the extracted stars.
        Failed extractions are represented as `None`.

    overlap_fail_count : int
        The number of stars that failed extraction because their cutout
        region extended beyond the input image.
    """
    stars_out = []
    total_overlap_fail_count = 0
    for img, cat in zip(data, catalogs, strict=True):
        extracted, overlap_fail_count = _extract_stars(
            img, cat, size=size, use_xy=True)
        stars_out.extend(extracted)
        total_overlap_fail_count += overlap_fail_count

    # Filter out None values
    return ([star for star in stars_out if star is not None],
            total_overlap_fail_count)


def _extract_stars(data, catalog, *, size=(11, 11), use_xy=True):
    """
    Extract cutout images from a single image centered on stars defined
    in the single input catalog.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the 2D image from
        which to extract the stars. If the input ``catalog`` contains
        only the sky coordinates (i.e., not the pixel coordinates) of
        the stars then the `~astropy.nddata.NDData` object must have a
        valid ``wcs`` attribute.

    catalog : `~astropy.table.Table`
        A single catalog of sources to be extracted from the
        input ``data``. The center of each source can be defined
        either in pixel coordinates (in ``x`` and ``y`` columns)
        or sky coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object). If both are specified,
        then the value of the ``use_xy`` keyword determines which
        coordinates will be used.

    size : int or array_like (int), optional
        The extraction box size along each axis. If ``size`` is a scalar
        then a square box of size ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` order. ``size`` must
        have odd values and be greater than or equal to 3 for both axes.

    use_xy : bool, optional
        Whether to use the ``x`` and ``y`` pixel positions when both
        pixel and sky coordinates are present in the input catalog
        table. If `False` then sky coordinates are used instead of pixel
        coordinates (e.g., for linked stars). The default is `True`.

    Returns
    -------
    stars : list of `EPSFStar` objects
        A list of `EPSFStar` instances containing the extracted stars.
        Failed extractions are represented as `None`.

    overlap_fail_count : int
        The number of stars that failed extraction because their cutout
        region extended beyond the input image.
    """
    colnames = catalog.colnames
    if ('x' not in colnames or 'y' not in colnames) or not use_xy:
        xcenters, ycenters = data.wcs.world_to_pixel(catalog['skycoord'])
    else:
        xcenters = np.asarray(catalog['x'])
        ycenters = np.asarray(catalog['y'])

    if 'id' in colnames:
        ids = catalog['id']
    else:
        ids = np.arange(len(catalog), dtype=int) + 1

    fluxes = catalog['flux'] if 'flux' in colnames else None

    # Prepare uncertainty handling - defer weight array creation
    # until we know which cutouts we need
    uncertainty_info = _prepare_uncertainty_info(data)
    data_mask = data.mask  # Cache mask reference

    stars = []
    nonfinite_weights_count = 0
    overlap_fail_count = 0
    flux_failures = []  # Collect flux estimation failures
    all_zero_stars = []  # Collect stars with all-zero data
    for i, (xcenter, ycenter) in enumerate(zip(xcenters, ycenters,
                                               strict=True)):
        try:
            large_slc, _ = overlap_slices(data.data.shape, size,
                                          (ycenter, xcenter), mode='strict')
        except (PartialOverlapError, NoOverlapError):
            stars.append(None)
            overlap_fail_count += 1
            continue

        # Extract data cutout
        data_cutout = data.data[large_slc]

        # Create weights cutout only for this specific region
        weights_cutout, has_nonfinite = _create_weights_cutout(
            uncertainty_info, data_mask, large_slc)
        if has_nonfinite:
            nonfinite_weights_count += 1

        origin = (large_slc[1].start, large_slc[0].start)
        cutout_center = (xcenter - origin[0], ycenter - origin[1])
        flux = fluxes[i] if fluxes is not None else None

        try:
            # Suppress all-zero warning in EPSFStar (we emit our own below)
            with warnings.catch_warnings():
                msg = 'All unmasked data values in star cutout are zero'
                warnings.filterwarnings('ignore', message=msg,
                                        category=AstropyUserWarning)
                star = EPSFStar(data_cutout, weights=weights_cutout,
                                cutout_center=cutout_center, origin=origin,
                                wcs_large=data.wcs, id_label=ids[i], flux=flux)
            stars.append(star)

            # Track stars with all-zero data
            if hasattr(star, '_has_all_zero_data') and star._has_all_zero_data:
                all_zero_stars.append((xcenter, ycenter))
        except ValueError as exc:
            # Collect flux estimation failures; emit warnings later
            flux_failures.append((xcenter, ycenter, exc))
            stars.append(None)

    # Emit consolidated warning for non-finite weights
    if nonfinite_weights_count > 0:
        warnings.warn(f'{nonfinite_weights_count} star cutout(s) had '
                      'non-finite weight values which were set to zero. '
                      'Please check the input uncertainty values in the '
                      'NDData object.', AstropyUserWarning)

    # Emit individual flux estimation failure warnings. These may be a
    # consequence of having all non-finite weights (data then becomes
    # completely masked), so we emit them after the non-finite weights
    # warning.
    for xcenter, ycenter, exc in flux_failures:
        warnings.warn(f'Failed to create EPSFStar for object at '
                      f'({xcenter:.2f}, {ycenter:.2f}): {exc}',
                      AstropyUserWarning)

    # Emit warnings for stars with all-zero data
    for xcenter, ycenter in all_zero_stars:
        warnings.warn(f'Star at ({xcenter:.1f}, {ycenter:.1f}) has all '
                      'unmasked data values equal to zero',
                      AstropyUserWarning)

    return stars, overlap_fail_count


def _prepare_uncertainty_info(data):
    """
    Prepare uncertainty information for efficient weight computation.

    This function analyzes the input NDData's uncertainty and returns
    a dictionary with information needed to compute weights for cutout
    regions without creating the full weight array.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`
        The NDData object containing the data and possibly uncertainty.

    Returns
    -------
    info : dict
        A dictionary with keys:
        - 'type' : str
            One of 'none', 'weights', or 'uncertainty'.
        - 'array' : `~numpy.ndarray` (only if type='weights')
            The weight array from the input data.
        - 'uncertainty' : `~astropy.nddata.NDUncertainty` (only if
            type='uncertainty')
            The uncertainty object for on-the-fly conversion to weights.
    """
    if data.uncertainty is None:
        return {'type': 'none'}

    if data.uncertainty.uncertainty_type == 'weights':
        return {
            'type': 'weights',
            'array': data.uncertainty.array,
        }

    # For other uncertainties, prepare the conversion
    return {
        'type': 'uncertainty',
        'uncertainty': data.uncertainty,
    }


def _create_weights_cutout(uncertainty_info, data_mask, slices):
    """
    Create a weights cutout for a specific region.

    This avoids creating the full weights array when only a small cutout
    is needed, improving memory efficiency.

    Parameters
    ----------
    uncertainty_info : dict
        Dictionary containing uncertainty information.

    data_mask : `~numpy.ndarray` or None
        Mask array for the data.

    slices : tuple of slice
        Slices defining the cutout region.

    Returns
    -------
    weights_cutout : `~numpy.ndarray`
        The weights array for the cutout region.

    has_nonfinite : bool
        True if non-finite weights were found and set to zero.
    """
    cutout_shape = (slices[0].stop - slices[0].start,
                    slices[1].stop - slices[1].start)

    if uncertainty_info['type'] == 'none':
        weights_cutout = np.ones(cutout_shape, dtype=float)
    elif uncertainty_info['type'] == 'weights':
        weights_cutout = np.asarray(
            uncertainty_info['array'][slices], dtype=float)
    else:
        # Convert uncertainty to weights for this cutout only
        uncertainty_cutout = uncertainty_info['uncertainty'].array[slices]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Convert to standard deviation representation if needed
            if hasattr(uncertainty_info['uncertainty'], 'represent_as'):
                uncertainty_cutout = (
                    uncertainty_info['uncertainty']
                    .represent_as(StdDevUncertainty).array[slices])
            # First compute weights, then check for non-finite values
            weights_cutout = 1.0 / uncertainty_cutout

    # Check for non-finite weights and track if found
    has_nonfinite = not np.all(np.isfinite(weights_cutout))
    if has_nonfinite:
        # Set non-finite weights to 0
        weights_cutout = np.where(np.isfinite(weights_cutout),
                                  weights_cutout, 0.0)

    # Apply mask if present
    if data_mask is not None:
        mask_cutout = data_mask[slices]
        weights_cutout[mask_cutout] = 0.0

    return weights_cutout, has_nonfinite
