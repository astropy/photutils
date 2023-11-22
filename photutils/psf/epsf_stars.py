# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to extract cutouts of stars and data
structures to hold the cutouts for fitting and building ePSFs.
"""

import warnings

import numpy as np
from astropy.nddata import NDData
from astropy.nddata.utils import (NoOverlapError, PartialOverlapError,
                                  overlap_slices)
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
        input cutout ``data`` array.  If `None`, then the center of of
        the input cutout ``data`` array will be used.

    origin : tuple of two int, optional
        The ``(x, y)`` index of the origin (bottom-left corner) pixel of
        the input cutout array with respect to the original array from
        which the cutout was extracted.  This can be used to convert
        positions within the cutout image to positions in the original
        image.  ``origin`` and ``wcs_large`` must both be input for a
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

    def __init__(self, data, *, weights=None, cutout_center=None,
                 origin=(0, 0), wcs_large=None, id_label=None):

        self._data = np.asanyarray(data)
        self.shape = self._data.shape

        if weights is not None:
            if weights.shape != data.shape:
                raise ValueError('weights must have the same shape as the '
                                 'input data array.')
            self.weights = np.asanyarray(weights, dtype=float).copy()
        else:
            self.weights = np.ones_like(self._data, dtype=float)

        self.mask = (self.weights <= 0.0)

        # mask out invalid image data
        invalid_data = np.logical_not(np.isfinite(self._data))
        if np.any(invalid_data):
            self.weights[invalid_data] = 0.0
            self.mask[invalid_data] = True

        self._cutout_center = cutout_center
        self.origin = np.asarray(origin)
        self.wcs_large = wcs_large
        self.id_label = id_label

        self.flux = self.estimate_flux()

        self._excluded_from_fit = False
        self._fitinfo = None

    def __array__(self):
        """
        Array representation of the mask data array (e.g., for
        matplotlib).
        """
        return self._data

    @property
    def data(self):
        """The 2D cutout image."""
        return self._data

    @property
    def cutout_center(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of the star's
        center with respect to the input cutout ``data`` array.
        """
        return self._cutout_center

    @cutout_center.setter
    def cutout_center(self, value):
        if value is None:
            value = ((self.shape[1] - 1) / 2.0, (self.shape[0] - 1) / 2.0)
        else:
            if len(value) != 2:
                raise ValueError('The "cutout_center" attribute must have '
                                 'two elements in (x, y) form.')

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
        """
        if np.any(self.mask):
            data_interp = _interpolate_missing_data(self.data, method='cubic',
                                                    mask=self.mask)
            data_interp = _interpolate_missing_data(data_interp,
                                                    method='nearest',
                                                    mask=self.mask)
            flux = np.sum(data_interp, dtype=float)

        else:
            flux = np.sum(self.data, dtype=float)

        return flux

    def register_epsf(self, epsf):
        """
        Register and scale (in flux) the input ``epsf`` to the star.

        Parameters
        ----------
        epsf : `EPSFModel`
            The ePSF to register.

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the registered/scaled ePSF.
        """
        yy, xx = np.indices(self.shape, dtype=float)
        xx = xx - self.cutout_center[0]
        yy = yy - self.cutout_center[1]

        return self.flux * epsf.evaluate(xx, yy, flux=1.0, x_0=0.0, y_0=0.0)

    def compute_residual_image(self, epsf):
        """
        Compute the residual image of the star data minus the
        registered/scaled ePSF.

        Parameters
        ----------
        epsf : `EPSFModel`
            The ePSF to subtract.

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the residual image.
        """
        return self.data - self.register_epsf(epsf)

    @lazyproperty
    def _xy_idx(self):
        """
        1D arrays of x and y indices of unmasked pixels in the cutout
        reference frame.
        """
        yidx, xidx = np.indices(self._data.shape)
        return xidx[~self.mask].ravel(), yidx[~self.mask].ravel()

    @lazyproperty
    def _xidx(self):
        """
        1D arrays of x indices of unmasked pixels in the cutout
        reference frame.
        """
        return self._xy_idx[0]

    @lazyproperty
    def _yidx(self):
        """
        1D arrays of y indices of unmasked pixels in the cutout
        reference frame.
        """
        return self._xy_idx[1]

    @property
    def _xidx_centered(self):
        """
        1D array of x indices of unmasked pixels, with respect to the
        star center, in the cutout reference frame.
        """
        return self._xy_idx[0] - self.cutout_center[0]

    @property
    def _yidx_centered(self):
        """
        1D array of y indices of unmasked pixels, with respect to the
        star center, in the cutout reference frame.
        """
        return self._xy_idx[1] - self.cutout_center[1]

    @lazyproperty
    def _data_values(self):
        """1D array of unmasked cutout data values."""
        return self.data[~self.mask].ravel()

    @lazyproperty
    def _data_values_normalized(self):
        """
        1D array of unmasked cutout data values, normalized by the
        star's total flux.
        """
        return self._data_values / self.flux

    @lazyproperty
    def _weight_values(self):
        """
        1D array of unmasked weight values.
        """
        return self.weights[~self.mask].ravel()


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
            raise ValueError('stars_list must be a list of EPSFStar and/or '
                             'LinkedEPSFStar objects.')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __delitem__(self, index):
        del self._data[index]

    def __iter__(self):
        yield from self._data

    # explicit set/getstate to avoid infinite recursion
    # from pickler using __getattr__
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, attr):
        if attr in ['cutout_center', 'center', 'flux',
                    '_excluded_from_fit']:
            result = np.array([getattr(star, attr) for star in self._data])
        else:
            result = [getattr(star, attr) for star in self._data]
        if len(self._data) == 1:
            result = result[0]
        return result

    def _getattr_flat(self, attr):
        values = []
        for item in self._data:
            if isinstance(item, LinkedEPSFStar):
                values.extend(getattr(item, attr))
            else:
                values.append(getattr(item, attr))

        return np.array(values)

    @property
    def cutout_center_flat(self):
        """
        A `~numpy.ndarray` of the ``(x, y)`` position of all the
        stars' centers (including linked stars) with respect to the
        input cutout ``data`` array, as a 2D array (``n_all_stars`` x
        2).

        Note that when `EPSFStars` contains any `LinkedEPSFStar`, the
        ``cutout_center`` attribute will be a nested 3D array.
        """
        return self._getattr_flat('cutout_center')

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
        return self._getattr_flat('center')

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
        stars within `LinkedEPSFStar`.  Each linked star is included in
        the count.
        """
        return len(self.all_stars)

    @property
    def n_good_stars(self):
        """
        The total number of `EPSFStar` objects, including all the linked
        stars within `LinkedEPSFStar`, that have not been excluded from
        fitting.  Each non-excluded linked star is included in the
        count.
        """
        return len(self.all_good_stars)

    @lazyproperty
    def _max_shape(self):
        """
        The maximum x and y shapes of all the `EPSFStar` objects
        (including linked stars).
        """
        return np.max([star.shape for star in self.all_stars],
                      axis=0)


class LinkedEPSFStar(EPSFStars):
    """
    A class to hold a list of `EPSFStar` objects for linked stars.

    Linked stars are `EPSFStar` cutouts from different images that
    represent the same physical star.  When building the ePSF, linked
    stars are constrained to have the same sky coordinates.

    Parameters
    ----------
    stars_list : list of `EPSFStar` objects
        A list of `EPSFStar` objects for the same physical star.  Each
        `EPSFStar` object must have a valid ``wcs_large`` attribute to
        convert between pixel and sky coordinates.
    """

    def __init__(self, stars_list):
        for star in stars_list:
            if not isinstance(star, EPSFStar):
                raise ValueError('stars_list must contain only EPSFStar '
                                 'objects.')
            if star.wcs_large is None:
                raise ValueError('Each EPSFStar object must have a valid '
                                 'wcs_large attribute.')

        super().__init__(stars_list)

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

        idx = np.logical_not(self._excluded_from_fit).nonzero()[0]
        if idx.shape == (0,):  # pylint: disable=no-member
            warnings.warn('Cannot constrain centers of linked stars because '
                          'all the stars have been excluded during the ePSF '
                          'build process.', AstropyUserWarning)
            return

        good_stars = [self._data[i] for i in idx]  # pylint: disable=not-an-iterable

        coords = []
        for star in good_stars:
            wcs = star.wcs_large
            xposition = star.center[0]
            yposition = star.center[1]
            coords.append(wcs.pixel_to_world_values(xposition, yposition))

        # compute mean cartesian coordinates
        lon, lat = np.transpose(coords)
        lon *= np.pi / 180.0
        lat *= np.pi / 180.0
        x_mean = np.mean(np.cos(lat) * np.cos(lon))
        y_mean = np.mean(np.cos(lat) * np.sin(lon))
        z_mean = np.mean(np.sin(lat))

        # convert mean cartesian coordinates back to spherical
        hypot = np.hypot(x_mean, y_mean)
        mean_lon = np.arctan2(y_mean, x_mean)
        mean_lat = np.arctan2(z_mean, hypot)
        mean_lon *= 180.0 / np.pi
        mean_lat *= 180.0 / np.pi

        # convert mean sky coordinates back to center pixel coordinates
        # for each star
        for star in good_stars:
            center = star.wcs_large.world_to_pixel_values(mean_lon, mean_lat)
            star.cutout_center = np.array(center) - star.origin


def extract_stars(data, catalogs, *, size=(11, 11)):
    """
    Extract cutout images centered on stars defined in the input
    catalog(s).

    Stars where the cutout array bounds partially or completely lie
    outside of the input ``data`` image will not be extracted.

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
        the input ``data``.  To link stars in multiple images as a
        single source, you must use a single source catalog where the
        positions defined in sky coordinates.

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
        extracted from every 2D image in the input ``data``.  In this
        case, the sky coordinates for each source must be specified as a
        `~astropy.coordinates.SkyCoord` object contained in a column
        called ``skycoord``.  Each `~astropy.nddata.NDData` object in
        the input ``data`` must also have a valid ``wcs`` attribute.
        Pixel coordinates (in ``x`` and ``y`` columns) will be ignored.

        Optionally, each catalog may also contain an ``id`` column
        representing the ID/name of stars.  If this column is not
        present then the extracted stars will be given an ``id`` number
        corresponding the the table row number (starting at 1).  Any
        other columns present in the input ``catalogs`` will be ignored.

    size : int or array_like (int), optional
        The extraction box size along each axis.  If ``size`` is a
        scalar then a square box of size ``size`` will be used.  If
        ``size`` has two elements, they must be in ``(ny, nx)`` order.
        ``size`` must have odd values and be greater than or equal to 3
        for both axes.

    Returns
    -------
    stars : `EPSFStars` instance
        A `EPSFStars` instance containing the extracted stars.
    """
    if isinstance(data, NDData):
        data = [data]

    if isinstance(catalogs, Table):
        catalogs = [catalogs]

    for img in data:
        if not isinstance(img, NDData):
            raise ValueError('data must be a single or list of NDData '
                             'objects.')

    for cat in catalogs:
        if not isinstance(cat, Table):
            raise ValueError('catalogs must be a single or list of Table '
                             'objects.')

    if len(catalogs) == 1 and len(data) > 1:
        if 'skycoord' not in catalogs[0].colnames:
            raise ValueError('When inputting a single catalog with multiple '
                             'NDData objects, the catalog must have a '
                             '"skycoord" column.')

        if any(img.wcs is None for img in data):
            raise ValueError('When inputting a single catalog with multiple '
                             'NDData objects, each NDData object must have '
                             'a wcs attribute.')
    else:
        for cat in catalogs:
            if 'x' not in cat.colnames or 'y' not in cat.colnames:
                if 'skycoord' not in cat.colnames:
                    raise ValueError('When inputting multiple catalogs, '
                                     'each one must have a "x" and "y" '
                                     'column or a "skycoord" column.')

                if any(img.wcs is None for img in data):
                    raise ValueError('When inputting catalog(s) with only '
                                     'skycoord positions, each NDData object '
                                     'must have a wcs attribute.')

        if len(data) != len(catalogs):
            raise ValueError('When inputting multiple catalogs, the number '
                             'of catalogs must match the number of input '
                             'images.')

    size = as_pair('size', size, lower_bound=(3, 0), check_odd=True)

    if len(catalogs) == 1:  # may included linked stars
        use_xy = True
        if len(data) > 1:
            use_xy = False  # linked stars require skycoord positions

        stars = []
        # stars is a list of lists, one list of stars in each image
        for img in data:
            stars.append(_extract_stars(img, catalogs[0], size=size,
                                        use_xy=use_xy))

        # transpose the list of lists, to associate linked stars
        stars = list(map(list, zip(*stars)))

        # remove 'None' stars (i.e., no or partial overlap in one or
        # more images) and handle the case of only one "linked" star
        stars_out = []
        n_input = len(catalogs[0]) * len(data)
        n_extracted = 0
        for star in stars:
            good_stars = [i for i in star if i is not None]
            n_extracted += len(good_stars)
            if not good_stars:
                continue  # no overlap in any image

            if len(good_stars) == 1:
                good_stars = good_stars[0]  # only one star, cannot be linked
            else:
                good_stars = LinkedEPSFStar(good_stars)

            stars_out.append(good_stars)
    else:  # no linked stars
        stars_out = []
        for img, cat in zip(data, catalogs):
            stars_out.extend(_extract_stars(img, cat, size=size, use_xy=True))

        n_input = len(stars_out)
        stars_out = [star for star in stars_out if star is not None]
        n_extracted = len(stars_out)

    n_excluded = n_input - n_extracted
    if n_excluded > 0:
        warnings.warn(f'{n_excluded} star(s) were not extracted because '
                      'their cutout region extended beyond the input image.',
                      AstropyUserWarning)

    return EPSFStars(stars_out)


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
        A single catalog of sources to be extracted from the input
        ``data``.  The center of each source can be defined either in
        pixel coordinates (in ``x`` and ``y`` columns) or sky
        coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object).  If both are specified,
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
    """
    size = as_pair('size', size, lower_bound=(3, 0), check_odd=True)

    colnames = catalog.colnames
    if ('x' not in colnames or 'y' not in colnames) or not use_xy:
        xcenters, ycenters = data.wcs.world_to_pixel(catalog['skycoord'])
    else:
        xcenters = catalog['x'].data.astype(float)
        ycenters = catalog['y'].data.astype(float)

    if 'id' in colnames:
        ids = catalog['id']
    else:
        ids = np.arange(len(catalog), dtype=int) + 1

    if data.uncertainty is None:
        weights = np.ones_like(data.data)
    else:
        if data.uncertainty.uncertainty_type == 'weights':
            weights = np.asanyarray(data.uncertainty.array, dtype=float)
        else:
            warnings.warn('The data uncertainty attribute has an unsupported '
                          'type.  Only uncertainty_type="weights" can be '
                          'used to set weights.  Weights will be set to 1.',
                          AstropyUserWarning)
            weights = np.ones_like(data.data)

    if data.mask is not None:
        weights[data.mask] = 0.0

    stars = []
    for xcenter, ycenter, obj_id in zip(xcenters, ycenters, ids):
        try:
            large_slc, _ = overlap_slices(data.data.shape, size,
                                          (ycenter, xcenter), mode='strict')
            data_cutout = data.data[large_slc]
            weights_cutout = weights[large_slc]
        except (PartialOverlapError, NoOverlapError):
            stars.append(None)
            continue

        origin = (large_slc[1].start, large_slc[0].start)
        cutout_center = (xcenter - origin[0], ycenter - origin[1])
        star = EPSFStar(data_cutout, weights=weights_cutout,
                        cutout_center=cutout_center, origin=origin,
                        wcs_large=data.wcs, id_label=obj_id)

        stars.append(star)

    return stars
