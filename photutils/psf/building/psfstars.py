from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import numpy as np
from astropy.nddata import NDData
from astropy.nddata.utils import (overlap_slices, PartialOverlapError,
                                  NoOverlapError)
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.wcs.utils import skycoord_to_pixel

from ...aperture import BoundingBox
from .utils import interpolate_missing_data


__all__ = ['PSFStar', 'PSFStars', 'LinkedPSFStar', 'extract_stars']


class PSFStar(object):
    """
    A class to hold a 2D cutout image and associated metadata of a star.

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

    wcs_large : `~astropy.wcs.WCS` or None, optional
        A WCS object associated with the large image from which the
        cutout array was extracted.  It should not be the WCS object of
        the input cutout ``data`` array.  ``origin`` and ``wcs_large``
        must both be input for a linked star (a single star extracted
        from different images).

    id_label : int, str, or `None`, optional
        An optional identification number or label for the star.

    pixel_scale : float or tuple of two floats, optional
        The pixel scale (in arbitrary units) of the input ``data``.  The
        ``pixel_scale`` can either be a single float or tuple of two
        floats of the form ``(x_pixscale, y_pixscale)``.  If
        ``pixel_scale`` is a scalar then the pixel scale will be the
        same for both the x and y axes.  The star ``pixel_scale`` is
        used in conjunction with the PSF pixel scale or oversampling
        factor when building and fitting the PSF.  The ratio of the
        star-to-PSF pixel scales represents the PSF oversampling factor.
        ``pixel_scale`` allows for building (and fitting) a PSF using
        images of stars with different pixel scales (e.g. velocity
        aberrations).
    """

    def __init__(self, data, weights=None, cutout_center=None, origin=(0, 0),
                 wcs_large=None, id_label=None, pixel_scale=1.):

        self._data = np.asanyarray(data)
        self.shape = self._data.shape

        if weights is not None:
            if weights.shape != data.shape:
                raise ValueError('weights must have the same shape as the '
                                 'input data array.')
            self.weights = np.asanyarray(weights, dtype=np.float).copy()
        else:
            self.weights = np.ones_like(self._data, dtype=np.float)

        self.mask = (self.weights <= 0.)

        # mask out invalid image data
        invalid_data = np.logical_not(np.isfinite(self._data))
        if np.any(invalid_data):
            self.weights[invalid_data] = 0.
            self.mask[invalid_data] = True

        self._cutout_center = cutout_center
        self.origin = np.asarray(origin)
        self.wcs_large = wcs_large
        self.id_label = id_label

        pixel_scale = np.atleast_1d(pixel_scale)
        if len(pixel_scale) == 1:
            pixel_scale = np.repeat(pixel_scale, 2)
        self.pixel_scale = pixel_scale  # ndarray

        self.flux = self.estimate_flux()

        self._excluded_from_fit = False

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
            value = ((self.shape[1] - 1) / 2., (self.shape[0] - 1) / 2.)
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

        return (self.cutout_center + self.origin)

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
            data_interp = interpolate_missing_data(self.data,
                                                   method='cubic',
                                                   mask=self.mask)
            data_interp = interpolate_missing_data(data_interp,
                                                   method='nearest',
                                                   mask=self.mask)
            flux = np.sum(data_interp, dtype=np.float64)

        else:
            flux = np.sum(self.data, dtype=np.float64)

        return flux

    def register_psf(self, psf):
        """
        Register and scale (in flux) the input ``psf`` to the star.

        Parameters
        ----------
        psf : `PSF2DModel`
            The point-spread function (PSF).

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the registered/scaled PSF.
        """

        x_oversamp = self.pixel_scale[0] / psf.pixel_scale[0]
        y_oversamp = self.pixel_scale[1] / psf.pixel_scale[1]

        yy, xx = np.indices(self.shape, dtype=np.float)
        xx = x_oversamp * (xx - self.cutout_center[0])
        yy = y_oversamp * (yy - self.cutout_center[1])

        return (self.flux * x_oversamp * y_oversamp *
                psf.evaluate(xx, yy, flux=1.0, x_0=0.0, y_0=0.0))

    def compute_residual_image(self, psf):
        """
        Compute the residual image of the star data minus the
        registered/scaled PSF.

        Parameters
        ----------
        psf : `PSF2DModel`
            The point-spread function (PSF).

        Returns
        -------
        data : `~numpy.ndarray`
            A 2D array of the residual image.
        """

        return self.data - self.register_psf(psf)

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


class PSFStars(object):
    """
    Class to hold a list of `PSFStar` and/or `LinkedPSFStar` objects.

    Parameters
    ----------
    star_list : list of `PSFStar` or `LinkedPSFStar` objects
        A list of `PSFStar` and/or `LinkedPSFStar` objects.
    """

    def __init__(self, stars_list):
        if (isinstance(stars_list, PSFStar) or
                isinstance(stars_list, LinkedPSFStar)):
            self._data = [stars_list]
        elif isinstance(stars_list, list):
            self._data = stars_list
        else:
            raise ValueError('stars_list must be a list of PSFStar and/or '
                             'LinkedPSFStar objects.')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    # needed for python 2
    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __delitem__(self, index):
        del self._data[index]

    def __iter__(self):
        for i in self._data:
            yield i

    def __getattr__(self, attr):
        if attr in ['cutout_center', 'center', 'pixel_scale', 'flux',
                    '_excluded_from_fit']:
            return np.array([getattr(star, attr) for star in self._data])
        else:
            return [getattr(star, attr) for star in self._data]

    @lazyproperty
    def all_psfstars(self):
        """
        A list of all `PSFStar` objects stored in this object, including
        those that comprise linked stars (i.e. `LinkedPSFStar`), as a
        flat list.
        """

        psf_stars = []
        for item in self._data:
            if isinstance(item, LinkedPSFStar):
                psf_stars.extend(item.all_psfstars)
            else:
                psf_stars.append(item)

        return psf_stars

    @property
    def all_good_psfstars(self):
        """
        A list of all `PSFStar` objects stored in this object that have
        not been excluded from fitting, including those that comprise
        linked stars (i.e. `LinkedPSFStar`), as a flat list.
        """

        psf_stars = []
        for psf_star in self.all_psfstars:
            if psf_star._excluded_from_fit:
                continue
            else:
                psf_stars.append(psf_star)

        return psf_stars

    @lazyproperty
    def n_stars(self):
        """
        The total number of stars.

        A linked star is counted only once.
        """

        return len(self._data)

    @lazyproperty
    def n_psfstars(self):
        """
        The total number of `PSFStar` objects, including all the linked
        stars within `LinkedPSFStar`.  Each linked star is included in
        the count.
        """

        return len(self.all_psfstars)

    @property
    def n_good_psfstars(self):
        """
        The total number of `PSFStar` objects, including all the linked
        stars within `LinkedPSFStar`, that have not been excluded from
        fitting.  Each non-excluded linked star is included in the
        count.
        """

        return np.count_nonzero(~self._excluded_from_fit)

    @lazyproperty
    def _min_pixel_scale(self):
        """
        The minimum x and y pixel scale of all the `PSFStar`s (including
        linked stars).
        """

        return np.min([star.pixel_scale for star in self.all_psfstars],
                      axis=0)

    @lazyproperty
    def _max_shape(self):
        """
        The maximum x and y shapes of all the `PSFStar`s (including
        linked stars).
        """

        return np.max([star.shape for star in self.all_psfstars],
                      axis=0)


class LinkedPSFStar(PSFStars):
    """
    A class to hold a list of `PSFStar` objects for linked stars.

    Linked stars are `PSFStar` cutouts from different images that
    represent the same physical star.  When building the PSF, linked
    stars are constrained to have the same sky coordinates.

    Parameters
    ----------
    star_list : list of `PSFStar` objects
        A list of `PSFStar` objects for the same physical star.  Each
        `PSFStar` object must have a valid ``wcs_large`` attribute to
        convert between pixel and sky coordinates.
    """

    def __init__(self, stars_list):
        for star in stars_list:
            if not isinstance(star, PSFStar):
                return ValueError('stars_list must contain only PSFStar '
                                  'objects.')
            if star.wcs_large is None:
                return ValueError('Each PSFStar object must have a valid '
                                  'wcs_large attribute.')

        super(LinkedPSFStar, self).__init__(stars_list)

    def constrain_centers(self):
        """
        Constrain the centers of linked PSFStar objects (i.e. the same
        physical star) to have the same sky coordinate.

        The single sky coordinate is calculated as the mean of sky
        coordinates of the linked stars.
        """

        if len(self._data) < 2:   # no linked stars
            return

        good_stars = self._data[np.logical_not(self._excluded_from_fit)]

        coords = []
        for star in good_stars:
            coords.append(star.wcs_large.all_pix2world(self.center[0],
                                                       self.center[1], 0))

        # compute mean cartesian coordinates
        lon, lat = np.transpose(coords)
        x_mean = np.mean(np.cos(lat) * np.cos(lon))
        y_mean = np.mean(np.cos(lat) * np.sin(lon))
        z_mean = np.mean(np.sin(lat))

        # convert mean cartesian coordinates back to spherical
        hypot = np.hypot(x_mean, y_mean)
        lon = np.arctan2(y_mean, x_mean)
        lat = np.arctan2(z_mean, hypot)

        # convert mean sky coordinates back to center pixel coordinates
        # for each star
        for star in good_stars:
            center = np.array(star.wcs_large.all_world2pix(lon, lat, 0))
            star.cutout_center = center - star.origin


def extract_stars(data, catalogs, size=(11, 11)):
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
        which to extract the stars.  If the input ``catalogs`` contain
        only the sky coordinates (i.e. not the pixel coordinates) of the
        stars then each of the `~astropy.nddata.NDData` objects must
        have a valid ``wcs`` attribute.

    catalogs : `~astropy.table.Table`, list of `~astropy.table.Table`
        A catalog or list of catalogs of sources to be extracted from
        the input ``data``.  To link stars in multiple images as a
        single source, you must use a single source catalog where the
        positions defined in sky coordinates.

        If a list of catalogs is input (or a single catalog with a
        single `~astropy.nddata.NDData` object), they are assumed to
        correspond to the list of `~astropy.nddata.NDData` objects input
        in ``data`` (i.e. a separate source catalog for each 2D image).
        For this case, the center of each source can be defined either
        in pixel coordinates (in ``x`` and ``y`` columns) or sky
        coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object).  If both are specified,
        then the pixel coordinates will be used.

        If a single source catalog is input with multiple
        `~astropy.nddata.NDData` objects, then these sources will be
        extracted from every 2D image in the input ``data``.  In this
        case, the sky coordinates for each source must be specified as a
        `~astropy.coordinates.SkyCoord` object contained in a column
        called ``skycoord``.  Each `~astropy.nddata.NDData` object in
        the input ``data`` must also have a valid ``wcs`` attribute.

        Optionally, each catalog may also contain an ``id`` column
        representing the ID/name of stars.  If this column is not
        present then the extracted stars will be given an ``id`` number
        corresponding the the table row number (starting at 1).  Any
        other columns present in the input ``catalogs`` will be ignored.

    size : int or array_like (int), optional
        The extraction box size along each axis.  If ``size`` is a
        scalar then a square box of size ``size`` will be used.  If
        ``size`` has two elements, they should be in ``(ny, nx)`` order.
        The size must be greater than or equal to 3 pixel for both axes.

    Returns
    -------
    psfstars : `PSFStars` instance
        A `PSFStars` instance containing the extracted stars.
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

        if any([img.wcs is None for img in data]):
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
                else:
                    if any([img.wcs is None for img in data]):
                        raise ValueError('When inputting catalog(s) with '
                                         'only skycoord positions, each '
                                         'NDData object must have a wcs '
                                         'attribute.')

        if len(data) != len(catalogs):
            raise ValueError('When inputting multiple catalogs, the number '
                             'of catalogs must match the number of input '
                             'images.')

    size = np.atleast_1d(size)
    if len(size) == 1:
        size = np.repeat(size, 2)

    min_size = 3
    if size[0] < min_size or size[1] < min_size:
        raise ValueError('size must be >= {} for x and y'.format(min_size))

    if len(catalogs) == 1:    # may included linked stars
        stars = []
        # stars is a list of lists, one list of stars in each image
        for img in data:
            stars.append(_extract_stars(img, catalogs[0], size=size))

        # transpose the list of lists, to associate linked stars
        stars = list(map(list, zip(*stars)))

        # remove 'None' stars (i.e. no or partial overlap in one or more
        # images) and handle the case of only one "linked" star
        stars_out = []
        for star in stars:
            good_stars = [i for i in star if i is not None]
            if len(good_stars) == 0:
                continue    # no overlap in any image
            elif len(good_stars) == 1:
                good_stars = good_stars[0]  # only one star, cannot be linked
            else:
                good_stars = LinkedPSFStar(good_stars)

            stars_out.append(good_stars)
    else:    # no linked stars
        stars_out = []
        for img, cat in zip(data, catalogs):
            stars_out.append(_extract_stars(img, cat, size=size))

    return PSFStars(stars_out)


def _extract_stars(data, catalog, size=(11, 11)):
    """
    Extract cutout images from a single image centered on stars defined
    in the single input catalog.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the 2D image from
        which to extract the stars.  If the input ``catalog`` contains
        only the sky coordinates (i.e. not the pixel coordinates) of the
        stars then the `~astropy.nddata.NDData` object must have a valid
        ``wcs`` attribute.

    catalogs : `~astropy.table.Table`
        A single catalog of sources to be extracted from the input
        ``data``.  The center of each source can be defined either in
        pixel coordinates (in ``x`` and ``y`` columns) or sky
        coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object).  If both are specified,
        then the pixel coordinates will be used.

    size : int or array_like (int), optional
        The extraction box size along each axis.  If ``size`` is a
        scalar then a square box of size ``size`` will be used.  If
        ``size`` has two elements, they should be in ``(ny, nx)`` order.
        The size must be greater than or equal to 3 pixel for both axes.

    Returns
    -------
    stars : list of `PSFStar` objects
        A list of `PSFStar` instances containing the extracted stars.
    """

    colnames = catalog.colnames
    if 'x' not in colnames or 'y' not in colnames:
        xcenters, ycenters = skycoord_to_pixel(catalog['skycoord'], data.wcs,
                                               origin=0, mode='all')
    else:
        xcenters = catalog['x'].data.astype(np.float)
        ycenters = catalog['y'].data.astype(np.float)

    if 'id' in colnames:
        ids = catalog['id']
    else:
        ids = np.arange(len(catalog), dtype=np.int) + 1

    if data.uncertainty is None:
        weights = np.ones_like(data.data)
    else:
        if data.uncertainty.uncertainty_type == 'weights':
            weights = np.asanyarray(data.uncertainty.array, dtype=np.float)
        else:
            warnings.warn('The data uncertainty attribute has an unsupported '
                          'type.  Only uncertainty_type="weights" can be '
                          'used to set weights.  Weights will be set to 1.')
            weights = np.ones_like(data.data)

    if data.mask is not None:
        weights[data.mask] = 0.

    stars = []
    for xcenter, ycenter, obj_id in zip(xcenters, ycenters, ids):
        try:
            large_slc, small_slc = overlap_slices(data.data.shape, size,
                                                  (ycenter, xcenter),
                                                  mode='strict')
            data_cutout = data.data[large_slc]
            weights_cutout = weights[large_slc]
        except (PartialOverlapError, NoOverlapError):
            stars.append(None)
            continue

        origin = (large_slc[1].start, large_slc[0].start)
        cutout_center = (xcenter - origin[0], ycenter - origin[1])
        star = PSFStar(data_cutout, weights_cutout,
                       cutout_center=cutout_center, origin=origin,
                       wcs_large=data.wcs, id_label=obj_id)

        stars.append(star)

    return stars
