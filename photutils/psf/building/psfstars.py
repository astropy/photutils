from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
import numpy as np
from astropy.table import Table
from astropy.nddata import NDData
from astropy.nddata.nddata import UnknownUncertainty
from astropy.nddata.utils import (overlap_slices, PartialOverlapError,
                                  NoOverlapError)
from astropy import wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.utils import lazyproperty

from ...aperture import BoundingBox
from .centroid import find_peak
from .utils import interpolate_missing_data


__all__ = ['PSFStar', 'LinkedPSFStar', 'PSFStars', 'extract_stars']


class PSFStar(object):
    """
    A class to hold a 2D cutout image and associated metadata of a star.

    Parameters
    ----------
    data : `~numpy.ndarray`
        A 2D cutout image of a single star/source.

    weights : `~numpy.ndarray` or `None`, optional
        A 2D array of the weights associated with the input ``data``.

    center : tuple of two floats or `None`, optional
        The ``(x, y)`` position of the star's center with respect to the
        input ``data`` array.  If `None`, then the center of of the
        input ``data`` array will be used.

        TODO: One can use the :meth:`~Star.recenter` method to further
        refine the center position.

    origin : tuple of two int, optional
        The ``(x, y)`` index of the origin (bottom-left corner) pixel of
        the input cutout array with respect to the original array from
        which the cutout was extracted.  This can be used to convert
        positions within the cutout image to positions in the original
        image.  The ``origin`` and ``wcs`` must both be input for linked
        stars (i.e. the same star extracted from different images).

    wcs_original : `~astropy.wcs.WCS` or None, optional
        A WCS object associated with the *original* image from which the
        cutout array was extracted.  It should *not* be a WCS object
        associated with the input cutout ``data`` array.  The ``origin``
        and ``wcs`` must both be input for linked stars (i.e. the same
        star extracted from different images).

    id : int, str, or `None`, optional
        An identification number or label for the star.

    todo_flux: float, None
        Fitted flux or initial estimate of the flux.

    todo_pixel_scale : float, str {'wcs'}, optional
        Pixel scale. When pixel_scale is 'wcs', pixel scale will be inferred
        from the ``wcs`` argument (which *must* be provided in this case).
    """

    def __init__(self, data, weights=None, center=None, origin=(0, 0),
                 wcs_original=None, id=None, flux=None, pixel_scale=1):

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

        if center is None:
            center = np.array((data.shape[1] - 1) / 2.,
                              (data.shape[0] - 1) / 2.)
        self.center = center

        self.origin = origin
        self.wcs_original = wcs_original
        self.id = id

        if flux is None:
            # compute flux so that sum(data)/flux = 1:
            if np.any(self.mask):
                # fill in missing data to better estimate the total flux
                data_interp = interpolate_missing_data(self.data,
                                                       method='cubic',
                                                       mask=self.mask)
                data_interp = interpolate_missing_data(data_interp,
                                                       method='nearest',
                                                       mask=self.mask)
                flux = np.sum(data_interp, dtype=np.float64)

            else:
                flux = np.sum(self.data, dtype=np.float64)
        self.flux = flux

        self.pixel_scale = pixel_scale

        # TODO
        self._fit_failed = False

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

    @lazyproperty
    def _xy_idx(self):
        yidx, xidx = np.indices(self._data.shape)
        return xidx[~self.mask].ravel(), yidx[~self.mask].ravel()

    @lazyproperty
    def _xidx(self):
        return self._xy_idx[0]

    @lazyproperty
    def _yidx(self):
        return self._xy_idx[1]

    @property
    def _xidx_centered(self):
        return self._xy_idx[0] - self.center[0]

    @property
    def _yidx_centered(self):
        return self._xy_idx[1] - self.center[1]

    @lazyproperty
    def _data_values(self):
        return self.data[~self.mask].ravel()

    @lazyproperty
    def _data_values_normalized(self):
        return self._data_values / self.flux

    @lazyproperty
    def _weight_values(self):
        return self.weights[~self.mask].ravel()

    @lazyproperty
    def bbox(self):
        """
        The minimal `~photutils.aperture.BoundingBox` for cutout region
        with respect to the original image.
        """

        return BoundingBox(self.origin[0], self.origin[0] + self.shape[0],
                           self.origin[1], self.origin[1] + self.shape[1])




    #@property
    #def flux(self):
    #    """
    #    Set/get fitted flux. Setting flux to `None` will set the flux
    #    to the sum of the values of all data pixels.

    #    .. note::
    #        If ``data`` contains invalid values (i.e., not finite or for which
    #        ``weights`` == 0), those pixels are first interpolated over using
    #        a cubic spline if possible, and when they cannot be interpolated
    #        over using a spline, they are interpolated using nearest-neighbor
    #        interpolation.
#
#        """
#        return self._flux

#    #@flux.setter
#    def __zflux(self, flux):
#        if flux is None:
#            # compute flux so that sum(data)/flux = 1:
#            if self._has_bad_data:
#                # fill in missing data so as to better estimate "total"
#                # (within image data) flux of the star:
#                idata = interpolate_missing_data(self._data, method='cubic',
#                                                 mask=self._mask)
#                idata = interpolate_missing_data(idata, method='nearest',
#                                                 mask=self._mask)
#                self._flux = np.abs(np.sum(idata, dtype=np.float64))
#
#            else:
#                self._flux = np.abs(np.sum(self._data, dtype=np.float64))
#
#            if not np.isfinite(self._flux):
#                self._flux = 1.0
#
#        else:
#            if not np.isfinite(flux):
#                raise ValueError("'flux' must be a finite number.")
#            self._flux = float(flux)


    @property
    def center_original(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the center of the star
        relative to the center of the coordinate system (not relative to BLC
        as returned by the ``center``).

        When setting the center of the image data, a tuple of two `int` or
        `float` may be used. If ``center`` is set to `None`, the center will be
        derived by looking for a peak near the position of the maximum
        value in the data.

        When setting absolute coordinates, the (relative) center of the star is
        re-computed using the current value of the ``blc``.

        """

        return (self.center[0] + self.origin[0],
                self.center[1] + self.origin[1])

    def zz_refine_center(self, **kwargs):
        """
        Improve star center by finding the maximum of a quadratic
        polynomial fitted to data in a square window of width ``peak_fit_box``
        near currently defined star's center.

        Parameters
        ----------
    recenter : bool, optional
        Indicates that a new source position should be estimated by fitting a
        quadratic polynomial to pixels around the star's center
        (either provided by ``center`` or by performing a brute-search of
        the peak pixel value within a specified search box - see
        ``peak_search_box`` parameter for more details). This may be useful if
        the position of the center of the star is not very accurate.

        .. note::
            Keep in mind that the results of finding star's peak position
            may be sub-optimal on undersampled images. However, this
            method of peak finding (fitting a quadratic 2D polynomial)
            is used only at this stage of determining the PSF (i.e., at the
            stage of extracting star cutouts) and
            the iterative process of refining PSF uses PSF fitting to stars at
            all subsequent stages.

    peak_fit_box : int, tuple of int, optional
        Size (in pixels) of the box around the center of the star (or around
        stars' peak if peak was searched for - see ``peak_search_box`` for
        more details) to be used for quadratic fitting from which peak location
        is computed. If a single integer number is provided, then it is assumed
        that fitting box is a square with sides of length given by
        ``peak_fit_box``. If a tuple of two values is provided, then first
        value indicates the width of the box and the second value indicates
        the height of the box.

    peak_search_box :  str {'all', 'off', 'fitbox'}, int, tuple of int, None,\
optional
        Size (in pixels) of the box around the center of the input star
        to be used for brute-force search of the maximum value pixel. This
        search is performed before quadratic fitting in order to improve
        the original estimate of the peak location. If a single integer
        number is provided, then it is assumed that search box is a square
        with sides of length given by ``peak_fit_box``. If a tuple of two
        values is provided, then first value indicates the width of the box
        and the second value indicates the height of the box. ``'off'`` or
        `None` turns off brute-force search of the maximum. When
        ``peak_search_box`` is ``'all'`` then the entire cutout of the
        star is searched for maximum and when it is set to ``'fitbox'`` then
        the brute-force search is performed in the same box as
        ``peak_fit_box``.





        **kwargs : dict-like, optional
            Additional optional keyword arguments. When present, these
            arguments override values set in a `Star` object when it was
            itnitialized.

            Possible values are:

            - **peak_fit_box** : int, tuple of int, optional
              Size (in pixels) of the box around the center of the star
              used for quadratic fitting based on which peak location
              is computed. See `Star` for more details.

            - **peak_search_box** : str {'all', 'off', 'fitbox'}, int, \
tuple of int, None, optional
              Size (in pixels) of the box around the center of the input star
              to be used for brute-force search of the maximum value pixel.
              See `Star` for more details.
        """
        fbox = kwargs.pop('peak_fit_box', self.peak_fit_box)
        sbox = kwargs.pop('peak_search_box', self.peak_search_box)

        if len(kwargs) > 0:
            print("Unrecognized keyword arguments to 'refine_center' "
                  "will be ignored.")

        self._cx, self._cy = find_peak(
            self._data, self._cx, self._cy,
            peak_fit_box=fbox, peak_search_box=sbox, mask=self.mask
        )

    @property
    def pixel_scale(self):
        """
        Set/Get pixel scale (in arbitrary units). Pixel scale of the
        star can be used in conjunction with pixel scale of a PSF to
        determine PSF's oversampling factor. Either a single floating
        point value or a 1D iterable of length at least 2 (x-scale,
        y-scale) can be provided.  When getting pixel scale, a tuple of
        two values is returned with a pixel scale for each axis.
        """

        return self._pixscale

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale):
        if pixel_scale == 'wcs':
            if wcs:
                pixel_scale = wcs.utils.proj_plane_pixel_scales(wcs)

            else:
                raise ValueError("'wcs' attribute not set.")

        if hasattr(pixel_scale, '__iter__'):
            if len(pixel_scale) != 2:
                raise TypeError("Parameter 'pixel_scale' must be either a "
                                "scalar or an iterable with two elements.")
            self._pixscale = (float(pixel_scale[0]), float(pixel_scale[1]))

        else:
            self._pixscale = (float(pixel_scale), float(pixel_scale))

    # TODO
    #def compute_residuals(self, psf_stars):
    #    return 2D array of (cutout - best-fit PSF)


class PSFStars(object):
    """
    Class to hold a list of `PSFStar` objects.
    """

    def __init__(self, stars_list):
        if isinstance(stars_list, PSFStar):
            self._data = [stars_list]
        elif isinstance(stars_list, list):
            self._data = stars_list
        else:
            raise ValueError('stars_list must be a list of PSFStar objects '
                             'or a single PSFStar object.')

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
        if attr is 'center':
            return np.array([getattr(p, attr) for p in self._data])
        else:
            return [getattr(p, attr) for p in self._data]

    @lazyproperty
    def all_psfstars(self):
        """
        A list of all `PSFStar` objects, including linked stars, as a
        flat list.
        """

        # assumes only a single level of lists
        psf_stars = []
        for item in self._data:
            if isinstance(item, LinkedPSFStar):
                psf_stars.extend(item.all_psfstars)
            elif isinstance(item, list):
                psf_stars.extend(item)
            else:
                psf_stars.append(item)

        return psf_stars

    @lazyproperty
    def n_stars(self):
        """The number of stars."""

        return len(self._data)

    @lazyproperty
    def n_psfstars(self):
        """The number of `PSFStar` objects, including linked stars."""

        return len(self.all_psfstars)

    def get_centers(self):
        centers = []
        #for star in self._data:
        #    if star.
        #    zzzzzz
        pass


class LinkedPSFStar(PSFStars):
    def __init__(self, stars_list):
        super(LinkedPSFStar, self).__init__(stars_list)

    def todo_constrain_linked_centers(self, ignore_badfit_stars=True):
        """
        Constrains the coordinates of star centers (in image
        coordinates).

        This is achieved by constraining star centers of all linked
        stars to correspond to a single sky coordinate obtained by
        computing weighted mean of world coorinates (before
        constraining) of star centers of linked stars.

        Parameters
        ----------

        ignore_badfit_stars : bool, optional
            Do not use stars that have fit error status >0 or that have
            ``ignore`` attribute set to ``True`` in computing mean
            world coordinate.

        """
        # first, check that this star is linked to other stars:
        if self.next is None and self.prev is None:
            return  # nothing to do

        # second, select only those linked stars that have a valid WCS:
        stars = [s for s in self.get_linked_list() if s.wcs is not None]
        if len(stars) < 2:
            return  # nothing to do

        # find centers of the stars in world coordinates:

        w = np.asarray(
            [s.wcs.all_pix2world(s.x_abs_center, s.y_abs_center, 0) +
             [s.star_weight]
             for s in stars if (ignore_badfit_stars and
                                ((s.fit_error_status is not None and
                                  s.fit_error_status > 0) or s.ignore))
             ]
        )

        lon = w[:, 0]
        lat = w[:, 1]

        # compute mean cartesian coordinates:
        wt = w[:, 2] / np.sum(w[:, 2], dtype=np.float64)
        xm = (wt * np.cos(lat) * np.cos(lon)).sum(dtype=np.float)
        ym = (wt * np.cos(lat) * np.sin(lon)).sum(dtype=np.float)
        zm = (wt * np.sin(lat)).sum(dtype=np.float)

        # convert cartesian coordinates back to spherical:
        hyp = np.hypot(xm, ym)
        lon = np.arctan2(ym, xm)
        lat = np.arctan2(zm, hyp)

        # compute new centers:
        for s in stars:
            s.abs_center = list(map(float, s.wcs.all_world2pix(lon, lat, 0)))


def extract_stars(data, catalogs, size=(11, 11)):
    """
    Extract cutout images centered on stars defined in the input
    catalog(s).

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
    size : tuple of two int, optional
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
        center_cutout = (xcenter - origin[0], ycenter - origin[1])
        star = PSFStar(data_cutout, weights_cutout, center=center_cutout,
                       origin=origin, wcs_original=data.wcs, id=obj_id)

        stars.append(star)

    return stars
