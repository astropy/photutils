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

from .centroid import find_peak
from .utils import interpolate_missing_data


__all__ = ['Stars', 'Star', 'extract_stars']


class Stars(object):
    """
    Class to hold `Star` objects.
    """

    def __init__(self, stars_list):
        if isinstance(stars_list, Star):
            self._data = [Star]
        elif isinstance(stars_list, list):
            self._data = stars_list
        else:
            raise ValueError('invalid input.')

    def __len__(self):
        return len(self._data)

    # needed for python 2
    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __delitem__(self, index):
        del self._data[index]

    def __iter__(self):
        for i in self._data:
            yield i

    def __getattr__(self, attr):
        return [getattr(p, attr) for p in self._data]

    #def flatten(self):
    #    pass

    def constrain_linked_centers(self, ignore_badfit_stars=True):
        """ Constrains the coordinates of star centers (in image coordinates).

        This is achieved by constraining star centers of all linked stars to
        correspond to a single sky coordinate obtained by computing weighted
        mean of world coorinates (before constraining) of star centers of
        linked stars.

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


class Star(object):
    """
    A class for holding information about a star cutout from 2D images such as
    its coordinates, WCS, pixel scale, location of the cutout in the original
    image, fit information (if, e.g., a PSF was fit to the star), etc.

    In addition, it provides a mechanism of linking multiple stars together
    for the purpose of computing their average position on the sky and
    converting these world coordinate to the image coordinates of the linked
    stars.


    Parameters
    ----------

    data : numpy.ndarray
        A cutout (sub-image) from a 2D image containing image of a single star.

    weights : numpy.ndarray, None, optional
        Weights associated with each pixel in the ``data``.

    star_weight : float, optional
        Weight of the star. This can be used, e.g., to compute weighted average
        of the world coordinates of linked stars.

    flux: float, None
        Fitted flux or initial estimate of the flux.

    pixel_scale : float, str {'wcs'}, optional
        Pixel scale. When pixel_scale is 'wcs', pixel scale will be inferred
        from the ``wcs`` argument (which *must* be provided in this case).

    center : tuple of two float, None, optional
        Position of the center of the star in input ``data`` array. When
        ``center`` is `None`, center will be set at at the coordinates of the
        detected peak of intensity.

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

    origin : tuple of two int
        Position (``x``, ``y``) of the bottom-left corner of star's cutout
        in the original image. This is useful for recovering star' position
        in the original image from which star's cutout was extracted.

    wcs : astropy.wcs.WCS, None, optional
        :py:class:`~astropy.wcs.WCS` of the *original* image from which star's
        cutout was extracted.

    meta : dict-like, optional
        Additional meta information about the star.


    Attributes
    ----------
    image_name : str
        Name of the image from which star's cutout was extracted.
        Default: 'Unknown'.

    catalog_name : str
        Name of the catalog that provided the coordinates of the source.
        Default: 'Unknown'.

    id : int, str, None
        Some identification of the source in the catalog. Default: `None`.
    """

    def __init__(self, data, weights=None, star_weight=1.0,
                 flux=None, pixel_scale=1, center=None, recenter=False,
                 peak_fit_box=5, peak_search_box='fitbox',
                 origin=(0, 0), wcs=None, meta={}, id=None):

        self._data = data
        self.weights = weights  # we must set weights ASAP to have a valid mask
        self.peak_fit_box = peak_fit_box
        self.peak_search_box = peak_search_box
        self.pixel_scale = pixel_scale
        self.star_weight = star_weight

        # set/compute star's flux:
        self.flux = flux

        # set input image related parameters:
        self._ny, self._nx = data.shape

        # center of the "star" in 'data' grid:
        self.center = center
        if center is not None and recenter:
            self.refine_center(peak_fit_box=peak_fit_box,
                               peak_search_box=peak_search_box)

        # coordinate of the bottom-left pixel of input 'data' in the original
        # image from which 'data' have been "cut-out"
        self.origin = origin

        self._wcs = wcs
        self._meta = meta

        self._prev = None
        self._next = None

        # fit information:
        self._fit_residual = None
        self._fit_info = None
        self._fit_error_status = None
        self._iter_fit_status = None
        self._iter_fit_eps = None
        self._ignore = False

        # useful info: ids, names, etc.
        self.image_name = 'Unknown'
        self.catalog_name = 'Unknown'
        self.id = id

    @property
    def data(self):
        """
        Get star's image data.

        """
        return self._data

    @property
    def weights(self):
        """
        Get effective weights of image data.

        When setting weights, :py:class:`numpy.ndarray` or `None` may be used.
        Effective weights of star's data are computed by setting elements of
        the input ``weights`` array that correspond to invalid image data
        (such as `~numpy.nan`, `~numpy.inf`, etc.) to 0.

        """
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is not None:
            if weights.shape != self._data.shape:
                raise ValueError("'data' and 'weights' arrays must have "
                                 "identical shapes.")
            self._weights = np.array(weights, dtype=np.float, copy=True)
            self._mask = self.weights > 0.0
            self._has_bad_data = True

        else:
            self._weights = np.ones_like(self._data, dtype=np.float)
            self._mask = np.ones_like(self._data, dtype=np.bool)
            self._has_bad_data = False

        # mask out invalid image data:
        invalid_data = np.logical_not(np.isfinite(self._data))
        if np.any(invalid_data):
            self._mask[invalid_data] = False
            self._weights[invalid_data] = 0.0
            self._has_bad_data = True

        self._x, self._y, self._v, self._w = self._compute_data_vectors()

    @property
    def star_weight(self):
        """ Get/Set star's weight. """
        return self._star_weight

    @star_weight.setter
    def star_weight(self, star_weight):
        star_weight = float(star_weight)
        if star_weight <= 0.0:
            raise ValueError("Star's weight must be a strictly positive "
                             "number.")

    @property
    def mask(self):
        """
        Effective mask indicating which pixels are "valid" (True) and
        which pixels are "defective" (False). Effective mask is computed
        as ``weights>0``.

        """
        return self._mask

    @property
    def flux(self):
        """
        Set/get fitted flux. Setting flux to `None` will set the flux
        to the sum of the values of all data pixels.

        .. note::
            If ``data`` contains invalid values (i.e., not finite or for which
            ``weights`` == 0), those pixels are first interpolated over using
            a cubic spline if possible, and when they cannot be interpolated
            over using a spline, they are interpolated using nearest-neighbor
            interpolation.

        """
        return self._flux

    @flux.setter
    def flux(self, flux):
        if flux is None:
            # compute flux so that sum(data)/flux = 1:
            if self._has_bad_data:
                # fill in missing data so as to better estimate "total"
                # (within image data) flux of the star:
                idata = interpolate_missing_data(self._data, method='cubic',
                                                 mask=self._mask)
                idata = interpolate_missing_data(idata, method='nearest',
                                                 mask=self._mask)
                self._flux = np.abs(np.sum(idata, dtype=np.float64))

            else:
                self._flux = np.abs(np.sum(self._data, dtype=np.float64))

            if not np.isfinite(self._flux):
                self._flux = 1.0

        else:
            if not np.isfinite(flux):
                raise ValueError("'flux' must be a finite number.")
            self._flux = float(flux)

    @property
    def shape(self):
        """ Numpy style tuple of dimensions of the data array (ny, nx). """
        return self._data.shape

    @property
    def nx(self):
        """ Number of columns in the data array. """
        return self._nx

    @property
    def ny(self):
        """ Number of rows in the data array. """
        return self._ny

    @property
    def center(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the center of the star
        in terms of pixels of star's image cutout.

        When setting the center of the image data, a tuple of two `int` or
        `float` may be used. If `center` is set to `None`, the center will be
        derived by looking for a peak near the position of the maximum
        value in the data.

        """
        return (self._cx, self._cy)

    @center.setter
    def center(self, center):
        if center is None:
            self._cx, self._cy = find_peak(
                self._data, xmax=None, ymax=None,
                peak_fit_box=self._peak_fit_box, peak_search_box=None,
                mask=self.mask
            )

        elif hasattr(center, '__iter__') and len(center) == 2:
            self._cx, self._cy = center

        else:
            raise TypeError("Parameter 'center' must be either None or an "
                            "iterable with two elements.")

    @property
    def x_center(self):
        """ X-coordinate of the center. """
        return self._cx

    @x_center.setter
    def x_center(self, x_center):
        self._cx = x_center

    @property
    def y_center(self):
        """ Y-coordinate of the center. """
        return self._cy

    @y_center.setter
    def y_center(self, y_center):
        self._cy = y_center

    @property
    def abs_center(self):
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
        return (self._cx + self._blc[0], self._cy + self._blc[1])

    @abs_center.setter
    def abs_center(self, abs_center):
        if abs_center is None:
            self.center = None
            return
        self._cx = abs_center[0] - self._blc[0]
        self._cy = abs_center[1] - self._blc[1]

    @property
    def x_abs_center(self):
        """
        Get/set absolute X-coordinate of the center (including BLC).
        When setting absolute coordinate, the (relative) center of the star is
        re-computed using the current value of the ``blc``.

        """
        return self._cx + self._blc[0]

    @x_abs_center.setter
    def x_abs_center(self, x_abs_center):
        self._cx = x_abs_center - self._blc[0]

    @property
    def y_abs_center(self):
        """
        Get/set absolute Y-coordinate of the center (including BLC).
        When setting absolute coordinate, the (relative) center of the star is
        re-computed using the current value of the ``blc``.

        """
        return self._cy + self._blc[1]

    @y_abs_center.setter
    def y_abs_center(self, y_abs_center):
        self._cy = y_abs_center - self._blc[1]

    def refine_center(self, **kwargs):
        """
        Improve star center by finding the maximum of a quadratic
        polynomial fitted to data in a square window of width ``peak_fit_box``
        near currently defined star's center.

        Parameters
        ----------
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
    def blc(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the bottom-left corner of the
        star data relative to the origin of the original image from which
        the star was extracted.

        """
        return self._blc

    @blc.setter
    def blc(self, blc):
        if hasattr(blc, '__iter__') and len(blc) == 2:
            self._blc = tuple(np.asarray(blc).tolist())
        else:
            raise TypeError("Parameter 'blc' must be an iterable with two "
                            "elements.")

    @property
    def wcs(self):
        """ Get the wcs object attached to the `Star`. """
        return self._wcs

    @property
    def meta(self):
        """
        Get/Set the meta object attached to the `Star`. Meta object must
        be a `dict`. When setting meta to `None`, an empty `dict` will be
        assigned to the `Star` object.

        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is None:
            self._meta = {}
        elif not isinstance(meta, dict):
            raise TypeError("'meta' must be a dictionary or 'None'.")
        else:
            self._meta = meta

    def _compute_data_vectors(self):
        y, x = np.indices(self._data.shape)
        x = x[self._mask].ravel()
        y = y[self._mask].ravel()
        v = self._data[self._mask].ravel()
        w = self._weights[self._mask].ravel()
        return(x, y, v, w)

    def centered_plist(self, normalized=True):
        """
        Return a list of coordinates, data values, and weights of *valid*
        pixels in input data.

        Parameters
        ----------

        normalized : bool, optional
            Normalize image data values by ``flux``.

        Returns
        -------

        plist : numpy.ndarray
            A `numpy.ndarray` of shape ``Nx4`` where ``N`` is the number of
            valid pixels (``weights`` > 0 and ``numpy.isfinite(data)``)
            in image data. The first two columns contain ``x`` and ``y``
            coordinates of the *valid* pixels relative to the center of the
            star. The third column contains (normalized) pixel values.
            The last column contains weight associated with pixel values.

        """
        plist = np.empty((self._x.shape[0], 4), dtype=np.float)
        plist[:, 0] = self._x - self._cx
        plist[:, 1] = self._y - self._cy
        if normalized:
            plist[:, 2] = self._v / self.flux
        else:
            plist[:, 2] = self._v
        plist[:, 3] = self._w
        return plist

    def absolute_plist(self, normalized=True):
        """
        Return a list of absolute coordinates in the original image from
        which star's cutout was obtained, data values, and weights of *valid*
        pixels in input data.

        .. note::
            Use `abs_center` to retrieve the center of the star in "absolute"
            coordinates.

        Parameters
        ----------

        normalized : bool, optional
            Normalize image data values by ``flux``.

        Returns
        -------

        plist : numpy.ndarray
            A `numpy.ndarray` of shape ``Nx4`` where N is the number of valid
            pixels (``weights`` > 0 and ``numpy.isfinite(data)``) in image
            data. The first two columns contain ``x`` and ``y`` coordinates
            of the *valid* pixels in the original image from which star's
            cutout was obtained. The third column contains (normalized)
            pixel values. The last column contains weights associated with
            pixel values.

        """
        plist = np.empty((self._x.shape[0], 4), dtype=np.float)
        x0, y0 = self.blc
        plist[:, 0] = self._x - self._cx + x0
        plist[:, 1] = self._y - self._cy + y0
        if normalized:
            plist[:, 2] = self._v / self.flux
        else:
            plist[:, 2] = self._v
        plist[:, 3] = self._w
        return plist

    @property
    def fit_residual(self):
        """
        Set/Get fit residual. It must be either a `None` or
        a `~numpy.ndarray` object of the same shape as data. This attribute
        is intended to hold the residual of the fit of a PSF to this
        `Star` object.

        """
        return self._fit_residual

    @fit_residual.setter
    def fit_residual(self, fit_residual):
        if fit_residual is None:
            self._fit_residual = None
        else:
            fit_residual = np.asarray(fit_residual)
            if fit_residual.shape != self.shape:
                raise ValueError("'fit_residual' must be 'None' or a 2D array "
                                 "of the same shape as this Star's data.")
            self._fit_residual = fit_residual

    @property
    def fit_info(self):
        """
        Set/Get the results of the PSF fit. This attribute is intended to
        store any kind of fit information as returned by the fitter.

        """
        return self._fit_info

    @fit_info.setter
    def fit_info(self, fit_info):
        self._fit_info = fit_info

    @property
    def fit_error_status(self):
        """
        Set/Get PSF fit error status. The value of `None` indicates that
        the fit has not been performed and the value of 0 indicates that the
        fit was successful.

        """
        return self._fit_error_status

    @fit_error_status.setter
    def fit_error_status(self, fit_error_status):
        self._fit_error_status = fit_error_status

    @property
    def iter_fit_status(self):
        """
        Set/Get PSF fit error status. The value of `None` indicates that
        the fit has not been performed and the value of 0 indicates that the
        fit was successful.

        """
        return self._iter_fit_status

    @iter_fit_status.setter
    def iter_fit_status(self, iter_fit_status):
        self._iter_fit_status = iter_fit_status

    @property
    def iter_fit_eps(self):
        """
        Set/Get PSF fit error status. The value of `None` indicates that
        the fit has not been performed and the value of 0 indicates that the
        fit was successful.

        """
        return self._iter_fit_eps

    @iter_fit_eps.setter
    def iter_fit_eps(self, iter_fit_eps):
        self._iter_fit_eps = iter_fit_eps

    @property
    def ignore(self):
        """
        Set/Get 'ignore' attribute that indicates whether this star
        should be used when constructing a PSF (`True` - use this star,
        `False` - ignore this star).

        """
        return self._ignore

    @ignore.setter
    def ignore(self, ignore):
        self._ignore = ignore

    @property
    def pixel_scale(self):
        """
        Set/Get pixel scale (in arbitrary units). Pixel scale of the star
        can be used in conjunction with pixel scale of a PSF to determine
        PSF's oversampling factor. Either a single floating point value or
        a 1D iterable of length at least 2 (x-scale, y-scale) can be provided.
        When getting pixel scale, a tuple of two values is returned with a
        pixel scale for each axis.

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

    @property
    def peak_fit_box(self):
        """ Set/Get ``peak_fit_box`` property. See `Star` for more details. """
        return self._peak_fit_box

    @peak_fit_box.setter
    def peak_fit_box(self, peak_fit_box):
        self._peak_fit_box = peak_fit_box

    @property
    def peak_search_box(self):
        """
        Set/Get ``peak_search_box`` property. See `Star` for more details.

        """
        return self._peak_search_box

    @peak_search_box.setter
    def peak_search_box(self, peak_search_box):
        self._peak_search_box = peak_search_box


class Weights(UnknownUncertainty):
    """ Convenience class for defining weights to be used
    with `~astropy.nddata.NDData` input images.

    """
    @property
    def uncertainty_type(self):
        return 'weights'


def extract_stars(data, catalogs, box_size=11, recenter=False):
    """
    Extract cutout images centered on stars defined in the input
    catalog(s).

    Parameters
    ----------
    data : `~astropy.nddata.NDData`, list of `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object or a list of
        `~astropy.nddata.NDData` objects containing the 2D image(s) from
        which to extract the stars.  If the input ``catalogs`` contain
        only the sky coordinates of the stars then each of the
        `~astropy.nddata.NDData` objects must have a valid ``wcs``
        attribute.

    catalogs : `~astropy.table.Table`, list of `~astropy.table.Table`
        A catalog or list of catalogs of sources to be extracted from
        the input ``data``.  To link stars in multiple images as a
        single source, you must use a single source catalog where the
        positions defined in sky coordinates.

        If a list of catalogs is input, they are assumed to correspond
        to the list of `~astropy.nddata.NDData` objects input in
        ``data`` (i.e. a separate source catalog for each 2D image).
        For multiple input catalogs, the center of each source can be
        defined either in pixel coordinates (in ``x`` and ``y`` columns)
        or sky coordinates (in a ``skycoord`` column containing a
        `~astropy.coordinates.SkyCoord` object).  If both are specified,
        then the pixel coordinates will be used.

        If a single source catalog is input, then these sources will be
        extracted from every 2D image in the input ``data``.  In this
        case, the sky coordinates for each source must be specified as a
        `~astropy.coordinates.SkyCoord` object contained in a column
        called ``skycoord``.  Each `~astropy.nddata.NDData` object in
        the input ``data`` must also have a valid ``wcs`` attribute.

        Optionally, each catalog may also contain an ``id`` column
        representing the ID/name of stars.  If this column is not
        present then the extracted stars will be given an ``id`` number
        corresponding the the table row number (starting at 1).  Any
        other columns presents in the input ``catalogs`` will be
        ignored.

    box_size : int or array_like (int), optional
        The extraction box size along each axis.  If ``box_size`` is a
        scalar then a square box of size ``box_size`` will be used.  If
        ``box_size`` has two elements, they should be in ``(ny, nx)``
        order.

    recenter : bool, optional
        Whether to estimate a new source position within the extracted
        box.

    Returns
    -------
    starlist : `Stars` instance
        A `Stars` instance containing the extracted stars.
    """

    if isinstance(data, NDData):
        data = [data]

    if isinstance(catalogs, Table):
        catalogs = [catalogs]

    for im in data:
        if not isinstance(im, NDData):
            raise ValueError('data must be a single or list of NDData '
                             'objects.')

    for cat in catalogs:
        if not isinstance(cat, Table):
            raise ValueError('catalogs must be a single or list of Table '
                             'objects.')

    if len(catalogs) == 1:
        if 'skycoord' not in catalogs[0].colnames:
            raise ValueError('When inputting a single catalog, it must '
                             'have a "skycoord" column.')

        if any([im.wcs is None for im in data]):
            raise ValueError('When inputting a single catalog, each NDData '
                             'object must have a wcs attribute.')
    else:
        for cat in catalogs:
            if (('x' not in cat.colnames or 'y' not in cat.colnames) and
                    ('skycoord' not in cat.colnames)):
                raise ValueError('When inputting multiple catalogs, '
                                 'each one must have a "x" and "y" '
                                 'column or a "skycoord" column.')

        if len(data) != len(catalogs):
            raise ValueError('When inputting multiple catalogs, the number '
                             'of catalogs must match the number of input '
                             'images.')

    if len(catalogs) == 1:
        stars = []
        for im in data:
            stars.append(_extract_stars(im, catalogs[0], box_size=box_size,
                                        recenter=recenter))

        # transpose the list of lists to associate linked stars
        stars = list(map(list, zip(*stars)))

        # remove 'None' stars (i.e. no or partial overlap in one or more
        # images) and handle the case of only one "linked" star
        stars_out = []
        for star in stars:
            st = [i for i in star if i is not None]
            if len(st) == 0:
                continue    # no overlap in any image
            elif len(st) == 1:
                st = st[0]    # only one star, so cannot be linked
            stars_out.append(st)
    else:
        stars_out = []
        for im, cat in zip(data, catalogs):
            stars_out.append(_extract_stars(im, cat, box_size=box_size,
                                            recenter=recenter))

    return Stars(stars_out)


def _extract_stars(data, catalog, box_size=11, recenter=False):
    """
    Extract cutout images from a single image centered on stars defined
    in the single input catalog.
    """

    colnames = catalog.colnames
    if 'x' not in colnames or 'y' not in colnames:
        if 'skycoord' not in colnames:
            raise ValueError('catalog does not contain required source '
                             'sky coordinates.')

        if data.wcs is None:
            raise ValueError('When the source catalog contains sky '
                             'coordinates, the data object must have a '
                             'valid wcs attribute.')

        xcens, ycens = skycoord_to_pixel(catalog['skycoord'], data.wcs,
                                         origin=0, mode='all')
    else:
        xcens = catalog['x'].data.astype(np.float)
        ycens = catalog['y'].data.astype(np.float)

    box_size = np.atleast_1d(box_size)
    if len(box_size) == 1:
        box_size = np.repeat(box_size, 2)
    min_size = 3
    if box_size[0] < min_size or box_size[1] < min_size:
        raise ValueError('box_size must be >= {} for x and y'
                         .format(min_size))
    shape = box_size[::-1]

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
    for xcen, ycen, idi in zip(xcens, ycens, ids):
        try:
            large_slc, small_slc = overlap_slices(data.data.shape, shape,
                                                  (ycen, xcen), mode='strict')
            cutout = data.data[large_slc]
            weights_cutout = weights[large_slc]
        except (PartialOverlapError, NoOverlapError):
            stars.append(None)
            continue

        if recenter:
            mask = (weights_cutout > 0.0)

            x0 = xcen - large_slc[1].start
            y0 = ycen - large_slc[0].start
            xnew, ynew = find_peak(cutout, x0, y0, mask=mask)
            xcen = np.int(xnew + large_slc[1].start + 0.5)
            ycen = np.int(ynew + large_slc[0].start + 0.5)

            try:
                large_slc, small_slc = overlap_slices(data.data.shape, shape,
                                                      (ycen, xcen),
                                                      mode='strict')
                cutout = data.data[large_slc]
                weights_cutout = weights[large_slc]
            except (PartialOverlapError, NoOverlapError):
                stars.append(None)

        origin = (large_slc[1].start, large_slc[0].start)
        star = Star(cutout, weights_cutout, center=(xcen, ycen),
                    recenter=False, origin=origin, wcs=data.wcs, id=idi)

        stars.append(star)

    return stars
