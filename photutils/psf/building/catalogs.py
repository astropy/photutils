"""
This module provides tools for source extraction from images and catalogs.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np
from astropy.table import Table
from astropy.nddata import NDData
from astropy.nddata.nddata import UnknownUncertainty
from astropy import wcs

from .centroid import find_peak
from .utils import py2round, interpolate_missing_data

__all__ = [
    'extract_stars', 'sim_extract_stars', 'Weights', 'Star', 'expand_starlist'
]

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


class Weights(UnknownUncertainty):
    """ Convenience class for defining weights to be used
    with `~astropy.nddata.NDData` input images.

    """
    @property
    def uncertainty_type(self):
        return 'weights'


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

    blc : tuple of two int
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
                 blc=(0, 0), wcs=None, meta={}):

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
        self.blc = blc

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
        self.id = None

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
            log.debug("Unrecognized keyword arguments to 'refine_center' "
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
    def is_linked(self):
        """ Check if this star is linked to other stars. """
        return (self._next is None and self._prev is None)

    @property
    def prev(self):
        """ Get/Set previous (i.e., to the left) linked star. """
        return self._prev

    @prev.setter
    def prev(self, new_previous):
        if new_previous is not None:
            ll1 = set(self.get_linked_list_right())
            ll2 = set(new_previous.get_linked_list_left())
            if not ll1.isdisjoint(ll2):
                raise ValueError("New node or one of its linked nodes is "
                                 "already a member of the linked list.")
            new_previous._next = self
        if self._prev is not None:
            self._prev._next = None
        self._prev = new_previous

    @property
    def next(self):
        """ Get/Set next (i.e., to the right) linked star. """
        return self._next

    @next.setter
    def next(self, new_next):
        if new_next is not None:
            ll1 = set(self.get_linked_list_left())
            ll2 = set(new_next.get_linked_list_right())
            if not ll1.isdisjoint(ll2):
                raise ValueError("New node or one of its linked nodes is "
                                 "already a member of the linked list.")
            new_next._prev = self
        if self._next is not None:
            self._next._prev = None
        self._next = new_next

    @property
    def last(self):
        """ Get last (i.e., furthest away to the right) linked star. """
        node = self
        while node.next is not None:
            node = node.next
        return node

    @property
    def first(self):
        """ Get first (i.e., furthest away to the left) linked star. """
        node = self
        while node.prev is not None:
            node = node.prev
        return node

    def remove_this_node(self):
        """ Remove this node (star) from the linked list. """
        prev = self.prev
        next = self.next
        self.prev = None
        self.next = None
        if prev is not None:
            prev.next = next
        if next is not None:
            next.prev = prev

    def append_first(self, node):
        """ Append ``node`` to the left of the first node in the list. """
        if node is None:
            return

        ll1 = set(self.get_linked_list())
        ll2 = set(node.get_linked_list_left())
        if not ll1.isdisjoint(ll2):
            raise ValueError("New node or one of its linked nodes is "
                             "already a member of the linked list.")

        first = self.first
        first._prev = node
        node._next = first

    def append_last(self, node):
        """ Append ``node`` to the right of the last node in the list. """
        if node is None:
            return

        ll1 = set(self.get_linked_list())
        ll2 = set(node.get_linked_list_right())
        if not ll1.isdisjoint(ll2):
            raise ValueError("New node or one of its linked nodes is "
                             "already a member of the linked list.")

        last = self.last
        last._next = node
        node._prev = last

    def insert_prev(self, new_previous):
        """ Insert ``node`` between this star and previous left star. """
        if new_previous is None:
            raise TypeError("New previous node must be a valid Star object.")
        if new_next in self.get_linked_list():
            raise ValueError("New node already a member of the linked list.")
        new_previous.remove_this_node()
        new_previous._next = self
        new_previous._prev = self._prev
        if self._prev is not None:
            self._prev._next = new_previous
        self._prev = new_previous

    def insert_next(self, new_next):
        """ Insert ``node`` between this star and previous right star. """
        if new_next is None:
            raise TypeError("New next node must be a valid Star object.")
        if new_next in self.get_linked_list():
            raise ValueError("New node already a member of the linked list.")
        new_next.remove_this_node()
        new_next._prev = self
        new_next._next = self._next
        if self._next is not None:
            self._next._prev = new_next
        self._next = new_next

    def get_linked_list(self):
        """ Get a list of all linked stars. """
        node = self.first
        nodes = [node]
        while node.next is not None:
            node = node.next
            nodes.append(node)
        return nodes

    def get_linked_list_left(self):
        """ Get a list of all linked to the left stars including self. """
        node = self
        nodes = [node]
        while node.prev is not None:
            node = node.prev
            nodes.append(node)
        return nodes

    def get_linked_list_right(self):
        """ Get a list of all linked to the right stars including self. """
        node = self
        nodes = [node]
        while node.next is not None:
            node = node.next
            nodes.append(node)
        return nodes

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


def expand_starlist(stars):
    """ Get a list of all stars including "hidden" linked stars. """

    if not hasattr(stars, '__iter__'):
        stars = [stars]

    expanded_starlist = []
    for s in stars:
        expanded_starlist += s.get_linked_list()

    return expanded_starlist


def extract_stars(images, catalogs, common_catalog=None, extract_size=11,
                  recenter=False, peak_fit_box=5, peak_search_box='fitbox',
                  catmap={'x': 'x', 'y': 'y',
                          'lon': 'lon', 'lat': 'lat',
                          'weight': 'weight', 'id': 'id'},
                  cat_name_kwd='name', image_name_kwd='name'):
    """Extracts sub-images centered on stars from a catalog.

    Given input catalogs of source coordinates in an image, and corrsponding
    images, this function extracts small sub-images centered
    on source coordinates from input catalogs. These sub-images are re-packaged
    as discrete 2D fittable models.

    ``common_catalog`` provides world coordinates of sources that need to be
    extracted from *all* input images. For each source coordinate in this
    catalog only one `Star` object is returned (extracted from one of the
    input images) and the `Star` objects correesponding to the same star in
    other images are *linked* to the first (returned) `Star`.

    Parameters
    ----------
    images : astropy.nddata.NDData, list of astropy.nddata.NDData
        A :py:class:`~astropy.nddata.NDData` container or a list of
        :py:class:`~astropy.nddata.NDData` containers. If input ``catalog``
        contains only world coordinates of the sources or if
        ``common_catalog`` is provided, then ``image`` must contain a valid
        ``wcs`` attribute.

    catalogs : astropy.table.Table, list of astropy.table.Table
        A catalog or list of catalogs of sources to be "extracted" from the
        input image(s). It must be a (a list of)
        :py:class:`~astropy.table.Table` catalog(s) containing image
        (or world) coordinates of the sources to be extracted. If input
        ``catalogs`` contains only world coordinates of the sources, then
        each image in ``images`` must contain a valid ``wcs`` attribute.
        The number of input catalogs must match the number of input images.

        Optionally, catalog may contain the following columns (all other
        columns will be ignored):

        - ``weight``: weight to be assigned to a star. This weight,
          for example, can be used for computing weighted world coordinates
          of a group of linked stars.

        - ``id``: name/ID of the star. If this column is not present in the
          catalog then each extracted star's ``id`` attribute will be set
          to the row number (first row being 1).

    common_catalog : astropy.table.Table
        A catalog of sources to be "extracted" from **all** input images. It
        must be a :py:class:`~astropy.table.Table` catalog containing *world*
        coordinates of the sources to be extracted. When ``common_catalog``
        is provided, input ``images`` must contain a valid ``wcs`` attribute.
        This catalog is used to create "linked" `Star` objects: all linked
        `Star` correspond to the same star on the sky and this information can
        be used to impose additional constrain on star coordinates when
        building a PSF (see :py:meth:`~Star.constrain_linked_centers` for
        more details).

        Optionally, catalog may contain the following columns (all other
        columns will be ignored):

        - ``weight``: weight to be assigned to a star. This weight,
          for example, can be used for computing weighted world coordinates
          of a group of linked stars.

        - ``id``: name/ID of the star. If this column is not present in the
          catalog then each extracted star's ``id`` attribute will be set
          to the row number (first row being 1).

    extract_size : int, tuple, numpy.ndarray, optional
        Indicates the size of the extraction region for each source. If a
        single number is provided, then a square extraction region with sides
        of length ``extract_size`` will be used for each source. A tuple can be
        used to indicate a rectangular extraction region (for all sources)
        with different sides along ``X-`` and ``Y-`` axes. Alternatively, a
        2D array of size ``Nx2`` can be used to indicate a different extraction
        box size for each source.

    recenter : bool, optional
        Indicates that a new source position should be estimated by fitting a
        quadratic polynomial to pixels around the star's position
        (either taken from the catalog or by performing a brute-search of
        the peak pixel value within a specified search box - see
        ``peak_search_box`` parameter for more details). This may be useful if
        the positions of the sources in the `catalog` are not very accurate.

        See `Star.recenter` for more details.

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
        ``peak_search_box`` is ``'all'`` then the entire extraction box of the
        star is searched for maximum and when it is set to ``'fitbox'`` then
        the brute-force search is performed in the same box as
        ``peak_fit_box``.

    catmap : dict, optional
        A `dict` that provides mapping between source's ``x``, ``y``,
        ``longitude``, ``latitude``, ``weight``, and ``id`` and the
        corresponding column names in a :py:class:`~astropy.table.Table`
        catalog. This parameter is ignored if input catalog is a
        :py:class:`~numpy.ndarray`.

    cat_name_kwd : str, optional
        Keyword name that identifies catalog name in its ``meta`` attribute
        when input catalog is a :py:class:`~astropy.table.Table`. This
        parameter is ignored if input catalog is a :py:class:`~numpy.ndarray`.
        If catalog does not contain name in its meta, then ``catalog_name``
        attribute of the returned stars will be set to 'Unknown'.

    image_name_kwd : str, optional
        Keyword name that identifies image name in its ``meta`` attribute
        when input image is a :py:class:`~astropy.nddata.NDData`. This
        parameter is ignored if input catalog is a :py:class:`~numpy.ndarray`.
        If image does not contain name in its meta, then ``image_name``
        attribute of the returned stars will be set to 'Unknown'.

    Returns
    -------
    starlist : list of Star
        A list of :py:class:`Star` objects one for each source in the catalog
        that is within the input image(s).

    """
    # first, check that input catalog column map contains required keywords
    # and if not - set them to defaults:
    loncol = catmap.get('lon', 'lon')
    latcol = catmap.get('lat', 'lat')

    # make sure the number of catalogs matches the number of catalogs provided:
    if isinstance(images, NDData):
        images = [images]

        if isinstance(catalogs, Table):
            catalogs = [catalogs]
        elif not isinstance(catalogs[0], Table):
            raise TypeError("Unsupported catalog type. ")

    else:
        if hasattr(images, '__iter__'):
            for i in images:
                if not isinstance(i, NDData):
                    raise ValueError("'images' must be a list of 'NDData' "
                                     "objects or a single 'NDData' object.")

        else:
            raise ValueError("'images' must be a list of 'NDData' objects or "
                             "a single 'NDData' object.")

        if not hasattr(catalogs, '__iter__') or len(images) != len(catalogs):
            raise ValueError("Number of catalogs must match the number of "
                             "input images")

        for i in catalogs:
            if not isinstance(i, Table):
                raise ValueError("'catalogs' must be a list of 'Table' "
                                 "objects.")

    if common_catalog is not None:
        # check to see that at least some images have valid WCS
        # if 'common_catalog' is provided:
        if not any([i.wcs is not None for i in images]):
            raise ValueError("At least one of the images must have a valid "
                             "WCS when 'common_catalog' is provided.")

        # check that common_catalog has world coordinates:
        colnames = common_catalog.colnames
        if loncol not in colnames or latcol not in colnames:
            raise ValueError("Source coordinates in 'common_catalog' must be "
                             "world coordinates.")

    # extract stars from individual image catalogs:
    stars = []
    for image, catalog in zip(images, catalogs):
        stars += sim_extract_stars(
            image=image,
            catalog=catalog,
            extract_size=extract_size,
            peak_fit_box=peak_fit_box,
            peak_search_box=peak_search_box,
            recenter=recenter,
            catmap=catmap,
            cat_name_kwd=cat_name_kwd,
            image_name_kwd=image_name_kwd,
            _ignored_as_None=False
        )

    if common_catalog is None:
        return stars

    # extract stars from "common_catalog" producing linked stars:
    linked_stars = []
    for image in images:
        if image.wcs is None:
            continue
        linked_stars.append(
            sim_extract_stars(
                image=image,
                catalog=catalog,
                extract_size=extract_size,
                peak_fit_box=peak_fit_box,
                peak_search_box=peak_search_box,
                recenter=recenter,
                catmap=catmap,
                cat_name_kwd=cat_name_kwd,
                image_name_kwd=image_name_kwd,
                _ignored_as_None=True
            )
        )

    if len(linked_stars) == 1:  # special case (for performance)
        stars += [s for s in linked_stars[0] if s is not None]
        return stars

    nlink = len(linked_stars)
    nstar = len(linked_stars[0])
    for k in range(nstar):
        lst = []
        for l in range(nlink):
            s = linked_stars[l][k]
            if s is not None:
                lst.append(s)

        if len(lst) == 0:
            continue

        s = lst[0]
        sp = s
        for si in lst[1:]:
            sp.next = si
            sp = si
        stars.append(s)


def sim_extract_stars(image, catalog, extract_size=11, recenter=False,
                      peak_fit_box=5, peak_search_box='fitbox',
                      catmap={'x': 'x', 'y': 'y',
                              'lon': 'lon', 'lat': 'lat',
                              'weight': 'weight', 'id': 'id'},
                      cat_name_kwd='name', image_name_kwd='name',
                      _ignored_as_None=False):
    """Extracts sub-images centered on stars from a catalog from **single** \
images.

    Given an input catalog of source coordinates in an image, the image and
    optionally a weight map, this function extracts small sub-images centered
    on source coordinates in the catalog. These sub-images are re-packaged
    as discrete 2D fittable models.

    Parameters
    ----------
    image : astropy.nddata.NDData, numpy.ndarray
        A :py:class:`~astropy.nddata.NDData` container or a
        :py:class:`~numpy.ndarray`. If input ``catalog``
        contains only world coordinates of the sources, then ``image`` must
        be a :py:class:`~astropy.nddata.NDData` container and contain a valid
        ``wcs`` attribute.

    catalog : astropy.table.Table, numpy.ndarray
        A catalog of sources to be "extracted" from the input image.
        The following are acceptable forms for catalog:

        * A :py:class:`numpy.ndarray` with 2 or 3 columns or a string to a text
          file containing 2 or three columns. First column must contain the
          ``x`` coordinate (in pixels) of the sources in the input image and
          the second column must contain the ``y`` coordinates of the sources.
          The optional third column, when present, is assumed to contain
          stars' weights.

        * A :py:class:`~astropy.table.Table` catalog containing image
          (or world) coordinates of the sources to be extracted. If input
          ``catalog`` contains only world coordinates of the sources, then
          ``image`` must contain a valid ``wcs`` attribute.

          Optionally, a :py:class:`~astropy.table.Table` catalog may contain
          the following columns (all other columns will be ignored):

          - ``weight``: weight to be assigned to a star. This weight,
            for example, can be used for computing weighted world coordinates
            of a group of linked stars.

          - ``id``: name/ID of the star. If this column is not present in the
            catalog then each extracted star's ``id`` attribute will be set
            to the row number (first row being 1).

    extract_size : int, tuple, numpy.ndarray, optional
        Indicates the size of the extraction region for each source. If a
        single number is provided, then a square extraction region with sides
        of length ``extract_size`` will be used for each source. A tuple can be
        used to indicate a rectangular extraction region (for all sources)
        with different sides along ``X-`` and ``Y-`` axes. Alternatively, a
        2D array of size ``Nx2`` can be used to indicate a different extraction
        box size for each source.

    recenter : bool, optional
        Indicates that a new source position should be estimated by fitting a
        quadratic polynomial to pixels around the star's position
        (either taken from the catalog or by performing a brute-search of
        the peak pixel value within a specified search box - see
        ``peak_search_box`` parameter for more details). This may be useful if
        the positions of the sources in the `catalog` are not very accurate.

        See `Star.recenter` for more details.

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
        ``peak_search_box`` is ``'all'`` then the entire extraction box of the
        star is searched for maximum and when it is set to ``'fitbox'`` then
        the brute-force search is performed in the same box as
        ``peak_fit_box``.

    catmap : dict, optional
        A `dict` that provides mapping between source's ``x``, ``y``,
        ``longitude``, ``latitude``, ``weight``, and ``id`` and the
        corresponding column names in a :py:class:`~astropy.table.Table`
        catalog. This parameter is ignored if input catalog is a
        :py:class:`~numpy.ndarray`.

    cat_name_kwd : str, optional
        Keyword name that identifies catalog name in its ``meta`` attribute
        when input catalog is a :py:class:`~astropy.table.Table`. This
        parameter is ignored if input catalog is a :py:class:`~numpy.ndarray`.
        If catalog does not contain name in its meta, then ``catalog_name``
        attribute of the returned stars will be set to 'Unknown'.

    image_name_kwd : str, optional
        Keyword name that identifies image name in its ``meta`` attribute
        when input image is a :py:class:`~astropy.nddata.NDData`. This
        parameter is ignored if input catalog is a :py:class:`~numpy.ndarray`.
        If image does not contain name in its meta, then ``image_name``
        attribute of the returned stars will be set to 'Unknown'.


    Other Parameters
    ----------------

    _ignored_as_None : bool, optional
        Indicates that sources that have not been extracted (for example,
        because they are outside the image) be returned as `None` in the
        returned list. By default, omitted sources are not included in the
        returned list.


    Returns
    -------
    starlist : list of Star
        A list of :py:class:`Star` objects one for each source in the catalog
        that is within the input image.

    """
    fitwindow = peak_fit_box
    # first, check that input catalog column map contains required keywords
    # and if not - set them to defaults:
    xcol = catmap.get('x', 'x')
    ycol = catmap.get('y', 'y')
    loncol = catmap.get('lon', 'lon')
    latcol = catmap.get('lat', 'lat')
    weightcol = catmap.get('weight', 'weight')
    idcol = catmap.get('id', 'id')

    catname = 'Unknown'
    imname = 'Unknown'

    if isinstance(catalog, Table):
        colnames = catalog.colnames
        if xcol not in colnames or ycol not in colnames:
            if loncol not in colnames or latcol not in colnames:
                raise ValueError("Catalog does not contain required "
                                 "source coordinates")

            if not isinstance(image, NDData) or image.wcs is None:
                raise ValueError("When source catalog contains world "
                                 "coordinates, 'image' object must be an"
                                 "NDData object with a valid WCS object.")

            # convert sky coordinates to image coordinates:
            lon = np.asarray(catalog[loncol], dtype=np.float)
            lat = np.asarray(catalog[latcol], dtype=np.float)
            srcx, srxy = image.wcs.all_world2pix(lon, lat, 0)

        else:
            # catalog already contains coordinates in image coordinates
            srcx = np.asarray(catalog[xcol], dtype=np.float)
            srcy = np.asarray(catalog[ycol], dtype=np.float)

        # extract source weights and ids:
        if weightcol in colnames:
            wght = np.asarray(catalog[weightcol], dtype=np.float)
        else:
            wght = np.ones_like(srcx)

        if idcol in colnames:
            sid = catalog[idcol]
        else:
            sid = np.arange(1, len(srcx) + 1, dtype=np.int)

        if cat_name_kwd in catalog.meta:
            catname = catalog.meta[cat_name_kwd]

    else:
        # assume catalog is a Nx2 array of image coordinates
        tmp = np.asarray(catalog)
        if len(tmp.shape) != 2:
            raise ValueError("Catalogs must be arrays of shape NxM with M>=2.")
        srcx = tmp[:, 0]
        srcy = tmp[:, 1]

        # extract source weights and ids:
        if tmp.shape[-1] > 2:
            wght = tmp[:, 2]
        else:
            wght = np.ones_like(srcx)

        sid = np.arange(1, len(srcx) + 1, dtype=np.int)

    if isinstance(image, NDData):
        wcs = image.wcs

        if image.uncertainty is None:
            weights = None
        else:
            if image.uncertainty.uncertainty_type != 'weights':
                log.warning("Input image's uncertainty has an unsupported "
                            "'uncertainty_type'. Only 'weights' type can be "
                            "used as weight. Setting weights to 1.")

            weights = np.asarray(image.uncertainty.array, dtype=np.float)

        if image.mask is not None:
            if weights is None:
                weights = np.asarray(image.mask == 0, dtype=np.float)
            else:
                weights[image.mask] = 0.0

        if image_name_kwd in image.meta:
            imname = image.meta[image_name_kwd]

        image = image.data

    elif isinstance(image, np.ndarray):
        wcs = None
        weights = None

    else:
        raise TypeError("'image' must be either a NDData or a numpy.ndarray "
                        "object.")

    # process extraction region size:
    bsize = np.empty((len(catalog), 2), dtype=np.int)
    if hasattr(extract_size, '__iter__'):
        es = np.asarray(extract_size, dtype=np.int)
        if len(es.shape) == 1:
            esx = es[0]
            esy = es[1]
            if esx < 3:
                log.warning("'extract_size' along X-axis is too small. "
                            "Setting 'extract_size' to 3.")
                esx = 3
            if esy < 3:
                log.warning("'extract_size' along Y-axis is too small. "
                            "Setting 'extract_size' to 3.")
                esy = 3
            bsize[:, 0] = esx
            bsize[:, 1] = esy
        elif (len(es.shape) == 2 and es.shape[1] > 1 and
              (es.shape[0] == len(srcx) or es.shape[0] == 1)):
            bsize[:, :] = es[:, :2]
            # check 'extract_size':
            if not np.all(bsize > 2):
                log.warning("'extract_size' is too small for some sources. "
                            "Setting 'extract_size' to min 3 in those cases.")
                bsize[bsize < 3] = 3
        else:
            raise ValueError("'extract_size' must be an integer number, or a "
                             "1D vector of length 2, or a 2D iterable of "
                             "shape Nx2 where N is equal to the number of "
                             "sources in the input catalog.")
    else:
        # check 'extract_size':
        if extract_size < 3:
            log.warning("'extract_size' is too small. Setting it to 3")
            extract_size = 3
        bsize[:, :] = extract_size

    # extract stars:
    starlist = []
    blc = []
    ny, nx = image.shape
    for x, y, (w, h), wt, i in zip(srcx, srcy, bsize, wght, sid):
        xc = int(x)
        yc = int(y)
        x1 = max(0, xc - (w - 1) // 2)
        x2 = min(nx, x1 + w)
        y1 = max(0, yc - (h - 1) // 2)
        y2 = min(ny, y1 + h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            log.warning("Source with coordinates ({}, {}) is being ignored "
                        "because there are too few pixels available around "
                        "its center pixel.".format(x, y))
            if _ignored_as_None:
                starlist.append(None)
            continue

        cutout = image[y1:y2, x1:x2]

        if weights is None:
            wghts = None
            effmask = None
        else:
            wghts = weights[y1:y2, x1:x2]
            effmask = (wghts > 0.0)

        if recenter:
            xnew, ynew = find_peak(
                cutout, x - x1, y - y1,
                peak_fit_box=fitwindow, peak_search_box=peak_search_box,
                mask=effmask
            )
            xnew += x1
            ynew += y1
            # re-compute extraction box with improved center value:
            xc = int(py2round(xnew))
            yc = int(py2round(ynew))
            x1 = max(0, xc - (w - 1) // 2)
            x2 = min(nx, x1 + w)
            y1 = max(0, yc - (h - 1) // 2)
            y2 = min(ny, y1 + h)
            if x2 - x1 < 3 or y2 - y1 < 3:
                log.warning("Source with coordinates ({}, {}) is being "
                            "ignored because there are too few pixels "
                            "available around its center pixel.".format(x, y))
                if _ignored_as_None:
                    starlist.append(None)
                continue
            x_center = xnew - x1
            y_center = ynew - y1
            cutout = image[y1:y2, x1:x2]
        else:
            x_center = x - x1
            y_center = y - y1

        # construct Star object and set its attributes
        star = Star(data=cutout, weights=wghts, flux=None,
                    center=(x_center, y_center),
                    recenter=recenter, peak_fit_box=fitwindow,
                    blc=(x1, y1), wcs=wcs)

        star.star_weight = wt
        star.id = i
        star.catalog_name = catname
        star.image_name = imname

        starlist.append(star)

    return starlist
