# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by an Aperture.
"""

from copy import deepcopy
import functools
import inspect
import warnings

from astropy.stats import (SigmaClip, mad_std, biweight_location,
                           biweight_midvariance)
from astropy.table import QTable
import astropy.units as u
from astropy.utils import lazyproperty
import numpy as np

from . import Aperture
from ..utils._misc import _get_meta
from ..utils._moments import _moments, _moments_central
from ..utils._quantity_helpers import process_quantities

__all__ = ['ApertureStats']


# default table columns for `to_table()` output
DEFAULT_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'area', 'min', 'max', 'sum', 'sum_err', 'mean',
                   'median', 'mode', 'std', 'mad_std', 'var',
                   'biweight_location', 'biweight_midvariance',
                   'fwhm', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity']


def as_scalar(method):
    """
    Return a scalar value from a method if the class is scalar.
    """
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return (result[0] if args[0].isscalar and len(result) == 1
                    else result)
        except TypeError:  # if result has no len
            return result
    return _decorator


class ApertureStats:
    """
    Class to create a catalog of statistics for sources defined by an
    aperture.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """

    def __init__(self, data, aperture, *, error=None, mask=None, wcs=None,
                 sigma_clip=None, sum_method='exact', subpixels=5):

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        self._data = self._validate_array(data, 'data', shape=False)
        self._data_unit = unit
        self.aperture = self._validate_aperture(aperture)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._wcs = wcs
        if sigma_clip is not None and not isinstance(sigma_clip, SigmaClip):
            raise TypeError('sigma_clip must be a SigmaClip instance')
        self.sigma_clip = sigma_clip
        self.sum_method = sum_method
        self.subpixels = subpixels

        self._ids = np.arange(self.n_apertures) + 1
        self._data_mask = self._make_data_mask()
        self.default_columns = DEFAULT_COLUMNS
        self.meta = _get_meta()

    @staticmethod
    def _validate_aperture(aperture):
        if not isinstance(aperture, Aperture):
            raise TypeError('aperture must be an Aperture object')
        return aperture

    def _validate_array(self, array, name, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                raise ValueError(f'{name} must be a 2D array.')
            if shape and array.shape != self._data.shape:
                raise ValueError(f'data and {name} must have the same shape.')
        return array

    @property
    def _properties(self):
        """
        A list of all class properties, including lazyproperties (even in
        superclasses).
        """
        def isproperty(obj):
            return isinstance(obj, property)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=isproperty)]

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
        """
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    @property
    def properties(self):
        """
        A sorted list of built-in source properties.
        """
        lazyproperties = [name for name in self._lazyproperties if not
                          name.startswith('_')]
        lazyproperties.sort()
        return lazyproperties

    def __getitem__(self, index):
        if self.isscalar:
            raise TypeError(f'A scalar {self.__class__.__name__!r} object '
                            'cannot be indexed')

        newcls = object.__new__(self.__class__)

        # attributes defined in __init__ that are copied directly to the
        # new class
        init_attr = ('_data', '_data_unit', '_error', '_mask', '_wcs',
                     '_data_mask', 'default_columns', 'meta')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # need to slice _aperture and _ids;
        # aperture determines isscalar (needed below)
        attrs = ('aperture', '_ids')
        for attr in attrs:
            setattr(newcls, attr, getattr(self, attr)[index])

        # slice evaluated lazyproperty objects
        keys = (set(self.__dict__.keys()) & set(self._lazyproperties))
        for key in keys:
            value = self.__dict__[key]

            # do not insert attributes that are always scalar (e.g.,
            # isscalar, n_apertures), i.e., not an array/list for each
            # source
            if np.isscalar(value):
                continue

            try:
                # keep _<attrs> as length-1 iterables
                if newcls.isscalar and key.startswith('_'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                # apply fancy indices (e.g., array/list or bool
                # mask) to lists
                val = (np.array(value + [None],
                                dtype=object)[:-1][index]).tolist()

            newcls.__dict__[key] = val
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.n_apertures}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            raise TypeError(f'Scalar {self.__class__.__name__!r} object has '
                            'no len()')
        return self.n_apertures

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single aperture position).
        """
        return self.aperture.isscalar

    def copy(self):
        """
        Return a deep copy of this object.
        """
        return deepcopy(self)

    def _make_data_mask(self):
        """
        Create a mask of non-finite ``data`` values combined with the
        input ``mask`` array.
        """
        mask = ~np.isfinite(self._data)
        if self._mask is not None:
            mask |= self._mask
        return mask

    @lazyproperty
    def _null_object(self):
        """
        Return `None` values.
        """
        return np.array([None] * self.n_apertures)

    @lazyproperty
    def _null_value(self):
        """
        Return np.nan values.
        """
        values = np.empty(self.n_apertures)
        values.fill(np.nan)
        return values

    def get_id(self, id_num):
        """
        Return a new `ApertureStats` object for the input ID number
        only.

        Parameters
        ----------
        id_num : int
            The aperture ID number.

        Returns
        -------
        result : `ApertureStats`
            A new `ApertureStats` object containing only the source with
            the input ID number.
        """
        return self.get_ids(id_num)

    def get_ids(self, id_nums):
        """
        Return a new `ApertureStats` object for the input ID numbers
        only.

        Parameters
        ----------
        id_nums : list, tuple, or `~numpy.ndarray` of int
            The aperture ID number(s).

        Returns
        -------
        result : `ApertureStats`
            A new `ApertureStats` object containing only the sources with
            the input ID numbers.
        """
        idx = np.searchsorted(self.id, id_nums)
        return self[idx]

    @property
    def ids(self):
        """
        The aperture identification number(s), always as an iterable
        `~numpy.ndarray`.
        """
        return self._id_iter

    @property
    def _id_iter(self):
        """
        The aperture identification number(s), always as an iterable
        `~numpy.ndarray`.
        """
        _id = self.id
        if self.isscalar:
            _id = np.array((_id,))
        return _id

    @property
    @as_scalar
    def id(self):
        """
        The aperture identification number(s).
        """
        return np.arange(self.n_apertures) + 1

    def to_table(self, columns=None):
        """
        Create a `~astropy.table.QTable` of source properties.

        Parameters
        ----------
        columns : str, list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are any of
            the `ApertureStats` properties. If ``columns`` is `None`,
            then a default list of scalar-valued properties (as defined
            by the ``default_columns`` attribute) will be used.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of sources properties with one row per source.
        """
        if columns is None:
            table_columns = self.default_columns
        else:
            table_columns = np.atleast_1d(columns)

        tbl = QTable(meta=self.meta)
        for column in table_columns:
            values = getattr(self, column)

            # column assignment requires an object with a length
            if self.isscalar:
                values = (values,)

            tbl[column] = values
        return tbl

    @lazyproperty
    def n_apertures(self):
        """
        The number of positions in the input aperture.
        """
        if self.isscalar:
            return 1
        return len(self.aperture)

    @property
    def _apertures(self):
        """
        The input apertures, always as an iterable.
        """
        apertures = self.aperture
        if self.isscalar:
            apertures = (apertures,)
        return apertures

    @lazyproperty
    def _aperture_masks(self):
        """
        The aperture masks (`ApertureMask`) generated with the 'center'
        method, always as an iterable.
        """
        aperture_masks = self.aperture.to_mask(method='center')
        if self.isscalar:
            aperture_masks = (aperture_masks,)
        return aperture_masks

    @lazyproperty
    def _overlap_slices(self):
        """
        The aperture mask overlap slices with the data, always as an
        iterable.

        The overlap slices are the same for all aperture mask methods.
        """
        overlap_slices = []
        for apermask in self._aperture_masks:
            (slc_large, slc_small) = apermask.get_overlap_slices(
                self._data.shape)
            overlap_slices.append((slc_large, slc_small))
        return overlap_slices

    @lazyproperty
    def _data_cutouts(self):
        """
        The data cutouts from the aperture bounding box, always as a
        iterable.
        """
        cutouts = []
        for slc_large, _ in self._overlap_slices:
            cutouts.append(self._data[slc_large])
        return cutouts

    @lazyproperty
    def _data_mask_cutouts(self):
        """
        The data mask cutouts from the aperture bounding box, with
        masked values set to zero, always as a iterable.

        The masked values here include the input mask and the non-finite
        data values.
        """
        mask_cutouts = []
        for slc_large, _ in self._overlap_slices:
            mask_cutouts.append(self._data_mask[slc_large])
        return mask_cutouts

    def _make_aperture_cutouts(self, aperture_masks):
        """
        Make cutouts for the data, error, and total mask.
        """
        data_cutouts = []
        error_cutouts = []
        mask_cutouts = []

        for (apermask, slices, data_cutout, data_mask) in zip(
                aperture_masks, self._overlap_slices, self._data_cutouts,
                self._data_mask_cutouts):

            slc_large, slc_small = slices
            if slc_large is None:  # aperture does not overlap the data
                mask_cutout = False
                data_cutout = None
                error_cutout = None
            else:
                apermask_cutout = apermask.data[slc_small]

                # apply the aperture mask; for "exact" and "subpixel"
                # this is an expanded boolean mask using the aperture
                # mask zero values
                mask_cutout = (apermask_cutout == 0) | data_mask

                if self.sigma_clip is None:
                    # data_cutout will have zeros where mask_cutout is True
                    data_cutout *= ~mask_cutout
                else:
                    # to input a mask, SigmaClip needs a MaskedArray
                    data_cutout_ma = np.ma.masked_array(data_cutout,
                                                        mask=mask_cutout)
                    data_sigclip = self.sigma_clip(data_cutout_ma)

                    mask_cutout = data_sigclip.mask
                    data_cutout = data_sigclip.filled(0.)

                if self._error is None:
                    error_cutout = None
                else:
                    # error_cutout will have zeros where mask_cutout is True
                    error_cutout = self._error[slc_large] * ~mask_cutout

            data_cutouts.append(data_cutout)
            error_cutouts.append(error_cutout)
            mask_cutouts.append(mask_cutout)

        # use zip (instead of np.transpose) because these may contain
        # arrays that have different shapes
        return list(zip(data_cutouts, error_cutouts, mask_cutouts))

    @lazyproperty
    def _aperture_cutouts(self):
        """
        Cutouts for the data, error, and total mask using the "center"
        aperture mask method.
        """
        return self._make_aperture_cutouts(self._aperture_masks)

    @lazyproperty
    def _aperture_cutouts_sum(self):
        """
        Cutouts for the data, error, and total mask using the input
        ``sum_method`` aperture mask method.
        """
        aperture_masks = self.aperture.to_mask(method=self.sum_method)
        if self.isscalar:
            aperture_masks = (aperture_masks,)
        return self._make_aperture_cutouts(aperture_masks)

    @lazyproperty
    def _cutout_total_mask(self):
        """
        Boolean mask representing the combination of ``_data_mask``,
        the cutout aperture mask, and the sigma-clip mask.

        This mask is applied to ``data`` and ``error`` inputs when
        calculating properties.
        """
        return list(zip(*self._aperture_cutouts))[2]

    @as_scalar
    def _make_masked_array(self, array):
        """
        Make masked arrays from cutouts.

        Units are not applied.
        """
        total_mask = self._cutout_total_mask

        return [np.ma.masked_array(arr, mask=mask)
                for arr, mask in zip(array, total_mask)]

    @lazyproperty
    @as_scalar
    def data(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the data using the
        aperture bounding box.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels outside of the aperture mask,
        pixels from the input ``mask``, non-finite ``data`` values (NaN
        and inf), and sigma-clipped pixels.
        """
        return self._make_masked_array(list(zip(*self._aperture_cutouts))[0])

    @lazyproperty
    @as_scalar
    def error(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the error array using
        the aperture bounding box.

        The cutout does not have units due to current limitations of
        masked quantity arrays.

        The mask is `True` for pixels outside of the aperture mask,
        pixels from the input ``mask``, non-finite ``data`` values (NaN
        and inf), and sigma-clipped pixels.
        """
        return self._make_masked_array(list(zip(*self._aperture_cutouts))[1])

    @lazyproperty
    def _cutout_moment_data(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the data. The
        following pixels are set to zero in these arrays:

            * any masked pixels
            * invalid values (NaN and inf)

        These arrays are used to derive moment-based properties.
        """
        data = deepcopy(self.data)  # self.data is a list
        if self.isscalar:
            data = (data,)

        cutouts = []
        for arr in data:
            arr_ = arr.data
            arr_[arr.mask] = 0.
            cutouts.append(arr_)

        return cutouts

    @lazyproperty
    def _all_masked(self):
        """
        True if all pixels within the aperture are masked.
        """
        return np.array([np.all(mask) for mask in self._cutout_total_mask])

    def _get_values(self, array):
        """
        Get a 1D array of unmasked values from the input array within
        the aperture.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self.isscalar:
            array = (array,)
        return [arr.compressed() if len(arr.compressed()) > 0
                else np.array([np.nan]) for arr in array]

    @lazyproperty
    def _data_values(self):
        """
        A 1D array of unmasked data values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.data)

    @lazyproperty
    def _error_values(self):
        """
        A 1D array of unmasked error values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.error)

    @lazyproperty
    @as_scalar
    def moments(self):
        """
        Spatial moments up to 3rd order of the source.
        """
        return np.array([_moments(arr, order=3) for arr in
                         self._cutout_moment_data])

    @lazyproperty
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        cutout_centroid = self.cutout_centroid
        if self.isscalar:
            cutout_centroid = cutout_centroid[np.newaxis, :]
        return np.array([_moments_central(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in
                         zip(self._cutout_moment_data, cutout_centroid[:, 0],
                             cutout_centroid[:, 1])])

    @lazyproperty
    @as_scalar
    def cutout_centroid(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of
        the centroid within the aperture.
        """
        moments = self.moments
        if self.isscalar:
            moments = moments[np.newaxis, :]

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ycentroid = moments[:, 1, 0] / moments[:, 0, 0]
            xcentroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((xcentroid, ycentroid))

    @lazyproperty
    @as_scalar
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid of the unmasked pixels
        within the aperture.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @lazyproperty
    def _xcentroid(self):
        """
        The ``x`` coordinate of the centroid of the unmasked pixels
        within the aperture, always as an iterable.
        """
        xcentroid = np.transpose(self.centroid)[0]
        if self.isscalar:
            xcentroid = (xcentroid,)
        return xcentroid

    @lazyproperty
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid of the unmasked pixels
        within the aperture.
        """
        return self._xcentroid

    @lazyproperty
    def _ycentroid(self):
        """
        The ``y`` coordinate of the centroid of the unmasked pixels
        within the aperture, always as an iterable.
        """
        ycentroid = np.transpose(self.centroid)[1]
        if self.isscalar:
            ycentroid = (ycentroid,)
        return ycentroid

    @lazyproperty
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid of the unmasked pixels
        within the aperture.
        """
        return self._ycentroid

    @lazyproperty
    @as_scalar
    def sky_centroid(self):
        """
        The sky coordinate of the centroid of the unmasked pixels within
        the aperture, returned as a `~astropy.coordinates.SkyCoord`
        object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(self.xcentroid, self.ycentroid)

    @lazyproperty
    @as_scalar
    def sky_centroid_icrs(self):
        """
        The sky coordinate in the International Celestial
        Reference System (ICRS) frame of the centroid of the
        unmasked pixels within the aperture, returned as a
        `~astropy.coordinates.SkyCoord` object.

        `None` if ``wcs`` is not input.
        """
        if self._wcs is None:
            return self._null_object
        return self.sky_centroid.icrs

    @lazyproperty
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the aperture, always as
        an iterable.
        """
        return [aperture.bbox for aperture in self._apertures]

    @lazyproperty
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the aperture.

        Note that the aperture bounding box is calculated using the
        exact size of the aperture, which may be slightly larger than
        the aperture mask calculated using the "center" mode.
        """
        return self._bbox

    @lazyproperty
    @as_scalar
    def _bbox_minmax(self):
        """
        The bounding box x/y minimum and maximum.
        """
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        return np.array([(bbox_.ixmin, bbox_.ixmax - 1,
                          bbox_.iymin, bbox_.iymax - 1)
                         for bbox_ in bbox])

    @lazyproperty
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel index of the bounding box.
        """
        return np.transpose(self._bbox_minmax)[0]

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index of the bounding box.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.transpose(self._bbox_minmax)[1]

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index of the bounding box.
        """
        return np.transpose(self._bbox_minmax)[2]

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index of the bounding box.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.transpose(self._bbox_minmax)[3]

    def _calculate_stats(self, stat_func, unit=None):
        """
        Apply the input ``stat_func`` to the 1D array of unmasked data
        values in the aperture.

        Units are applied if the input ``data`` has units.

        Parameters
        ----------
        stat_func : callable
            The callable to apply to the 1D `~numpy.ndarray` of unmasked
            data values.

        unit : `None` or `astropy.unit.Unit`, optional
            The unit to apply to the output data. This is used only
            if the input ``data`` has units. If `None` then the input
            ``data`` unit will be used.
        """
        if unit is None:
            unit = self._data_unit
        result = np.array([stat_func(arr) for arr in self._data_values])
        if self._data_unit is not None:
            result <<= unit
        return result

    @lazyproperty
    @as_scalar
    def sum(self):
        r"""
        The sum of the unmasked ``data`` values within the aperture.

        .. math:: F = \sum_{i \in A} I_i

        where :math:`F` is ``sum``, :math:`I_i` is the
        background-subtracted ``data``, and :math:`A` are the unmasked
        pixels in the aperture.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        return self._calculate_stats(np.sum)

    @lazyproperty
    @as_scalar
    def sum_err(self):
        r"""
        The uncertainty of `sum` , propagated from the input ``error``
        array.

        ``sum_err`` is the quadrature sum of the total errors over the
        unmasked pixels within the aperture:

        .. math:: \Delta F = \sqrt{\sum_{i \in A}
                  \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `sum`, :math:`\sigma_{\mathrm{tot,
        i}}` are the pixel-wise total errors (``error``), and :math:`A`
        are the unmasked pixels in the aperture.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the error array.
        """
        if self._error is None:
            err = self._null_value
        else:
            err = np.sqrt(np.array([np.sum(arr**2)
                                    for arr in self._error_values]))

        if self._data_unit is not None:
            err <<= self._data_unit
        return err

    @lazyproperty
    @as_scalar
    def min(self):
        """
        The minimum of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.min)

    @lazyproperty
    @as_scalar
    def max(self):
        """
        The maximum of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.max)

    @lazyproperty
    @as_scalar
    def mean(self):
        """
        The mean of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.mean)

    @lazyproperty
    @as_scalar
    def median(self):
        """
        The median of the unmasked pixel values within the aperture.
        """
        return self._calculate_stats(np.median)

    @lazyproperty
    @as_scalar
    def mode(self):
        """
        The mode of the unmasked pixel values within the aperture.

        The mode is estimated as ``(3 * median) - (2 * mean)``.
        """
        return 3. * self.median - 2. * self.mean

    @lazyproperty
    @as_scalar
    def std(self):
        """
        The standard deviation of the unmasked pixel values within the
        aperture.
        """
        return self._calculate_stats(np.std)

    @lazyproperty
    @as_scalar
    def mad_std(self):
        r"""
        The standard deviation calculated using
        the `median absolute deviation (MAD)
        <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

        The standard deviation estimator is given by:

        .. math::

            \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
                \approx 1.4826 \ \textrm{MAD}

        where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
        distribution function evaluated at probability :math:`P = 3/4`.
        """
        return self._calculate_stats(mad_std)

    @lazyproperty
    @as_scalar
    def var(self):
        """
        The variance of the unmasked pixel values within the aperture.
        """
        unit = self._data_unit
        if unit is not None:
            unit **= 2
        return self._calculate_stats(np.var, unit=unit)

    @lazyproperty
    @as_scalar
    def biweight_location(self):
        """
        The biweight location of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_location`.
        """
        return self._calculate_stats(biweight_location)

    @lazyproperty
    @as_scalar
    def biweight_midvariance(self):
        """
        The biweight midvariance of the unmasked pixel values within the
        aperture.

        See `astropy.stats.biweight_midvariance`
        """
        unit = self._data_unit
        if unit is not None:
            unit **= 2
        return self._calculate_stats(biweight_midvariance, unit=unit)

    @lazyproperty
    @as_scalar
    def area(self):
        """
        The total area of the unmasked pixel values within the aperture.

        Note that the source area may be smaller than the aperture area
        if a mask is input to `ApertureStats` or if the ``data`` within
        the aperture contains invalid values (NaN and inf).
        """
        areas = np.array([arr.size for arr in self._data_values]).astype(float)
        areas[self._all_masked] = np.nan
        return areas << (u.pix ** 2)

    @lazyproperty
    @as_scalar
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        mu_02 = moments[:, 0, 2]
        mu_11 = -moments[:, 1, 1]
        mu_20 = moments[:, 2, 0]
        tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
        return tensor.reshape((tensor.shape[0], 2, 2)) * u.pix**2

    @lazyproperty
    def _covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source, always as an iterable.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu_norm = moments / moments[:, 0, 0][:, np.newaxis, np.newaxis]

        covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                          mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
        covar = covar.reshape((covar.shape[0], 2, 2))

        # Modify the covariance matrix in the case of "infinitely" thin
        # detections. This follows SourceExtractor's prescription of
        # incrementally increasing the diagonal elements by 1/12.
        delta = 1. / 12
        delta2 = delta**2
        # ignore RuntimeWarning from NaN values in covar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            covar_det = np.linalg.det(covar)
            idx = np.where(covar_det < delta2)[0]
            while idx.size > 0:  # pragma: no cover
                covar[idx, 0, 0] += delta
                covar[idx, 1, 1] += delta
                covar_det = np.linalg.det(covar)
                idx = np.where(covar_det < delta2)[0]
        return covar

    @lazyproperty
    @as_scalar
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """
        return self._covariance * (u.pix**2)

    @lazyproperty
    @as_scalar
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.empty((self.n_apertures, 2))
        eigvals.fill(np.nan)
        # np.linalg.eivals requires finite input values
        idx = np.unique(np.where(np.isfinite(self._covariance))[0])
        eigvals[idx] = np.linalg.eigvals(self._covariance[idx])

        # check for negative variance
        # (just in case covariance matrix is not positive (semi)definite)
        idx2 = np.unique(np.where(eigvals < 0)[0])  # pragma: no cover
        eigvals[idx2] = (np.nan, np.nan)  # pragma: no cover

        # sort each eigenvalue pair in descending order
        eigvals.sort(axis=1)
        eigvals = np.fliplr(eigvals)

        return eigvals * u.pix**2

    @lazyproperty
    @as_scalar
    def semimajor_sigma(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self.covariance_eigvals
        if self.isscalar:
            eigvals = eigvals[np.newaxis, :]
        # this matches SourceExtractor's A parameter
        return np.sqrt(eigvals[:, 0])

    @lazyproperty
    @as_scalar
    def semiminor_sigma(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self.covariance_eigvals
        if self.isscalar:
            eigvals = eigvals[np.newaxis, :]
        # this matches SourceExtractor's B parameter
        return np.sqrt(eigvals[:, 1])

    @lazyproperty
    @as_scalar
    def fwhm(self):
        r"""
        The circularized full width at half maximum (FWHM) of the 2D
        Gaussian function that has the same second-order central moments
        as the source.

        .. math::

           \mathrm{FWHM} & = 2 \sqrt{2 \ln(2)} \sqrt{0.5 (a^2 + b^2)}
           \\
                          & = 2 \sqrt{\ln(2) \ (a^2 + b^2)}

        where :math:`a` and :math:`b` are the 1-sigma lengths of the
        semimajor (`semimajor_sigma`) and semiminor (`semiminor_sigma`)
        axes, respectively.
        """
        return 2.0 * np.sqrt(np.log(2.0) * (self.semimajor_sigma**2
                                            + self.semiminor_sigma**2))

    @lazyproperty
    @as_scalar
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction.
        """
        covar = self._covariance
        orient_radians = 0.5 * np.arctan2(2. * covar[:, 0, 1],
                                          (covar[:, 0, 0] - covar[:, 1, 1]))
        return orient_radians * 180. / np.pi * u.deg

    @lazyproperty
    @as_scalar
    def eccentricity(self):
        r"""
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math:: e = \sqrt{1 - \frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        semimajor_var, semiminor_var = np.transpose(self.covariance_eigvals)
        return np.sqrt(1. - (semiminor_var / semimajor_var))

    @lazyproperty
    @as_scalar
    def elongation(self):
        r"""
        The ratio of the lengths of the semimajor and semiminor axes.

        .. math:: \mathrm{elongation} = \frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_sigma / self.semiminor_sigma

    @lazyproperty
    @as_scalar
    def ellipticity(self):
        r"""
        1.0 minus the ratio of the lengths of the semimajor and
        semiminor axes (or 1.0 minus the `elongation`).

        .. math:: \mathrm{ellipticity} = 1 - \frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_sigma / self.semimajor_sigma)

    @lazyproperty
    @as_scalar
    def covar_sigx2(self):
        r"""
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\sigma_x^2`, in units of pixel**2.
        """
        return self._covariance[:, 0, 0] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigy2(self):
        r"""
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.
        """
        return self._covariance[:, 1, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigxy(self):
        r"""
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
        pixel**2.
        """
        return self._covariance[:, 0, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def cxx(self):
        r"""
        `SourceExtractor`_'s CXX ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return ((np.cos(self.orientation) / self.semimajor_sigma)**2
                + (np.sin(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cyy(self):
        r"""
        `SourceExtractor`_'s CYY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return ((np.sin(self.orientation) / self.semimajor_sigma)**2
                + (np.cos(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cxy(self):
        r"""
        `SourceExtractor`_'s CXY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
                cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \approx 3`.
        """
        return (2. * np.cos(self.orientation) * np.sin(self.orientation)
                * ((1. / self.semimajor_sigma**2)
                   - (1. / self.semiminor_sigma**2)))

    @lazyproperty
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        unmasked pixel values within the aperture.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        as:

        .. math::
            G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
            \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\bar{x}` is the mean over pixel values :math:`x_i`
        within the aperture.

        The Gini coefficient is a way of measuring the inequality in a
        given set of values. In the context of galaxy morphology, it
        measures how the light of a galaxy image is distributed among
        its pixels. A Gini coefficient value of 0 corresponds to a
        galaxy image with the light evenly distributed over all pixels
        while a Gini coefficient value of 1 represents a galaxy image
        with all its light concentrated in just one pixel.
        """
        gini = []
        for arr in self._data_values:
            if np.all(np.isnan(arr)):
                gini.append(np.nan)
                continue
            npix = np.size(arr)
            normalization = np.abs(np.mean(arr)) * npix * (npix - 1)
            kernel = ((2. * np.arange(1, npix + 1) - npix - 1)
                      * np.abs(np.sort(arr)))
            gini.append(np.sum(kernel) / normalization)
        return np.array(gini)
