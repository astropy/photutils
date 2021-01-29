# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by a segmentation image.
"""

import functools

from copy import copy
import inspect
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from ..aperture import BoundingBox
from ..utils._convolution import _filter_data
from ..utils._moments import _moments, _moments_central
from ..utils._wcs_helpers import _pixel_to_world

__all__ = ['SourceCatalog']
__doctest_requires__ = {('SourceCatalog', 'SourceCatalog.*'): ['scipy']}


# default table columns for `to_table()` output
DEFAULT_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'sky_centroid_icrs', 'source_sum', 'source_sum_err',
                   'background_sum', 'background_mean',
                   'background_at_centroid', 'bbox_xmin', 'bbox_xmax',
                   'bbox_ymin', 'bbox_ymax', 'min_value', 'max_value',
                   'minval_xpos', 'minval_ypos', 'maxval_xpos', 'maxval_ypos',
                   'area', 'equivalent_radius', 'perimeter',
                   'semimajor_axis_sigma', 'semiminor_axis_sigma',
                   'orientation', 'eccentricity', 'ellipticity', 'elongation',
                   'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy',
                   'cyy', 'gini']


def as_scalar(method):
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return (result[0] if args[0].isscalar and len(result) == 1
                    else result)
        except TypeError:
            return result
    return _decorator


class SourceCatalog:
    def __init__(self, data, segment_img, error=None, mask=None,
                 kernel=None, background=None, wcs=None):

        self._data_unit = None
        data, error, background = self._process_quantities(data, error,
                                                           background)
        self._data = data
        self._segment_img = self._validate_segment_img(segment_img)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._kernel = kernel
        self._background = self._validate_array(background, 'background')
        self._wcs = wcs

        # needed for ordering and isscalar
        self._labels = self._segment_img.labels
        self._slices = self._segment_img.slices
        self.default_columns = DEFAULT_COLUMNS

    def _process_quantities(self, data, error, background):
        """
        Check units of input arrays.

        If any of the input arrays have units then they all must have
        units and the units must be the same.

        Return unitless ndarrays with the array unit set in
        self._data_unit.
        """
        inputs = (data, error, background)
        has_unit = [hasattr(x, 'unit') for x in inputs if x is not None]
        use_units = all(has_unit)
        if any(has_unit) and not use_units:
            raise ValueError('If any of data, error, or background has '
                             'units, then they all must all have units.')
        if use_units:
            self._data_unit = data.unit
            data = data.value
            if error is not None:
                if error.unit != self._data_unit:
                    raise ValueError('error must have the same units as data')
                error = error.value
            if background is not None:
                if background.unit != self._data_unit:
                    raise ValueError('background must have the same units as '
                                     'data')
                background = background.value
        return data, error, background

    def _validate_segment_img(self, segment_img):
        if not isinstance(segment_img, SegmentationImage):
            raise ValueError('segment_img must be a SegmentationImage')
        if segment_img.shape != self._data.shape:
            raise ValueError('segment_img and data must have the same shape.')
        return segment_img

    def _validate_array(self, array, name, check_units=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.shape != self._data.shape:
                raise ValueError(f'error and {name} must have the same shape.')
        return array

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).
        """
        def islazyproperty(object):
            return isinstance(object, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def __getitem__(self, index):
        newcls = object.__new__(self.__class__)

        segm = copy(self._segment_img)  # TODO: add segm copy method?
        # TODO: test non-consecutive labels
        segm.keep_labels(segm.labels[index])
        newcls._segment_img = segm

        # attributes defined in __init__ (_segment_img was set above)
        init_attr = ('_data', '_error', '_mask', '_kernel', '_background',
                     '_wcs', '_data_unit', 'default_columns')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # _labels determines ordering and isscalar
        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_slices'
        # Use a numpy object array to allow for fancy and bool indices.
        # NOTE: None is appended to the list (and then removed) to keep
        # the array only on the outer level (i.e., prevents recursion).
        # Otherwise, the tuple of (y, x) slices are not preserved.
        value = np.array(getattr(self, attr) + [None],
                         dtype=object)[:-1][index]
        if not newcls.isscalar:
            value = value.tolist()
        setattr(newcls, attr, value)

        # lazy properties to keep, but not slice
        ref_attr = ('_convolved_data', '_data_mask')

        # evaluated lazyproperty objects
        keys = set(self.__dict__.keys()) & set(self._lazyproperties)
        for key in keys:
            value = self.__dict__[key]
            if key in ref_attr:  # do not slice
                newcls.__dict__[key] = value
            else:
                # do not insert attributes that are always scalar (e.g.,
                # isscalar, nlabels), i.e., not an array/list for each
                # source
                if np.isscalar(value):
                    continue

                # TODO: copy(value)?
                try:
                    val = value[index]

                    if newcls.isscalar and key.startswith('_'):
                        # keep _<attrs> as length-1 iterables
                        # NOTE: these attributes will not exactly match
                        # the values if evaluated for the first time in
                        # a scalar class (e.g., _bbox_corner_ll)
                        val = (val,)
                except TypeError:
                    # apply fancy indices (e.g., array/list or bool
                    # mask) to lists
                    val = (np.array(value + [None],
                                    dtype=object)[:-1][index]).tolist()

                newcls.__dict__[key] = val
        return newcls

    def __len__(self):
        if self.isscalar:
            raise TypeError(f'Scalar {self.__class__.__name__!r} object has '
                            'no len()')
        return self.nlabels

    @lazyproperty
    def isscalar(self):
        return self._labels.shape == ()

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    @lazyproperty
    def _null_object(self):
        """
        Return None values.

        Used for SkyCoord properties if ``wcs`` is `None`.
        """
        return np.array([None] * self.nlabels)

    @lazyproperty
    def _null_value(self):
        """
        Return np.nan values.

        Used for background properties if ``background`` is `None`.
        """
        values = np.empty(self.nlabels)
        values.fill(np.nan)
        return values

    @lazyproperty
    def _convolved_data(self):
        if self._kernel is None:
            return self._data
        return _filter_data(self._data, self._kernel, mode='constant',
                            fill_value=0.0, check_normalization=True)

    @lazyproperty
    def _data_mask(self):
        mask = ~np.isfinite(self._data)
        if self._mask is not None:
            mask |= self._mask
        return mask

    @lazyproperty
    def _cutout_segment_mask(self):
        label = self.label
        if self.isscalar:
            label = (label,)
        return [self._segment_img.data[slc] != label_
                for label_, slc in zip(label, self._slices_iter)]

    @lazyproperty
    def _cutout_total_mask(self):
        """
        Boolean mask representing the combination of ``_data_mask`` and
        ``_cutout_segment_mask``.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """
        masks = []
        for mask, slc in zip(self._cutout_segment_mask, self._slices_iter):
            masks.append(mask | self._data_mask[slc])
        return masks

    @as_scalar
    def _make_cutout(self, array, units=True, masked=False):
        cutouts = [array[slc] for slc in self._slices_iter]
        if units and self._data_unit is not None:
            cutouts = [(cutout << self._data_unit) for cutout in cutouts]
        if masked:
            return [np.ma.masked_array(cutout, mask=mask)
                    for cutout, mask in zip(cutouts, self._cutout_total_mask)]
        return cutouts

    @lazyproperty
    def _cutout_moment_data(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the input
        ``convolved_data``. The following pixels are set to zero in
        these arrays:

            * any masked pixels
            * invalid values (NaN and +/- inf)
            * negative data values - negative pixels (especially at
              large radii) can give image moments that have negative
              variances.

        These arrays are used to derive moment-based properties.
        """
        mask = ~np.isfinite(self._convolved_data) | (self._convolved_data < 0)
        if self._mask is not None:
            mask |= self._mask

        cutout = self.convdata_cutout
        if self.isscalar:
            cutout = (cutout,)

        cutouts = []
        for slc, cutout_, mask_ in zip(self._slices_iter, cutout,
                                       self._cutout_segment_mask):
            try:
                cutout = cutout_.value.copy()  # Quantity array
            except AttributeError:
                cutout = cutout_.copy()
            cutout[(mask[slc] | mask_)] = 0.
            cutouts.append(cutout)
        return cutouts

    def to_table(self, columns=None, exclude_columns=None):
        return _properties_table(self, columns=columns,
                                 exclude_columns=exclude_columns)

    @lazyproperty
    def nlabels(self):
        if self.isscalar:
            return 1
        return len(self._labels)

    @property
    @as_scalar
    def label(self):
        """
        The source label number(s) in the segmentation image
        """
        return self._labels

    @property
    @as_scalar
    def id(self):
        """
        The source identification number corresponding to the object
        label in the segmentation image.
        """
        return self.label

    @property
    @as_scalar
    def slices(self):
        """
        Slice tuples.
        """
        return self._slices

    @lazyproperty
    def _slices_iter(self):
        _slices = self.slices
        if self.isscalar:
            _slices = (_slices,)
        return _slices

    @lazyproperty
    @as_scalar
    def segm_cutout(self):
        return self._make_cutout(self._segment_img.data, units=True,
                                 masked=False)

    @lazyproperty
    @as_scalar
    def segm_cutout_ma(self):
        return self._make_cutout(self._segment_img.data, units=False,
                                 masked=True)

    @lazyproperty
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source segment.
        """
        return self._make_cutout(self._data, units=True, masked=False)

    @lazyproperty
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """
        return self._make_cutout(self._data, units=False, masked=True)

    @lazyproperty
    def convdata_cutout(self):
        return self._make_cutout(self._convolved_data, units=True,
                                 masked=False)

    @lazyproperty
    def convdata_cutout_ma(self):
        return self._make_cutout(self._convolved_data, units=False,
                                 masked=True)

    @lazyproperty
    def error_cutout(self):
        if self._error is None:
            return self._null_object
        return self._make_cutout(self._error, units=True,
                                 masked=False)

    @lazyproperty
    def error_cutout_ma(self):
        if self._error is None:
            return self._null_object
        return self._make_cutout(self._error, units=False,
                                 masked=True)

    @lazyproperty
    def background_cutout(self):
        if self._background is None:
            return self._null_object
        return self._make_cutout(self._background, units=True,
                                 masked=False)

    @lazyproperty
    def background_cutout_ma(self):
        if self._error is None:
            return self._null_object
        return self._make_cutout(self._background, units=False,
                                 masked=True)

    def _get_values(self, array):
        if self.isscalar:
            array = (array,)
        return [arr.compressed() if len(arr.compressed()) > 0 else np.nan
                for arr in array]

    @lazyproperty
    def _data_values(self):
        return self._get_values(self.data_cutout_ma)

    @lazyproperty
    def _error_values(self):
        if self._error is None:
            return self._null_value
        return self._get_values(self.error_cutout_ma)

    @lazyproperty
    def _background_values(self):
        if self._background is None:
            return self._null_value
        return self._get_values(self.background_cutout_ma)

    @lazyproperty
    @as_scalar
    def moments(self):
        """Spatial moments up to 3rd order of the source."""
        return np.array([_moments(arr, order=3) for arr in
                         self._cutout_moment_data])

    @lazyproperty
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        xcen = self.xcentroid
        ycen = self.ycentroid
        if self.isscalar:
            xcen = (xcen,)
            ycen = (ycen,)
        return np.array([_moments_central(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in
                         zip(self._cutout_moment_data, xcen, ycen)])

    @lazyproperty
    @as_scalar
    def cutout_centroid(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the centroid within the source segment.
        """
        moments = self.moments
        if self.isscalar:
            moments = moments[np.newaxis, :]
        mu_00 = moments[:, 0, 0]
        badmask = (mu_00 == 0)
        ycentroid = np.where(badmask, np.nan, moments[:, 1, 0] / mu_00)
        xcentroid = np.where(badmask, np.nan, moments[:, 0, 1] / mu_00)
        return np.transpose((ycentroid, xcentroid))

    @lazyproperty
    @as_scalar
    def centroid(self):
        """
        The ``(y, x)`` coordinate of the centroid within the source
        segment.
        """
        origin = np.transpose((self.bbox_ymin, self.bbox_xmin))
        return self.cutout_centroid + origin

    @lazyproperty
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """
        return np.transpose(self.centroid)[1]

    @lazyproperty
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """
        return np.transpose(self.centroid)[0]

    @lazyproperty
    def sky_centroid(self):
        """
        The sky coordinate of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input WCS.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(self.xcentroid, self.ycentroid)

    @lazyproperty
    def sky_centroid_icrs(self):
        """
        The sky coordinate, in the International Celestial Reference
        System (ICRS) frame, of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.
        """
        if self._wcs is None:
            return self._null_object
        return self.sky_centroid.icrs

    @lazyproperty
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices_iter]

    @lazyproperty
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel location within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel location within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    def _bbox_corner_ll(self):
        """
        Lower-left outside pixel corner location.
        """
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox_ in bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ul(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox_ in bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymax + 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_lr(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox_ in bbox:
            xypos.append((bbox_.ixmax + 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ur(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox_ in bbox:
            xypos.append((bbox_.ixmax + 0.5, bbox_.iymax + 0.5))
        return np.array(xypos)

    @lazyproperty
    @as_scalar
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ll))

    @lazyproperty
    @as_scalar
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ul))

    @lazyproperty
    @as_scalar
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_lr))

    @lazyproperty
    @as_scalar
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_object
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ur))

    @lazyproperty
    @as_scalar
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """
        values = np.array([np.min(array) for array in self._data_values])
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    @as_scalar
    def max_value(self):
        """
        The maximum pixel value of the ``data`` within the source
        segment.
        """
        values = np.array([np.max(array) for array in self._data_values])
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    @as_scalar
    def minval_cutout_index(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        data = self.data_cutout_ma
        if self.isscalar:
            data = (data,)
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmin(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    @as_scalar
    def maxval_cutout_index(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        data = self.data_cutout_ma
        if self.isscalar:
            data = (data,)
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmax(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    @as_scalar
    def minval_index(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        index = self.minval_cutout_index
        if self.isscalar:
            index = (index,)
        out = []
        for idx, slc in zip(index, self._slices_iter):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    @as_scalar
    def maxval_index(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        index = self.maxval_cutout_index
        if self.isscalar:
            index = (index,)
        out = []
        for idx, slc in zip(index, self._slices_iter):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    @as_scalar
    def minval_xindex(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        return np.transpose(self.minval_index)[1]

    @lazyproperty
    @as_scalar
    def minval_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        return np.transpose(self.minval_index)[0]

    @lazyproperty
    @as_scalar
    def maxval_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        return np.transpose(self.maxval_index)[1]

    @lazyproperty
    @as_scalar
    def maxval_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        return np.transpose(self.maxval_index)[0]
