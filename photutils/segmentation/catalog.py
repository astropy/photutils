# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by a segmentation image.
"""

from copy import deepcopy
import functools
import inspect
import warnings

from astropy.stats import SigmaClip
from astropy.table import QTable
import astropy.units as u
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from ..aperture import (BoundingBox, CircularAperture, EllipticalAperture,
                        RectangularAnnulus)
from ..background import SExtractorBackground
from ..utils._convolution import _filter_data
from ..utils._moments import _moments, _moments_central

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
                   'semimajor_sigma', 'semiminor_sigma',
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
    def __init__(self, data, segment_img, error=None, mask=None, kernel=None,
                 background=None, wcs=None, localbkg_width=None,
                 kron_params=('mask', 2.5, 0.0, 'exact', 5)):

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

        if localbkg_width is not None and localbkg_width <= 0:
            raise ValueError('localbkg_width must be > 0')
        self.localbkg_width = localbkg_width

        if kron_params[0] not in ('none', 'mask', 'mask_all', 'correct'):
            raise ValueError('Invalid value for kron_params[0]')
        self.kron_params = kron_params

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

    def _validate_array(self, array, name):
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
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def __getitem__(self, index):
        if self.isscalar:
            raise TypeError(f'A scalar {self.__class__.__name__!r} object '
                            'cannot be indexed')

        newcls = object.__new__(self.__class__)

        # attributes defined in __init__ (_segment_img was set above)
        init_attr = ('_data', '_segment_img', '_error', '_mask', '_kernel',
                     '_background', '_wcs', '_data_unit', 'localbkg_width',
                     'kron_params', 'default_columns')
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

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        fmt = [f'Sources: {len(self)}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

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
        return [self._segment_img.data[slc] != label
                for label, slc in zip(self._label_iter, self._slices_iter)]

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
            * invalid values (NaN and inf)
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
    def _label_iter(self):
        _label = self.label
        if self.isscalar:
            _label = (_label,)
        return _label

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
        input, or any non-finite ``data`` values (NaN and inf).
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

    @lazyproperty
    def _all_masked(self):
        return [np.all(mask) for mask in self._cutout_total_mask]

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
        cutout_centroid = self.cutout_centroid
        if self.isscalar:
            cutout_centroid = cutout_centroid[np.newaxis, :]
        return np.array([_moments_central(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in
                         zip(self._cutout_moment_data, cutout_centroid[:, 1],
                             cutout_centroid[:, 0])])

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

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ycentroid = moments[:, 1, 0] / moments[:, 0, 0]
            xcentroid = moments[:, 0, 1] / moments[:, 0, 0]
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
    def _xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """
        xcentroid = np.transpose(self.centroid)[1]
        if self.isscalar:
            xcentroid = (xcentroid,)
        return xcentroid

    @lazyproperty
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """
        return self._xcentroid

    @lazyproperty
    def _ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """
        ycentroid = np.transpose(self.centroid)[0]
        if self.isscalar:
            ycentroid = (ycentroid,)
        return ycentroid

    @lazyproperty
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """
        return self._ycentroid

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
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices_iter]

    @lazyproperty
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.
        """
        return self._bbox

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
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ul(self):
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymax + 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_lr(self):
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmax + 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ur(self):
        xypos = []
        for bbox_ in self._bbox:
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
        values -= self._local_background
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
        values -= self._local_background
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

    @lazyproperty
    @as_scalar
    def source_sum(self):
        """
        The sum of the unmasked ``data`` values within the source segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``source_sum``, :math:`(I_i - B_i)` is the
        ``data``, and :math:`S` are the unmasked pixels in the source
        segment.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        source_sum = np.array([np.sum(arr) for arr in self._data_values])
        source_sum -= self.area.value * self._local_background
        if self._data_unit is not None:
            source_sum <<= self._data_unit
        return source_sum

    @lazyproperty
    @as_scalar
    def source_sum_err(self):
        """
        The uncertainty of
        `~photutils.segmentation.LegacySourceProperties.source_sum`,
        propagated from the input ``error`` array.

        ``source_sum_err`` is the quadrature sum of the total errors
        over the non-masked pixels within the source segment:

        .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
                  \\sigma_{\\mathrm{tot}, i}^2}

        where :math:`\\Delta F` is ``source_sum_err``,
        :math:`\\sigma_{\\mathrm{tot, i}}` are the pixel-wise total
        errors, and :math:`S` are the non-masked pixels in the source
        segment.

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
    def background_sum(self):
        """
        The sum of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the background array.
        """
        if self._background is None:
            bkg_sum = self._null_value
        else:
            bkg_sum = np.array([np.sum(arr)
                                for arr in self._background_values])

        if self._data_unit is not None:
            bkg_sum <<= self._data_unit
        return bkg_sum

    @lazyproperty
    @as_scalar
    def background_mean(self):
        """
        The mean of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the background array.
        """
        if self._background is None:
            bkg_mean = self._null_value
        else:
            bkg_mean = np.array([np.mean(arr)
                                 for arr in self._background_values])

        if self._data_unit is not None:
            bkg_mean <<= self._data_unit
        return bkg_mean

    @lazyproperty
    @as_scalar
    def background_centroid(self):
        """
        The value of the ``background`` at the position of the source
        centroid.

        The background value at fractional position values are
        determined using bilinear interpolation.
        """
        if self._background is None:
            bkg = self._null_value
        else:
            from scipy.ndimage import map_coordinates

            xcen = self._xcentroid
            ycen = self._ycentroid
            bkg = map_coordinates(self._background, (xcen, ycen), order=1,
                                  mode='nearest')

            mask = np.isfinite(xcen) & np.isfinite(ycen)
            bkg[~mask] = np.nan

        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    @lazyproperty
    @as_scalar
    def area(self):
        """
        The total unmasked area of the source segment in units of
        pixels**2.

        Note that the source area may be smaller than its segment area
        if a mask is input to `SourceProperties` or if the ``data``
        within the segment contains invalid values (NaN and inf).
        """
        return np.array([arr.shape[0]
                         if isinstance(arr, np.ndarray) else np.nan
                         for arr in self._data_values]) << u.pix**2

    @lazyproperty
    @as_scalar
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """
        return np.sqrt(self.area / np.pi)

    @lazyproperty
    @as_scalar
    def perimeter(self):
        """
        The perimeter of the source segment, approximated as the total
        length of lines connecting the centers of the border pixels
        defined by a 4-pixel connectivity.

        If any masked pixels make holes within the source segment, then
        the perimeter around the inner hole (e.g., an annulus) will also
        contribute to the total perimeter.

        References
        ----------
        .. [1] K. Benkrid, D. Crookes, and A. Benkrid.  "Design and FPGA
               Implementation of a Perimeter Estimator".  Proceedings of
               the Irish Machine Vision and Image Processing Conference,
               pp. 51-57 (2000).
               http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
        """
        from scipy.ndimage import binary_erosion, convolve

        selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]])
        size = 34
        weights = np.zeros(size, dtype=float)
        weights[[5, 7, 15, 17, 25, 27]] = 1.
        weights[[21, 33]] = np.sqrt(2.)
        weights[[13, 23]] = (1 + np.sqrt(2.)) / 2.

        perimeter = []
        for mask in self._cutout_total_mask:
            if np.all(mask):
                perimeter.append(np.nan)
                continue

            data = ~mask
            data_eroded = binary_erosion(data, selem, border_value=0)
            border = np.logical_xor(data, data_eroded).astype(int)
            perimeter_data = convolve(border, kernel, mode='constant', cval=0)
            perimeter_hist = np.bincount(perimeter_data.ravel(),
                                         minlength=size)
            perimeter.append(perimeter_hist[0:size] @ weights)

        return np.array(perimeter) * u.pix

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
        same second-order moments as the source.
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
        while idx.size > 0:
            covar[idx, 0, 0] += delta
            covar[idx, 1, 1] += delta
            # ignore RuntimeWarning from NaN values in covar
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                covar_det = np.linalg.det(covar)
            idx = np.where(covar_det < delta2)[0]
        return covar

    @lazyproperty
    @as_scalar
    def covariance(self):
        return self._covariance * (u.pix**2)

    @lazyproperty
    @as_scalar
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.empty((self.nlabels, 2))
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
        # this matches SourceExtractor's A parameter
        return np.sqrt(eigvals[:, 1])

    @lazyproperty
    @as_scalar
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.  The angle increases in the counter-clockwise direction.
        """
        covar = self._covariance
        orient_radians = 0.5 * np.arctan2(2. * covar[:, 0, 1],
                                          (covar[:, 0, 0] - covar[:, 1, 1]))
        return orient_radians * 180. / np.pi * u.deg

    @lazyproperty
    @as_scalar
    def eccentricity(self):
        """
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math:: e = \\sqrt{1 - \\frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        semimajor_var, semiminor_var = np.transpose(self.covariance_eigvals)
        return np.sqrt(1. - (semiminor_var / semimajor_var))

    @lazyproperty
    @as_scalar
    def elongation(self):
        """
        The ratio of the lengths of the semimajor and semiminor axes:

        .. math:: \\mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_sigma / self.semiminor_sigma

    @lazyproperty
    @as_scalar
    def ellipticity(self):
        """
        ``1`` minus the ratio of the lengths of the semimajor and
        semiminor axes (or ``1`` minus the `elongation`):

        .. math:: \\mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_sigma / self.semimajor_sigma)

    @lazyproperty
    @as_scalar
    def covar_sigx2(self):
        """
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\\sigma_x^2`, in units of pixel**2.
        """
        return self._covariance[:, 0, 0] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigy2(self):
        """
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\\sigma_y^2`, in units of pixel**2.
        """
        return self._covariance[:, 1, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def covar_sigxy(self):
        """
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\\sigma_x \\sigma_y`, in units of
        pixel**2.
        """
        return self._covariance[:, 0, 1] * u.pix**2

    @lazyproperty
    @as_scalar
    def cxx(self):
        """
        `SourceExtractor`_'s CXX ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """
        return ((np.cos(self.orientation) / self.semimajor_sigma)**2
                + (np.sin(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cyy(self):
        """
        `SourceExtractor`_'s CYY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """
        return ((np.sin(self.orientation) / self.semimajor_sigma)**2
                + (np.cos(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @as_scalar
    def cxy(self):
        """
        `SourceExtractor`_'s CXY ellipse parameter in units of
        pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths). `SourceExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """
        return (2. * np.cos(self.orientation) * np.sin(self.orientation)
                * ((1. / self.semimajor_sigma**2)
                   - (1. / self.semiminor_sigma**2)))

    @lazyproperty
    @as_scalar
    def gini(self):
        """
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        as:

        .. math::
            G = \\frac{1}{\\left | \\bar{x} \\right | n (n - 1)}
            \\sum^{n}_{i} (2i - n - 1) \\left | x_i \\right |

        where :math:`\\bar{x}` is the mean over all pixel values
        :math:`x_i`.

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

    @lazyproperty
    @as_scalar
    def local_background_aperture(self):
        """
        The rectangular annulus aperture used to estimate the local
        background.
        """
        if self.localbkg_width is None:
            return self._null_object

        aperture = []
        for bbox_ in self._bbox:
            xpos = 0.5 * (bbox_.ixmin + bbox_.ixmax - 1)
            ypos = 0.5 * (bbox_.iymin + bbox_.iymax - 1)
            scale = 1.5
            width_bbox = bbox_.ixmax - bbox_.ixmin
            width_in = width_bbox * scale
            width_out = width_in + 2 * self.localbkg_width
            height_bbox = bbox_.iymax - bbox_.iymin
            height_in = height_bbox * scale
            height_out = height_in + 2 * self.localbkg_width
            aperture.append(RectangularAnnulus((xpos, ypos), width_in,
                                               width_out, height_out,
                                               height_in, theta=0.))
        return aperture

    @lazyproperty
    def _local_background(self):
        """
        The local background value estimated using a rectangular annulus
        aperture around the source.

        This property is always an `~numpy.ndarray` without units.
        """
        if self.localbkg_width is None:
            bkg = np.zeros(self.nlabels)
        else:
            mask = self._data_mask | self._segment_img.data.astype(bool)
            sigma_clip = SigmaClip(sigma=3.0, cenfunc='median', maxiters=20)
            bkg_func = SExtractorBackground(sigma_clip)
            bkg_aper = self.local_background_aperture
            if self.isscalar:
                bkg_aper = (bkg_aper,)

            bkg = []
            for aperture in bkg_aper:
                aperture_mask = aperture.to_mask(method='center')
                values = aperture_mask.get_values(self._data, mask=mask)
                if len(values) < 10:  # not enough unmasked pixels
                    bkg.append(0.)
                    continue
                bkg.append(bkg_func(values))
            bkg = np.array(bkg)

        bkg[self._all_masked] = np.nan
        return bkg

    @lazyproperty
    @as_scalar
    def local_background(self):
        """
        The local background value estimated using a rectangular annulus
        aperture around the source.
        """
        bkg = self._local_background
        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    def _mask_neighbors(self, label, aperture_mask, method='none'):
        if method == 'none':
            return None

        slc_lg, slc_sm = aperture_mask.get_overlap_slices(self._data.shape)
        segment_img = np.copy(self._segment_img.data[slc_lg])

        # mask all pixels outside of the source segment
        if method in ('mask_all', ):
            segm_mask = (segment_img != label)

        # mask pixels *only* in neighboring segments (not including
        # background pixels)
        if method in ('mask', 'correct'):
            segm_mask = np.logical_and(segment_img != label,
                                       segment_img != 0)

        return segm_mask

    def _make_circular_apertures(self, radius):
        if radius <= 0:
            return self._null_object
        apertures = []
        for (xcen, ycen) in zip(self._xcentroid, self._ycentroid):
            if np.any(~np.isfinite((xcen, ycen))):
                apertures.append(None)
                continue
            apertures.append(CircularAperture((xcen, ycen), r=radius))
        return apertures

    def _make_elliptical_apertures(self, scale=6.):
        """
        Return a list of elliptical apertures based on the scaled
        isophotal shape of the sources.

        Parameters
        ----------
        scale : float or `~numpy.ndarray`, optional
            The scale factor to apply to the ellipse major and minor
            axes. The default value of 6.0 is roughly two times the
            isophotal extent of the source. A `~numpy.ndarray` input
            must be a 1D array of length ``nlabels``.
        """
        xcen = self._xcentroid
        ycen = self._ycentroid
        major_size = self.semimajor_sigma.value * scale
        minor_size = self.semiminor_sigma.value * scale
        theta = self.orientation.to(u.radian).value
        if self.isscalar:
            major_size = (major_size,)
            minor_size = (minor_size,)
            theta = (theta,)

        aperture = []
        for xcen_, ycen_, major_, minor_, theta_ in zip(xcen, ycen,
                                                        major_size,
                                                        minor_size,
                                                        theta):
            values = (xcen_, ycen_, major_, minor_, theta_)
            if np.any(~np.isfinite(values)):
                aperture.append(None)
                continue
            aperture.append(EllipticalAperture((xcen_, ycen_), major_, minor_,
                                               theta=theta_))
        return aperture

    def _prepare_kron_data(self, label, xcentroid, ycentroid, aperture_mask,
                           background=0.):
        """
        Make cutouts from data and error, applying various types of
        masking and/or pixel corrections for a single ``aperture_mask``.
        """
        slc_lg, slc_sm = aperture_mask.get_overlap_slices(self._data.shape)
        data = np.copy(self._data[slc_lg])
        mask = self._data_mask[slc_lg]
        segm_mask = self._mask_neighbors(label, aperture_mask,
                                         method=self.kron_params[0])
        if segm_mask is not None:
            mask |= segm_mask

        data -= background  # subtract local background before masking
        data[mask] = 0.

        if self._error is not None:
            error = np.copy(self._error[slc_lg])
            error[mask] = 0.
        else:
            error = None

        # Correct masked pixels in neighboring segments.  Masked pixels
        # are replaced with pixels on the opposite side of the source.
        if self.kron_params[0] == 'correct':
            from ..utils.interpolation import _mask_to_mirrored_num
            xypos = (xcentroid - max(0, aperture_mask.bbox.ixmin),
                     ycentroid - max(0, aperture_mask.bbox.iymin))
            data = _mask_to_mirrored_num(data, segm_mask, xypos)
            if self._error is not None:
                error = _mask_to_mirrored_num(error, segm_mask, xypos)

        return data, error

    def _correct_kron_mask(self, data, mask, xycenter, error=None):
        # Correct masked pixels in neighboring segments.  Masked pixels
        # are replaced with pixels on the opposite side of the source.
        from ..utils.interpolation import _mask_to_mirrored_num
        data = _mask_to_mirrored_num(data, mask, xycenter)
        if error is not None:
            error = _mask_to_mirrored_num(error, mask, xycenter)
        return data, error

    @lazyproperty
    @as_scalar
    def kron_radius(self):
        """
        The unscaled first-moment Kron radius.

        If the source is completely masked, then ``np.nan`` will be
        returned for both the Kron radius and Kron flux.

        The unscaled first-moment Kron radius is given by:

        .. math::
            k_r = \\frac{\\sum_{i \\in A} \\ r_i I_i}{\\sum_{i \\in A} I_i}

        where the sum is over all pixels in an elliptical aperture whose
        axes are defined by six times the ``semimajor_axis_sigma`` and
        ``semiminor_axis_sigma`` at the calculated ``orientation``
        (all properties derived from the central image moments of the
        source). :math:`r_i` is the elliptical "radius" to the pixel
        given by:

        .. math::
            r_i^2 = cxx(x_i - \\bar{x})^2 +
                cxx \\ cyy (x_i - \\bar{x})(y_i - \\bar{y}) +
                cyy(y_i - \\bar{y})^2

        where :math:`\\bar{x}` and :math:`\\bar{y}` represent the source
        centroid.

        If either the numerator or denominator <= 0, then ``np.nan``
        will be returned. In this case, the Kron aperture will
        be defined as a circular aperture with a radius equal to
        ``kron_params[2]``. If ``kron_params[2] <= 0``, then the Kron
        aperture will be `None` and the Kron flux will be ``np.nan``.
        """
        labels = self._label_iter
        apertures = self._make_elliptical_apertures(scale=6.0)
        xcen = self._xcentroid
        ycen = self._ycentroid
        cxx = self.cxx.value
        cxy = self.cxy.value
        cyy = self.cyy.value
        if self.isscalar:
            cxx = (cxx,)
            cxy = (cxy,)
            cyy = (cyy,)

        kron_radius = []
        for (label, aperture, xcen_, ycen_, cxx_, cxy_, cyy_) in zip(
                labels, apertures, xcen, ycen, cxx, cxy, cyy):
            if aperture is None:
                kron_radius.append(np.nan)
                continue

            # prepare cutouts of the data based on the aperture size
            method = 'center'  # need whole pixels to compute Kron radius
            aperture_mask = aperture.to_mask(method=method)
            slc_lg, slc_sm = aperture_mask.get_overlap_slices(self._data.shape)

            # prepare kron mask (method?)
            mask = self._data_mask[slc_lg]
            segm_mask = self._mask_neighbors(label, aperture_mask,
                                             method=self.kron_params[0])
            if segm_mask is not None:
                mask |= segm_mask

            data = self._data[slc_lg]

            xycen = (xcen_ - max(0, aperture_mask.bbox.ixmin),
                     ycen_ - max(0, aperture_mask.bbox.iymin))

            if self.kron_params[0] == 'correct':
                data, _ = self._correct_kron_mask(data, mask, xycen,
                                                  error=None)

            x = np.arange(data.shape[1]) - xycen[0]
            y = np.arange(data.shape[0]) - xycen[1]
            xx, yy = np.meshgrid(x, y)
            rr = np.sqrt(cxx_ * xx**2 + cxy_ * xx * yy + cyy_ * yy**2)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels
            flux_numer = np.sum((aperture_weights * data * rr)[pixel_mask])
            flux_denom = np.sum((aperture_weights * data)[pixel_mask])

            if flux_numer <= 0 or flux_denom <= 0:
                kron_radius.append(np.nan)
                continue

            kron_radius.append(flux_numer / flux_denom)

        kron_radius = np.array(kron_radius) * u.pix
        return kron_radius

    @lazyproperty
    @as_scalar
    def kron_aperture(self):
        """
        The Kron aperture.

        If ``kron_radius * np.sqrt(semimajor_sigma * semiminor__sigma) <
        kron_params[2]`` then a circular aperture with a radius equal to
        ``kron_params[2]`` will be returned. If ``kron_params[2] <= 0``,
        then the Kron aperture will be `None`.

        If ``kron_radius = np.nan`` then a circular aperture with a
        radius equal to ``kron_params[2]`` will be returned if the
        source is not completely masked, otherwise `None` will be
        returned.

        Note that if the Kron aperture is `None`, the Kron flux will be
        ``np.nan``.
        """
        scale = self.kron_radius.value * self.kron_params[1]
        kron_aperture = self._make_elliptical_apertures(scale=scale)
        kron_radius = self.kron_radius.value

        # check for minimum Kron radius
        major_sigma = self.semimajor_sigma.value
        minor_sigma = self.semiminor_sigma.value
        circ_radius = kron_radius * np.sqrt(major_sigma * minor_sigma)
        min_radius = self.kron_params[2]
        mask = np.isnan(kron_radius) | (circ_radius < min_radius)
        idx = mask.nonzero()[0]
        if idx.size > 0:
            circ_aperture = self._make_circular_apertures(self.kron_params[2])
            for i in idx:
                kron_aperture[i] = circ_aperture[i]

        return kron_aperture

    @lazyproperty
    def _kron_flux_fluxerr(self):
        """
        The flux and flux error in the Kron aperture.

        If the Kron aperture is `None`, then ``np.nan`` will be returned.
        """
        kron_aperture = deepcopy(self.kron_aperture)
        if self.isscalar:
            kron_aperture = (kron_aperture,)

        kron_flux = []
        kron_fluxerr = []
        for label, xcen, ycen, aperture, bkg in zip(self._label_iter,
                                                    self._xcentroid,
                                                    self._ycentroid,
                                                    kron_aperture,
                                                    self._local_background):
            if aperture is None:
                kron_flux.append(np.nan)
                kron_fluxerr.append(np.nan)
                continue

            xycen = (xcen_ - max(0, aperture_mask.bbox.ixmin),
                     ycen_ - max(0, aperture_mask.bbox.iymin))

            aperture_mask = aperture.to_mask()
            data, error = self._prepare_kron_data(label, xcen, ycen,
                                                  aperture_mask, bkg)
            aperture.positions -= (aperture_mask.bbox.ixmin,
                                   aperture_mask.bbox.iymin)

            method = self.kron_params[3]
            subpixels = self.kron_params[4]
            flux, fluxerr = aperture.do_photometry(data, error=error,
                                                   method=method,
                                                   subpixels=subpixels)

            kron_flux.append(flux[0])
            if error is None:
                kron_fluxerr.append(np.nan)
            else:
                kron_fluxerr.append(fluxerr[0])

        return np.transpose((kron_flux, kron_fluxerr))

    @lazyproperty
    @as_scalar
    def kron_flux(self):
        """
        The flux in the Kron aperture.

        If the Kron aperture is `None`, then ``np.nan`` will be returned.
        """
        kron_flux = self._kron_flux_fluxerr[:, 0]
        if self._data_unit is not None:
            kron_flux <<= self._data_unit
        return kron_flux

    @lazyproperty
    @as_scalar
    def kron_fluxerr(self):
        """
        The flux error in the Kron aperture.

        If the Kron aperture is `None`, then ``np.nan`` will be returned.
        """
        kron_fluxerr = self._kron_flux_fluxerr[:, 1]
        if self._data_unit is not None:
            kron_fluxerr <<= self._data_unit
        return kron_fluxerr
