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
from astropy.utils.decorators import deprecated
import numpy as np

from .core import SegmentationImage
from ..aperture import (BoundingBox, CircularAperture, EllipticalAperture,
                        RectangularAnnulus)
from ..background import SExtractorBackground
from ..utils._convolution import _filter_data
from ..utils._misc import _get_meta
from ..utils._moments import _moments, _moments_central

__all__ = ['SourceCatalog']
__doctest_requires__ = {('SourceCatalog', 'SourceCatalog.*'): ['scipy']}


# default table columns for `to_table()` output
DEFAULT_COLUMNS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity', 'min_value', 'max_value',
                   'local_background', 'segment_flux', 'segment_fluxerr',
                   'kron_flux', 'kron_fluxerr']


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


class SourceCatalog:
    r"""
    Class to create a catalog of photometry and morphological properties
    for sources defined by a segmentation image.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array from which to calculate the source photometry and
        properties. If ``kernel`` is input, then a convolved version
        of ``data`` will be used instead of ``data`` to calculate the
        source centroid and morphological properties. Source photometry
        is always measured from ``data``. For accurate source properties
        and photometry, ``data`` should be background-subtracted.
        Non-finite ``data`` values (NaN and inf) are automatically
        masked.

    segment_img : `~photutils.segmentation.SegmentationImage`
        A `~photutils.segmentation.SegmentationImage` object defining
        the sources.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data``
        array. ``error`` is assumed to include *all* sources of
        error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) . ``error`` must have
        the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vice versa) with identical
        units. Non-finite ``error`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array. Such pixels can
        be masked using the ``mask`` keyword. See the Notes section
        below for details on the error propagation.

    mask : 2D `~numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations. Non-finite
        values (NaN and inf) in the input ``data`` are automatically
        masked.

    kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the data prior to
        calculating the source centroid and morphological parameters.
        The kernel should be the same one used in defining the
        source segments, i.e., the detection image (e.g., see
        :func:`~photutils.segmentation.detect_sources`). If `None`, then
        the unfiltered ``data`` will be used instead.

    background : float, 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``. ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``. If ``data``
        is a `~astropy.units.Quantity` array then ``background`` must
        be a `~astropy.units.Quantity` array (and vice versa) with
        identical units. Inputing the ``background`` merely allows
        for its properties to be measured within each source segment.
        The input ``background`` does *not* get subtracted from the
        input ``data``, which should already be background-subtracted.
        Non-finite ``background`` values (NaN and inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array. Such pixels can
        be masked using the ``mask`` keyword.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If `None`, then all
        sky-based properties will be set to `None`.

    localbkg_width : int, optional
        The width of the rectangular annulus used to compute a
        local background around each source. If 0.0, then no local
        background subtraction is performed. The local background
        affects the ``min_value``, ``max_value``, ``segment_flux``, and
        ``kron_flux`` properties. It does not affect the moment-based
        morphological properties of the source.

    apermask_method : {'correct', 'mask', 'none'}, optional
        The method used to handle neighboring sources when performing
        aperture photometry (e.g., circular apertures or elliptical Kron
        apertures).  This parameter also affects the Kron radius.

          * 'correct':  replace pixels assigned to neighboring sources
                        by replacing them with pixels on the opposite
                        side of the source center (equivalent to
                        MASK_TYPE=CORRECT in SourceExtractor).
          * 'mask':  mask pixels assigned to neighboring sources
                     (equivalent to MASK_TYPE=BLANK in SourceExtractor).
          * 'none':  do not mask any pixels (equivalent to
                     MASK_TYPE=NONE in SourceExtractor).

    kron_params : list of 2 floats, optional
        A list of two parameters used to determine how the Kron
        radius and flux are calculated. The first item is the scaling
        parameter of the Kron radius and the second item represents
        the minimum circular radius. If the Kron radius times
        sqrt(``semimajor_sigma`` * ``semiminor_sigma``) is less than
        than this radius, then the Kron flux will be measured in a
        circle with this minimum radius.

    detection_cat : `SourceCatalog`, optional
        A `SourceCatalog` object for the detection image. The source
        labels in ``detection_cat`` must correspond to the labels in the
        input ``segment_img``. If input, this detection catalog will
        be used to define the source centroids for all aperture-based
        photometry (e.g., local background aperture, circular aperture,
        Kron aperture). It will also be used to define the object
        elliptical shape parameters when calculating the Kron radius.
        This keyword affects the local-background value, circular
        aperture photometry, Kron radius, and Kron photometry.

    Notes
    -----
    ``data`` should be background-subtracted for accurate source
    photometry and properties. The previously-subtracted background can
    be passed into this class to calculate properties of the background
    for each source.

    `SourceExtractor`_'s centroid and morphological parameters are
    always calculated from a filtered "detection" image, i.e., the
    image used to define the segmentation image. The usual downside of
    the filtering is the sources will be made more circular than they
    actually are. If you wish to reproduce `SourceExtractor`_ centroid
    and morphology results, then input a ``kernel``. If ``kernel`` is
    `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values within the source segment are set to zero
    when calculating morphological properties based on image moments.
    Negative values could occur, for example, if the segmentation
    image was defined from a different image (e.g., different
    bandpass) or if the background was oversubtracted. However,
    `~photutils.segmentation.SourceCatalog.segment_flux` always includes
    the contribution of negative ``data`` values.

    The input ``error`` array is assumed to include *all* sources
    of error, including the Poisson error of the sources.
    `~photutils.segmentation.SourceCatalog.segment_fluxerr` is simply
    the quadrature sum of the pixel-wise total errors over the
    unmasked pixels within the source segment:

    .. math:: \Delta F = \sqrt{\sum_{i \in S}
              \sigma_{\mathrm{tot}, i}^2}

    where :math:`\Delta F` is
    `~photutils.segmentation.SourceCatalog.segment_fluxerr`,
    :math:`S` are the unmasked pixels in the source segment, and
    :math:`\sigma_{\mathrm{tot}, i}` is the input ``error`` array.

    Custom errors for source segments can be calculated using
    the `~photutils.segmentation.SourceCatalog.error_ma` and
    `~photutils.segmentation.SourceCatalog.background_ma` properties,
    which are 2D `~numpy.ma.MaskedArray` cutout versions of the input
    ``error`` and ``background`` arrays. The mask is `True` for pixels
    outside of the source segment, masked pixels from the ``mask``
    input, or any non-finite ``data`` values (NaN and inf).

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """

    def __init__(self, data, segment_img, *, error=None, mask=None,
                 kernel=None, background=None, wcs=None, localbkg_width=0,
                 apermask_method='correct', kron_params=(2.5, 1.0),
                 detection_cat=None):

        self._data_unit = None
        data, error, background = self._process_quantities(data, error,
                                                           background)
        self._data = self._validate_array(data, 'data', shape=False)
        self._segment_img = self._validate_segment_img(segment_img)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._kernel = kernel
        self._background = self._validate_array(background, 'background')
        self._wcs = wcs

        self._convolved_data = self._convolve_data()
        self._data_mask = self._make_data_mask()
        self._localbkg_width = self._validate_localbkg_width(localbkg_width)
        self._apermask_method = self._validate_apermask_method(apermask_method)
        self._kron_params = self._validate_kron_params(kron_params)

        # needed for ordering and isscalar
        self._labels = self._segment_img.labels
        self._slices = self._segment_img.slices
        self.default_columns = DEFAULT_COLUMNS

        self._extra_properties = []

        if detection_cat is not None:
            if not isinstance(detection_cat, SourceCatalog):
                raise TypeError('detection_cat must be a SourceCatalog '
                                'instance')
            if not np.array_equal(detection_cat.labels, self.labels):
                raise ValueError('detection_cat must have same source labels '
                                 'as the input segment_img')
        self._detection_cat = detection_cat

        self.meta = _get_meta()

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
            raise TypeError('segment_img must be a SegmentationImage')
        if segment_img.shape != self._data.shape:
            raise ValueError('segment_img and data must have the same shape.')
        return segment_img

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

    @staticmethod
    def _validate_localbkg_width(localbkg_width):
        if localbkg_width < 0:
            raise ValueError('localbkg_width must be >= 0')
        localbkg_width_int = int(localbkg_width)
        if localbkg_width_int != localbkg_width:
            raise ValueError('localbkg_width must be an integer')
        return localbkg_width_int

    @staticmethod
    def _validate_apermask_method(apermask_method):
        if apermask_method not in ('none', 'mask', 'correct'):
            raise ValueError('Invalid apermask_method value')
        return apermask_method

    @staticmethod
    def _validate_kron_params(kron_params):
        kron_params = np.atleast_1d(kron_params)
        if len(kron_params) != 2:
            raise ValueError('kron_params must have 2 elements')
        if kron_params[0] <= 0:
            raise ValueError('kron_params[0] must be > 0')
        if kron_params[1] <= 0:
            raise ValueError('kron_params[1] must be > 0')
        return kron_params

    @property
    def _properties(self):
        """
        A list of all class properties, include lazyproperties (even in
        superclasses).
        """
        def isproperty(obj):
            return isinstance(obj, property)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=isproperty)]

    @property
    def properties(self):
        """
        A list of built-in source properties.
        """
        lazyproperties = [name for name in self._lazyproperties if not
                          name.startswith('_')]
        lazyproperties.remove('isscalar')
        lazyproperties.remove('nlabels')
        lazyproperties.extend(['label', 'labels', 'slices'])
        lazyproperties.sort()
        return lazyproperties

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
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

        # attributes defined in __init__ that are copied directly to the
        # new class
        init_attr = ('_data', '_segment_img', '_error', '_mask', '_kernel',
                     '_background', '_wcs', '_data_unit', '_convolved_data',
                     '_data_mask', '_localbkg_width', '_apermask_method',
                     '_kron_params', 'default_columns', '_extra_properties',
                     'meta')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # _labels determines ordering and isscalar
        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        # need to slice detection_cat, if input
        attr = '_detection_cat'
        if getattr(self, attr) is None:
            setattr(newcls, attr, None)
        else:
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

        # evaluated lazyproperty objects and extra properties
        keys = (set(self.__dict__.keys())
                & (set(self._lazyproperties) | set(self._extra_properties)))
        for key in keys:
            value = self.__dict__[key]

            # do not insert attributes that are always scalar (e.g.,
            # isscalar, nlabels), i.e., not an array/list for each
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
            fmt = [f'Length: {self.nlabels}',
                   f'labels: {self.labels}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            raise TypeError(f'Scalar {self.__class__.__name__!r} object has '
                            'no len()')
        return self.nlabels

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self._labels.shape == ()

    @staticmethod
    def _has_len(value):
        if isinstance(value, str):
            return False
        try:
            # NOTE: cannot just check for __len__ attribute, because
            # it could exist, but raise an Exception for scalar objects
            len(value)
        except TypeError:
            return False
        return True

    def copy(self):
        """
        Return a deep copy of this SourceCatalog.
        """
        return deepcopy(self)

    @property
    def extra_properties(self):
        return self._extra_properties

    def add_extra_property(self, name, value, overwrite=False):
        """
        Add extra properties as attributes.

        For example, this property ``name`` can then be included in the
        `to_table` ``columns`` keyword list to output the results in the
        table.

        The complete list of user-defined extra properties is stored in
        the ``extra_properties`` attribute.

        Parameters
        ----------
        name : str
            The name of property. The name must not conflict with any of
            the built-in property names or attributes.

        value : array-like or float
            The value to assign.

        overwrite : bool, option
            If `True`, will overwrite the existing property ``name``.
        """
        internal_attributes = ((set(self.__dict__.keys())
                               | set(self._properties))
                               - set(self.extra_properties))
        if name in internal_attributes:
            raise ValueError(f'{name} cannot be set because it is a '
                             'built-in attribute')

        if not overwrite:
            if hasattr(self, name):
                raise ValueError(f'{name} already exists as an attribute. '
                                 'Set overwrite=True to overwrite an existing '
                                 'attribute.')
            if name in self._extra_properties:
                raise ValueError(f'{name} already exists in the '
                                 '"extra_properties" attribute list.')

        property_error = False
        if self.isscalar:
            # this allows fluxfrac_radius to add len-1 array values for
            # scalar self
            if self._has_len(value) and len(value) == 1:
                value = value[0]

            if hasattr(value, 'isscalar'):
                # e.g., Quantity, SkyCoord, Time
                if not value.isscalar:
                    property_error = True
            else:
                if not np.isscalar(value):
                    property_error = True
        else:
            if not self._has_len(value) or len(value) != self.nlabels:
                property_error = True
        if property_error:
            raise ValueError('value must have the same number of elements as '
                             'the catalog in order to add it as an extra '
                             'property.')

        setattr(self, name, value)
        if not overwrite:
            self._extra_properties.append(name)

    def remove_extra_property(self, name):
        """
        Remove a user-defined extra property.

        The property must have been defined using `add_extra_property`.
        The complete list of user-defined extra properties is stored in
        the ``extra_properties`` attribute.

        Parameters
        ----------
        name : str
            The name of the property to remove.
        """
        self.remove_extra_properties(name)

    def remove_extra_properties(self, names):
        """
        Remove user-defined extra properties.

        The properties must have been defined using
        `add_extra_property`. The complete list of user-defined extra
        properties is stored in the ``extra_properties`` attribute.

        Parameters
        ----------
        name : list of str
            The names of the properties to remove.
        """
        names = np.atleast_1d(names)
        for name in names:
            if name in self._extra_properties:
                delattr(self, name)
                self._extra_properties.remove(name)
            else:
                raise ValueError(f'{name} is not a defined extra property.')

    def rename_extra_property(self, name, new_name):
        """
        Rename an extra property.

        The renamed property will remain at the same index in the
        ``extra_properties`` list.

        Parameters
        ----------
        name : str
            The old attribute name.

        new_name : str
            The new attribute name.
        """
        self.add_extra_property(new_name, getattr(self, name))
        idx = self.extra_properties.index(name)
        self.remove_extra_property(name)
        # preserve the order of self.extra_properties
        self.extra_properties.remove(new_name)
        self.extra_properties.insert(idx, new_name)

    def _convolve_data(self):
        """
        Convolve the input data with the input kernel.
        """
        if self._kernel is None:
            return self._data
        return _filter_data(self._data, self._kernel, mode='constant',
                            fill_value=0.0, check_normalization=True)

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

        For example, this is used for SkyCoord properties if ``wcs`` is
        `None`.
        """
        return np.array([None] * self.nlabels)

    @lazyproperty
    def _null_value(self):
        """
        Return np.nan values.

        For example, this is used for background properties if
        ``background`` is `None`.
        """
        values = np.empty(self.nlabels)
        values.fill(np.nan)
        return values

    @lazyproperty
    def _cutout_segment_mask(self):
        """
        Boolean mask for source segment.

        The mask is `True` for all pixels (background and from other
        source segments) outside of the source segment.
        """
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
        """
        Make cutouts from in the input array using the source minimal
        bounding box.

        Masks and units are optionally applied.
        """
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
        A list of 2D `~numpy.ndarray` cutouts from the (convolved) data
        The following pixels are set to zero in these arrays:

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

        cutout = self.convdata
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

    def get_label(self, label):
        """
        Return a new `SourceCatalog` object for the input ``label``
        only.

        Parameters
        ----------
        label : int
            The source label.

        Returns
        -------
        cat : `SourceCatalog`
            A new `SourceCatalog` object containing only the source with
            the input ``label``.
        """
        return self.get_labels(label)

    def get_labels(self, labels):
        """
        Return a new `SourceCatalog` object for the input ``labels``
        only.

        Parameters
        ----------
        labels : list, tuple, or `~numpy.ndarray` of int
            The source label(s).

        Returns
        -------
        cat : `SourceCatalog`
            A new `SourceCatalog` object containing only the sources with
            the input ``labels``.
        """
        idx = np.searchsorted(self.label, labels)
        return self[idx]

    def to_table(self, columns=None):
        """
        Create a `~astropy.table.QTable` of source properties.

        Parameters
        ----------
        columns : str, list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are any of
            the `SourceCatalog` properties or custom properties added
            using `add_extra_property`. If ``columns`` is `None`, then a
            default list of scalar-valued properties (as defined by the
            ``default_columns`` attribute) will be used.

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
    def nlabels(self):
        """
        The number of source labels.
        """
        if self.isscalar:
            return 1
        return len(self._labels)

    @property
    @as_scalar
    def label(self):
        """
        The source label number(s).

        This label number corresponds to the assigned pixel value in the
        `~photutils.segmentation.SegmentationImage`.
        """
        return self._labels

    @property
    def labels(self):
        """
        The source label number(s), always as an iterable
        `~numpy.ndarray`.

        This label number corresponds to the assigned pixel value in the
        `~photutils.segmentation.SegmentationImage`.
        """
        return self._label_iter

    @property
    def _label_iter(self):
        """
        The source label, always as a iterable.
        """
        _label = self.label
        if self.isscalar:
            _label = np.array((_label,))
        return _label

    @property
    @as_scalar
    def slices(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source.
        """
        return self._slices

    @lazyproperty
    def _slices_iter(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source, always as an iterable.
        """
        _slices = self.slices
        if self.isscalar:
            _slices = (_slices,)
        return _slices

    @lazyproperty
    @as_scalar
    def segment(self):
        """
        A 2D `~numpy.ndarray` cutout of the segmentation image using the
        minimal bounding box of the source.
        """
        return self._make_cutout(self._segment_img.data, units=False,
                                 masked=False)

    @lazyproperty
    @as_scalar
    def segment_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout of the segmentation image
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).
        """
        return self._make_cutout(self._segment_img.data, units=False,
                                 masked=True)

    @lazyproperty
    def data(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source.
        """
        return self._make_cutout(self._data, units=True, masked=False)

    @lazyproperty
    def data_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the data using the
        minimal bounding box of the source.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).
        """
        return self._make_cutout(self._data, units=False, masked=True)

    @lazyproperty
    def convdata(self):
        """
        A 2D `~numpy.ndarray` cutout from the convolved data using the
        minimal bounding box of the source.
        """
        return self._make_cutout(self._convolved_data, units=True,
                                 masked=False)

    @lazyproperty
    def convdata_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the convolved data
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).
        """
        return self._make_cutout(self._convolved_data, units=False,
                                 masked=True)

    @lazyproperty
    @as_scalar
    def error(self):
        """
        A 2D `~numpy.ndarray` cutout from the error array using the
        minimal bounding box of the source.
        """
        if self._error is None:
            return self._null_object
        return self._make_cutout(self._error, units=True,
                                 masked=False)

    @lazyproperty
    @as_scalar
    def error_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the error array using
        the minimal bounding box of the source.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).
        """
        if self._error is None:
            return self._null_object
        return self._make_cutout(self._error, units=False,
                                 masked=True)

    @lazyproperty
    @as_scalar
    def background(self):
        """
        A 2D `~numpy.ndarray` cutout from the background array using the
        minimal bounding box of the source.
        """
        if self._background is None:
            return self._null_object
        return self._make_cutout(self._background, units=True,
                                 masked=False)

    @lazyproperty
    @as_scalar
    def background_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the background array.
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).
        """
        if self._background is None:
            return self._null_object
        return self._make_cutout(self._background, units=False,
                                 masked=True)

    @lazyproperty
    def _all_masked(self):
        """
        True if all pixels over the source segment are masked.
        """
        return np.array([np.all(mask) for mask in self._cutout_total_mask])

    def _get_values(self, array):
        """
        Get a 1D array of unmasked values from the input array within
        the source segment.

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
        return self._get_values(self.data_ma)

    @lazyproperty
    def _error_values(self):
        """
        A 1D array of unmasked error values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.error_ma)

    @lazyproperty
    def _background_values(self):
        """
        A 1D array of unmasked background values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.background_ma)

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
        return np.transpose((xcentroid, ycentroid))

    @lazyproperty
    @as_scalar
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid within the source
        segment.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @lazyproperty
    def _xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment,
        always as an iterable.
        """
        xcentroid = np.transpose(self.centroid)[0]
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
        The ``y`` coordinate of the centroid within the source segment,
        always as an iterable.
        """
        ycentroid = np.transpose(self.centroid)[1]
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
    @as_scalar
    def sky_centroid(self):
        """
        The sky coordinate of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

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
        The sky coordinate in the International Celestial Reference
        System (ICRS) frame of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        `None` if ``wcs`` is not input.
        """
        if self._wcs is None:
            return self._null_object
        return self.sky_centroid.icrs

    @lazyproperty
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment, always as an iterable.
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
        The minimum ``x`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices_iter])

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    def _bbox_corner_ll(self):
        """
        Lower-left *outside* pixel corner location (not index).
        """
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ul(self):
        """
        Upper-left *outside* pixel corner location (not index).
        """
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymax + 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_lr(self):
        """
        Lower-right *outside* pixel corner location (not index).
        """
        xypos = []
        for bbox_ in self._bbox:
            xypos.append((bbox_.ixmax + 0.5, bbox_.iymin - 0.5))
        return np.array(xypos)

    @lazyproperty
    def _bbox_corner_ur(self):
        """
        Upper-right *outside* pixel corner location (not index).
        """
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
        their entirety, thus the vertices are at the pixel *corners*,
        not their centers.

        `None` if ``wcs`` is not input.
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
        their entirety, thus the vertices are at the pixel *corners*,
        not their centers.

        `None` if ``wcs`` is not input.
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
        their entirety, thus the vertices are at the pixel *corners*,
        not their centers.

        `None` if ``wcs`` is not input.
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
        their entirety, thus the vertices are at the pixel *corners*,
        not their centers.

        `None` if ``wcs`` is not input.
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
    def cutout_minval_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        data = self.data_ma
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
    def cutout_maxval_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        data = self.data_ma
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
        first occurrence is returned.
        """
        index = self.cutout_minval_index
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
        first occurrence is returned.
        """
        index = self.cutout_maxval_index
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
        first occurrence is returned.
        """
        return np.transpose(self.minval_index)[1]

    @lazyproperty
    @as_scalar
    def minval_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        return np.transpose(self.minval_index)[0]

    @lazyproperty
    @as_scalar
    def maxval_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        return np.transpose(self.maxval_index)[1]

    @lazyproperty
    @as_scalar
    def maxval_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        return np.transpose(self.maxval_index)[0]

    @lazyproperty
    @as_scalar
    def segment_flux(self):
        r"""
        The sum of the unmasked ``data`` values within the source segment.

        .. math:: F = \sum_{i \in S} I_i

        where :math:`F` is ``segment_flux``, :math:`I_i` is the
        background-subtracted ``data``, and :math:`S` are the unmasked
        pixels in the source segment.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        localbkg = self._local_background
        if self.isscalar:
            localbkg = localbkg[0]
        source_sum = np.array([np.sum(arr) for arr in self._data_values])
        source_sum -= self.area.value * localbkg
        if self._data_unit is not None:
            source_sum <<= self._data_unit
        return source_sum

    @lazyproperty
    @as_scalar
    def segment_fluxerr(self):
        r"""
        The uncertainty of `segment_flux` , propagated from the input
        ``error`` array.

        ``segment_fluxerr`` is the quadrature sum of the total errors
        over the unmasked pixels within the source segment:

        .. math:: \Delta F = \sqrt{\sum_{i \in S}
                  \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `segment_flux`,
        :math:`\sigma_{\mathrm{tot, i}}` are the pixel-wise total
        errors (``error``), and :math:`S` are the unmasked pixels in the
        source segment.

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
        if a mask is input to `SourceCatalog` or if the ``data``
        within the segment contains invalid values (NaN and inf).
        """
        areas = np.array([arr.size for arr in self._data_values]).astype(float)
        areas[self._all_masked] = np.nan
        return areas << (u.pix ** 2)

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
        source.  The angle increases in the counter-clockwise direction.
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
        The ratio of the lengths of the semimajor and semiminor axes:

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
        semiminor axes (or 1.0 minus the `elongation`):

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
        source.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        as:

        .. math::
            G = \frac{1}{\left | \bar{x} \right | n (n - 1)}
            \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\bar{x}` is the mean over pixel values :math:`x_i`
        within the source segment.

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
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.
        """
        if self._detection_cat is not None:
            # local background aperture defined using the source
            # centroid and bbox defined by detection image
            return self._detection_cat.local_background_aperture

        if self._localbkg_width == 0:
            return self._null_object

        aperture = []
        for bbox_ in self._bbox:
            xpos = 0.5 * (bbox_.ixmin + bbox_.ixmax - 1)
            ypos = 0.5 * (bbox_.iymin + bbox_.iymax - 1)
            scale = 1.5
            width_bbox = bbox_.ixmax - bbox_.ixmin
            width_in = width_bbox * scale
            width_out = width_in + 2 * self._localbkg_width
            height_bbox = bbox_.iymax - bbox_.iymin
            height_in = height_bbox * scale
            height_out = height_in + 2 * self._localbkg_width
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
        if self._localbkg_width == 0:
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
                # check not enough unmasked pixels
                if len(values) < 10:  # pragma: no cover
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

    def _make_aperture_data(self, label, xcentroid, ycentroid, aperture_bbox,
                            local_background, make_error=True):
        """
        Make cutouts of data, error, and mask arrays for aperture
        photometry (e.g., circular or Kron).

        Neighboring sources can be included, masked, or corrected based
        on the ``apermask_method`` keyword.
        """
        # make cutouts of the data based on the aperture bbox
        slc_lg, slc_sm = aperture_bbox.get_overlap_slices(self._data.shape)
        data = self._data[slc_lg] - local_background
        data_mask = self._data_mask[slc_lg]
        if make_error and self._error is not None:
            error = self._error[slc_lg]
        else:
            error = None

        # calculate cutout centroid position
        cutout_xycen = (xcentroid - max(0, aperture_bbox.ixmin),
                        ycentroid - max(0, aperture_bbox.iymin))

        # mask or correct neighboring sources
        if self._apermask_method != 'none':
            segment_img = self._segment_img.data[slc_lg]
            segm_mask = np.logical_and(segment_img != label,
                                       segment_img != 0)
        if self._apermask_method == 'mask':
            mask = data_mask | segm_mask
        else:
            mask = data_mask

        if self._apermask_method == 'correct':
            from ._utils import mask_to_mirrored_value
            data = mask_to_mirrored_value(data, segm_mask, cutout_xycen,
                                          mask=mask)
            if error is not None:
                error = mask_to_mirrored_value(error, segm_mask, cutout_xycen,
                                               mask=mask)

        return data, error, mask, cutout_xycen, slc_sm

    def make_circular_apertures(self, radius):
        """
        Return a list of circular apertures with the specified radius
        centered at the source centroid position.

        If provided, the `SourceCatalog` ``detection_cat`` will be used
        for the source centroids.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        Returns
        -------
        result : list of `~photutils.aperture.CircularAperture`
            A list of `~photutils.aperture.CircularAperture` instances.
            The aperture will be `None` where the source centroid
            position is not finite or where the source is completely
            masked.
        """
        if self._detection_cat is not None:
            # use detection catalog for centroids
            detcat = self._detection_cat
        else:
            detcat = self

        if radius <= 0:
            raise ValueError('radius must be > 0')

        apertures = []
        for (xcen, ycen, all_masked) in zip(detcat._xcentroid,
                                            detcat._ycentroid,
                                            self._all_masked):

            if all_masked or np.any(~np.isfinite((xcen, ycen))):
                apertures.append(None)
                continue

            apertures.append(CircularAperture((xcen, ycen), r=radius))

        return apertures

    @as_scalar
    def plot_circular_apertures(self, radius, axes=None, origin=(0, 0),
                                **kwargs):
        """
        Plot circular apertures on a matplotlib `~matplotlib.axes.Axes`
        instance.

        The apertures are defined by the specified radius and are
        centered at the source centroid position.

        If provided, the `SourceCatalog` ``detection_cat`` will be used
        for the source centroids.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        axes : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : list of `~matplotlib.patches.Patch`
            A list of matplotlib patches for the plotted aperture. The
            patches can be used, for example, when adding a plot legend.
        """
        apertures = self.make_circular_apertures(radius)
        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(axes=axes, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    @deprecated('1.1', alternative='make_circular_apertures')
    def circular_aperture(self, radius):
        """
        Return a list of circular apertures with the specified radius
        centered at the source centroid position.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        Returns
        -------
        result : list of `~photutils.aperture.CircularAperture`
            A list of `~photutils.aperture.CircularAperture` instances.
            The aperture will be `None` where the source centroid
            position is not finite or where the source is completely
            masked.
        """
        return self.make_circular_apertures(radius)

    def circular_photometry(self, radius, name=None, overwrite=False):
        """
        Perform aperture photometry for each source using a circular
        aperture of the specified radius centered at the source centroid
        position.

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the flux and flux error. The attribute names
            ``[name]_flux`` and ``[name]_fluxerr`` will store the
            photometry results. For example, these names can then be
            included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, fluxerr : `~numpy.ndarray` of floats, floats, or `~astropy.units.Quantity`
            The aperture fluxes and flux errors. NaN will be returned
            where the circular aperture is `None` (e.g., where the
            source centroid position is not finite).
        """
        if radius <= 0:
            raise ValueError('radius must be > 0')

        if self._detection_cat is not None:
            # use source centroid defined by detection image
            detcat = self._detection_cat
        else:
            detcat = self

        apertures = self.make_circular_apertures(radius)

        flux = []
        fluxerr = []
        for (label, aperture, xcen, ycen, bkg) in zip(
                self.labels, apertures, detcat._xcentroid, detcat._ycentroid,
                self._local_background):

            if aperture is None:
                flux.append(np.nan)
                fluxerr.append(np.nan)
                continue

            aperture_mask = aperture.to_mask(method='exact')
            data, error, mask, _, slc_sm = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels
            # ignore RuntimeWarning for invalid data or error values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                flux.append(np.sum((aperture_weights * data)[pixel_mask]))
                if error is None:
                    fluxerr.append(np.nan)
                else:
                    fluxerr.append(np.sqrt(np.sum(
                        (aperture_weights * error**2)[pixel_mask])))

        flux = np.array(flux)
        fluxerr = np.array(fluxerr)

        if self._data_unit is not None:
            flux <<= self._data_unit
            fluxerr <<= self._data_unit

        if self.isscalar:
            flux = flux[0]
            fluxerr = fluxerr[0]

        if name is not None:
            flux_name = f'{name}_flux'
            fluxerr_name = f'{name}_fluxerr'
            self.add_extra_property(flux_name, flux, overwrite=overwrite)
            self.add_extra_property(fluxerr_name, fluxerr,
                                    overwrite=overwrite)

        return flux, fluxerr

    def _make_elliptical_apertures(self, scale=6.):
        """
        Return a list of elliptical apertures based on the scaled
        isophotal shape of the sources.

        If provided, the `SourceCatalog` ``detection_cat`` will be used
        for the source centroids.

        Parameters
        ----------
        scale : float or `~numpy.ndarray`, optional
            The scale factor to apply to the ellipse major and minor
            axes. The default value of 6.0 is roughly two times the
            isophotal extent of the source. A `~numpy.ndarray` input
            must be a 1D array of length ``nlabels``.

        Returns
        -------
        result : list of `~photutils.aperture.EllipticalAperture`
            A list of `~photutils.aperture.EllipticalAperture`
            instances. The aperture will be `None` where the source
            centroid position or elliptical shape parameters are not
            finite or where the source is completely masked.
        """
        if self._detection_cat is not None:
            # use detection catalog for centroids and elliptical shape
            # parameters
            detcat = self._detection_cat
        else:
            detcat = self

        xcen = detcat._xcentroid
        ycen = detcat._ycentroid
        major_size = detcat.semimajor_sigma.value * scale
        minor_size = detcat.semiminor_sigma.value * scale
        theta = detcat.orientation.to(u.radian).value
        if self.isscalar:
            major_size = (major_size,)
            minor_size = (minor_size,)
            theta = (theta,)

        aperture = []
        for values in zip(xcen, ycen, major_size, minor_size, theta,
                          self._all_masked):
            if values[-1] or np.any(~np.isfinite(values[:-1])):
                aperture.append(None)
                continue

            (xcen_, ycen_, major_, minor_, theta_) = values[:-1]
            aperture.append(EllipticalAperture((xcen_, ycen_), major_, minor_,
                                               theta=theta_))
        return aperture

    @lazyproperty
    @as_scalar
    def kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius.

        The *unscaled* first-moment Kron radius is given by:

        .. math::
            k_r = \frac{\sum_{i \in A} \ r_i I_i}{\sum_{i \in A} I_i}

        where :math:`I_i` are the data values and the sum is over
        pixels in an elliptical aperture whose axes are defined by
        six times the semimajor (`semimajor_sigma`) and semiminor
        axes (`semiminor_sigma`) at the calculated `orientation` (all
        properties derived from the central image moments of the
        source). :math:`r_i` is the elliptical "radius" to the pixel
        given by:

        .. math::
            r_i^2 = cxx (x_i - \bar{x})^2 +
                cxy (x_i - \bar{x})(y_i - \bar{y}) +
                cyy (y_i - \bar{y})^2

        where :math:`\bar{x}` and :math:`\bar{y}` represent the source
        centroid and the coefficients are based on image moments (`cxx`,
        `cxy`, and `cyy`).

        The scaling parameter of the `kron_radius` is defined using the
        `SourceCatalog` ``kron_params`` keyword.

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If either the numerator or denominator above is less than or
        equal to 0, then ``np.nan`` will be returned for both the Kron
        radius and Kron flux.

        If the source is completely masked, then ``np.nan`` will be
        returned for both the Kron radius and Kron flux.

        If the `SourceCatalog` ``detection_cat`` was provided, then
        its ``kron_radius`` will be returned if the source is not
        completely masked.
        """
        if self._detection_cat is not None:
            kron_radius = self._detection_cat.kron_radius
            if self.isscalar:
                kron_radius = np.atleast_1d(kron_radius)
            kron_radius[self._all_masked] = np.nan
            return kron_radius

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

            # use 'center' (whole pixels) to compute Kron radius
            aperture_mask = aperture.to_mask(method='center')

            # prepare cutouts of the data based on the aperture size
            # local background explicitly set to zero for SE agreement
            data, _, mask, xycen, slc_sm = self._make_aperture_data(
                label, xcen_, ycen_, aperture_mask.bbox, 0.0, make_error=False)

            xval = np.arange(data.shape[1]) - xycen[0]
            yval = np.arange(data.shape[0]) - xycen[1]
            xx, yy = np.meshgrid(xval, yval)
            rr = np.sqrt(cxx_ * xx**2 + cxy_ * xx * yy + cyy_ * yy**2)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels

            # ignore RuntimeWarning for invalid data values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                flux_numer = np.sum((aperture_weights * data * rr)[pixel_mask])
                flux_denom = np.sum((aperture_weights * data)[pixel_mask])

            if flux_numer <= 0 or flux_denom <= 0:
                kron_radius.append(np.nan)
                continue

            kron_radius.append(flux_numer / flux_denom)

        kron_radius = np.array(kron_radius) * u.pix
        return kron_radius

    def make_kron_apertures(self, kron_params):
        """
        Return a list of Kron elliptical apertures with the specified
        scaling and centered at the source centroid position.

        If provided, the `SourceCatalog` ``detection_cat`` will be used
        for the source centroids and elliptical shape parameters.

        Parameters
        ----------
        kron_params : list of 2 floats, optional
            A list of two parameters used to determine how the Kron
            radius and flux are calculated. The first item is the
            scaling parameter of the Kron radius (`kron_radius`)
            and the second item represents the minimum circular
            radius. If the Kron radius times sqrt( `semimajor_sigma` *
            `semiminor_sigma`) is less than than this radius, then the
            Kron flux will be measured in a circle with this minimum
            radius.

        Returns
        -------
        result : list of `~photutils.aperture.PixelAperture`
            A list of `~photutils.aperture.EllipticalAperture` or
            `~photutils.aperture.CircularAperture` instances. The
            aperture will be `None` where the source centroid position
            is not finite or where the source is completely masked.
        """
        if self._detection_cat is not None:
            # use detection catalog for centroids and elliptical shape
            # parameters
            detcat = self._detection_cat
        else:
            detcat = self

        kron_radius = detcat.kron_radius.value
        scale = kron_radius * kron_params[0]
        # NOTE: if kron_radius = NaN, scale = NaN and kron_aperture = None
        kron_apertures = self._make_elliptical_apertures(scale=scale)

        # check for minimum Kron radius
        major_sigma = detcat.semimajor_sigma.value
        minor_sigma = detcat.semiminor_sigma.value
        circ_radius = kron_radius * np.sqrt(major_sigma * minor_sigma)
        min_radius = kron_params[1]

        mask = (circ_radius < min_radius)
        idx = np.atleast_1d(mask).nonzero()[0]
        if idx.size > 0:
            circ_aperture = self.make_circular_apertures(min_radius)
            for i in idx:
                kron_apertures[i] = circ_aperture[i]

        return kron_apertures

    @as_scalar
    def plot_kron_apertures(self, kron_params, axes=None, origin=(0, 0),
                            **kwargs):
        """
        Plot Kron elliptical apertures on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The apertures are defined by the specified radius and are
        centered at the source centroid position.

        If provided, the `SourceCatalog` ``detection_cat`` will be used
        for the source centroids and elliptical shape parameters.

        Parameters
        ----------
        kron_params : list of 2 floats, optional
            A list of two parameters used to determine how the Kron
            radius and flux are calculated. The first item is the
            scaling parameter of the Kron radius (`kron_radius`)
            and the second item represents the minimum circular
            radius. If the Kron radius times sqrt( `semimajor_sigma` *
            `semiminor_sigma`) is less than than this radius, then the
            Kron flux will be measured in a circle with this minimum
            radius.

        axes : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : list of `~matplotlib.patches.Patch`
            A list of matplotlib patches for the plotted aperture. The
            patches can be used, for example, when adding a plot legend.
        """
        apertures = self.make_kron_apertures(kron_params)
        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(axes=axes, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    @lazyproperty
    @as_scalar
    def kron_aperture(self):
        r"""
        The elliptical Kron aperture.

        For sources where

        .. math::
            k_r \ \sqrt{a \cdot b} < rc_{min}

        where :math:`k_r` is the `kron_radius`, :math:`a` and
        :math:`b` are the semimajor (`semimajor_sigma`) and semiminor
        (`semiminor_sigma`) axes, respectively, and :math:`rc_{min}` is
        the minimum circular radius defined by ``kron_params[1]`` (see
        `SourceCatalog`), then a circular aperture with a radius equal
        to ``kron_params[1]`` will be returned. If ``kron_params[1] <=
        0``, then the Kron aperture will be `None`.

        If ``kron_radius = np.nan`` then a circular aperture with a
        radius equal to ``kron_params[1]`` will be returned if the
        source is not completely masked, otherwise `None` will be
        returned.

        Note that if the Kron aperture is `None`, the Kron flux will be
        ``np.nan``.
        """
        if self._detection_cat is not None:
            return self._detection_cat.kron_aperture
        return self.make_kron_apertures(self._kron_params)

    def _calc_kron_photometry(self, kron_params):
        """
        Calculate the flux and flux error in the Kron aperture (without
        units).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned.

        Returns
        -------
        kron_flux, kron_fluxerr : tuple of `~numpy.ndarray`
            The Kron flux and flux error.
        """
        # check kron_params
        kron_params = self._validate_kron_params(kron_params)

        if self._detection_cat is not None:
            detcat = self._detection_cat
        else:
            detcat = self

        kron_flux = []
        kron_fluxerr = []
        kron_aperture = self.make_kron_apertures(kron_params)
        for label, xcen, ycen, aperture, bkg in zip(detcat._label_iter,
                                                    detcat._xcentroid,
                                                    detcat._ycentroid,
                                                    kron_aperture,
                                                    self._local_background):
            if aperture is None:
                kron_flux.append(np.nan)
                kron_fluxerr.append(np.nan)
                continue

            aperture_mask = aperture.to_mask(method='exact')

            # prepare cutouts of the data based on the aperture size
            data, error, mask, _, slc_sm = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels
            # ignore RuntimeWarning for invalid data or error values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                kron_flux.append(np.sum((aperture_weights * data)[pixel_mask]))
                if error is None:
                    kron_fluxerr.append(np.nan)
                else:
                    kron_fluxerr.append(
                        np.sqrt(np.sum((aperture_weights
                                        * error**2)[pixel_mask])))

        return kron_flux, kron_fluxerr

    def kron_photometry(self, kron_params, name=None, overwrite=False):
        """
        Perform photometry for each source using an elliptical Kron
        aperture.

        This method can be used to calculate the Kron photometry using
        different scalings of the Kron radius (`kron_radius`).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        Parameters
        ----------
        kron_params : list of 2 floats, optional
            A list of two parameters used to determine how the Kron
            radius and flux are calculated. The first item is the
            scaling parameter of the Kron radius (`kron_radius`)
            and the second item represents the minimum circular
            radius. If the Kron radius times sqrt( `semimajor_sigma` *
            `semiminor_sigma`) is less than than this radius, then the
            Kron flux will be measured in a circle with this minimum
            radius.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the Kron flux and flux error. The attribute
            names ``[name]_flux`` and ``[name]_fluxerr`` will store
            the photometry results. For example, these names can then
            be included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, fluxerr : `~numpy.ndarray` of floats, floats, or `~astropy.units.Quantity`
            The aperture fluxes and flux errors. NaN will be returned
            where the circular aperture is `None` (e.g., where the
            source centroid position is not finite).
        """
        kron_flux, kron_fluxerr = self._calc_kron_photometry(kron_params)
        if self._data_unit is not None:
            kron_flux <<= self._data_unit
            kron_fluxerr <<= self._data_unit

        if self.isscalar:
            kron_flux = kron_flux[0]
            kron_fluxerr = kron_fluxerr[0]

        if name is not None:
            flux_name = f'{name}_flux'
            fluxerr_name = f'{name}_fluxerr'
            self.add_extra_property(flux_name, kron_flux, overwrite=overwrite)
            self.add_extra_property(fluxerr_name, kron_fluxerr,
                                    overwrite=overwrite)

        return kron_flux, kron_fluxerr

    @lazyproperty
    def _kron_flux_fluxerr(self):
        """
        The flux and flux error in the Kron aperture (without units).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned.
        """
        return np.transpose(self._calc_kron_photometry(self._kron_params))

    @lazyproperty
    @as_scalar
    def kron_flux(self):
        """
        The flux in the Kron aperture.

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

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

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be returned.
        """
        kron_fluxerr = self._kron_flux_fluxerr[:, 1]
        if self._data_unit is not None:
            kron_fluxerr <<= self._data_unit
        return kron_fluxerr

    @lazyproperty
    def _max_circular_kron_radius(self):
        """
        The maximum circular Kron radius used as the upper limit of
        fluxfrac_radius.
        """
        if self._detection_cat is not None:
            detcat = self._detection_cat
        else:
            detcat = self

        semimajor_sig = detcat.semimajor_sigma.value
        kron_radius = detcat.kron_radius.value
        radius = semimajor_sig * kron_radius * self._kron_params[0]
        if self.isscalar:
            radius = np.array([radius])
        return radius

    @staticmethod
    def _fluxfrac_radius_fcn(radius, data, mask, aperture, normflux):
        """
        Function whose root is found to compute the fluxfrac_radius.
        """
        aperture.r = radius
        flux, _ = aperture.do_photometry(data, mask=mask)
        return 1.0 - (flux[0] / normflux)

    @lazyproperty
    def _fluxfrac_optimizer_args(self):
        if self._detection_cat is not None:
            detcat = self._detection_cat
        else:
            detcat = self

        kron_flux = self._kron_flux_fluxerr[:, 0]  # unitless
        max_radius = self._max_circular_kron_radius

        args = []
        for label, xcen, ycen, kronflux, bkg, max_radius_ in zip(
                self.labels, detcat._xcentroid, detcat._ycentroid,
                kron_flux, self._local_background, max_radius):

            if (np.any(~np.isfinite((xcen, ycen, kronflux, max_radius_)))
                    or kronflux == 0):
                args.append(None)
                continue

            aperture = CircularAperture((xcen, ycen), r=max_radius_)
            aperture_mask = aperture.to_mask(method='exact')

            # prepare cutouts of the data based on the maximum aperture size
            data, _, mask, xycen, _ = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg,
                make_error=False)

            aperture.positions = xycen
            args.append([data, mask, aperture, kronflux, max_radius_])

        return args

    @as_scalar
    def fluxfrac_radius(self, fluxfrac, name=None, overwrite=False):
        """
        Calculate the circular radius that encloses the specified
        fraction of the Kron flux.

        To estimate the half-light radius, use ``fluxfrac = 0.5``.

        Parameters
        ----------
        fluxfrac : float
            The fraction of the Kron flux at which to find the circular
            radius.

        name : str or `None`, optional
            The attribute name which will be assigned to the value
            of the output array. For example, this name can then be
            included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        radius : 1D `~numpy.ndarray`
            The circular radius that encloses the specified fraction of
            the Kron flux. NaN is returned where no solution was found
            or if the Kron flux is zero.
        """
        if fluxfrac <= 0 or fluxfrac > 1:
            raise ValueError('fluxfrac must be > 0 and <= 1')

        from scipy.optimize import root_scalar

        radius = []
        for fluxfrac_args in self._fluxfrac_optimizer_args:
            if fluxfrac_args is None:
                radius.append(np.nan)
                continue

            max_radius = fluxfrac_args[-1]
            args = fluxfrac_args[:-1]
            args[-1] *= fluxfrac
            args = tuple(args)

            # Try to find the root of self._fluxfrac_radius_fnc, which
            # is bracketed by a min and max radius. A ValueError is
            # raised if the bracket points do not have different signs,
            # indicating no solution or multiple solutions (e.g., a
            # multi-valued function). This can happen when at some
            # radius, flux starts decreasing with increasing radius (due
            # to negative data values), resulting in multiple possible
            # solutions. If no solution is found, we iteratively
            # decrease the max radius to narrow the bracket range until
            # the root is found. If max radius drops below the min
            # radius (0.1), then no solution is possible and NaN will be
            # returned as the result.
            found = False
            min_radius = 0.1
            max_radius_delta = 1.0
            while max_radius > min_radius and found is False:
                try:
                    bracket = [min_radius, max_radius]
                    result = root_scalar(self._fluxfrac_radius_fcn, args=args,
                                         bracket=bracket, method='brentq')
                    result = result.root
                    found = True
                except ValueError:  # pragma: no cover
                    # ValueError is raised if the bracket points do not
                    # have different signs
                    max_radius -= max_radius_delta

            # no solution found between min_radius and max_radius
            if found is False:
                result = np.nan

            radius.append(result)

        result = np.array(radius) << u.pix

        if name is not None:
            self.add_extra_property(name, result, overwrite=overwrite)

        return result
