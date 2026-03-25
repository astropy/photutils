# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating the properties of sources defined by a
segmentation image.
"""

import functools
import inspect
import math
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import binary_erosion, convolve, map_coordinates
from scipy.optimize import root_scalar

from photutils.aperture import (BoundingBox, CircularAperture,
                                EllipticalAperture, RectangularAnnulus)
from photutils.background import SExtractorBackground
from photutils.centroids import centroid_quadratic
from photutils.morphology import gini as gini_func
from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.utils import _mask_to_mirrored_value
from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._misc import _get_meta
from photutils.utils._moments import _image_moments
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils.cutouts import CutoutImage

__all__ = ['SourceCatalog']


# Default table columns for `to_table()` output
DEFAULT_COLUMNS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity', 'min_value', 'max_value',
                   'local_background', 'segment_flux', 'segment_fluxerr',
                   'kron_flux', 'kron_fluxerr']


def as_scalar(method):
    """
    Return a decorated method where it will always return a scalar value
    (instead of a length-1 tuple/list/array) if the class is scalar.

    Note that lazyproperties that begin with '_' should not have this
    decorator applied. Such properties are assumed to always be iterable
    and when slicing (see __getitem__) from a cached multi-object
    catalog to create a single-object catalog, they will no longer be
    scalar.

    Parameters
    ----------
    method : function
        The method to be decorated.

    Returns
    -------
    decorator : function
        The decorated method.
    """
    @functools.wraps(method)
    def _as_scalar(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return (result[0] if args[0].isscalar and len(result) == 1
                    else result)
        except TypeError:  # if result has no len
            return result

    return _as_scalar


def use_detcat(method):
    """
    Return a decorated method where it will return the value from the
    detection image catalog instead of using the method to calculate it.

    Parameters
    ----------
    method : function
        The method to be decorated.

    Returns
    -------
    decorator : function
        The decorated method.
    """
    @functools.wraps(method)
    def _use_detcat(self, *args, **kwargs):
        if self._detection_cat is None:
            return method(self, *args, **kwargs)

        return getattr(self._detection_cat, method.__name__)

    return _use_detcat


class SourceCatalog:
    """
    Class to create a catalog of photometry and morphological properties
    for sources defined by a segmentation image.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array from which to calculate the source photometry and
        properties. If ``convolved_data`` is input, then a convolved
        version of ``data`` will be used instead of ``data`` to
        calculate the source centroid and morphological properties.
        Source photometry is always measured from ``data``. For
        accurate source properties and photometry, ``data`` should be
        background-subtracted. Non-finite ``data`` values (NaN and inf)
        are automatically masked.

    segment_img : `~photutils.segmentation.SegmentationImage`
        A `~photutils.segmentation.SegmentationImage` object defining
        the sources.

    convolved_data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array used to calculate the source centroid and
        morphological properties. Typically, ``convolved_data``
        should be the input ``data`` array convolved by the
        same smoothing kernel that was applied to the detection
        image when deriving the source segments (e.g., see
        :func:`~photutils.segmentation.detect_sources`). If
        ``convolved_data`` is `None`, then the unconvolved ``data`` will
        be used instead. Non-finite ``convolved_data`` values (NaN and
        inf) are not automatically masked, unless they are at the same
        position of non-finite values in the input ``data`` array. Such
        pixels can be masked using the ``mask`` keyword.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data``
        array. ``error`` is assumed to include *all* sources of
        error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have
        the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vice versa) with identical
        units. Non-finite ``error`` values (NaN and inf) are not
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

    background : float, 2D `~numpy.ndarray`, or `~astropy.units.Quantity`, \
            optional
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
        sky-based properties will be set to `None`. This keyword will be
        ignored if ``detection_cat`` is input.

    localbkg_width : int, optional
        The width of the rectangular annulus used to compute a local
        background around each source. If zero, then no local background
        subtraction is performed. The local background affects the
        ``min_value``, ``max_value``, ``segment_flux``, ``kron_flux``,
        and ``fluxfrac_radius`` properties. It is also used when
        calculating circular and Kron aperture photometry (i.e.,
        `circular_photometry` and `kron_photometry`). It does not affect
        the moment-based morphological properties of the source.

    apermask_method : {'correct', 'mask', 'none'}, optional
        The method used to handle neighboring sources when performing
        aperture photometry (e.g., circular apertures or elliptical Kron
        apertures). This parameter also affects the Kron radius. The
        options are:

        * 'correct': replace pixels assigned to neighboring sources by
          replacing them with pixels on the opposite side of the source
          center (equivalent to MASK_TYPE=CORRECT in SourceExtractor).

        * 'mask': mask pixels assigned to neighboring sources
          (equivalent to MASK_TYPE=BLANK in SourceExtractor).

        * 'none': do not mask any pixels (equivalent to MASK_TYPE=NONE
          in SourceExtractor).

        This keyword will be ignored if ``detection_cat`` is input. In
        that case, the ``apermask_method`` set in the ``detection_cat``
        will be used.

    kron_params : tuple of 2 or 3 floats, optional
        A list of parameters used to determine the Kron aperture.
        The first item is the scaling parameter of the unscaled Kron
        radius and the second item represents the minimum value for
        the unscaled Kron radius in pixels. The optional third item is
        the minimum circular radius in pixels. If ``kron_params[0]``
        * `kron_radius` * sqrt(`semimajor_sigma` * `semiminor_sigma`)
        is less than or equal to this radius, then the Kron aperture
        will be a circle with this minimum radius. This keyword will be
        ignored if ``detection_cat`` is input.

    detection_cat : `SourceCatalog`, optional
        A `SourceCatalog` object for the detection image. The
        segmentation image used to create the detection catalog must
        be the same one input to ``segment_img``. If input, then the
        detection catalog source centroids and morphological/shape
        properties will be returned instead of calculating them
        from the input ``data``. The detection catalog centroids
        and shape properties will also be used to perform aperture
        photometry (i.e., circular and Kron). If ``detection_cat``
        is input, then the input ``wcs``, ``apermask_method``, and
        ``kron_params`` keywords will be ignored. This keyword affects
        `circular_photometry` (including returned apertures), all Kron
        parameters (Kron radius, flux, flux errors, apertures, and
        custom `kron_photometry`), and `fluxfrac_radius` (which is based
        on the Kron flux).

    progress_bar : bool, optional
        Whether to display a progress bar when calculating
        some properties (e.g., ``kron_radius``, ``kron_flux``,
        ``fluxfrac_radius``, ``circular_photometry``, ``centroid_win``,
        ``centroid_quad``). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Notes
    -----
    ``data`` should be background-subtracted for accurate source
    photometry and properties. The previously-subtracted background can
    be passed into this class to calculate properties of the background
    for each source.

    Note that this class does not convert input data in
    surface-brightness units to flux or counts. Conversion from
    surface-brightness units should be performed before using this
    class.

    `SourceExtractor`_'s centroid and morphological parameters are
    always calculated from a convolved, or filtered, "detection"
    image (``convolved_data``), i.e., the image used to define the
    segmentation image. The usual downside of the filtering is the
    sources will be made more circular than they actually are. If
    you wish to reproduce `SourceExtractor`_ centroid and morphology
    results, then input the ``convolved_data``. If ``convolved_data``
    is `None`, then the unfiltered ``data`` will be used for the source
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

    .. math::

        \\Delta F = \\sqrt{\\sum_{i \\in S}
            \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.segmentation.SourceCatalog.segment_fluxerr`,
    :math:`S` are the unmasked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    Custom errors for source segments can be calculated using
    the `~photutils.segmentation.SourceCatalog.error_ma` and
    `~photutils.segmentation.SourceCatalog.background_ma` properties,
    which are 2D `~numpy.ma.MaskedArray` cutout versions of the input
    ``error`` and ``background`` arrays. The mask is `True` for pixels
    outside the source segment, masked pixels from the ``mask`` input,
    or any non-finite ``data`` values (NaN and inf).

    **Scalar vs. Multi-source Catalogs**

    A `SourceCatalog` can represent a single source or multiple
    sources. Most properties adapt their return type accordingly: for
    a multi-source catalog, properties return arrays or lists (one
    element per source); for a single-source (scalar) catalog, the
    same properties return a scalar value or a single object. For
    example, `kron_aperture` returns a list of aperture objects for a
    multi-source catalog, but a single aperture object for a scalar
    catalog. Similarly, `data` returns a list of 2D cutout arrays for a
    multi-source catalog, but a single 2D array for a scalar catalog. A
    scalar catalog is created when the input segmentation image contains
    only one source or when a multi-source catalog is indexed to select
    a single source.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """

    def __init__(self, data, segment_img, *, convolved_data=None, error=None,
                 mask=None, background=None, wcs=None, localbkg_width=0,
                 apermask_method='correct', kron_params=(2.5, 1.4, 0.0),
                 detection_cat=None, progress_bar=False):

        inputs = (data, convolved_data, error, background)
        names = ('data', 'convolved_data', 'error', 'background')
        inputs, unit = process_quantities(inputs, names)
        (data, convolved_data, error, background) = inputs

        self._data_unit = unit
        self._data = self._validate_array(data, 'data', shape=False)
        self._convolved_data = self._validate_array(convolved_data,
                                                    'convolved_data')
        self._segment_img = self._validate_segment_img(segment_img)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._background = self._validate_array(background, 'background')
        self.wcs = wcs
        self.localbkg_width = self._validate_localbkg_width(localbkg_width)
        self.apermask_method = self._validate_apermask_method(apermask_method)
        self.kron_params = self._validate_kron_params(kron_params)
        self.progress_bar = progress_bar

        # Needed for ordering and isscalar
        # NOTE: calculate slices before labels for performance.
        #       _labels is initially always a non-scalar array, but
        #       it can become a numpy scalar after indexing/slicing.
        self._slices = self._segment_img.slices
        self._labels = self._segment_img.labels

        if self._labels.shape == (0,):
            msg = 'segment_img must have at least one non-zero label'
            raise ValueError(msg)

        self._detection_cat = self._validate_detection_cat(detection_cat)
        attrs = ('wcs', 'apermask_method', 'kron_params')
        if self._detection_cat is not None:
            for attr in attrs:
                setattr(self, attr, getattr(self._detection_cat, attr))

        if convolved_data is None:
            self._convolved_data = self._data

        self._apermask_kwargs = {
            'circ': {'method': 'exact'},
            'kron': {'method': 'exact'},
            'fluxfrac': {'method': 'exact'},
            'cen_win': {'method': 'center'},
        }

        self.default_columns = DEFAULT_COLUMNS
        self._extra_properties = []
        self._fluxfrac_cache = {}
        self.meta = _get_meta()
        self._update_meta()

    def _validate_segment_img(self, segment_img):
        if not isinstance(segment_img, SegmentationImage):
            msg = 'segment_img must be a SegmentationImage'
            raise TypeError(msg)
        if segment_img.shape != self._data.shape:
            msg = 'segment_img and data must have the same shape'
            raise ValueError(msg)
        return segment_img

    def _validate_array(self, array, name, *, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            # UFuncTypeError is raised when subtracting float
            # local_background from int data; convert to float
            array = np.asanyarray(array)
            if array.ndim != 2:
                msg = f'{name} must be a 2D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    @staticmethod
    def _validate_localbkg_width(localbkg_width):
        if localbkg_width < 0:
            msg = 'localbkg_width must be >= 0'
            raise ValueError(msg)
        localbkg_width_int = int(localbkg_width)
        if localbkg_width_int != localbkg_width:
            msg = 'localbkg_width must be an integer'
            raise ValueError(msg)
        return localbkg_width_int

    @staticmethod
    def _validate_apermask_method(apermask_method):
        if apermask_method not in ('none', 'mask', 'correct'):
            msg = 'Invalid apermask_method value'
            raise ValueError(msg)
        return apermask_method

    @staticmethod
    def _validate_kron_params(kron_params):
        if np.ndim(kron_params) != 1:
            msg = 'kron_params must be 1D'
            raise ValueError(msg)
        nparams = len(kron_params)
        if nparams not in (2, 3):
            msg = 'kron_params must have 2 or 3 elements'
            raise ValueError(msg)
        if kron_params[0] <= 0:
            msg = 'kron_params[0] must be > 0'
            raise ValueError(msg)
        if kron_params[1] <= 0:
            msg = 'kron_params[1] must be > 0'
            raise ValueError(msg)
        if nparams == 3 and kron_params[2] < 0:
            msg = 'kron_params[2] must be >= 0'
            raise ValueError(msg)
        return tuple(kron_params)

    def _validate_detection_cat(self, detection_cat):
        if detection_cat is None:
            return None

        if not isinstance(detection_cat, SourceCatalog):
            msg = 'detection_cat must be a SourceCatalog instance'
            raise TypeError(msg)
        if not np.array_equal(detection_cat._segment_img, self._segment_img):
            msg = ('detection_cat must have same segment_img as the '
                   'input segment_img')
            raise ValueError(msg)
        return detection_cat

    def _update_meta(self):
        attrs = ('localbkg_width', 'apermask_method', 'kron_params')
        for attr in attrs:
            self.meta[attr] = getattr(self, attr)

    def _set_semode(self):
        # SE emulation
        self._apermask_kwargs = {
            'circ': {'method': 'subpixel', 'subpixels': 5},
            'kron': {'method': 'center'},
            'fluxfrac': {'method': 'subpixel', 'subpixels': 5},
            'cen_win': {'method': 'subpixel', 'subpixels': 11},
        }

    @property
    def _properties(self):
        """
        A list of all class properties, include lazyproperties (even in
        superclasses).

        The result is cached on the class to avoid repeated
        introspection via `inspect.getmembers`.
        """
        cls = self.__class__
        attr = '_cached_properties'
        # Subclasses get their own property list
        if attr not in cls.__dict__:
            def isproperty(obj):
                return isinstance(obj, property)

            setattr(cls, attr,
                    [i[0] for i in inspect.getmembers(
                        cls, predicate=isproperty)])
        return getattr(cls, attr)

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

        The result is cached on the class to avoid repeated
        introspection via `inspect.getmembers`.
        """
        cls = self.__class__
        attr = '_cached_lazyproperties'
        # Subclasses get their own lazyproperty list
        if attr not in cls.__dict__:
            def islazyproperty(obj):
                return isinstance(obj, lazyproperty)

            setattr(cls, attr,
                    [i[0] for i in inspect.getmembers(
                        cls, predicate=islazyproperty)])
        return getattr(cls, attr)

    @staticmethod
    def _index_object_list(lst, index):
        """
        Index a list of heterogeneous objects using numpy-style
        indexing.

        A numpy object array is used to support fancy and boolean
        indices on lists of tuples or other structured objects.

        A sentinel ``None`` is appended (and then removed) to prevent
        numpy from recursing into nested sequences (e.g., tuples of
        slices).

        Parameters
        ----------
        lst : list
            The list of objects to index.

        index : int, slice, list, or array
            The index to apply to the list.

        Returns
        -------
        result : list or object
            A list for array results or the element itself for scalar
            (integer) indices.
        """
        result = np.array([*lst, None], dtype=object)[:-1][index]
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot '
                   'be indexed')
            raise TypeError(msg)

        newcls = object.__new__(self.__class__)

        # Attributes defined in __init__ that are copied directly to the
        # new class
        init_attr = ('_data', '_segment_img', '_error', '_mask', '_background',
                     'wcs', '_data_unit', '_convolved_data', 'localbkg_width',
                     'apermask_method', 'kron_params', 'default_columns',
                     '_extra_properties', 'meta', '_apermask_kwargs',
                     'progress_bar')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # _labels determines ordering and isscalar
        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        # Need to slice detection_cat, if input
        attr = '_detection_cat'
        if getattr(self, attr) is None:
            setattr(newcls, attr, None)
        else:
            setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_slices'
        value = self._index_object_list(getattr(self, attr), index)
        setattr(newcls, attr, value)

        # Slice the fluxfrac_radius cache values
        newcls._fluxfrac_cache = {key: value[index]
                                  for key, value
                                  in self._fluxfrac_cache.items()}

        # Evaluated lazyproperty objects and extra properties
        keys = (set(self.__dict__.keys())
                & (set(self._lazyproperties) | set(self._extra_properties)))
        for key in keys:
            value = self.__dict__[key]

            # Do not insert attributes that are always scalar (e.g.,
            # isscalar, nlabels), i.e., not an array/list for each
            # source
            if np.isscalar(value):
                continue

            try:
                # Keep _<attr> lazyproperties as length-1 iterables;
                # _<attr> lazyproperties should not have @as_scalar applied
                if newcls.isscalar and key.startswith('_'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                # Apply fancy indices (e.g., array/list or bool
                # mask) to lists
                val = self._index_object_list(value, index)

            newcls.__dict__[key] = val
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.nlabels}', f'labels: {self.labels}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            msg = f'Scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
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

        Returns
        -------
        result : `SourceCatalog`
            A deep copy of this object.
        """
        return deepcopy(self)

    @property
    def extra_properties(self):
        """
        A list of the user-defined source properties.
        """
        return self._extra_properties

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def add_extra_property(self, name, value, overwrite=False):
        """
        Add a user-defined extra property as a new attribute.

        For example, the property ``name`` can then be included in the
        `to_table` ``columns`` keyword list to output the results in the
        table.

        The complete list of user-defined extra properties is stored in
        the `extra_properties` attribute.

        Parameters
        ----------
        name : str
            The name of property. The name must not conflict with any of
            the built-in property names or attributes.

        value : array_like or float
            The value to assign.

        overwrite : bool, option
            If `True`, will overwrite the existing property ``name``.
        """
        internal_attributes = ((set(self.__dict__.keys())
                               | set(self._properties))
                               - set(self.extra_properties))
        if name in internal_attributes:
            msg = f'{name} cannot be set because it is a built-in attribute'
            raise ValueError(msg)

        if not overwrite:
            if hasattr(self, name):
                msg = (f'{name} already exists as an attribute. Set '
                       'overwrite=True to overwrite an existing attribute.')
                raise ValueError(msg)
            if name in self._extra_properties:
                msg = (f'{name} already exists in the extra_properties '
                       'attribute list.')
                raise ValueError(msg)

        property_error = False
        if self.isscalar:
            # This allows fluxfrac_radius to add len-1 array values for
            # scalar self
            if self._has_len(value) and len(value) == 1:
                value = value[0]

            if hasattr(value, 'isscalar'):
                # e.g., Quantity, SkyCoord, Time
                if not value.isscalar:
                    property_error = True
            elif not np.isscalar(value):
                property_error = True
        elif not self._has_len(value) or len(value) != self.nlabels:
            property_error = True
        if property_error:
            msg = ('value must have the same number of elements as the '
                   'catalog in order to add it as an extra property.')
            raise ValueError(msg)

        setattr(self, name, value)
        if name not in self._extra_properties:
            self._extra_properties.append(name)

    def remove_extra_property(self, name):
        """
        Remove a user-defined extra property.

        The property must have been defined using `add_extra_property`.
        The complete list of user-defined extra properties is stored in
        the `extra_properties` attribute.

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
        properties is stored in the `extra_properties` attribute.

        Parameters
        ----------
        names : list of str or str
            The names of the properties to remove.
        """
        if isinstance(names, str):
            names = [names]

        # We copy the list here to prevent changing the list in-place
        # during the for loop below, e.g., in case a user inputs
        # self.extra_properties to ``names``
        extra_properties = self._extra_properties.copy()

        for name in names:
            if name in extra_properties:
                delattr(self, name)
                extra_properties.remove(name)
            else:
                msg = f'{name} is not a defined extra property'
                raise ValueError(msg)
        self._extra_properties = extra_properties

    def rename_extra_property(self, name, new_name):
        """
        Rename a user-defined extra property.

        The renamed property will remain at the same index in the
        `extra_properties` list.

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
        # Preserve the order of self.extra_properties
        self.extra_properties.remove(new_name)
        self.extra_properties.insert(idx, new_name)

    @lazyproperty
    def _null_objects(self):
        """
        Return `None` values.

        For example, this is used for SkyCoord properties if ``wcs`` is
        `None`.
        """
        return np.array([None] * self.nlabels)

    @lazyproperty
    def _null_values(self):
        """
        Return np.nan values.

        For example, this is used for background properties if
        ``background`` is `None`.
        """
        values = np.empty(self.nlabels)
        values.fill(np.nan)
        return values

    @lazyproperty
    def _data_cutouts(self):
        """
        A list of data cutouts using the segmentation image slices.
        """
        return [self._data[slc] for slc in self._slices_iter]

    @lazyproperty
    def _segment_img_cutouts(self):
        """
        A list of segmentation image cutouts using the segmentation
        image slices.
        """
        return [self._segment_img.data[slc] for slc in self._slices_iter]

    @lazyproperty
    def _mask_cutouts(self):
        """
        A list of mask cutouts using the segmentation image slices.

        If the input ``mask`` is None then a list of None is returned.
        """
        if self._mask is None:
            return self._null_objects
        return [self._mask[slc] for slc in self._slices_iter]

    @lazyproperty
    def _error_cutouts(self):
        """
        A list of error cutouts using the segmentation image slices.

        If the input ``error`` is None then a list of None is returned.
        """
        if self._error is None:
            return self._null_objects
        return [self._error[slc] for slc in self._slices_iter]

    @lazyproperty
    def _convdata_cutouts(self):
        """
        A list of convolved data cutouts using the segmentation image
        slices.
        """
        return [self._convolved_data[slc] for slc in self._slices_iter]

    @lazyproperty
    def _background_cutouts(self):
        """
        A list of background cutouts using the segmentation image
        slices.
        """
        if self._background is None:
            return self._null_objects
        return [self._background[slc] for slc in self._slices_iter]

    @staticmethod
    def _make_cutout_data_mask(data_cutout, mask_cutout):
        """
        Make a cutout data mask, combining both the input ``mask`` and
        non-finite ``data`` values.
        """
        data_mask = ~np.isfinite(data_cutout)
        if mask_cutout is not None:
            data_mask |= mask_cutout
        return data_mask

    def _make_cutout_data_masks(self, data_cutouts, mask_cutouts):
        """
        Make a list of cutout data masks, combining both the input
        ``mask`` and non-finite ``data`` values for each source.
        """
        data_masks = []
        for (data_cutout, mask_cutout) in zip(data_cutouts, mask_cutouts,
                                              strict=True):
            data_masks.append(self._make_cutout_data_mask(data_cutout,
                                                          mask_cutout))
        return data_masks

    @lazyproperty
    def _cutout_segment_masks(self):
        """
        Cutout boolean mask for source segment.

        The mask is `True` for all pixels (background and from other
        source segments) outside the source segment.
        """
        return [segm != label
                for label, segm in zip(self.labels,
                                       self._segment_img_cutouts, strict=True)]

    @lazyproperty
    def _cutout_data_masks(self):
        """
        Cutout boolean mask of non-finite ``data`` values combined with
        the input ``mask`` array.

        The mask is `True` for non-finite ``data`` values and where the
        input ``mask`` is `True`.
        """
        return self._make_cutout_data_masks(self._data_cutouts,
                                            self._mask_cutouts)

    @lazyproperty
    def _cutout_total_masks(self):
        """
        Boolean mask representing the combination of
        ``_cutout_segment_masks`` and ``_cutout_data_masks``.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """
        masks = []
        for mask1, mask2 in zip(self._cutout_segment_masks,
                                self._cutout_data_masks, strict=True):
            masks.append(mask1 | mask2)
        return masks

    @lazyproperty
    def _moment_data_cutouts(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the (convolved) data.

        The following pixels are set to zero in these arrays:

        * pixels outside the source segment
        * any masked pixels from the input ``mask``
        * invalid convolved data values (NaN and inf)
        * negative convolved data values; negative pixels (especially
          at large radii) can give image moments that have negative
          variances.

        These arrays are used to derive moment-based properties.
        """
        cutouts = []
        for convdata_cutout, mask_cutout, segmmask_cutout in zip(
                self._convdata_cutouts, self._mask_cutouts,
                self._cutout_segment_masks, strict=True):

            convdata_mask = (~np.isfinite(convdata_cutout)
                             | (convdata_cutout < 0) | segmmask_cutout)

            if self._mask is not None:
                convdata_mask |= mask_cutout

            cutout = convdata_cutout.copy()
            cutout[convdata_mask] = 0.0
            cutouts.append(cutout)
        return cutouts

    def _prepare_cutouts(self, arrays, *, units=True, masked=False,
                         dtype=None):
        """
        Prepare cutouts by applying optional units, masks, or dtype.
        """
        if units and masked:
            msg = 'Both units and masked cannot be True'
            raise ValueError(msg)

        if dtype is not None:
            cutouts = [cutout.astype(dtype, copy=True) for cutout in arrays]
        else:
            cutouts = arrays

        if units and self._data_unit is not None:
            cutouts = [(cutout << self._data_unit) for cutout in cutouts]
        if masked:
            return [np.ma.masked_array(cutout, mask=mask)
                    for cutout, mask in zip(cutouts, self._cutout_total_masks,
                                            strict=True)]

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
        self._segment_img.check_labels(labels)
        sorter = np.argsort(self.labels)
        indices = sorter[np.searchsorted(self.labels, labels, sorter=sorter)]
        return self[indices]

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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
        elif isinstance(columns, str):
            table_columns = [columns]
        else:
            table_columns = columns

        tbl = QTable()
        tbl.meta.update(self.meta)  # keep tbl.meta type
        for column in table_columns:
            values = getattr(self, column)

            # Column assignment requires an object with a length
            if self.isscalar:
                values = (values,)

            tbl[column] = values
        return tbl

    @lazyproperty
    def nlabels(self):
        """
        The number of source labels.
        """
        return len(self.labels)

    @property
    @as_scalar
    def label(self):
        """
        The source label number(s).

        This label number corresponds to the assigned pixel value in the
        `~photutils.segmentation.SegmentationImage`.

        Returns an array for multi-source catalogs, or a scalar for a
        single-source catalog.
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
        labels = self.label
        if self.isscalar:
            labels = np.array((labels,))
        return labels

    @property
    @as_scalar
    def slices(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source.

        Returns a list for multi-source catalogs, or a single tuple for
        a single-source catalog.
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

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._segment_img_cutouts, units=False,
                                     masked=False)

    @lazyproperty
    @as_scalar
    def segment_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout of the segmentation image
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._segment_img_cutouts, units=False,
                                     masked=True)

    @lazyproperty
    @as_scalar
    def data(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._data_cutouts, units=True,
                                     masked=False, dtype=float)

    @lazyproperty
    @as_scalar
    def data_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the data using the
        minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._data_cutouts, units=False,
                                     masked=True, dtype=float)

    @lazyproperty
    @as_scalar
    def convdata(self):
        """
        A 2D `~numpy.ndarray` cutout from the convolved data using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._convdata_cutouts, units=True,
                                     masked=False, dtype=float)

    @lazyproperty
    @as_scalar
    def convdata_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the convolved data
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._convdata_cutouts, units=False,
                                     masked=True, dtype=float)

    @lazyproperty
    @as_scalar
    def error(self):
        """
        A 2D `~numpy.ndarray` cutout from the error array using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._error is None:
            return self._null_objects
        return self._prepare_cutouts(self._error_cutouts, units=True,
                                     masked=False)

    @lazyproperty
    @as_scalar
    def error_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the error array using
        the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._error is None:
            return self._null_objects
        return self._prepare_cutouts(self._error_cutouts, units=False,
                                     masked=True)

    @lazyproperty
    @as_scalar
    def background(self):
        """
        A 2D `~numpy.ndarray` cutout from the background array using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._background is None:
            return self._null_objects
        return self._prepare_cutouts(self._background_cutouts, units=True,
                                     masked=False)

    @lazyproperty
    @as_scalar
    def background_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the background array.
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._background is None:
            return self._null_objects
        return self._prepare_cutouts(self._background_cutouts, units=False,
                                     masked=True)

    @lazyproperty
    def _all_masked(self):
        """
        True if all pixels over the source segment are masked.
        """
        return np.array([np.all(mask) for mask in self._cutout_total_masks])

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

    @staticmethod
    def _reduceat(values, ufunc, *, transform=None):
        """
        Apply ``ufunc.reduceat`` to a list of arrays.

        This is significantly faster than a list comprehension with
        individual NumPy calls for each array.

        Parameters
        ----------
        values : list of 1D `~numpy.ndarray`
            A list of 1D arrays.

        ufunc : `~numpy.ufunc`
            The NumPy ufunc to apply (e.g., `~numpy.add`,
            `~numpy.minimum`, `~numpy.maximum`).

        transform : callable or None, optional
            An optional transformation to apply to the concatenated
            array before reducing (e.g., `~numpy.square`).

        Returns
        -------
        result : `~numpy.ndarray`
            The reduceat result.

        sizes : `~numpy.ndarray`
            The sizes of the input arrays.
        """
        if not values:
            return np.array([]), np.array([], dtype=int)

        sizes = np.array([len(arr) for arr in values])
        splits = np.concatenate(([0], np.cumsum(sizes[:-1])))
        concat = np.concatenate(values)
        if transform is not None:
            concat = transform(concat)
        return ufunc.reduceat(concat, splits), sizes

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
        if self._error is None:
            return self._null_objects
        return self._get_values(self.error_ma)

    @lazyproperty
    def _background_values(self):
        """
        A 1D array of unmasked background values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self._background is None:
            return self._null_objects
        return self._get_values(self.background_ma)

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments(self):
        """
        Spatial moments up to 3rd order of the source.
        """
        return np.array([_image_moments(arr, order=3)
                         for arr in self._moment_data_cutouts])

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        cutout_centroid = self.cutout_centroid
        if self.isscalar:
            cutout_centroid = cutout_centroid[np.newaxis, :]
        return np.array([_image_moments(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in
                         zip(self._moment_data_cutouts, cutout_centroid[:, 0],
                             cutout_centroid[:, 1], strict=True)])

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        centroid within the isophotal source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        moments = self.moments
        if self.isscalar:
            moments = moments[np.newaxis, :]

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ycentroid = moments[:, 1, 0] / moments[:, 0, 0]
            xcentroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((xcentroid, ycentroid))

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid within the isophotal
        source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @lazyproperty
    @use_detcat
    def _xcentroid(self):
        """
        The ``x`` coordinate of the `centroid` within the source
        segment, always as an iterable.
        """
        if self.isscalar:
            xcentroid = self.centroid[0:1]  # scalar array
        else:
            xcentroid = self.centroid[:, 0]
        return xcentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._xcentroid

    @lazyproperty
    @use_detcat
    def _ycentroid(self):
        """
        The ``y`` coordinate of the `centroid` within the source
        segment, always as an iterable.
        """
        if self.isscalar:
            ycentroid = self.centroid[1:2]  # scalar array
        else:
            ycentroid = self.centroid[:, 1]
        return ycentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._ycentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_win(self):
        """
        The ``(x, y)`` coordinate of the "windowed" centroid.

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE and YWIN_IMAGE parameters.

        Notes
        -----
        On each iteration, the centroid is calculated using all pixels
        within a circular aperture of ``4 * sigma`` from the current
        position, weighting pixel values with a 2D Gaussian with a
        standard deviation of ``sigma``. ``sigma`` is the half-light
        radius (i.e., ``fluxfrac_radius(0.5)``) times (2.0 / 2.35). A
        minimum half-light radius of 0.5 pixels is used. Iteration stops
        when the change in centroid position falls below a pre-defined
        threshold or a maximum number of iterations is reached.

        If the windowed centroid falls outside the 1-sigma ellipse
        shape based on the image moments, then the isophotal `centroid`
        will be used instead. If the half-light radius is not finite
        (e.g., due to a non-finite Kron radius), then ``np.nan`` will be
        returned.
        """
        # Use .copy() to avoid mutating the cached fluxfrac_radius value
        radius_hl = self.fluxfrac_radius(0.5).value.copy()
        if self.isscalar:
            radius_hl = np.array([radius_hl])

        # Track which sources have non-finite half-light radii (e.g.,
        # due to NaN kron_radius). These sources cannot have a
        # meaningful windowed centroid.
        nan_hl = ~np.isfinite(radius_hl)

        # Apply a minimum half-light radius of 0.5 pixels (matching
        # SourceExtractor) for valid but very small values
        min_radius = 0.5
        small_mask = np.isfinite(radius_hl) & (radius_hl < min_radius)
        radius_hl[small_mask] = min_radius

        labels = self.labels
        if self.progress_bar:
            desc = 'centroid_win'
            labels = add_progress_bar(labels, desc=desc)

        # Pre-fetch arrays used in the inner loop
        data_arr = self._data
        mask_arr = self._mask
        segm_data = self._segment_img.data
        data_shape = data_arr.shape
        do_correct = self.apermask_method == 'correct'
        do_segm_mask = self.apermask_method != 'none'
        max_aper_size = max(data_arr.size, 1_000_000)

        max_iters = 16
        centroid_threshold = 0.0001

        xcen_win = []
        ycen_win = []
        for label, xcen, ycen, rad_hl, nan_hl_ in zip(
                labels, self._xcentroid, self._ycentroid, radius_hl,
                nan_hl, strict=True):

            if nan_hl_ or math.isnan(xcen) or math.isnan(ycen):
                xcen_win.append(np.nan)
                ycen_win.append(np.nan)
                continue

            sigma = 2.0 * rad_hl * gaussian_fwhm_to_sigma
            inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)
            radius = 4.0 * sigma
            radius_sq = radius * radius

            # Compute the full (unclipped) bounding box for the aperture
            # using the initial centroid. The radius is fixed, so the
            # bbox size stays the same across iterations even if the
            # center shifts slightly.
            bbox_halfsize = int(radius + 1.5)
            full_ny = full_nx = 2 * bbox_halfsize + 1

            # OOM guard
            if full_ny * full_nx > max_aper_size:
                xcen_win.append(np.nan)
                ycen_win.append(np.nan)
                continue

            # Cache for cutout data when the integer bbox doesn't change
            prev_ixcen = prev_iycen = None
            cached_data = None
            cached_mask = None

            iter_ = 0
            dcen = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                while iter_ < max_iters and dcen > centroid_threshold:
                    # Compute integer bounding box
                    ixmin = int(xcen + 0.5) - bbox_halfsize
                    ixmax = ixmin + full_nx
                    iymin = int(ycen + 0.5) - bbox_halfsize
                    iymax = iymin + full_ny

                    # Clip to data boundaries
                    slc_y = slice(max(0, iymin), min(data_shape[0], iymax))
                    slc_x = slice(max(0, ixmin), min(data_shape[1], ixmax))
                    if (slc_y.start >= slc_y.stop
                            or slc_x.start >= slc_x.stop):
                        xcen = np.nan
                        ycen = np.nan
                        break

                    cur_ixcen = int(xcen + 0.5)
                    cur_iycen = int(ycen + 0.5)

                    # Recompute cutout data only when the integer center
                    # changes to avoid redundant _mask_to_mirrored_value
                    # calls
                    if cur_ixcen != prev_ixcen or cur_iycen != prev_iycen:
                        prev_ixcen = cur_ixcen
                        prev_iycen = cur_iycen

                        data = data_arr[slc_y, slc_x].astype(float)
                        data_mask = ~np.isfinite(data)
                        if mask_arr is not None:
                            data_mask |= mask_arr[slc_y, slc_x]

                        cutout_xycen = (xcen - max(0, ixmin),
                                        ycen - max(0, iymin))

                        if do_segm_mask:
                            seg_cut = segm_data[slc_y, slc_x]
                            segm_mask = ((seg_cut != label)
                                         & (seg_cut != 0))
                            if self.apermask_method == 'mask':
                                data_mask = data_mask | segm_mask

                        if do_correct:
                            data = _mask_to_mirrored_value(
                                data, segm_mask, cutout_xycen,
                                mask=data_mask)

                        cached_data = data
                        cached_mask = data_mask

                    # Centroid position in cutout coordinates
                    cx = xcen - max(0, ixmin)
                    cy = ycen - max(0, iymin)

                    ny = slc_y.stop - slc_y.start
                    nx = slc_x.stop - slc_x.start

                    # Build coordinate grids relative to centroid
                    # (reused for circle mask, Gaussian, and moments)
                    xvals = np.arange(nx) - cx
                    yvals = np.arange(ny) - cy
                    xx = xvals[np.newaxis, :]
                    yy = yvals[:, np.newaxis]

                    # Inline binary circle mask
                    rr2 = xx * xx + yy * yy
                    aper_weights = (rr2 <= radius_sq).astype(float)

                    # Inline Gaussian weight
                    gweight = np.exp(rr2 * inv_2sigma2)

                    # Apply weights and mask
                    weighted = (cached_data * aper_weights * gweight)
                    weighted[cached_mask] = 0.0

                    # Inline moment computation
                    total = np.sum(weighted)
                    dx = np.sum(weighted * xx) / total
                    dy = np.sum(weighted * yy) / total

                    dcen = math.sqrt(dx * dx + dy * dy)
                    xcen += dx * 2.0
                    ycen += dy * 2.0
                    iter_ += 1

            xcen_win.append(xcen)
            ycen_win.append(ycen)

        xcen_win = np.array(xcen_win)
        ycen_win = np.array(ycen_win)

        # Reset to the isophotal centroid if the windowed centroid is
        # outside the 1-sigma ellipse or if the iteration failed (NaN
        # from aperture off-image). Sources with NaN half-light radius
        # keep NaN (no valid window size).
        dx = self._xcentroid - xcen_win
        dy = self._ycentroid - ycen_win
        cxx = self.cxx.value
        cxy = self.cxy.value
        cyy = self.cyy.value
        if self.isscalar:
            cxx = (cxx,)
            cxy = (cxy,)
            cyy = (cyy,)
        reset = ((cxx * dx**2 + cxy * dx * dy + cyy * dy**2) > 1)
        nan_cen = np.isnan(xcen_win) | np.isnan(ycen_win)
        reset |= nan_cen & ~nan_hl
        if np.any(reset):
            xcen_win[reset] = self._xcentroid[reset]
            ycen_win[reset] = self._ycentroid[reset]

        return np.transpose((xcen_win, ycen_win))

    @lazyproperty
    @use_detcat
    @as_scalar
    def xcentroid_win(self):
        """
        The ``x`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE parameters.
        """
        if self.isscalar:
            xcentroid = self.centroid_win[0]  # scalar array
        else:
            xcentroid = self.centroid_win[:, 0]
        return xcentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def ycentroid_win(self):
        """
        The ``y`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s YWIN_IMAGE parameters.
        """
        if self.isscalar:
            ycentroid = self.centroid_win[1]  # scalar array
        else:
            ycentroid = self.centroid_win[:, 1]
        return ycentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_win(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        "windowed" centroid.

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE and YWIN_IMAGE parameters. See
        `centroid_win` for further details about the algorithm.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.centroid_win - origin

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_quad(self):
        """
        The ``(x, y)`` centroid coordinate, relative to the cutout data,
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.

        Notes
        -----
        `~photutils.centroids.centroid_quadratic` is used to calculate
        the centroid with ``fit_boxsize=3``.

        Because this centroid is based on fitting data, it can fail for
        many reasons including:

        * quadratic fit failed
        * quadratic fit does not have a maximum
        * quadratic fit maximum falls outside image
        * not enough unmasked data points (6 are required)

        In these cases, then the isophotal `centroid` will be used
        instead.

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        centroid_quad = []
        with warnings.catch_warnings():
            # Ignore fit warnings:
            #   - quadratic fit failed
            #   - quadratic fit does not have a maximum
            #   - quadratic fit maximum falls outside image
            #   - not enough unmasked data points (6 are required)
            #   - maximum value is at the edge of the data
            warnings.simplefilter('ignore', AstropyUserWarning)

            cutouts = self._data_cutouts
            if self.progress_bar:
                desc = 'centroid_quad'
                cutouts = add_progress_bar(cutouts, desc=desc)

            for data, mask in zip(cutouts, self._cutout_total_masks,
                                  strict=True):
                try:
                    centroid = centroid_quadratic(data, mask=mask,
                                                  fit_boxsize=3)
                except ValueError:
                    centroid = (np.nan, np.nan)
                centroid_quad.append(centroid)

        centroid_quad = np.array(centroid_quad)

        # Use the segment barycenter if fit returned NaN
        nan_mask = (np.isnan(centroid_quad[:, 0])
                    | np.isnan(centroid_quad[:, 1]))
        if np.any(nan_mask):
            centroid_quad[nan_mask] = self.cutout_centroid[nan_mask]

        return centroid_quad

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_quad(self):
        """
        The ``(x, y)`` centroid coordinate, calculated by fitting a 2D
        quadratic polynomial to the unmasked pixels in the source
        segment.

        Notes
        -----
        `~photutils.centroids.centroid_quadratic` is used to calculate
        the centroid with ``fit_boxsize=3``.

        Because this centroid is based on fitting data, it can fail for
        many reasons, returning (np.nan, np.nan):

        * quadratic fit failed
        * quadratic fit does not have a maximum
        * quadratic fit maximum falls outside image
        * not enough unmasked data points (6 are required)

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid_quad + origin

    @lazyproperty
    @use_detcat
    @as_scalar
    def xcentroid_quad(self):
        """
        The ``x`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        if self.isscalar:
            xcentroid = self.centroid_quad[0]  # scalar array
        else:
            xcentroid = self.centroid_quad[:, 0]
        return xcentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def ycentroid_quad(self):
        """
        The ``y`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        if self.isscalar:
            ycentroid = self.centroid_quad[1]  # scalar array
        else:
            ycentroid = self.centroid_quad[:, 1]
        return ycentroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid(self):
        """
        The sky coordinate of the `centroid` within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.xcentroid, self.ycentroid)

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_icrs(self):
        """
        The sky coordinate in the International Celestial Reference
        System (ICRS) frame of the `centroid` within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.sky_centroid.icrs

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_win(self):
        """
        The sky coordinate of the "windowed" centroid (`centroid_win`)
        within the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.xcentroid_win, self.ycentroid_win)

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_quad(self):
        """
        The sky coordinate of the centroid (`centroid_quad`), calculated
        by fitting a 2D quadratic polynomial to the unmasked pixels in
        the source segment.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.xcentroid_quad,
                                       self.ycentroid_quad)

    @lazyproperty
    @use_detcat
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment, always as an iterable.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices_iter]

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.

        Returns a list for multi-source catalogs, or a single
        `~photutils.aperture.BoundingBox` for a single-source catalog.
        """
        return self._bbox

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ll(self):
        """
        Lower-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ul(self):
        """
        Upper-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_lr(self):
        """
        Lower-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ur(self):
        """
        Upper-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ll))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ul))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_lr))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ur))

    @lazyproperty
    @as_scalar
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """
        values, _ = self._reduceat(self._data_values, np.minimum)
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
        values, _ = self._reduceat(self._data_values, np.maximum)
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
        for idx, slc in zip(index, self._slices_iter, strict=True):
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
        for idx, slc in zip(index, self._slices_iter, strict=True):
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
        if self.isscalar:
            xidx = self.minval_index[1]
        else:
            xidx = self.minval_index[:, 1]
        return xidx

    @lazyproperty
    @as_scalar
    def minval_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            yidx = self.minval_index[0]
        else:
            yidx = self.minval_index[:, 0]
        return yidx

    @lazyproperty
    @as_scalar
    def maxval_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            xidx = self.maxval_index[1]
        else:
            xidx = self.maxval_index[:, 1]
        return xidx

    @lazyproperty
    @as_scalar
    def maxval_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            yidx = self.maxval_index[0]
        else:
            yidx = self.maxval_index[:, 0]
        return yidx

    @lazyproperty
    @as_scalar
    def segment_flux(self):
        r"""
        The sum of the unmasked ``data`` values within the source
        segment.

        .. math::

            F = \sum_{i \in S} I_i

        where :math:`F` is ``segment_flux``, :math:`I_i` is the
        background-subtracted ``data``, and :math:`S` are the unmasked
        pixels in the source segment.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        localbkg = self._local_background
        if self.isscalar:
            localbkg = localbkg[0]
        source_sum, _ = self._reduceat(self._data_values, np.add)
        source_sum -= self.area.value * localbkg
        if self._data_unit is not None:
            source_sum <<= self._data_unit
        return source_sum

    @lazyproperty
    @as_scalar
    def segment_fluxerr(self):
        r"""
        The uncertainty of `segment_flux`, propagated from the input
        ``error`` array.

        ``segment_fluxerr`` is the quadrature sum of the total errors
        over the unmasked pixels within the source segment:

        .. math::

            \Delta F = \sqrt{\sum_{i \in S} \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `segment_fluxerr`,
        :math:`\sigma_{\mathrm{tot, i}}` are the pixel-wise total errors
        (``error``), and :math:`S` are the unmasked pixels in the source
        segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the error array.
        """
        if self._error is None:
            err = self._null_values
        else:
            err_sq, _ = self._reduceat(self._error_values, np.add,
                                       transform=np.square)
            err = np.sqrt(err_sq)

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
            bkg_sum = self._null_values
        else:
            bkg_sum, _ = self._reduceat(
                self._background_values, np.add)

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
            bkg_mean = self._null_values
        else:
            bkg_sum, sizes = self._reduceat(
                self._background_values, np.add)
            bkg_mean = bkg_sum / sizes

        if self._data_unit is not None:
            bkg_mean <<= self._data_unit
        return bkg_mean

    @lazyproperty
    @as_scalar
    def background_centroid(self):
        """
        The value of the per-pixel ``background`` at the position of the
        isophotal (center-of-mass) `centroid`.

        If ``detection_cat`` is input, then its `centroid` will be used.

        The background values at fractional position values are
        determined using bilinear interpolation.
        """
        if self._background is None:
            bkg = self._null_values
        else:
            xcen = self._xcentroid
            ycen = self._ycentroid
            bkg = map_coordinates(self._background, (ycen, xcen), order=1,
                                  mode='nearest')

            mask = np.isfinite(xcen) & np.isfinite(ycen)
            bkg[~mask] = np.nan

        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    @lazyproperty
    @use_detcat
    @as_scalar
    def segment_area(self):
        """
        The total area of the source segment in units of pixels**2.

        This area is simply the area of the source segment from the
        input ``segment_img``. It does not take into account any data
        masking (i.e., a ``mask`` input to `SourceCatalog` or invalid
        ``data`` values).
        """
        areas = []
        for label, slices in zip(self.labels, self._slices_iter, strict=True):
            areas.append(np.count_nonzero(self._segment_img[slices] == label))
        return np.array(areas) << (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def area(self):
        """
        The total unmasked area of the source in units of pixels**2.

        Note that the source area may be smaller than its `segment_area`
        if a mask is input to `SourceCatalog` or if the ``data`` within
        the segment contains invalid values (NaN and inf).
        """
        areas = np.array([arr.size for arr in self._data_values]).astype(float)
        areas[self._all_masked] = np.nan
        return areas << (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """
        return np.sqrt(self.area / np.pi)

    @lazyproperty
    @use_detcat
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
        .. [1] K. Benkrid, D. Crookes, and A. Benkrid. "Design and FPGA
               Implementation of a Perimeter Estimator". Proceedings of
               the Irish Machine Vision and Image Processing Conference,
               pp. 51-57 (2000).
        """
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]])
        size = 34
        weights = np.zeros(size, dtype=float)
        weights[[5, 7, 15, 17, 25, 27]] = 1.0
        weights[[21, 33]] = np.sqrt(2.0)
        weights[[13, 23]] = (1 + np.sqrt(2.0)) / 2.0

        perimeter = []
        for mask in self._cutout_total_masks:
            if np.all(mask):
                perimeter.append(np.nan)
                continue

            data = ~mask
            data_eroded = binary_erosion(data, footprint, border_value=0)
            border = np.logical_xor(data, data_eroded).astype(int)
            perimeter_data = convolve(border, kernel, mode='constant', cval=0)
            perimeter_hist = np.bincount(perimeter_data.ravel(),
                                         minlength=size)
            perimeter.append(perimeter_hist[0:size] @ weights)

        return np.array(perimeter) * u.pix

    @lazyproperty
    @use_detcat
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
    @use_detcat
    def _covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source, always as an iterable.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu_norm = moments / moments[:, 0, 0][:, np.newaxis, np.newaxis]

        covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                          mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
        covar = covar.reshape((covar.shape[0], 2, 2))

        # Modify the covariance matrix in the case of "infinitely" thin
        # detections. This follows SourceExtractor's prescription of
        # incrementally increasing the diagonal elements by 1/12.
        delta = 1.0 / 12
        delta2 = delta**2
        # Ignore RuntimeWarning from NaN values in covar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            covar_det = np.linalg.det(covar)

            # Covariance should be positive semidefinite
            idx = np.where(covar_det < 0)[0]
            covar[idx] = np.array([[np.nan, np.nan], [np.nan, np.nan]])

            idx = np.where(covar_det < delta2)[0]
            while idx.size > 0:
                covar[idx, 0, 0] += delta
                covar[idx, 1, 1] += delta
                covar_det = np.linalg.det(covar)
                idx = np.where(covar_det < delta2)[0]
        return covar

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """
        return self._covariance * (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.empty((self.nlabels, 2))
        eigvals.fill(np.nan)
        # np.linalg.eigvalsh requires finite input values
        idx = np.unique(np.where(np.isfinite(self._covariance))[0])
        eigvals[idx] = np.linalg.eigvalsh(self._covariance[idx])

        # Check for negative variance
        # (just in case covariance matrix is not positive semidefinite)
        idx2 = np.unique(np.where(eigvals < 0)[0])
        eigvals[idx2] = (np.nan, np.nan)

        # Sort each eigenvalue pair in descending order
        # (eigvalsh returns values in ascending order)
        eigvals = np.fliplr(eigvals)

        return eigvals * u.pix**2

    @lazyproperty
    @use_detcat
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
        # This matches SourceExtractor's A parameter
        return np.sqrt(eigvals[:, 0])

    @lazyproperty
    @use_detcat
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
        # This matches SourceExtractor's B parameter
        return np.sqrt(eigvals[:, 1])

    @lazyproperty
    @use_detcat
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
    @use_detcat
    @as_scalar
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction and
        will be in the range [0, 360) degrees.
        """
        covar = self._covariance
        orient_radians = 0.5 * np.arctan2(2.0 * covar[:, 0, 1],
                                          (covar[:, 0, 0] - covar[:, 1, 1]))
        return (np.rad2deg(orient_radians) % 360) << u.deg

    @lazyproperty
    @use_detcat
    @as_scalar
    def eccentricity(self):
        r"""
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math::

            e = \sqrt{1 - \frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        semimajor_var, semiminor_var = np.transpose(self.covariance_eigvals)
        return np.sqrt(1.0 - (semiminor_var / semimajor_var))

    @lazyproperty
    @use_detcat
    @as_scalar
    def elongation(self):
        r"""
        The ratio of the lengths of the semimajor and semiminor axes.

        .. math::

            \mathrm{elongation} = \frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_sigma / self.semiminor_sigma

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipticity(self):
        r"""
        1.0 minus the ratio of the lengths of the semimajor and
        semiminor axes.

        .. math::

            \mathrm{ellipticity} = \frac{a - b}{a} = 1 - \frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_sigma / self.semimajor_sigma)

    @lazyproperty
    @use_detcat
    @as_scalar
    def covar_sigx2(self):
        r"""
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\sigma_x^2`, in units of pixel**2.
        """
        return self._covariance[:, 0, 0] * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def covar_sigy2(self):
        r"""
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.
        """
        return self._covariance[:, 1, 1] * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def covar_sigxy(self):
        r"""
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
        pixel**2.
        """
        return self._covariance[:, 0, 1] * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def cxx(self):
        r"""
        Coefficient for ``x**2`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return ((np.cos(self.orientation) / self.semimajor_sigma)**2
                + (np.sin(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def cyy(self):
        r"""
        Coefficient for ``y**2`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return ((np.sin(self.orientation) / self.semimajor_sigma)**2
                + (np.cos(self.orientation) / self.semiminor_sigma)**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def cxy(self):
        r"""
        Coefficient for ``x * y`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return (2.0 * np.cos(self.orientation) * np.sin(self.orientation)
                * ((1.0 / self.semimajor_sigma**2)
                   - (1.0 / self.semiminor_sigma**2)))

    @lazyproperty
    @use_detcat
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

        The Gini coefficient of the distribution of absolute flux values
        is calculated using the prescription from `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        (Eq. 6) as:

        .. math::

            G = \frac{1}{\overline{|x|} \, n \, (n - 1)}
                \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\overline{|x|}` is the mean of the absolute value
        of all pixel values :math:`x_i`. If the sum of all pixel values
        is zero, the Gini coefficient is zero.

        Negative pixel values are used via their absolute value. Invalid
        values (NaN and inf) in the input are automatically excluded
        from the calculation. If only a single finite pixel remains
        after filtering, the Gini coefficient is 0.0.
        """
        return np.array([gini_func(arr) for arr in self._data_values])

    @lazyproperty
    def _local_background_apertures(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.
        """
        if self.localbkg_width == 0:
            return self._null_objects

        apertures = []
        for bbox_ in self._bbox:
            xpos = 0.5 * (bbox_.ixmin + bbox_.ixmax - 1)
            ypos = 0.5 * (bbox_.iymin + bbox_.iymax - 1)
            scale = 1.5
            width_in = (bbox_.ixmax - bbox_.ixmin) * scale
            width_out = width_in + 2 * self.localbkg_width
            height_in = (bbox_.iymax - bbox_.iymin) * scale
            height_out = height_in + 2 * self.localbkg_width
            apertures.append(RectangularAnnulus((xpos, ypos), width_in,
                                                width_out, height_out,
                                                h_in=height_in, theta=0.0))
        return apertures

    @lazyproperty
    @use_detcat
    @as_scalar
    def local_background_aperture(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._local_background_apertures

    @lazyproperty
    def _local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.

        Pixels are masked where the input ``mask`` is `True`, where the
        input ``data`` is non-finite, and within any non-zero pixel
        label in the segmentation image.

        This property is always an `~numpy.ndarray` without units.
        """
        if self.localbkg_width == 0:
            local_bkgs = np.zeros(self.nlabels)
        else:
            sigma_clip = SigmaClip(sigma=3.0, cenfunc='median', maxiters=20)
            bkg_func = SExtractorBackground(sigma_clip=sigma_clip)
            bkg_apers = self._local_background_apertures

            local_bkgs = []
            for aperture in bkg_apers:
                aperture_mask = aperture.to_mask(method='center')
                slc_lg, slc_sm = aperture_mask.get_overlap_slices(
                    self._data.shape)

                data_cutout = self._data[slc_lg].astype(float, copy=True)
                # All non-zero segment labels are masked
                segm_mask_cutout = self._segment_img.data[slc_lg].astype(bool)
                if self._mask is None:
                    mask_cutout = None
                else:
                    mask_cutout = self._mask[slc_lg]
                data_mask_cutout = self._make_cutout_data_mask(data_cutout,
                                                               mask_cutout)
                data_mask_cutout |= segm_mask_cutout

                aperweight_cutout = aperture_mask.data[slc_sm]
                good_mask = (aperweight_cutout > 0) & ~data_mask_cutout

                data_cutout *= aperweight_cutout
                data_values = data_cutout[good_mask]  # 1D array

                # Check not enough unmasked pixels
                if len(data_values) < 10:
                    local_bkgs.append(0.0)
                    continue
                local_bkgs.append(bkg_func(data_values))
            local_bkgs = np.array(local_bkgs)

        local_bkgs[self._all_masked] = np.nan
        return local_bkgs

    @lazyproperty
    @as_scalar
    def local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.
        """
        bkg = self._local_background
        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    def _aperture_to_mask(self, aperture, **kwargs):
        """
        Call ``aperture.to_mask()``, but first check that the aperture
        bounding box is not larger than the input data to prevent
        out-of-memory errors from pathologically large apertures.

        The aperture mask is allocated at the full (unclipped) bounding
        box size by ``to_mask()``, before ``get_overlap_slices`` clips
        it to the data shape. For pathological apertures (e.g., from
        huge Kron radii), this allocation can cause out-of-memory
        issues.

        Returns `None` if the aperture mask would be unreasonably large.
        """
        bbox = aperture.bbox
        # Limit the aperture mask size to prevent OOM errors
        max_size = max(self._data.size, 1_000_000)
        if bbox.shape[0] * bbox.shape[1] > max_size:
            return None
        return aperture.to_mask(**kwargs)

    def _make_aperture_data(self, label, xcentroid, ycentroid, aperture_bbox,
                            local_background, *, make_error=True):
        """
        Make cutouts of data, error, and mask arrays for aperture
        photometry (e.g., circular or Kron).

        Neighboring sources can be included, masked, or corrected based
        on the ``apermask_method`` keyword.
        """
        # Make cutouts of the data based on the aperture bbox
        slc_lg, slc_sm = aperture_bbox.get_overlap_slices(self._data.shape)
        if slc_lg is None:
            return (None,) * 5

        data = self._data[slc_lg].astype(float) - local_background

        mask_cutout = None if self._mask is None else self._mask[slc_lg]
        data_mask = self._make_cutout_data_mask(data, mask_cutout)

        if make_error and self._error is not None:
            error = self._error[slc_lg]
        else:
            error = None

        # Calculate cutout centroid position
        cutout_xycen = (xcentroid - max(0, aperture_bbox.ixmin),
                        ycentroid - max(0, aperture_bbox.iymin))

        # Mask or correct neighboring sources
        if self.apermask_method == 'none':
            mask = data_mask
        else:
            segment_img = self._segment_img.data[slc_lg]
            segm_mask = np.logical_and(segment_img != label,
                                       segment_img != 0)
            if self.apermask_method == 'mask':
                mask = data_mask | segm_mask
            else:
                mask = data_mask

        if self.apermask_method == 'correct':
            data = _mask_to_mirrored_value(data, segm_mask, cutout_xycen,
                                           mask=mask)
            if error is not None:
                error = _mask_to_mirrored_value(error, segm_mask, cutout_xycen,
                                                mask=mask)

        return data, error, mask, cutout_xycen, slc_sm

    def _make_circular_apertures(self, radius):
        """
        Make circular aperture for each source.

        The aperture for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        it will be used for the source centroids.

        Parameters
        ----------
        radius : float, 1D `~numpy.ndarray`
            The radius of the circle in pixels.

        Returns
        -------
        result : list of `~photutils.aperture.CircularAperture`
            A list of `~photutils.aperture.CircularAperture` instances.
            The aperture will be `None` where the source `centroid`
            position is not finite or where the source is completely
            masked.
        """
        radius = np.broadcast_to(radius, len(self._xcentroid))
        if np.any(radius <= 0):
            msg = 'radius must be > 0'
            raise ValueError(msg)

        apertures = []
        for (xcen, ycen, radius_, all_masked) in zip(self._xcentroid,
                                                     self._ycentroid,
                                                     radius,
                                                     self._all_masked,
                                                     strict=True):
            if all_masked or np.any(~np.isfinite((xcen, ycen, radius_))):
                apertures.append(None)
                continue

            apertures.append(CircularAperture((xcen, ycen), r=radius_))

        return apertures

    @as_scalar
    def make_circular_apertures(self, radius):
        """
        Make circular aperture for each source.

        The aperture for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        then its `centroid` values will be used.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        Returns
        -------
        result : `~photutils.aperture.CircularAperture` or \
                 list of `~photutils.aperture.CircularAperture`
            The circular aperture for each source. The aperture will be
            `None` where the source `centroid` position is not finite or
            where the source is completely masked.
        """
        return self._make_circular_apertures(radius)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_circular_apertures(self, radius, ax=None, origin=(0, 0),
                                **kwargs):
        """
        Plot circular apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        then its `centroid` values will be used.

        An aperture will not be plotted for sources where the source
        `centroid` position is not finite or where the source is
        completely masked.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or \
                list of `~matplotlib.patches.Patch`
            The matplotlib patch for each plotted aperture. The patches
            can be used, for example, when adding a plot legend.
        """
        apertures = self._make_circular_apertures(radius)
        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(ax=ax, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def circular_photometry(self, radius, name=None, overwrite=False):
        """
        Perform circular aperture photometry for each source.

        The circular aperture for each source will be centered at
        its `centroid` position. If a ``detection_cat`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

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
        flux, fluxerr : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position is not finite or the source is
            completely masked).
        """
        if radius <= 0:
            msg = 'radius must be > 0'
            raise ValueError(msg)

        apertures = self._make_circular_apertures(radius)
        kwargs = self._apermask_kwargs['circ']
        flux, fluxerr = self._aperture_photometry(apertures,
                                                  desc='circular_photometry',
                                                  **kwargs)

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

    def _make_elliptical_apertures(self, *, scale=6.0):
        """
        Return a list of elliptical apertures based on the scaled
        isophotal shape of the sources.

        If a ``detection_cat`` was input to `SourceCatalog`, then its
        source `centroid` and shape parameters will be used.

        If scale is zero (due to a minimum circular radius set in
        ``kron_params``) then a circular aperture will be returned with
        the minimum circular radius.

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
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked.
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
        for values in zip(xcen, ycen, major_size, minor_size, theta,
                          self._all_masked, strict=True):
            if values[-1] or np.any(~np.isfinite(values[:-1])):
                aperture.append(None)
                continue

            # kron_radius = 0 -> scale = 0 -> major/minor_size = 0
            if values[2] == 0 and values[3] == 0:
                aperture.append(CircularAperture((values[0], values[1]),
                                                 r=self.kron_params[2]))
                continue

            (xcen_, ycen_, major_, minor_, theta_) = values[:-1]
            aperture.append(EllipticalAperture((xcen_, ycen_), major_, minor_,
                                               theta=theta_))

        return aperture

    @lazyproperty
    @use_detcat
    def _measured_kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius, always as an array
        (without units).

        The returned value is the measured Kron radius without applying
        any minimum Kron or circular radius.
        """
        apertures = self._make_elliptical_apertures(scale=6.0)
        cxx = self.cxx.value
        cxy = self.cxy.value
        cyy = self.cyy.value
        if self.isscalar:
            cxx = (cxx,)
            cxy = (cxy,)
            cyy = (cyy,)

        labels = self.labels
        if self.progress_bar:
            desc = 'kron_radius'
            labels = add_progress_bar(labels, desc=desc)

        kron_radius = []
        for (label, aperture, cxx_, cxy_, cyy_) in zip(labels, apertures,
                                                       cxx, cxy, cyy,
                                                       strict=True):
            if aperture is None:
                kron_radius.append(np.nan)
                continue

            xcen, ycen = aperture.positions
            # Use 'center' (whole pixels) to compute Kron radius
            aperture_mask = self._aperture_to_mask(aperture, method='center')
            if aperture_mask is None:
                kron_radius.append(np.nan)
                continue

            # Prepare cutouts of the data based on the aperture size
            # local background explicitly set to zero for SE agreement
            data, _, mask, xycen, slc_sm = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, 0.0, make_error=False)

            xval = np.arange(data.shape[1]) - xycen[0]
            yval = np.arange(data.shape[0]) - xycen[1]
            xx, yy = np.meshgrid(xval, yval)
            rr = np.sqrt(cxx_ * xx**2 + cxy_ * xx * yy + cyy_ * yy**2)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels

            # Ignore RuntimeWarning for invalid data values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                flux_numer = np.sum((aperture_weights * data * rr)[pixel_mask])
                flux_denom = np.sum((aperture_weights * data)[pixel_mask])

            # Set Kron radius to the minimum Kron radius if numerator or
            # denominator is negative
            if flux_numer <= 0 or flux_denom <= 0:
                kron_radius.append(self.kron_params[1])
                continue

            kron_radius.append(flux_numer / flux_denom)

        return np.array(kron_radius)

    @as_scalar
    def _calc_kron_radius(self, kron_params):
        """
        Calculate the *unscaled* first-moment Kron radius, applying any
        minimum Kron or circular radius to the measured Kron radius.

        Returned as a Quantity array or scalar (if self isscalar) with
        pixel units.
        """
        kron_radius = self._measured_kron_radius.copy()

        # Set values exceeding the measurement aperture scale (6.0)
        # to NaN. Such values are unphysical (the Kron radius cannot
        # meaningfully exceed the aperture used to measure it) and are
        # caused by near-cancellation in the denominator of the Kron
        # formula due to outlier pixels or noise.
        max_kron_radius = 6.0
        kron_radius[kron_radius > max_kron_radius] = np.nan

        # Set minimum (unscaled) kron radius
        kron_radius[kron_radius < kron_params[1]] = kron_params[1]

        # Check for minimum circular radius
        if len(kron_params) == 3:
            major_sigma = self.semimajor_sigma.value
            minor_sigma = self.semiminor_sigma.value
            circ_radius = (kron_params[0] * kron_radius
                           * np.sqrt(major_sigma * minor_sigma))
            kron_radius[circ_radius <= kron_params[2]] = 0.0

        return kron_radius << u.pix

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius.

        The *unscaled* first-moment Kron radius is given by:

        .. math::

            r_k = \frac{\sum_{i \in A} \ r_i I_i}{\sum_{i \in A} I_i}

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
        `centroid` and the coefficients are based on image moments
        (`cxx`, `cxy`, and `cyy`).

        The `kron_radius` value is the unscaled moment value. The
        minimum unscaled radius can be set using the second element of
        the `SourceCatalog` ``kron_params`` keyword. If the measured
        unscaled Kron radius exceeds 6.0 (the measurement aperture
        scale factor), ``np.nan`` will be returned. Such values are
        unphysical, typically caused by near-cancellation in the
        denominator of the Kron formula due to outlier pixels or noise.

        If either the numerator or denominator above is less than
        or equal to 0, then the minimum unscaled Kron radius
        (``kron_params[1]``) will be used.

        The Kron aperture is calculated for each source using its shape
        parameters, `kron_radius`, and the ``kron_params`` scaling and
        minimum values input into `SourceCatalog`. The Kron aperture is
        used to compute the Kron photometry.

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_sigma` *
        `semiminor_sigma`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron radius will be set to
        zero and the Kron aperture will be a circle with this minimum
        radius.

        If the source is completely masked, then ``np.nan`` will be
        returned for both the Kron radius and Kron flux (the Kron
        aperture will be `None`).

        If a ``detection_cat`` was input to `SourceCatalog`, then its
        ``kron_radius`` will be returned.

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.
        """
        return self._calc_kron_radius(self.kron_params)

    def _make_kron_apertures(self, kron_params):
        """
        Make Kron apertures for each source, always returned as a list.
        """
        # NOTE: if kron_radius = NaN, scale = NaN and kron_aperture = None
        kron_radius = self._calc_kron_radius(kron_params)
        scale = kron_radius.value * kron_params[0]
        return self._make_elliptical_apertures(scale=scale)

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_aperture(self):
        r"""
        The elliptical (or circular) Kron aperture.

        The Kron aperture is calculated for each source using its shape
        parameters, `kron_radius`, and the ``kron_params`` scaling and
        minimum values input into `SourceCatalog`. The Kron aperture is
        used to compute the Kron photometry.

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_sigma` *
        `semiminor_sigma`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron aperture will be a
        circle with this minimum radius.

        The aperture will be `None` where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.

        If a ``detection_cat`` was input to `SourceCatalog`, then its
        ``kron_aperture`` will be returned.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._make_kron_apertures(self.kron_params)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_kron_apertures(self, kron_params=None):
        """
        Make Kron apertures for each source.

        The aperture for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        then its `centroid` values will be used.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_fluxerr`) in the source catalog. This method should
        be used only to explore alternative ``kron_params`` with a
        detection image.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats or `None`, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_sigma`
            * `semiminor_sigma`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius. If `None`, then the ``kron_params`` input into
            `SourceCatalog` will be used (the apertures will be the same
            as those in `kron_aperture`).

        Returns
        -------
        result : `~photutils.aperture.PixelAperture` \
                 or list of `~photutils.aperture.PixelAperture`
            The Kron apertures for each source. Each aperture will
            either be a `~photutils.aperture.EllipticalAperture`,
            `~photutils.aperture.CircularAperture`, or `None`. The
            aperture will be `None` where the source `centroid` position
            or elliptical shape parameters are not finite or where the
            source is completely masked.
        """
        if kron_params is None:
            return self.kron_aperture
        return self._make_kron_apertures(kron_params)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_kron_apertures(self, kron_params=None, ax=None, origin=(0, 0),
                            **kwargs):
        """
        Plot Kron apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        then its `centroid` values will be used.

        An aperture will not be plotted for sources where the source
        `centroid` position or elliptical shape parameters are not
        finite or where the source is completely masked.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_fluxerr`) in the source catalog. This method should be
        used only to visualize/explore alternative ``kron_params`` with
        a detection image.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats or `None`, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_sigma`
            * `semiminor_sigma`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius. If `None`, then the ``kron_params`` input into
            `SourceCatalog` will be used (the apertures will be the same
            as those in `kron_aperture`).

        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : list of `~matplotlib.patches.Patch`
            A list of matplotlib patches for the plotted aperture. The
            patches can be used, for example, when adding a plot legend.
        """
        if kron_params is None:
            apertures = self.kron_aperture
            if self.isscalar:
                apertures = (apertures,)
        else:
            apertures = self._make_kron_apertures(kron_params)

        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(ax=ax, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    def _aperture_photometry(self, apertures, *, desc='', **kwargs):
        """
        Perform aperture photometry on cutouts of the data and optional
        error arrays.

        The appropriate ``apermask_method`` is applied to the cutouts to
        handle neighboring sources.

        Parameters
        ----------
        apertures : list of `PixelAperture`
            A list of the apertures.

        desc : str, optional
            The description displayed before the progress bar.

        **kwargs : dict, optional
            Additional keyword arguments passed to the aperture
            ``to_mask`` method.

        Returns
        -------
        flux, fluxerr : 1D `~numpy.ndaray`
            The flux and flux error arrays.
        """
        labels = self.labels
        if self.progress_bar:
            labels = add_progress_bar(labels, desc=desc)

        flux = []
        fluxerr = []
        for label, aperture, bkg in zip(labels, apertures,
                                        self._local_background, strict=True):
            # Return NaN for completely masked sources or sources where
            # the centroid is not finite
            if aperture is None:
                flux.append(np.nan)
                fluxerr.append(np.nan)
                continue

            xcen, ycen = aperture.positions
            aperture_mask = self._aperture_to_mask(aperture, **kwargs)
            if aperture_mask is None:
                flux.append(np.nan)
                fluxerr.append(np.nan)
                continue

            # Prepare cutouts of the data based on the aperture size
            data, error, mask, _, slc_sm = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels
            # Ignore RuntimeWarning for invalid data or error values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                values = (aperture_weights * data)[pixel_mask]
                flux_ = np.nan if values.shape == (0,) else np.sum(values)
                flux.append(flux_)

                if error is None:
                    fluxerr_ = np.nan
                else:
                    values = (aperture_weights * error**2)[pixel_mask]
                    if values.shape == (0,):
                        fluxerr_ = np.nan
                    else:
                        fluxerr_ = np.sqrt(np.sum(values))
                fluxerr.append(fluxerr_)

        flux = np.array(flux)
        fluxerr = np.array(fluxerr)

        return flux, fluxerr

    def _calc_kron_photometry(self, *, kron_params=None):
        """
        Calculate the flux and flux error in the Kron aperture (without
        units).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned.

        If ``detection_cat`` is input, then its `centroid` values will
        be used.

        Returns
        -------
        kron_flux, kron_fluxerr : tuple of `~numpy.ndarray`
            The Kron flux and flux error.
        """
        if kron_params is None:
            kron_aperture = self.kron_aperture
            if self.isscalar:
                kron_aperture = (kron_aperture,)
        else:
            kron_params = self._validate_kron_params(kron_params)
            kron_aperture = self._make_kron_apertures(kron_params)

        kwargs = self._apermask_kwargs['kron']
        flux, fluxerr = self._aperture_photometry(kron_aperture,
                                                  desc='kron_photometry',
                                                  **kwargs)

        return flux, fluxerr

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def kron_photometry(self, kron_params, name=None, overwrite=False):
        """
        Perform photometry for each source using an elliptical Kron
        aperture.

        This method can be used to calculate the Kron photometry using
        alternate ``kron_params`` (e.g., different scalings of the Kron
        radius).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_sigma`
            * `semiminor_sigma`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
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
        flux, fluxerr : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked).
        """
        kron_flux, kron_fluxerr = self._calc_kron_photometry(
            kron_params=kron_params)
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
    def _kron_photometry(self):
        """
        The flux and flux error in the Kron aperture (without units).

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        return np.transpose(self._calc_kron_photometry(kron_params=None))

    @lazyproperty
    @as_scalar
    def kron_flux(self):
        """
        The flux in the Kron aperture.

        See the `SourceCatalog` ``apermask_method`` keyword for options
        to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_flux = self._kron_photometry[:, 0]
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

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_fluxerr = self._kron_photometry[:, 1]
        if self._data_unit is not None:
            kron_fluxerr <<= self._data_unit
        return kron_fluxerr

    @lazyproperty
    @use_detcat
    def _max_circular_kron_radius(self):
        """
        The maximum circular Kron radius used as the upper limit of
        fluxfrac_radius.
        """
        semimajor_sig = self.semimajor_sigma.value
        kron_radius = self.kron_radius.value
        radius = semimajor_sig * kron_radius * self.kron_params[0]
        mask = radius == 0
        if np.any(mask):
            radius[mask] = self.kron_params[2]
        if self.isscalar:
            radius = np.array([radius])
        return radius

    @staticmethod
    def _fluxfrac_radius_fcn(radius, data, mask, aperture, normflux, kwargs):
        """
        Function whose root is found to compute the fluxfrac_radius.
        """
        aperture.r = radius
        flux, _ = aperture.do_photometry(data, mask=mask, **kwargs)
        return 1.0 - (flux[0] / normflux)

    @lazyproperty
    @use_detcat
    def _fluxfrac_optimizer_args(self):
        kron_flux = self._kron_photometry[:, 0]  # unitless
        max_radius = self._max_circular_kron_radius
        kwargs = self._apermask_kwargs['fluxfrac']

        labels = self.labels
        if self.progress_bar:
            desc = 'fluxfrac_radius prep'
            labels = add_progress_bar(labels, desc=desc)

        args = []
        for label, xcen, ycen, kronflux, bkg, max_radius_ in zip(
                labels, self._xcentroid, self._ycentroid,
                kron_flux, self._local_background, max_radius, strict=True):

            if (np.any(~np.isfinite((xcen, ycen, kronflux, max_radius_)))
                    or kronflux == 0):
                args.append(None)
                continue

            aperture = CircularAperture((xcen, ycen), r=max_radius_)
            aperture_mask = self._aperture_to_mask(aperture, **kwargs)
            if aperture_mask is None:
                args.append(None)
                continue

            # Prepare cutouts of the data based on the maximum aperture size
            data, _, mask, xycen, _ = self._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg,
                make_error=False)

            aperture.positions = xycen
            args.append([data, mask, aperture, kronflux, kwargs, max_radius_])

        return args

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
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
            or where the Kron flux is zero or non-finite.
        """
        if fluxfrac <= 0 or fluxfrac > 1:
            msg = 'fluxfrac must be > 0 and <= 1'
            raise ValueError(msg)

        # Return cached result if available
        if fluxfrac in self._fluxfrac_cache:
            result = self._fluxfrac_cache[fluxfrac]
            if name is not None:
                self.add_extra_property(name, result, overwrite=overwrite)
            return result

        args = self._fluxfrac_optimizer_args
        if self.progress_bar:
            desc = 'fluxfrac_radius'
            args = add_progress_bar(args, desc=desc)

        radius = []
        for fluxfrac_args in args:
            if fluxfrac_args is None:
                radius.append(np.nan)
                continue

            max_radius = fluxfrac_args[-1]
            args = fluxfrac_args[:-1]
            args[3] *= fluxfrac
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
            max_radius_delta = 0.1 * max_radius
            while max_radius > min_radius and found is False:
                try:
                    bracket = [min_radius, max_radius]
                    result = root_scalar(self._fluxfrac_radius_fcn, args=args,
                                         bracket=bracket, method='brentq')
                    result = result.root
                    found = True
                except ValueError:
                    # ValueError is raised if the bracket points do not
                    # have different signs
                    max_radius -= max_radius_delta

            # No solution found between min_radius and max_radius
            if found is False:
                result = np.nan

            radius.append(result)

        result = np.array(radius) << u.pix
        self._fluxfrac_cache[fluxfrac] = result

        if name is not None:
            self.add_extra_property(name, result, overwrite=overwrite)

        return result

    @as_scalar
    def make_cutouts(self, shape, *, array=None, mode='partial',
                     fill_value=np.nan):
        """
        Make cutout arrays for each source.

        The cutout for each source will be centered at its `centroid`
        position. If a ``detection_cat`` was input to `SourceCatalog`,
        then its `centroid` values will be used.

        Parameters
        ----------
        shape : 2-tuple
            The cutout shape along each axis in ``(ny, nx)`` order.

        array : `None` or 2D `~numpy.ndarray`
            A 2D array with the same shape as the ``data`` array input
            to `~photutils.segmentation.SourceCatalog`. If `None` then
            the ``data`` array will be used. If any cutout arrays
            are not fully contained within the ``array`` array and
            ``mode='partial'`` with ``fill_value=np.nan``, then the
            input ``array`` must have a float data type.

        mode : {'partial', 'trim'}, optional
            The mode used for extracting the cutout array. In
            ``'partial'`` mode, positions in the cutout array that
            do not overlap with the large array will be filled with
            ``fill_value``. In ``'trim'`` mode, only the overlapping
            elements are returned, thus the resulting small array may be
            smaller than the requested ``shape``.

        fill_value : number, optional
            If ``mode='partial'``, the value to fill pixels in the
            extracted cutout array that do not overlap with the input
            ``array_large``. ``fill_value`` will be changed to have
            the same ``dtype`` as the ``array_large`` array, with
            one exception. If ``array_large`` has integer type and
            ``fill_value`` is ``np.nan``, then a `ValueError` will be
            raised.

        Returns
        -------
        cutouts : `~photutils.utils.CutoutImage` \
                  or list of `~photutils.utils.CutoutImage`
            The `~photutils.utils.CutoutImage` for each source. The
            cutout will be `None` where the source `centroid` position
            is not finite or where the source is completely masked.
        """
        if array is None:
            array = self._data
        elif array.shape != self._data.shape:
            msg = 'array must have the same shape as data'
            raise ValueError(msg)

        if mode not in ('partial', 'trim'):
            msg = 'mode must be "partial" or "trim"'
            raise ValueError(msg)

        cutouts = []
        for (xcen, ycen, all_masked) in zip(self._xcentroid,
                                            self._ycentroid,
                                            self._all_masked, strict=True):

            if all_masked or np.any(~np.isfinite((xcen, ycen))):
                cutouts.append(None)
                continue

            cutouts.append(CutoutImage(array, (ycen, xcen), shape,
                                       mode=mode, fill_value=fill_value))

        return cutouts
