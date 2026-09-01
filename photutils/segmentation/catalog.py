# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating the properties of sources defined by a
segmentation image.
"""

import functools
import inspect
import warnings
from copy import deepcopy
from functools import cached_property

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from scipy.ndimage import map_coordinates

from photutils.aperture import (BoundingBox, CircularAperture,
                                EllipticalAperture, RectangularAnnulus)
from photutils.aperture._batch_photometry import (FLAG_COL_BBOX_CLIPPED,
                                                  FLAG_COL_MASKED,
                                                  FLAG_COL_N_PIXELS,
                                                  FLAG_COL_NONFINITE_DATA,
                                                  FLAG_COL_SEG,
                                                  FLAG_COL_SEG_MASKED,
                                                  FLAG_COL_UNCORRECTED,
                                                  FLAG_COL_UNCORRECTED_MASKED,
                                                  FLAG_COL_VALID, SHAPE_CIRCLE,
                                                  SHAPE_ELLIPSE,
                                                  batch_aperture_sums)
from photutils.background import SExtractorBackground
from photutils.segmentation._batch_catalog import (batch_central_moments,
                                                   batch_centroid_win,
                                                   batch_flux_radius_prepare,
                                                   batch_flux_radius_solve,
                                                   batch_kron_radius,
                                                   batch_moment_err,
                                                   batch_perimeter,
                                                   batch_quad_boxes,
                                                   batch_raw_moments,
                                                   batch_segment_gather)
from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.flags import (SEGMENTATION_FLAGS,
                                          decode_segmentation_flags)
from photutils.utils._deprecation import (_get_future_column_names,
                                          create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._flags import update_flag_docstring
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import validate_table_columns
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._wcs_helpers import compute_pixel_to_sky_jacobians
from photutils.utils.cutouts import CutoutImage

__all__ = ['SourceCatalog']


# Segmentation masking codes used by the batch Cython drivers
_SEG_METHOD_CODES = {'none': 0, 'mask': 1, 'correct': 3}


def _batch_gini(packed, offsets, counts):
    """
    Compute the Gini coefficient of the packed pixel values of many
    sources.

    This is the vectorized form of `photutils.morphology.gini` applied
    to each source's unmasked (finite) absolute pixel values: NaN for a
    source with no pixels, 0.0 for a single pixel or a zero mean, and
    otherwise the Lotz et al. (2004) sum over the sorted values.

    Parameters
    ----------
    packed : 1D `~numpy.ndarray`
        The packed pixel values of all sources (see
        ``batch_segment_gather``), with a single placeholder value for
        a source with no pixels.

    offsets : 1D `~numpy.ndarray`
        The start offset of each source in ``packed`` (length
        ``n_sources + 1``).

    counts : 1D `~numpy.ndarray`
        The number of pixels of each source.

    Returns
    -------
    result : 1D `~numpy.ndarray`
        The Gini coefficient of each source.
    """
    sizes = np.diff(offsets)
    starts = offsets[:-1]
    # Sort the absolute values within each source in place (a single
    # sort per source is much cheaper than one lexsort over all of the
    # values); the placeholder values of empty sources are harmless
    # because their results are set to NaN below
    values = np.abs(np.nan_to_num(packed))
    for start, stop in zip(starts, offsets[1:], strict=True):
        values[start:stop].sort()
    # 1-based rank of each value within its source
    rank = np.arange(len(values)) - np.repeat(starts, sizes) + 1
    n_pixels = np.repeat(counts, sizes)
    kernel = (2.0 * rank - n_pixels - 1) * values

    # Ignore RuntimeWarning from the empty and single-pixel sources
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.add.reduceat(values, starts) / counts
        normalization = mean * counts * (counts - 1)
        result = np.add.reduceat(kernel, starts) / normalization
    result[normalization == 0.0] = 0.0
    result[counts == 1] = 0.0
    result[counts == 0] = np.nan
    return result


# Remove in 4.0
_DEPRECATED_ATTRIBUTES = {
    'add_extra_property': 'add_property',
    'apermask_method': 'aperture_mask_method',
    'background': 'background_cutout',
    'background_ma': 'background_cutout_masked',
    'convdata': 'conv_data_cutout',
    'convdata_ma': 'conv_data_cutout_masked',
    'covar_sigx2': 'covariance_xx',
    'covar_sigxy': 'covariance_xy',
    'covar_sigy2': 'covariance_yy',
    'cutout_maxval_index': 'cutout_max_value_index',
    'cutout_minval_index': 'cutout_min_value_index',
    'cxx': 'ellipse_cxx',
    'cxy': 'ellipse_cxy',
    'cyy': 'ellipse_cyy',
    'data': 'data_cutout',
    'data_ma': 'data_cutout_masked',
    'error': 'error_cutout',
    'error_ma': 'error_cutout_masked',
    'extra_properties': 'custom_properties',
    'fluxfrac_radius': 'flux_radius',
    'get_label': 'select_label',
    'get_labels': 'select_labels',
    'kron_fluxerr': 'kron_flux_err',
    'localbkg_width': 'local_bkg_width',
    'maxval_index': 'max_value_index',
    'maxval_xindex': 'max_value_xindex',
    'maxval_yindex': 'max_value_yindex',
    'minval_index': 'min_value_index',
    'minval_xindex': 'min_value_xindex',
    'minval_yindex': 'min_value_yindex',
    'nlabels': 'n_labels',
    'remove_extra_properties': 'remove_properties',
    'remove_extra_property': 'remove_property',
    'rename_extra_property': 'rename_property',
    'segment': 'segment_cutout',
    'segment_fluxerr': 'segment_flux_err',
    'segment_ma': 'segment_cutout_masked',
    'semimajor_sigma': 'semimajor_axis',
    'semiminor_sigma': 'semiminor_axis',
    'xcentroid': 'x_centroid',
    'xcentroid_quad': 'x_centroid_quad',
    'xcentroid_win': 'x_centroid_win',
    'ycentroid': 'y_centroid',
    'ycentroid_quad': 'y_centroid_quad',
    'ycentroid_win': 'y_centroid_win',
}

_DEPRECATED_META_KEYS = {
    'localbkg_width': 'local_bkg_width',
    'apermask_method': 'aperture_mask_method',
}

# Public non-underscore attributes that must never be collapsed to a
# scalar for a single-source (scalar) catalog (see __getattribute__).
_SCALAR_EXCLUDE = frozenset({'isscalar', 'n_labels', 'labels', 'properties',
                             'custom_properties', 'default_columns',
                             'kron_params'})


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
        if self._detection_catalog is None:
            return method(self, *args, **kwargs)

        return getattr(self._detection_catalog, method.__name__)

    return _use_detcat


def _update_flags_docstring(func):
    """
    Decorator to insert the segmentation flag descriptions into a
    docstring.

    The ``<flag_descriptions>`` placeholder in the function docstring is
    replaced with a bullet list generated from ``SEGMENTATION_FLAGS``.

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function with an updated docstring.
    """
    return update_flag_docstring(func, SEGMENTATION_FLAGS, indent=8)


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

    segmentation_image : `~photutils.segmentation.SegmentationImage`
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

    background : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the
        input ``data``. ``background`` must be a 2D image with
        the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity` array then ``background`` must
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
        ignored if ``detection_catalog`` is input.

    local_bkg_width : int, optional
        The width of the rectangular annulus used to compute a
        local background around each source. If zero, then no local
        background subtraction is performed. The local background
        affects the ``min_value``, ``max_value``, ``segment_flux``,
        ``kron_flux``, and ``flux_radius`` properties. It is also used
        when calculating circular and Kron aperture photometry (i.e.,
        `circular_photometry` and `kron_photometry`). It does not affect
        the moment-based morphological properties of the source.

    aperture_mask_method : {'correct', 'mask', 'none'}, optional
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

        This keyword will be ignored if ``detection_catalog`` is
        input. In that case, the ``aperture_mask_method`` set in the
        ``detection_catalog`` will be used.

    kron_params : tuple of 2 or 3 floats, optional
        A list of parameters used to determine the Kron aperture.
        The first item is the scaling parameter of the unscaled Kron
        radius and the second item represents the minimum value for
        the unscaled Kron radius in pixels. The optional third item is
        the minimum circular radius in pixels. If ``kron_params[0]``
        * `kron_radius` * sqrt(`semimajor_axis` * `semiminor_axis`)
        is less than or equal to this radius, then the Kron aperture
        will be a circle with this minimum radius. This keyword will be
        ignored if ``detection_catalog`` is input.

    detection_catalog : `SourceCatalog`, optional
        A `SourceCatalog` object for the detection image. The
        segmentation image used to create the detection catalog must be
        the same one input to ``segmentation_image``. If input, then
        the detection catalog source centroids and morphological/shape
        properties will be returned instead of calculating them from
        the input ``data``. The detection catalog centroids and shape
        properties will also be used to perform aperture photometry
        (i.e., circular and Kron). If ``detection_catalog`` is
        input, then the input ``wcs``, ``aperture_mask_method``, and
        ``kron_params`` keywords will be ignored. This keyword affects
        `circular_photometry` (including returned apertures), all Kron
        parameters (Kron radius, flux, flux errors, apertures, and
        custom `kron_photometry`), and `flux_radius` (which is based on
        the Kron flux).

    progress_bar : bool, optional
        Whether to display a progress bar when calculating properties.

        .. deprecated:: 3.1
           This keyword argument is deprecated and will be removed in a
           future version. All property calculations are now computed
           for all sources at once in compiled code, so no progress bar
           is displayed and this keyword has no effect.

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
    `~photutils.segmentation.SourceCatalog.segment_flux_err` is simply
    the quadrature sum of the pixel-wise total errors over the
    unmasked pixels within the source segment:

    .. math::

        \\Delta F = \\sqrt{\\sum_{i \\in S}
            \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.segmentation.SourceCatalog.segment_flux_err`,
    :math:`S` are the unmasked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    For the aperture-based fluxes (e.g.,
    `~photutils.segmentation.SourceCatalog.kron_flux` and
    `~photutils.segmentation.SourceCatalog.circular_photometry`), the
    flux is the sum of the pixel values weighted by their aperture
    overlap fractions. The flux error therefore weights each pixel
    variance by the squared overlap fraction: :math:`\\Delta F =
    \\sqrt{\\sum_{i \\in A} w_i^2 \\sigma_{\\mathrm{tot}, i}^2}`, where
    :math:`w_i` are the aperture overlap fractions and :math:`A` are the
    unmasked pixels within the aperture.

    Custom errors for source segments can be calculated using the
    `~photutils.segmentation.SourceCatalog.error_cutout_masked` and
    `~photutils.segmentation.SourceCatalog.background_cutout_masked`
    properties, which are 2D `~numpy.ma.MaskedArray` cutout versions of
    the input ``error`` and ``background`` arrays. The mask is `True`
    for pixels outside the source segment, masked pixels from the
    ``mask`` input, or any non-finite ``data`` values (NaN and inf).

    **Scalar vs. Multi-source Catalogs**

    A `SourceCatalog` can represent a single source or multiple
    sources. Most properties adapt their return type accordingly: for
    a multi-source catalog, properties return arrays or lists (one
    element per source); for a single-source (scalar) catalog, the
    same properties return a scalar value or a single object. For
    example, `kron_aperture` returns a list of aperture objects for a
    multi-source catalog, but a single aperture object for a scalar
    catalog. Similarly, `data_cutout` returns a list of 2D cutout arrays
    for a multi-source catalog, but a single 2D array for a scalar
    catalog. A scalar catalog is created when the a multi-source catalog
    is indexed to select a single source.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """

    @deprecated_renamed_argument('segment_img', 'segmentation_image', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('localbkg_width', 'local_bkg_width',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('apermask_method', 'aperture_mask_method',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('detection_cat', 'detection_catalog', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('progress_bar', None, '3.1', until='4.0')
    def __init__(self, data, segmentation_image, *, convolved_data=None,
                 error=None, mask=None, background=None, wcs=None,
                 local_bkg_width=0, aperture_mask_method='correct',
                 kron_params=(2.5, 1.4, 0.0), detection_catalog=None,
                 progress_bar=False):

        inputs = (data, convolved_data, error, background)
        names = ('data', 'convolved_data', 'error', 'background')
        inputs, unit = process_quantities(inputs, names)
        (data, convolved_data, error, background) = inputs

        self._data_unit = unit
        self._data = self._validate_array(data, 'data', shape=False)
        self._convolved_data = self._validate_array(convolved_data,
                                                    'convolved_data')
        self._segmentation_image = self._validate_segmentation_image(
            segmentation_image)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._background = self._validate_array(background, 'background')
        self.wcs = wcs
        self.local_bkg_width = self._validate_local_bkg_width(
            local_bkg_width)
        self.aperture_mask_method = self._validate_aperture_mask_method(
            aperture_mask_method)
        self.kron_params = self._validate_kron_params(kron_params)
        self.progress_bar = progress_bar

        # Needed for ordering and isscalar
        # NOTE: calculate slices before labels for performance.
        #       _labels is initially always a non-scalar array, but
        #       it can become a numpy scalar after indexing/slicing.
        self._slices = self._segmentation_image.slices
        self._labels = self._segmentation_image.labels

        if self._labels.shape == (0,):
            msg = 'segmentation_image must have at least one non-zero label'
            raise ValueError(msg)

        self._detection_catalog = self._validate_detection_catalog(
            detection_catalog)
        attrs = ('wcs', 'aperture_mask_method', 'kron_params')
        if self._detection_catalog is not None:
            for attr in attrs:
                setattr(self, attr, getattr(self._detection_catalog, attr))

        if convolved_data is None:
            self._convolved_data = self._data

        self._aperture_mask_kwargs = {
            'circ': {'method': 'exact'},
            'kron': {'method': 'exact'},
            'flux_radius': {'method': 'exact'},
            'cen_win': {'method': 'center'},
        }

        self.default_columns = ['label', 'x_centroid', 'y_centroid',
                                'sky_centroid', 'bbox_xmin', 'bbox_xmax',
                                'bbox_ymin', 'bbox_ymax', 'area',
                                'semimajor_axis', 'semiminor_axis',
                                'orientation', 'eccentricity', 'min_value',
                                'max_value', 'segment_flux',
                                'segment_flux_err', 'kron_flux',
                                'kron_flux_err', 'flags']

        self._custom_properties = []
        self._flux_radius_cache = {}
        # The full-image arrays used by the batch Cython drivers, filled
        # lazily by _get_batch_arrays. The dict is shared by reference
        # with sliced catalogs (see __getitem__), so whichever catalog
        # builds the arrays first populates it for all of them.
        self._batch_arrays_cache = {}
        self.meta = _get_meta()
        self._update_meta()

    def _validate_segmentation_image(self, segmentation_image):
        if not isinstance(segmentation_image, SegmentationImage):
            msg = 'segmentation_image must be a SegmentationImage'
            raise TypeError(msg)
        if segmentation_image.shape != self._data.shape:
            msg = 'segmentation_image and data must have the same shape'
            raise ValueError(msg)
        return segmentation_image

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
    def _validate_local_bkg_width(local_bkg_width):
        if local_bkg_width < 0:
            msg = 'local_bkg_width must be >= 0'
            raise ValueError(msg)
        local_bkg_width_int = int(local_bkg_width)
        if local_bkg_width_int != local_bkg_width:
            msg = 'local_bkg_width must be an integer'
            raise ValueError(msg)
        return local_bkg_width_int

    @staticmethod
    def _validate_aperture_mask_method(aperture_mask_method):
        if aperture_mask_method not in ('none', 'mask', 'correct'):
            msg = 'Invalid aperture_mask_method value'
            raise ValueError(msg)
        return aperture_mask_method

    @staticmethod
    def _validate_kron_params(kron_params):
        if np.ndim(kron_params) != 1:
            msg = 'kron_params must be 1D'
            raise ValueError(msg)
        n_params = len(kron_params)
        if n_params not in (2, 3):
            msg = 'kron_params must have 2 or 3 elements'
            raise ValueError(msg)
        if kron_params[0] <= 0:
            msg = 'kron_params[0] must be > 0'
            raise ValueError(msg)
        if kron_params[1] <= 0:
            msg = 'kron_params[1] must be > 0'
            raise ValueError(msg)
        if n_params == 3 and kron_params[2] < 0:
            msg = 'kron_params[2] must be >= 0'
            raise ValueError(msg)
        return tuple(kron_params)

    def _validate_detection_catalog(self, detection_catalog):
        if detection_catalog is None:
            return None

        if not isinstance(detection_catalog, SourceCatalog):
            msg = 'detection_catalog must be a SourceCatalog instance'
            raise TypeError(msg)
        if not np.array_equal(detection_catalog._segmentation_image,
                              self._segmentation_image):
            msg = ('detection_catalog must have same segmentation_image '
                   'as the input segmentation_image')
            raise ValueError(msg)
        return detection_catalog

    def _update_meta(self):
        meta_values = {}
        attrs = ('local_bkg_width', 'aperture_mask_method', 'kron_params')
        for attr in attrs:
            meta_values[attr] = getattr(self, attr)

        if not _get_future_column_names():
            for old_name, new_name in _DEPRECATED_META_KEYS.items():
                if new_name in meta_values:
                    meta_values[old_name] = meta_values[new_name]

        self.meta.update(meta_values)

    def _set_semode(self):
        # SE emulation
        self._aperture_mask_kwargs = {
            'circ': {'method': 'subpixel', 'subpixels': 5},
            'kron': {'method': 'center'},
            'flux_radius': {'method': 'subpixel', 'subpixels': 5},
            'cen_win': {'method': 'subpixel', 'subpixels': 11},
        }

    @property
    def properties(self):
        """
        A list of built-in source properties.
        """
        cached_properties = [name for name in self._cached_properties
                             if not name.startswith('_')]
        cached_properties.remove('isscalar')
        cached_properties.remove('n_labels')
        cached_properties.extend(['label', 'labels', 'slices'])
        cached_properties.sort()
        return cached_properties

    @property
    def _cached_properties(self):
        """
        A list of all class cached properties (even in superclasses).

        The result is cached on the class to avoid repeated
        introspection via `inspect.getmembers`.
        """
        cls = self.__class__
        attr = '_cached_properties_cache'
        # Subclasses get their own cached-property list
        if attr not in cls.__dict__:
            def is_cached_property(obj):
                return isinstance(obj, cached_property)

            setattr(cls, attr,
                    [i[0] for i in inspect.getmembers(
                        cls, predicate=is_cached_property)])
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
        init_attr = ('_data', '_segmentation_image', '_error', '_mask',
                     '_background', 'wcs', '_data_unit', '_convolved_data',
                     'local_bkg_width', 'aperture_mask_method',
                     'kron_params', 'progress_bar', '_batch_arrays_cache')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # Copy mutable containers so that mutating the sliced catalog
        # (e.g., via add_property) does not also mutate the parent
        # catalog (and vice versa).
        newcls.default_columns = self.default_columns.copy()
        newcls._custom_properties = self._custom_properties.copy()
        newcls.meta = self.meta.copy()
        newcls._aperture_mask_kwargs = {
            key: value.copy()
            for key, value in self._aperture_mask_kwargs.items()}

        # _labels determines ordering and isscalar
        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        # Need to slice detection_catalog, if input
        attr = '_detection_catalog'
        if getattr(self, attr) is None:
            setattr(newcls, attr, None)
        else:
            setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_slices'
        value = self._index_object_list(getattr(self, attr), index)
        setattr(newcls, attr, value)

        # Slice the flux_radius cache values
        newcls._flux_radius_cache = {key: value[index]
                                     for key, value
                                     in self._flux_radius_cache.items()}

        # Evaluated cached-property objects and extra properties
        keys = (set(self.__dict__.keys())
                & (set(self._cached_properties)
                   | set(self._custom_properties)))
        for key in keys:
            value = self.__dict__[key]

            # Do not insert attributes that are always scalar (e.g.,
            # isscalar, n_labels), i.e., not an array/list for each
            # source
            if np.isscalar(value):
                continue

            try:
                # Keep private ('_'-prefixed) cached properties as length-1
                # iterables. Such properties are never collapsed by
                # __getattribute__ and internal code relies on them
                # always being iterable.
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
            fmt = [f'Length: {self.n_labels}', f'labels: {self.labels}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            msg = f'Scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.n_labels

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def __getattribute__(self, name):
        # Centralized scalar collapse: for a scalar (single-source)
        # catalog, public properties that return a length-1
        # array/list/tuple (or a scalar SkyCoord) are returned as a
        # scalar value. Private ('_'-prefixed) attributes are always
        # returned unchanged so that internal code can rely on iterable
        # values.
        value = super().__getattribute__(name)
        if (not name.startswith('_')
                and name not in _SCALAR_EXCLUDE
                and isinstance(value, (np.ndarray, SkyCoord, list, tuple))
                and self.isscalar):
            try:
                if len(value) == 1:
                    return value[0]
            except TypeError:  # value has no len
                pass
        return value

    def _make_scalar(self, value):
        """
        Return a scalar value for a scalar catalog if ``value`` is a
        length-1 sequence, otherwise return ``value`` unchanged.

        This is the method-call analog of the scalar collapse performed
        for attribute access by `__getattribute__`.
        """
        try:
            if self.isscalar and len(value) == 1:
                return value[0]
        except TypeError:  # value has no len
            pass
        return value

    def _array(self, name):
        """
        Return the per-source attribute ``name`` with a leading (source)
        axis, so that internal code can always index along the source
        axis without repeating ``if self.isscalar`` checks.

        For a non-scalar (multi-source) catalog this simply returns the
        attribute unchanged. For a scalar (single-source) catalog the
        value is first read through the normal attribute access, which
        applies the scalar collapse performed by `__getattribute__`
        (e.g., ``moments`` becomes a ``(4, 4)`` array and ``centroid`` a
        length-2 array). A single leading length-1 axis is then restored
        (``ndarray`` values are reshaped, while other objects are
        wrapped in a length-1 list).

        Reading the collapsed value rather than the raw stored value
        is important. A per-source array can be stored either with
        its leading length-1 axis intact (when computed directly for
        a scalar catalog) or with that axis already removed (when a
        multi-source value is cached and then sliced to a scalar). Going
        through the scalar collapse normalizes both cases to the same
        shape, so this method is robust regardless of how the value was
        produced.
        """
        value = getattr(self, name)
        if not self.isscalar:
            return value
        if isinstance(value, np.ndarray):
            return value[np.newaxis, ...]
        return [value]

    def _get_batch_arrays(self):
        """
        Return the cached C-contiguous full-image arrays used by the
        batch Cython drivers.

        The dict holds float64 ``data`` and ``error`` arrays, a uint8
        ``mask`` plane (bit 1 = input mask, bit 2 = non-finite data),
        and an intp ``segm`` array. The dict itself is created in
        ``__init__`` and shared by reference with sliced catalogs (see
        ``__getitem__``), so the arrays are built once no matter which
        of the parent or sliced catalogs first needs them.

        Returns
        -------
        result : dict
            A dict with keys ``'data'``, ``'error'``, ``'mask'``, and
            ``'segm'``. The ``'error'`` value is `None` if no error
            array was input. The ``'convdata'`` and ``'background'``
            keys are added lazily by ``_get_batch_convdata`` and
            ``_get_batch_background``.
        """
        arrays = self._batch_arrays_cache
        if 'data' not in arrays:
            data = np.ascontiguousarray(self._data, dtype=np.float64)
            error = None
            if self._error is not None:
                error = np.ascontiguousarray(self._error, dtype=np.float64)
            mask_plane = np.zeros(data.shape, dtype=np.uint8)
            if self._mask is not None:
                mask_plane[self._mask] |= 1
            mask_plane[~np.isfinite(data)] |= 2
            segm = np.ascontiguousarray(
                self._segmentation_image.data, dtype=np.intp)
            arrays.update(data=data, error=error, mask=mask_plane,
                          segm=segm)
        return arrays

    def _get_batch_convdata(self):
        """
        Return the cached C-contiguous float64 convolved-data array
        used by the batch moment kernels.
        """
        arrays = self._get_batch_arrays()
        if 'convdata' not in arrays:
            arrays['convdata'] = np.ascontiguousarray(
                self._convolved_data, dtype=np.float64)
        return arrays['convdata']

    def _get_batch_background(self):
        """
        Return the cached C-contiguous float64 background array used by
        the batch segment gather.
        """
        arrays = self._get_batch_arrays()
        if 'background' not in arrays:
            arrays['background'] = np.ascontiguousarray(
                self._background, dtype=np.float64)
        return arrays['background']

    def _get_batch_bboxes(self):
        """
        Return the per-source segment bounding boxes as
        ``(iymin, iymax, ixmin, ixmax)`` intp arrays.
        """
        slices = self._slices_iter
        iymin = np.array([slc[0].start for slc in slices],
                         dtype=np.intp)
        iymax = np.array([slc[0].stop for slc in slices],
                         dtype=np.intp)
        ixmin = np.array([slc[1].start for slc in slices],
                         dtype=np.intp)
        ixmax = np.array([slc[1].stop for slc in slices],
                         dtype=np.intp)
        return iymin, iymax, ixmin, ixmax

    @cached_property
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
    def custom_properties(self):
        """
        A list of the user-defined source properties.
        """
        return self._custom_properties

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def add_property(self, name, value, overwrite=False):
        """
        Add a user-defined property as a new attribute.

        For example, the property ``name`` can then be included in the
        `to_table` ``columns`` keyword list to output the results in the
        table.

        The complete list of user-defined properties is stored in the
        `custom_properties` attribute.

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
        # dir() includes all class-level attributes (properties, cached
        # properties, and methods), so built-in methods (e.g., to_table)
        # cannot be clobbered even with overwrite=True
        internal_attributes = ((set(self.__dict__.keys())
                                | set(dir(self.__class__)))
                               - set(self.custom_properties))
        if name in internal_attributes:
            msg = f'{name} cannot be set because it is a built-in attribute'
            raise ValueError(msg)

        if not overwrite:
            if hasattr(self, name):
                msg = (f'{name} already exists as an attribute. Set '
                       'overwrite=True to overwrite an existing attribute.')
                raise ValueError(msg)
            if name in self._custom_properties:
                msg = (f'{name} already exists in the custom_properties '
                       'attribute list.')
                raise ValueError(msg)

        property_error = False
        if self.isscalar:
            # This allows flux_radius to add len-1 array values for
            # scalar self
            if self._has_len(value) and len(value) == 1:
                value = value[0]

            if hasattr(value, 'isscalar'):
                # E.g., Quantity, SkyCoord, Time
                if not value.isscalar:
                    property_error = True
            elif not np.isscalar(value):
                property_error = True
        elif not self._has_len(value) or len(value) != self.n_labels:
            property_error = True
        if property_error:
            msg = ('value must have the same number of elements as the '
                   'catalog in order to add it as a property.')
            raise ValueError(msg)

        setattr(self, name, value)
        if name not in self._custom_properties:
            self._custom_properties.append(name)

    def remove_property(self, name):
        """
        Remove a user-defined property.

        The property must have been defined using `add_property`. The
        complete list of user-defined properties is stored in the
        `custom_properties` attribute.

        Parameters
        ----------
        name : str
            The name of the property to remove.
        """
        self.remove_properties(name)

    def remove_properties(self, names):
        """
        Remove user-defined properties.

        The properties must have been defined using `add_property`.
        The complete list of user-defined properties is stored in the
        `custom_properties` attribute.

        Parameters
        ----------
        names : list of str or str
            The names of the properties to remove.
        """
        if isinstance(names, str):
            names = [names]

        # We copy the list here to prevent changing the list in-place
        # during the for loop below, e.g., in case a user inputs
        # self.custom_properties to ``names``
        custom_properties = self._custom_properties.copy()

        for name in names:
            if name in custom_properties:
                delattr(self, name)
                custom_properties.remove(name)
            else:
                msg = f'{name} is not a defined property'
                raise ValueError(msg)
        self._custom_properties = custom_properties

    def rename_property(self, name, new_name):
        """
        Rename a user-defined property.

        The renamed property will remain at the same index in the
        `custom_properties` list.

        Parameters
        ----------
        name : str
            The old attribute name.

        new_name : str
            The new attribute name.
        """
        self.add_property(new_name, getattr(self, name))
        idx = self.custom_properties.index(name)
        self.remove_property(name)
        # Preserve the order of self.custom_properties
        self.custom_properties.remove(new_name)
        self.custom_properties.insert(idx, new_name)

    @cached_property
    def _null_objects(self):
        """
        Return `None` values.

        For example, this is used for SkyCoord properties if ``wcs`` is
        `None`.
        """
        return np.array([None] * self.n_labels)

    @cached_property
    def _null_values(self):
        """
        Return np.nan values.

        For example, this is used for background properties if
        ``background`` is `None`.
        """
        values = np.empty(self.n_labels)
        values.fill(np.nan)
        return values

    @cached_property
    def _data_cutouts(self):
        """
        A list of data cutouts using the segmentation image slices.
        """
        return [self._data[slc] for slc in self._slices_iter]

    @cached_property
    def _segmentation_image_cutouts(self):
        """
        A list of segmentation image cutouts using the segmentation
        image slices.
        """
        return [self._segmentation_image.data[slc]
                for slc in self._slices_iter]

    @cached_property
    def _mask_cutouts(self):
        """
        A list of mask cutouts using the segmentation image slices.

        If the input ``mask`` is None then a list of None is returned.
        """
        if self._mask is None:
            return self._null_objects
        return [self._mask[slc] for slc in self._slices_iter]

    @cached_property
    def _error_cutouts(self):
        """
        A list of error cutouts using the segmentation image slices.

        If the input ``error`` is None then a list of None is returned.
        """
        if self._error is None:
            return self._null_objects
        return [self._error[slc] for slc in self._slices_iter]

    @cached_property
    def _convdata_cutouts(self):
        """
        A list of convolved data cutouts using the segmentation image
        slices.
        """
        return [self._convolved_data[slc] for slc in self._slices_iter]

    @cached_property
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

    @cached_property
    def _cutout_segment_masks(self):
        """
        Cutout boolean mask for source segment.

        The mask is `True` for all pixels (background and from other
        source segments) outside the source segment.
        """
        return [segm != label
                for label, segm
                in zip(self.labels,
                       self._segmentation_image_cutouts,
                       strict=True)]

    @cached_property
    def _cutout_data_masks(self):
        """
        Cutout boolean mask of non-finite ``data`` values combined with
        the input ``mask`` array.

        The mask is `True` for non-finite ``data`` values and where the
        input ``mask`` is `True`.
        """
        return self._make_cutout_data_masks(self._data_cutouts,
                                            self._mask_cutouts)

    @cached_property
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

    def select_label(self, label):
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
        return self.select_labels(label)

    def select_labels(self, labels):
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
        self._segmentation_image.check_labels(labels)
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
            using `add_property`. If ``columns`` is `None`, then a
            default list of scalar-valued properties (as defined by the
            ``default_columns`` attribute) will be used.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of sources properties with one row per source.

        Raises
        ------
        ValueError
            If any name in ``columns`` is not a valid (or deprecated)
            column name.
        """
        if columns is None:
            table_columns = self.default_columns
        else:
            allowed_columns = (set(self.properties)
                               | set(self.custom_properties))
            # Remove 2D cutout images from the allowed columns
            allowed_columns = {col for col in allowed_columns
                               if '_cutout' not in col}

            deprecated_names = _DEPRECATED_ATTRIBUTES.copy()
            # These are not valid column names, but are deprecated
            # attributes that are still accessible in 3.x. They will be
            # removed in 4.0.
            invalid = ('add_extra_property', 'apermask_method', 'background',
                       'background_ma', 'convdata', 'convdata_ma', 'data',
                       'data_ma', 'error', 'error_ma', 'extra_properties',
                       'fluxfrac_radius', 'get_label', 'get_labels',
                       'localbkg_width', 'nlabels', 'remove_extra_properties',
                       'remove_extra_property', 'rename_extra_property',
                       'segment', 'segment_ma')
            for name in invalid:
                deprecated_names.pop(name)

            table_columns = validate_table_columns(
                columns, allowed_columns, deprecated_names=deprecated_names)

        # Replace with QTable() in 4.0
        tbl = create_empty_deprecated_qtable(
            _DEPRECATED_ATTRIBUTES, since='3.0', until='4.0')

        tbl.meta.update(self.meta)  # keep tbl.meta type
        for column in table_columns:
            values = getattr(self, column)

            # Column assignment requires an object with a length
            if self.isscalar:
                values = (values,)

            # Use the canonical (non-deprecated) column name so that
            # assigning into ``tbl`` does not trigger a second,
            # redundant deprecation warning (the deprecated attribute
            # access above already warned once).
            canonical_column = _DEPRECATED_ATTRIBUTES.get(column, column)
            tbl[canonical_column] = values
        return tbl

    @cached_property
    def n_labels(self):
        """
        The number of source labels.
        """
        return len(self.labels)

    @property
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
    def slices(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source.

        Returns a list for multi-source catalogs, or a single tuple for
        a single-source catalog.
        """
        return self._slices

    @cached_property
    def _slices_iter(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source, always as an iterable.
        """
        _slices = self.slices
        if self.isscalar:
            _slices = (_slices,)
        return _slices

    @cached_property
    def segment_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout of the segmentation image using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(
            self._segmentation_image_cutouts, units=False,
            masked=False)

    @cached_property
    def segment_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout of the segmentation image
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(
            self._segmentation_image_cutouts, units=False,
            masked=True)

    @cached_property
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._data_cutouts, units=True,
                                     masked=False, dtype=float)

    @cached_property
    def data_cutout_masked(self):
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

    @cached_property
    def conv_data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the convolved data using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._convdata_cutouts, units=True,
                                     masked=False, dtype=float)

    @cached_property
    def conv_data_cutout_masked(self):
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

    @cached_property
    def error_cutout(self):
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

    @cached_property
    def error_cutout_masked(self):
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

    @cached_property
    def background_cutout(self):
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

    @cached_property
    def background_cutout_masked(self):
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

    @cached_property
    def _all_masked(self):
        """
        True if all pixels over the source segment are masked.
        """
        # The gathered data values hold a single NaN for a completely
        # masked source, and a non-finite data value is always masked,
        # so no other source has a NaN value
        values = self._data_values
        sizes = np.array([arr.size for arr in values])
        first = np.array([arr[0] for arr in values])
        return (sizes == 1) & np.isnan(first)

    @cached_property
    @_update_flags_docstring
    def flags(self):
        # numpydoc ignore: RT01
        """
        The per-source bitwise quality flags.

        The flags combine deblending-provenance flags carried
        by the input segmentation image with measurement-time
        flags computed from this catalog's ``data``,
        ``mask``, ``error``, and segmentation image. Use
        `~photutils.segmentation.decode_segmentation_flags` to decode
        the values.

        If a ``detection_catalog`` was input, the shape
        (``undefined_shape``, ``singular_covariance``), centroid
        (``centroid_win_fallback``, ``centroid_quad_failed``),
        and Kron aperture geometry (``kron_undefined``,
        ``kron_minimum_radius``) flags derive from the detection
        catalog. The Kron photometry-loop flags (``kron_no_overlap``,
        ``kron_partial_overlap``, ``kron_masked_pixels``,
        ``kron_neighbor_pixels``, ``kron_uncorrected_pixels``), like the
        segment-level edge, mask, non-finite, and ``all_masked`` flags,
        are evaluated on this catalog's own inputs.

        Accessing ``flags`` computes the moment, covariance, centroid,
        and Kron-aperture properties if they have not already
        been computed (the results are cached and reused by those
        properties).

        The flags are:

        <flag_descriptions>
        """
        flags = np.zeros(len(self.labels), dtype=int)

        # Deblending provenance flags from the segmentation image
        flags_map = self._segmentation_image._flags_map
        if flags_map:
            flags |= np.array([flags_map.get(int(label), 0)
                               for label in self.labels])

        # Source segment touches an image boundary
        ny, nx = self._data.shape
        edge = np.array([(slc[0].start == 0 or slc[1].start == 0
                          or slc[0].stop == ny or slc[1].stop == nx)
                         for slc in self._slices_iter])
        flags[edge] |= SEGMENTATION_FLAGS.EDGE_TOUCH

        # Input-masked pixels within the source segment
        if self._mask is not None:
            masked = np.array(
                [np.any(mask_cut & ~segm_mask)
                 for mask_cut, segm_mask in zip(
                     self._mask_cutouts,
                     self._cutout_segment_masks, strict=True)])
            flags[masked] |= SEGMENTATION_FLAGS.MASKED_PIXELS

        # Non-finite data values within the source segment
        nonfinite = np.array(
            [np.any(~np.isfinite(data_cut) & ~segm_mask)
             for data_cut, segm_mask in zip(
                 self._data_cutouts, self._cutout_segment_masks,
                 strict=True)])
        flags[nonfinite] |= SEGMENTATION_FLAGS.NON_FINITE_DATA

        # Non-finite error values within the source segment
        if self._error is not None:
            nonfinite_err = np.array(
                [np.any(~np.isfinite(err_cut) & ~segm_mask)
                 for err_cut, segm_mask in zip(
                     self._error_cutouts,
                     self._cutout_segment_masks, strict=True)])
            flags[nonfinite_err] |= SEGMENTATION_FLAGS.NON_FINITE_ERROR

        # All pixels within the source segment are masked
        flags[self._all_masked] |= SEGMENTATION_FLAGS.ALL_MASKED

        # Non-positive net flux (the zeroth image moment over the
        # source segment). The moment-derived shape properties are
        # undefined. Fully-masked sources and sources with no positive
        # (convolved) data values have a zero net flux, so they are also
        # flagged here. The ``~(m00 > 0)`` form is defensive against
        # a non-finite net flux, which cannot currently occur because
        # the moment cutouts are zero-filled for masked and non-finite
        # pixels, but the form would still catch it.
        m00 = self._array('moments')[:, 0, 0]
        flags[~(m00 > 0)] |= SEGMENTATION_FLAGS.UNDEFINED_SHAPE

        # Singular, nearly singular, or rank-1 degenerate source
        # covariance, evaluated on the raw (unregularized) covariance
        # matrix
        flags[self._singular_covariance_flag_mask] |= (
            SEGMENTATION_FLAGS.SINGULAR_COVARIANCE)

        # Windowed centroid is NaN or fell back to the isophotal
        # centroid
        flags[self._centroid_win_fallback] |= (
            SEGMENTATION_FLAGS.CENTROID_WIN_FALLBACK)

        # Quadratic-fit centroid is non-finite
        quad = self._array('centroid_quad')
        quad_failed = ~np.all(np.isfinite(quad), axis=1)
        flags[quad_failed] |= SEGMENTATION_FLAGS.CENTROID_QUAD_FAILED

        # Kron-aperture flags
        flags |= self._kron_flags

        # Minimum Kron radius or minimum circular radius applied.
        # _measured_kron_radius is the unscaled measured value, which
        # _calc_kron_radius clips to kron_params[1] and sets to zero
        # when the minimum circular aperture is used instead. A
        # measured value equal to the minimum is also flagged because
        # _measured_kron_radius returns exactly kron_params[1] when the
        # Kron numerator or denominator is not positive. NaN comparisons
        # are False, so sources with an undefined Kron radius are not
        # flagged here (they are flagged as kron_undefined instead). The
        # Kron radii are measured by the detection catalog, if input, so
        # its kron_params define the applied minimum.
        detcat = (self if self._detection_catalog is None
                  else self._detection_catalog)
        measured = self._measured_kron_radius
        kron_radius = self._array('kron_radius').value
        min_applied = ((measured <= detcat.kron_params[1])
                       | (kron_radius == 0))
        flags[min_applied] |= SEGMENTATION_FLAGS.KRON_MINIMUM_RADIUS

        return flags

    def decode_flags(self, *, return_bit_values=False):
        """
        Decode the source quality flags into individual components.

        This is a convenience method that calls
        `~photutils.segmentation.decode_segmentation_flags` with the
        `flags` property.

        Parameters
        ----------
        return_bit_values : bool, optional
            If `True`, return the decoded bit flags (integers) instead
            of the flag names (strings).

        Returns
        -------
        decoded : dict
            A dictionary mapping each source label to the list of its
            active flag names (or bit values). The entries follow the
            catalog order.

        See Also
        --------
        photutils.segmentation.decode_segmentation_flags

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.modeling.models import Gaussian2D
        >>> from photutils.segmentation import SourceCatalog, detect_sources
        >>> yy, xx = np.mgrid[0:51, 0:51]
        >>> data = (Gaussian2D(100, 25, 25, 3, 3)(xx, yy)
        ...         + Gaussian2D(100, 2, 40, 3, 3)(xx, yy))
        >>> segm = detect_sources(data, 10, 5)
        >>> cat = SourceCatalog(data, segm)
        >>> for label, names in cat.decode_flags().items():
        ...     print(label, names)
        1 []
        2 ['edge_touch', 'kron_partial_overlap']
        """
        decoded = decode_segmentation_flags(
            self._array('flags'), return_bit_values=return_bit_values)
        return {int(label): flags
                for label, flags in zip(self.labels, decoded, strict=True)}

    def _segment_gather(self, array):
        """
        Pack the unmasked values of ``array`` within each source
        segment into one array (see ``batch_segment_gather``).

        Parameters
        ----------
        array : 2D `~numpy.ndarray`
            The C-contiguous float64 full-image array to gather from.

        Returns
        -------
        packed : 1D `~numpy.ndarray`
            The packed values of all sources, in row-major pixel order,
            with a single NaN for a completely-masked source.

        offsets : 1D `~numpy.ndarray`
            The start offset of each source in ``packed`` (length
            ``n_labels + 1``).

        counts : 1D `~numpy.ndarray`
            The number of unmasked segment pixels of each source.
        """
        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        return batch_segment_gather(
            array, mask=arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax)

    def _segment_values(self, array):
        """
        Get a list of 1D arrays of the unmasked values of ``array``
        within each source segment.

        An array with a single NaN is returned for completely-masked
        sources.

        Parameters
        ----------
        array : 2D `~numpy.ndarray`
            The C-contiguous float64 full-image array to gather from.

        Returns
        -------
        result : list of 1D `~numpy.ndarray`
            The per-source values, in row-major pixel order.
        """
        packed, offsets, _ = self._segment_gather(array)
        return np.split(packed, offsets[1:-1])

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

    @cached_property
    def _data_values(self):
        """
        A list of 1D arrays of the unmasked data values within each
        source segment.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._segment_values(self._get_batch_arrays()['data'])

    @cached_property
    def _error_values(self):
        """
        A list of 1D arrays of the unmasked error values within each
        source segment.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self._error is None:
            return self._null_objects
        return self._segment_values(self._get_batch_arrays()['error'])

    @cached_property
    def _background_values(self):
        """
        A list of 1D arrays of the unmasked background values within
        each source segment.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self._background is None:
            return self._null_objects
        return self._segment_values(self._get_batch_background())

    @cached_property
    @use_detcat
    def moments(self):
        """
        Spatial moments up to 3rd order of the source.
        """
        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        return batch_raw_moments(
            self._get_batch_convdata(), mask=arrays['mask'],
            segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax)

    @cached_property
    @use_detcat
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        cutout_centroid = self._array('cutout_centroid')
        return batch_central_moments(
            self._get_batch_convdata(), mask=arrays['mask'],
            segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax,
            xcen=np.ascontiguousarray(cutout_centroid[:, 0]),
            ycen=np.ascontiguousarray(cutout_centroid[:, 1]))

    @cached_property
    @use_detcat
    def cutout_centroid(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        centroid within the isophotal source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        moments = self._array('moments')

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            y_centroid = moments[:, 1, 0] / moments[:, 0, 0]
            x_centroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((x_centroid, y_centroid))

    @cached_property
    @use_detcat
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid within the isophotal
        source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @cached_property
    @use_detcat
    def x_centroid(self):
        """
        The ``x`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._array('centroid')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid(self):
        """
        The ``y`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._array('centroid')[:, 1]

    @cached_property
    @use_detcat
    def _centroid_err_cov(self):
        """
        The ``(n_labels, 2, 2)`` pixel error covariance matrices of the
        isophotal centroid, ``[[var_x, cov_xy], [cov_xy, var_y]]``.

        The matrices are all-NaN where the errors cannot be computed (no
        ``error`` input or non-positive total flux). See `centroid_err`
        for the propagation details.
        """
        if self._error is None:
            return np.full((self.n_labels, 2, 2), np.nan)

        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        cutout_centroid = self._array('cutout_centroid')

        # The accumulators are the summed pixel variance and its
        # dx**2, dy**2, and dx*dy weighted sums. The variance is zeroed
        # at masked pixels and at pixels that were excluded from the
        # moment data (e.g., non-finite or negative convolved values),
        # so that pixels that do not contribute flux weight also do not
        # contribute error variance.
        acc = batch_moment_err(
            arrays['error'], convdata=self._get_batch_convdata(),
            mask=arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax,
            xcen=np.ascontiguousarray(cutout_centroid[:, 0]),
            ycen=np.ascontiguousarray(cutout_centroid[:, 1]))

        # The total flux is the zeroth raw moment of the moment data.
        total_flux = self._array('moments')[:, 0, 0]
        is_singular = np.atleast_1d(self._singular_covariance_mask)
        pixel_var = 1.0 / 12.0

        # Ignore divide-by-zero and invalid-value RuntimeWarnings for
        # sources with non-positive or non-finite total flux; those
        # values are replaced by NaN below.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Propagate the error through the centroid formula.
            norm = 1.0 / total_flux**2
            err_var_x = acc[:, 1] * norm
            err_var_y = acc[:, 2] * norm
            err_cov_xy = acc[:, 3] * norm

            # Singularity correction for point-like sources. If the
            # error covariance matrix is nearly singular, add the
            # variance of a uniform distribution across a single pixel
            # (1/12) scaled by the summed pixel variance. The
            # correction is added to the variances only, never to the
            # covariance.
            err_sum_norm = acc[:, 0] * pixel_var * norm
            singular = (is_singular
                        & ((err_var_x * err_var_y - err_cov_xy**2)
                           < err_sum_norm**2))
            err_var_x[singular] += err_sum_norm[singular]
            err_var_y[singular] += err_sum_norm[singular]

            bad = ~np.isfinite(total_flux) | (total_flux <= 0)
            err_var_x[bad] = np.nan
            err_var_y[bad] = np.nan
            err_cov_xy[bad] = np.nan

        cov = np.empty((len(err_var_x), 2, 2))
        cov[:, 0, 0] = err_var_x
        cov[:, 1, 1] = err_var_y
        cov[:, 0, 1] = err_cov_xy
        cov[:, 1, 0] = err_cov_xy
        return cov

    def _sky_err_from_cov(self, pix_cov, xycen):
        """
        Transport pixel error covariances to sky position errors.

        The pixel covariance of each source is mapped to the local
        tangent plane with the forward WCS Jacobian ``F`` evaluated at
        the source position (``sky_cov = F pix_cov F^T``). The returned
        errors are the square roots of the tangent-plane variances along
        East and North.

        Parameters
        ----------
        pix_cov : `~numpy.ndarray`
            The ``(N, 2, 2)`` pixel error covariance matrices.

        xycen : `~numpy.ndarray`
            The ``(N, 2)`` pixel ``(x, y)`` centroid positions at which
            to evaluate the WCS Jacobians.

        Returns
        -------
        sky_err : `~astropy.units.Quantity`
            The ``(N, 2)`` array of 1-sigma errors in arcsec, with
            columns ``(east_err, north_err)``. Rows are NaN where the
            position or covariance is not finite.
        """
        sky_err = np.full(xycen.shape, np.nan)
        good = (np.all(np.isfinite(xycen), axis=1)
                & np.all(np.isfinite(pix_cov), axis=(1, 2)))
        if np.any(good):
            jac = compute_pixel_to_sky_jacobians(xycen[good, 0],
                                                 xycen[good, 1],
                                                 self.wcs)
            sky_cov = np.einsum('nij,njk,nlk->nil', jac, pix_cov[good], jac)
            sky_err[good, 0] = np.sqrt(sky_cov[:, 0, 0])
            sky_err[good, 1] = np.sqrt(sky_cov[:, 1, 1])
        return sky_err << u.arcsec

    @cached_property
    @use_detcat
    def centroid_err(self):
        """
        The ``(x, y)`` 1-sigma errors on the `centroid` position.

        The errors are computed by propagating the input ``error`` array
        through the center-of-mass centroid formula. If ``error`` was
        not input, then the errors are NaN.

        For a centroid defined as :math:`x_c = \\sum f_i x_i / F`,
        the variance is:

        .. math::

            \\text{Var}(x_c) = \\frac{\\sum_i (x_i - x_c)^2
            \\sigma_i^2}{F^2}

        For point-like sources whose shape covariance determinant
        is below :math:`(1/12)^2`, a singularity correction of
        :math:`\\sum_i \\sigma_i^2 / (12 F^2)` is added to the variances
        if the error covariance matrix is itself nearly singular.
        """
        cov = self._centroid_err_cov
        return np.sqrt(np.stack((cov[:, 0, 0], cov[:, 1, 1]), axis=1))

    @cached_property
    @use_detcat
    def x_centroid_err(self):
        """
        The 1-sigma error on the ``x`` coordinate of the `centroid`.

        See `centroid_err` for details. If ``error`` was not input, then
        the errors are NaN.
        """
        return self._array('centroid_err')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid_err(self):
        """
        The 1-sigma error on the ``y`` coordinate of the `centroid`.

        See `centroid_err` for details. If ``error`` was not input, then
        the errors are NaN.
        """
        return self._array('centroid_err')[:, 1]

    def _normalize_centroid_win_err(self, err_sum, err_var_x, err_var_y,
                                    err_cov_xy, weighted_flux):
        """
        Normalize the windowed centroid position-error variances.

        Normalize by ``step_factor^2 / weighted_flux^2``, add the
        pixel-size (1/12) variance correction, and apply the singularity
        correction for point-like sources. All inputs are arrays with
        one element per source. Sources with non-positive or non-finite
        ``weighted_flux`` have ``np.nan`` variances.

        Parameters
        ----------
        err_sum : `~numpy.ndarray`
            Sum of the weighted error variance over the aperture.

        err_var_x : `~numpy.ndarray`
            Weighted error variance in ``x``.

        err_var_y : `~numpy.ndarray`
            Weighted error variance in ``y``.

        err_cov_xy : `~numpy.ndarray`
            Weighted error covariance in ``xy``.

        weighted_flux : `~numpy.ndarray`
            Total weighted flux from the last iteration.

        Returns
        -------
        err_var_x : `~numpy.ndarray`
            Normalized error variance in ``x``.

        err_var_y : `~numpy.ndarray`
            Normalized error variance in ``y``.

        err_cov_xy : `~numpy.ndarray`
            Normalized error covariance in ``xy``. The pixel-size and
            singularity corrections are not applied to the covariance.
        """
        step_factor = 2.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            norm = step_factor**2 / weighted_flux**2

            # Add the pixel-size variance correction (1/12).
            # The finite-pixel term is added per pixel.
            err_sum_norm = err_sum * (1.0 / 12) * norm
            err_var_x = err_var_x * norm + err_sum_norm
            err_var_y = err_var_y * norm + err_sum_norm
            err_cov_xy = err_cov_xy * norm

            # Handle fully correlated profiles of point-like sources
            # that cause a singularity. The determinant check
            # includes the pixel-size correction in the variances
            # but not in the covariance.
            singular = (self._singular_covariance_mask
                        & ((err_var_x * err_var_y - err_cov_xy**2)
                           < err_sum_norm**2))
            err_var_x[singular] += err_sum_norm[singular]
            err_var_y[singular] += err_sum_norm[singular]

            bad = ~np.isfinite(weighted_flux) | (weighted_flux <= 0)
            err_var_x[bad] = np.nan
            err_var_y[bad] = np.nan
            err_cov_xy[bad] = np.nan

        return err_var_x, err_var_y, err_cov_xy

    def _apply_centroid_win_fallback(self, xcen_win, ycen_win,
                                     win_weighted_flux,
                                     win_cen_mom_xx, win_cen_mom_yy,
                                     win_cen_mom_xy,
                                     win_err_var_x, win_err_var_y,
                                     win_err_cov_xy,
                                     nan_hl, x_centroid, y_centroid):
        """
        Apply the fallback conditions for the windowed centroid.

        Reset the centroid to the isophotal centroid when any of the
        following conditions hold: the centroid diverged far from the
        isophotal centroid and lies outside the 1-sigma ellipse, the
        total weighted flux is non-positive, the windowed 2nd-order
        moments are negative, or the windowed covariance determinant
        is negative. Fallback sources also use the isophotal centroid
        errors. Sources with NaN half-light radius keep NaN.

        Parameters
        ----------
        xcen_win : `~numpy.ndarray`
            Windowed x centroids.

        ycen_win : `~numpy.ndarray`
            Windowed y centroids.

        win_weighted_flux : `~numpy.ndarray`
            Weighted flux from each source.

        win_cen_mom_xx : `~numpy.ndarray`
            Windowed central 2nd-order moment xx.

        win_cen_mom_yy : `~numpy.ndarray`
            Windowed central 2nd-order moment yy.

        win_cen_mom_xy : `~numpy.ndarray`
            Windowed central 2nd-order moment xy.

        win_err_var_x : `~numpy.ndarray`
            Error variance in x.

        win_err_var_y : `~numpy.ndarray`
            Error variance in y.

        win_err_cov_xy : `~numpy.ndarray`
            Error covariance in xy.

        nan_hl : `~numpy.ndarray`
            Boolean mask indicating sources with NaN half-light radius.

        x_centroid : `~numpy.ndarray`
            The isophotal x centroids.

        y_centroid : `~numpy.ndarray`
            The isophotal y centroids.

        Returns
        -------
        result : `~numpy.ndarray`
            The windowed centroid coordinates, their error variances and
            covariance, and the fallback indicator as columns ``(x, y,
            var_x, var_y, cov_xy, fallback)``, shape ``(n_sources, 6)``.
            The ``fallback`` column is 1.0 for sources whose windowed
            centroid was reset to the isophotal centroid or is NaN
            (non-finite half-light radius), and 0.0 otherwise.
        """
        dx = x_centroid - xcen_win
        dy = y_centroid - ycen_win
        cxx = self._array('ellipse_cxx').value
        cxy = self._array('ellipse_cxy').value
        cyy = self._array('ellipse_cyy').value

        reset = ((dx**2 > 1) & (dy**2 > 1)
                 & ((cxx * dx**2 + cxy * dx * dy + cyy * dy**2) > 1))
        reset |= win_weighted_flux <= 0.0
        reset |= (win_cen_mom_xx < 0.0) | (win_cen_mom_yy < 0.0)
        reset |= (win_cen_mom_xx * win_cen_mom_yy
                  - win_cen_mom_xy**2) < 0.0
        nan_cen = np.isnan(xcen_win) | np.isnan(ycen_win)
        reset |= nan_cen & ~nan_hl
        reset &= ~nan_hl
        if np.any(reset):
            xcen_win[reset] = x_centroid[reset]
            ycen_win[reset] = y_centroid[reset]
            # Fallback sources use the isophotal centroid, so also
            # use the isophotal centroid error covariance
            iso_cov = self._centroid_err_cov
            win_err_var_x[reset] = iso_cov[reset, 0, 0]
            win_err_var_y[reset] = iso_cov[reset, 1, 1]
            win_err_cov_xy[reset] = iso_cov[reset, 0, 1]

        fallback = (reset | nan_hl).astype(float)
        return np.transpose((xcen_win, ycen_win, win_err_var_x,
                             win_err_var_y, win_err_cov_xy, fallback))

    @cached_property
    @use_detcat
    def _centroid_win_results(self):
        """
        The "windowed" centroid coordinates, their error variances
        and covariance, and the fallback indicator as a 2D array with
        columns ``(x, y, var_x, var_y, cov_xy, fallback)`` and shape
        ``(n_labels, 6)``.

        The ``fallback`` column is 1.0 for sources whose windowed
        centroid was reset to the isophotal centroid or is NaN, and 0.0
        otherwise.

        This is the single computation behind `centroid_win`,
        `centroid_win_err`, and ``_centroid_win_fallback``. See
        `centroid_win` for the algorithm details.
        """
        # Use .copy() to avoid mutating the cached flux_radius value
        radius_hl = self.flux_radius(0.5).value.copy()
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

        compute_err = self._error is not None

        # Centroids as iterable arrays regardless of scalar state
        x_centroid = np.atleast_1d(self.x_centroid).astype(float)
        y_centroid = np.atleast_1d(self.y_centroid).astype(float)

        sigma = 2.0 * radius_hl * gaussian_fwhm_to_sigma
        # np.isnan (not ~np.isfinite) for parity with the previous
        # per-source math.isnan checks
        skip = (nan_hl | np.isnan(x_centroid)
                | np.isnan(y_centroid)).astype(np.uint8)
        arrays = self._get_batch_arrays()
        results = batch_centroid_win(
            arrays['data'], error=arrays['error'],
            mask=arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            xcen0=x_centroid, ycen0=y_centroid, sigma=sigma,
            skip=skip,
            seg_method=_SEG_METHOD_CODES[self.aperture_mask_method],
            compute_err=int(compute_err),
            max_aper_size=max(self._data.size, 1_000_000))

        (xcen_win, ycen_win, win_weighted_flux,
         win_cen_mom_xx, win_cen_mom_yy, win_cen_mom_xy,
         win_err_sum, win_err_var_x, win_err_var_y,
         win_err_cov_xy) = results.T.copy()

        # Normalize error terms by step_factor^2 / weighted_flux^2
        if compute_err:
            (win_err_var_x, win_err_var_y,
             win_err_cov_xy) = self._normalize_centroid_win_err(
                win_err_sum, win_err_var_x, win_err_var_y,
                win_err_cov_xy, win_weighted_flux)

        return self._apply_centroid_win_fallback(
            xcen_win, ycen_win, win_weighted_flux,
            win_cen_mom_xx, win_cen_mom_yy, win_cen_mom_xy,
            win_err_var_x, win_err_var_y, win_err_cov_xy, nan_hl,
            x_centroid, y_centroid)

    @cached_property
    @use_detcat
    def _centroid_win_fallback(self):
        """
        A boolean array that is `True` where the windowed centroid is
        NaN or fell back to the isophotal centroid.
        """
        return self._centroid_win_results[:, 5].astype(bool)

    @cached_property
    @use_detcat
    def centroid_win(self):
        """
        The ``(x, y)`` coordinate of the "windowed" centroid.

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE and YWIN_IMAGE parameters.

        Notes
        -----
        During each iteration, the centroid is calculated using all
        pixels within a circular aperture of ``4 * sigma`` from the
        current position, weighting pixel values with a 2D Gaussian with
        a standard deviation of ``sigma``. ``sigma`` is the half-light
        radius (i.e., ``flux_radius(0.5)``) times (2.0 / 2.35). A
        minimum half-light radius of 0.5 pixels is used. Iteration stops
        when the change in centroid position falls below a pre-defined
        threshold or a maximum number of iterations is reached.

        The windowed centroid is reset to the isophotal `centroid`
        if any of the following conditions hold: the centroid diverged
        far from the isophotal centroid and lies outside the 1-sigma
        ellipse, the total weighted flux is non-positive, the windowed
        2nd-order moments are negative, or the windowed covariance
        determinant is negative. If the half-light radius is not finite
        (e.g., due to a non-finite Kron radius), then ``np.nan`` will be
        returned.
        """
        return self._centroid_win_results[:, 0:2].copy()

    @cached_property
    @use_detcat
    def x_centroid_win(self):
        """
        The ``x`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE parameters.
        """
        return self._array('centroid_win')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid_win(self):
        """
        The ``y`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s YWIN_IMAGE parameters.
        """
        return self._array('centroid_win')[:, 1]

    @cached_property
    @use_detcat
    def centroid_win_err(self):
        """
        The ``(x, y)`` 1-sigma errors on the "windowed" centroid
        (`centroid_win`) position.

        The errors are computed by propagating the input ``error`` array
        through the Gaussian-weighted windowed centroid formula. If
        ``error`` was not input, then the errors are NaN.

        Sources where the windowed centroid fell back to the
        isophotal `centroid` have the isophotal centroid errors (see
        `centroid_err`). The errors are NaN where the half-light radius
        is not finite.
        """
        results = self._centroid_win_results
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.sqrt(results[:, 2:4])

    @cached_property
    @use_detcat
    def _centroid_win_err_cov(self):
        """
        The ``(n_labels, 2, 2)`` pixel error covariance matrices of the
        windowed centroid, ``[[var_x, cov_xy], [cov_xy, var_y]]``.

        Fallback sources hold the isophotal covariance (see
        ``_centroid_err_cov``); matrices are all-NaN where errors are
        unavailable.
        """
        results = self._centroid_win_results
        cov = np.empty((len(results), 2, 2))
        cov[:, 0, 0] = results[:, 2]
        cov[:, 1, 1] = results[:, 3]
        cov[:, 0, 1] = results[:, 4]
        cov[:, 1, 0] = results[:, 4]
        return cov

    @cached_property
    @use_detcat
    def x_centroid_win_err(self):
        """
        The 1-sigma error on the ``x`` coordinate of the "windowed"
        centroid (`centroid_win`).

        See `centroid_win_err` for details. If ``error`` was not input,
        then the errors are NaN.
        """
        return self._array('centroid_win_err')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid_win_err(self):
        """
        The 1-sigma error on the ``y`` coordinate of the "windowed"
        centroid (`centroid_win`).

        See `centroid_win_err` for details. If ``error`` was not input,
        then the errors are NaN.
        """
        return self._array('centroid_win_err')[:, 1]

    @cached_property
    @use_detcat
    def cutout_centroid_win(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        "windowed" centroid.

        The window centroid is computed using an iterative algorithm to
        derive a more accurate centroid. See `centroid_win` for further
        details about the algorithm.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.centroid_win - origin

    @cached_property
    @use_detcat
    def _centroid_quad_results(self):
        """
        The quadratic centroid coordinates, relative to the cutout data,
        and their error variances and covariance as a 2D array with
        columns ``(x, y, var_x, var_y, cov_xy)`` and shape ``(n_labels,
        5)``.

        This is the single computation behind `cutout_centroid_quad`,
        `centroid_quad`, and `centroid_quad_err`. See
        `cutout_centroid_quad` for the algorithm details.
        """
        # Precompute the pseudo-inverse for the 3x3 relative coordinate
        # design matrix [1, x, y, xy, x^2, y^2]. This is constant for
        # all sources.
        xi = np.arange(3)
        x, y = np.meshgrid(xi, xi)
        x = x.ravel()
        y = y.ravel()
        coeff_matrix = np.empty((9, 6), dtype=float)
        coeff_matrix[:, 0] = 1
        coeff_matrix[:, 1] = x
        coeff_matrix[:, 2] = y
        coeff_matrix[:, 3] = x * y
        coeff_matrix[:, 4] = x * x
        coeff_matrix[:, 5] = y * y
        pinv = np.linalg.pinv(coeff_matrix)

        compute_err = self._error is not None

        # Gather the 3x3 box around the peak pixel of each masked
        # (zero-filled) cutout
        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        status, peak, boxes, box_var = batch_quad_boxes(
            arrays['data'], error=arrays['error'], mask=arrays['mask'],
            segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax, compute_err=int(compute_err))
        nx = ixmax - ixmin
        ny = iymax - iymin
        n_src = len(status)

        results = np.full((n_src, 5), np.nan)

        # If the peak is at the edge of the cutout, return the peak
        # position. No fit is performed, so no errors can be
        # propagated.
        edge = status == 3
        results[edge, 0] = peak[edge, 0]
        results[edge, 1] = peak[edge, 1]

        # Fit the quadratic to the 3x3 box of the remaining sources
        fit = status == 0
        xidx0 = peak[:, 0] - 1
        yidx0 = peak[:, 1] - 1
        # einsum (rather than a BLAS matmul) keeps the per-source
        # arithmetic independent of the number of sources, so a sliced
        # catalog reproduces the parent catalog values exactly
        coeffs = np.einsum('ij,nj->ni', pinv, boxes)
        c10 = coeffs[:, 1]
        c01 = coeffs[:, 2]
        c11 = coeffs[:, 3]
        c20 = coeffs[:, 4]
        c02 = coeffs[:, 5]
        det = 4.0 * c20 * c02 - c11 * c11
        fit &= ~((det <= 0) | (c20 > 0))
        # Ignore RuntimeWarning from the sources without a fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Maximum in relative coords, then convert to cutout coords
            xm_rel = (c01 * c11 - 2.0 * c02 * c10) / det
            ym_rel = (c10 * c11 - 2.0 * c20 * c01) / det
            xm = xm_rel + xidx0
            ym = ym_rel + yidx0
            fit &= ((xm > 0.0) & (xm < (nx - 1.0))
                    & (ym > 0.0) & (ym < (ny - 1.0)))
            results[fit, 0] = xm[fit]
            results[fit, 1] = ym[fit]

            if compute_err:
                # Propagate the pixel variances through the quadratic
                # peak solution. The fit coefficients are linear in the
                # pixel values (``c = pinv @ pixels``) and the peak
                # position is a rational function of the coefficients,
                # so the position variances follow from the delta
                # method. Masked pixels have a fixed (zeroed) data
                # value, so they contribute no variance.
                zeros = np.zeros(n_src)
                grad_x = np.column_stack(
                    [zeros,
                     -2.0 * c02 / det,
                     c11 / det,
                     (c01 + 2.0 * c11 * xm_rel) / det,
                     -4.0 * c02 * xm_rel / det,
                     (-2.0 * c10 - 4.0 * c20 * xm_rel) / det])
                grad_y = np.column_stack(
                    [zeros,
                     c11 / det,
                     -2.0 * c20 / det,
                     (c10 + 2.0 * c11 * ym_rel) / det,
                     (-2.0 * c01 - 4.0 * c02 * ym_rel) / det,
                     -4.0 * c20 * ym_rel / det])
                u_x = np.einsum('ni,ij->nj', grad_x, pinv)
                u_y = np.einsum('ni,ij->nj', grad_y, pinv)
                results[fit, 2] = np.sum(u_x**2 * box_var, axis=1)[fit]
                results[fit, 3] = np.sum(u_y**2 * box_var, axis=1)[fit]
                results[fit, 4] = np.sum(u_x * u_y * box_var, axis=1)[fit]

        # Use the segment barycenter if the fit returned NaN. Fallback
        # sources use the isophotal centroid, so also use the isophotal
        # centroid error covariance.
        nan_mask = (np.isnan(results[:, 0]) | np.isnan(results[:, 1]))
        if np.any(nan_mask):
            cutout_centroid = self._array('cutout_centroid')
            results[nan_mask, 0:2] = cutout_centroid[nan_mask]
            iso_cov = self._centroid_err_cov
            results[nan_mask, 2] = iso_cov[nan_mask, 0, 0]
            results[nan_mask, 3] = iso_cov[nan_mask, 1, 1]
            results[nan_mask, 4] = iso_cov[nan_mask, 0, 1]

        return results

    @cached_property
    @use_detcat
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
        * no unmasked pixels within the source segment

        In these cases, then the isophotal `centroid` will be used
        instead.

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        return self._centroid_quad_results[:, 0:2].copy()

    @cached_property
    @use_detcat
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
        * no unmasked pixels within the source segment

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid_quad + origin

    @cached_property
    @use_detcat
    def x_centroid_quad(self):
        """
        The ``x`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        return self._array('centroid_quad')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid_quad(self):
        """
        The ``y`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        return self._array('centroid_quad')[:, 1]

    @cached_property
    @use_detcat
    def centroid_quad_err(self):
        """
        The ``(x, y)`` 1-sigma errors on the quadratic centroid
        (`centroid_quad`) position.

        The errors are computed by propagating the input ``error`` array
        through the least-squares quadratic fit and its peak solution
        using the delta method. If ``error`` was not input, then the
        errors are NaN.

        Sources where the quadratic centroid fell back to the
        isophotal `centroid` have the isophotal centroid errors (see
        `centroid_err`). The errors are NaN for sources where the
        maximum data value is at the edge of the source segment (where
        the position of the maximum pixel is returned instead of a fit).
        """
        results = self._centroid_quad_results
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.sqrt(results[:, 2:4])

    @cached_property
    @use_detcat
    def _centroid_quad_err_cov(self):
        """
        The ``(n_labels, 2, 2)`` pixel error covariance matrices of the
        quadratic centroid, ``[[var_x, cov_xy], [cov_xy, var_y]]``.

        Fallback sources hold the isophotal covariance (see
        ``_centroid_err_cov``); matrices are all-NaN where errors are
        unavailable.
        """
        results = self._centroid_quad_results
        cov = np.empty((len(results), 2, 2))
        cov[:, 0, 0] = results[:, 2]
        cov[:, 1, 1] = results[:, 3]
        cov[:, 0, 1] = results[:, 4]
        cov[:, 1, 0] = results[:, 4]
        return cov

    @cached_property
    @use_detcat
    def x_centroid_quad_err(self):
        """
        The 1-sigma error on the ``x`` coordinate of the quadratic
        centroid (`centroid_quad`).

        See `centroid_quad_err` for details. If ``error`` was not input,
        then the errors are NaN.
        """
        return self._array('centroid_quad_err')[:, 0]

    @cached_property
    @use_detcat
    def y_centroid_quad_err(self):
        """
        The 1-sigma error on the ``y`` coordinate of the quadratic
        centroid (`centroid_quad`).

        See `centroid_quad_err` for details. If ``error`` was not input,
        then the errors are NaN.
        """
        return self._array('centroid_quad_err')[:, 1]

    @cached_property
    @use_detcat
    def sky_centroid(self):
        """
        The sky coordinate of the `centroid` within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.x_centroid, self.y_centroid)

    @cached_property
    @use_detcat
    def sky_centroid_err(self):
        """
        The 1-sigma errors on the `sky_centroid` position as a ``(east,
        north)`` `~astropy.units.Quantity` in arcsec.

        The pixel error covariance of the `centroid` is transported
        to the local tangent plane with the WCS Jacobian evaluated
        at each source position. The first column is the error
        along East (the Right Ascension direction as a great-circle
        angle, i.e., including the cos(dec) factor) and the second
        is along North (Declination).

        `None` if ``wcs`` is not input. NaN where the pixel errors are
        unavailable (e.g., no ``error`` input).
        """
        if self.wcs is None:
            return self._null_objects
        xycen = self._array('centroid')
        return self._sky_err_from_cov(self._centroid_err_cov, xycen)

    @cached_property
    @use_detcat
    def sky_centroid_ra_err(self):
        """
        The 1-sigma error on the `sky_centroid` position along the Right
        Ascension direction, as a great-circle angle in arcsec (i.e.,
        including the cos(dec) factor).

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_err')[:, 0]

    @cached_property
    @use_detcat
    def sky_centroid_dec_err(self):
        """
        The 1-sigma error on the `sky_centroid` position along the
        Declination direction, in arcsec.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_err')[:, 1]

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
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
        return self.wcs.pixel_to_world(self.x_centroid_win,
                                       self.y_centroid_win)

    @cached_property
    @use_detcat
    def sky_centroid_win_err(self):
        """
        The 1-sigma errors on the `sky_centroid_win` position as a
        ``(east, north)`` `~astropy.units.Quantity` in arcsec.

        The pixel error covariance of the `centroid_win` is transported
        to the local tangent plane with the WCS Jacobian evaluated at
        each source position. The first column is the error along East
        (the Right Ascension direction as a great-circle angle, i.e.,
        including the cos(dec) factor) and the second is along North
        (Declination). Fallback sources use the isophotal centroid error
        covariance.

        `None` if ``wcs`` is not input. NaN where the pixel errors are
        unavailable (e.g., no ``error`` input).
        """
        if self.wcs is None:
            return self._null_objects
        xycen = self._array('centroid_win')
        return self._sky_err_from_cov(self._centroid_win_err_cov,
                                      xycen)

    @cached_property
    @use_detcat
    def sky_centroid_win_ra_err(self):
        """
        The 1-sigma error on the `sky_centroid_win` position along the
        Right Ascension direction, as a great-circle angle in arcsec
        (i.e., including the cos(dec) factor).

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_win_err')[:, 0]

    @cached_property
    @use_detcat
    def sky_centroid_win_dec_err(self):
        """
        The 1-sigma error on the `sky_centroid_win` position along the
        Declination direction, in arcsec.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_win_err')[:, 1]

    @cached_property
    @use_detcat
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
        return self.wcs.pixel_to_world(self.x_centroid_quad,
                                       self.y_centroid_quad)

    @cached_property
    @use_detcat
    def sky_centroid_quad_err(self):
        """
        The 1-sigma errors on the `sky_centroid_quad` position as a
        ``(east, north)`` `~astropy.units.Quantity` in arcsec.

        The pixel error covariance of the `centroid_quad` is transported
        to the local tangent plane with the WCS Jacobian evaluated at
        each source position. The first column is the error along East
        (the Right Ascension direction as a great-circle angle, i.e.,
        including the cos(dec) factor) and the second is along North
        (Declination). Fallback sources use the isophotal centroid error
        covariance.

        `None` if ``wcs`` is not input. NaN where the pixel errors are
        unavailable (e.g., no ``error`` input).
        """
        if self.wcs is None:
            return self._null_objects
        xycen = self._array('centroid_quad')
        return self._sky_err_from_cov(self._centroid_quad_err_cov,
                                      xycen)

    @cached_property
    @use_detcat
    def sky_centroid_quad_ra_err(self):
        """
        The 1-sigma error on the `sky_centroid_quad` position along the
        Right Ascension direction, as a great-circle angle in arcsec
        (i.e., including the cos(dec) factor).

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_quad_err')[:, 0]

    @cached_property
    @use_detcat
    def sky_centroid_quad_dec_err(self):
        """
        The 1-sigma error on the `sky_centroid_quad` position along the
        Declination direction, in arcsec.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self._array('sky_centroid_quad_err')[:, 1]

    @cached_property
    @use_detcat
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment, always as an iterable.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices_iter]

    @cached_property
    @use_detcat
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.

        Returns a list for multi-source catalogs, or a single
        `~photutils.aperture.BoundingBox` for a single-source catalog.
        """
        return self._bbox

    @cached_property
    @use_detcat
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices_iter])

    @cached_property
    @use_detcat
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices_iter])

    @cached_property
    @use_detcat
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices_iter])

    @cached_property
    @use_detcat
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices_iter])

    @cached_property
    @use_detcat
    def _bbox_corner_ll(self):
        """
        Lower-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @cached_property
    @use_detcat
    def _bbox_corner_ul(self):
        """
        Upper-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @cached_property
    @use_detcat
    def _bbox_corner_lr(self):
        """
        Lower-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @cached_property
    @use_detcat
    def _bbox_corner_ur(self):
        """
        Upper-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
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

    @cached_property
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

    @cached_property
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

    @cached_property
    def cutout_min_value_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        data = self._array('data_cutout_masked')
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmin(arr), arr.shape))
        return np.array(idx)

    @cached_property
    def cutout_max_value_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        data = self._array('data_cutout_masked')
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmax(arr), arr.shape))
        return np.array(idx)

    @cached_property
    def min_value_index(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        index = self._array('cutout_min_value_index')
        out = []
        for idx, slc in zip(index, self._slices_iter, strict=True):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @cached_property
    def max_value_index(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        index = self._array('cutout_max_value_index')
        out = []
        for idx, slc in zip(index, self._slices_iter, strict=True):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @cached_property
    def min_value_xindex(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        return self._array('min_value_index')[:, 1]

    @cached_property
    def min_value_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        return self._array('min_value_index')[:, 0]

    @cached_property
    def max_value_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        return self._array('max_value_index')[:, 1]

    @cached_property
    def max_value_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        return self._array('max_value_index')[:, 0]

    @cached_property
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
        source_sum, sizes = self._reduceat(self._data_values, np.add)
        # Subtract the local background over the number of unmasked
        # pixels actually included in the sum. The "area" property is
        # not used here because it is taken from the detection catalog
        # (if input), whose pixel count can differ when the data masks
        # differ.
        source_sum -= sizes * self._local_background
        if self._data_unit is not None:
            source_sum <<= self._data_unit
        return source_sum

    @cached_property
    def segment_flux_err(self):
        r"""
        The uncertainty of `segment_flux`, propagated from the input
        ``error`` array.

        ``segment_flux_err`` is the quadrature sum of the total errors
        over the unmasked pixels within the source segment:

        .. math::

            \Delta F = \sqrt{\sum_{i \in S} \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `segment_flux_err`,
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

    @cached_property
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

    @cached_property
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

    @cached_property
    def background_centroid(self):
        """
        The value of the per-pixel ``background`` at the position of the
        isophotal (center-of-mass) `centroid`.

        If ``detection_catalog`` is input, then its `centroid` will be
        used.

        The background values at fractional position values are
        determined using bilinear interpolation.
        """
        if self._background is None:
            bkg = self._null_values
        else:
            xcen = np.atleast_1d(self.x_centroid)
            ycen = np.atleast_1d(self.y_centroid)
            bkg = map_coordinates(self._background, (ycen, xcen), order=1,
                                  mode='nearest')

            mask = np.isfinite(xcen) & np.isfinite(ycen)
            bkg[~mask] = np.nan

        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    @cached_property
    @use_detcat
    def segment_area(self):
        """
        The total area of the source segment in units of pixels**2.

        This area is simply the area of the source segment from the
        input ``segmentation_image``. It does not take into account
        any data masking (i.e., a ``mask`` input to `SourceCatalog` or
        invalid ``data`` values).
        """
        segm = self._segmentation_image
        areas = segm.areas[segm.get_indices(np.atleast_1d(self.labels))]
        return np.asarray(areas, dtype=int) << (u.pix**2)

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """
        return np.sqrt(self.area / np.pi)

    @cached_property
    @use_detcat
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
        size = 34
        weights = np.zeros(size, dtype=float)
        weights[[5, 7, 15, 17, 25, 27]] = 1.0
        weights[[21, 33]] = np.sqrt(2.0)
        weights[[13, 23]] = (1 + np.sqrt(2.0)) / 2.0

        # Histogram of the convolved border-pixel patterns of each
        # source (see batch_perimeter); the weights are applied with
        # einsum so the per-source arithmetic is independent of the
        # number of sources
        arrays = self._get_batch_arrays()
        iymin, iymax, ixmin, ixmax = self._get_batch_bboxes()
        hist = batch_perimeter(
            arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            bbox_iymin=iymin, bbox_iymax=iymax, bbox_ixmin=ixmin,
            bbox_ixmax=ixmax)
        perimeter = np.einsum('nk,k->n', hist.astype(float), weights)
        perimeter[self._all_masked] = np.nan
        return np.array(perimeter) * u.pix

    @cached_property
    @use_detcat
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """
        moments = self._array('moments_central')
        mu_02 = moments[:, 0, 2]
        mu_11 = -moments[:, 1, 1]
        mu_20 = moments[:, 2, 0]
        tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
        return tensor.reshape((tensor.shape[0], 2, 2)) * u.pix**2

    @cached_property
    def _singular_covariance_mask(self):
        """
        A boolean mask with a leading source axis that is `True` for
        sources whose raw covariance matrix is singular or nearly
        singular (i.e., point-like sources).

        A source is flagged as singular when the determinant of its raw
        (unregularized) covariance matrix is less than ``(1 / 12)**2``,
        the squared variance of a uniform distribution across a single
        pixel. Sources with non-finite covariance are not flagged.
        """
        return self._raw_covariance_det < (1.0 / 12.0)**2

    @cached_property
    def _singular_covariance_flag_mask(self):
        """
        A boolean mask with a leading source axis that is `True` for
        sources whose raw covariance matrix is singular or nearly
        singular, including rank-1 degenerate sources.

        This is the mask used for the ``'singular_covariance'``
        flag. It matches the equivalent aperture flag (see
        `~photutils.aperture.decode_aperture_flags`): in addition to
        the determinant test used by ``_singular_covariance_mask``, a
        source is flagged when its minor-axis variance (the smaller
        eigenvalue of the raw covariance matrix) is less than ``1 /
        12``. The determinant test alone misses elongated sources that
        are unresolved along only one axis. Sources with non-finite
        covariance are not flagged.
        """
        covar = self._raw_covariance
        # Ignore RuntimeWarning from NaN values in the covariance
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Smaller eigenvalue (the minor-axis variance) of each
            # 2x2 symmetric covariance matrix, via the closed form
            # lambda = tr/2 -/+ sqrt((tr/2)**2 - det). The discriminant
            # is non-negative for a real symmetric matrix. Clip tiny
            # negative rounding to zero.
            half_trace = 0.5 * (covar[:, 0, 0] + covar[:, 1, 1])
            disc = np.maximum(half_trace**2 - self._raw_covariance_det,
                              0.0)
            min_eigval = half_trace - np.sqrt(disc)
            degenerate = (np.isfinite(min_eigval)
                          & (min_eigval < 1.0 / 12.0))
        return self._singular_covariance_mask | degenerate

    @cached_property
    def _raw_covariance_det(self):
        """
        The determinant of the raw ``(N, 2, 2)`` covariance matrix.
        """
        # Ignore RuntimeWarning from NaN values in the covariance
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.linalg.det(self._raw_covariance)

    @cached_property
    def _raw_covariance(self):
        """
        The raw ``(N, 2, 2)`` covariance matrix of the 2D Gaussian
        function that has the same normalized second-order moments as
        the source, before any regularization.

        This unregularized matrix is shared by `_covariance`
        (which regularizes a copy) and by the masks that test
        it for singularity (``_singular_covariance_mask`` and
        ``_singular_covariance_flag_mask``). Callers that modify the
        matrix in place must operate on a copy so the cached value is
        not corrupted.
        """
        moments = self._array('moments_central')
        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu_norm = moments / moments[:, 0, 0][:, np.newaxis, np.newaxis]
        covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                          mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
        return covar.reshape((covar.shape[0], 2, 2))

    @cached_property
    @use_detcat
    def _covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source, always as an iterable.
        """
        # Copy so the regularization below does not mutate the cached
        # raw covariance shared with `_singular_covariance_mask`.
        covar = self._raw_covariance.copy()

        # Regularize the covariance matrix for "infinitely" thin
        # detections by incrementally increasing the diagonal elements
        # by 1/12, the variance of a uniform distribution across a
        # single pixel (the smallest second moment a resolved source can
        # have given finite pixel size).
        delta = 1.0 / 12
        delta2 = delta**2
        # Ignore RuntimeWarning from NaN values in covar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            covar_det = np.linalg.det(covar)
            covar_trace = covar[:, 0, 0] + covar[:, 1, 1]

            # A valid covariance is positive semidefinite (det >= 0
            # and trace >= 0). Any matrix that is not (e.g., from
            # net-negative flux weighting) has an undefined shape and is
            # set to NaN.
            bad = (covar_det < 0) | (covar_trace < 0)
            covar[bad] = np.nan

            # Regularize "infinitely" thin detections by adding 1/12
            # (delta) to each diagonal. A single bump is sufficient.
            # For a positive semidefinite matrix the bumped determinant
            # exceeds the raw determinant by delta times the trace plus
            # delta squared. Since the raw determinant and trace are
            # both non-negative, the result is at least delta squared,
            # which equals the delta2 threshold, so it clears the
            # threshold in a single step.
            idx = np.where(covar_det < delta2)[0]
            covar[idx, 0, 0] += delta
            covar[idx, 1, 1] += delta
        return covar

    @cached_property
    @use_detcat
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """
        return self._covariance * (u.pix**2)

    @cached_property
    @use_detcat
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.empty((self.n_labels, 2))
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

    @cached_property
    @use_detcat
    def semimajor_axis(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self._array('covariance_eigvals')
        # This matches SourceExtractor's A parameter
        return np.sqrt(eigvals[:, 0])

    @cached_property
    @use_detcat
    def semiminor_axis(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        eigvals = self._array('covariance_eigvals')
        # This matches SourceExtractor's B parameter
        return np.sqrt(eigvals[:, 1])

    @cached_property
    @use_detcat
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
        semimajor (`semimajor_axis`) and semiminor (`semiminor_axis`)
        axes, respectively.
        """
        return 2.0 * np.sqrt(np.log(2.0) * (self.semimajor_axis**2
                                            + self.semiminor_axis**2))

    @cached_property
    @use_detcat
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction and is
        in the range (-90, 90] degrees.
        """
        covar = self._covariance
        orient_radians = 0.5 * np.arctan2(2.0 * covar[:, 0, 1],
                                          (covar[:, 0, 0] - covar[:, 1, 1]))
        return np.rad2deg(orient_radians) * u.deg

    @cached_property
    @use_detcat
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

    @cached_property
    @use_detcat
    def elongation(self):
        r"""
        The ratio of the lengths of the semimajor and semiminor axes.

        .. math::

            \mathrm{elongation} = \frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_axis / self.semiminor_axis

    @cached_property
    @use_detcat
    def ellipticity(self):
        r"""
        1.0 minus the ratio of the lengths of the semiminor and
        semimajor axes.

        .. math::

            \mathrm{ellipticity} = \frac{a - b}{a} = 1 - \frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_axis / self.semimajor_axis)

    @cached_property
    @use_detcat
    def covariance_xx(self):
        r"""
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\sigma_x^2`, in units of pixel**2.
        """
        return self._covariance[:, 0, 0] * u.pix**2

    @cached_property
    @use_detcat
    def covariance_yy(self):
        r"""
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.
        """
        return self._covariance[:, 1, 1] * u.pix**2

    @cached_property
    @use_detcat
    def covariance_xy(self):
        r"""
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
        pixel**2.
        """
        return self._covariance[:, 0, 1] * u.pix**2

    @cached_property
    @use_detcat
    def ellipse_cxx(self):
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
        return ((np.cos(self.orientation) / self.semimajor_axis)**2
                + (np.sin(self.orientation) / self.semiminor_axis)**2)

    @cached_property
    @use_detcat
    def ellipse_cyy(self):
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
        return ((np.sin(self.orientation) / self.semimajor_axis)**2
                + (np.cos(self.orientation) / self.semiminor_axis)**2)

    @cached_property
    @use_detcat
    def ellipse_cxy(self):
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
                * ((1.0 / self.semimajor_axis**2)
                   - (1.0 / self.semiminor_axis**2)))

    @cached_property
    @use_detcat
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
        packed, offsets, counts = self._segment_gather(
            self._get_batch_arrays()['data'])
        return _batch_gini(packed, offsets, counts)

    @cached_property
    def _local_background_apertures(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.
        """
        if self.local_bkg_width == 0:
            return self._null_objects

        apertures = []
        for bbox_ in self._bbox:
            xpos = 0.5 * (bbox_.ixmin + bbox_.ixmax - 1)
            ypos = 0.5 * (bbox_.iymin + bbox_.iymax - 1)
            scale = 1.5
            width_in = (bbox_.ixmax - bbox_.ixmin) * scale
            width_out = width_in + 2 * self.local_bkg_width
            height_in = (bbox_.iymax - bbox_.iymin) * scale
            height_out = height_in + 2 * self.local_bkg_width
            apertures.append(RectangularAnnulus((xpos, ypos), width_in,
                                                width_out, height_out,
                                                h_in=height_in, theta=0.0))
        return apertures

    @cached_property
    @use_detcat
    def local_background_aperture(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._local_background_apertures

    @cached_property
    def _local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.

        Pixels are masked where the input ``mask`` is `True`, where the
        input ``data`` is non-finite, and within any non-zero pixel
        label in the segmentation image.

        This property is always an `~numpy.ndarray` without units.
        """
        if self.local_bkg_width == 0:
            local_bkgs = np.zeros(self.n_labels)
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
                segm_mask_cutout = (
                    self._segmentation_image.data[slc_lg].astype(bool))
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

    @cached_property
    def local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.
        """
        bkg = self._local_background
        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    def _make_circular_apertures(self, radius):
        """
        Make circular aperture for each source.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, it will be used for the source centroids.

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
        x_centroid = np.atleast_1d(self.x_centroid)
        y_centroid = np.atleast_1d(self.y_centroid)
        radius = np.broadcast_to(radius, len(x_centroid))
        if np.any(radius <= 0):
            msg = 'radius must be > 0'
            raise ValueError(msg)

        apertures = []
        for (xcen, ycen, radius_, all_masked) in zip(x_centroid,
                                                     y_centroid,
                                                     radius,
                                                     self._all_masked,
                                                     strict=True):
            if all_masked or np.any(~np.isfinite((xcen, ycen, radius_))):
                apertures.append(None)
                continue

            apertures.append(CircularAperture((xcen, ycen), r=radius_))

        return apertures

    def make_circular_apertures(self, radius):
        """
        Make circular aperture for each source.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

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
        return self._make_scalar(self._make_circular_apertures(radius))

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_circular_apertures(self, radius, ax=None, origin=(0, 0),
                                **kwargs):
        """
        Plot circular apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

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
        return self._make_scalar(patches)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def circular_photometry(self, radius, name=None, overwrite=False):
        """
        Perform circular aperture photometry for each source.

        The circular aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the flux and flux error. The attribute names
            ``[name]_flux`` and ``[name]_flux_err`` will store the
            photometry results. For example, these names can then be
            included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, flux_err : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position is not finite or the source is
            completely masked).
        """
        if radius <= 0:
            msg = 'radius must be > 0'
            raise ValueError(msg)

        flux, flux_err = self._calc_circular_photometry(radius)

        if self._data_unit is not None:
            flux <<= self._data_unit
            flux_err <<= self._data_unit

        if self.isscalar:
            flux = flux[0]
            flux_err = flux_err[0]

        if name is not None:
            flux_name = f'{name}_flux'
            flux_err_name = f'{name}_flux_err'
            self.add_property(flux_name, flux, overwrite=overwrite)
            self.add_property(flux_err_name, flux_err,
                              overwrite=overwrite)

        return flux, flux_err

    def _make_elliptical_apertures(self, *, scale=6.0):
        """
        Return a list of elliptical apertures based on the scaled
        isophotal shape of the sources.

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its source `centroid` and shape parameters will be used.

        If scale is zero (due to a minimum circular radius set in
        ``kron_params``) then a circular aperture will be returned with
        the minimum circular radius.

        Parameters
        ----------
        scale : float or `~numpy.ndarray`, optional
            The scale factor to apply to the ellipse major and minor
            axes. The default value of 6.0 is roughly two times the
            isophotal extent of the source. A `~numpy.ndarray` input
            must be a 1D array of length ``n_labels``.

        Returns
        -------
        result : list of `~photutils.aperture.EllipticalAperture`
            A list of `~photutils.aperture.EllipticalAperture`
            instances. The aperture will be `None` where the source
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked.
        """
        xcen = np.atleast_1d(self.x_centroid)
        ycen = np.atleast_1d(self.y_centroid)
        major_size = np.atleast_1d(self.semimajor_axis.value) * scale
        minor_size = np.atleast_1d(self.semiminor_axis.value) * scale
        theta = np.atleast_1d(self.orientation.to_value(u.radian))

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

    @cached_property
    @use_detcat
    def _measured_kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius, always as an array
        (without units).

        The returned value is the measured Kron radius without applying
        any minimum Kron or circular radius.
        """
        scale = 6.0
        xcen = np.ascontiguousarray(self._array('x_centroid'),
                                    dtype=np.float64)
        ycen = np.ascontiguousarray(self._array('y_centroid'),
                                    dtype=np.float64)
        semimajor = np.ascontiguousarray(
            self._array('semimajor_axis').value, dtype=np.float64)
        semiminor = np.ascontiguousarray(
            self._array('semiminor_axis').value, dtype=np.float64)
        theta = np.ascontiguousarray(
            self._array('orientation').to_value(u.radian),
            dtype=np.float64)
        cxx = np.ascontiguousarray(self._array('ellipse_cxx').value,
                                   dtype=np.float64)
        cxy = np.ascontiguousarray(self._array('ellipse_cxy').value,
                                   dtype=np.float64)
        cyy = np.ascontiguousarray(self._array('ellipse_cyy').value,
                                   dtype=np.float64)

        # Completely masked sources and sources with a non-finite
        # centroid or shape have no measured Kron radius (NaN)
        skip = (self._all_masked | ~np.isfinite(xcen) | ~np.isfinite(ycen)
                | ~np.isfinite(semimajor * scale)
                | ~np.isfinite(semiminor * scale)
                | ~np.isfinite(theta)).astype(np.uint8)

        kron_min = self.kron_params[1]
        min_circ_radius = (self.kron_params[2]
                           if len(self.kron_params) == 3 else 0.0)

        arrays = self._get_batch_arrays()
        sums = batch_kron_radius(
            arrays['data'], mask=arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            xcen=xcen, ycen=ycen, semimajor=semimajor,
            semiminor=semiminor, theta=theta, cxx=cxx, cxy=cxy, cyy=cyy,
            skip=skip,
            seg_method=_SEG_METHOD_CODES[self.aperture_mask_method],
            scale=scale, min_circ_radius=min_circ_radius,
            max_aper_size=max(self._data.size, 1_000_000))
        flux_numer = sums[:, 0]
        flux_denom = sums[:, 1]

        # Ignore RuntimeWarning from undefined (NaN) sums and from a
        # zero denominator
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            kron_radius = flux_numer / flux_denom
            # Set the Kron radius to the minimum Kron radius if the
            # numerator or denominator is not positive (NaN comparisons
            # are False, so undefined sources stay NaN)
            kron_radius[(flux_numer <= 0) | (flux_denom <= 0)] = kron_min

        return kron_radius

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
            semimajor_axis = self.semimajor_axis.value
            semiminor_axis = self.semiminor_axis.value
            circ_radius = (kron_params[0] * kron_radius
                           * np.sqrt(semimajor_axis * semiminor_axis))
            kron_radius[circ_radius <= kron_params[2]] = 0.0

        return kron_radius << u.pix

    @cached_property
    @use_detcat
    def kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius.

        The *unscaled* first-moment Kron radius is given by:

        .. math::

            r_k = \frac{\sum_{i \in A} \ r_i I_i}{\sum_{i \in A} I_i}

        where :math:`I_i` are the data values and the sum is over
        pixels in an elliptical aperture whose axes are defined by
        six times the semimajor (`semimajor_axis`) and semiminor
        axes (`semiminor_axis`) at the calculated `orientation` (all
        properties derived from the central image moments of the
        source). :math:`r_i` is the elliptical "radius" to the pixel
        given by:

        .. math::

            r_i^2 = cxx (x_i - \bar{x})^2 +
                cxy (x_i - \bar{x})(y_i - \bar{y}) +
                cyy (y_i - \bar{y})^2

        where :math:`\bar{x}` and :math:`\bar{y}` represent the source
        `centroid` and the coefficients are based on image moments
        (`ellipse_cxx`, `ellipse_cxy`, and `ellipse_cyy`).

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

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis` *
        `semiminor_axis`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron radius will be set to
        zero and the Kron aperture will be a circle with this minimum
        radius.

        If the source is completely masked, then ``np.nan`` will be
        returned for both the Kron radius and Kron flux (the Kron
        aperture will be `None`).

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its ``kron_radius`` will be returned.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.
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

    @cached_property
    @use_detcat
    def kron_aperture(self):
        r"""
        The elliptical (or circular) Kron aperture.

        The Kron aperture is calculated for each source using its shape
        parameters, `kron_radius`, and the ``kron_params`` scaling and
        minimum values input into `SourceCatalog`. The Kron aperture is
        used to compute the Kron photometry.

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis` *
        `semiminor_axis`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron aperture will be a
        circle with this minimum radius.

        The aperture will be `None` where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its ``kron_aperture`` will be returned.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._make_kron_apertures(self.kron_params)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_kron_apertures(self, kron_params=None):
        """
        Make Kron apertures for each source.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_flux_err`) in the source catalog. This method should
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
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
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
        return self._make_scalar(self._make_kron_apertures(kron_params))

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_kron_apertures(self, kron_params=None, ax=None, origin=(0, 0),
                            **kwargs):
        """
        Plot Kron apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        An aperture will not be plotted for sources where the source
        `centroid` position or elliptical shape parameters are not
        finite or where the source is completely masked.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_flux_err`) in the source catalog. This method should be
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
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
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
            apertures = self._array('kron_aperture')
        else:
            apertures = self._make_kron_apertures(kron_params)

        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(ax=ax, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return self._make_scalar(patches)

    @staticmethod
    def _overlap_params(kwargs):
        """
        Translate aperture ``to_mask`` keyword arguments to the
        ``(use_exact, subpixels)`` overlap parameters of the batch
        drivers.

        Parameters
        ----------
        kwargs : dict
            The ``to_mask`` keyword arguments (``method`` and, for the
            subpixel method, ``subpixels``).

        Returns
        -------
        use_exact, subpixels : int
            The overlap parameters.
        """
        method = kwargs.get('method', 'exact')
        if method == 'exact':
            return 1, 1
        if method == 'center':
            return 0, 1
        return 0, kwargs.get('subpixels', 5)

    def _calc_circular_photometry(self, radius):
        """
        Calculate the flux and flux error (without units) in a circular
        aperture of the given radius centered on each source centroid.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If ``detection_catalog`` is input, then its `centroid` values
        will be used.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        Returns
        -------
        flux, flux_err : 1D `~numpy.ndarray`
            The aperture flux and flux error arrays. NaN where the
            source is completely masked, its centroid is not finite, the
            aperture does not overlap the data or has no contributing
            pixels, or the aperture is unreasonably large.
        """
        xcen = np.atleast_1d(self.x_centroid).astype(float)
        ycen = np.atleast_1d(self.y_centroid).astype(float)
        n_src = len(xcen)
        flux = np.full(n_src, np.nan)
        flux_err = np.full(n_src, np.nan)

        # Exclude undefined apertures (completely masked sources and
        # non-finite centroids) and apertures whose bounding box would
        # be unreasonably large (out-of-memory guard)
        max_size = max(self._data.size, 1_000_000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            nx = (np.floor(xcen + radius + 1.5)
                  - np.floor(xcen - radius + 0.5))
            ny = (np.floor(ycen + radius + 1.5)
                  - np.floor(ycen - radius + 0.5))
            valid = (~self._all_masked & np.isfinite(xcen)
                     & np.isfinite(ycen) & np.isfinite(radius)
                     & (nx * ny <= max_size))
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return flux, flux_err

        arrays = self._get_batch_arrays()
        seg_code = _SEG_METHOD_CODES[self.aperture_mask_method]
        seg_arr = None
        seg_labels = None
        if seg_code != 0:
            seg_arr = arrays['segm']
            seg_labels = np.ascontiguousarray(
                np.atleast_1d(self.labels), dtype=np.intp)[idx]
        positions = np.ascontiguousarray(
            np.column_stack((xcen[idx], ycen[idx])))
        local_bkg = np.ascontiguousarray(self._local_background,
                                         dtype=np.float64)[idx]
        use_exact, subpixels = self._overlap_params(
            self._aperture_mask_kwargs['circ'])

        (sums, sum_vars, _, overlap, _, _, _, _, _, fcounts,
         _) = batch_aperture_sums(
            arrays['data'], arrays['error'], arrays['mask'], positions,
            SHAPE_CIRCLE, np.array([radius], dtype=np.float64), radius,
            radius, 0.0, 0.0, use_exact, subpixels, seg_arr, seg_labels,
            seg_code, local_bkg, 0)

        # Membership matches the previous cutout-based
        # ``in_aperture & ~mask`` rule: under 'correct', uncorrectable
        # neighbor pixels stay members (with value zero)
        members = fcounts[:, FLAG_COL_VALID].copy()
        if seg_code == 3:
            members += fcounts[:, FLAG_COL_UNCORRECTED]
        good = overlap & (members > 0)
        flux[idx] = np.where(good, sums, np.nan)
        if self._error is not None:
            flux_err[idx] = np.where(good, np.sqrt(sum_vars), np.nan)

        return flux, flux_err

    def _kron_aperture_params(self, kron_params):
        """
        Return the Kron aperture geometry of each source as arrays.

        This is the array form of the apertures made by
        ``_make_kron_apertures`` (see `kron_aperture`), without
        constructing aperture objects.

        Parameters
        ----------
        kron_params : tuple
            The validated Kron parameters.

        Returns
        -------
        xcen, ycen : 1D `~numpy.ndarray`
            The aperture centers.

        major, minor : 1D `~numpy.ndarray`
            The semimajor and semiminor axes of the elliptical
            apertures (the scaled Kron radius times the isophotal
            axes).

        theta : 1D `~numpy.ndarray`
            The ellipse orientation in radians.

        circular : 1D `~numpy.ndarray` (bool)
            `True` where the aperture is instead the circle with the
            minimum circular radius ``kron_params[2]`` (i.e., where the
            Kron radius is zero).

        defined : 1D `~numpy.ndarray` (bool)
            `False` where the aperture is undefined (`None` in
            `kron_aperture`): where the source is completely masked or
            its centroid or elliptical shape parameters are not finite.
        """
        # NOTE: if kron_radius = NaN, scale = NaN and the aperture is
        # undefined
        kron_radius = self._calc_kron_radius(kron_params)
        scale = kron_radius.value * kron_params[0]
        xcen = np.atleast_1d(self.x_centroid)
        ycen = np.atleast_1d(self.y_centroid)
        major = np.atleast_1d(self.semimajor_axis.value) * scale
        minor = np.atleast_1d(self.semiminor_axis.value) * scale
        theta = np.atleast_1d(self.orientation.to_value(u.radian))
        defined = (~self._all_masked & np.isfinite(xcen) & np.isfinite(ycen)
                   & np.isfinite(major) & np.isfinite(minor)
                   & np.isfinite(theta))
        # kron_radius = 0 -> scale = 0 -> major/minor = 0
        circular = defined & (major == 0) & (minor == 0)
        return xcen, ycen, major, minor, theta, circular, defined

    def _calc_kron_photometry(self, *, kron_params=None):
        """
        Calculate the flux and flux error in the Kron aperture (without
        units).

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is undefined (`None` in `kron_aperture`),
        then ``np.nan`` will be returned.

        If ``detection_catalog`` is input, then its `centroid` values
        will be used.

        Parameters
        ----------
        kron_params : tuple or `None`, optional
            The Kron parameters. If `None`, the apertures are those
            of `kron_aperture` (from the detection catalog, if input).

        Returns
        -------
        kron_flux, kron_flux_err, kron_flags : tuple of `~numpy.ndarray`
            The Kron flux, flux error, and bitwise quality flags (see
            `~photutils.segmentation.decode_segmentation_flags`).
        """
        if kron_params is None:
            # The geometry of the kron_aperture property, which is
            # taken from the detection catalog if input
            source = (self if self._detection_catalog is None
                      else self._detection_catalog)
            kron_params = source.kron_params
        else:
            source = self
            kron_params = self._validate_kron_params(kron_params)
        (xcen, ycen, major, minor, theta, circular,
         defined) = source._kron_aperture_params(kron_params)

        n_src = len(xcen)
        flux = np.full(n_src, np.nan)
        flux_err = np.full(n_src, np.nan)
        kron_flags = np.zeros(n_src, dtype=int)
        kron_flags[~defined] = SEGMENTATION_FLAGS.KRON_UNDEFINED

        # Exclude apertures whose bounding box would be unreasonably
        # large (out-of-memory guard)
        max_size = max(self._data.size, 1_000_000)
        circ_radius = kron_params[2] if len(kron_params) == 3 else 0.0
        # Ignore RuntimeWarning from non-finite values of undefined
        # apertures
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_ext = np.sqrt((major * cos_t) ** 2 + (minor * sin_t) ** 2)
            y_ext = np.sqrt((major * sin_t) ** 2 + (minor * cos_t) ** 2)
            x_ext = np.where(circular, circ_radius, x_ext)
            y_ext = np.where(circular, circ_radius, y_ext)
            nx = (np.floor(xcen + x_ext + 1.5)
                  - np.floor(xcen - x_ext + 0.5))
            ny = (np.floor(ycen + y_ext + 1.5)
                  - np.floor(ycen - y_ext + 0.5))
            oversized = defined & (nx * ny > max_size)
        kron_flags[oversized] = SEGMENTATION_FLAGS.KRON_UNDEFINED
        usable = defined & ~oversized

        # Group the apertures by shape for the batch driver
        circ_idx = np.flatnonzero(usable & circular)
        ell_idx = np.flatnonzero(usable & ~circular)
        circ_params = np.full((len(circ_idx), 1), circ_radius)
        ell_params = np.column_stack((major, minor, theta))[ell_idx]
        positions = np.column_stack((xcen, ycen)).astype(float)

        arrays = self._get_batch_arrays()
        seg_code = _SEG_METHOD_CODES[self.aperture_mask_method]
        seg_arr = arrays['segm'] if seg_code != 0 else None
        labels_arr = np.ascontiguousarray(np.atleast_1d(self.labels),
                                          dtype=np.intp)
        local_bkg = np.ascontiguousarray(self._local_background,
                                         dtype=np.float64)
        has_error = self._error is not None

        for idx, psrc, shape_code in (
                (circ_idx, circ_params, SHAPE_CIRCLE),
                (ell_idx, ell_params, SHAPE_ELLIPSE)):
            if idx.size == 0:
                continue
            psrc = np.ascontiguousarray(psrc, dtype=np.float64)
            pos = np.ascontiguousarray(positions[idx])
            seg_labels = labels_arr[idx] if seg_code != 0 else None
            (sums, sum_vars, _, overlap, _, _, _, _, _, fcounts,
             weights_out) = batch_aperture_sums(
                arrays['data'], arrays['error'], arrays['mask'], pos,
                shape_code, None, 0.0, 0.0, 0.0, 0.0, 1, 1, seg_arr,
                seg_labels, seg_code, local_bkg[idx], 0,
                params_per_source=psrc)

            n_pix = fcounts[:, FLAG_COL_N_PIXELS]
            clipped = fcounts[:, FLAG_COL_BBOX_CLIPPED].astype(bool)
            weights_out = weights_out.astype(bool)
            # Membership matches the previous cutout-based
            # ``in_aperture & ~mask`` rule: under 'correct',
            # uncorrectable neighbor pixels stay members (with value
            # zero), so they keep the flux at 0.0 rather than NaN
            members = fcounts[:, FLAG_COL_VALID].copy()
            if seg_code == 3:
                members += fcounts[:, FLAG_COL_UNCORRECTED]
            good = overlap & (members > 0)
            flux[idx] = np.where(good, sums, np.nan)
            if has_error:
                flux_err[idx] = np.where(good, np.sqrt(sum_vars),
                                         np.nan)

            grp_flags = np.zeros(len(idx), dtype=int)
            no_ovl = ~overlap | (clipped & (n_pix == 0))
            grp_flags[no_ovl] |= SEGMENTATION_FLAGS.KRON_NO_OVERLAP
            grp_flags[(n_pix > 0) & weights_out] |= (
                SEGMENTATION_FLAGS.KRON_PARTIAL_OVERLAP)
            masked_any = (fcounts[:, FLAG_COL_MASKED]
                          + fcounts[:, FLAG_COL_NONFINITE_DATA]) > 0
            grp_flags[masked_any] |= (
                SEGMENTATION_FLAGS.KRON_MASKED_PIXELS)
            seg_any = (fcounts[:, FLAG_COL_SEG]
                       + fcounts[:, FLAG_COL_SEG_MASKED]) > 0
            grp_flags[seg_any] |= (
                SEGMENTATION_FLAGS.KRON_NEIGHBOR_PIXELS)
            unc_any = (fcounts[:, FLAG_COL_UNCORRECTED]
                       + fcounts[:, FLAG_COL_UNCORRECTED_MASKED]) > 0
            grp_flags[unc_any] |= (
                SEGMENTATION_FLAGS.KRON_UNCORRECTED_PIXELS)
            kron_flags[idx] |= grp_flags

        return flux, flux_err, kron_flags

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def kron_photometry(self, kron_params, name=None, overwrite=False):
        """
        Perform photometry for each source using an elliptical Kron
        aperture.

        This method can be used to calculate the Kron photometry using
        alternate ``kron_params`` (e.g., different scalings of the Kron
        radius).

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the Kron flux and flux error. The attribute
            names ``[name]_flux`` and ``[name]_flux_err`` will store
            the photometry results. For example, these names can then
            be included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, flux_err : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked).
        """
        kron_flux, kron_flux_err, _ = self._calc_kron_photometry(
            kron_params=kron_params)
        if self._data_unit is not None:
            kron_flux <<= self._data_unit
            kron_flux_err <<= self._data_unit

        if self.isscalar:
            kron_flux = kron_flux[0]
            kron_flux_err = kron_flux_err[0]

        if name is not None:
            flux_name = f'{name}_flux'
            flux_err_name = f'{name}_flux_err'
            self.add_property(flux_name, kron_flux, overwrite=overwrite)
            self.add_property(flux_err_name, kron_flux_err,
                              overwrite=overwrite)

        return kron_flux, kron_flux_err

    @cached_property
    def _kron_photometry(self):
        """
        The flux, flux error, and bitwise quality flags of the Kron
        aperture (without units) as a 2D array with columns ``(flux,
        flux_err, flags)`` and shape ``(n_labels, 3)``.

        The flags are stored as floats so that all three values can be
        kept in a single array that slices along the source axis. All of
        the flag bit values are exactly representable as floats.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be returned
        for the flux and flux error. This will occur where the source
        `centroid` position or elliptical shape parameters are not
        finite or where the source is completely masked.
        """
        return np.transpose(self._calc_kron_photometry(kron_params=None))

    @cached_property
    def _kron_flags(self):
        """
        The per-source bitwise quality flags from the Kron-aperture
        photometry.
        """
        return self._kron_photometry[:, 2].astype(int)

    @cached_property
    def kron_flux(self):
        """
        The flux in the Kron aperture.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_flux = self._kron_photometry[:, 0]
        if self._data_unit is not None:
            kron_flux <<= self._data_unit
        return kron_flux

    @cached_property
    def kron_flux_err(self):
        """
        The flux error in the Kron aperture.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_flux_err = self._kron_photometry[:, 1]
        if self._data_unit is not None:
            kron_flux_err <<= self._data_unit
        return kron_flux_err

    @cached_property
    @use_detcat
    def _max_circular_kron_radius(self):
        """
        The maximum circular Kron radius used as the upper limit of
        ``flux_radius``.
        """
        semimajor_sig = self._array('semimajor_axis').value
        kron_radius = self._array('kron_radius').value
        radius = semimajor_sig * kron_radius * self.kron_params[0]
        mask = radius == 0
        if np.any(mask):
            radius[mask] = self.kron_params[2]
        return radius

    @cached_property
    @use_detcat
    def _flux_radius_optimizer_args(self):
        """
        The prepared per-source arguments of the flux-radius root-find
        (see ``batch_flux_radius_prepare``), always as a list with one
        entry per source.

        Each entry is `None` for a source that cannot have a solution
        or the list ``[clean_data, grid_params, kronflux,
        max_radius]``. This cached property snapshots the current
        ``_aperture_mask_kwargs['flux_radius']`` settings. Changing
        them (e.g., via ``_set_semode``) after the first flux_radius
        call has no effect.
        """
        kron_flux = np.ascontiguousarray(self._kron_photometry[:, 0],
                                         dtype=np.float64)  # unitless
        max_radius = np.ascontiguousarray(self._max_circular_kron_radius,
                                          dtype=np.float64)
        x_centroid = np.ascontiguousarray(np.atleast_1d(self.x_centroid),
                                          dtype=np.float64)
        y_centroid = np.ascontiguousarray(np.atleast_1d(self.y_centroid),
                                          dtype=np.float64)
        local_bkg = np.ascontiguousarray(self._local_background,
                                         dtype=np.float64)
        use_exact, subpixels = self._overlap_params(
            self._aperture_mask_kwargs['flux_radius'])

        # Sources with a non-finite centroid, Kron flux, or maximum
        # radius, or a zero Kron flux, have no solution
        skip = (~np.isfinite(x_centroid) | ~np.isfinite(y_centroid)
                | ~np.isfinite(kron_flux) | ~np.isfinite(max_radius)
                | (kron_flux == 0)).astype(np.uint8)

        arrays = self._get_batch_arrays()
        return batch_flux_radius_prepare(
            arrays['data'], mask=arrays['mask'], segm=arrays['segm'],
            labels=np.ascontiguousarray(np.atleast_1d(self.labels),
                                        dtype=np.intp),
            xcen=x_centroid, ycen=y_centroid, local_bkg=local_bkg,
            kronflux=kron_flux, max_radius=max_radius, skip=skip,
            seg_method=_SEG_METHOD_CODES[self.aperture_mask_method],
            use_exact=use_exact, subpixels=subpixels,
            max_aper_size=max(self._data.size, 1_000_000))

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def flux_radius(self, fraction, name=None, overwrite=False):
        """
        Calculate the circular radius that encloses the specified
        fraction of the Kron flux.

        To estimate the half-light radius, use ``fraction = 0.5``.

        Parameters
        ----------
        fraction : float
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
        if fraction <= 0 or fraction > 1:
            msg = 'fraction must be > 0 and <= 1'
            raise ValueError(msg)

        # Return cached result if available
        if fraction in self._flux_radius_cache:
            result = self._flux_radius_cache[fraction]
            if name is not None:
                self.add_property(name, result, overwrite=overwrite)
            return self._make_scalar(result)

        args = self._flux_radius_optimizer_args
        radius = batch_flux_radius_solve(args, fraction=fraction)
        result = radius << u.pix
        self._flux_radius_cache[fraction] = result

        if name is not None:
            self.add_property(name, result, overwrite=overwrite)

        return self._make_scalar(result)

    def make_cutouts(self, shape, *, array=None, mode='partial',
                     fill_value=np.nan):
        """
        Make cutout arrays for each source.

        The cutout for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

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
            msg = "mode must be 'partial' or 'trim'"
            raise ValueError(msg)

        cutouts = []
        for (xcen, ycen, all_masked) in zip(np.atleast_1d(self.x_centroid),
                                            np.atleast_1d(self.y_centroid),
                                            self._all_masked, strict=True):

            if all_masked or np.any(~np.isfinite((xcen, ycen))):
                cutouts.append(None)
                continue

            cutouts.append(CutoutImage(array, (ycen, xcen), shape,
                                       mode=mode, fill_value=fill_value))

        return self._make_scalar(cutouts)
