# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by a segmentation image.
"""

import warnings

from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
from astropy.utils import deprecated, lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from ..aperture import BoundingBox
from ..utils._moments import _moments, _moments_central
from ..utils.convolution import _filter_data
from ..utils._wcs_helpers import _pixel_to_world

__all__ = ['SourceProperties', 'source_properties', 'SourceCatalog']

__doctest_requires__ = {('SourceProperties', 'SourceProperties.*',
                         'SourceCatalog', 'SourceCatalog.*',
                         'source_properties', 'properties_table'):
                        ['scipy']}

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


class SourceProperties:
    """
    Class to calculate photometry and morphological properties of a
    single labeled source.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.  If ``filtered_data`` is input, then it will be used
        instead of ``data`` to calculate the source centroid and
        morphological properties.  Source photometry is always measured
        from ``data``.  For accurate source properties and photometry,
        ``data`` should be background-subtracted.  Non-finite ``data``
        values (NaN and +/- inf) are automatically masked.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    label : int
        The label number of the source whose properties are calculated.

    filtered_data : array-like or `~astropy.units.Quantity`, optional
        The filtered version of the background-subtracted ``data`` from
        which to calculate the source centroid and morphological
        properties.  The kernel used to perform the filtering should be
        the same one used in defining the source segments (e.g., see
        :func:`~photutils.detect_sources`).  If ``data`` is a
        `~astropy.units.Quantity` array then ``filtered_data`` must be a
        `~astropy.units.Quantity` array (and vise versa) with identical
        units.  Non-finite ``filtered_data`` values (NaN and +/- inf)
        are not automatically masked, unless they are at the same
        position of non-finite values in the input ``data`` array.  Such
        pixels can be masked using the ``mask`` keyword.  If `None`,
        then the unfiltered ``data`` will be used instead.

    error : array_like or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data`` array.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vise versa) with identical
        units.  Non-finite ``error`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array.  Such pixels can
        be masked using the ``mask`` keyword.  See the Notes section
        below for details on the error propagation.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.  Non-finite
        values (NaN and +/- inf) in the input ``data`` are automatically
        masked.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  If ``data`` is
        a `~astropy.units.Quantity` array then ``background`` must be a
        `~astropy.units.Quantity` array (and vise versa) with identical
        units.  Inputting the ``background`` merely allows for its
        properties to be measured within each source segment.  The input
        ``background`` does *not* get subtracted from the input
        ``data``, which should already be background-subtracted.
        Non-finite ``background`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array.  Such pixels can
        be masked using the ``mask`` keyword.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).  If `None`, then all sky-based
        properties will be set to `None`.

    Notes
    -----
    ``data`` (and optional ``filtered_data``) should be
    background-subtracted for accurate source photometry and properties.

    `SExtractor`_'s centroid and morphological parameters are always
    calculated from a filtered "detection" image, i.e. the image used to
    define the segmentation image.  The usual downside of the filtering
    is the sources will be made more circular than they actually are.
    If you wish to reproduce `SExtractor`_ centroid and morphology
    results, then input a filtered and background-subtracted "detection"
    image into the ``filtered_data`` keyword.  If ``filtered_data`` is
    `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values (``filtered_data`` or ``data``) within the
    source segment are set to zero when calculating morphological
    properties based on image moments.  Negative values could occur, for
    example, if the segmentation image was defined from a different
    image (e.g., different bandpass) or if the background was
    oversubtracted. Note that `~photutils.SourceProperties.source_sum`
    always includes the contribution of negative ``data`` values.

    The input ``error`` array is assumed to include *all* sources of
    error, including the Poisson error of the sources.
    `~photutils.SourceProperties.source_sum_err` is simply the
    quadrature sum of the pixel-wise total errors over the non-masked
    pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.SourceProperties.source_sum_err`, :math:`S` are the
    non-masked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    Custom errors for source segments can be calculated using the
    `~photutils.SourceProperties.error_cutout_ma` and
    `~photutils.SourceProperties.background_cutout_ma` properties, which
    are 2D `~numpy.ma.MaskedArray` cutout versions of the input
    ``error`` and ``background``.  The mask is `True` for pixels outside
    of the source segment, masked pixels from the ``mask`` input, or any
    non-finite ``data`` values (NaN and +/- inf).

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    def __init__(self, data, segment_img, label, filtered_data=None,
                 error=None, mask=None, background=None, wcs=None):

        if not isinstance(segment_img, SegmentationImage):
            segment_img = SegmentationImage(segment_img)

        if segment_img.shape != data.shape:
            raise ValueError('segment_img and data must have the same shape.')

        inputs = (data, filtered_data, error, background)
        has_unit = [hasattr(x, 'unit') for x in inputs if x is not None]
        use_units = all(has_unit)
        if any(has_unit) and not use_units:
            raise ValueError('If any of data, filtered_data, error, or '
                             'background has units, then they all must have '
                             'the same units.')

        if use_units:
            self._data_unit = data.unit
        else:
            self._data_unit = 1

        if error is not None:
            error = np.asanyarray(error)
            if error.shape != data.shape:
                raise ValueError('error and data must have the same shape.')
            if use_units and error.unit != self._data_unit:
                raise ValueError('error and data must have the same units.')

        if mask is np.ma.nomask:
            mask = None
        if mask is not None:
            mask = np.asanyarray(mask)
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape.')

        if background is not None:
            background = np.atleast_1d(background)
            if len(background) == 1:
                background = np.zeros(data.shape) + background
            else:
                background = np.asanyarray(background)
                if background.shape != data.shape:
                    raise ValueError('background and data must have the same '
                                     'shape.')
            if use_units and background.unit != self._data_unit:
                raise ValueError('background and data must have the same '
                                 'units.')

        if filtered_data is not None:
            filtered_data = np.asanyarray(filtered_data)
            if filtered_data.shape != data.shape:
                raise ValueError('filtered_data and data must have the same '
                                 'shape.')
            if use_units and filtered_data.unit != self._data_unit:
                raise ValueError('filtered_data and data must have the same '
                                 'units.')
            self._filtered_data = filtered_data
        else:
            self._filtered_data = data

        self._data = data
        self._segment_img = segment_img
        self._error = error
        self._mask = mask
        self._background = background  # 2D array
        self._wcs = wcs

        segment_img.check_labels(label)
        self.label = label

        self.segment = segment_img[segment_img.get_index(label)]
        self.slices = self.segment.slices

    def __str__(self):
        cls_name = '<{0}.{1}>'.format(self.__class__.__module__,
                                      self.__class__.__name__)

        cls_info = []
        params = ['label', 'sky_centroid']
        for param in params:
            cls_info.append((param, getattr(self, param)))
        fmt = (['{0}: {1}'.format(key, val) for key, val in cls_info])
        fmt.insert(1, 'centroid (x, y): ({0:0.4f}, {1:0.4f})'
                   .format(self.xcentroid.value, self.ycentroid.value))

        return '{}\n'.format(cls_name) + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    @lazyproperty
    def _segment_mask(self):
        """
        Boolean mask for source segment.

        ``_segment_mask`` is `True` for all pixels outside of the source
        segment for this label.  Pixels from other source segments
        within the rectangular cutout are `True`.
        """

        return self._segment_img.data[self.slices] != self.label

    @lazyproperty
    def _input_mask(self):
        """
        Boolean mask for the user-input mask.
        """

        if self._mask is not None:
            return self._mask[self.slices]
        else:
            return None

    @lazyproperty
    def _data_mask(self):
        """
        Boolean mask for non-finite (NaN and +/- inf) ``data`` values.
        """

        return ~np.isfinite(self.data_cutout)

    @lazyproperty
    def _total_mask(self):
        """
        Boolean mask representing the combination of the
        ``_segment_mask``, ``_input_mask``, and ``_data_mask``.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """

        mask = self._segment_mask | self._data_mask

        if self._input_mask is not None:
            mask |= self._input_mask

        return mask

    @lazyproperty
    def _is_completely_masked(self):
        """
        `True` if all pixels within the source segment are masked,
        otherwise `False`.
        """

        return np.all(self._total_mask)

    @lazyproperty
    def _data_zeroed(self):
        """
        A 2D `~numpy.ndarray` cutout from the input ``data`` where any
        masked pixels (``_segment_mask``, ``_input_mask``, or
        ``_data_mask``) are set to zero.  Invalid values (NaN and +/-
        inf) are set to zero via the ``_data_mask``.  Any units are
        dropped on the input ``data``.

        This is a 2D array representation (with zeros as placeholders
        for the masked/removed values) of the 1D ``_data_values``
        property, which is used for ``source_sum``, ``area``,
        ``min_value``, ``max_value``, ``minval_pos``, ``maxval_pos``,
        etc.
        """

        # NOTE: using np.where is faster than
        #     _data = np.copy(self.data_cutout)
        #     self._data[self._total_mask] = 0.
        return np.where(self._total_mask, 0,
                        self.data_cutout).astype(np.float64)  # copy

    @lazyproperty
    def _filtered_data_zeroed(self):
        """
        A 2D `~numpy.ndarray` cutout from the input ``filtered_data``
        (or ``data`` if ``filtered_data`` is `None`) where any masked
        pixels (``_segment_mask``, ``_input_mask``, or ``_data_mask``)
        are set to zero.  Invalid values (NaN and +/- inf) are set to
        zero.  Any units are dropped on the input ``filtered_data`` (or
        ``data``).

        Negative data values are also set to zero because negative
        pixels (especially at large radii) can result in image moments
        that result in negative variances.

        This array is used for moment-based properties.
        """

        filt_data = self._filtered_data[self.slices]
        filt_data = np.where(self._total_mask, 0., filt_data)  # copy
        filt_data[filt_data < 0] = 0.
        return filt_data.astype(np.float64)

    def make_cutout(self, data, masked_array=False):
        """
        Create a (masked) cutout array from the input ``data`` using the
        minimal bounding box of the source segment.

        If ``masked_array`` is `False` (default), then the returned
        cutout array is simply a `~numpy.ndarray`.  The returned cutout
        is a view (not a copy) of the input ``data``.  No pixels are
        altered (e.g. set to zero) within the bounding box.

        If ``masked_array` is `True`, then the returned cutout array is
        a `~numpy.ma.MaskedArray`.  The mask is `True` for pixels
        outside of the source segment (labeled region of interest),
        masked pixels from the ``mask`` input, or any non-finite
        ``data`` values (NaN and +/- inf).  The data part of the masked
        array is a view (not a copy) of the input ``data``.

        Parameters
        ----------
        data : array-like (2D)
            The data array from which to create the masked cutout array.
            ``data`` must have the same shape as the segmentation image
            input into `SourceProperties`.

        masked_array : bool, optional
            If `True` then a `~numpy.ma.MaskedArray` will be returned,
            where the mask is `True` for pixels outside of the source
            segment (labeled region of interest), masked pixels from the
            ``mask`` input, or any non-finite ``data`` values (NaN and
            +/- inf).  If `False`, then a `~numpy.ndarray` will be
            returned.

        Returns
        -------
        result : 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            The 2D cutout array.
        """

        data = np.asanyarray(data)
        if data.shape != self._segment_img.shape:
            raise ValueError('data must have the same shape as the '
                             'segmentation image input to SourceProperties')

        if masked_array:
            return np.ma.masked_array(data[self.slices],
                                      mask=self._total_mask)
        else:
            return data[self.slices]

    def to_table(self, columns=None, exclude_columns=None):
        """
        Create a `~astropy.table.QTable` of properties.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.QTable` will include a default list of
        scalar-valued properties.

        Parameters
        ----------
        columns : str or list of str, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`.  The allowed column names are any
            of the attributes of `SourceProperties`.

        exclude_columns : str or list of str, optional
            Names of columns to exclude from the default columns in the
            output `~astropy.table.QTable`.  The default columns are
            defined in the
            ``photutils.segmentation.properties.DEFAULT_COLUMNS``
            variable.

        Returns
        -------
        table : `~astropy.table.QTable`
            A single-row table of properties of the source.
        """

        return _properties_table(self, columns=columns,
                                 exclude_columns=exclude_columns)

    @lazyproperty
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source segment.
        """

        return self._data[self.slices]

    @lazyproperty
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """

        return np.ma.masked_array(self._data[self.slices],
                                  mask=self._total_mask)

    @lazyproperty
    def filtered_data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``filtered_data``.

        If ``filtered_data`` was not input, then the cutout will be from
        the input ``data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """

        return np.ma.masked_array(self._filtered_data[self.slices],
                                  mask=self._total_mask)

    @lazyproperty
    def error_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input ``error``
        image.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).

        If ``error`` is `None`, then ``error_cutout_ma`` is also `None`.
        """

        if self._error is None:
            return None
        else:
            return np.ma.masked_array(self._error[self.slices],
                                      mask=self._total_mask)

    @lazyproperty
    def background_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input
        ``background``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).

        If ``background`` is `None`, then ``background_cutout_ma`` is
        also `None`.
        """

        if self._background is None:
            return None
        else:
            return np.ma.masked_array(self._background[self.slices],
                                      mask=self._total_mask)

    @lazyproperty
    @deprecated('0.7')
    def values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``data`` values within the
        source segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked).

        If all pixels are masked, ``values`` will be an empty array.
        """

        return self._data_values  # pragma: no cover

    @lazyproperty
    def _data_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``data`` values within the
        source segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked) via the ``_data_mask``.

        If all pixels are masked, an empty array will be returned.

        This array is used for ``source_sum``, ``area``, ``min_value``,
        ``max_value``, ``minval_pos``, ``maxval_pos``, etc.
        """

        return self.data_cutout_ma.compressed()

    @lazyproperty
    def _filtered_data_values(self):
        return self.filtered_data_cutout_ma.compressed()

    @lazyproperty
    def _error_values(self):
        return self.error_cutout_ma.compressed()

    @lazyproperty
    def _background_values(self):
        return self.background_cutout_ma.compressed()

    @lazyproperty
    def indices(self):
        """
        A tuple of two `~numpy.ndarray` containing the ``y`` and ``x``
        pixel indices, respectively, of unmasked pixels within the
        source segment.

        Non-finite ``data`` values (NaN and +/- inf) are excluded.

        If all ``data`` pixels are masked, a tuple of two empty arrays
        will be returned.
        """

        yindices, xindices = np.nonzero(self.data_cutout_ma)
        return (yindices + self.slices[0].start,
                xindices + self.slices[1].start)

    @lazyproperty
    @deprecated('0.7', 'indices')
    def coords(self):
        """
        A tuple of two `~numpy.ndarray` containing the ``y`` and ``x``
        pixel indices, respectively, of unmasked pixels within the
        source segment.

        Non-finite ``data`` values (NaN and +/- inf) are excluded.

        If all ``data`` pixels are masked, a tuple of two empty arrays
        will be returned.
        """

        return self.indices  # pragma: no cover

    @lazyproperty
    def moments(self):
        """Spatial moments up to 3rd order of the source."""

        return _moments(self._filtered_data_zeroed, order=3)

    @lazyproperty
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """

        ycentroid, xcentroid = self.cutout_centroid.value
        return _moments_central(self._filtered_data_zeroed,
                                center=(xcentroid, ycentroid), order=3)

    @lazyproperty
    def id(self):
        """
        The source identification number corresponding to the object
        label in the segmentation image.
        """

        return self.label

    @lazyproperty
    def cutout_centroid(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the centroid within the source segment.
        """

        moments = self.moments
        if moments[0, 0] != 0:
            ycentroid = moments[1, 0] / moments[0, 0]
            xcentroid = moments[0, 1] / moments[0, 0]
            return (ycentroid, xcentroid) * u.pix
        else:
            return (np.nan, np.nan) * u.pix

    @lazyproperty
    def centroid(self):
        """
        The ``(y, x)`` coordinate of the centroid within the source
        segment.
        """

        ycen, xcen = self.cutout_centroid.value
        return (ycen + self.slices[0].start,
                xcen + self.slices[1].start) * u.pix

    @lazyproperty
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """

        return self.centroid[1]

    @lazyproperty
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """

        return self.centroid[0]

    @lazyproperty
    def sky_centroid(self):
        """
        The sky coordinates of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input WCS.
        """

        return _pixel_to_world(self.xcentroid.value, self.ycentroid.value,
                               self._wcs)

    @lazyproperty
    def sky_centroid_icrs(self):
        """
        The sky coordinates, in the International Celestial Reference
        System (ICRS) frame, of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.
        """

        if self._wcs is None:
            return None
        else:
            return self.sky_centroid.icrs

    @lazyproperty
    def bbox(self):
        """
        The `~photutils.BoundingBox` of the minimal rectangular region
        containing the source segment.
        """

        return BoundingBox(self.slices[1].start, self.slices[1].stop,
                           self.slices[0].start, self.slices[0].stop)

    @lazyproperty
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel location within the minimal bounding box
        containing the source segment.
        """

        return self.bbox.ixmin * u.pix

    @lazyproperty
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """

        return (self.bbox.ixmax - 1) * u.pix

    @lazyproperty
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel location within the minimal bounding box
        containing the source segment.
        """

        return self.bbox.iymin * u.pix

    @lazyproperty
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """

        return (self.bbox.iymax - 1) * u.pix

    @lazyproperty
    @deprecated('0.7', 'bbox_xmin')
    def xmin(self):
        """
        The minimum ``x`` pixel location within the minimal bounding box
        containing the source segment.
        """

        return self.bbox_xmin  # pragma: no cover

    @lazyproperty
    @deprecated('0.7', 'bbox_xmax')
    def xmax(self):
        """
        The maximum ``x`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """

        return self.bbox_xmax  # pragma: no cover

    @lazyproperty
    @deprecated('0.7', 'bbox_ymin')
    def ymin(self):
        """
        The minimum ``y`` pixel location within the minimal bounding box
        containing the source segment.
        """

        return self.bbox_ymin  # pragma: no cover

    @lazyproperty
    @deprecated('0.7', 'bbox_ymax')
    def ymax(self):
        """
        The maximum``y`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """

        return self.bbox_ymax  # pragma: no cover

    @lazyproperty
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        return _calc_sky_bbox_corner(self.bbox, 'll', self._wcs)

    @lazyproperty
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        return _calc_sky_bbox_corner(self.bbox, 'ul', self._wcs)

    @lazyproperty
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        return _calc_sky_bbox_corner(self.bbox, 'lr', self._wcs)

    @lazyproperty
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        return _calc_sky_bbox_corner(self.bbox, 'ur', self._wcs)

    @lazyproperty
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit
        else:
            return np.min(self._data_values)

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value of the ``data`` within the source
        segment.
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit
        else:
            return np.max(self._data_values)

    @lazyproperty
    def minval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            arr = self.data_cutout_ma
            # multiplying by unit converts int to float, but keep as
            # float in case the array contains a NaN
            return np.asarray(np.unravel_index(np.argmin(arr),
                                               arr.shape)) * u.pix

    @lazyproperty
    def maxval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            arr = self.data_cutout_ma
            # multiplying by unit converts int to float, but keep as
            # float in case the array contains a NaN
            return np.asarray(np.unravel_index(np.argmax(arr),
                                               arr.shape)) * u.pix

    @lazyproperty
    def minval_pos(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            yposition, xposition = self.minval_cutout_pos.value
            return (yposition + self.slices[0].start,
                    xposition + self.slices[1].start) * u.pix

    @lazyproperty
    def maxval_pos(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            yposition, xposition = self.maxval_cutout_pos.value
            return (yposition + self.slices[0].start,
                    xposition + self.slices[1].start) * u.pix

    @lazyproperty
    def minval_xpos(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        return self.minval_pos[1]

    @lazyproperty
    def minval_ypos(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        return self.minval_pos[0]

    @lazyproperty
    def maxval_xpos(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        return self.maxval_pos[1]

    @lazyproperty
    def maxval_ypos(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        return self.maxval_pos[0]

    @lazyproperty
    def source_sum(self):
        """
        The sum of the unmasked ``data`` values within the source segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``source_sum``, :math:`(I_i - B_i)` is the
        ``data``, and :math:`S` are the unmasked pixels in the source
        segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked).
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit  # table output needs unit
        else:
            return np.sum(self._data_values)

    @lazyproperty
    def source_sum_err(self):
        """
        The uncertainty of `~photutils.SourceProperties.source_sum`,
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
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the error array.
        """

        if self._error is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # table output needs unit
            else:
                return np.sqrt(np.sum(self._error_values ** 2))
        else:
            return None

    @lazyproperty
    def background_sum(self):
        """
        The sum of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the background array.
        """

        if self._background is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # unit for table
            else:
                return np.sum(self._background_values)
        else:
            return None

    @lazyproperty
    def background_mean(self):
        """
        The mean of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the background array.
        """

        if self._background is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # unit for table
            else:
                return np.mean(self._background_values)
        else:
            return None

    @lazyproperty
    def background_at_centroid(self):
        """
        The value of the ``background`` at the position of the source
        centroid.

        The background value at fractional position values are
        determined using bilinear interpolation.
        """

        if self._background is not None:
            from scipy.ndimage import map_coordinates

            # centroid can be NaN if segment is completely masked or if
            # all data values are <= 0
            if np.any(~np.isfinite(self.centroid)):
                return np.nan * self._data_unit  # unit for table
            else:
                value = map_coordinates(self._background,
                                        [[self.ycentroid.value],
                                         [self.xcentroid.value]], order=1,
                                        mode='nearest')[0]
                return value * self._data_unit
        else:
            return None

    @lazyproperty
    def area(self):
        """
        The total unmasked area of the source segment in units of
        pixels**2.

        Note that the source area may be smaller than its segment area
        if a mask is input to `SourceProperties` or `source_properties`,
        or if the ``data`` within the segment contains invalid values
        (NaN and +/- inf).
        """

        if self._is_completely_masked:
            return np.nan * u.pix**2
        else:
            return len(self._data_values) * u.pix**2

    @lazyproperty
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """

        return np.sqrt(self.area / np.pi)

    @lazyproperty
    def perimeter(self):
        """
        The perimeter of the source segment, approximated as the total
        length of lines connecting the centers of the border pixels
        defined by a 4-pixel connectivity.

        If any masked pixels make holes within the source segment, then
        the perimeter around the inner hole (e.g. an annulus) will also
        contribute to the total perimeter.

        References
        ----------
        .. [1] K. Benkrid, D. Crookes, and A. Benkrid.  "Design and FPGA
               Implementation of a Perimeter Estimator".  Proceedings of
               the Irish Machine Vision and Image Processing Conference,
               pp. 51-57 (2000).
               http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
        """

        if self._is_completely_masked:
            return np.nan * u.pix  # unit for table
        else:
            from scipy.ndimage import binary_erosion, convolve

            data = ~self._total_mask
            selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            data_eroded = binary_erosion(data, selem, border_value=0)
            border = np.logical_xor(data, data_eroded).astype(np.int)

            kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]])
            perimeter_data = convolve(border, kernel, mode='constant', cval=0)

            size = 34
            perimeter_hist = np.bincount(perimeter_data.ravel(),
                                         minlength=size)

            weights = np.zeros(size, dtype=np.float)
            weights[[5, 7, 15, 17, 25, 27]] = 1.
            weights[[21, 33]] = np.sqrt(2.)
            weights[[13, 23]] = (1 + np.sqrt(2.)) / 2.

            return (perimeter_hist[0:size] @ weights) * u.pix

    @lazyproperty
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """

        moments = self.moments_central
        mu_02 = moments[0, 2]
        mu_11 = -moments[1, 1]
        mu_20 = moments[2, 0]
        return np.array([[mu_02, mu_11], [mu_11, mu_20]]) * u.pix**2

    @lazyproperty
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """

        moments = self.moments_central
        if moments[0, 0] != 0:
            mu_norm = moments / moments[0, 0]
            covariance = self._check_covariance(
                np.array([[mu_norm[0, 2], mu_norm[1, 1]],
                          [mu_norm[1, 1], mu_norm[2, 0]]]))
            return covariance * u.pix**2
        else:
            return np.empty((2, 2)) * np.nan * u.pix**2

    @staticmethod
    def _check_covariance(covariance):
        """
        Check and modify the covariance matrix in the case of
        "infinitely" thin detections.  This follows SExtractor's
        prescription of incrementally increasing the diagonal elements
        by 1/12.
        """

        increment = 1. / 12  # arbitrary SExtractor value
        value = (covariance[0, 0] * covariance[1, 1]) - covariance[0, 1]**2
        if value >= increment**2:
            return covariance
        else:
            covar = np.copy(covariance)
            while value < increment**2:
                covar[0, 0] += increment
                covar[1, 1] += increment
                value = (covar[0, 0] * covar[1, 1]) - covar[0, 1]**2
            return covar

    @lazyproperty
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """

        if np.any(~np.isfinite(self.covariance)):
            return (np.nan, np.nan) * u.pix**2
        else:
            eigvals = np.linalg.eigvals(self.covariance)
            if np.any(eigvals < 0):  # negative variance
                return (np.nan, np.nan) * u.pix**2  # pragma: no cover
            return (np.max(eigvals), np.min(eigvals)) * u.pix**2

    @lazyproperty
    def semimajor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """

        # this matches SExtractor's A parameter
        return np.sqrt(self.covariance_eigvals[0])

    @lazyproperty
    def semiminor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """

        # this matches SExtractor's B parameter
        return np.sqrt(self.covariance_eigvals[1])

    @lazyproperty
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

        semimajor_var, semiminor_var = self.covariance_eigvals
        if semimajor_var == 0:
            return 0.  # pragma: no cover
        return np.sqrt(1. - (semiminor_var / semimajor_var))

    @lazyproperty
    def orientation(self):
        """
        The angle in radians between the ``x`` axis and the major axis
        of the 2D Gaussian function that has the same second-order
        moments as the source.  The angle increases in the
        counter-clockwise direction.
        """

        covar_00, covar_01, _, covar_11 = self.covariance.flat
        if covar_00 < 0 or covar_11 < 0:  # negative variance
            return np.nan * u.deg  # pragma: no cover

        # Quantity output in radians because inputs are Quantities
        orient_radians = 0.5 * np.arctan2(2. * covar_01,
                                          (covar_00 - covar_11))
        return orient_radians.to(u.deg)

    @lazyproperty
    def elongation(self):
        """
        The ratio of the lengths of the semimajor and semiminor axes:

        .. math:: \\mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

        Note that this is the same as `SExtractor`_'s elongation
        parameter.
        """

        return self.semimajor_axis_sigma / self.semiminor_axis_sigma

    @lazyproperty
    def ellipticity(self):
        """
        ``1`` minus the ratio of the lengths of the semimajor and
        semiminor axes (or ``1`` minus the `elongation`):

        .. math:: \\mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

        Note that this is the same as `SExtractor`_'s ellipticity
        parameter.
        """

        return 1.0 - (self.semiminor_axis_sigma / self.semimajor_axis_sigma)

    @lazyproperty
    def covar_sigx2(self):
        """
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\\sigma_x^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s X2 parameter.
        """

        return self.covariance[0, 0]

    @lazyproperty
    def covar_sigy2(self):
        """
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\\sigma_y^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s Y2 parameter.
        """

        return self.covariance[1, 1]

    @lazyproperty
    def covar_sigxy(self):
        """
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\\sigma_x \\sigma_y`, in units of
        pixel**2.

        Note that this is the same as `SExtractor`_'s XY parameter.
        """

        return self.covariance[0, 1]

    @lazyproperty
    def cxx(self):
        """
        `SExtractor`_'s CXX ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return ((np.cos(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.sin(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def cyy(self):
        """
        `SExtractor`_'s CYY ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return ((np.sin(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.cos(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def cxy(self):
        """
        `SExtractor`_'s CXY ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return (2. * np.cos(self.orientation) * np.sin(self.orientation) *
                ((1. / self.semimajor_axis_sigma**2) -
                 (1. / self.semiminor_axis_sigma**2)))

    @lazyproperty
    def gini(self):
        """
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <http://adsabs.harvard.edu/abs/2004AJ....128..163L>`_ as:

        .. math::
            G = \\frac{1}{\\left | \\bar{x} \\right | n (n - 1)}
            \\sum^{n}_{i} (2i - n - 1) \\left | x_i \\right |

        where :math:`\\bar{x}` is the mean over all pixel values
        :math:`x_i`.

        The Gini coefficient is a way of measuring the inequality in a
        given set of values.  In the context of galaxy morphology, it
        measures how the light of a galaxy image is distributed among
        its pixels.  A Gini coefficient value of 0 corresponds to a
        galaxy image with the light evenly distributed over all pixels
        while a Gini coefficient value of 1 represents a galaxy image
        with all its light concentrated in just one pixel.
        """

        npix = np.size(self._data_values)
        normalization = (np.abs(np.mean(self._data_values)) * npix *
                         (npix - 1))
        kernel = ((2. * np.arange(1, npix + 1) - npix - 1) *
                  np.abs(np.sort(self._data_values)))

        return np.sum(kernel) / normalization


def source_properties(data, segment_img, error=None, mask=None,
                      background=None, filter_kernel=None, wcs=None,
                      labels=None):
    """
    Calculate photometry and morphological properties of sources defined
    by a labeled segmentation image.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.  ``data`` should be background-subtracted.
        Non-finite ``data`` values (NaN and +/- inf) are automatically
        masked.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data`` array.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  Non-finite ``error`` values
        (NaN and +/- inf) are not automatically masked, unless they are
        at the same position of non-finite values in the input ``data``
        array.  Such pixels can be masked using the ``mask`` keyword.
        See the Notes section below for details on the error
        propagation.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.  Non-finite
        values (NaN and +/- inf) in the input ``data`` are automatically
        masked.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.  Non-finite ``background`` values (NaN
        and +/- inf) are not automatically masked, unless they are at
        the same position of non-finite values in the input ``data``
        array.  Such pixels can be masked using the ``mask`` keyword.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the data prior to
        calculating the source centroid and morphological parameters.
        The kernel should be the same one used in defining the source
        segments, i.e. the detection image (e.g., see
        :func:`~photutils.detect_sources`).  If `None`, then the
        unfiltered ``data`` will be used instead.

    wcs : `None` or WCS object, optional
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).  If `None`, then all sky-based
        properties will be set to `None`.

    labels : int, array-like (1D, int)
        The segmentation labels for which to calculate source
        properties.  If `None` (default), then the properties will be
        calculated for all labeled sources.

    Returns
    -------
    output : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.

    Notes
    -----
    `SExtractor`_'s centroid and morphological parameters are always
    calculated from a filtered "detection" image, i.e. the image used to
    define the segmentation image.  The usual downside of the filtering
    is the sources will be made more circular than they actually are.
    If you wish to reproduce `SExtractor`_ centroid and morphology
    results, then input a filtered and background-subtracted "detection"
    image into the ``filtered_data`` keyword.  If ``filtered_data`` is
    `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values (``filtered_data`` or ``data``) within the
    source segment are set to zero when calculating morphological
    properties based on image moments.  Negative values could occur, for
    example, if the segmentation image was defined from a different
    image (e.g., different bandpass) or if the background was
    oversubtracted. Note that `~photutils.SourceProperties.source_sum`
    always includes the contribution of negative ``data`` values.

    The input ``error`` is assumed to include *all* sources of error,
    including the Poisson error of the sources.
    `~photutils.SourceProperties.source_sum_err` is simply the
    quadrature sum of the pixel-wise total errors over the non-masked
    pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.SourceProperties.source_sum_err`, :math:`S` are the
    non-masked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    .. _SExtractor: http://www.astromatic.net/software/sextractor

    See Also
    --------
    SegmentationImage, SourceProperties, detect_sources

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import SegmentationImage, source_properties
    >>> image = np.arange(16.).reshape(4, 4)
    >>> print(image)  # doctest: +SKIP
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]
     [12. 13. 14. 15.]]
    >>> segm = SegmentationImage([[1, 1, 0, 0],
    ...                           [1, 0, 0, 2],
    ...                           [0, 0, 2, 2],
    ...                           [0, 2, 2, 0]])
    >>> props = source_properties(image, segm)

    Print some properties of the first object (labeled with ``1`` in the
    segmentation image):

    >>> props[0].id  # id corresponds to segment label number
    1
    >>> props[0].centroid  # doctest: +FLOAT_CMP
    <Quantity [0.8, 0.2] pix>
    >>> props[0].source_sum  # doctest: +FLOAT_CMP
    5.0
    >>> props[0].area  # doctest: +FLOAT_CMP
    <Quantity 3. pix2>
    >>> props[0].max_value  # doctest: +FLOAT_CMP
    4.0

    Print some properties of the second object (labeled with ``2`` in
    the segmentation image):

    >>> props[1].id  # id corresponds to segment label number
    2
    >>> props[1].centroid  # doctest: +FLOAT_CMP
    <Quantity [2.36363636, 2.09090909] pix>
    >>> props[1].perimeter  # doctest: +FLOAT_CMP
    <Quantity 5.41421356 pix>
    >>> props[1].orientation  # doctest: +FLOAT_CMP
    <Quantity -42.4996777 deg>
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('segment_img and data must have the same shape.')

    # filter the data once, instead of repeating for each source
    if filter_kernel is not None:
        filtered_data = _filter_data(data, filter_kernel, mode='constant',
                                     fill_value=0.0, check_normalization=True)
    else:
        filtered_data = None

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)

    sources_props = []
    for label in labels:
        if label not in segment_img.labels:
            warnings.warn('label {} is not in the segmentation image.'
                          .format(label), AstropyUserWarning)
            continue  # skip invalid labels

        sources_props.append(SourceProperties(
            data, segment_img, label, filtered_data=filtered_data,
            error=error, mask=mask, background=background, wcs=wcs))

    if not sources_props:
        raise ValueError('No sources are defined.')

    return SourceCatalog(sources_props, wcs=wcs)


class SourceCatalog:
    """
    Class to hold source catalogs.
    """

    def __init__(self, properties_list, wcs=None):
        if isinstance(properties_list, SourceProperties):
            self._data = [properties_list]
        elif isinstance(properties_list, list):
            if not properties_list:
                raise ValueError('properties_list must not be an empty list.')
            self._data = properties_list
        else:
            raise ValueError('invalid input.')

        self.wcs = wcs
        self._cache = {}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __delitem__(self, index):
        del self._data[index]

    def __iter__(self):
        for i in self._data:
            yield i

    def __str__(self):
        cls_name = '<{0}.{1}>'.format(self.__class__.__module__,
                                      self.__class__.__name__)
        fmt = ['Catalog length: {0}'.format(len(self))]

        return '{}\n'.format(cls_name) + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        if attr not in self._cache:
            values = [getattr(p, attr) for p in self._data]

            if isinstance(values[0], u.Quantity):
                # turn list of Quantities into a Quantity array
                values = u.Quantity(values)
            if isinstance(values[0], SkyCoord):  # pragma: no cover
                # failsafe: turn list of SkyCoord into a SkyCoord array
                values = SkyCoord(values)

            self._cache[attr] = values

        return self._cache[attr]

    @lazyproperty
    def _none_list(self):
        """
        Return a list of `None` values, used by SkyCoord properties if
        ``wcs`` is `None`.
        """

        return [None] * len(self._data)

    @lazyproperty
    def background_at_centroid(self):
        background = self._data[0]._background
        if background is None:
            return self._none_list
        else:
            from scipy.ndimage import map_coordinates

            values = map_coordinates(background,
                                     [[self.ycentroid.value],
                                      [self.xcentroid.value]], order=1,
                                     mode='nearest')[0]

            mask = np.isfinite(self.xcentroid) & np.isfinite(self.ycentroid)
            values[~mask] = np.nan

            return values * self._data[0]._data_unit

    @lazyproperty
    def sky_centroid(self):
        if self.wcs is None:
            return self._none_list
        else:
            # For a large catalog, it's much faster to calculate world
            # coordinates using the complete list of (x, y) instead of
            # looping through the individual (x, y).  It's also much
            # faster to recalculate the world coordinates than to create a
            # SkyCoord array from a loop-generated SkyCoord list.  The
            # assumption here is that the wcs is the same for each
            # SourceProperties instance.
            return _pixel_to_world(self.xcentroid, self.ycentroid, self.wcs)

    @lazyproperty
    def sky_centroid_icrs(self):
        if self.wcs is None:
            return self._none_list
        else:
            return self.sky_centroid.icrs

    @lazyproperty
    def sky_bbox_ll(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'll', self.wcs)

    @lazyproperty
    def sky_bbox_ul(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'ul', self.wcs)

    @lazyproperty
    def sky_bbox_lr(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'lr', self.wcs)

    @lazyproperty
    def sky_bbox_ur(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'ur', self.wcs)

    def to_table(self, columns=None, exclude_columns=None):
        """
        Construct a `~astropy.table.QTable` of source properties from a
        `SourceCatalog` object.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.QTable` will include a default list of
        scalar-valued properties.

        Multi-dimensional properties, e.g.
        `~photutils.SourceProperties.data_cutout`, can be included in
        the ``columns`` input, but they will not be preserved when
        writing the table to a file.  This is a limitation of
        multi-dimensional columns in astropy tables.

        Parameters
        ----------
        columns : str or list of str, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`.  The allowed column names are any
            of the attributes of `SourceProperties`.

        exclude_columns : str or list of str, optional
            Names of columns to exclude from the default columns in the
            output `~astropy.table.QTable`.  The default columns are
            defined in the
            ``photutils.segmentation.properties.DEFAULT_COLUMNS``
            variable.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of source properties with one row per source.

        See Also
        --------
        SegmentationImage, SourceProperties, source_properties, detect_sources

        Examples
        --------
        >>> import numpy as np
        >>> from photutils import source_properties
        >>> image = np.arange(16.).reshape(4, 4)
        >>> print(image)  # doctest: +SKIP
        [[ 0.  1.  2.  3.]
         [ 4.  5.  6.  7.]
         [ 8.  9. 10. 11.]
         [12. 13. 14. 15.]]
        >>> segm = SegmentationImage([[1, 1, 0, 0],
        ...                           [1, 0, 0, 2],
        ...                           [0, 0, 2, 2],
        ...                           [0, 2, 2, 0]])
        >>> cat = source_properties(image, segm)
        >>> columns = ['id', 'xcentroid', 'ycentroid', 'source_sum']
        >>> tbl = cat.to_table(columns=columns)
        >>> tbl['xcentroid'].info.format = '.10f'  # optional format
        >>> tbl['ycentroid'].info.format = '.10f'  # optional format
        >>> print(tbl)
        id  xcentroid    ycentroid   source_sum
                pix          pix
        --- ------------ ------------ ----------
        1 0.2000000000 0.8000000000        5.0
        2 2.0909090909 2.3636363636       55.0
        """

        return _properties_table(self, columns=columns,
                                 exclude_columns=exclude_columns)


def _properties_table(obj, columns=None, exclude_columns=None):
    """
    Construct a `~astropy.table.QTable` of source properties from a
    `SourceProperties` or `SourceCatalog` object.

    Parameters
    ----------
    obj : `SourceProperties` or `SourceCatalog` instance
        The object containing the source properties.

    columns : str or list of str, optional
        Names of columns, in order, to include in the output
        `~astropy.table.QTable`.  The allowed column names are any
        of the attributes of `SourceProperties`.

    exclude_columns : str or list of str, optional
        Names of columns to exclude from the default columns in the
        output `~astropy.table.QTable`.  The default columns are defined
        in the ``photutils.segmentation.properties.DEFAULT_COLUMNS``
        variable.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of source properties with one row per source.
    """

    # start with the default columns
    columns_all = DEFAULT_COLUMNS

    table_columns = None
    if exclude_columns is not None:
        table_columns = [s for s in columns_all if s not in exclude_columns]
    if columns is not None:
        table_columns = np.atleast_1d(columns)
    if table_columns is None:
        table_columns = columns_all

    tbl = QTable()
    for column in table_columns:
        values = getattr(obj, column)

        if isinstance(obj, SourceProperties):
            # turn scalar values into length-1 arrays because QTable
            # column assignment requires an object with a length
            values = np.atleast_1d(values)

            # Unfortunately np.atleast_1d creates an array of SkyCoord
            # instead of a SkyCoord array (Quantity does work correctly
            # with np.atleast_1d).  Here we make a SkyCoord array for
            # the output table column.
            if isinstance(values[0], SkyCoord):
                values = SkyCoord(values)  # length-1 SkyCoord array

        tbl[column] = values

    return tbl


def _calc_sky_bbox_corner(bbox, corner, wcs):
    """
    Calculate the sky coordinates at the corner of a minimal bounding
    box.

    The bounding box encloses all of the source segment pixels in their
    entirety, thus the vertices are at the pixel *corners*.

    Parameters
    ----------
    bbox : `~photutils.BoundingBox`
        The source bounding box.

    corner : {'ll', 'ul', 'lr', 'ur'}
        The desired bounding box corner:
            * 'll':  lower left
            * 'ul':  upper left
            * 'lr':  lower right
            * 'ur':  upper right

    wcs : `None` or WCS object
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    skycoord : `~astropy.coordinates.SkyCoord` or `None`
        The sky coordinate at the bounding box corner.  If ``wcs`` is
        `None`, then `None` will be returned.
    """

    if corner == 'll':
        xpos = bbox.ixmin - 0.5
        ypos = bbox.iymin - 0.5
    elif corner == 'ul':
        xpos = bbox.ixmin - 0.5
        ypos = bbox.iymax + 0.5
    elif corner == 'lr':
        xpos = bbox.ixmax + 0.5
        ypos = bbox.iymin - 0.5
    elif corner == 'ur':
        xpos = bbox.ixmax + 0.5
        ypos = bbox.iymax + 0.5
    else:
        raise ValueError('Invalid corner name.')

    return _pixel_to_world(xpos, ypos, wcs)
