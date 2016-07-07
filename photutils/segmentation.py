# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
import numpy as np
from astropy.table import Table
from astropy.utils import lazyproperty
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord
from .utils.convolution import _convolve_data
from .utils.prepare_data import _prepare_data


__all__ = ['SegmentationImage', 'SourceProperties', 'source_properties',
           'properties_table']

# outline_segments requires scikit-image >= 0.11
__doctest_skip__ = {'SegmentationImage.outline_segments'}

__doctest_requires__ = {('SegmentationImage', 'SegmentationImage.*',
                         'SourceProperties', 'SourceProperties.*',
                         'source_properties', 'properties_table'): ['scipy'],
                        ('SegmentationImage', 'SegmentationImage.*',
                         'SourceProperties', 'SourceProperties.*',
                         'source_properties', 'properties_table'):
                        ['skimage']}


class SegmentationImage(object):
    """
    Class for a segmentation image.

    Parameters
    ----------
    data : array_like (int)
        A 2D segmentation image where sources are labeled by different
        positive integer values.  A value of zero is reserved for the
        background.
    """

    def __init__(self, data):

        self.data = np.asanyarray(data, dtype=np.int)

    @property
    def data(self):
        """
        The 2D segmentation image.
        """

        return self._data

    @data.setter
    def data(self, value):
        if np.min(value) < 0:
            raise ValueError('The segmentation image cannot contain '
                             'negative integers.')
        self._data = value
        # be sure to delete any lazy properties to reset their values.
        del (self.data_masked, self.shape, self.labels, self.nlabels,
             self.max, self.slices, self.is_sequential)

    @property
    def array(self):
        """
        The 2D segmentation image.
        """

        return self._data

    def __array__(self):
        """
        Array representation of the segmentation image (e.g., for
        matplotlib).
        """

        return self._data

    @lazyproperty
    def data_masked(self):
        """
        A `~numpy.ma.MaskedArray` version of the segmentation image
        where the background (label = 0) has been masked.
        """

        return np.ma.masked_where(self.data == 0, self.data)

    @staticmethod
    def _labels(data):
        """
        Return a sorted array of the non-zero labels in the segmentation
        image.

        Parameters
        ----------
        data : array_like (int)
            A 2D segmentation image where sources are labeled by
            different positive integer values.  A value of zero is
            reserved for the background.

        Returns
        -------
        result : `~numpy.ndarray`
            An array of non-zero label numbers.

        Notes
        -----
        This is a separate static method so it can be used on masked
        versions of the segmentation image (cf.
        ``~photutils.SegmentationImage.remove_masked_labels``.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm._labels(segm.data)
        array([1, 3, 4, 5, 7])
        """

        return np.unique(data[data != 0])

    @lazyproperty
    def shape(self):
        """
        The shape of the 2D segmentation image.
        """

        return self._data.shape

    @lazyproperty
    def labels(self):
        """The sorted non-zero labels in the segmentation image."""

        return self._labels(self.data)

    @lazyproperty
    def nlabels(self):
        """The number of non-zero labels in the segmentation image."""

        return len(self.labels)

    @lazyproperty
    def max(self):
        """The maximum non-zero label in the segmentation image."""

        return np.max(self.data)

    @lazyproperty
    def slices(self):
        """The minimal bounding box slices for each labeled region."""

        from scipy.ndimage import find_objects
        return find_objects(self._data)

    @lazyproperty
    def is_sequential(self):
        """
        Determine whether or not the non-zero labels in the segmenation
        image are sequential (with no missing values).
        """

        if (self.labels[-1] - self.labels[0] + 1) == self.nlabels:
            return True
        else:
            return False

    def check_label(self, label):
        """
        Check for a valid label label number within the segmentation
        image.

        Parameters
        ----------
        label : int
            The label number to check.

        Raises
        ------
        ValueError
            If the input ``label`` is invalid.
        """

        if label == 0:
            raise ValueError('label "0" is reserved for the background')
        if label < 0:
            raise ValueError('label must be a positive integer, got '
                             '"{0}"'.format(label))
        if label not in self.labels:
            raise ValueError('label "{0}" is not in the segmentation '
                             'image'.format(label))

    def outline_segments(self, mask_background=False):
        """
        Outline the labeled segments.

        The "outlines" represent the pixels *just inside* the segments,
        leaving the background pixels unmodified.  This corresponds to
        the ``mode='inner'`` in `skimage.segmentation.find_boundaries`.

        Parameters
        ----------
        mask_background : bool, optional
            Set to `True` to mask the background pixels (labels = 0) in
            the returned image.  This is useful for overplotting the
            segment outlines on an image.  The default is `False`.

        Returns
        -------
        boundaries : 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            An image with the same shape of the segmenation image
            containing only the outlines of the labeled segments.  The
            pixel values in the outlines correspond to the labels in the
            segmentation image.  If ``mask_background`` is `True`, then
            a `~numpy.ma.MaskedArray` is returned.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[0, 0, 0, 0, 0, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 0, 0, 0, 0, 0]])
        >>> segm.outline_segments()
        array([[0, 0, 0, 0, 0, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 0, 0, 0, 0, 0]])
        """

        import skimage
        if LooseVersion(skimage.__version__) < LooseVersion('0.11'):
            raise ImportError('The outline_segments() function requires '
                              'scikit-image >= 0.11')
        from skimage.segmentation import find_boundaries

        outlines = self.data * find_boundaries(self.data, mode='inner')
        if mask_background:
            outlines = np.ma.masked_where(outlines == 0, outlines)
        return outlines

    def relabel(self, labels, new_label):
        """
        Relabel one or more label numbers.

        The input ``labels`` will all be relabeled to ``new_label``.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to relabel.

        new_label : int
            The relabeled label number.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.relabel(labels=[1, 7], new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [2, 0, 0, 0, 0, 5],
               [2, 2, 0, 5, 5, 5],
               [2, 2, 0, 0, 5, 5]])
        """

        labels = np.atleast_1d(labels)
        for label in labels:
            data = self.data
            data[np.where(data == label)] = new_label
            self.data = data     # needed to call the data setter

    def relabel_sequential(self, start_label=1):
        """
        Relabel the label numbers sequentially, such that there are no
        missing label numbers (up to the maximum label number).

        Parameters
        ----------
        start_label : int, optional
            The starting label number, which should be a positive
            integer.  The default is 1.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.relabel_sequential()
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [5, 0, 0, 0, 0, 4],
               [5, 5, 0, 4, 4, 4],
               [5, 5, 0, 0, 4, 4]])
        """

        if start_label <= 0:
            raise ValueError('start_label must be > 0.')

        if self.is_sequential and (self.labels[0] == start_label):
            return

        forward_map = np.zeros(self.max + 1, dtype=np.int)
        forward_map[self.labels] = np.arange(self.nlabels) + start_label
        self.data = forward_map[self.data]

    def keep_labels(self, labels, relabel=False):
        """
        Keep only the specified label numbers.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to keep.  Labels of zero and those not
            in the segmentation image will be ignored.

        relabel : bool, optional
            If `True`, then the segmentation image will be relabeled
            such that the labels are in sequential order starting from
            1.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=3)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=[5, 3])
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 5],
               [0, 0, 0, 5, 5, 5],
               [0, 0, 0, 0, 5, 5]])
        """

        labels = np.atleast_1d(labels)
        labels_tmp = list(set(self.labels) - set(labels))
        self.remove_labels(labels_tmp, relabel=relabel)

    def remove_labels(self, labels, relabel=False):
        """
        Remove one or more label numbers.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to remove.  Labels of zero and those not
            in the segmentation image will be ignored.

        relabel : bool, optional
            If `True`, then the segmentation image will be relabeled
            such that the labels are in sequential order starting from
            1.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=5)
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=[5, 3])
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])
        """

        self.relabel(labels, new_label=0)
        if relabel:
            self.relabel_sequential()

    def remove_border_labels(self, border_width, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments near the image border.

        Labels within the defined border region will be removed.

        Parameters
        ----------
        border_width : int
            The width of the border region in pixels.

        partial_overlap : bool, optional
            If this is set to `True` (the default), a segment that
            partially extends into the border region will be removed.
            Segments that are completely within the border region are
            always removed.

        relabel : bool, optional
            If `True`, then the segmentation image will be relabeled
            such that the labels are in sequential order starting from
            1.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1,
        ...                           partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """

        if border_width >= min(self.shape) / 2:
            raise ValueError('border_width must be smaller than half the '
                             'image size in either dimension')
        border = np.zeros(self.shape, dtype=np.bool)
        border[:border_width, :] = True
        border[-border_width:, :] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True
        self.remove_masked_labels(border, partial_overlap=partial_overlap,
                                  relabel=relabel)

    def remove_masked_labels(self, mask, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments located within a masked region.

        Parameters
        ----------
        mask : array_like (bool)
            A boolean mask, with the same shape as the segmentation
            image (``.data``), where `True` values indicate masked
            pixels.

        partial_overlap : bool, optional
            If this is set to `True` (the default), a segment that
            partially extends into a masked region will also be removed.
            Segments that are completely within a masked region are
            always removed.

        relabel : bool, optional
            If `True`, then the segmentation image will be relabeled
            such that the labels are in sequential order starting from
            1.

        Examples
        --------
        >>> from photutils import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> mask = np.zeros_like(segm.data, dtype=np.bool)
        >>> mask[0, :] = True    # mask the first row
        >>> segm.remove_masked_labels(mask)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_masked_labels(mask, partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """

        if mask.shape != self.shape:
            raise ValueError('mask must have the same shape as the '
                             'segmentation image')
        remove_labels = self._labels(self.data[mask])
        if not partial_overlap:
            interior_labels = self._labels(self.data[~mask])
            remove_labels = list(set(remove_labels) - set(interior_labels))
        self.remove_labels(remove_labels, relabel=relabel)


class SourceProperties(object):
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
        from ``data``.  ``data`` should be background-subtracted.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    label : int
        The label number of the source whose properties to calculate.

    filtered_data : array-like or `~astropy.units.Quantity`, optional
        The filtered version of the background-subtracted ``data`` from
        which to calculate the source centroid and morphological
        properties.  The kernel used to perform the filtering should be
        the same one used in defining the source segments (e.g., see
        :func:`~photutils.detect_sources`).  If `None`, then the
        unfiltered ``data`` will be used instead.  Note that
        `SExtractor`_'s centroid and morphological parameters are
        calculated from the filtered "detection" image.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        If ``effective_gain`` is input, then ``error`` should include
        all sources of "background" error but *exclude* the Poisson
        error of the sources.  If ``effective_gain`` is `None`, then
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources.  ``error`` must have
        the same shape as ``data``.  See the Notes section below for
        details on the error propagation.

    effective_gain : float, array-like, or `~astropy.units.Quantity`, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data``.  This ratio is used to calculate the Poisson error of
        the sources when it is not included in ``error``.  If
        ``effective_gain`` is `None`, then ``error`` is assumed to
        include *all* sources of error.  See the Notes section below for
        details on the error propagation.

        If you are calculating the properties of many sources from the
        same data, it is highly recommended that you input a *total*
        error array instead of using ``effective_gain``.  Otherwise a
        total error array will need to be repeatedly recalculated.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.

    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use.  If `None`, then
        `~photutils.SourceProperties.icrs_centroid`,
        `~photutils.SourceProperties.ra_icrs_centroid`, and
        `~photutils.SourceProperties.dec_icrs_centroid` will be `None`.

    Notes
    -----
    `SExtractor`_'s centroid and morphological parameters are always
    calculated from the filtered "detection" image.  The usual downside
    of the filtering is the sources will be made more circular than they
    actually are.  If you wish to reproduce `SExtractor`_ results, then
    use the ``filtered_data`` input.  If ``filtered_data`` is `None`,
    then the unfiltered ``data`` will be used for the source centroid
    and morphological parameters.

    Negative (background-subtracted) data values within the source
    segment are set to zero when measuring morphological properties
    based on image moments.  This could occur, for example, if the
    segmentation image was defined from a different image (e.g.,
    different bandpass) or if the background was oversubtracted.  Note
    that `~photutils.SourceProperties.source_sum` includes the
    contribution of negative (background-subtracted) data values.
    `~photutils.SourceProperties.source_sum_err` will ignore such pixels
    when calculating the source Poission error (i.e. when if
    ``effective_gain`` is input; see below).

    If ``effective_gain`` is input, then ``error`` should include all
    sources of "background" error but *exclude* the Poisson error of the
    sources.  The total error image, :math:`\sigma_{\mathrm{tot}}` is
    then:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
              \\frac{(I - B)}{g}}

    where :math:`\sigma_b`, :math:`(I - B)`, and :math:`g` are the
    background ``error`` image, the background-subtracted ``data``
    image, and ``effective_gain``, respectively.

    Pixels where :math:`(I_i - B_i)` is negative do not contribute
    additional Poisson noise to the total error, i.e.
    :math:`\sigma_{\mathrm{tot}, i} = \sigma_{\mathrm{b}, i}`.  Note
    that this is different from `SExtractor`_, which sums the total
    variance in the segment, including pixels where :math:`(I_i - B_i)`
    is negative.  In such cases, `SExtractor`_ underestimates the total
    errors.

    If ``effective_gain`` is `None`, then ``error`` is assumed to
    include *all* sources of error, including the Poisson error of the
    sources, i.e. :math:`\sigma_{\mathrm{tot}} = \sigma_{\mathrm{b}} =
    \mathrm{error}`.

    For example, if your input ``data`` are in units of ADU, then
    ``effective_gain`` should represent electrons/ADU.  If your input
    ``data`` are in units of electrons/s then ``effective_gain`` should
    be the exposure time or an exposure time map (e.g., for mosaics with
    non-uniform exposure times).

    ``effective_gain`` can be a 2D gain image with the same shape as the
    ``data``.  This is useful with mosaic images that have variable
    depths (i.e., exposure times) across the field.  For example, one
    should use an exposure-time map as the ``effective_gain`` for a
    variable depth mosaic image in count-rate units.

    `~photutils.SourceProperties.source_sum_err` is simply the
    quadrature sum of the pixel-wise total errors over the non-masked
    pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\Delta F` is
    `~photutils.SourceProperties.source_sum_err` and :math:`S` are the
    non-masked pixels in the source segment.

    Custom errors for source segments can be calculated using the
    `~photutils.SourceProperties.error_cutout_ma` and
    `~photutils.SourceProperties.background_cutout_ma` properties, which
    are 2D `~numpy.ma.MaskedArray` cutout versions of the input
    ``error`` and ``background``.  The mask is `True` for both pixels
    outside of the source segment and masked pixels.

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    def __init__(self, data, segment_img, label, filtered_data=None,
                 error=None, effective_gain=None, mask=None, background=None,
                 wcs=None):

        if not isinstance(segment_img, SegmentationImage):
            segment_img = SegmentationImage(segment_img)

        if segment_img.shape != data.shape:
            raise ValueError('The data and segmentation image must have '
                             'the same shape')
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('The data and mask must have the same shape')

        segment_img.check_label(label)
        self.label = label
        self._slice = segment_img.slices[label - 1]
        self._segment_img = segment_img
        self._mask = mask
        self._wcs = wcs

        data, error, background = _prepare_data(
            data, error=error, effective_gain=effective_gain,
            background=background)
        # data and filtered_data should be background-subtracted
        self._data = data
        if filtered_data is None:
            self._filtered_data = data
        else:
            self._filtered_data = filtered_data
        self._error = error    # *total* error
        self._background = background    # 2D array

    def __getitem__(self, key):
        return getattr(self, key, None)

    def make_cutout(self, data, masked_array=False):
        """
        Create a (masked) cutout array from the input ``data`` using the
        minimal bounding box of the source segment.

        Parameters
        ----------
        data : array-like (2D)
            The data array from which to create the masked cutout array.
            ``data`` must have the same shape as the segmentation image
            input into `SourceProperties`.

        masked_array : bool, optional
            If `True` then a `~numpy.ma.MaskedArray` will be created
            where the mask is `True` for both pixels outside of the
            source segment and any masked pixels.  If `False`, then a
            `~numpy.ndarray` will be generated.

        Returns
        -------
        result : `~numpy.ndarray` or `~numpy.ma.MaskedArray` (2D)
            The 2D cutout array or masked array.
        """

        if data is None:
            return None

        data = np.asarray(data)
        if data.shape != self._data.shape:
            raise ValueError('data must have the same shape as the '
                             'segmentation image input to SourceProperties')
        if masked_array:
            return np.ma.masked_array(data[self._slice],
                                      mask=self._cutout_total_mask)
        else:
            return data[self._slice]

    def to_table(self, columns=None, exclude_columns=None):
        """
        Create a `~astropy.table.Table` of properties.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.Table` will include all scalar-valued
        properties.  Multi-dimensional properties, e.g.
        `~photutils.SourceProperties.data_cutout`, can be included in
        the ``columns`` input.

        Parameters
        ----------
        columns : str or list of str, optional
            Names of columns, in order, to include in the output
            `~astropy.table.Table`.  The allowed column names are any of
            the attributes of `SourceProperties`.

        exclude_columns : str or list of str, optional
            Names of columns to exclude from the default properties list
            in the output `~astropy.table.Table`.  The default
            properties are those with scalar values.

        Returns
        -------
        table : `~astropy.table.Table`
            A single-row table of properties of the source.
        """

        return properties_table(self, columns=columns,
                                exclude_columns=exclude_columns)

    @lazyproperty
    def _cutout_segment_bool(self):
        """
        _cutout_segment_bool is `True` only for pixels in the source
        segment of interest.  Pixels from other sources within the
        rectangular cutout are not included.
        """

        return self._segment_img.data[self._slice] == self.label

    @lazyproperty
    def _cutout_total_mask(self):
        """
        _cutout_total_mask is `True` for regions outside of the source
        segment or where the input mask is `True`.
        """

        mask = ~self._cutout_segment_bool
        if self._mask is not None:
            mask |= self._mask[self._slice]
        return mask

    @lazyproperty
    def data_cutout(self):
        """
        A 2D cutout from the (background-subtracted) data of the source
        segment.
        """

        return self.make_cutout(self._data, masked_array=False)

    @lazyproperty
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the
        (background-subtracted) data, where the mask is `True` for both
        pixels outside of the source segment and masked pixels.
        """

        return self.make_cutout(self._data, masked_array=True)

    @lazyproperty
    def _data_cutout_maskzeroed_double(self):
        """
        A 2D cutout from the (background-subtracted) (filtered) data,
        where pixels outside of the source segment and masked pixels are
        set to zero.  Negative data values are also set to zero because
        negative pixels (especially at large radii) can result in image
        moments that result in negative variances.  The cutout image is
        double precision, which is required for scikit-image's
        Cython-based moment functions.
        """

        cutout = self.make_cutout(self._filtered_data, masked_array=False)
        cutout = np.where(cutout > 0, cutout, 0.)    # negative pixels -> 0
        return (cutout * ~self._cutout_total_mask).astype(np.float64)

    @lazyproperty
    def error_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input ``error``
        image, where the mask is `True` for both pixels outside of the
        source segment and masked pixels.  If ``error`` is `None`, then
        ``error_cutout_ma`` is also `None`.
        """

        return self.make_cutout(self._error, masked_array=True)

    @lazyproperty
    def background_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input
        ``background``, where the mask is `True` for both pixels outside
        of the source segment and masked pixels.  If ``background`` is
        `None`, then ``background_cutout_ma`` is also `None`.
        """

        return self.make_cutout(self._background, masked_array=True)

    @lazyproperty
    def coords(self):
        """
        A tuple of `~numpy.ndarray`\s containing the ``y`` and ``x``
        pixel coordinates of the source segment.  Masked pixels are not
        included.
        """

        yy, xx = np.nonzero(self.data_cutout_ma)
        coords = (yy + self._slice[0].start, xx + self._slice[1].start)
        return coords

    @lazyproperty
    def values(self):
        """
        A `~numpy.ndarray` of the (background-subtracted) pixel values
        within the source segment.  Masked pixels are not included.
        """

        return self.data_cutout[~self._cutout_total_mask]

    @lazyproperty
    def moments(self):
        """Spatial moments up to 3rd order of the source."""

        from skimage.measure import moments
        return moments(self._data_cutout_maskzeroed_double, 3)

    @lazyproperty
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """

        from skimage.measure import moments_central
        ycentroid, xcentroid = self.cutout_centroid.value
        return moments_central(self._data_cutout_maskzeroed_double,
                               ycentroid, xcentroid, 3)

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

        m = self.moments
        if m[0, 0] != 0:
            ycentroid = m[0, 1] / m[0, 0]
            xcentroid = m[1, 0] / m[0, 0]
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
        return (ycen + self._slice[0].start,
                xcen + self._slice[1].start) * u.pix

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
    def icrs_centroid(self):
        """
        The International Celestial Reference System (ICRS) coordinates
        of the centroid within the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xcentroid.value,
                                     self.ycentroid.value,
                                     self._wcs, origin=1).icrs
        else:
            return None

    @lazyproperty
    def ra_icrs_centroid(self):
        """
        The ICRS Right Ascension coordinate (in degrees) of the centroid
        within the source segment.
        """

        if self._wcs is not None:
            return self.icrs_centroid.ra.degree * u.deg
        else:
            return None

    @lazyproperty
    def dec_icrs_centroid(self):
        """
        The ICRS Declination coordinate (in degrees) of the centroid
        within the source segment.
        """

        if self._wcs is not None:
            return self.icrs_centroid.dec.degree * u.deg
        else:
            return None

    @lazyproperty
    def bbox(self):
        """
        The bounding box ``(ymin, xmin, ymax, xmax)`` of the minimal
        rectangular region containing the source segment.
        """

        # (stop - 1) to return the max pixel location, not the slice index
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop - 1, self._slice[1].stop - 1) * u.pix

    @lazyproperty
    def xmin(self):
        """
        The minimum ``x`` pixel location of the minimal bounding box
        (`~photutils.SourceProperties.bbox`) of the source segment.
        """

        return self.bbox[1]

    @lazyproperty
    def xmax(self):
        """
        The maximum ``x`` pixel location of the minimal bounding box
        (`~photutils.SourceProperties.bbox`) of the source segment.
        """

        return self.bbox[3]

    @lazyproperty
    def ymin(self):
        """
        The minimum ``y`` pixel location of the minimal bounding box
        (`~photutils.SourceProperties.bbox`) of the source segment.
        """

        return self.bbox[0]

    @lazyproperty
    def ymax(self):
        """
        The maximum ``y`` pixel location of the minimal bounding box
        (`~photutils.SourceProperties.bbox`) of the source segment.
        """

        return self.bbox[2]

    @lazyproperty
    def min_value(self):
        """
        The minimum pixel value of the (background-subtracted) data
        within the source segment.
        """

        return np.min(self.values)

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value of the (background-subtracted) data
        within the source segment.
        """

        return np.max(self.values)

    @lazyproperty
    def minval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the (background-subtracted) data.
        """

        return np.argwhere(self.data_cutout_ma == self.min_value)[0] * u.pix

    @lazyproperty
    def maxval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the (background-subtracted) data.
        """

        return np.argwhere(self.data_cutout_ma == self.max_value)[0] * u.pix

    @lazyproperty
    def minval_pos(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        (background-subtracted) data.
        """

        yp, xp = np.array(self.minval_cutout_pos)
        return (yp + self._slice[0].start, xp + self._slice[1].start) * u.pix

    @lazyproperty
    def maxval_pos(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        (background-subtracted) data.
        """

        yp, xp = np.array(self.maxval_cutout_pos)
        return (yp + self._slice[0].start, xp + self._slice[1].start) * u.pix

    @lazyproperty
    def minval_xpos(self):
        """
        The ``x`` coordinate of the minimum pixel value of the
        (background-subtracted) data.
        """

        return self.minval_pos[1]

    @lazyproperty
    def minval_ypos(self):
        """
        The ``y`` coordinate of the minimum pixel value of the
        (background-subtracted) data.
        """

        return self.minval_pos[0]

    @lazyproperty
    def maxval_xpos(self):
        """
        The ``x`` coordinate of the maximum pixel value of the
        (background-subtracted) data.
        """

        return self.maxval_pos[1]

    @lazyproperty
    def maxval_ypos(self):
        """
        The ``y`` coordinate of the maximum pixel value of the
        (background-subtracted) data.
        """

        return self.maxval_pos[0]

    @lazyproperty
    def area(self):
        """The area of the source segment in units of pixels**2."""

        return len(self.values) * u.pix**2

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
        The perimeter of the source segment, approximated lines through
        the centers of the border pixels using a 4-connectivity.
        """

        from skimage.measure import perimeter
        return perimeter(self._cutout_segment_bool, 4) * u.pix

    @lazyproperty
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """

        mu = self.moments_central
        a = mu[2, 0]
        b = -mu[1, 1]
        c = mu[0, 2]
        return np.array([[a, b], [b, c]]) * u.pix**2

    @lazyproperty
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """

        mu = self.moments_central
        if mu[0, 0] != 0:
            m = mu / mu[0, 0]
            covariance = self._check_covariance(
                np.array([[m[2, 0], m[1, 1]], [m[1, 1], m[0, 2]]]))
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

        p = 1. / 12     # arbitrary SExtractor value
        val = (covariance[0, 0] * covariance[1, 1]) - covariance[0, 1]**2
        if val >= p**2:
            return covariance
        else:
            covar = np.copy(covariance)
            while val < p**2:
                covar[0, 0] += p
                covar[1, 1] += p
                val = (covar[0, 0] * covar[1, 1]) - covar[0, 1]**2
            return covar

    @lazyproperty
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """

        if not np.isnan(np.sum(self.covariance)):
            eigvals = np.linalg.eigvals(self.covariance)
            if np.any(eigvals < 0):    # negative variance
                return (np.nan, np.nan) * u.pix**2
            return (np.max(eigvals), np.min(eigvals)) * u.pix**2
        else:
            return (np.nan, np.nan) * u.pix**2

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

        l1, l2 = self.covariance_eigvals
        if l1 == 0:
            return 0.
        return np.sqrt(1. - (l2 / l1))

    @lazyproperty
    def orientation(self):
        """
        The angle in radians between the ``x`` axis and the major axis
        of the 2D Gaussian function that has the same second-order
        moments as the source.  The angle increases in the
        counter-clockwise direction.
        """

        a, b, b, c = self.covariance.flat
        if a < 0 or c < 0:    # negative variance
            return np.nan * u.rad
        return 0.5 * np.arctan2(2. * b, (a - c))

    @lazyproperty
    def elongation(self):
        """
        The ratio of the lengths of the semimajor and semiminor axes:

        .. math:: \mathrm{elongation} = \\frac{a}{b}

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

        .. math:: \mathrm{ellipticity} = 1 - \\frac{b}{a}

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
        :math:`\sigma_x^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s X2 parameter.
        """

        return self.covariance[0, 0]

    @lazyproperty
    def covar_sigy2(self):
        """
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s Y2 parameter.
        """

        return self.covariance[1, 1]

    @lazyproperty
    def covar_sigxy(self):
        """
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
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
    def source_sum(self):
        """
        The sum of the non-masked (background-subtracted) data values
        within the source segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``source_sum``, :math:`(I_i - B_i)` is the
        background-subtracted input ``data``, and :math:`S` are the
        non-masked pixels in the source segment.
        """

        return np.sum(np.ma.masked_array(self._data[self._slice],
                                         mask=self._cutout_total_mask))

    @lazyproperty
    def source_sum_err(self):
        """
        The uncertainty of `~photutils.SourceProperties.source_sum`,
        propagated from the input ``error`` array.

        ``source_sum_err`` is the quadrature sum of the total errors
        over the non-masked pixels within the source segment:

        .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
                  \\sigma_{\\mathrm{tot}, i}^2}

        where :math:`\Delta F` is ``source_sum_err``,
        :math:`\sigma_{\mathrm{tot, i}}` are the pixel-wise total
        errors, and :math:`S` are the non-masked pixels in the source
        segment.
        """

        if self._error is not None:
            # power doesn't work here, see astropy #2968
            # return np.sqrt(np.sum(self.error_cutout_ma**2))
            return np.sqrt(np.sum(
                np.ma.masked_array(self.error_cutout_ma.data**2,
                                   mask=self.error_cutout_ma.mask)))
        else:
            return None

    @lazyproperty
    def background_sum(self):
        """The sum of ``background`` values within the source segment."""

        if self._background is not None:
            return np.sum(self.background_cutout_ma)
        else:
            return None

    @lazyproperty
    def background_mean(self):
        """The mean of ``background`` values within the source segment."""

        if self._background is not None:
            return np.mean(self.background_cutout_ma)
        else:
            return None

    @lazyproperty
    def background_at_centroid(self):
        """
        The value of the ``background`` at the position of the source
        centroid.  Fractional position values are determined using
        bilinear interpolation.
        """

        from scipy.ndimage import map_coordinates

        if self._background is None:
            return None
        else:
            return map_coordinates(
                self._background, [[self.ycentroid.value],
                                   [self.xcentroid.value]])[0]


def source_properties(data, segment_img, error=None, effective_gain=None,
                      mask=None, background=None, filter_kernel=None,
                      wcs=None, labels=None):
    """
    Calculate photometry and morphological properties of sources defined
    by a labeled segmentation image.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.  ``data`` should be background-subtracted.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        If ``effective_gain`` is input, then ``error`` should include
        all sources of "background" error but *exclude* the Poisson
        error of the sources.  If ``effective_gain`` is `None`, then
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources.  ``error`` must have
        the same shape as ``data``.  See the Notes section below for
        details on the error propagation.

    effective_gain : float, array-like, or `~astropy.units.Quantity`, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data``.  This ratio is used to calculate the Poisson error of
        the sources when it is not included in ``error``.  If
        ``effective_gain`` is `None`, then ``error`` is assumed to
        include *all* sources of error.  See the Notes section below for
        details on the error propagation.

        If you are calculating the properties of many sources from the
        same data, it is highly recommended that you input a *total*
        error array instead of using ``effective_gain``.  Otherwise a
        total error array will need to be repeatedly recalculated.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the data prior to
        calculating the source centroid and morphological parameters.
        The kernel should be the same one used in defining the source
        segments (e.g., see :func:`~photutils.detect_sources`).  If
        `None`, then the unfiltered ``data`` will be used instead.  Note
        that `SExtractor`_'s centroid and morphological parameters are
        calculated from the filtered "detection" image.

    wcs : `~astropy.wcs.WCS`
        The WCS transformation to use.  If `None`, then
        `~photutils.SourceProperties.icrs_centroid`,
        `~photutils.SourceProperties.ra_icrs_centroid`, and
        `~photutils.SourceProperties.dec_icrs_centroid` will be `None`.

    labels : int or list of ints
        Subset of segmentation labels for which to calculate the
        properties.  If `None`, then the properties will be calculated
        for all labeled sources (the default).

    Returns
    -------
    output : list of `SourceProperties` objects
        A list of `SourceProperties` objects, one for each source.  The
        properties can be accessed as attributes or keys.

    Notes
    -----
    `SExtractor`_'s centroid and morphological parameters are always
    calculated from the filtered "detection" image.  The usual downside
    of the filtering is the sources will be made more circular than they
    actually are.  If you wish to reproduce `SExtractor`_ results, then
    use the ``filtered_data`` input.  If ``filtered_data`` is `None`,
    then the unfiltered ``data`` will be used for the source centroid
    and morphological parameters.

    Negative (background-subtracted) data values within the source
    segment are set to zero when measuring morphological properties
    based on image moments.  This could occur, for example, if the
    segmentation image was defined from a different image (e.g.,
    different bandpass) or if the background was oversubtracted.  Note
    that `~photutils.SourceProperties.source_sum` includes the
    contribution of negative (background-subtracted) data values.
    `~photutils.SourceProperties.source_sum_err` will ignore such pixels
    when calculating the source Poission error (i.e. when if
    ``effective_gain`` is input; see below).

    If ``effective_gain`` is input, then ``error`` should include all
    sources of "background" error but *exclude* the Poisson error of the
    sources.  The total error image, :math:`\sigma_{\mathrm{tot}}` is
    then:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
              \\frac{(I - B)}{g}}

    where :math:`\sigma_b`, :math:`(I - B)`, and :math:`g` are the
    background ``error`` image, the background-subtracted ``data``
    image, and ``effective_gain``, respectively.

    Pixels where :math:`(I_i - B_i)` is negative do not contribute
    additional Poisson noise to the total error, i.e.
    :math:`\sigma_{\mathrm{tot}, i} = \sigma_{\mathrm{b}, i}`.  Note
    that this is different from `SExtractor`_, which sums the total
    variance in the segment, including pixels where :math:`(I_i - B_i)`
    is negative.  In such cases, `SExtractor`_ underestimates the total
    errors.

    If ``effective_gain`` is `None`, then ``error`` is assumed to
    include *all* sources of error, including the Poisson error of the
    sources, i.e. :math:`\sigma_{\mathrm{tot}} = \sigma_{\mathrm{b}} =
    \mathrm{error}`.

    For example, if your input ``data`` are in units of ADU, then
    ``effective_gain`` should represent electrons/ADU.  If your input
    ``data`` are in units of electrons/s then ``effective_gain`` should
    be the exposure time or an exposure time map (e.g., for mosaics with
    non-uniform exposure times).

    ``effective_gain`` can be a 2D gain image with the same shape as the
    ``data``.  This is useful with mosaic images that have variable
    depths (i.e., exposure times) across the field.  For example, one
    should use an exposure-time map as the ``effective_gain`` for a
    variable depth mosaic image in count-rate units.

    `~photutils.SourceProperties.source_sum_err` is simply the
    quadrature sum of the pixel-wise total errors over the non-masked
    pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\Delta F` is
    `~photutils.SourceProperties.source_sum_err` and :math:`S` are the
    non-masked pixels in the source segment.

    .. _SExtractor: http://www.astromatic.net/software/sextractor

    See Also
    --------
    SegmentationImage, SourceProperties, properties_table,
    :func:`photutils.detection.detect_sources`

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import SegmentationImage, source_properties
    >>> image = np.arange(16.).reshape(4, 4)
    >>> print(image)
    [[  0.   1.   2.   3.]
     [  4.   5.   6.   7.]
     [  8.   9.  10.  11.]
     [ 12.  13.  14.  15.]]
    >>> segm = SegmentationImage([[1, 1, 0, 0],
    ...                           [1, 0, 0, 2],
    ...                           [0, 0, 2, 2],
    ...                           [0, 2, 2, 0]])
    >>> props = source_properties(image, segm)

    Print some properties of the first object (labeled with ``1`` in the
    segmentation image):

    >>> props[0].id    # id corresponds to segment label number
    1
    >>> props[0].centroid    # doctest: +FLOAT_CMP
    <Quantity [ 0.8, 0.2] pix>
    >>> props[0].source_sum    # doctest: +FLOAT_CMP
    5.0
    >>> props[0].area    # doctest: +FLOAT_CMP
    <Quantity 3.0 pix2>
    >>> props[0].max_value    # doctest: +FLOAT_CMP
    4.0

    Print some properties of the second object (labeled with ``2`` in
    the segmentation image):

    >>> props[1].id    # id corresponds to segment label number
    2
    >>> props[1].centroid    # doctest: +FLOAT_CMP
    <Quantity [ 2.36363636, 2.09090909] pix>
    >>> props[1].perimeter    # doctest: +FLOAT_CMP
    <Quantity 5.414213562373095 pix>
    >>> props[1].orientation    # doctest: +FLOAT_CMP
    <Quantity -0.7417593069227176 rad>
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)

    # prepare the input data once, instead of repeating for each source
    data, error_total, background = _prepare_data(
        data, error=error, effective_gain=effective_gain,
        background=background)

    # filter the data once, instead of repeating for each source
    if filter_kernel is not None:
        filtered_data = _convolve_data(data, filter_kernel, mode='constant',
                                       fill_value=0.0,
                                       check_normalization=True)
    else:
        filtered_data = None

    sources_props = []
    for label in labels:
        if label not in segment_img.labels:
            continue      # skip invalid labels (without warnings)
        sources_props.append(SourceProperties(
            data, segment_img, label, filtered_data=filtered_data,
            error=error_total, effective_gain=None, mask=mask,
            background=background, wcs=wcs))

    return sources_props


def properties_table(source_props, columns=None, exclude_columns=None):
    """
    Construct a `~astropy.table.Table` of properties from a list of
    `SourceProperties` objects.

    If ``columns`` or ``exclude_columns`` are not input, then the
    `~astropy.table.Table` will include all scalar-valued properties.
    Multi-dimensional properties, e.g.
    `~photutils.SourceProperties.data_cutout`, can be included in the
    ``columns`` input.

    Parameters
    ----------
    source_props : `SourceProperties` or list of `SourceProperties`
        A `SourceProperties` object or list of `SourceProperties`
        objects, one for each source.

    columns : str or list of str, optional
        Names of columns, in order, to include in the output
        `~astropy.table.Table`.  The allowed column names are any of the
        attributes of `SourceProperties`.

    exclude_columns : str or list of str, optional
        Names of columns to exclude from the default properties list in
        the output `~astropy.table.Table`.  The default properties are
        those with scalar values.

    Returns
    -------
    table : `~astropy.table.Table`
        A table of properties of the segmented sources, one row per
        source.

    See Also
    --------
    SegmentationImage, SourceProperties, source_properties,
    :func:`photutils.detection.detect_sources`

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import source_properties, properties_table
    >>> image = np.arange(16.).reshape(4, 4)
    >>> print(image)
    [[  0.   1.   2.   3.]
     [  4.   5.   6.   7.]
     [  8.   9.  10.  11.]
     [ 12.  13.  14.  15.]]
    >>> segm = SegmentationImage([[1, 1, 0, 0],
    ...                           [1, 0, 0, 2],
    ...                           [0, 0, 2, 2],
    ...                           [0, 2, 2, 0]])
    >>> props = source_properties(image, segm)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'source_sum']
    >>> tbl = properties_table(props, columns=columns)
    >>> print(tbl)
     id   xcentroid     ycentroid   source_sum
             pix           pix
    --- ------------- ------------- ----------
      1           0.2           0.8        5.0
      2 2.09090909091 2.36363636364       55.0
    """

    if isinstance(source_props, list) and len(source_props) == 0:
        raise ValueError('source_props is an empty list')
    source_props = np.atleast_1d(source_props)

    # all scalar-valued properties
    columns_all = ['id', 'xcentroid', 'ycentroid', 'ra_icrs_centroid',
                   'dec_icrs_centroid', 'source_sum',
                   'source_sum_err', 'background_sum', 'background_mean',
                   'background_at_centroid', 'xmin', 'xmax', 'ymin', 'ymax',
                   'min_value', 'max_value', 'minval_xpos', 'minval_ypos',
                   'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius',
                   'perimeter', 'semimajor_axis_sigma',
                   'semiminor_axis_sigma', 'eccentricity', 'orientation',
                   'ellipticity', 'elongation', 'covar_sigx2',
                   'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy', 'cyy']

    table_columns = None
    if exclude_columns is not None:
        table_columns = [s for s in columns_all if s not in exclude_columns]
    if columns is not None:
        table_columns = np.atleast_1d(columns)
    if table_columns is None:
        table_columns = columns_all

    # it's *much* faster to calculate world coordinates using the
    # complete list of (x, y) instead of from the individual (x, y).
    # The assumption here is that the wcs is the same for each
    # element of source_props.
    if ('ra_icrs_centroid' in table_columns or
            'dec_icrs_centroid' in table_columns or
            'icrs_centroid' in table_columns):
        xcentroid = [props.xcentroid.value for props in source_props]
        ycentroid = [props.ycentroid.value for props in source_props]
        if source_props[0]._wcs is not None:
            icrs_centroid = pixel_to_skycoord(
                xcentroid, ycentroid, source_props[0]._wcs, origin=1).icrs
            icrs_ra = icrs_centroid.ra.degree * u.deg
            icrs_dec = icrs_centroid.dec.degree * u.deg
        else:
            nprops = len(source_props)
            icrs_ra = [None] * nprops
            icrs_dec = [None] * nprops
            icrs_centroid = [None] * nprops

    props_table = Table()
    for column in table_columns:
        if column == 'ra_icrs_centroid':
            props_table[column] = icrs_ra
        elif column == 'dec_icrs_centroid':
            props_table[column] = icrs_dec
        elif column == 'icrs_centroid':
            props_table[column] = icrs_centroid
        else:
            values = [getattr(props, column) for props in source_props]
            if isinstance(values[0], u.Quantity):
                # turn list of Quantities into a Quantity array
                values = u.Quantity(values)
            props_table[column] = values

    return props_table
