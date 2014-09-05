# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table, Column
import skimage
from skimage.measure._regionprops import _cached_property


__all__ = ['SegmentProperties', 'segment_properties', 'segment_photometry']


class SegmentProperties(object):
    """Class to calculate morphological properties of source segments."""

    def __init__(self, data, segment_image, label, labelslice=None,
                 mask=None, mask_method='exclude', background=None):
        """
        Construct a `SegmentProperties` object.

        Parameters
        ----------
        data : array_like
            The 2D array from which to calculate the source properties.

        segment_image : array_like
            A 2D segmentation image, with the same shape as ``data``, where
            sources are marked by different positive integer values.  A
            value of zero is reserved for the background.

        label : int
            The label number of a source segment in ``segment_image``
            for which to calculate morphological properties.

        labelslice : 2-tuple of slice objects, optional
            A ``(y_slice, x_slice)`` tuple of slice objects defining the
            minimal box enclosing the source segment.  If `None` (the
            default), then ``labelslice`` will be calculated.

        mask : array_like, bool, optional
            A boolean mask, with the same shape as ``data``, where a
            `True` value indicates the corresponding element of
            ``image`` is masked.  Use the ``mask_method`` keyword to
            select the method used to treat masked pixels.

        mask_method : {'exclude', 'interpolate'}, optional
            Method used to treat masked pixels.  The currently supported
            methods are:

            'exclude':
                Exclude masked pixels from all calculations.  This is
                the default.

            'interpolate':
                The value of masked pixels are replaced by the mean
                value of the 8-connected neighboring non-masked pixels.

        background : float or array_like, optional
            The background level of the input ``data``.  ``background``
            may either be a scalar value or a 2D image with the same
            shape as the input ``data``.  If the input ``data`` has been
            background-subtracted, then set ``background`` to `None`
            (the default).
        """

        from scipy import ndimage

        if segment_image.shape != data.shape:
            raise ValueError('segment_image and data must have the same shape')

        if label == 0:
            raise ValueError('label "0" is reserved for the background')
        elif label < 0:
            raise ValueError('label must be a positive integer')

        data, variance, background = _condition_data(
            data, error=None, gain=None, mask=mask, mask_method=mask_method,
            background=background)

        self._image = data
        self._segment_image = segment_image
        self.label = label
        if labelslice is not None:
            self._slice = labelslice
        else:
            labelslices = ndimage.find_objects(segment_image)
            self._slice = labelslices[label - 1]
            if self._slice is None:
                raise ValueError('label "{0}" is not in the input '
                                 'segment_image'.format(label))
        if mask_method == 'interpolate':
            # interpolated masked pixels are used like unmasked pixels,
            # so no further masking is needed
            self._mask = np.zeros_like(data, dtype=np.bool)
        else:
            # excluded masked pixels still need the mask
            self._mask = mask
        self._background = background
        self._cache_active = True

    def __getitem__(self, key):
        return getattr(self, key, None)

    @_cached_property
    def _in_segment(self):
        """
        _in_segment is `True` for pixels in the labeled source segment.
        """
        return self._segment_image[self._slice] == self.label

    @_cached_property
    def _local_mask(self):
        """
        _local_mask is `True` for regions outside of the labeled source
        segment or where the input mask ("excluded" mask) is `True`.
        """
        if self._mask is None:
            return ~self._in_segment
        else:
            return np.logical_or(~self._in_segment, self._mask[self._slice])

    @_cached_property
    def cutout_image(self):
        """
        A 2D cutout image of the source segment.
        """
        return self._image[self._slice]

    @_cached_property
    def isolated_cutout_image(self):
        """
        A 2D cutout image of the source segment where pixels outside of
        the source segment have a value of ``numpy.nan``.
        """
        return np.where(self._in_segment, self.cutout_image, np.nan)

    @_cached_property
    def cutout_image_maskedarray(self):
        """
        A 2D cutout image as a masked array, where the mask is `True`
        for pixels outside of the source segment and "excluded" masked
        pixels.
        """
        return np.ma.masked_array(self.cutout_image, mask=self._local_mask)

    @_cached_property
    def _cutout_image_maskzeroed(self):
        """
        A 2D cutout image where pixels outside of the source segment and
        "excluded" masked pixels are set to zero.
        """
        return self.cutout_image * ~self._local_mask

    @_cached_property
    def _cutout_image_maskzeroed_double(self):
        """
        Double-precision version of ``_cutout_image_maskedzeroed``.
        Required for scikit-image's Cython moment functions.
        """
        return self._cutout_image_maskzeroed.astype(np.double)

    @_cached_property
    def coords(self):
        """
        A list of the ``(y, x)`` pixel coordinates of the source
        segment.

        "Excluded" masked pixels are not included, but interpolated
        masked pixels are included.
        """
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO:  use masked array in case good pixel is zero
        yy, xx = np.nonzero(self._cutout_image_maskzeroed)
        return (yy + self._slice[0].start, xx + self._slice[1].start)

    @_cached_property
    def values(self):
        """
        A list of the pixel values within the source segment.

        Values of "excluded" masked pixels are not included, but
        interpolated masked pixels are included.
        """
        return self.cutout_image[~self._local_mask]

    @_cached_property
    def moments(self):
        """Spatial moments up to 3rd order of the source segment."""
        return skimage.measure.moments(
            self._cutout_image_maskzeroed_double, 3)

    @_cached_property
    def moments_central(self):
        """
        Central moments (translation invariant) of the source segment up
        to 3rd order.
        """
        ycentroid, xcentroid = self.local_centroid
        return skimage.measure.moments_central(
            self._cutout_image_maskzeroed_double, ycentroid, xcentroid, 3)

    @_cached_property
    def id(self):
        """
        The source identification number corresponding to the object
        label in the ``segment_image``.
        """
        return self.label

    @_cached_property
    def local_centroid(self):
        """
        The ``(y, x)`` coordinate, relative to the `cutout_image`, of
        the centroid within the source segment.
        """
        # TODO: allow alternative centroid methods?
        m = self.moments
        ycentroid = m[0, 1] / m[0, 0]
        xcentroid = m[1, 0] / m[0, 0]
        return ycentroid, xcentroid

    @_cached_property
    def centroid(self):
        """
        The ``(y, x)`` coordinate of the centroid within the source
        segment.
        """
        ycen, xcen = self.local_centroid
        return ycen + self._slice[0].start, xcen + self._slice[1].start

    @_cached_property
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """
        return self.centroid[1]

    @_cached_property
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """
        return self.centroid[0]

    @_cached_property
    def bbox(self):
        """
        The bounding box ``(ymin, xmin, ymax, xmax)`` of the region
        containing the source segment.
        """
        # (stop - 1) to return the max pixel location, not the slice index
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop - 1, self._slice[1].stop - 1)

    @_cached_property
    def xmin(self):
        """
        The left ``x`` pixel location of the bounding box of the source
        segment.
        """
        return self.bbox[1]

    @_cached_property
    def xmax(self):
        """
        The right ``x`` pixel location of the bounding box of the source
        segment.
        """
        return self.bbox[3]

    @_cached_property
    def ymin(self):
        """
        The bottom ``y`` pixel location of the bounding box of the
        source segment.
        """
        return self.bbox[0]

    @_cached_property
    def ymax(self):
        """
        The top ``y`` pixel location of the bounding box of the
        source segment.
        """
        return self.bbox[2]

    @_cached_property
    def min_value(self):
        """The minimum pixel value within the source segment."""
        return np.min(self.values)

    @_cached_property
    def max_value(self):
        """The maximum pixel value within the source segment."""
        return np.max(self.values)

    @_cached_property
    def minval_local_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `cutout_image`, of
        the minimum pixel value.
        """
        return np.argwhere(self.cutout_image_maskedarray == self.min_value)[0]

    @_cached_property
    def maxval_local_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `cutout_image`, of
        the maximum pixel value.
        """
        return np.argwhere(self.cutout_image_maskedarray == self.max_value)[0]

    @_cached_property
    def minval_pos(self):
        """The ``(y, x)`` coordinate of the minimum pixel value."""
        yp, xp = self.minval_local_pos
        return yp + self._slice[0].start, xp + self._slice[1].start

    @_cached_property
    def maxval_pos(self):
        """The ``(y, x)`` coordinate of the maximum pixel value."""
        yp, xp = self.maxval_local_pos
        return yp + self._slice[0].start, xp + self._slice[1].start

    @_cached_property
    def minval_xpos(self):
        """The ``x`` coordinate of the minimum pixel value."""
        return self.minval_pos[1]

    @_cached_property
    def minval_ypos(self):
        """The ``y`` coordinate of the minimum pixel value."""
        return self.minval_pos[0]

    @_cached_property
    def maxval_xpos(self):
        """The ``x`` coordinate of the maximum pixel value."""
        return self.maxval_pos[1]

    @_cached_property
    def maxval_ypos(self):
        """The ``y`` coordinate of the maximum pixel value."""
        return self.maxval_pos[0]

    @_cached_property
    def area(self):
        """The area of the source segment in units of pixels**2."""
        return len(self.values)

    @_cached_property
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """
        return np.sqrt(self.area / np.pi)

    @_cached_property
    def perimeter(self):
        """
        The perimeter of the source segment, approximated using a line
        through the centers of the border pixels using a 4-connectivity.
        """
        return skimage.measure.perimeter(self._in_segment, 4)

    @_cached_property
    def inertia_tensor(self):
        """
        Inertia tensor of the source segment for the rotation around its
        center of mass.
        """
        mu = self.moments_central
        a = mu[2, 0]
        b = -mu[1, 1]
        c = mu[0, 2]
        return np.array([[a, b], [b, c]])

    @_cached_property
    def covariance(self):
        """
        The covariance matrix of the ellipse that has the same
        second-order moments as the source segment.
        """
        mu = self.moments_central
        m = mu / mu[0, 0]
        return np.array([[m[2, 0], m[1, 1]], [m[1, 1], m[0, 2]]])

    @_cached_property
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.linalg.eigvals(self.covariance)
        return np.max(eigvals), np.min(eigvals)

    @_cached_property
    def semimajor_axis_length(self):
        """
        The length of the semimajor axis of the ellipse that has the
        same second-order central moments as the region.
        """
        # this matches SExtractor's A parameter
        return np.sqrt(self.covariance_eigvals[0])

    @_cached_property
    def semiminor_axis_length(self):
        """
        The length of the semiminor axis of the ellipse that has the
        same second-order central moments as the region.
        """
        # this matches SExtractor's B parameter
        return np.sqrt(self.covariance_eigvals[1])

    @_cached_property
    def eccentricity(self):
        """
        The eccentricity of the ellipse that has the same second-order
        moments as the source segment.

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

    @_cached_property
    def orientation(self):
        """
        The angle in radians between the ``x`` axis and the major axis
        of the ellipse that has the same second-order moments as the
        source segment.  The angle increases in the counter-clockwise
        direction.
        """
        a, b, b, c = self.covariance.flat
        return 0.5 * np.arctan2(2. * b, (a - c))

    @_cached_property
    def background_centroid(self):
        """
        The value of the background at the position of the source
        centroid.
        """
        if self._background is None:
            return None
        else:
            return self._background[self.ycentroid, self.xcentroid]

    @_cached_property
    def se_elongation(self):
        """
        SExtractor's elongation parameter.

        .. math:: \mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_axis_length / self.semiminor_axis_length

    @_cached_property
    def se_ellipticity(self):
        """
        SExtractor's ellipticity parameter.

        .. math:: \mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_axis_length / self.semimajor_axis_length)

    @_cached_property
    def se_x2(self):
        """
        SExtractor's X2 parameter, in units of pixel**2, which
        corresponds to the ``(0, 0)`` element of the `covariance`
        matrix.
        """
        return self.covariance[0, 0]

    @_cached_property
    def se_y2(self):
        """
        SExtractor's Y2 parameter, in units of pixel**2, which
        corresponds to the ``(1, 1)`` element of the `covariance`
        matrix.
        """
        return self.covariance[1, 1]

    @_cached_property
    def se_xy(self):
        """
        SExtractor's XY parameter, in units of pixel**2, which
        corresponds to the ``(0, 1)`` element of the `covariance`
        matrix.
        """
        return self.covariance[0, 1]

    @_cached_property
    def se_cxx(self):
        """
        SExtractor's CXX ellipse parameter in units of pixel**(-2).
        """
        return ((np.cos(self.orientation) / self.semimajor_axis_length)**2 +
                (np.sin(self.orientation) / self.semiminor_axis_length)**2)

    @_cached_property
    def se_cyy(self):
        """
        SExtractor's CYY ellipse parameter in units of pixel**(-2).
        """
        return ((np.sin(self.orientation) / self.semimajor_axis_length)**2 +
                (np.cos(self.orientation) / self.semiminor_axis_length)**2)

    @_cached_property
    def se_cxy(self):
        """
        SExtractor's CXY ellipse parameter in units of pixel**(-2).
        """
        return (2. * np.cos(self.orientation) * np.sin(self.orientation) *
                ((1./self.semimajor_axis_length**2) -
                 (1./self.semiminor_axis_length**2)))


def segment_properties(data, segment_image, mask=None, mask_method='exclude',
                       background=None, labels=None, return_table=False):
    """
    Calculate morphological properties of sources defined by a labeled
    segmentation image.

    Parameters
    ----------
    data : array_like
        The 2D array from which to calculate the source properties.

    segment_image : array_like
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    mask : array_like, bool, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is
        masked.  Use the ``mask_method`` keyword to select the method
        used to treat masked pixels.

    mask_method : {'exclude', 'interpolate'}, optional
        Method used to treat masked pixels.  The currently supported
        methods are:

        'exclude':
            Exclude masked pixels from all calculations.  This is the
            default.

        'interpolate':
            The value of masked pixels are replaced by the mean value of
            the 8-connected neighboring non-masked pixels.

    background : float or array_like, optional
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None` (the
        default).

    labels : int or list of ints
        Subset of ``segment_image`` labels for which to calculate
        morphological properties.  If `None`, then morphological
        properties will be calculated for all source segments (the
        default).

    return_table : bool, optional
        If `True` then return an `astropy.table.Table`, otherwise return
        a list of `SegmentProperties` objects.

    Returns
    -------
    output : `astropy.table.Table` or list of `SegmentProperties` objects

        * If ``return_table = True``: `astropy.table.Table`
              A table of the properties of the segmented sources
              containing the columns listed below.

        * If ``return_table = False``: list
              A list of `SegmentProperties` objects, one for each source
              segment.  The properties can be accessed using the
              attributes listed below.

    See Also
    --------
    detect_sources, segment_photometry

    Notes
    -----
    The following properties can be accessed either as columns in an
    `astropy.table.Table` or as attributes or keys of property objects:

    **id** : int
        The source identification number corresponding to the object
        label in the ``segment_image``.

    **xcentroid**, **ycentroid** : float
        The ``x`` and ``y`` coordinates of the centroid within the
        source segment.

    **xmin**, **xmax**, **ymin**, **ymax** : float
        The pixel locations defining the bounding box of the source
        segment.

    **min_value**, **max_value** : float
        The minimum and maximum pixel values within the source segment.

    **minval_xpos**, **minval_ypos** : float
        The ``x`` and ``y`` coordinates of the minimum pixel value.

    **maxval_xpos**, **maxval_ypos** : float
        The ``x`` and ``y`` coordinates of the maximum pixel value.

    **area** : float
        The area of the source segment in units of pixels**2.

    **equivalent_radius** : float
        The radius of a circle with the same ``area`` as the source
        segment.

    **perimeter** : float
        The perimeter of the source segment, approximated using a line
        through the centers of the border pixels using a 4-connectivity.

    **semimajor_axis_length** : float
        The length of the semimajor axis of the ellipse that has the
        same second-order central moments as the region.

    **semiminor_axis_length** : float
        The length of the semiminor axis of the ellipse that has the
        same second-order central moments as the region.

    **eccentricity** : float
        The eccentricity of the ellipse that has the same second-order
        moments as the source segment.  The eccentricity is the fraction
        of the distance along the semimajor axis at which the focus
        lies.

        .. math:: e = \\sqrt{1 - \\frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

    **orientation** : float
        The angle in radians between the ``x`` axis and the major axis
        of the ellipse that has the same second-order moments as the
        source segment.  The angle increases in the counter-clockwise
        direction.

    See `SegmentProperties` for the additional properties that can be
    accessed as attributes or keys, returned when ``return_table =
    False``.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import segment_properties
    >>> image = np.arange(16.).reshape(4, 4)
    >>> segm_image = np.array([[1, 1, 0, 0],
    ...                        [1, 0, 0, 2],
    ...                        [0, 0, 2, 2],
    ...                        [0, 2, 2, 0]])
    >>> props = segment_properties(image, segm_image)

    Print some properties of the first object (labeled with ``1`` in the
    segmentation image):

    >>> props[0].id    # id corresponds to label number
    1
    >>> props[0].centroid
    (0.80000000000000004, 0.20000000000000001)
    >>> props[0].area
    3
    >>> props[0].max_value
    4.0

    Print some properties of the second object (labelel with ``2`` in
    the segmentation image):

    >>> props[1].id    # id corresponds to label number
    2
    >>> props[1].centroid
    (2.3636363636363633, 2.0909090909090908)
    >>> props[1].area
    5

    Use ``return_table = True`` to return the properties as a
    `~astropy.table.Table`:

    >>> t = segment_properties(image, segm_image, return_table=True)
    >>> print(t)
     id   xcentroid     ycentroid   ...  eccentricity    orientation
    --- ------------- ------------- ... -------------- ---------------
      1           0.2           0.8 ...            1.0 -0.785398163397
      2 2.09090909091 2.36363636364 ... 0.930987270026 -0.741759306923
    >>> t[0]['max_value']
    4.0
    """

    from scipy import ndimage

    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')

    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    labelslices = ndimage.find_objects(segment_image)
    segm_propslist = []
    for i, labelslice in enumerate(labelslices):
        label = i + 1    # consecutive even if some label numbers are missing
        # labelslice is None for missing label numbers
        if labelslice is None or label not in label_ids:
            continue
        segm_props = SegmentProperties(data, segment_image, label,
                                       labelslice=labelslice, mask=mask,
                                       background=background)
        segm_propslist.append(segm_props)

    if not return_table:
        return segm_propslist
    else:
        props_table = Table()
        columns = ['id', 'xcentroid', 'ycentroid', 'xmin', 'xmax', 'ymin',
                   'ymax', 'min_value', 'max_value', 'minval_xpos',
                   'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area',
                   'equivalent_radius', 'perimeter', 'semimajor_axis_length',
                   'semiminor_axis_length', 'eccentricity', 'orientation']
        for column in columns:
            values = [getattr(props, column) for props in segm_propslist]
            props_table[column] = Column(values)
        return props_table


def segment_photometry(data, segment_image, error=None, gain=None,
                       mask=None, mask_method='exclude', background=None,
                       labels=None):
    """
    Perform photometry of sources defined by a labeled segmentation
    image.

    When the segmentation image is defined using a thresholded flux
    level (e.g., see `detect_sources`), this is equivalent to performing
    isophotal photometry in `SExtractor`_.

    .. _SExtractor : http://www.astromatic.net/software/sextractor

    Parameters
    ----------
    data : array_like
        The 2D array on which to perform photometry.

    segment_image : array_like
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``image``.  If
        ``gain`` is input, then ``error`` should include all sources of
        "background" error but *exclude* the Poission error of the
        sources.  If ``gain`` is `None`, then the ``error_image`` is
        assumed to include *all* sources of error, including the
        Poission error of the sources.  ``error`` must have the same
        shape as ``image``.

    gain : float or array-like, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.  If
        ``gain`` is input, then ``error`` should include all sources of
        "background" error but *exclude* the Poission error of the
        sources.  If ``gain`` is `None`, then the ``error`` is assumed
        to include *all* sources of error, including the Poission error
        of the sources.  For example, if your input ``data`` is in units
        of ADU, then ``gain`` should represent electrons/ADU.  If your
        input ``data`` is in units of electrons/s then ``gain`` should
        be the exposure time.

    mask : array_like, bool, optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is
        masked.  Use the ``mask_method`` keyword to select the method
        used to treat masked pixels.

    mask_method : {'exclude', 'interpolate'}, optional
        Method used to treat masked pixels.  The currently supported
        methods are:

        'exclude':
            Exclude masked pixels from all calculations.  This is the
            default.

        'interpolate':
            The value of masked pixels are replaced by the mean value of
            the 8-connected neighboring non-masked pixels.

    background : float or array_like, optional
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None` (the
        default).

    labels : int, sequence of ints or None
        Subset of ``segment_image`` labels for which to perform the
        photometry.  If `None`, then photometry will be performed for
        all source segments (the default).

    Returns
    -------
    table : `astropy.table.Table`
        A table of the photometry of the segmented sources containing
        the following columns:

        * ``'id'``: The source identification number corresponding to
          the object label in ``segment_image``.
        * ``'segment_sum'``: The sum of the image values within the
          source segment.
        * ``'segment_sum_err'``: The corresponding uncertainty of
          ``'segment_sum'`` values.  Returned only if ``error`` is
          input.
        * ``'background_sum'``: The sum of background values within the
          source segment.  Returned only if ``background`` is input.
        * ``'background_mean'``: The mean of background values within
          the source segment.  Returned only if ``background`` is input.

    See Also
    --------
    detect_sources, segment_properties

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import segment_photometry
    >>> image = np.arange(16.).reshape(4, 4)
    >>> error = np.sqrt(image)
    >>> segm_image = np.array([[1, 1, 0, 0],
    ...                        [1, 0, 0, 2],
    ...                        [0, 0, 2, 2],
    ...                        [0, 2, 2, 0]])
    >>> t = segment_photometry(image, segm_image, error=error)
    >>> print(t)
     id segment_sum segment_sum_err
    --- ----------- ---------------
      1         5.0    2.2360679775
      2        55.0    7.4161984871
    """

    from scipy import ndimage

    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')

    data, variance, background = _condition_data(
        data, error=error, gain=gain, mask=mask, mask_method=mask_method,
        background=background)

    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)
    segment_sum = ndimage.measurements.sum(
        data, labels=segment_image, index=label_ids)
    columns = [label_ids, segment_sum]
    names = ('id', 'segment_sum')
    phot_table = Table(columns, names=names)

    if error is not None:
        segment_sum_var = ndimage.measurements.sum(
            variance, labels=segment_image, index=label_ids)
        segment_sum_err = np.sqrt(segment_sum_var)
        phot_table['segment_sum_err'] = segment_sum_err

    if background is not None:
        background_sum = ndimage.measurements.sum(
            background, labels=segment_image, index=label_ids)
        background_mean = ndimage.measurements.mean(
            background, labels=segment_image, index=label_ids)
        phot_table['background_sum'] = background_sum
        phot_table['background_mean'] = background_mean

    return phot_table


def _condition_data(data, error=None, gain=None, mask=None,
                    mask_method='exclude', background=None):
    """Condition the data, error, and background inputs."""

    if background is not None:
        data, background = _subtract_background(data, background)

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        variance = error**2
        if gain is not None:
            variance = _apply_gain(data, variance, gain)
    else:
        variance = None

    if mask is not None:
        data, variance, background = _apply_mask(
            data, mask, mask_method, variance=variance, background=background)

    return data, variance, background


def _subtract_background(data, background):
    """Subtract background from data."""
    if np.isscalar(background):
        bkgrd_image = np.zeros_like(data) + background
    else:
        if background.shape != data.shape:
            raise ValueError('If input background is 2D, then it must '
                             'have the same shape as the input data.')
        bkgrd_image = background
    return (data - bkgrd_image), bkgrd_image


def _apply_mask(data, mask, mask_method, variance=None, background=None):
    """Apply mask to data, variance, and background images."""
    if data.shape != mask.shape:
        raise ValueError('data and mask must have the same shape')

    mask_idx = mask.nonzero()
    if mask_method == 'exclude':
        # masked pixels will not contribute to sums
        data[mask_idx] = 0.0
        if background is not None:
            background[mask_idx] = 0.0
        if variance is not None:
            variance[mask_idx] = 0.0
    elif mask_method == 'interpolate':
        for j, i in zip(*mask_idx):
            y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
            x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
            goodpix = ~mask[y0:y1, x0:x1]
            data[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])
            if background is not None:
                background[j, i] = np.mean(
                    background[y0:y1, x0:x1][goodpix])
            if variance is not None:
                variance[j, i] = np.sqrt(np.mean(
                    variance[y0:y1, x0:x1][goodpix]))
    else:
        raise ValueError(
            'mask_method "{0}" is not valid'.format(mask_method))
    return data, variance, background


def _apply_gain(data, variance, gain):
    """Apply gain to variance images."""
    if np.isscalar(gain):
        gain = np.broadcast_arrays(gain, data)[0]
    gain = np.asarray(gain)
    if gain.shape != data.shape:
        raise ValueError('If input gain is 2D, then it must have '
                         'the same shape as the input data.')
    if np.any(gain <= 0):
        raise ValueError('gain must be positive everywhere')
    return (variance + (data / gain))
