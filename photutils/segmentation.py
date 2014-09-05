# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from astropy.table import Table, Column
import skimage
from skimage.measure._regionprops import _cached_property


__all__ = ['segment_properties', 'segment_photometry']


class _SegmentProperties(object):
    """Class to calculate morphological properties of source segments."""

    def __init__(self, image, segment_image, label, slice, mask=None,
                 background=None):
        self._image = image
        self._segment_image = segment_image
        self.label = label
        self._slice = slice
        self._mask = mask
        self._background = background
        self._cache_active = True

    def __getitem__(self, key):
        return getattr(self, key, None)

    @_cached_property
    def _in_segment(self):
        """
        _in_segment is True for pixels in the labeled object segment.
        """
        return self._segment_image[self._slice] == self.label

    @_cached_property
    def _local_mask(self):
        """
        _local_mask is True for regions outside of the labeled object
        segment or where the input mask is True.
        """
        if self._mask is None:
            return ~self._in_segment
        else:
            return np.logical_or(~self._in_segment, self._mask[self._slice])

    @_cached_property
    def cutout_image(self):
        return self._image[self._slice]

    @_cached_property
    def isolated_cutout_image(self):
        return np.where(self._in_segment, self.cutout_image, np.nan)

    @_cached_property
    def cutout_image_maskedarray(self):
        return np.ma.masked_array(self.cutout_image, mask=self._local_mask)

    @_cached_property
    def _cutout_image_maskzeroed(self):
        return self.cutout_image * ~self._local_mask

    @_cached_property
    def _cutout_image_maskzeroed_double(self):
        return self._cutout_image_maskzeroed.astype(np.double)

    @_cached_property
    def coords(self):
        """(y, x) coordinates of pixels in the object segment."""
        yy, xx = np.nonzero(self._cutout_image_maskzeroed)
        return (yy + self._slice[0].start, xx + self._slice[1].start)

    @_cached_property
    def values(self):
        """Non-masked values in the object segment."""
        return self.cutout_image[~self._local_mask]

    @_cached_property
    def moments(self):
        """Image moments up to third order."""
        return skimage.measure.moments(
            self._cutout_image_maskzeroed_double, 3)

    @_cached_property
    def moments_central(self):
        """Central moments up to third order."""
        ycentroid, xcentroid = self.local_centroid
        return skimage.measure.moments_central(
            self._cutout_image_maskzeroed_double, ycentroid, xcentroid, 3)

    @_cached_property
    def id(self):
        """Unique identification number."""
        return self.label

    @_cached_property
    def local_centroid(self):
        """
        Centroid (y, x) coordinates, relative to the segment bounding
        box region.
        """
        # TODO: allow alternative centroid methods?
        m = self.moments
        ycentroid = m[0, 1] / m[0, 0]
        xcentroid = m[1, 0] / m[0, 0]
        return ycentroid, xcentroid

    @_cached_property
    def centroid(self):
        """Centroid (y, x) coordinates, relative to the input image."""
        ycen, xcen = self.local_centroid
        return ycen + self._slice[0].start, xcen + self._slice[1].start

    @_cached_property
    def xcentroid(self):
        return self.centroid[1]

    @_cached_property
    def ycentroid(self):
        return self.centroid[0]

    @_cached_property
    def bbox(self):
        """(ymin, xmin, ymax, xmax) of the region containing the segment."""
        # (stop - 1) to return the max pixel location, not the slice index
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop - 1, self._slice[1].stop - 1)

    @_cached_property
    def xmin(self):
        return self.bbox[1]

    @_cached_property
    def xmax(self):
        return self.bbox[3]

    @_cached_property
    def ymin(self):
        return self.bbox[0]

    @_cached_property
    def ymax(self):
        return self.bbox[2]

    @_cached_property
    def min_value(self):
        return np.min(self.values)

    @_cached_property
    def max_value(self):
        return np.max(self.values)

    @_cached_property
    def minval_local_pos(self):
        return np.argwhere(self.cutout_image_maskedarray == self.min_value)[0]

    @_cached_property
    def maxval_local_pos(self):
        return np.argwhere(self.cutout_image_maskedarray == self.max_value)[0]

    @_cached_property
    def minval_pos(self):
        yp, xp = self.minval_local_pos
        return yp + self._slice[0].start, xp + self._slice[1].start

    @_cached_property
    def maxval_pos(self):
        yp, xp = self.maxval_local_pos
        return yp + self._slice[0].start, xp + self._slice[1].start

    @_cached_property
    def minval_xpos(self):
        return self.minval_pos[1]

    @_cached_property
    def minval_ypos(self):
        return self.minval_pos[0]

    @_cached_property
    def maxval_xpos(self):
        return self.maxval_pos[1]

    @_cached_property
    def maxval_ypos(self):
        return self.maxval_pos[0]

    @_cached_property
    def area(self):
        return len(self.values)

    @_cached_property
    def equivalent_radius(self):
        return np.sqrt(self.area / np.pi)

    @_cached_property
    def perimeter(self):
        return skimage.measure.perimeter(self._in_segment, 4)

    @_cached_property
    def inertia_tensor(self):
        mu = self.moments_central
        a = mu[2, 0]
        b = -mu[1, 1]
        c = mu[0, 2]
        return np.array([[a, b], [b, c]])

    @_cached_property
    def covariance(self):
        mu = self.moments_central
        m = mu / mu[0, 0]
        return np.array([[m[2, 0], m[1, 1]], [m[1, 1], m[0, 2]]])

    @_cached_property
    def covariance_eigvals(self):
        eigvals = np.linalg.eigvals(self.covariance)
        return np.max(eigvals), np.min(eigvals)

    @_cached_property
    def semimajor_axis_length(self):
        # this matches SExtractor's A parameter
        return np.sqrt(self.covariance_eigvals[0])

    @_cached_property
    def semiminor_axis_length(self):
        # this matches SExtractor's B parameter
        return np.sqrt(self.covariance_eigvals[1])

    @_cached_property
    def eccentricity(self):
        l1, l2 = self.covariance_eigvals
        if l1 == 0:
            return 0.
        return np.sqrt(1. - (l2 / l1))

    @_cached_property
    def orientation(self):
        a, b, b, c = self.covariance.flat
        return 0.5 * np.arctan2(2. * b, (a - c))

    @_cached_property
    def background_centroid(self):
        """Background value at the centroid position."""
        if self._background is None:
            return None
        else:
            return self._background[self.ycentroid, self.xcentroid]

    @_cached_property
    def se_elongation(self):
        """SExtractor's elongation property."""
        return self.semimajor_axis_length / self.semiminor_axis_length

    @_cached_property
    def se_ellipticity(self):
        """SExtractor's ellipticity property."""
        return 1.0 - (self.semiminor_axis_length / self.semimajor_axis_length)

    @_cached_property
    def se_x2(self):
        """SExtractor's X2 property."""
        return self.covariance[0, 0]

    @_cached_property
    def se_y2(self):
        """SExtractor's Y2 property."""
        return self.covariance[1, 1]

    @_cached_property
    def se_xy(self):
        """SExtractor's XY property."""
        return self.covariance[0, 1]

    @_cached_property
    def se_cxx(self):
        """SExtractor's CXX property."""
        return ((np.cos(self.orientation) / self.semimajor_axis_length)**2 +
                (np.sin(self.orientation) / self.semiminor_axis_length)**2)

    @_cached_property
    def se_cyy(self):
        """SExtractor's CYY property."""
        return ((np.sin(self.orientation) / self.semimajor_axis_length)**2 +
                (np.cos(self.orientation) / self.semiminor_axis_length)**2)

    @_cached_property
    def se_cxy(self):
        """SExtractor's CXY property."""
        return (2. * np.cos(self.orientation) * np.sin(self.orientation) *
                ((1./self.semimajor_axis_length**2) -
                 (1./self.semiminor_axis_length**2)))


def segment_properties(data, segment_image, mask=None, mask_method='exclude',
                       background=None, labels=None, return_table=False):
    """
    Calculate morphological properties of source segments.

    Parameters
    ----------
    data : array_like
        The 2D array on which to perform photometry.

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
        If `True` then return an astropy `astropy.table.Table`,
        otherwise return a list of property objects.

    Returns
    -------
    output : `astropy.table.Table` or list of property objects

        * If ``return_table = True``: `astropy.table.Table`
              A table of the properties of the segmented sources
              containing the columns listed below.

        * If ``return_table = False``: list
              A list of property objects, one for each source segment.
              The properties can be accessed using the attributes listed
              below.

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

    **The following properties can be accessed only as attributes or
    keys of property objects, returned when** ``return_table = False``

    **background_centroid** : float
        The value of the background at the source centroid.

    **bbox** : 4-tuple
        The bounding box ``(ymin, xmin, ymax, xmax)`` of the source
        segment.

    **centroid** : 2-tuple
        The image ``(y, x)`` coordinates of the centroid.

    **coords** : (N, 2) `numpy.ndarray`
        The ``(y, x)`` pixel coordinate list of the source segment.

    **covariance** : (2, 2) `numpy.ndarray`
        The covariance matrix of the ellipse that has the same
        second-order moments as the source segment.

    **covariance_eigvals** : 2-tuple
        The two eigenvalues of the covariance matrix in decreasing order.

    **cutout_image** : `numpy.ndarray`
        A 2D cutout image based on the bounding box (``bbox``) of the
        source segment.

    **cutout_image_maskedarray** : `numpy.ma.MaskedArray`
        A 2D cutout image of the source segment where pixels outside of
        the segment are masked.

    **inertia_tensor** : (2, 2) `numpy.ndarray`
        Inertia tensor of the source segment for the rotation around its
        mass.

    **isolated_cutout_image** : `numpy.ndarray`
        A 2D cutout image based on the bounding box (``bbox``) of the
        source segment.  Pixels outside of the source segment have a
        value of ``numpy.nan``.

    **local_centroid** : 2-tuple
        The ``(y, x)`` coordinates, relative to the ``cutout_image``, of
        the centroid.

    **minval_local_pos**, **maxval_local_pos** : 2-tuple
        The ``(y, x)`` coordinates, relative to the ``cutout_image``, of
        the minimum and maximum pixel values.

    **minval_pos**, **maxval_pos** : 2-tuple
        The image coordinates ``(y, x)`` of the minimum and maximum
        pixel values.

    **moments** : (3, 3) `numpy.ndarray`
        Spatial moments up to 3rd order of the source segment.

    **moments_central** : (3, 3) `numpy.ndarray`
        Central moments (translation invariant) of the source segment
        up to 3rd order.

    **se_cxx**, **se_cxy**, **se_cyy** : float
        SExtractor's ``CXX``, ``CXY``, and ``CYY`` ellipse parameters in
        units of :math:`\mathrm{pixel}^{-2}`.

    **se_ellipticity** : float
        SExtractor's ellipticity parameter.

        .. math:: \mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

    **se_elongation** : float
        SExtractor's elongation parameter.

        .. math:: \mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

    **se_x2**, **se_xy**, **se_y2** : float
        SExtractor's ``X2_IMAGE``, ``XY_IMAGE``, and ``Y2_IMAGE``
        parameters, in units of :math:`\mathrm{pixel}^{2}`, which
        correspond to elements of the ``covariance`` matrix.

    **values**: `numpy.ndarray`
        The pixel values within the source segment.
    """

    from scipy import ndimage
    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')

    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    if background is not None:
        data = _subtract_background(data, background)

    objslices = ndimage.find_objects(segment_image)
    objpropslist = []
    for i, objslice in enumerate(objslices):
        label = i + 1    # consecutive even if some label numbers are missing
        # objslice is None for missing label numbers
        if objslice is None or label not in label_ids:
            continue
        # input mask here only if 'exclude'
        objprops = _SegmentProperties(data, segment_image, label, objslice,
                                      mask=mask, background=background)
        objpropslist.append(objprops)

    if not return_table:
        return objpropslist
    else:
        props_table = Table()
        columns = ['id', 'xcentroid', 'ycentroid', 'xmin', 'xmax', 'ymin',
                   'ymax', 'min_value', 'max_value', 'minval_xpos',
                   'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area',
                   'equivalent_radius', 'perimeter', 'semimajor_axis_length',
                   'semiminor_axis_length', 'eccentricity', 'orientation']
        for column in columns:
            values = [getattr(props, column) for props in objpropslist]
            props_table[column] = Column(values)
        return props_table


def segment_photometry(data, segment_image, error=None, gain=None,
                       mask=None, mask_method='exclude', background=None,
                       labels=None):
    """
    Perform photometry of sources whose extents are defined by a labeled
    segmentation image.

    When the segmentation image is defined using a thresholded flux
    level (e.g., see `detect_sources`), this is equivalent to performing
    isophotal photometry in `SExtractor`_.

    .. _SExtractor : http://www.astromatic.net/software/sextractor

    Parameters
    ----------
    data : array_like
        The 2D array on which to perform photometry.

    segment_image : array_like
        A 2D segmentation image where sources are marked by different
        positive integer values.  A value of zero is reserved for the
        background.

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
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is masked
        when computing the photometry.  Use the ``mask_method`` keyword
        to select the method used to treat masked pixels.

    mask_method : {'exclude', 'interpolate'}, optional
        Method used to treat masked pixels.  The currently supported
        methods are:

        'exclude'
            Exclude masked pixels from all calculations.  This is the
            default.

        'interpolate'
            The value of masked pixels are replaced by the mean value of
            the neighboring non-masked pixels.

    background : float or array_like, optional
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None` (which
        is the default).

    labels : int, sequence of ints or None
        Subset of ``segment_image`` labels for which to perform the
        photometry.  If `None`, then photometry will be performed for
        all source segments.

    Returns
    -------
    table : `astropy.table.Table`
        A table of the photometry of the segmented sources containing
        the following columns:

        * ``'id'``: The source identification number corresponding to
          the object label in the ``segment_image``.
        * ``'segment_sum'``: The sum of image values within the source
          segment.
        * ``'segment_sum_err'``: The corresponding uncertainty in
          ``'segment_sum'`` values.  Returned only if ``error`` is
          input.
        * ``'background_sum'``: The sum of background values within the
          source segment.  Returned only if ``background`` is input.
        * ``'background_mean'``: The mean of background values within
          the source segment.  Returned only if ``background`` is input.

    See Also
    --------
    detect_sources, segment_properties
    """

    from scipy import ndimage
    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')
    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    data_iscopy = False
    if background is not None:
        if np.isscalar(background):
            bkgrd_image = np.zeros_like(data) + background
        else:
            if background.shape != data.shape:
                raise ValueError('If input background is 2D, then it must '
                                 'have the same shape as the input data.')
            bkgrd_image = background
        data = copy.deepcopy(data)
        data_iscopy = True
        data -= bkgrd_image

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        variance = error**2

    if mask is not None:
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        if not data_iscopy:
            data = copy.deepcopy(data)

        mask_idx = mask.nonzero()
        if mask_method == 'exclude':
            # masked pixels will not contribute to sums
            data[mask_idx] = 0.0
            if background is not None:
                bkgrd_image[mask_idx] = 0.0
            if error is not None:
                variance[mask_idx] = 0.0
        elif mask_method == 'interpolate':
            for j, i in zip(*mask_idx):
                y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
                x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
                goodpix = ~mask[y0:y1, x0:x1]
                data[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])
                if background is not None:
                    bkgrd_image[j, i] = np.mean(
                        bkgrd_image[y0:y1, x0:x1][goodpix])
                if error is not None:
                    variance[j, i] = np.sqrt(np.mean(
                        variance[y0:y1, x0:x1][goodpix]))
        else:
            raise ValueError(
                'mask_method "{0}" is not valid'.format(mask_method))

    segment_sum = ndimage.measurements.sum(data, labels=segment_image,
                                           index=label_ids)
    columns = [label_ids, segment_sum]
    names = ('id', 'segment_sum')
    phot_table = Table(columns, names=names)

    if error is not None:
        if gain is not None:
            if np.isscalar(gain):
                gain = np.broadcast_arrays(gain, data)[0]
            gain = np.asarray(gain)
            if gain.shape != data.shape:
                raise ValueError('If input gain is 2D, then it must have '
                                 'the same shape as the input data.')
            if np.any(gain <= 0):
                raise ValueError('gain must be positive everywhere')
            variance += data / gain
        segment_sum_var = ndimage.measurements.sum(variance,
                                                   labels=segment_image,
                                                   index=label_ids)
        segment_sum_err = np.sqrt(segment_sum_var)
        phot_table['segment_sum_err'] = segment_sum_err

    if background is not None:
        background_sum = ndimage.measurements.sum(bkgrd_image,
                                                  labels=segment_image,
                                                  index=label_ids)
        background_mean = ndimage.measurements.mean(bkgrd_image,
                                                    labels=segment_image,
                                                    index=label_ids)
        phot_table['background_sum'] = background_sum
        phot_table['background_mean'] = background_mean
    return phot_table


def _subtract_background(data, background):
    if np.isscalar(background):
        bkgrd_image = np.zeros_like(data) + background
    else:
        if background.shape != data.shape:
            raise ValueError('If input background is 2D, then it must '
                             'have the same shape as the input data.')
        bkgrd_image = background
    data = copy.deepcopy(data)
    return data - bkgrd_image
