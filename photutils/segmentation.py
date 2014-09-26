# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from astropy.table import Table
from astropy.utils import lazyproperty
import astropy.units as u
from astropy.utils.misc import isiterable


__all__ = ['SegmentProperties', 'segment_properties', 'properties_table']
__doctest_requires__ = {('segment_properties', 'properties_table'): ['scipy'],
                        ('segment_properties', 'properties_table'):
                        ['skimage']}


class SegmentProperties(object):
    """
    Class to calculate photometry and morphological properties of source
    segments.
    """

    def __init__(self, data, segment_image, label, label_slice=None,
                 error=None, effective_gain=None, mask=None,
                 mask_method='exclude', background=None):
        """
        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity`
            The 2D array from which to calculate the source photometry
            and properties.

        segment_image : array_like
            A 2D segmentation image, with the same shape as ``data``,
            where sources are marked by different positive integer
            values.  A value of zero is reserved for the background.

        label : int
            The label number of a source segment in ``segment_image``
            for which to calculate properties.

        label_slice : 2-tuple of slice objects, optional
            A ``(y_slice, x_slice)`` tuple of slice objects defining the
            minimal box enclosing the source segment.  If `None` (the
            default), then ``label_slice`` will be calculated.

        error : array_like or `~astropy.units.Quantity`, optional
            The pixel-wise Gaussian 1-sigma errors of the input
            ``data``.  If ``effective_gain`` is input, then ``error``
            should include all sources of "background" error but
            *exclude* the Poisson error of the sources.  If
            ``effective_gain`` is `None`, then the ``error_image`` is
            assumed to include *all* sources of error, including the
            Poisson error of the sources.  ``error`` must have the same
            shape as ``data``.  See the Notes section below for details
            on the error propagation.

        effective_gain : float, array-like, or `~astropy.units.Quantity`, optional
            Ratio of counts (e.g., electrons or photons) to the units of
            ``data`` used to calculate the Poisson error of the sources.
            If ``effective_gain`` is `None`, then the ``error`` is
            assumed to include *all* sources of error.  See the Notes
            section below for details on the error propagation.

        mask : array_like (bool), optional
            A boolean mask, with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Use the ``mask_method`` keyword to select the
            method used to treat masked pixels.

        mask_method : {'exclude', 'interpolate'}, optional
            Method used to treat masked pixels.  The currently supported
            methods are:

            'exclude':
                Exclude masked pixels from all calculations.  This is
                the default.

            'interpolate':
                The value of masked pixels are replaced by the mean
                value of the 8-connected neighboring non-masked pixels.

        background : float, array_like, or `~astropy.units.Quantity`, optional
            The background level of the input ``data``.  ``background``
            may either be a scalar value or a 2D image with the same
            shape as the input ``data``.  If the input ``data`` has been
            background-subtracted, then set ``background`` to `None`
            (the default) or ``0.``.

        Notes
        -----
        If ``effective_gain`` is input, then ``error`` should include
        all sources of "background" error but *exclude* the Poisson
        error of the sources.  The total error image,
        :math:`\sigma_{\mathrm{tot}}` is then:

        .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                      \\frac{I}{g}}

        where :math:`\sigma_b`, :math:`I`, and :math:`g` are the
        background ``error`` image, ``data`` image, and
        ``effective_gain``, respectively.

        If ``effective_gain`` is `None`, then ``error`` is assumed to
        include *all* sources of error, including the Poisson error of
        the sources, i.e. :math:`\sigma_{\mathrm{tot}} =
        \mathrm{error}`.

        For example, if your input ``data`` are in units of ADU, then
        ``effective_gain`` should represent electrons/ADU.  If your
        input ``data`` are in units of electrons/s then
        ``effective_gain`` should be the exposure time.

        `~photutils.SegmentProperties.segment_sum_err` is simply the
        quadrature sum of the pixel-wise total errors over the pixels
        within the source segment:

        .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
                  \\sigma_{\\mathrm{tot}, i}^2}

        where :math:`\Delta F` is
        `~photutils.SegmentProperties.segment_sum_err` and :math:`S` are
        the pixels in the source segment.

        Custom errors for source segments can be calculated using the
        `~photutils.SegmentProperties.error_cutout_ma` and
        `~photutils.SegmentProperties.background_cutout_ma` properties,
        which are 2D `~numpy.ma.MaskedArray` cutout versions of the
        input ``error`` and ``background``.  The mask is `True` for both
        pixels outside of the source segment and "excluded" masked
        pixels.
        """

        from scipy import ndimage

        if segment_image.shape != data.shape:
            raise ValueError('segment_image and data must have the same '
                             'shape')

        if label == 0:
            raise ValueError('label "0" is reserved for the background')
        elif label < 0:
            raise ValueError('label must be a positive integer')

        self._inputimage = data
        self._segment_image = segment_image
        image, error, background = _prepare_data(
            data, error=error, effective_gain=effective_gain, mask=mask,
            mask_method=mask_method, background=background)
        self._image = image
        self._error = error
        self._background = background

        self.label = label
        if label_slice is not None:
            self._slice = label_slice
        else:
            label_slices = ndimage.find_objects(segment_image)
            self._slice = label_slices[label - 1]
            if self._slice is None:
                raise ValueError('label "{0}" is not in the input '
                                 'segment_image'.format(label))
        if mask_method == 'interpolate':
            # interpolated masked pixels are used like unmasked pixels,
            # so no further masking is needed
            self._mask = np.zeros_like(image, dtype=np.bool)
        else:
            # excluded masked pixels still need the mask
            self._mask = mask
        self._cache_active = True

    def __getitem__(self, key):
        return getattr(self, key, None)

    @lazyproperty
    def _in_segment(self):
        """
        _in_segment is `True` for pixels in the labeled source segment.
        """
        return self._segment_image[self._slice] == self.label

    @lazyproperty
    def _local_mask(self):
        """
        _local_mask is `True` for regions outside of the labeled source
        segment or where the input mask ("excluded" mask) is `True`.
        """
        if self._mask is None:
            return ~self._in_segment
        else:
            return np.logical_or(~self._in_segment, self._mask[self._slice])

    @lazyproperty
    def data_cutout(self):
        """
        A 2D cutout from the data of the source segment.
        """
        return self._inputimage[self._slice]

    @lazyproperty
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the data, where the
        mask is `True` for both pixels outside of the source segment and
        "excluded" masked pixels.
        """
        return np.ma.masked_array(self.data_cutout, mask=self._local_mask)

    @lazyproperty
    def _data_cutout_maskzeroed_double(self):
        """
        A 2D cutout from the data, where pixels outside of the source
        segment and "excluded" masked pixels are set to zero.  The
        cutout image is double precision, which is required for
        scikit-image's Cython moment functions.
        """
        return (self.data_cutout * ~self._local_mask).astype(np.float64)

    @lazyproperty
    def error_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input ``error``
        image, where the mask is `True` for both pixels outside of the
        source segment and "excluded" masked pixels.  If ``error`` is
        `None`, then ``error_cutout_ma`` is also `None`.
        """
        if self._error is not None:
            return np.ma.masked_array(self._error[self._slice],
                                      mask=self._local_mask)
        else:
            return None

    @lazyproperty
    def background_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input
        ``background``, where the mask is `True` for both pixels outside
        of the source segment and "excluded" masked pixels.  If
        ``background`` is `None`, then ``background_cutout_ma`` is also
        `None`.
        """
        if self._background is not None:
            return np.ma.masked_array(self._background[self._slice],
                                      mask=self._local_mask)
        else:
            return None

    @lazyproperty
    def coords(self):
        """
        A list of the ``(y, x)`` pixel coordinates of the source
        segment.

        "Excluded" masked pixels are not included, but interpolated
        masked pixels are included.
        """
        yy, xx = np.nonzero(self.data_cutout_ma)
        coords = (yy + self._slice[0].start, xx + self._slice[1].start)
        return coords

    @lazyproperty
    def values(self):
        """
        A list of the pixel values within the source segment.

        Values of "excluded" masked pixels are not included, but
        interpolated masked pixels are included.
        """
        return self.data_cutout[~self._local_mask]

    @lazyproperty
    def moments(self):
        """Spatial moments up to 3rd order of the source segment."""
        from skimage.measure import moments
        return moments(self._data_cutout_maskzeroed_double, 3)

    @lazyproperty
    def moments_central(self):
        """
        Central moments (translation invariant) of the source segment up
        to 3rd order.
        """
        from skimage.measure import moments_central
        ycentroid, xcentroid = self.local_centroid.value
        return moments_central(self._data_cutout_maskzeroed_double,
                               ycentroid, xcentroid, 3)

    @lazyproperty
    def id(self):
        """
        The source identification number corresponding to the object
        label in the ``segment_image``.
        """
        return self.label

    @lazyproperty
    def local_centroid(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the centroid within the source segment.
        """
        # TODO: allow alternative centroid methods?
        m = self.moments
        ycentroid = m[0, 1] / m[0, 0]
        xcentroid = m[1, 0] / m[0, 0]
        return (ycentroid, xcentroid) * u.pix

    @lazyproperty
    def centroid(self):
        """
        The ``(y, x)`` coordinate of the centroid within the source
        segment.
        """
        ycen, xcen = self.local_centroid.value
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
        The left ``x`` pixel location of the minimal bounding box
        (`~photutils.SegmentProperties.bbox`) of the source segment.
        """
        return self.bbox[1]

    @lazyproperty
    def xmax(self):
        """
        The right ``x`` pixel location of the minimal bounding box
        (`~photutils.SegmentProperties.bbox`) of the source segment.
        """
        return self.bbox[3]

    @lazyproperty
    def ymin(self):
        """
        The bottom ``y`` pixel location of the minimal bounding box
        (`~photutils.SegmentProperties.bbox`) of the source segment.
        """
        return self.bbox[0]

    @lazyproperty
    def ymax(self):
        """
        The top ``y`` pixel location of the minimal bounding box
        (`~photutils.SegmentProperties.bbox`) of the source segment.
        """
        return self.bbox[2]

    @lazyproperty
    def min_value(self):
        """The minimum pixel value within the source segment."""
        return np.min(self.values)

    @lazyproperty
    def max_value(self):
        """The maximum pixel value within the source segment."""
        return np.max(self.values)

    @lazyproperty
    def minval_local_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the minimum pixel value.
        """
        return np.argwhere(self.data_cutout_ma == self.min_value)[0] * u.pix

    @lazyproperty
    def maxval_local_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the maximum pixel value.
        """
        return np.argwhere(self.data_cutout_ma == self.max_value)[0] * u.pix

    @lazyproperty
    def minval_pos(self):
        """The ``(y, x)`` coordinate of the minimum pixel value."""
        yp, xp = np.array(self.minval_local_pos)
        return (yp + self._slice[0].start, xp + self._slice[1].start) * u.pix

    @lazyproperty
    def maxval_pos(self):
        """The ``(y, x)`` coordinate of the maximum pixel value."""
        yp, xp = np.array(self.maxval_local_pos)
        return (yp + self._slice[0].start, xp + self._slice[1].start) * u.pix

    @lazyproperty
    def minval_xpos(self):
        """The ``x`` coordinate of the minimum pixel value."""
        return self.minval_pos[1]

    @lazyproperty
    def minval_ypos(self):
        """The ``y`` coordinate of the minimum pixel value."""
        return self.minval_pos[0]

    @lazyproperty
    def maxval_xpos(self):
        """The ``x`` coordinate of the maximum pixel value."""
        return self.maxval_pos[1]

    @lazyproperty
    def maxval_ypos(self):
        """The ``y`` coordinate of the maximum pixel value."""
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
        The perimeter of the source segment, approximated using a line
        through the centers of the border pixels using a 4-connectivity.
        """
        from skimage.measure import perimeter
        return perimeter(self._in_segment, 4) * u.pix

    @lazyproperty
    def inertia_tensor(self):
        """
        Inertia tensor of the source segment for the rotation around its
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
        same second-order moments as the source segment.
        """
        mu = self.moments_central
        m = mu / mu[0, 0]
        return np.array([[m[2, 0], m[1, 1]], [m[1, 1], m[0, 2]]]) * u.pix**2

    @lazyproperty
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        eigvals = np.linalg.eigvals(self.covariance)
        return (np.max(eigvals), np.min(eigvals)) * u.pix**2

    @lazyproperty
    def semimajor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source segment.
        """
        # this matches SExtractor's A parameter
        return np.sqrt(self.covariance_eigvals[0])

    @lazyproperty
    def semiminor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source segment.
        """
        # this matches SExtractor's B parameter
        return np.sqrt(self.covariance_eigvals[1])

    @lazyproperty
    def eccentricity(self):
        """
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source segment.

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
        moments as the source segment.  The angle increases in the
        counter-clockwise direction.
        """
        a, b, b, c = self.covariance.flat
        return 0.5 * np.arctan2(2. * b, (a - c))

    @lazyproperty
    def elongation(self):
        """
        SExtractor's elongation parameter.

        .. math:: \mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self.semimajor_axis_sigma / self.semiminor_axis_sigma

    @lazyproperty
    def ellipticity(self):
        """
        SExtractor's ellipticity parameter.

        .. math:: \mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return 1.0 - (self.semiminor_axis_sigma / self.semimajor_axis_sigma)

    @lazyproperty
    def se_x2(self):
        """
        SExtractor's X2 parameter, in units of pixel**2, which
        corresponds to the ``(0, 0)`` element of the `covariance`
        matrix.
        """
        return self.covariance[0, 0]

    @lazyproperty
    def se_y2(self):
        """
        SExtractor's Y2 parameter, in units of pixel**2, which
        corresponds to the ``(1, 1)`` element of the `covariance`
        matrix.
        """
        return self.covariance[1, 1]

    @lazyproperty
    def se_xy(self):
        """
        SExtractor's XY parameter, in units of pixel**2, which
        corresponds to the ``(0, 1)`` element of the `covariance`
        matrix.
        """
        return self.covariance[0, 1]

    @lazyproperty
    def se_cxx(self):
        """
        SExtractor's CXX ellipse parameter in units of pixel**(-2).
        """
        return ((np.cos(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.sin(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def se_cyy(self):
        """
        SExtractor's CYY ellipse parameter in units of pixel**(-2).
        """
        return ((np.sin(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.cos(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def se_cxy(self):
        """
        SExtractor's CXY ellipse parameter in units of pixel**(-2).
        """
        return (2. * np.cos(self.orientation) * np.sin(self.orientation) *
                ((1. / self.semimajor_axis_sigma**2) -
                 (1. / self.semiminor_axis_sigma**2)))

    @lazyproperty
    def segment_sum(self):
        """
        The sum of the background-subtracted data values within the source
        segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``segment_sum``, :math:`I_i` is the ``data``,
        :math:`B_i` is the ``background``, and :math:`S` are the pixels
        in the source segment.
        """
        return np.sum(np.ma.masked_array(self._image[self._slice],
                                         mask=self._local_mask))

    @lazyproperty
    def segment_sum_err(self):
        """
        The uncertainty of `~photutils.SegmentProperties.segment_sum`,
        propagated from the input ``error`` array.

        ``segment_sum_err`` is the quadrature sum of the total errors
        over the pixels within the source segment:

        .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
                  \\sigma_{\\mathrm{tot}, i}^2}

        where :math:`\Delta F` is ``segment_sum_err``,
        :math:`\sigma_{\mathrm{tot, i}}` are the pixel-wise total
        errors, and :math:`S` are the pixels in the source segment.
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
    def background_atcentroid(self):
        """
        The value of the ``background`` at the position of the source
        centroid.
        """
        if self._background is None:
            return None
        else:
            return self._background[self.ycentroid.value,
                                    self.xcentroid.value]


def segment_properties(data, segment_image, error=None, effective_gain=None,
                       mask=None, mask_method='exclude', background=None,
                       labels=None):
    """
    Calculate photometry and morphological properties of sources defined
    by a labeled segmentation image.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.

    segment_image : array_like
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        If ``effective_gain`` is input, then ``error`` should include
        all sources of "background" error but *exclude* the Poisson
        error of the sources.  If ``effective_gain`` is `None`, then the
        ``error_image`` is assumed to include *all* sources of error,
        including the Poisson error of the sources.  ``error`` must have
        the same shape as ``data``.  See the Notes section below for
        details on the error propagation.

    effective_gain : float, array-like, or `~astropy.units.Quantity`, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data`` used to calculate the Poisson error of the sources.  If
        ``effective_gain`` is `None`, then the ``error`` is assumed to
        include *all* sources of error.  See the Notes section below for
        details on the error propagation.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Use the ``mask_method`` keyword to select the method used to
        treat masked pixels.

    mask_method : {'exclude', 'interpolate'}, optional
        Method used to treat masked pixels.  The currently supported
        methods are:

        'exclude':
            Exclude masked pixels from all calculations.  This is the
            default.

        'interpolate':
            The value of masked pixels are replaced by the mean value of
            the 8-connected neighboring non-masked pixels.

    background : float, array-like, or `~astropy.units.Quantity`, optional
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None` (the
        default).

    labels : int or list of ints
        Subset of ``segment_image`` labels for which to calculate the
        properties.  If `None`, then the properties will be calculated
        for all source segments (the default).

    Returns
    -------
    output : list of `SegmentProperties` objects
        A list of `SegmentProperties` objects, one for each source
        segment.  The properties can be accessed as attributes or keys.

    Notes
    -----
    If ``effective_gain`` is input, then ``error`` should include all
    sources of "background" error but *exclude* the Poisson error of the
    sources.  The total error image, :math:`\sigma_{\mathrm{tot}}` is
    then:

    .. math:: \\sigma_{\\mathrm{tot}} = \\sqrt{\\sigma_{\\mathrm{b}}^2 +
                  \\frac{I}{g}}

    where :math:`\sigma_b`, :math:`I`, and :math:`g` are the background
    ``error`` image, ``data`` image, and ``effective_gain``,
    respectively.

    If ``effective_gain`` is `None`, then ``error`` is assumed to
    include *all* sources of error, including the Poisson error of the
    sources, i.e. :math:`\sigma_{\mathrm{tot}} = \mathrm{error}`.

    For example, if your input ``data`` are in units of ADU, then
    ``effective_gain`` should represent electrons/ADU.  If your input
    ``data`` are in units of electrons/s then ``effective_gain`` should
    be the exposure time.

    See Also
    --------
    detect_sources, properties_table

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

    >>> print(props[0].id)    # id corresponds to segment label number
    1
    >>> print(props[0].centroid)    # doctest: +FLOAT_CMP
    [ 0.8  0.2] pix
    >>> print(props[0].segment_sum)    # doctest: +FLOAT_CMP
    5.0
    >>> print(props[0].area)    # doctest: +FLOAT_CMP
    3.0 pix2
    >>> print(props[0].max_value)    # doctest: +FLOAT_CMP
    4.0

    Print some properties of the second object (labeled with ``2`` in
    the segmentation image):

    >>> print(props[1].id)    # id corresponds to segment label number
    2
    >>> print(props[1].centroid)    # doctest: +FLOAT_CMP
    [ 2.36363636  2.09090909] pix
    >>> print(props[1].perimeter)    # doctest: +FLOAT_CMP
    5.41421356237 pix
    >>> print(props[1].orientation)    # doctest: +FLOAT_CMP
    -0.741759306923 rad
    """

    from scipy import ndimage

    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')

    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    label_slices = ndimage.find_objects(segment_image)
    segm_propslist = []
    for i, label_slice in enumerate(label_slices):
        label = i + 1    # consecutive even if some label numbers are missing
        # label_slice is None for missing label numbers
        if label_slice is None or label not in label_ids:
            continue
        segm_props = SegmentProperties(
            data, segment_image, label, label_slice=label_slice, error=error,
            effective_gain=effective_gain, mask=mask, mask_method=mask_method,
            background=background)
        segm_propslist.append(segm_props)
    return segm_propslist


def properties_table(segment_props, columns=None, exclude_columns=None):
    """
    Construct an `astropy.table.Table` of properties from a list of
    `SegmentProperties` objects.

    If ``columns`` or ``exclude_columns`` are not input, then the
    `~astropy.table.Table` will include all scalar-valued properties.
    Multi-dimensional properties, e.g.
    `~photutils.SegmentProperties.data_cutout`, can be included in the
    ``columns`` input.

    Parameters
    ----------
    segment_props : `SegmentProperties` or list of `SegmentProperties`
        A `SegmentProperties` object or list of `SegmentProperties`
        objects, one for each source segment.

    columns : str or list of str, optional
        Names of columns, in order, to include in the output
        `~astropy.table.Table`.  The allowed column names are any of the
        attributes of `SegmentProperties`.

    exclude_columns : str or list of str, optional
        Names of columns to exclude from the default properties list in
        the output `~astropy.table.Table`.  The default properties are
        those with scalar values.

    Returns
    -------
    table : `~astropy.table.Table`
        A table of properties of the segmented sources, one row per
        source segment.

    See Also
    --------
    detect_sources, segment_properties

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import segment_properties, properties_table
    >>> image = np.arange(16.).reshape(4, 4)
    >>> segm_image = np.array([[1, 1, 0, 0],
    ...                        [1, 0, 0, 2],
    ...                        [0, 0, 2, 2],
    ...                        [0, 2, 2, 0]])
    >>> segm_props = segment_properties(image, segm_image)
    >>> columns = ['id', 'xcentroid', 'ycentroid', 'segment_sum']
    >>> t = properties_table(segm_props, columns=columns)
    >>> print(t)
     id   xcentroid     ycentroid   segment_sum
             pix           pix
    --- ------------- ------------- -----------
      1           0.2           0.8         5.0
      2 2.09090909091 2.36363636364        55.0
    """

    if len(segment_props) == 0:
        raise ValueError('segment_props is an empty list')

    props_table = Table()
    # all scalar-valued properties
    columns_all = ['id', 'xcentroid', 'ycentroid', 'segment_sum',
                   'segment_sum_err', 'background_sum', 'background_mean',
                   'background_atcentroid', 'xmin', 'xmax', 'ymin', 'ymax',
                   'min_value', 'max_value', 'minval_xpos', 'minval_ypos',
                   'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius',
                   'perimeter', 'semimajor_axis_sigma',
                   'semiminor_axis_sigma', 'eccentricity', 'orientation',
                   'ellipticity', 'elongation', 'se_x2', 'se_xy',
                   'se_y2', 'se_cxx', 'se_cxy', 'se_cyy']

    table_columns = None
    if exclude_columns is not None:
        table_columns = [s for s in columns_all if s not in exclude_columns]

    if columns is not None:
        table_columns = np.atleast_1d(columns)

    if table_columns is None:
        table_columns = columns_all

    segment_props = np.atleast_1d(segment_props)
    for column in table_columns:
        values = [getattr(props, column) for props in segment_props]
        if isinstance(values[0], u.Quantity):
            # turn list of Quantities into a Quantity array
            values = u.Quantity(values)
        props_table[column] = values
    return props_table


def _check_units(inputs):
    """Check for consistent units on data, error, and background."""
    has_unit = [hasattr(x, 'unit') for x in inputs]
    if any(has_unit) and not all(has_unit):
        raise ValueError('If any of data, error, or background has units, '
                         'then they all must all have units.')
    if all(has_unit):
        if any([inputs[0].unit != getattr(x, 'unit') for x in inputs[1:]]):
            raise u.UnitsError(
                'data, error, and background units do not match.')


def _prepare_data(data, error=None, effective_gain=None, mask=None,
                  mask_method='exclude', background=None):
    """
    Prepare the data, error, and background arrays.

    Notes
    -----
    ``data``, ``error``, and ``background`` must all have the same units
    if they are `~astropy.units.Quantity`\s.

    If ``effective_gain`` is a `~astropy.units.Quantity`, then it must
    have units such that ``effective_gain * data`` is in units of counts
    (e.g. counts, electrons, or photons).
    """

    inputs = [data, error, background]
    _check_units(inputs)

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        variance = error**2
        if effective_gain is not None:
            # data here should include the background
            variance = _calculate_total_variance(data, variance,
                                                 effective_gain)
    else:
        variance = None

    # subtract background after calculating total variance, but
    # before applying mask
    if background is not None:
        data, background = _subtract_background(data, background)

    if mask is not None:
        # _apply_mask needs 2D background
        data, variance, background = _apply_mask(
            data, mask, mask_method, variance=variance, background=background)

    if error is not None:
        error = np.sqrt(variance)

    return data, error, background


def _subtract_background(data, background):
    """
    Subtract background from data and return a 2D background image.
    """
    if not isiterable(background):
        # NOTE: np.broadcast_arrays() never returns a Quantity
        # background = np.broadcast_arrays(background, data)[0]
        background = np.zeros(data.shape) + background
    else:
        if background.shape != data.shape:
            raise ValueError('If input background is 2D, then it must '
                             'have the same shape as the input data.')
    return (data - background), background


def _calculate_total_variance(data, variance, effective_gain):
    """
    Calculate the total variance, including Poisson noise of sources.

    Notes
    -----
    ``data`` here should not be background-subtracted.  ``data`` should
    include all detected counts, including the background, to properly
    calculate the Poisson errors.  ``data`` is converted to counts using
    the ``effective_gain``.
    """

    has_unit = [hasattr(x, 'unit') for x in [data, effective_gain]]
    if any(has_unit) and not all(has_unit):
        raise ValueError('If either data or effective_gain has units, then '
                         'they both must have units.')
    if all(has_unit):
        count_units = [u.electron, u.photon]
        datagain_unit = (data * effective_gain).unit
        if datagain_unit not in count_units:
            raise u.UnitsError('(data * effective_gain) has units of "{0}", '
                               'but it must have count units (u.electron '
                               'or u.photon).'.format(datagain_unit))

    if not isiterable(effective_gain):
        # NOTE: np.broadcast_arrays() never returns a Quantity
        # effective_gain = np.broadcast_arrays(effective_gain, data)[0]
        effective_gain = np.zeros(data.shape) + effective_gain
    else:
        if effective_gain.shape != data.shape:
            raise ValueError('If input effective_gain is 2D, then it must '
                             'have the same shape as the input data.')
    if np.any(effective_gain <= 0):
        raise ValueError('effective_gain must be positive everywhere')

    if all(has_unit):
        variance_total = np.maximum(
            variance + ((data * data.unit) / effective_gain.value),
            0. * variance.unit)
    else:
        variance_total = np.maximum(variance + (data / effective_gain), 0)
    return variance_total


def _apply_mask(data, mask, mask_method, variance=None, background=None):
    """Apply mask to data, variance, and background images."""
    if data.shape != mask.shape:
        raise ValueError('data and mask must have the same shape')

    data = copy.deepcopy(data)    # ensure input data is not modified
    mask_idx = mask.nonzero()
    if mask_method == 'exclude':
        # excluded masked pixels will not contribute to sums
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
                background[j, i] = np.mean(background[y0:y1, x0:x1][goodpix])
            if variance is not None:
                variance[j, i] = np.mean(variance[y0:y1, x0:x1][goodpix])
    else:
        raise ValueError(
            'mask_method "{0}" is not valid'.format(mask_method))
    return data, variance, background
