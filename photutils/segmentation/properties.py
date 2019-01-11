# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs.utils import pixel_to_skycoord


from .core import SegmentationImage
from ..utils.convolution import filter_data
from ..utils._moments import _moments, _moments_central


__all__ = ['SourceProperties', 'source_properties', 'SourceCatalog']

__doctest_requires__ = {('SourceProperties', 'SourceProperties.*',
                         'SourceCatalog', 'SourceCatalog.*',
                         'source_properties', 'properties_table'):
                        ['scipy', 'skimage']}


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
        ``data`` should be background-subtracted.

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
        SExtractor's centroid and morphological parameters are
        calculated from the filtered "detection" image.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  See the Notes section below
        for details on the error propagation.

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
        The WCS transformation to use.  If `None`, then any sky-based
        properties will be set to `None`.

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

    Custom errors for source segments can be calculated using the
    `~photutils.SourceProperties.error_cutout_ma` and
    `~photutils.SourceProperties.background_cutout_ma` properties, which
    are 2D `~numpy.ma.MaskedArray` cutout versions of the input
    ``error`` and ``background``.  The mask is `True` for both pixels
    outside of the source segment and masked pixels from the ``mask``
    input.

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    def __init__(self, data, segment_img, label, filtered_data=None,
                 error=None, mask=None, background=None, wcs=None):

        if not isinstance(segment_img, SegmentationImage):
            segment_img = SegmentationImage(segment_img)

        if segment_img.shape != data.shape:
            raise ValueError('segment_img and data must have the same shape.')

        if error is not None:
            error = np.atleast_1d(error)
            if len(error) == 1:
                error = np.zeros(data.shape) + error
            if error.shape != data.shape:
                raise ValueError('error and data must have the same shape.')

        if mask is np.ma.nomask:
            mask = np.zeros(data.shape).astype(bool)
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape.')

        if background is not None:
            background = np.atleast_1d(background)
            if len(background) == 1:
                background = np.zeros(data.shape) + background
            if background.shape != data.shape:
                raise ValueError('background and data must have the same '
                                 'shape.')

        # data and filtered_data should be background-subtracted
        # for accurate source photometry and properties
        self._data = data
        if filtered_data is None:
            self._filtered_data = data
        else:
            self._filtered_data = filtered_data
        self._error = error    # total error; 2D array
        self._background = background    # 2D array

        segment_img.check_labels(label)
        self.label = label
        self._slice = segment_img.slices[label - 1]
        self._segment_img = segment_img
        self._mask = mask
        self._wcs = wcs

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

        data = np.asanyarray(data)
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
        Create a `~astropy.table.QTable` of properties.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.QTable` will include all scalar-valued
        properties.  Multi-dimensional properties, e.g.
        `~photutils.SourceProperties.data_cutout`, can be included in
        the ``columns`` input.

        Parameters
        ----------
        columns : str or list of str, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`.  The allowed column names are any
            of the attributes of `SourceProperties`.

        exclude_columns : str or list of str, optional
            Names of columns to exclude from the default properties list
            in the output `~astropy.table.QTable`.  The default
            properties are those with scalar values.

        Returns
        -------
        table : `~astropy.table.QTable`
            A single-row table of properties of the source.
        """

        return _properties_table(self, columns=columns,
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
    def _data_cutout_maskzeroed(self):
        """
        A 2D cutout from the (background-subtracted) (filtered) data,
        where pixels outside of the source segment and masked pixels are
        set to zero.

        Invalid values (e.g. NaNs or infs) are set to zero.  Negative
        data values are also set to zero because negative pixels
        (especially at large radii) can result in image moments that
        result in negative variances.
        """

        cutout = self.make_cutout(self._filtered_data, masked_array=False)
        cutout = np.where(np.isfinite(cutout), cutout, 0.)
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
        A tuple of `~numpy.ndarray` containing the ``y`` and ``x`` pixel
        coordinates of the source segment.  Masked pixels are not
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

        return _moments(self._data_cutout_maskzeroed, order=3)

    @lazyproperty
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """

        ycentroid, xcentroid = self.cutout_centroid.value
        return _moments_central(self._data_cutout_maskzeroed,
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

        m = self.moments
        if m[0, 0] != 0:
            ycentroid = m[1, 0] / m[0, 0]
            xcentroid = m[0, 1] / m[0, 0]
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
    def sky_centroid(self):
        """
        The sky coordinates of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input WCS.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xcentroid.value,
                                     self.ycentroid.value,
                                     self._wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_centroid_icrs(self):
        """
        The sky coordinates, in the International Celestial Reference
        System (ICRS) frame, of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.
        """

        if self._wcs is not None:
            return self.sky_centroid.icrs
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
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xmin.value - 0.5,
                                     self.ymin.value - 0.5,
                                     self._wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xmin.value - 0.5,
                                     self.ymax.value + 0.5,
                                     self._wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xmax.value + 0.5,
                                     self.ymin.value - 0.5,
                                     self._wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """

        if self._wcs is not None:
            return pixel_to_skycoord(self.xmax.value + 0.5,
                                     self.ymax.value + 0.5,
                                     self._wcs, origin=0)
        else:
            return None

    @lazyproperty
    def min_value(self):
        """
        The minimum pixel value of the (background-subtracted) data
        within the source segment.
        """

        return np.nanmin(self.values)

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value of the (background-subtracted) data
        within the source segment.
        """

        return np.nanmax(self.values)

    @lazyproperty
    def minval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the (background-subtracted) data.
        """

        # NOTE:  Quantity converts this to float
        return np.argwhere(self.data_cutout_ma == self.min_value)[0] * u.pix

    @lazyproperty
    def maxval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the (background-subtracted) data.
        """

        # NOTE:  Quantity converts this to float
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
        a = mu[0, 2]
        b = -mu[1, 1]
        c = mu[2, 0]
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
                np.array([[m[0, 2], m[1, 1]], [m[1, 1], m[2, 0]]]))
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
    def source_sum(self):
        """
        The sum of the non-masked (background-subtracted) data values
        within the source segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``source_sum``, :math:`(I_i - B_i)` is the
        background-subtracted input ``data``, and :math:`S` are the
        non-masked pixels in the source segment.
        """

        return np.nansum(np.ma.masked_array(self._data[self._slice],
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

        where :math:`\\Delta F` is ``source_sum_err``,
        :math:`\\sigma_{\\mathrm{tot, i}}` are the pixel-wise total
        errors, and :math:`S` are the non-masked pixels in the source
        segment.
        """

        if self._error is not None:
            # power doesn't work here, see astropy #2968
            # return np.sqrt(np.sum(self.error_cutout_ma**2))
            return np.sqrt(np.nansum(
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
            value = map_coordinates(self._background,
                                    [[self.ycentroid.value],
                                     [self.xcentroid.value]])[0]

            # map_coordinates works if self._background is a Quantity, but
            # the returned value is a numpy array (without units)
            if isinstance(self._background, u.Quantity):
                value *= self._background.unit

            return value


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

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  See the Notes section below
        for details on the error propagation.

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
        The WCS transformation to use.  If `None`, then any sky-based
        properties will be set to `None`.

    labels : int, array-like (1D, int)
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

    >>> props[0].id    # id corresponds to segment label number
    1
    >>> props[0].centroid    # doctest: +FLOAT_CMP
    <Quantity [0.8, 0.2] pix>
    >>> props[0].source_sum    # doctest: +FLOAT_CMP
    5.0
    >>> props[0].area    # doctest: +FLOAT_CMP
    <Quantity 3. pix2>
    >>> props[0].max_value    # doctest: +FLOAT_CMP
    4.0

    Print some properties of the second object (labeled with ``2`` in
    the segmentation image):

    >>> props[1].id    # id corresponds to segment label number
    2
    >>> props[1].centroid    # doctest: +FLOAT_CMP
    <Quantity [2.36363636, 2.09090909] pix>
    >>> props[1].perimeter    # doctest: +FLOAT_CMP
    <Quantity 5.41421356 pix>
    >>> props[1].orientation    # doctest: +FLOAT_CMP
    <Quantity -0.74175931 rad>
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('segment_img and data must have the same shape.')

    # filter the data once, instead of repeating for each source
    if filter_kernel is not None:
        filtered_data = filter_data(data, filter_kernel, mode='constant',
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

    return SourceCatalog(sources_props, wcs=wcs)


class SourceCatalog:
    """
    Class to hold source catalogs.
    """

    def __init__(self, properties_list, wcs=None):
        if isinstance(properties_list, SourceProperties):
            self._data = [properties_list]
        elif isinstance(properties_list, list):
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

    def __getattr__(self, attr):
        exclude = ['sky_centroid', 'sky_centroid_icrs', 'icrs_centroid',
                   'ra_icrs_centroid', 'dec_icrs_centroid', 'sky_bbox_ll',
                   'sky_bbox_ul', 'sky_bbox_lr', 'sky_bbox_ur']
        if attr not in exclude:
            if attr not in self._cache:
                values = [getattr(p, attr) for p in self._data]

                if isinstance(values[0], u.Quantity):
                    # turn list of Quantities into a Quantity array
                    values = u.Quantity(values)
                if isinstance(values[0], SkyCoord):   # pragma: no cover
                    # turn list of SkyCoord into a SkyCoord array
                    values = SkyCoord(values)

                self._cache[attr] = values

            return self._cache[attr]

    @lazyproperty
    def sky_centroid(self):
        if self.wcs is not None:
            # For a large catalog, it's much faster to calculate world
            # coordinates using the complete list of (x, y) instead of
            # looping through the individual (x, y).  It's also much
            # faster to recalculate the world coordinates than to create a
            # SkyCoord array from a loop-generated SkyCoord list.  The
            # assumption here is that the wcs is the same for each
            # SourceProperties instance.
            return pixel_to_skycoord(self.xcentroid, self.ycentroid,
                                     self.wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_centroid_icrs(self):
        if self.wcs is not None:
            return self.sky_centroid.icrs
        else:
            return None

    @lazyproperty
    def sky_bbox_ll(self):
        if self.wcs is not None:
            return pixel_to_skycoord(self.xmin.value - 0.5,
                                     self.ymin.value - 0.5,
                                     self.wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_ul(self):
        if self.wcs is not None:
            return pixel_to_skycoord(self.xmin.value - 0.5,
                                     self.ymax.value + 0.5,
                                     self.wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_lr(self):
        if self.wcs is not None:
            return pixel_to_skycoord(self.xmax.value + 0.5,
                                     self.ymin.value - 0.5,
                                     self.wcs, origin=0)
        else:
            return None

    @lazyproperty
    def sky_bbox_ur(self):
        if self.wcs is not None:
            return pixel_to_skycoord(self.xmax.value + 0.5,
                                     self.ymax.value + 0.5,
                                     self.wcs, origin=0)
        else:
            return None

    def to_table(self, columns=None, exclude_columns=None):
        """
        Construct a `~astropy.table.QTable` of source properties from a
        `SourceCatalog` object.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.QTable` will include most scalar-valued source
        properties.  Multi-dimensional properties, e.g.
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
            Names of columns to exclude from the default properties list
            in the output `~astropy.table.QTable`.  The default
            properties are those with scalar values:

            'id', 'xcentroid', 'ycentroid', 'sky_centroid',
            'sky_centroid_icrs', 'source_sum', 'source_sum_err',
            'background_sum', 'background_mean',
            'background_at_centroid', 'xmin', 'xmax', 'ymin', 'ymax',
            'min_value', 'max_value', 'minval_xpos', 'minval_ypos',
            'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius',
            'perimeter', 'semimajor_axis_sigma', 'semiminor_axis_sigma',
            'eccentricity', 'orientation', 'ellipticity', 'elongation',
            'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy',
            'cyy'

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

    if isinstance(obj, SourceCatalog) and len(obj) == 0:
        raise ValueError('SourceCatalog contains no sources.')

    # all scalar-valued properties
    columns_all = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'sky_centroid_icrs', 'source_sum', 'source_sum_err',
                   'background_sum', 'background_mean',
                   'background_at_centroid', 'xmin', 'xmax', 'ymin',
                   'ymax', 'min_value', 'max_value', 'minval_xpos',
                   'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area',
                   'equivalent_radius', 'perimeter',
                   'semimajor_axis_sigma', 'semiminor_axis_sigma',
                   'eccentricity', 'orientation', 'ellipticity',
                   'elongation', 'covar_sigx2', 'covar_sigxy',
                   'covar_sigy2', 'cxx', 'cxy', 'cyy']

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
            values = np.atleast_1d(values)
            if isinstance(values[0], u.Quantity):
                # turn list of Quantities into a Quantity array
                values = u.Quantity(values)
            if isinstance(values[0], SkyCoord):   # pragma: no cover
                # turn list of SkyCoord into a SkyCoord array
                values = SkyCoord(values)

        if isinstance(obj, SourceCatalog) and values is None:
            values = [None] * len(obj)

        tbl[column] = values

    return tbl
