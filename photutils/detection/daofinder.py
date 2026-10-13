# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
DAOStarFinder class.
"""

import warnings
from functools import cached_property

import astropy.units as u
import numpy as np

from photutils.detection.core import (_DEPR_DEFAULT, StarFinderBase,
                                      StarFinderCatalogBase,
                                      _handle_deprecated_range,
                                      _StarFinderKernel, _validate_n_brightest,
                                      _validate_xycoords_bounds)
from photutils.utils._convolution import _filter_data
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._quantity_helpers import check_units, isscalar
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['DAOStarFinder']


class DAOStarFinder(StarFinderBase):
    """
    Detect stars in an image using the DAOFIND algorithm (Stetson 1987
    [1]_).

    DAOFIND searches images for local density maxima that have a peak
    amplitude greater than ``threshold`` (approximately, since
    ``threshold`` is applied to a convolved image) and have a size and
    shape similar to the defined 2D Gaussian kernel. The Gaussian
    kernel is defined by the ``fwhm``, ``ratio``, ``theta``, and
    ``sigma_radius`` input parameters.

    The kernel has a positive core and negative wings, and it sums to
    zero. Convolving the image with this kernel is equivalent to a
    linear least-squares fit, at every pixel, of the amplitude of the
    Gaussian centered on that pixel plus a constant local background.
    Each pixel of the convolved image is therefore the height of the
    best-fitting Gaussian at that position, in the units of the input
    ``data``. Because the local background is fitted, the convolved
    image does not depend on the background level. For uncorrelated
    noise, the uncertainty of the fitted height is the pixel noise
    multiplied by the kernel relative error, which is why ``threshold``
    is scaled by that factor by default (see ``scale_threshold``). The
    local background is fitted using only the pixels within the kernel
    footprint, so a larger ``sigma_radius`` gives a less noisy fitted
    height.

    ``DAOStarFinder`` finds the object centroid by fitting the marginal
    x and y 1D distributions of the Gaussian kernel to the marginal x
    and y distributions of the input (unconvolved) ``data`` image.

    ``DAOStarFinder`` calculates the object roundness using two methods.
    The ``roundness_range`` bounds are applied to both measures
    of roundness. The first method (``roundness1``, called ``SROUND``
    in DAOFIND) is based on the source symmetry and is the ratio of a
    measure of the object's bilateral (2-fold) to four-fold symmetry.
    The second roundness statistic (``roundness2``, called ``GROUND``
    in DAOFIND) is the difference between the heights of the
    best-fitting Gaussian functions in x and y, divided by the average
    of the two heights. A circular source will have a zero roundness. A
    source extended in x or y will have a negative or positive
    roundness, respectively.

    The sharpness statistic is the ratio of the difference between
    the height of the central pixel and the mean of the surrounding
    pixels within the kernel footprint in the input (unconvolved)
    image, to the height of the best-fitting Gaussian function at that
    point (the central pixel value of the convolved image).

    Parameters
    ----------
    threshold : float or 2D `~numpy.ndarray`
        The absolute image value above which to select sources. If
        ``threshold`` is a 2D array, it must have the same shape as the
        input ``data``. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

        By default, ``threshold`` is internally scaled by a factor
        derived from the Gaussian kernel, so the effective threshold
        applied to the convolved data may differ from the input value.
        Set ``scale_threshold=False`` to apply the value exactly as
        given.

    fwhm : float
        The full width at half maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor to major axis standard deviations of
        the Gaussian kernel. ``ratio`` must be strictly positive and
        less than or equal to 1.0. The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counterclockwise from the positive
        x-axis. The default is 0.0.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) (:math:`\\sigma = \\mbox{FWHM} / (2
        \\sqrt{2 \\log(2)})`). The default is 1.5.

    sharplo : float, optional
        The lower bound on sharpness for object detection.

        .. deprecated:: 3.0
            This keyword is deprecated and will be removed in version
            4.0. Use ``sharpness_range=(lower, upper)`` instead.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

        .. deprecated:: 3.0
            This keyword is deprecated and will be removed in version
            4.0. Use ``sharpness_range=(lower, upper)`` instead.

    roundlo : float, optional
        The lower bound on roundness for object detection.

        .. deprecated:: 3.0
            This keyword is deprecated and will be removed in version
            4.0. Use ``roundness_range=(lower, upper)`` instead.

    roundhi : float, optional
        The upper bound on roundness for object detection.

        .. deprecated:: 3.0
            This keyword is deprecated and will be removed in version
            4.0. Use ``roundness_range=(lower, upper)`` instead.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders. The default is
        `False`, which is the mode used by DAOFIND.

    n_brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``n_brightest`` is set to `None`, all objects
        will be selected.

    peak_max : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peak_max`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peak_max`` must have the
        same units. If ``peak_max`` is set to `None`, then no peak pixel
        value filtering will be performed.

    xycoords : `None` or Nx2 array_like, optional
        The (x, y) pixel coordinates of the approximate centroid
        positions of identified sources. If ``xycoords`` are input,
        the algorithm will skip the source-finding step, and
        ``threshold``, ``min_separation``, ``exclude_border``, and the
        ``mask`` input to ``find_stars`` are not used to select the
        sources. The positions are rounded to the nearest pixel and
        must be within the bounds of the input ``data``.

    min_separation : `None` or float, optional
        The minimum separation (in pixels) for detected objects. If
        `None` (default) then the minimum separation is calculated as
        ``2.5 * fwhm``. Set to 0 to disable minimum separation. Note
        that large values may result in long run times.

        .. versionchanged:: 3.0
            The default ``min_separation`` changed from 0 to
            ``2.5 * fwhm``. To recover the previous behavior, set
            ``min_separation=0``.

    scale_threshold : bool, optional
        If `True` (default), the input ``threshold`` is multiplied by
        the kernel relative error before being applied to the convolved
        data. This is the behavior of the original DAOFIND algorithm. If
        `False`, the input ``threshold`` is used directly without any
        scaling.

    sharpness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on sharpness for object
        detection. Objects with sharpness outside this range will be
        rejected. If `None`, no sharpness filtering is performed. The
        default is ``(0.2, 1.0)``.

    roundness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on roundness for object
        detection. Objects with roundness outside this range will be
        rejected. Both ``roundness1`` and ``roundness2`` are tested
        against this range. If `None`, no roundness filtering is
        performed. The default is ``(-1.0, 1.0)``.

    See Also
    --------
    IRAFStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peak_max``
    must have the same units as the image.

    For the convolution step, this routine sets pixels beyond the
    image borders to 0.0. The equivalent parameters in DAOFIND are
    ``boundary='constant'`` and ``constant=0.0``. This is not the
    DAOFIND default, which is ``boundary='nearest'``. The two modes
    differ only for sources near the image borders. With the zero
    boundary used here, a star that is truncated by the image border
    can be detected. Its centroid is biased toward the image interior
    and its sharpness and roundness values are not reliable. With the
    DAOFIND default, the centroid of such a star usually falls outside
    of the image and the star is rejected. Set ``exclude_border=True``
    to exclude the sources near the image borders.

    As in DAOFIND, the centroid is computed using the major-axis
    standard deviation of the Gaussian kernel for the x-axis fit and
    the minor-axis standard deviation for the y-axis fit, whatever the
    kernel ``theta``. The centroids are therefore biased for an
    elliptical kernel (``ratio < 1``) whose major axis is not along
    the x-axis. The largest bias occurs for a ``theta`` of 90 or 270
    degrees, where the x offset of the centroid from the peak pixel is
    too large by a factor of ``1 / ratio**2`` and the y offset is too
    small by the same factor. Circular kernels and elliptical kernels
    with a ``theta`` of 0 or 180 degrees are not affected.

    The main differences between `~photutils.detection.DAOStarFinder`
    and `~photutils.detection.IRAFStarFinder` are:

    * `~photutils.detection.IRAFStarFinder` always uses a 2D
      circular Gaussian kernel, while
      `~photutils.detection.DAOStarFinder` can use an elliptical
      Gaussian kernel.

    * `~photutils.detection.IRAFStarFinder` calculates the objects'
      centroid, roundness, and sharpness using image moments.

    This routine differs from DAOFIND for pixels of the convolved
    image that have identical values, which are uncommon for
    floating-point data (a saturated star usually has none). For a
    positive ``min_separation``, such pixels that are within
    ``min_separation`` of each other give only one source (see
    `~photutils.detection.find_peaks`). DAOFIND keeps every pixel that
    is tied for the maximum within the kernel footprint, except for
    those that follow a detected pixel within half the kernel width
    along the same image line. With ``min_separation=0``, this routine
    keeps every pixel that is tied for the maximum within the kernel
    footprint.

    References
    ----------
    .. [1] Stetson, P. 1987, PASP 99, 191
           (https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract)
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, fwhm, ratio=1.0, theta=0.0,
                 sigma_radius=1.5, sharplo=_DEPR_DEFAULT,
                 sharphi=_DEPR_DEFAULT, roundlo=_DEPR_DEFAULT,
                 roundhi=_DEPR_DEFAULT, exclude_border=False,
                 n_brightest=None, peak_max=None, xycoords=None,
                 min_separation=None, scale_threshold=True, *,
                 sharpness_range=(0.2, 1.0),
                 roundness_range=(-1.0, 1.0)):

        # Validate the units, but do not strip them
        inputs = (threshold, peak_max)
        names = ('threshold', 'peak_max')
        check_units(inputs, names)

        if not isscalar(fwhm):
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        sharpness_range = _handle_deprecated_range(
            sharplo, sharphi, sharpness_range,
            'sharp', 'sharpness_range', (0.2, 1.0))
        roundness_range = _handle_deprecated_range(
            roundlo, roundhi, roundness_range,
            'round', 'roundness_range', (-1.0, 1.0))

        if sharpness_range is not None:
            if np.ndim(sharpness_range) != 1 or np.size(sharpness_range) != 2:
                msg = ('sharpness_range must be a 2-element (lower, upper) '
                       'tuple or None')
                raise ValueError(msg)
            sharpness_range = tuple(sharpness_range)

        if roundness_range is not None:
            if np.ndim(roundness_range) != 1 or np.size(roundness_range) != 2:
                msg = ('roundness_range must be a 2-element (lower, upper) '
                       'tuple or None')
                raise ValueError(msg)
            roundness_range = tuple(roundness_range)

        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta % 360.0
        self.sigma_radius = sigma_radius
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * self.fwhm

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                msg = 'xycoords must be shaped as an Nx2 array'
                raise ValueError(msg)
        self.xycoords = xycoords
        self.scale_threshold = scale_threshold

        self.kernel = _StarFinderKernel(self.fwhm,
                                        ratio=self.ratio,
                                        theta=self.theta,
                                        sigma_radius=self.sigma_radius)
        if self.scale_threshold:
            self.threshold_eff = self.threshold * self.kernel.rel_err
        else:
            self.threshold_eff = self.threshold

    def _repr_str_params(self):
        params = ('threshold', 'fwhm', 'ratio', 'theta', 'sigma_radius',
                  'sharpness_range', 'roundness_range',
                  'exclude_border', 'n_brightest', 'peak_max', 'xycoords',
                  'min_separation', 'scale_threshold')
        overrides = {}
        if not isscalar(self.threshold):
            overrides['threshold'] = (
                f'<array; shape={np.shape(self.threshold)}>')
        if self.xycoords is not None:
            overrides['xycoords'] = (
                f'<array; shape={self.xycoords.shape}>')
        return params, overrides

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def _get_raw_catalog(self, data, *, mask=None):
        """
        Get the raw catalog of sources from the input data.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array. The image should be
            background-subtracted.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        cat : `_DAOStarFinderCatalog` or `None`
            A catalog of sources found in the input data. `None` is
            returned if no sources are found.
        """
        convolved_data = _filter_data(data, self.kernel.data, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        if self.xycoords is None:
            # _find_stars returns the integer peak pixel positions.
            # The subpixel centroids are computed later in the catalog.
            xypos = self._find_stars(convolved_data, self.kernel,
                                     self.threshold_eff, mask=mask,
                                     min_separation=self.min_separation,
                                     exclude_border=self.exclude_border)
        else:
            _validate_xycoords_bounds(self.xycoords, data.shape)
            xypos = self.xycoords

        if xypos is None:
            msg = 'No sources were found.'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        return _DAOStarFinderCatalog(data, convolved_data, xypos,
                                     self.threshold,
                                     self.kernel,
                                     sharpness_range=self.sharpness_range,
                                     roundness_range=self.roundness_range,
                                     n_brightest=self.n_brightest,
                                     peak_max=self.peak_max,
                                     scale_threshold=self.scale_threshold)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
            The 2D image array. The image should be
            background-subtracted.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.QTable` or `None`
            A table of found stars. `None` is returned if no stars are
            found. The table contains the following parameters:

            * ``id``: unique object identification number.
            * ``x_centroid, y_centroid``: object centroid.
            * ``sharpness``: object sharpness.
            * ``roundness1``: object roundness based on symmetry.
            * ``roundness2``: object roundness based on marginal Gaussian
              fits.
            * ``n_pixels``: the total number of pixels in the Gaussian
              kernel array.
            * ``peak``: the value of the input data at the central
              pixel of the object.
            * ``flux``: the object instrumental flux calculated as the
              sum of data values within the kernel footprint.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.
            * ``daofind_mag``: the "mag" parameter returned by the DAOFIND
              algorithm. It is a measure of the intensity ratio of the
              amplitude of the best fitting Gaussian function at the
              object position to the detection threshold. This parameter
              is reported only for comparison to the IRAF DAOFIND
              output. It should not be interpreted as a magnitude
              derived from an integrated flux.
        """
        # Validate the units, but do not strip them
        inputs = (data, self.threshold, self.peak_max)
        names = ('data', 'threshold', 'peak_max')
        check_units(inputs, names)

        cat = self._get_raw_catalog(data, mask=mask)
        if cat is None:
            return None

        # Apply all selection filters
        cat = cat.apply_all_filters()
        if cat is None:
            return None

        # Create the output table
        return cat.to_table()


class _DAOStarFinderCatalog(StarFinderCatalogBase):
    """
    Class to create a catalog of the properties of each detected star,
    as defined by DAOFIND.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image. The image should be background-subtracted.

    convolved_data : 2D `~numpy.ndarray`
        The convolved 2D image. If ``data`` is a
        `~astropy.units.Quantity` array, then ``convolved_data`` must
        have the same units.

    xypos : Nx2 `~numpy.ndarray`
        An Nx2 array of (x, y) pixel coordinates denoting the central
        positions of the stars.

    threshold : float or 2D `~numpy.ndarray`
        The absolute image value above which sources were selected. If
        ``threshold`` is a 2D array, it must have the same shape as
        ``data``. If ``data`` is a `~astropy.units.Quantity` array, then
        ``threshold`` must have the same units.

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    sharpness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on sharpness for object
        detection. Objects with sharpness outside this range will be
        rejected. If `None`, no sharpness filtering is performed. The
        default is ``(0.2, 1.0)``.

    roundness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on roundness for object
        detection. Objects with roundness outside this range will be
        rejected. Both ``roundness1`` and ``roundness2`` are tested
        against this range. If `None`, no roundness filtering is
        performed. The default is ``(-1.0, 1.0)``.

    n_brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``n_brightest`` is set to `None`, all objects
        will be selected.

    peak_max : float, None, optional
        The maximum allowed peak pixel value in an object. Objects with
        peak pixel values greater than ``peak_max`` will be rejected.
        This keyword may be used, for example, to exclude saturated
        sources. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``peak_max`` must have the
        same units. If ``peak_max`` is set to `None`, then no peak pixel
        value filtering will be performed.

    scale_threshold : bool, optional
        If `True` (default), the input ``threshold`` is multiplied by
        the kernel relative error to compute the effective threshold
        used by ``daofind_mag``. If `False`, the input ``threshold`` is
        used directly.
    """

    def __init__(self, data, convolved_data, xypos, threshold, kernel, *,
                 sharpness_range=(0.2, 1.0), roundness_range=(-1.0, 1.0),
                 n_brightest=None, peak_max=None, scale_threshold=True):

        # Validate the units, but do not strip them
        inputs = (data, convolved_data, threshold, peak_max)
        names = ('data', 'convolved_data', 'threshold', 'peak_max')
        check_units(inputs, names)

        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)

        self.convolved_data = convolved_data
        self.threshold = threshold
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range

        if scale_threshold:
            self.threshold_eff = threshold * kernel.rel_err
        else:
            self.threshold_eff = threshold
        self.cutout_center = tuple((size - 1) // 2 for size in kernel.shape)
        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'sharpness',
                                'roundness1', 'roundness2', 'n_pixels',
                                'peak', 'flux', 'mag', 'daofind_mag')

    def _get_init_attributes(self):
        """
        Return a tuple of attribute names to copy during slicing.
        """
        return ('data', 'unit', 'convolved_data', 'kernel', 'threshold',
                'sharpness_range', 'roundness_range', 'n_brightest',
                'peak_max', 'threshold_eff', 'cutout_shape',
                'cutout_center', 'default_columns')

    @cached_property
    def cutout_convdata(self):
        """
        The cutout of the convolved data centered on the source
        position.
        """
        return self.make_cutouts(self.convolved_data)

    @cached_property
    def peak(self):
        """
        The value of the original (unconvolved) data at the central
        pixel of each source cutout.
        """
        return self.cutout_data[:, self.cutout_center[0],
                                self.cutout_center[1]]

    @cached_property
    def flux(self):
        """
        The instrumental fluxes, calculated as the sum of the
        unconvolved data values within the kernel footprint.
        """
        return np.sum(self.cutout_data * self.kernel.mask, axis=(1, 2))

    @cached_property
    def convdata_peak(self):
        """
        The value of the convolved data at the central pixel of each
        source cutout.
        """
        return self.cutout_convdata[:, self.cutout_center[0],
                                    self.cutout_center[1]]

    @cached_property
    def roundness1(self):
        """
        The roundness of the source based on symmetry, defined as
        the ratio of a measure of the object's bilateral (2-fold) to
        four-fold symmetry.

        A circular source will have a zero roundness. A source
        extended in x or y will have a negative or positive roundness,
        respectively.

        As in DAOFIND, the symmetry is measured over the central square
        of the convolved-data cutout. For a non-square kernel, the
        pixels beyond the smaller kernel half-width are not used.
        """
        # DAOFIND computes the sums over the central square of the
        # cutout, with a half-width equal to the smaller of the two
        # cutout half-widths. This is the full cutout for a square
        # kernel.
        ycen, xcen = self.cutout_center
        center = min(ycen, xcen)
        cutout_conv = self.cutout_convdata[
            :, ycen - center:ycen + center + 1,
            xcen - center:xcen + center + 1].copy()

        # Set the central (peak) pixel to zero for the sum4 calculation
        cutout_conv[:, center, center] = 0.0

        # Calculate the four roundness quadrants of the central square,
        # which has odd dimensions.
        # quad1 = bottom right
        # quad2 = bottom left
        # quad3 = top left
        # quad4 = top right
        # 3 3 4 4 4
        # 3 3 4 4 4
        # 3 3 x 1 1
        # 2 2 2 1 1
        # 2 2 2 1 1
        quad1 = cutout_conv[:, 0:center + 1, center + 1:]
        quad2 = cutout_conv[:, 0:center, 0:center + 1]
        quad3 = cutout_conv[:, center:, 0:center]
        quad4 = cutout_conv[:, center + 1:, center:]

        axis = (1, 2)
        sum2 = (-quad1.sum(axis=axis) + quad2.sum(axis=axis)
                - quad3.sum(axis=axis) + quad4.sum(axis=axis))
        sum4 = np.abs(cutout_conv).sum(axis=axis)

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return 2.0 * sum2 / sum4

    @cached_property
    def sharpness(self):
        """
        The sharpness of the source, defined as the ratio of the
        difference between the height of the central pixel and the mean
        of the surrounding pixels within the kernel mask in the
        unconvolved data, to the central pixel value of the convolved
        data.
        """
        # Mean value of the unconvolved data (excluding the peak)
        cutout_data_masked = self.cutout_data * self.kernel.mask
        data_mean = ((np.sum(cutout_data_masked, axis=(1, 2)) - self.peak)
                     / (self.kernel.n_pixels - 1))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (self.peak - data_mean) / self.convdata_peak

    def _marginal_weights(self, axis):
        """
        Compute triangular weighting functions for the given axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis for which the marginal weights are computed:

            * 0: for the y-axis (rows)
            * 1: for the x-axis (columns)

        Returns
        -------
        wt : 1D `~numpy.ndarray`
            The 1D weighting function for the given axis.

        wts : 2D `~numpy.ndarray`
            The 2D weighting function for the given axis.

        size : int
            The size of the cutout along the given axis.

        center : int
            The center pixel position of the cutout along the given axis.

        sigma : float
            The standard deviation of the Gaussian kernel used for the
            given axis. As in DAOFIND, this is the major-axis value for
            the x-axis and the minor-axis value for the y-axis, whatever
            the kernel ``theta``.

        dxx : 1D `~numpy.ndarray`
            The array of pixel offsets from the center pixel along the
            given axis. As in DAOFIND, the offsets are the pixel index
            minus the center for the y-axis and the center minus the
            pixel index for the x-axis.
        """
        ycen, xcen = self.cutout_center
        xx = xcen - np.abs(np.arange(self.cutout_shape[1]) - xcen) + 1
        yy = ycen - np.abs(np.arange(self.cutout_shape[0]) - ycen) + 1
        xwt, ywt = np.meshgrid(xx, yy)

        if axis == 0:  # marginal distributions along y-axis (rows)
            wt = np.transpose(ywt)[0]  # 1D
            wts = xwt  # 2D
            size = self.cutout_shape[0]
            center = ycen
            sigma = self.kernel.y_sigma
            dxx = np.arange(size) - center
        elif axis == 1:  # marginal distributions along x-axis (columns)
            wt = xwt[0]  # 1D
            wts = ywt  # 2D
            size = self.cutout_shape[1]
            center = xcen
            sigma = self.kernel.x_sigma
            dxx = center - np.arange(size)

        return wt, wts, size, center, sigma, dxx

    def _marginal_kernel_sums(self, wt, wts, axis, center, size):
        """
        Compute weighted marginal kernel sums.

        Parameters
        ----------
        wt : 1D `~numpy.ndarray`
            The 1D weighting function for the given axis.

        wts : 2D `~numpy.ndarray`
            The 2D weighting function for the given axis.

        axis : {0, 1}
            The axis for which the marginal sums are computed:

            * 0: for the y-axis (rows)
            * 1: for the x-axis (columns)

        center : int
            The center pixel position of the cutout along the given axis.

        size : int
            The size of the cutout along the given axis.

        Returns
        -------
        result : dict
            A dict containing the following precomputed kernel-side
            quantities:

            * ``wt_sum``: the sum of the 1D weighting function.
            * ``kern_sum``: the sum of the 1D kernel distribution
              weighted by the 1D weighting function.
            * ``kern2_sum``: the sum of the square of the 1D kernel
              distribution weighted by the 1D weighting function.
            * ``kern_sum_1d``: the 1D kernel distribution (the kernel
              weighted by the 2D weighting function and summed over
              the perpendicular axis).
            * ``dkern_dx``: the derivative of the 1D kernel
              distribution (the 1D kernel distribution multiplied by
              the pixel offsets from the center).
            * ``dkern_dx_sum``: the sum of the derivative of the 1D
              kernel distribution weighted by the 1D weighting
              function.
            * ``dkern_dx2_sum``: the sum of the square of the
              derivative of the 1D kernel distribution weighted by the
              1D weighting function.
            * ``kern_dkern_dx_sum``: the sum of the product of the 1D
              kernel distribution and its derivative, weighted by the
              1D weighting function.
        """
        dx = center - np.arange(size)

        # Marginal sum: sum over the axis perpendicular to the given
        # axis, weighted by the 2D weighting function
        kern_sum_1d = np.sum(self.kernel.gaussian_kernel_unmasked * wts,
                             axis=1 - axis)
        wt_sum = np.sum(wt)
        kern_sum = np.sum(kern_sum_1d * wt)
        kern2_sum = np.sum(kern_sum_1d**2 * wt)

        dkern_dx = kern_sum_1d * dx
        dkern_dx_sum = np.sum(dkern_dx * wt)
        dkern_dx2_sum = np.sum(dkern_dx**2 * wt)
        kern_dkern_dx_sum = np.sum(kern_sum_1d * dkern_dx * wt)

        return {'wt_sum': wt_sum,
                'kern_sum': kern_sum,
                'kern2_sum': kern2_sum,
                'kern_sum_1d': kern_sum_1d,
                'dkern_dx': dkern_dx,
                'dkern_dx_sum': dkern_dx_sum,
                'dkern_dx2_sum': dkern_dx2_sum,
                'kern_dkern_dx_sum': kern_dkern_dx_sum}

    def _marginal_data_sums(self, wt, wts, axis, dxx, kern_sums):
        """
        Compute weighted marginal data sums.

        Parameters
        ----------
        wt : 1D `~numpy.ndarray`
            The 1D weighting function for the given axis.

        wts : 2D `~numpy.ndarray`
            The 2D weighting function for the given axis.

        axis : {0, 1}
            The axis for which the marginal sums are computed:

            * 0: for the y-axis (rows)
            * 1: for the x-axis (columns)

        dxx : 1D `~numpy.ndarray`
            The array of pixel offsets from the center pixel along the
            given axis.

        kern_sums : dict
            The precomputed kernel-side quantities returned by
            ``_marginal_kernel_sums``.

        Returns
        -------
        result : dict
            A dict containing the following precomputed data-side
            quantities:

            * ``data_sum``: the sum of the 1D data distribution
              weighted by the 1D weighting function.
            * ``data_kern_sum``: the sum of the 1D data distribution
              weighted by the 1D kernel distribution and the 1D
              weighting function.
            * ``data_dkern_dx_sum``: the sum of the 1D data
              distribution weighted by the derivative of the 1D kernel
              distribution and the 1D weighting function.
            * ``data_dx_sum``: the sum of the 1D data distribution
              weighted by the pixel offsets and the 1D weighting
              function.
        """
        cutout_data = self.cutout_data
        if isinstance(cutout_data, u.Quantity):
            cutout_data = cutout_data.value

        # Marginal sum: sum over the axis perpendicular to the given
        # axis, weighted by the 2D weighting function (cutout_data is
        # 3D with shape (N_sources, cutout_size_y, cutout_size_x))
        data_sum_1d = np.sum(cutout_data * wts, axis=2 - axis)
        data_sum = np.sum(data_sum_1d * wt, axis=1)
        data_kern_sum = np.sum(
            data_sum_1d * kern_sums['kern_sum_1d'] * wt, axis=1)
        data_dkern_dx_sum = np.sum(
            data_sum_1d * kern_sums['dkern_dx'] * wt, axis=1)
        data_dx_sum = np.sum(data_sum_1d * dxx * wt, axis=1)

        return {'data_sum': data_sum,
                'data_kern_sum': data_kern_sum,
                'data_dkern_dx_sum': data_dkern_dx_sum,
                'data_dx_sum': data_dx_sum}

    @staticmethod
    def _marginal_lstsq(kern_sums, data_sums, sigma, size):
        """
        Perform the marginal least-squares fit and apply masks.

        Parameters
        ----------
        kern_sums : dict
            The precomputed kernel-side quantities returned by
            ``_marginal_kernel_sums``.

        data_sums : dict
            The precomputed data-side quantities returned by
            ``_marginal_data_sums``.

        sigma : float
            The standard deviation of the Gaussian kernel used for the
            given axis (see ``_marginal_weights``).

        size : int
            The size of the cutout along the given axis.

        Returns
        -------
        result : Nx2 `~numpy.ndarray`
            An array of shape Nx2, where N is the number of detected
            sources, and each row contains the fitted fractional shift
            (dx) and amplitude (hx) for each source.
        """
        wt_sum = kern_sums['wt_sum']
        kern_sum = kern_sums['kern_sum']
        kern2_sum = kern_sums['kern2_sum']
        dkern_dx_sum = kern_sums['dkern_dx_sum']
        dkern_dx2_sum = kern_sums['dkern_dx2_sum']
        kern_dkern_dx_sum = kern_sums['kern_dkern_dx_sum']

        data_sum = data_sums['data_sum']
        data_kern_sum = data_sums['data_kern_sum']
        data_dkern_dx_sum = data_sums['data_dkern_dx_sum']
        data_dx_sum = data_sums['data_dx_sum']

        # Perform linear least-squares fit (where data = hx*kernel)
        # to find the amplitude (hx)
        hx_numer = data_kern_sum - (data_sum * kern_sum) / wt_sum
        hx_denom = kern2_sum - (kern_sum**2 / wt_sum)

        # Reject the star if the fit amplitude is not positive
        mask1 = (hx_numer <= 0.0) | (hx_denom <= 0.0)

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Compute fit amplitude
            hx = hx_numer / hx_denom

            # Compute centroid shift
            dx = ((kern_dkern_dx_sum
                   - (data_dkern_dx_sum - dkern_dx_sum * data_sum))
                  / (hx * dkern_dx2_sum / sigma**2))

            dx2 = data_dx_sum / data_sum

        hsize = size / 2.0
        mask2 = (np.abs(dx) > hsize)
        mask3 = (data_sum == 0.0)
        mask4 = (mask2 & mask3)
        mask5 = (mask2 & ~mask3)

        dx[mask4] = 0.0
        dx[mask5] = dx2[mask5]
        mask6 = (np.abs(dx) > hsize)
        dx[mask6] = 0.0

        hx[mask1] = np.nan
        dx[mask1] = np.nan

        return np.transpose((dx, hx))

    def daofind_marginal_fit(self, *, axis=0):
        """
        Fit 1D Gaussians, defined from the marginal x/y kernel
        distributions, to the marginal x/y distributions of the original
        (unconvolved) image.

        These fits are used to calculate the star centroid and roundness2
        ("GROUND") properties.

        Parameters
        ----------
        axis : {0, 1}, optional
            The axis for which the marginal fit is performed:

            * 0: for the y-axis (rows)
            * 1: for the x-axis (columns)

        Returns
        -------
        result : Nx2 `~numpy.ndarray`
            An array where each row contains the fractional shift (dx)
            of the image centroid relative to the central pixel and the
            height (hx) of the best-fitting Gaussian to the marginal
            distribution of the unconvolved source data, along x or y
            depending on ``axis``. Rejected fits are NaN.
        """
        wt, wts, size, center, sigma, dxx = (
            self._marginal_weights(axis))
        kern_sums = self._marginal_kernel_sums(wt, wts, axis, center,
                                               size)
        data_sums = self._marginal_data_sums(wt, wts, axis, dxx,
                                             kern_sums)
        return self._marginal_lstsq(kern_sums, data_sums, sigma, size)

    @cached_property
    def dx_hx(self):
        """
        The fitted fractional shift (dx) and amplitude (hx) from the
        marginal Gaussian fit along the x-axis.
        """
        return self.daofind_marginal_fit(axis=1)

    @cached_property
    def dy_hy(self):
        """
        The fitted fractional shift (dy) and amplitude (hy) from the
        marginal Gaussian fit along the y-axis.
        """
        return self.daofind_marginal_fit(axis=0)

    @cached_property
    def dx(self):
        """
        The fitted fractional shift in x of the image centroid relative
        to the maximum pixel.
        """
        return np.transpose(self.dx_hx)[0]

    @cached_property
    def dy(self):
        """
        The fitted fractional shift in y of the image centroid relative
        to the maximum pixel.
        """
        return np.transpose(self.dy_hy)[0]

    @cached_property
    def hx(self):
        """
        The height of the best-fitting Gaussian to the marginal x
        distribution of the unconvolved source data.
        """
        return np.transpose(self.dx_hx)[1]

    @cached_property
    def hy(self):
        """
        The height of the best-fitting Gaussian to the marginal y
        distribution of the unconvolved source data.
        """
        return np.transpose(self.dy_hy)[1]

    @cached_property
    def x_centroid(self):
        """
        The fitted x centroid of the source, calculated as the sum of
        the x position of the maximum pixel and the fitted fractional
        shift in x from the marginal Gaussian fit.
        """
        return np.transpose(self.xypos)[0] + self.dx

    @cached_property
    def y_centroid(self):
        """
        The fitted y centroid of the source, calculated as the sum of
        the y position of the maximum pixel and the fitted fractional
        shift in y from the marginal Gaussian fit.
        """
        return np.transpose(self.xypos)[1] + self.dy

    @cached_property
    def roundness2(self):
        """
        The star roundness.

        This roundness parameter is the difference between the heights
        of the best-fitting Gaussian functions in x and y, divided by
        the average of the two heights. A circular source will have a
        zero roundness. A source extended in x or y will have a negative
        or positive roundness, respectively.
        """
        return 2.0 * (self.hx - self.hy) / (self.hx + self.hy)

    @cached_property
    def _threshold_eff_per_source(self):
        """
        Per-source effective threshold values.

        If the input ``threshold`` is a scalar, then this returns an
        array of the same length as the number of sources, where each
        value is the same as the input ``threshold_eff``. If the input
        ``threshold`` is a 2D array, then this returns an array of the
        same length as the number of sources, where each value is the
        value of the input ``threshold_eff`` at the rounded (x, y)
        position of each source.
        """
        if np.ndim(self.threshold_eff) < 2:
            return np.ones(len(self)) * self.threshold_eff
        xpos = np.round(self.xypos[:, 0]).astype(int)
        ypos = np.round(self.xypos[:, 1]).astype(int)
        return self.threshold_eff[ypos, xpos]

    @cached_property
    def daofind_mag(self):
        """
        The "mag" parameter returned by the original DAOFIND algorithm.

        It is a measure of the intensity ratio of the amplitude of the
        best fitting Gaussian function at the object position to the
        detection threshold.
        """
        # Ignore RuntimeWarning if flux is <= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return -2.5 * np.log10(self.convdata_peak
                                   / self._threshold_eff_per_source)

    @cached_property
    def n_pixels(self):
        """
        The total number of pixels in the Gaussian kernel array.
        """
        return np.full(len(self), fill_value=self.kernel.data.size)

    def apply_filters(self):
        """
        Filter the catalog.
        """
        attrs = ('x_centroid', 'y_centroid', 'hx', 'hy', 'sharpness',
                 'roundness1', 'roundness2', 'peak', 'flux')
        newcat = self._filter_finite(attrs)
        if newcat is None:
            return None

        # As in DAOFIND, reject sources with a centroid outside the image
        ny, nx = newcat.data.shape
        mask = ((newcat.x_centroid >= -0.5)
                & (newcat.x_centroid <= nx - 0.5)
                & (newcat.y_centroid >= -0.5)
                & (newcat.y_centroid <= ny - 0.5))

        bounds = [
            ('sharpness', self.sharpness_range),
            ('roundness1', self.roundness_range),
            ('roundness2', self.roundness_range),
        ]
        return newcat._filter_bounds(bounds, initial_mask=mask)
