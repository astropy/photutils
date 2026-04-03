# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
IRAFStarFinder class.
"""

import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.detection.core import (_DEPR_DEFAULT, StarFinderBase,
                                      StarFinderCatalogBase,
                                      _handle_deprecated_range,
                                      _StarFinderKernel, _validate_n_brightest)
from photutils.utils._convolution import _filter_data
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._quantity_helpers import check_units, isscalar
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['IRAFStarFinder']


class IRAFStarFinder(StarFinderBase):
    """
    Detect stars in an image using IRAF's "starfind" algorithm.

    `IRAFStarFinder` searches images for local density maxima that
    have a peak amplitude greater than ``threshold`` above the local
    background and have a PSF full-width at half-maximum similar to the
    input ``fwhm``. The objects' centroid, roundness (ellipticity), and
    sharpness are calculated using image moments.

    Parameters
    ----------
    threshold : float or 2D `~numpy.ndarray`
        The absolute image value above which to select sources. If
        ``threshold`` is a 2D array, it must have the same shape as the
        input ``data``. If the star finder is run on an image that is a
        `~astropy.units.Quantity` array, then ``threshold`` must have
        the same units.

    fwhm : float
        The full-width half-maximum (FWHM) of the 2D circular Gaussian
        kernel in units of pixels.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) (:math:`\\sigma = \\mbox{FWHM} / (2
        \\sqrt{2 \\log(2)})`).

    minsep_fwhm : float, optional
        The separation (in units of ``fwhm``) for detected objects. The
        minimum separation is calculated as ``int((fwhm * minsep_fwhm) +
        0.5)`` and is clipped to a minimum value of 2. Note that large
        values may result in long run times.

        .. deprecated:: 3.0
            Use ``min_separation`` instead.

    sharplo : float, optional
        The lower bound on sharpness for object detection.

        .. deprecated:: 3.0
            Use ``sharpness_range=(lower, upper)`` instead.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

        .. deprecated:: 3.0
            Use ``sharpness_range=(lower, upper)`` instead.

    roundlo : float, optional
        The lower bound on roundness for object detection.

        .. deprecated:: 3.0
            Use ``roundness_range=(lower, upper)`` instead.

    roundhi : float, optional
        The upper bound on roundness for object detection.

        .. deprecated:: 3.0
            Use ``roundness_range=(lower, upper)`` instead.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders. The default is
        `False`, which is the mode used by starfind.

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

    xycoords : `None` or Nx2 `~numpy.ndarray`, optional
        The (x, y) pixel coordinates of the approximate centroid
        positions of identified sources. If ``xycoords`` are input, the
        algorithm will skip the source-finding step.

    min_separation : `None` or float, optional
        The minimum separation (in pixels) for detected objects. If
        `None` (default) then the minimum separation is calculated as
        ``2.5 * fwhm``. Note that large values may result in long run
        times.

        .. versionchanged:: 3.0
            The default ``min_separation`` changed from ``max(2,
            int(fwhm * 2.5 + 0.5))`` to ``2.5 * fwhm``. To recover the
            previous behavior, set ``min_separation=max(2, int(fwhm *
            2.5 + 0.5))``.

    sharpness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on sharpness for object
        detection. Objects with sharpness outside this range will be
        rejected. If `None`, no sharpness filtering is performed. The
        default is ``(0.5, 2.0)``.

    roundness_range : tuple of 2 floats or `None`, optional
        The ``(lower, upper)`` inclusive bounds on roundness for object
        detection. Objects with roundness outside this range will be
        rejected. If `None`, no roundness filtering is performed. The
        default is ``(0.0, 0.2)``.

    See Also
    --------
    DAOStarFinder

    Notes
    -----
    If the star finder is run on an image that is a
    `~astropy.units.Quantity` array, then ``threshold`` and ``peak_max``
    must have the same units as the image.

    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0. The equivalent parameters in IRAF's starfind are
    ``boundary='constant'`` and ``constant=0.0``.

    IRAF's starfind uses ``hwhmpsf``, ``fradius``, and ``sepmin`` as
    input parameters. The equivalent input values for `IRAFStarFinder`
    are:

    * ``fwhm = hwhmpsf * 2``
    * ``sigma_radius = fradius * sqrt(2.0 * log(2.0))``
    * ``min_separation = max(2, int((fwhm * sepmin) + 0.5))``

    The main differences between `~photutils.detection.DAOStarFinder`
    and `~photutils.detection.IRAFStarFinder` are:

    * `~photutils.detection.IRAFStarFinder` always uses a 2D circular
      Gaussian kernel, while `~photutils.detection.DAOStarFinder` can use
      an elliptical Gaussian kernel.

    * `IRAFStarFinder` internally calculates a "sky" background level
      based on unmasked pixels within the kernel footprint.

    * `~photutils.detection.IRAFStarFinder` calculates the objects'
      centroid, roundness, and sharpness using image moments.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('brightest', 'n_brightest', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('peakmax', 'peak_max', '3.0', until='4.0')
    def __init__(self, threshold, fwhm, sigma_radius=1.5,
                 minsep_fwhm=_DEPR_DEFAULT,
                 sharplo=_DEPR_DEFAULT, sharphi=_DEPR_DEFAULT,
                 roundlo=_DEPR_DEFAULT, roundhi=_DEPR_DEFAULT,
                 exclude_border=False, n_brightest=None, peak_max=None,
                 xycoords=None, min_separation=None, *,
                 sharpness_range=(0.5, 2.0),
                 roundness_range=(0.0, 0.2)):

        # Validate the units, but do not strip them
        inputs = (threshold, peak_max)
        names = ('threshold', 'peak_max')
        check_units(inputs, names)

        if not isscalar(fwhm):
            msg = 'fwhm must be a scalar value'
            raise TypeError(msg)

        sharpness_range = _handle_deprecated_range(
            sharplo, sharphi, sharpness_range,
            'sharp', 'sharpness_range', (0.5, 2.0))
        roundness_range = _handle_deprecated_range(
            roundlo, roundhi, roundness_range,
            'round', 'roundness_range', (0.0, 0.2))

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

        # Handle deprecated minsep_fwhm parameter
        if minsep_fwhm is not _DEPR_DEFAULT:
            msg = ("The 'minsep_fwhm' parameter is deprecated "
                   'and will be removed in a future version. Use '
                   "'min_separation' instead.")
            warnings.warn(msg, AstropyDeprecationWarning)
            if minsep_fwhm < 0:
                msg = 'minsep_fwhm must be >= 0'
                raise ValueError(msg)
            if min_separation is None:
                # Use the deprecated minsep_fwhm calculation to set the
                # min_separation
                min_separation = max(2, int((fwhm * minsep_fwhm) + 0.5))

        self.threshold = threshold
        self.fwhm = fwhm
        self.sigma_radius = sigma_radius
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.peak_max = peak_max

        if xycoords is not None:
            xycoords = np.asarray(xycoords)
            if xycoords.ndim != 2 or xycoords.shape[1] != 2:
                msg = 'xycoords must be shaped as an Nx2 array'
                raise ValueError(msg)
        self.xycoords = xycoords

        self.kernel = _StarFinderKernel(self.fwhm, ratio=1.0, theta=0.0,
                                        sigma_radius=self.sigma_radius)

        if min_separation is not None:
            if min_separation < 0:
                msg = 'min_separation must be >= 0'
                raise ValueError(msg)
            self.min_separation = min_separation
        else:
            self.min_separation = 2.5 * self.fwhm

    def _repr_str_params(self):
        params = ('threshold', 'fwhm', 'sigma_radius',
                  'sharpness_range', 'roundness_range',
                  'exclude_border', 'n_brightest', 'peak_max', 'xycoords',
                  'min_separation')
        overrides = {}
        if not isscalar(self.threshold):
            overrides['threshold'] = (
                f'<array; shape={np.shape(self.threshold)}>')
        if self.xycoords is not None:
            overrides['xycoords'] = f'<array; shape={self.xycoords.shape}>'
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
        cat : `_IRAFStarFinderCatalog` or `None`
            A catalog of sources found in the input data. `None` is
            returned if no sources are found.
        """
        convolved_data = _filter_data(data, self.kernel.data, mode='constant',
                                      fill_value=0.0,
                                      check_normalization=False)

        if self.xycoords is None:
            xypos = self._find_stars(convolved_data, self.kernel,
                                     self.threshold,
                                     min_separation=self.min_separation,
                                     mask=mask,
                                     exclude_border=self.exclude_border)
        else:
            xypos = self.xycoords

        if xypos is None:
            msg = 'No sources were found.'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        return _IRAFStarFinderCatalog(data,
                                      convolved_data,
                                      xypos,
                                      self.kernel,
                                      sharpness_range=self.sharpness_range,
                                      roundness_range=self.roundness_range,
                                      n_brightest=self.n_brightest,
                                      peak_max=self.peak_max)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
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
            * ``fwhm``: object FWHM.
            * ``sharpness``: object sharpness.
            * ``roundness``: object roundness.
            * ``orientation``: the angle between the ``x`` axis and the
              major axis source measured counter-clockwise in the range
              [0, 360) degrees.
            * ``n_pixels``: the total number of (positive) unmasked
              pixels.
            * ``peak``: the peak, sky-subtracted, pixel value of the object.
            * ``flux``: the object instrumental flux calculated as the
              sum of sky-subtracted data values within the kernel
              footprint.
            * ``mag``: the object instrumental magnitude calculated as
              ``-2.5 * log10(flux)``.
        """
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


class _IRAFStarFinderCatalog(StarFinderCatalogBase):
    """
    Class to create a catalog of the properties of each detected star,
    as defined by IRAF's ``starfind`` task.

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

    kernel : `_StarFinderKernel`
        The convolution kernel. This kernel must match the kernel used
        to create the ``convolved_data``.

    sharpness_range : tuple of 2 floats, optional
        The ``(lower, upper)`` inclusive bounds on sharpness for object
        detection. Objects with sharpness outside this range will be
        rejected.

    roundness_range : tuple of 2 floats, optional
        The ``(lower, upper)`` inclusive bounds on roundness for object
        detection. Objects with roundness outside this range will be
        rejected.

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
    """

    def __init__(self, data, convolved_data, xypos, kernel, *,
                 sharpness_range=(0.2, 1.0), roundness_range=(-1.0, 1.0),
                 n_brightest=None, peak_max=None):

        # Validate the units, but do not strip them
        inputs = (data, convolved_data, peak_max)
        names = ('data', 'convolved_data', 'peak_max')
        check_units(inputs, names)

        super().__init__(data, xypos, kernel,
                         n_brightest=n_brightest,
                         peak_max=peak_max)

        self.convolved_data = convolved_data
        self.sharpness_range = sharpness_range
        self.roundness_range = roundness_range

        self.default_columns = ('id', 'x_centroid', 'y_centroid', 'fwhm',
                                'sharpness', 'roundness', 'orientation',
                                'n_pixels', 'peak', 'flux', 'mag')

    def _get_init_attributes(self):
        """
        Return a tuple of attribute names to copy during slicing.
        """
        return ('data', 'unit', 'convolved_data', 'kernel',
                'sharpness_range', 'roundness_range', 'n_brightest',
                'peak_max', 'cutout_shape', 'default_columns')

    @lazyproperty
    def sky(self):
        """
        Calculate the sky background level.

        The local sky level is roughly estimated using the IRAF starfind
        calculation as the average value in the non-masked regions
        within the kernel footprint.
        """
        skymask = ~self.kernel.mask.astype(bool)  # True=sky, False=obj
        # nsky is always > 0 because the kernel mask never covers the
        # entire footprint (the Gaussian kernel is always truncated
        # within the array, leaving unmasked border pixels).
        nsky = np.count_nonzero(skymask)
        axis = (1, 2)
        sky = np.sum(self.cutout_data_nosub * skymask, axis=axis) / nsky

        if self.unit is not None:
            sky <<= self.unit

        return sky

    @lazyproperty
    def cutout_data_nosub(self):
        """
        The cutout data without sky subtraction or masking.
        """
        return self.make_cutouts(self.data)

    @lazyproperty
    def cutout_data(self):
        """
        The cutout data with sky subtraction and masking applied.
        """
        # This is a freshly computed array, so in-place modification is
        # safe.
        data = ((self.cutout_data_nosub - self.sky[:, np.newaxis, np.newaxis])
                * self.kernel.mask)
        # IRAF starfind discards negative pixels
        data[data < 0] = 0.0
        return data

    @lazyproperty
    def n_pixels(self):
        """
        The total number of (positive) unmasked pixels in the cutout
        data.
        """
        return np.count_nonzero(self.cutout_data, axis=(1, 2))

    @lazyproperty
    def cutout_xorigin(self):
        """
        The x pixel coordinate of the cutout origin.
        """
        return np.transpose(self.xypos)[0] - self.kernel.x_radius

    @lazyproperty
    def cutout_yorigin(self):
        """
        The y pixel coordinate of the cutout origin.
        """
        return np.transpose(self.xypos)[1] - self.kernel.y_radius

    @lazyproperty
    def x_centroid(self):
        """
        The x pixel coordinate of the object centroid.
        """
        return self.cutout_x_centroid + self.cutout_xorigin

    @lazyproperty
    def y_centroid(self):
        """
        The y pixel coordinate of the object centroid.
        """
        return self.cutout_y_centroid + self.cutout_yorigin

    @lazyproperty
    def sharpness(self):
        """
        The sharpness of the object.
        """
        return self.fwhm / self.kernel.fwhm

    def apply_filters(self):
        """
        Filter the catalog.
        """
        attrs = ('x_centroid', 'y_centroid', 'sharpness', 'roundness',
                 'orientation', 'sky', 'peak', 'flux')
        initial_mask = np.count_nonzero(self.cutout_data, axis=(1, 2)) > 1
        newcat = self._filter_finite(attrs, initial_mask=initial_mask)
        if newcat is None:
            return None

        bounds = [
            ('sharpness', self.sharpness_range),
            ('roundness', self.roundness_range),
        ]
        return newcat._filter_bounds(bounds)
