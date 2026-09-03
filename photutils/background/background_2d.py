# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for estimating the 2D background and background RMS in an image.
"""

import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import NamedTuple

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, block_replicate, reshape_as_blocks
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import generic_filter

from photutils.aperture import RectangularAperture
from photutils.aperture._batch_stats import batch_sigma_clip_stats
from photutils.background.core import (SIGMA_CLIP, BiweightLocationBackground,
                                       BiweightScaleBackgroundRMS,
                                       MADStdBackgroundRMS, MeanBackground,
                                       MedianBackground, MMMBackground,
                                       ModeEstimatorBackground,
                                       SExtractorBackground, StdBackgroundRMS)
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                _BkgZoomInterpolator)
from photutils.utils import ShepardIDWInterpolator
from photutils.utils._deprecation import (deprecated,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair, create_default_sigmaclip
from photutils.utils._repr import make_repr
from photutils.utils._stats import nanmedian, nanmin

__all__ = ['Background2D']

__doctest_skip__ = ['Background2D']


class _BoxStatsSpec(NamedTuple):
    """
    Parameters for the fused sigma-clipped box-statistics kernel.

    See ``Background2D._make_box_stats_spec``.
    """

    sigma_lower: float
    sigma_upper: float
    maxiters: int
    cenfunc_code: int
    stdfunc_code: int
    do_clip: bool
    bkg_kind: str
    median_factor: float
    mean_factor: float
    biloc_c: float
    biloc_m: float
    rms_kind: str
    biscale_c: float
    biscale_m: float


class Background2D:
    """
    Class to estimate a 2D background and background RMS noise in an
    image.

    The background is estimated using (sigma-clipped) statistics in
    each box of a grid that covers the input ``data`` to create a
    low-resolution, and possibly irregularly-gridded, background map.

    The final background map is calculated by interpolating the
    low-resolution background map.

    Invalid data values (i.e., NaN or inf) are automatically masked.

    .. note::

        Better performance will generally be obtained if you have the
        `Bottleneck <https://github.com/pydata/bottleneck>`_ package
        installed. See :ref:`performance-tips` for details, including
        notes on array byte order (endianness) when loading FITS data.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.nddata.NDData`
        The 2D array from which to estimate the background and/or
        background RMS map.

    box_size : int or array_like (int)
        The box size along each axis. If ``box_size`` is a scalar then
        a square box of size ``box_size`` will be used. If ``box_size``
        has two elements, they must be in ``(ny, nx)`` order. For best
        results, the box shape should be chosen such that the ``data``
        are covered by an integer number of boxes in both dimensions.
        When this is not the case, the image will be padded along the
        top and/or right edges. Ideally, the ``box_size`` should be
        chosen such that an integer number of boxes is only slightly
        larger than the ``data`` size to minimize the amount of padding.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from the background and background
        RMS calculations. ``mask`` is intended to mask sources or bad
        pixels, but a background and background RMS value will be
        calculated for them based on interpolation of the low-resolution
        background and background RMS maps. Use ``coverage_mask`` to
        mask blank areas of an image. ``coverage_mask`` pixels are
        assigned a value of ``fill_value`` (default = 0) in the output
        background and background RMS maps.

    coverage_mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        ``coverage_mask`` should be `True` where there is no coverage
        (i.e., no data) for a given pixel (e.g., blank areas in a mosaic
        image). It should not be used to mask sources or bad pixels (in
        that case use ``mask`` instead). ``coverage_mask`` pixels are
        assigned a value of ``fill_value`` (default = 0) in the output
        background and background RMS maps.

    fill_value : float, optional
        The value used to fill the output background and background RMS
        maps where the input ``coverage_mask`` is `True`.

    exclude_percentile : float in the range of [0, 100], optional
        The percentage of masked pixels allowed in a box for it to be
        included in the low-resolution map. If a box has more than
        ``exclude_percentile`` percent of its pixels masked then it
        will be excluded from the low-resolution map. Masked pixels
        include those from the input ``mask`` and ``coverage_mask``,
        non-finite ``data`` values, any padded area at the data
        edges, and those resulting from any sigma clipping. Setting
        ``exclude_percentile=0`` will exclude boxes that have any
        masked pixels. Note that completely masked boxes are
        always excluded. In general, ``exclude_percentile`` should be
        kept as low as possible to ensure there are a sufficient number
        of unmasked pixels in each box for reasonable statistical
        estimates. The default is 10.0.

    filter_size : int or array_like (int), optional
        The window size of the 2D median filter to apply to the
        low-resolution background map. If ``filter_size`` is a scalar
        then a square box of size ``filter_size`` will be used. If
        ``filter_size`` has two elements, they must be in ``(ny, nx)``
        order. ``filter_size`` must be odd along both axes. A filter
        size of ``1`` (or ``(1, 1)``) means no filtering.

    filter_threshold : float, optional
        The threshold value used for selective median filtering of
        the low-resolution 2D background map. The median filter will
        be applied to only the background boxes with values larger
        than ``filter_threshold``. Set to `None` to filter all boxes
        (default).

    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will
        be performed.

    bkg_estimator : callable, optional
        A callable object (a function or e.g., an instance of
        any `~photutils.background.BackgroundBase` subclass)
        used to estimate the background in each of the boxes.
        The callable object must take in a 2D `~numpy.ndarray`
        or `~numpy.ma.MaskedArray` and have an ``axis`` keyword.
        Internally, the background will be calculated along ``axis=1``
        and in this case the callable object must return a 1D
        `~numpy.ndarray`, where np.nan values are used for masked
        pixels. If ``bkg_estimator`` includes sigma clipping, it
        will be ignored (use the ``sigma_clip`` keyword here to
        define sigma clipping). The default is an instance of
        `~photutils.background.SExtractorBackground`.

    bkg_rms_estimator : callable, optional
        A callable object (a function or e.g., an instance of
        any `~photutils.background.BackgroundRMSBase` subclass)
        used to estimate the background RMS in each of the boxes.
        The callable object must take in a 2D `~numpy.ndarray`
        or `~numpy.ma.MaskedArray` and have an ``axis`` keyword.
        Internally, the background RMS will be calculated along
        ``axis=1`` and in this case the callable object must return a
        1D `~numpy.ndarray`, where np.nan values are used for masked
        pixels. If ``bkg_rms_estimator`` includes sigma clipping,
        it will be ignored (use the ``sigma_clip`` keyword here
        to define sigma clipping). The default is an instance of
        `~photutils.background.StdBackgroundRMS`.

    n_threads : int, optional
        The number of threads to use to compute the box statistics
        and to resize the low-resolution meshes to the full-size
        maps returned by the ``background`` and ``background_rms``
        properties. The default is 1 (no multithreading). When
        ``n_threads`` > 1, the work is divided into chunks along the
        y axis and processed concurrently. The box statistics are
        identical to the single-threaded computation and the resized
        maps are identical up to floating-point rounding. The underlying
        kernels release the Python global interpreter lock (GIL),
        so multithreading can significantly speed up the background
        estimation for large images. Resizing the meshes concurrently
        requires scipy 1.16 or later. With older versions the resizing
        step is single-threaded. If a custom ``bkg_estimator`` or
        ``bkg_rms_estimator`` callable is input, it must be thread-safe.

    interpolator : callable, optional
        A callable object (a function or object) used to interpolate
        the low-resolution background or background RMS image to the
        full-size background or background RMS maps. The default
        is an instance of `BkgZoomInterpolator`, which uses the
        `scipy.ndimage.zoom` function.

        .. deprecated:: 3.0
           This keyword argument is deprecated and will be removed in
           a future version. When removed, the `scipy.ndimage.zoom`
           cubic spline interpolator will always be used to resize the
           low-resolution background and background RMS arrays to the
           full image size.

    Notes
    -----
    Integer input data produce background and background RMS outputs
    with ``np.float32`` dtype to preserve precision from interpolation
    while minimizing memory usage. Float input data produce background
    and background RMS outputs with the same dtype as the input data.

    When the ``sigma_clip`` input is `None` or uses the string
    ``cenfunc`` and ``stdfunc`` options, and the ``bkg_estimator``
    and ``bkg_rms_estimator`` inputs are standard photutils estimator
    classes, the sigma clipping and box statistics are computed
    together in compiled code. Otherwise, a generic path that calls
    ``sigma_clip`` and the estimators is used. Both paths produce the
    same results.

    If there is only one background box element (i.e., ``box_size`` is
    the same size as (or larger than) the ``data``), then the background
    map will simply be a constant image.
    """

    @deprecated_renamed_argument('bkgrms_estimator', 'bkg_rms_estimator',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('interpolator', None, '3.0', until='4.0')
    def __init__(self, data, box_size, *, mask=None, coverage_mask=None,
                 fill_value=0.0, exclude_percentile=10.0, filter_size=(3, 3),
                 filter_threshold=None, sigma_clip=SIGMA_CLIP,
                 bkg_estimator=None, bkg_rms_estimator=None, n_threads=1,
                 interpolator=None):

        if isinstance(data, u.Quantity):
            self._unit = data.unit
            data = data.value
        elif isinstance(data, NDData):  # includes CCDData
            self._unit = data.unit
            data = data.data
        else:
            self._unit = None

        # For masked-array input, extract the data values. The data mask
        # is combined with the input mask below.
        data_mask = None
        if isinstance(data, np.ma.MaskedArray):
            if data.mask is not np.ma.nomask:
                data_mask = np.asarray(data.mask, dtype=bool)
            data = data.data

        # self._data is a temporary instance variable to store the input
        # data (the variable is deleted in self._calculate_stats)
        self._data = self._validate_array(data, 'data', shape=False)
        self._data_dtype = self._data.dtype
        self._data_shape = self._data.shape
        nonfinite_mask = ~np.isfinite(self._data)
        if np.all(nonfinite_mask):
            msg = ('Input data contains all non-finite (NaN or infinity) '
                   'values. Cannot compute a background.')
            raise ValueError(msg)

        # self._mask is a temporary instance variable to store the input
        # mask array (deleted in self._calculate_stats). self._has_mask
        # records whether a mask was provided for use after deletion.
        self._mask = self._validate_array(mask, 'mask')
        if data_mask is not None:
            if self._mask is None:
                self._mask = data_mask
            else:
                self._mask = np.logical_or(self._mask, data_mask)
        self._has_mask = self._mask is not None
        self.coverage_mask = self._validate_array(coverage_mask,
                                                  'coverage_mask')

        # box_size cannot be larger than the data array size
        self.box_size = as_pair('box_size', box_size, lower_bound=(0, 0),
                                upper_bound=self._data_shape)

        self.fill_value = fill_value
        if exclude_percentile < 0 or exclude_percentile > 100:
            msg = 'exclude_percentile must be between 0 and 100 (inclusive)'
            raise ValueError(msg)
        self.exclude_percentile = exclude_percentile
        self.filter_size = as_pair('filter_size', filter_size,
                                   lower_bound=(0, 0), check_odd=True)
        self.filter_threshold = filter_threshold

        if (isinstance(n_threads, bool)
                or not isinstance(n_threads, (int, np.integer))
                or n_threads < 1):
            msg = 'n_threads must be a positive integer'
            raise ValueError(msg)
        self.n_threads = int(n_threads)

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        self.sigma_clip = sigma_clip

        if interpolator is None:
            interpolator = _BkgZoomInterpolator()
        self.interpolator = interpolator

        if bkg_estimator is None:
            bkg_estimator = SExtractorBackground(sigma_clip=None)
        if bkg_rms_estimator is None:
            bkg_rms_estimator = StdBackgroundRMS(sigma_clip=None)

        # We perform sigma clipping as a separate step to avoid calling
        # it twice for the background and background RMS. Shallow-copy
        # the estimators before mutating their sigma_clip attribute so
        # that any user-supplied estimator object is not modified in
        # place.
        bkg_estimator = copy.copy(bkg_estimator)
        bkg_rms_estimator = copy.copy(bkg_rms_estimator)
        if hasattr(bkg_estimator, 'sigma_clip'):
            bkg_estimator.sigma_clip = None
        if hasattr(bkg_rms_estimator, 'sigma_clip'):
            bkg_rms_estimator.sigma_clip = None
        self.bkg_estimator = bkg_estimator
        self.bkg_rms_estimator = bkg_rms_estimator

        self._box_stats_spec = self._make_box_stats_spec()

        self._box_n_pixels = None

        # Store the interpolator keyword arguments for later use
        # (before self._data is deleted in self._calculate_stats)
        interp_dtype = self._data.dtype
        if interp_dtype.kind != 'f':
            interp_dtype = np.float32
        self._interp_kwargs = {'shape': self._data.shape,
                               'dtype': interp_dtype,
                               'box_size': self.box_size,
                               'n_threads': self.n_threads}

        # Perform the initial calculations to avoid storing large data
        # arrays and to keep the memory usage minimal
        bkg_stats, bkgrms_stats, self._n_good = self._calculate_stats(
            nonfinite_mask)

        # This is used to selectively filter the low-resolution maps
        self._min_bkg_stats = nanmin(bkg_stats)

        # Store a mask of the excluded mesh values (NaNs) in the
        # low-resolution maps
        self._mesh_nan_mask = np.isnan(bkg_stats)

        # Add keyword arguments needed for BkgZoomInterpolator.
        # BkgIDWInterpolator upscales the mesh based only on the good
        # pixels in the low-resolution mesh.
        if isinstance(self.interpolator, BkgIDWInterpolator):
            self._interp_kwargs['mesh_yxcen'] = self._calculate_mesh_yxcen()
            self._interp_kwargs['mesh_nan_mask'] = self._mesh_nan_mask

        # Compute the low-resolution background and background RMS
        # meshes. The mesh arrays are small, so this is inexpensive
        # compared to the box statistics above, and computing them
        # here keeps the instance immutable after construction. The
        # unfiltered background mesh defines the pixels selected by
        # filter_threshold for both meshes.
        bkg_mesh = self._interpolate_grid(bkg_stats)
        bkgrms_mesh = self._interpolate_grid(bkgrms_stats)
        bkg_mesh_filt = self._filter_grid(bkg_mesh, bkg_mesh)
        bkgrms_mesh_filt = self._filter_grid(bkgrms_mesh, bkg_mesh)
        self._background_mesh = self._apply_units(bkg_mesh_filt)
        self._background_rms_mesh = self._apply_units(bkgrms_mesh_filt)

    def _repr_str_params(self):
        params = ('data', 'box_size', 'mask', 'coverage_mask', 'fill_value',
                  'exclude_percentile', 'filter_size', 'filter_threshold',
                  'sigma_clip', 'bkg_estimator', 'bkg_rms_estimator',
                  'n_threads', 'interpolator')

        data_repr = f'<array; shape={self._interp_kwargs["shape"]}>'

        mask_repr = None if not self._has_mask else data_repr

        coverage_mask_repr = (None if self.coverage_mask is None
                              else data_repr)

        overrides = {'data': data_repr, 'mask': mask_repr,
                     'coverage_mask': coverage_mask_repr}

        return params, overrides

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def _validate_array(self, array, name, *, shape=True):
        """
        Validate the input data, mask, and coverage_mask arrays.
        """
        if name in ('mask', 'coverage_mask') and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if (name in ('mask', 'coverage_mask')
                    and array.dtype != bool):
                msg = f'{name} must be a boolean array'
                raise TypeError(msg)
            if array.ndim != 2:
                msg = f'{name} must be a 2D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    def _apply_units(self, data):
        """
        Apply units to the data.

        The units are based on the units of the input ``data`` array.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The input data array.

        Returns
        -------
        data : `~numpy.ndarray`
            The data array with units applied.
        """
        if self._unit is not None:
            data <<= self._unit
        return data

    def _combine_input_masks(self):
        """
        Combine the input mask and coverage_mask.
        """
        if self._mask is None and self.coverage_mask is None:
            return None
        if self._mask is None:
            return self.coverage_mask
        if self.coverage_mask is None:
            return self._mask

        return np.logical_or(self._mask, self.coverage_mask)

    def _combine_all_masks(self, mask):
        """
        Combine the input masks (mask and coverage_mask) with the mask
        of invalid data values.
        """
        input_mask = self._combine_input_masks()

        msg = ('Input data contains non-finite (NaN or infinity) values, '
               'which were automatically masked.')

        if input_mask is None:
            if np.any(mask):
                warnings.warn(msg, AstropyUserWarning)

            total_mask = mask
        else:
            condition = np.logical_and(np.logical_not(input_mask), mask)
            if np.any(condition):
                warnings.warn(msg, AstropyUserWarning)

            total_mask = np.logical_or(input_mask, mask)

        if np.all(total_mask):
            msg = 'All input pixels are masked. Cannot compute a background.'
            raise ValueError(msg)

        return total_mask

    @cached_property
    def _good_n_pixels_threshold(self):
        """
        The minimum number of required unmasked pixels in a box used for
        it to be included in the low-resolution map.

        A box is excluded if it contains fewer unmasked pixels than this
        threshold. For exclude_percentile=0, only boxes with zero masked
        pixels will be included. For exclude_percentile=100, all boxes
        will be included unless they are completely masked.

        Boxes that are completely masked are always excluded.
        """
        return (1 - (self.exclude_percentile / 100.0)) * self._box_n_pixels

    def _make_box_stats_spec(self):
        """
        The fused box-statistics kernel parameters, or `None`.

        Returns a `_BoxStatsSpec` when the ``sigma_clip``,
        ``bkg_estimator``, and ``bkg_rms_estimator`` inputs are all
        supported by the fused sigma-clipping and statistics kernel,
        and `None` otherwise (in which case the generic path based
        on `astropy.stats.SigmaClip` and the estimator callables is
        used). The fast path supports the string ``cenfunc`` values
        'median'/'mean'/'biweight', the string ``stdfunc`` values
        'std'/'mad_std'/'biweight', no spatial growing, and the standard
        photutils estimator classes. For the biweight-based estimators,
        the ``M`` location anchor must be `None` or a finite scalar.
        """
        sigma_clip = self.sigma_clip
        if sigma_clip is None:
            do_clip = False
            sigma_lower = sigma_upper = 0.0
            maxiters = 0
            cenfunc_code = stdfunc_code = 0
        else:
            # Spatial growing (``grow``) is not supported
            if (not isinstance(sigma_clip, SigmaClip)
                    or not (isinstance(sigma_clip.cenfunc, str)
                            and sigma_clip.cenfunc in ('median', 'mean',
                                                       'biweight'))
                    or not (isinstance(sigma_clip.stdfunc, str)
                            and sigma_clip.stdfunc in ('std', 'mad_std',
                                                       'biweight'))
                    or sigma_clip.grow):
                return None
            do_clip = True
            sigma_lower = float(sigma_clip.sigma_lower)
            sigma_upper = float(sigma_clip.sigma_upper)
            maxiters = sigma_clip.maxiters
            if maxiters is None or np.isinf(maxiters):
                maxiters = -1
            else:
                maxiters = int(maxiters)
            cenfunc_code = {'median': 0, 'mean': 1,
                            'biweight': 2}[sigma_clip.cenfunc]
            stdfunc_code = {'std': 0, 'mad_std': 1,
                            'biweight': 2}[sigma_clip.stdfunc]

        bkg_estimator = self.bkg_estimator
        median_factor = mean_factor = 0.0
        biloc_c = biloc_m = np.nan
        if type(bkg_estimator) is MeanBackground:
            bkg_kind = 'mean'
        elif type(bkg_estimator) is MedianBackground:
            bkg_kind = 'median'
        elif type(bkg_estimator) in (ModeEstimatorBackground,
                                     MMMBackground):
            bkg_kind = 'mode'
            median_factor = float(bkg_estimator.median_factor)
            mean_factor = float(bkg_estimator.mean_factor)
        elif type(bkg_estimator) is SExtractorBackground:
            bkg_kind = 'sextractor'
        elif type(bkg_estimator) is BiweightLocationBackground:
            m_value = bkg_estimator.M
            if m_value is not None and (np.ndim(m_value) != 0
                                        or not np.isfinite(m_value)):
                return None
            bkg_kind = 'biweight_location'
            biloc_c = float(bkg_estimator.c)
            biloc_m = np.nan if m_value is None else float(m_value)
        else:
            return None

        rms_estimator = self.bkg_rms_estimator
        biscale_c = biscale_m = np.nan
        if type(rms_estimator) is StdBackgroundRMS:
            rms_kind = 'std'
        elif type(rms_estimator) is MADStdBackgroundRMS:
            rms_kind = 'mad_std'
        elif type(rms_estimator) is BiweightScaleBackgroundRMS:
            m_value = rms_estimator.M
            if m_value is not None and (np.ndim(m_value) != 0
                                        or not np.isfinite(m_value)):
                return None
            rms_kind = 'biweight_scale'
            biscale_c = float(rms_estimator.c)
            biscale_m = np.nan if m_value is None else float(m_value)
        else:
            return None

        return _BoxStatsSpec(sigma_lower=sigma_lower,
                             sigma_upper=sigma_upper, maxiters=maxiters,
                             cenfunc_code=cenfunc_code,
                             stdfunc_code=stdfunc_code, do_clip=do_clip,
                             bkg_kind=bkg_kind,
                             median_factor=median_factor,
                             mean_factor=mean_factor, biloc_c=biloc_c,
                             biloc_m=biloc_m, rms_kind=rms_kind,
                             biscale_c=biscale_c, biscale_m=biscale_m)

    def _fast_box_statistics(self, data, *, axis=None):
        """
        Compute the background, background RMS, and surviving pixel
        count in each box using the fused sigma-clipping and statistics
        kernel.

        The kernel operates on the finite values of each box in a
        single pass, without generating a sigma-clipped copy of the box
        data, and produces the same results as sequentially applying
        `astropy.stats.SigmaClip` and the estimator classes. The kernel
        releases the GIL, so this method can run in parallel threads.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The array of box data, where NaN values mark masked pixels.

        axis : int or `None`, optional
            The axis along which to compute the statistics. Must be the
            last axis (``-1``). If `None`, the entire array is treated
            as a single box and scalar values are returned.

        Returns
        -------
        bkg : `~numpy.ndarray` or float
            The background statistics in each box.

        bkgrms : `~numpy.ndarray` or float
            The background RMS statistics in each box.

        n_good : `~numpy.ndarray` or int
            The number of surviving (unmasked and unclipped) pixels in
            each box.
        """
        spec = self._box_stats_spec

        if axis is None:
            data2d = data.reshape(1, -1)
        else:
            data2d = data.reshape(-1, data.shape[-1])

        # Sort each box (NaN values are placed last). The kernel
        # requires ascending-sorted rows.
        data2d = np.sort(data2d.astype(np.float64, copy=False), axis=-1)

        (mean, median, std, madstd, biloc,
         biscale, n_good) = batch_sigma_clip_stats(
            data2d, spec.sigma_lower, spec.sigma_upper, spec.maxiters,
            spec.cenfunc_code, spec.stdfunc_code, spec.do_clip,
            spec.rms_kind == 'mad_std',
            spec.bkg_kind == 'biweight_location', spec.biloc_c,
            spec.biloc_m, spec.rms_kind == 'biweight_scale',
            spec.biscale_c, spec.biscale_m)

        if spec.bkg_kind == 'mean':
            bkg = mean
        elif spec.bkg_kind == 'median':
            bkg = median
        elif spec.bkg_kind == 'mode':
            bkg = ((spec.median_factor * median)
                   - (spec.mean_factor * mean))
        elif spec.bkg_kind == 'biweight_location':
            bkg = biloc
        else:  # 'sextractor'
            bkg = (2.5 * median) - (1.5 * mean)

            # Set the background to the mean where the std is zero
            mean_mask = std == 0
            bkg[mean_mask] = mean[mean_mask]

            # Set the background to the median when the absolute
            # difference between the mean and median divided by the
            # standard deviation is greater than or equal to 0.3.
            with np.errstate(divide='ignore', invalid='ignore'):
                med_mask = (np.abs(mean - median) / std) >= 0.3
            mask = np.logical_and(med_mask, np.logical_not(mean_mask))
            bkg[mask] = median[mask]

        if spec.rms_kind == 'std':
            bkgrms = std
        elif spec.rms_kind == 'mad_std':
            bkgrms = madstd
        else:  # 'biweight_scale'
            bkgrms = biscale

        # Preserve the (float) dtype of the input box data
        if data.dtype != np.float64:
            bkg = bkg.astype(data.dtype)
            bkgrms = bkgrms.astype(data.dtype)

        if axis is None:
            return bkg[0], bkgrms[0], n_good[0]

        out_shape = data.shape[:-1]
        return (bkg.reshape(out_shape), bkgrms.reshape(out_shape),
                n_good.reshape(out_shape))

    def _sigmaclip_boxes(self, data, *, axis, sigma_clip=None):
        """
        Sigma clip the boxes along the specified axis.

        This method sigma clips the boxes along the specified axis and
        returns the sigma-clipped data. The input ``data`` is typically
        a 4D array where the first two dimensions represent the y and x
        positions of the boxes and the last two dimensions represent the
        y and x positions within each box.

        We perform sigma clipping as a separate step to avoid performing
        sigma clipping for both the background and background RMS
        estimators.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 4D array of box data.

        axis : int or tuple of int
            The axis or axes along which to sigma clip the data.

        sigma_clip : `~astropy.stats.SigmaClip` or `None`
            The sigma-clipping instance to apply. If `None`, no sigma
            clipping is performed. `~astropy.stats.SigmaClip` stores
            internal state on the instance during calls, so each thread
            must use its own instance.

        Returns
        -------
        data : `~numpy.ndarray`
            The sigma-clipped data.
        """
        if sigma_clip is not None:
            data = sigma_clip(data, axis=axis, masked=False, copy=False)
        return data

    def _compute_box_statistics(self, data, *, sigma_clip=None, axis=None):
        """
        Compute the background and background RMS statistics in each
        box.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 4D array of box data.

        sigma_clip : `~astropy.stats.SigmaClip` or `None`
            The sigma-clipping instance to apply to the box data. If
            `None`, no sigma clipping is performed.

        axis : int or tuple of int, optional
            The axis or axes along which to compute the statistics.

        Returns
        -------
        bkg : 2D `~numpy.ndarray` or float
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray` or float
            The background RMS statistics in each box.
        """
        if self._box_stats_spec is not None:
            bkg, bkgrms, n_good = self._fast_box_statistics(data,
                                                            axis=axis)
        else:
            data = self._sigmaclip_boxes(data, axis=axis,
                                         sigma_clip=sigma_clip)

            # Make 2D arrays of the box statistics
            bkg = self.bkg_estimator(data, axis=axis)
            bkgrms = self.bkg_rms_estimator(data, axis=axis)
            n_good = np.count_nonzero(~np.isnan(data), axis=axis)

        # Mask boxes with too few unmasked pixels. Completely masked
        # boxes are always excluded.
        box_mask = ((n_good < self._good_n_pixels_threshold)
                    | (n_good == 0))

        if np.ndim(bkg) == 0:
            if box_mask:  # single corner box
                # np.nan is float64, so use np.float32 to prevent numpy from
                # promoting the output data dtype to float64 if the
                # input data is float32
                bkg = np.float32(np.nan)
                bkgrms = np.float32(np.nan)
        else:
            bkg[box_mask] = np.nan
            bkgrms[box_mask] = np.nan

        return bkg, bkgrms, n_good

    def _parallel_box_statistics(self, data):
        """
        Compute the statistics for the core boxes, optionally using
        multiple threads.

        When ``n_threads`` > 1, the boxes are divided into chunks along
        the first (box row) axis and processed concurrently. The box
        statistics are independent between boxes, so the results are
        identical to the single-threaded computation. The underlying
        sigma-clipping and statistics kernels release the GIL, allowing
        the threads to run in parallel.

        Parameters
        ----------
        data : 3D `~numpy.ndarray`
            The array of box data, where the first two axes are the box
            grid positions and the last axis contains the (flattened)
            pixels within each box.

        Returns
        -------
        bkg : 2D `~numpy.ndarray`
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray`
            The background RMS statistics in each box.

        n_good : 2D `~numpy.ndarray`
            The number of unmasked pixels in each box.
        """
        n_chunks = min(self.n_threads, data.shape[0])
        if n_chunks == 1:
            return self._compute_box_statistics(
                data, axis=-1, sigma_clip=self.sigma_clip)

        def worker(chunk):
            # SigmaClip stores internal state on the instance during
            # calls, so each thread needs its own copy
            sigma_clip = (None if self.sigma_clip is None
                          else copy.copy(self.sigma_clip))
            return self._compute_box_statistics(chunk, axis=-1,
                                                sigma_clip=sigma_clip)

        chunks = np.array_split(data, n_chunks, axis=0)
        with ThreadPoolExecutor(max_workers=n_chunks) as executor:
            results = list(executor.map(worker, chunks))

        bkg = np.vstack([result[0] for result in results])
        bkgrms = np.vstack([result[1] for result in results])
        n_good = np.vstack([result[2] for result in results])
        return bkg, bkgrms, n_good

    def _calculate_stats(self, nonfinite_mask):
        """
        Calculate the background and background RMS statistics in each
        box.

        Parameters
        ----------
        nonfinite_mask : 2D bool `~numpy.ndarray`
            A mask of the non-finite (NaN or infinity) values in the
            input data.

        Returns
        -------
        bkg : 2D `~numpy.ndarray`
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray`
            The background RMS statistics in each box.

        n_good : 2D `~numpy.ndarray`
            The number of unmasked pixels in each box.
        """
        # If needed, copy the data to a float32 array to insert NaNs
        if self._data.dtype.kind != 'f':
            self._data = self._data.astype(np.float32)

        # Automatically mask non-finite values that aren't already
        # masked and combine all masks
        mask = self._combine_all_masks(nonfinite_mask)

        self._box_n_pixels = np.prod(self.box_size)
        n_boxes = self._data.shape // self.box_size
        y1, x1 = n_boxes * self.box_size

        # Suppress the AstropyUserWarning issued by sigma clipping
        # for the NaN values inserted for masked pixels. The warning
        # filter is set once here, in the main thread, because
        # warnings.catch_warnings manipulates global state and is not
        # thread-safe.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)

            # Core boxes - the part of the data array that is an
            # integer multiple of the box size.
            # Combine the last two axes for performance.
            # Below we transform both the data and mask arrays to avoid
            # making multiple copies of the data (one to insert NaN and
            # another for the reshape). Only one copy of the data and
            # mask array is made (except for the extra corner). The
            # boolean mask copy is much smaller than the data array.
            # An explicit copy of the data array is needed to avoid
            # modifying the original data array if the shape of the
            # data array is (y1, x1) (i.e., box_size = data.shape).
            core = reshape_as_blocks(self._data[:y1, :x1].copy(),
                                     self.box_size)
            core_mask = reshape_as_blocks(mask[:y1, :x1], self.box_size)
            core = core.reshape((*n_boxes, -1))
            core_mask = core_mask.reshape((*n_boxes, -1))
            core[core_mask] = np.nan
            bkg, bkgrms, n_good = self._parallel_box_statistics(core)

            extra_row = y1 < self._data.shape[0]
            extra_col = x1 < self._data.shape[1]
            if extra_row:
                # Extra row of boxes.
                # Here we need to make a copy of the data array to
                # avoid modifying the original data array.
                # Move the axes and combine the last two for
                # performance.
                row_data = self._data[y1:, :x1].copy()
                row_mask = mask[y1:, :x1]
                row_data[row_mask] = np.nan
                row_data = reshape_as_blocks(row_data,
                                             (1, self.box_size[1]))
                row_data = np.moveaxis(row_data, 0, -1)
                row_data = row_data.reshape((*row_data.shape[:-2], -1))
                row_stats = self._compute_box_statistics(
                    row_data, axis=-1, sigma_clip=self.sigma_clip)
                row_bkg, row_bkgrms, row_n_good = row_stats

            if extra_col:
                # Extra column of boxes.
                # Here we need to make a copy of the data array to
                # avoid modifying the original data array.
                # Move the axes and combine the last two for
                # performance.
                col_data = self._data[:y1, x1:].copy()
                col_mask = mask[:y1, x1:]
                col_data[col_mask] = np.nan
                col_data = reshape_as_blocks(col_data,
                                             (self.box_size[0], 1))
                col_data = np.transpose(col_data, (0, 3, 1, 2))
                col_data = col_data.reshape((*col_data.shape[:-2], -1))
                col_stats = self._compute_box_statistics(
                    col_data, axis=-1, sigma_clip=self.sigma_clip)
                col_bkg, col_bkgrms, col_n_good = col_stats

            if extra_row and extra_col:
                # Extra corner box, appended to the extra column.
                # Here we need to make a copy of the data array to
                # avoid modifying the original data array.
                corner_data = self._data[y1:, x1:].copy()
                corner_mask = mask[y1:, x1:]
                corner_data[corner_mask] = np.nan
                crn_stats = self._compute_box_statistics(
                    corner_data, axis=None, sigma_clip=self.sigma_clip)
                crn_bkg, crn_bkgrms, crn_n_good = crn_stats
                col_bkg = np.vstack((col_bkg, crn_bkg))
                col_bkgrms = np.vstack((col_bkgrms, crn_bkgrms))
                col_n_good = np.vstack((col_n_good, crn_n_good))

        # Combine the core and extra boxes to construct the complete
        # 2D bkg and bkgrms arrays
        if extra_row:
            bkg = np.vstack([bkg, row_bkg[:, 0]])
            bkgrms = np.vstack([bkgrms, row_bkgrms[:, 0]])
            n_good = np.vstack([n_good, row_n_good[:, 0]])

        if extra_col:
            bkg = np.hstack([bkg, col_bkg])
            bkgrms = np.hstack([bkgrms, col_bkgrms])
            n_good = np.hstack([n_good, col_n_good])

        if np.all(np.isnan(bkg)):
            msg = ('All boxes contain fewer than '
                   f'{self._good_n_pixels_threshold} unmasked or finite '
                   'pixels or are completely masked '
                   f'(box_size={tuple(self.box_size)}, '
                   f'exclude_percentile={self.exclude_percentile}). Please '
                   'check your data or increase "exclude_percentile" to '
                   'allow more boxes to be included.')
            raise ValueError(msg)

        # We no longer need the temporary input arrays
        del self._data
        del self._mask

        return bkg, bkgrms, n_good

    def _interpolate_grid(self, data, *, n_neighbors=10, eps=0.0, power=1.0,
                          regularization=0.0):
        """
        Fill in any NaN values in the low-resolution 2D mesh background
        and background RMS images using inverse distance weighting (IDW)
        interpolation.

        This method ensures that the low-resolution mesh contains no
        NaNs before applying a regular-grid interpolator to expand it to
        the full image size. If there are no NaNs, the input is returned
        (cast to the original dtype). Otherwise, NaN pixels are replaced
        by IDW interpolation using valid mesh values.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of the box statistics, possibly containing NaNs.

        n_neighbors : int, optional
            The maximum number of nearest neighbors to use during the
            interpolation.

        eps : float, optional
            Approximation parameter for nearest neighbors (see
            `scipy.spatial.cKDTree.query`).

        power : float, optional
            The power of the inverse distance used for the interpolation
            weights.

        regularization : float, optional
            Regularization parameter to control the smoothness of the
            interpolator.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The input array with NaNs replaced by interpolated values.
        """
        if not np.any(np.isnan(data)):
            return data

        mask = ~np.isnan(data)
        idx = np.where(mask)
        yx = np.column_stack(idx)
        interp_func = ShepardIDWInterpolator(yx, data[mask])

        # Interpolate the masked pixels where data is NaN
        idx = np.where(np.isnan(data))
        yx_indices = np.column_stack(idx)
        interp_values = interp_func(yx_indices, n_neighbors=n_neighbors,
                                    power=power, eps=eps,
                                    regularization=regularization)

        interp_data = np.copy(data)  # copy to avoid modifying the input data
        interp_data[idx] = interp_values

        return interp_data

    def _selective_filter(self, data, bkg_grid):
        """
        Filter only pixels above ``filter_threshold`` in a low-
        resolution 2D image.

        The pixels to be filtered are determined by applying the
        ``filter_threshold`` to the low-resolution background mesh. The
        same pixels are filtered in both the background and background
        RMS meshes.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of mesh values.

        bkg_grid : 2D `~numpy.ndarray`
            The NaN-interpolated, unfiltered background mesh used to
            select the pixels to filter.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The filtered 2D array of mesh values.
        """
        above_threshold = bkg_grid > self.filter_threshold
        if not np.any(above_threshold):
            return data

        # Apply the median filter across the whole mesh in one call,
        # then blend. Keep the filtered value only where the background
        # is above the threshold and use the original value everywhere
        # else.
        filtered = generic_filter(data, nanmedian, size=self.filter_size,
                                  mode='constant', cval=np.nan)
        return np.where(above_threshold, filtered, data)

    def _filter_grid(self, data, bkg_grid):
        """
        Apply a 2D median filter to a low-resolution 2D image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of mesh values.

        bkg_grid : 2D `~numpy.ndarray`
            The NaN-interpolated, unfiltered background mesh used to
            select the pixels to filter when ``filter_threshold`` is
            set.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The filtered 2D array of mesh values.
        """
        if tuple(self.filter_size) == (1, 1):
            return data

        if (self.filter_threshold is None
                or self.filter_threshold < self._min_bkg_stats):
            # Filter the entire array
            filtdata = generic_filter(data, nanmedian, size=self.filter_size,
                                      mode='constant', cval=np.nan)
        else:
            # Selectively filter the array
            filtdata = self._selective_filter(data, bkg_grid)

        return filtdata

    def _calculate_mesh_yxcen(self):
        """
        Calculate the y and x positions of the centers of the low-
        resolution background and background RMS meshes with respect to
        the input data array.

        This is used by the IDW interpolator to expand the low-
        resolution mesh to the full-size image. It is also used to plot
        the mesh boxes on the input image.
        """
        mesh_idx = np.where(~self._mesh_nan_mask)  # good mesh indices
        box_cen = (self.box_size - 1) / 2.0
        return (mesh_idx * self.box_size[:, None]) + box_cen[:, None]

    @property
    def background_mesh(self):
        """
        The low-resolution background image.

        This image is equivalent to the low-resolution "MINIBACK"
        background map check image in SourceExtractor.
        """
        return self._background_mesh

    @property
    def background_rms_mesh(self):
        """
        The low-resolution background RMS image.

        This image is equivalent to the low-resolution "MINIBACK_RMS"
        background rms map check image in SourceExtractor.
        """
        return self._background_rms_mesh

    @property
    @deprecated(since='3.0', alternative='n_pixels_mesh', until='4.0')
    def npixels_mesh(self):
        """
        A 2D array of the number pixels used to compute the statistics
        in each mesh.

        .. deprecated:: 3.0
            Use ``n_pixels_mesh`` instead.
        """
        return self._n_good

    @property
    def n_pixels_mesh(self):
        """
        A 2D array of the number pixels used to compute the statistics
        in each mesh.
        """
        return self._n_good

    @property
    @deprecated(since='3.0', alternative='n_pixels_map', until='4.0')
    def npixels_map(self):
        """
        A 2D map of the number of pixels used to compute the statistics
        in each mesh, resized to the shape of the input image.

        .. deprecated:: 3.0
            Use ``n_pixels_map`` instead.

        .. note::

            The returned image is (re)calculated each time this property
            is accessed. Store the result in a variable if you need to
            access it more than once.
        """
        return self.n_pixels_map

    @property
    def n_pixels_map(self):
        """
        A 2D map of the number of pixels used to compute the statistics
        in each mesh, resized to the shape of the input image.

        .. note::

            The returned image is (re)calculated each time this property
            is accessed. Store the result in a variable if you need to
            access it more than once.
        """
        n_pixels_map = block_replicate(self.n_pixels_mesh,
                                       self._interp_kwargs['box_size'],
                                       conserve_sum=False)
        return n_pixels_map[:self._interp_kwargs['shape'][0],
                            :self._interp_kwargs['shape'][1]]

    @cached_property
    def background_median(self):
        """
        The median value of the 2D low-resolution background map.

        This is equivalent to the value SourceExtractor prints to stdout
        (i.e., "(M+D) Background: <value>").

        .. note::

            This value is computed over the full ``background_mesh``,
            which includes IDW-interpolated values for any mesh boxes
            that were excluded from the statistics (e.g., due to masking
            or ``exclude_percentile``). It therefore represents the
            median of the final interpolated mesh, not solely the median
            of directly measured mesh values.
        """
        return np.median(self.background_mesh)

    @cached_property
    def background_rms_median(self):
        """
        The median value of the low-resolution background RMS map.

        This is equivalent to the value SourceExtractor prints to stdout
        (i.e., "(M+D) RMS: <value>").

        .. note::

            This value is computed over the full
            ``background_rms_mesh``, which includes IDW-interpolated
            values for any mesh boxes that were excluded from the
            statistics (e.g., due to masking or ``exclude_percentile``).
            It therefore represents the median of the final interpolated
            mesh, not solely the median of directly measured mesh
            values.
        """
        return np.median(self.background_rms_mesh)

    def _calculate_image(self, data):
        """
        Calculate the full-sized background or background rms image from
        the low-resolution mesh.
        """
        data = self.interpolator(data, **self._interp_kwargs)

        if self.coverage_mask is not None:
            data[self.coverage_mask] = self.fill_value

        return self._apply_units(data)

    @property
    def background(self):
        """
        A 2D `~numpy.ndarray` containing the background image.

        .. note::

            The returned image is (re)calculated each time this property
            is accessed. Store the result in a variable if you need to
            access it more than once.
        """
        return self._calculate_image(self.background_mesh)

    @property
    def background_rms(self):
        """
        A 2D `~numpy.ndarray` containing the background RMS image.

        .. note::

            The returned image is (re)calculated each time this property
            is accessed. Store the result in a variable if you need to
            access it more than once.
        """
        return self._calculate_image(self.background_rms_mesh)

    def plot_meshes(self, *, ax=None, marker='+', markersize=None,
                    color='blue', alpha=None, outlines=False, **kwargs):
        """
        Plot the low-resolution mesh boxes on a matplotlib Axes
        instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        marker : str, optional
            The `matplotlib marker
            <https://matplotlib.org/stable/api/markers_api.html>`_ to
            use to mark the center of the boxes.

        markersize : float, optional
            The box center marker size in ``points ** 2``
            (typographical points are 1/72 inch). The default is
            ``matplotlib.rcParams['lines.markersize'] ** 2``. If set to
            0, then the box center markers will not be plotted.

        color : str, optional
            The color for the box center markers and outlines.

        alpha : float, optional
            The alpha blending value, between 0 (transparent) and 1
            (opaque), for the box center markers and outlines.

        outlines : bool, optional
            Whether or not to plot the box outlines.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`, which is used to draw the box
            outlines. Used only if ``outlines`` is True.
        """
        import matplotlib.pyplot as plt

        kwargs['color'] = color
        if ax is None:
            ax = plt.gca()

        mesh_xycen = np.flipud(self._calculate_mesh_yxcen())
        ax.scatter(*mesh_xycen, s=markersize, marker=marker, color=color,
                   alpha=alpha)

        if outlines:
            xycen = np.column_stack(mesh_xycen)
            apers = RectangularAperture(xycen, w=self.box_size[1],
                                        h=self.box_size[0], theta=0.0)
            apers.plot(ax=ax, alpha=alpha, **kwargs)
