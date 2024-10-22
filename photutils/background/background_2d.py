# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines classes to estimate the 2D background and background
RMS in an image.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.nddata import NDData, block_replicate, reshape_as_blocks
from astropy.stats import SigmaClip
from astropy.utils import lazyproperty
from astropy.utils.decorators import deprecated, deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import generic_filter

from photutils.aperture import RectangularAperture
from photutils.background.core import SExtractorBackground, StdBackgroundRMS
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                BkgZoomInterpolator)
from photutils.utils import ShepardIDWInterpolator
from photutils.utils._parameters import as_pair
from photutils.utils._repr import make_repr
from photutils.utils._stats import nanmedian, nanmin

__all__ = ['Background2D']

__doctest_skip__ = ['Background2D']


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

        Better performance will generally be obtained if you have
        the `bottleneck`_ package installed. This acceleration also
        requires that the byte order of the input data array matches
        the byte order of the operating system. For example, the
        `astropy.io.fits` module loads data arrays as big-endian, even
        though most modern processors are little-endian. A big-endian
        array can be converted to native byte order ('=') in place
        using::

            >>> data.byteswap(inplace=True)
            >>> data = data.view(data.dtype.newbyteorder('='))

        One can also use, e.g.,::

            >>> data = data.astype(float)

        but this will temporarily create a new copy of the array in
        memory.

    Parameters
    ----------
    data : array_like or `~astropy.nddata.NDData`
        The 2D array from which to estimate the background and/or
        background RMS map.

    box_size : int or array_like (int)
        The box size along each axis. If ``box_size`` is a scalar then
        a square box of size ``box_size`` will be used. If ``box_size``
        has two elements, they must be in ``(ny, nx)`` order. For best
        results, the box shape should be chosen such that the ``data``
        are covered by an integer number of boxes in both dimensions.
        When this is not the case, see the ``edge_method`` keyword for
        more options.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        masked. Masked data are excluded from calculations. ``mask`` is
        intended to mask sources or bad pixels. Use ``coverage_mask``
        to mask blank areas of an image. ``mask`` and ``coverage_mask``
        differ only in that ``coverage_mask`` is applied to the output
        background and background RMS maps (see ``fill_value``).

    coverage_mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        ``coverage_mask`` should be `True` where there is no coverage
        (i.e., no data) for a given pixel (e.g., blank areas in a mosaic
        image). It should not be used for bad pixels (in that case use
        ``mask`` instead). ``mask`` and ``coverage_mask`` differ only in
        that ``coverage_mask`` is applied to the output background and
        background RMS maps (see ``fill_value``).

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
        ``exclude_percentile=0`` will exclude boxes that have any that
        have any masked pixels. Note that completely masked boxes are
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

    filter_threshold : int, optional
        The threshold value for used for selective median filtering of
        the low-resolution 2D background map. The median filter will
        be applied to only the background boxes with values larger
        than ``filter_threshold``. Set to `None` to filter all boxes
        (default).

    edge_method : {'pad', 'crop'}, optional
        This keyword will be removed in a future version and the default
        version of ``'pad'`` will always be used. The ``'crop'`` option
        has been strongly discouraged for some time now. Its usage
        creates a undesirable scaling of the low-resolution maps that
        leads to incorrect results.

        The method used to determine how to handle the case where the
        image size is not an integer multiple of the ``box_size``
        in either dimension. Both options will resize the image for
        internal calculations to give an exact multiple of ``box_size``
        in both dimensions.

        * ``'pad'``: pad the image along the top and/or right edges.
          This is the default and recommended method. Ideally, the
          ``box_size`` should be chosen such that an integer number
          of boxes is only slightly larger than the ``data`` size to
          minimize the amount of padding.
        * ``'crop'``: crop the image along the top and/or right edges.
          This method should be used sparingly, and it may be deprecated
          in the future. Best results will occur when ``box_size`` is
          chosen such that an integer number of boxes is only slightly
          smaller than the ``data`` size to minimize the amount of
          cropping.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will
        be performed. The default is to perform sigma clipping with
        ``sigma=3.0`` and ``maxiters=10``.

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

    bkgrms_estimator : callable, optional
        A callable object (a function or e.g., an instance of
        any `~photutils.background.BackgroundRMSBase` subclass)
        used to estimate the background RMS in each of the boxes.
        The callable object must take in a 2D `~numpy.ndarray`
        or `~numpy.ma.MaskedArray` and have an ``axis`` keyword.
        Internally, the background RMS will be calculated along
        ``axis=1`` and in this case the callable object must return a
        1D `~numpy.ndarray`, where np.nan values are used for masked
        pixels. If ``bkgrms_estimator`` includes sigma clipping,
        it will be ignored (use the ``sigma_clip`` keyword here
        to define sigma clipping). The default is an instance of
        `~photutils.background.StdBackgroundRMS`.

    interpolator : callable, optional
        A callable object (a function or object) used to interpolate
        the low-resolution background or background RMS image to the
        full-size background or background RMS maps. The default
        is an instance of `BkgZoomInterpolator`, which uses the
        `scipy.ndimage.zoom` function.

    Notes
    -----
    Better performance will generally be obtained if you have the
    `bottleneck`_ package installed.

    If there is only one background box element (i.e., ``box_size`` is
    the same size as (or larger than) the ``data``), then the background
    map will simply be a constant image.

    .. _bottleneck:  https://github.com/pydata/bottleneck
    """

    @deprecated_renamed_argument('edge_method', None, '2.0.0')
    def __init__(self, data, box_size, *, mask=None, coverage_mask=None,
                 fill_value=0.0, exclude_percentile=10.0, filter_size=(3, 3),
                 filter_threshold=None, edge_method='pad',
                 sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
                 bkg_estimator=SExtractorBackground(sigma_clip=None),
                 bkgrms_estimator=StdBackgroundRMS(sigma_clip=None),
                 interpolator=BkgZoomInterpolator()):

        if isinstance(data, (u.Quantity, NDData)):  # includes CCDData
            self._unit = data.unit
            data = data.data
        else:
            self._unit = None

        # this is a temporary instance variable to store the input data
        self._data = self._validate_array(data, 'data', shape=False)

        self._data_dtype = self._data.dtype

        self._mask = self._validate_array(mask, 'mask')
        self.coverage_mask = self._validate_array(coverage_mask,
                                                  'coverage_mask')

        # box_size cannot be larger than the data array size
        self.box_size = as_pair('box_size', box_size, lower_bound=(0, 1),
                                upper_bound=data.shape)

        self.fill_value = fill_value
        if exclude_percentile < 0 or exclude_percentile > 100:
            raise ValueError('exclude_percentile must be between 0 and 100 '
                             '(inclusive).')
        self.exclude_percentile = exclude_percentile
        self.filter_size = as_pair('filter_size', filter_size,
                                   lower_bound=(0, 1), check_odd=True)
        self.filter_threshold = filter_threshold
        if edge_method not in ('pad', 'crop'):
            raise ValueError('edge_method must be "pad" or "crop"')
        self.edge_method = edge_method
        self.sigma_clip = sigma_clip
        self.interpolator = interpolator

        # we perform sigma clipping as a separate step to avoid
        # calling it twice for the background and background RMS
        bkg_estimator.sigma_clip = None
        bkgrms_estimator.sigma_clip = None
        self.bkg_estimator = bkg_estimator
        self.bkgrms_estimator = bkgrms_estimator

        self._box_npixels = None
        self._params = ('box_size', 'coverage_mask',
                        'fill_value', 'exclude_percentile', 'filter_size',
                        'filter_threshold', 'edge_method', 'sigma_clip',
                        'bkg_estimator', 'bkgrms_estimator', 'interpolator')

        # store the interpolator keyword arguments for later use
        # (before self._data is deleted in self._calculate_stats)
        self._interp_kwargs = {'shape': self._data.shape,
                               'dtype': self._data.dtype,
                               'box_size': self.box_size,
                               'edge_method': self.edge_method}

        # perform the initial calculations to avoid storing large data
        # arrays and to keep the memory usage minimal
        (self._bkg_stats,
         self._bkgrms_stats,
         self._ngood) = self._calculate_stats()

        # this is used to selectively filter the low-resolution maps
        self._min_bkg_stats = nanmin(self._bkg_stats)

        # store a mask of the excluded mesh values (NaNs) in the
        # low-resolution maps
        self._mesh_nan_mask = np.isnan(self._bkg_stats)

        # add keyword arguments needed for BkgZoomInterpolator
        # BkgIDWInterpolator upscales the mesh based only on the good
        # pixels in the low-resolution mesh
        if isinstance(self.interpolator, BkgIDWInterpolator):
            self._interp_kwargs['mesh_yxcen'] = self._calculate_mesh_yxcen()
            self._interp_kwargs['mesh_nan_mask'] = self._mesh_nan_mask

    def __repr__(self):
        ellipsis = ('coverage_mask',)
        return make_repr(self, self._params, ellipsis=ellipsis)

    def __str__(self):
        ellipsis = ('coverage_mask',)
        return make_repr(self, self._params, ellipsis=ellipsis, long=True)

    def _validate_array(self, array, name, shape=True):
        """
        Validate the input data, mask, and coverage_mask arrays.
        """
        if name in ('mask', 'coverage_mask') and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                raise ValueError(f'{name} must be a 2D array.')
            if shape and array.shape != self._data.shape:
                raise ValueError(f'data and {name} must have the same shape.')
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

        mask = np.logical_or(self._mask, self.coverage_mask)
        del self._mask
        return mask

    def _combine_all_masks(self, mask):
        """
        Combine the input masks (mask and coverage_mask) with the mask
        of invalid data values.
        """
        input_mask = self._combine_input_masks()

        msg = ('Input data contains invalid values (NaNs or infs), which '
               'were automatically masked.')

        if input_mask is None:
            if np.any(mask):
                warnings.warn(msg, AstropyUserWarning)
            return mask

        total_mask = np.logical_or(input_mask, mask)

        if input_mask is not None:
            condition = np.logical_and(np.logical_not(input_mask), mask)
            if np.any(condition):
                warnings.warn(msg, AstropyUserWarning)

        return total_mask

    @lazyproperty
    def _good_npixels_threshold(self):
        """
        The minimum number of required unmasked pixels in a box used for
        it to be included in the low-resolution map.

        For exclude_percentile=0, only boxes where nmasked=0 will be
        included. For exclude_percentile=100, all boxes will be included
        *unless* they are completely masked.

        Boxes that are completely masked are always excluded.
        """
        return (1 - (self.exclude_percentile / 100.0)) * self._box_npixels

    def _sigmaclip_boxes(self, data, axis):
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

        Returns
        -------
        data : `~numpy.ndarray`
            The sigma-clipped data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            if self.sigma_clip is not None:
                data = self.sigma_clip(data, axis=axis, masked=False,
                                       copy=False)

        return data

    def _compute_box_statistics(self, data, axis=None):
        """
        Compute the background and background RMS statistics in each
        box.

        Parameters
        ----------
        data : `~numpy.ndarray`
            The 4D array of box data.

        axis : int or tuple of int, optional
            The axis or axes along which to compute the statistics.

        Returns
        -------
        bkg : 2D `~numpy.ndarray` or float
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray` or float
            The background RMS statistics in each box.
        """
        data = self._sigmaclip_boxes(data, axis=axis)

        # make 2D arrays of the box statistics
        bkg = self.bkg_estimator(data, axis=axis)
        bkgrms = self.bkgrms_estimator(data, axis=axis)

        # mask boxes with too few unmasked pixels
        ngood = np.count_nonzero(~np.isnan(data), axis=axis)
        box_mask = ngood <= self._good_npixels_threshold

        if np.ndim(bkg) == 0:
            if box_mask:  # single corner box
                # np.nan is float64; use np.float32 to prevent numpy from
                # promoting the output data dtype to float64 if the
                # input data is float32
                bkg = np.float32(np.nan)
                bkgrms = np.float32(np.nan)
        else:
            bkg[box_mask] = np.nan
            bkgrms[box_mask] = np.nan

        return bkg, bkgrms, ngood

    def _calculate_stats(self):
        """
        Calculate the background and background RMS statistics in each
        box.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D input data array. The data array is assumed to have
            been prepared by the ``_prepare_data`` method, where NaNs
            are used to mask invalid data values.

        Returns
        -------
        bkg : 2D `~numpy.ndarray`
            The background statistics in each box.

        bkgrms : 2D `~numpy.ndarray`
            The background RMS statistics in each box.

        ngood : 2D `~numpy.ndarray`
            The number of unmasked pixels in each box.
        """
        # if needed, copy the data to a float32 array to insert NaNs
        if self._data.dtype.kind != 'f':
            self._data = self._data.astype(np.float32)

        # automatically mask non-finite values that aren't already
        # masked and combine all masks
        mask = self._combine_all_masks(~np.isfinite(self._data))

        self._box_npixels = np.prod(self.box_size)
        nboxes = self._data.shape // self.box_size
        y1, x1 = nboxes * self.box_size

        # core boxes - the part of the data array that is an integer
        # multiple of the box size
        # combine the last two axes for performance
        # Below we transform both the data and mask arrays to avoid
        # making multiple copies of the data (one to insert NaN and
        # another for the reshape). Only one copy of the data and mask
        # array is made (except for the extra corner). The boolean mask
        # copy is much smaller than the data array.
        # An explicit copy of the data array is needed to avoid
        # modifying the original data array if the shape of the data
        # array is (y1, x1) (i.e., box_size = data.shape).
        core = reshape_as_blocks(self._data[:y1, :x1].copy(), self.box_size)
        core_mask = reshape_as_blocks(mask[:y1, :x1], self.box_size)
        core = core.reshape((*nboxes, -1))
        core_mask = core_mask.reshape((*nboxes, -1))
        core[core_mask] = np.nan
        bkg, bkgrms, ngood = self._compute_box_statistics(core, axis=-1)

        extra_row = y1 < self._data.shape[0]
        extra_col = x1 < self._data.shape[1]
        if self.edge_method == 'pad' and (extra_row or extra_col):
            if extra_row:
                # extra row of boxes
                # here we need to make a copy of the data array to avoid
                # modifying the original data array
                # move the axes and combine the last two for performance
                row_data = self._data[y1:, :x1].copy()
                row_mask = mask[y1:, :x1]
                row_data[row_mask] = np.nan
                row_data = reshape_as_blocks(row_data, (1, self.box_size[1]))
                row_data = np.moveaxis(row_data, 0, -1)
                row_data = row_data.reshape((*row_data.shape[:-2], -1))
                row_bkg, row_bkgrms, row_ngood = self._compute_box_statistics(
                    row_data, axis=-1)

            if extra_col:
                # extra column of boxes
                # here we need to make a copy of the data array to avoid
                # modifying the original data array
                # move the axes and combine the last two for performance
                col_data = self._data[:y1, x1:].copy()
                col_mask = mask[:y1, x1:]
                col_data[col_mask] = np.nan
                col_data = reshape_as_blocks(col_data, (self.box_size[0], 1))
                col_data = np.transpose(col_data, (0, 3, 1, 2))
                col_data = col_data.reshape((*col_data.shape[:-2], -1))
                col_bkg, col_bkgrms, col_ngood = self._compute_box_statistics(
                    col_data, axis=-1)

            if extra_row and extra_col:
                # extra corner box -- append to extra column
                # here we need to make a copy of the data array to avoid
                # modifying the original data array
                corner_data = self._data[y1:, x1:].copy()
                corner_mask = mask[y1:, x1:]
                corner_data[corner_mask] = np.nan
                crn_bkg, crn_bkgrms, crn_ngood = self._compute_box_statistics(
                    corner_data, axis=None)
                col_bkg = np.vstack((col_bkg, crn_bkg))
                col_bkgrms = np.vstack((col_bkgrms, crn_bkgrms))
                col_ngood = np.vstack((col_ngood, crn_ngood))

            # combine the core and extra boxes to construct the
            # complete 2D bkg and bkgrms arrays
            if extra_row:
                bkg = np.vstack([bkg, row_bkg[:, 0]])
                bkgrms = np.vstack([bkgrms, row_bkgrms[:, 0]])
                ngood = np.vstack([ngood, row_ngood[:, 0]])

            if extra_col:
                bkg = np.hstack([bkg, col_bkg])
                bkgrms = np.hstack([bkgrms, col_bkgrms])
                ngood = np.hstack([ngood, col_ngood])

        if np.all(np.isnan(bkg)):
            raise ValueError('All boxes contain <= '
                             f'{self._good_npixels_threshold} good pixels. '
                             'Please check your data or increase '
                             '"exclude_percentile" to allow more boxes to '
                             'be included.')

        # we no longer need the copy of the input array
        del self._data

        return bkg, bkgrms, ngood

    def _interpolate_grid(self, data, n_neighbors=10, eps=0.0, power=1.0,
                          reg=0.0):
        """
        Fill in any NaN values in the low-resolution 2D mesh background
        and background RMS images.

        IDW interpolation is used to replace the NaN pixels.

        This is required to use a regular-grid interpolator to expand
        the low-resolution image to the full size image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of the box statistics.

        n_neighbors : int, optional
            The maximum number of nearest neighbors to use during the
            interpolation.

        eps : float, optional
            Set to use approximate nearest neighbors; the kth neighbor
            is guaranteed to be no further than (1 + ``eps``) times the
            distance to the real *k*-th nearest neighbor. See
            `scipy.spatial.cKDTree.query` for further information.

        power : float, optional
            The power of the inverse distance used for the interpolation
            weights. See the Notes section for more details.

        reg : float, optional
            The regularization parameter. It may be used to control the
            smoothness of the interpolator. See the Notes section for
            more details.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            A 2D array of the box values where NaN values have been
            filled by IDW interpolation.
        """
        if not np.any(np.isnan(data)):
            # output integer dtype if input data was integer dtyle
            if data.dtype != self._data_dtype:
                data = data.astype(self._data_dtype)
            return data

        mask = ~np.isnan(data)
        idx = np.where(mask)
        yx = np.column_stack(idx)
        interp_func = ShepardIDWInterpolator(yx, data[mask])

        # interpolate the masked pixels where data is NaN
        idx = np.where(np.isnan(data))
        yx_indices = np.column_stack(idx)
        interp_values = interp_func(yx_indices, n_neighbors=n_neighbors,
                                    power=power, eps=eps, reg=reg)

        interp_data = np.copy(data)  # copy to avoid modifying the input data
        interp_data[idx] = interp_values

        # output integer dtype if input data was integer dtyle
        if interp_data.dtype != self._data_dtype:
            interp_data = interp_data.astype(self._data_dtype)

        return interp_data

    def _selective_filter(self, data):
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

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The filtered 2D array of mesh values.
        """
        data_filtered = np.copy(data)
        bkg_stats_interp = self._interpolate_grid(self._bkg_stats)
        yx_indices = np.column_stack(
            np.nonzero(bkg_stats_interp > self.filter_threshold))

        yfs, xfs = self.filter_size
        hyfs, hxfs = yfs // 2, xfs // 2
        for i, j in yx_indices:
            yidx0 = max(i - hyfs, 0)
            yidx1 = min(i - hyfs + yfs, data.shape[0])
            xidx0 = max(j - hxfs, 0)
            xidx1 = min(j - hxfs + xfs, data.shape[1])
            data_filtered[i, j] = np.median(data[yidx0:yidx1, xidx0:xidx1])

        return data_filtered

    def _filter_grid(self, data):
        """
        Apply a 2D median filter to a low-resolution 2D image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of mesh values.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The filtered 2D array of mesh values.
        """
        if tuple(self.filter_size) == (1, 1):
            return data

        if (self.filter_threshold is None
                or self.filter_threshold < self._min_bkg_stats):
            # filter the entire array
            filtdata = generic_filter(data, nanmedian, size=self.filter_size,
                                      mode='constant', cval=np.nan)
        else:
            # selectively filter the array
            filtdata = self._selective_filter(data)

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

    @lazyproperty
    def background_mesh(self):
        """
        The low-resolution background image.

        This image is equivalent to the low-resolution "MINIBACK"
        background map check image in SourceExtractor.
        """
        data = self._interpolate_grid(self._bkg_stats)
        if ('background_rms_mesh' in self.__dict__
                or self.filter_threshold is None):
            self._bkg_stats = None  # delete to save memory
        return self._apply_units(self._filter_grid(data))

    @lazyproperty
    def background_rms_mesh(self):
        """
        The low-resolution background RMS image.

        This image is equivalent to the low-resolution "MINIBACK_RMS"
        background rms map check image in SourceExtractor.
        """
        data = self._interpolate_grid(self._bkgrms_stats)
        self._bkgrms_stats = None  # delete to save memory
        return self._apply_units(self._filter_grid(data))

    @property
    @deprecated('2.0.0')
    def background_mesh_masked(self):
        """
        The low-resolution background image prior to any interpolation
        to fill NaN values.

        The array has NaN values where meshes were excluded.
        """
        data = self.background_mesh.copy()
        data[self._mesh_nan_mask] = np.nan
        return data

    @property
    @deprecated('2.0.0')
    def background_rms_mesh_masked(self):
        """
        The low-resolution background RMS image prior to any
        interpolation to fill NaN values.

        The array has NaN values where meshes were excluded.
        """
        data = self.background_rms_mesh.copy()
        data[self._mesh_nan_mask] = np.nan
        return data

    @property
    @deprecated('2.0.0', alternative='npixels_mesh')
    def mesh_nmasked(self):
        """
        A 2D array of the number of masked pixels in each mesh.

        NaN values indicate where meshes were excluded.
        """
        data = (np.prod(self.box_size) - self._ngood).astype(float)
        data[self._mesh_nan_mask] = np.nan
        return data

    @property
    def npixels_mesh(self):
        """
        A 2D array of the number pixels used to compute the statistics
        in each mesh.
        """
        return self._ngood

    @property
    def npixels_map(self):
        """
        A 2D map of the number of pixels used to compute the statistics
        in each mesh, resized to the shape of the input image.

        Note that the returned value is (re)calculated each time this
        property is accessed. If you need to access the returned image
        multiple times, you should store the result in a variable.
        """
        npixels_map = block_replicate(self.npixels_mesh,
                                      self._interp_kwargs['box_size'],
                                      conserve_sum=False)
        return npixels_map[:self._interp_kwargs['shape'][0],
                           :self._interp_kwargs['shape'][1]]

    @lazyproperty
    def background_median(self):
        """
        The median value of the 2D low-resolution background map.

        This is equivalent to the value SourceExtractor prints to stdout
        (i.e., "(M+D) Background: <value>").
        """
        return self._apply_units(np.median(self.background_mesh))

    @lazyproperty
    def background_rms_median(self):
        """
        The median value of the low-resolution background RMS map.

        This is equivalent to the value SourceExtractor prints to stdout
        (i.e., "(M+D) RMS: <value>").
        """
        return self._apply_units(np.median(self.background_rms_mesh))

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

        Note that the returned value is (re)calculated each time this
        property is accessed. If you need to access the background image
        multiple times, you should store the result in a variable.
        """
        return self._calculate_image(self.background_mesh)

    @property
    def background_rms(self):
        """
        A 2D `~numpy.ndarray` containing the background RMS image.

        Note that the returned value is (re)calculated each time this
        property is accessed. If you need to access the background rms
        image multiple times, you should store the result in a variable.
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
            (typographical points are 1/72 inch) . The default is
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
