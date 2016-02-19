# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
import numpy as np
from numpy.lib.index_tricks import index_exp
from astropy.stats import sigma_clip
from astropy.utils import lazyproperty
import warnings

import astropy
if LooseVersion(astropy.__version__) < LooseVersion('1.1'):
    ASTROPY_LT_1P1 = True
else:
    ASTROPY_LT_1P1 = False


__all__ = ['Background']

__doctest_requires__ = {('Background'): ['scipy']}


class Background(object):
    """
    Class to estimate a 2D background and background rms noise in an
    image.

    The background is estimated using sigma-clipped statistics in each
    mesh of a grid that covers the input ``data`` to create a
    low-resolution background map.  The final background map is the
    bicubic spline interpolation of the low-resolution map.

    The exact method used to estimate the background in each mesh can be
    set with the ``method`` parameter.  The background rms in each mesh
    is estimated by the sigma-clipped standard deviation.
    """

    def __init__(self, data, box_shape, filter_shape=(3, 3),
                 filter_threshold=None, mask=None, method='sextractor',
                 backfunc=None, interp_order=3, sigclip_sigma=3.,
                 sigclip_iters=10):
        """
        Parameters
        ----------
        data : array_like
            The 2D array from which to estimate the background and/or
            background rms map.

        box_shape : 2-tuple of int
            The ``(ny, nx)`` shape of the boxes in which to estimate the
            background.  For best results, the box shape should be
            chosen such that the ``data`` are covered by an integer
            number of boxes in both dimensions.

        filter_shape : 2-tuple of int, optional
            The ``(ny, nx)`` shape of the median filter to apply to the
            low-resolution background map.  A filter shape of ``(1, 1)``
            means no filtering.

        filter_threshold : int, optional
            The threshold value for used for selective median filtering
            of the low-resolution background map.  If not `None`, then
            the median filter will be applied to only the background
            meshes with values larger than ``filter_threshold``.

        mask : array_like (bool), optional
            A boolean mask, with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.  Masked data are excluded from all calculations.

        method : {'mean', 'median', 'sextractor', 'mode_estimate'}, optional
            The method use to estimate the background in the meshes.
            For all methods, the statistics are calculated from the
            sigma-clipped ``data`` values in each mesh.

            * 'mean':  Mean.
            * 'median':  Median.
            * 'sextractor':  The method used by `SExtractor`_.  The
              background in each mesh is a mode estimator: ``(2.5 *
              median) - (1.5 * mean)``.  If ``(mean - median) / std >
              0.3`` then the median is used instead.  Despite what the
              `SExtractor`_ User's Manual says, this is the method it
              *always* uses.
            * 'mode_estimate':  An alternative mode estimator:
              ``(3 * median) - (2 * mean)``.
            * 'custom': Use this method in combination with the
              ``backfunc`` parameter to specific a custom function to
              calculate the background in each mesh.

        backfunc : callable
            The function to compute the background in each mesh.  Must
            be a callable that takes in a 3D `~numpy.ma.MaskedArray` of
            size ``MxNxZ``, where the ``Z`` axis (axis=2) contains the
            sigma-clipped pixels in each background mesh, and outputs a
            2D `~numpy.ndarray` low-resolution background map of size
            ``MxN``.  ``backfunc`` is used only if ``method='custom'``.

        interp_order : int, optional
            The order of the spline interpolation used to resize the
            low-resolution background and background rms maps.  The
            value must be an integer in the range 0-5.  The default is 3
            (bicubic interpolation).

        sigclip_sigma : float, optional
            The number of standard deviations to use as the clipping limit
            when calculating the image background statistics.

        sigclip_iters : int, optional
           The number of iterations to perform sigma clipping, or `None` to
           clip until convergence is achieved (i.e., continue until the last
           iteration clips nothing) when calculating the image background
           statistics.  The default is 10.

        Notes
        -----
        If there is only 1 background mesh element (i.e., ``box_shape``
        is the same size as the ``data``), then the background map will
        simply be a constant image with the value in the background
        mesh.

        Limiting ``sigclip_iters`` will speed up the calculations,
        especially for large images, at the cost of some precision.

        .. _SExtractor: http://www.astromatic.net/software/sextractor
        """

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask shape must match data shape')
        valid_methods = ['mean', 'median', 'sextractor', 'mode_estimate',
                         'custom']
        if method not in valid_methods:
            raise ValueError('method "{0}" is not valid'.format(method))
        self.box_shape = (min(box_shape[0], data.shape[0]),
                          min(box_shape[1], data.shape[1]))
        self.filter_shape = filter_shape
        self.filter_threshold = filter_threshold
        self.mask = mask
        self.method = method
        self.backfunc = backfunc
        self.interp_order = interp_order
        self.sigclip_sigma = sigclip_sigma
        self.sigclip_iters = sigclip_iters
        self.yextra = data.shape[0] % box_shape[0]
        self.xextra = data.shape[1] % box_shape[1]
        self.data_shape = data.shape
        self.data_region = index_exp[0:data.shape[0], 0:data.shape[1]]
        if (self.yextra > 0) or (self.xextra > 0):
            self.padded = True
            data_ma = self._pad_data(data, mask)
        else:
            self.padded = False
            data_ma = np.ma.masked_array(data, mask=mask)
        self.data_ma_shape = data_ma.shape
        self._sigclip_data(data_ma)

    def _pad_data(self, data, mask=None):
        """
        Pad the ``data`` and ``mask`` on the right and top with zeros if
        necessary to have a integer number of background meshes of size
        ``box_shape``.
        """
        ypad, xpad = 0, 0
        if self.yextra > 0:
            ypad = self.box_shape[0] - self.yextra
        if self.xextra > 0:
            xpad = self.box_shape[1] - self.xextra

        pad_width = ((0, ypad), (0, xpad))
        mode = str('constant')
        padded_data = np.pad(data, pad_width, mode=mode,
                             constant_values=[np.nan])
        padded_mask = np.isnan(padded_data)

        if mask is not None:
            mask_pad = np.pad(mask, pad_width, mode=mode,
                              constant_values=[False])
            padded_mask = np.logical_or(padded_mask, mask_pad)
        return np.ma.masked_array(padded_data, mask=padded_mask)

    def _sigclip_data(self, data_ma):
        """
        Perform sigma clipping on the data in regions of size
        ``box_shape``.
        """

        ny, nx = data_ma.shape
        ny_box, nx_box = self.box_shape
        y_nbins = int(ny / ny_box)   # always integer because data were padded
        x_nbins = int(nx / nx_box)   # always integer because data were padded
        data_rebin = np.ma.swapaxes(data_ma.reshape(
            y_nbins, ny_box, x_nbins, nx_box), 1, 2).reshape(y_nbins, x_nbins,
                                                             ny_box * nx_box)
        del data_ma
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if ASTROPY_LT_1P1:
                self.data_sigclip = sigma_clip(
                    data_rebin, sig=self.sigclip_sigma, axis=2,
                    iters=self.sigclip_iters, cenfunc=np.ma.median,
                    varfunc=np.ma.var)
            else:
                self.data_sigclip = sigma_clip(
                    data_rebin, sigma=self.sigclip_sigma, axis=2,
                    iters=self.sigclip_iters, cenfunc=np.ma.median,
                    stdfunc=np.std)
        del data_rebin

    def _filter_meshes(self, data_low_res):
        """
        Apply a 2d median filter to the low-resolution background map,
        including only pixels inside the image at the borders.
        """

        from scipy.ndimage import generic_filter
        try:
            nanmedian_func = np.nanmedian    # numpy >= 1.9
        except AttributeError:
            from scipy.stats import nanmedian
            nanmedian_func = nanmedian

        if self.filter_threshold is None:
            return generic_filter(data_low_res, nanmedian_func,
                                  size=self.filter_shape, mode='constant',
                                  cval=np.nan)
        else:
            data_out = np.copy(data_low_res)
            for i, j in zip(*np.nonzero(data_low_res >
                                        self.filter_threshold)):
                yfs, xfs = self.filter_shape
                hyfs, hxfs = yfs // 2, xfs // 2
                y0, y1 = max(i - hyfs, 0), min(i - hyfs + yfs,
                                               data_low_res.shape[0])
                x0, x1 = max(j - hxfs, 0), min(j - hxfs + xfs,
                                               data_low_res.shape[1])
                data_out[i, j] = np.median(data_low_res[y0:y1, x0:x1])
            return data_out

    def _resize_meshes(self, data_low_res):
        """
        Resize the low-resolution background meshes to the original data
        size using bicubic interpolation.
        """

        if np.min(data_low_res) == np.max(data_low_res):
            # constant image (or only 1 mesh)
            return np.zeros(self.data_shape) + np.min(data_low_res)
        else:
            from scipy.ndimage import zoom
            zoom_factor = (int(self.data_ma_shape[0] / data_low_res.shape[0]),
                           int(self.data_ma_shape[1] / data_low_res.shape[1]))
            return zoom(data_low_res, zoom_factor, order=self.interp_order,
                        mode='reflect')

    @lazyproperty
    def background_low_res(self):
        """
        A 2D `~numpy.ndarray` containing the background estimate in each
        of the meshes of size ``box_shape``.

        This low-resolution background map is equivalent to the
        low-resolution "MINIBACKGROUND" background map in `SExtractor`_.
        """

        if self.method == 'mean':
            bkg_low_res = np.ma.mean(self.data_sigclip, axis=2)
        elif self.method == 'median':
            bkg_low_res = np.ma.median(self.data_sigclip, axis=2)
        elif self.method == 'sextractor':
            box_mean = np.ma.mean(self.data_sigclip, axis=2)
            box_median = np.ma.median(self.data_sigclip, axis=2)
            box_std = np.ma.std(self.data_sigclip, axis=2)
            condition = (np.abs(box_mean - box_median) / box_std) < 0.3
            bkg_est = (2.5 * box_median) - (1.5 * box_mean)
            bkg_low_res = np.ma.where(condition, bkg_est, box_median)
            bkg_low_res = np.ma.where(box_std == 0, box_mean, bkg_low_res)
        elif self.method == 'mode_estimate':
            bkg_low_res = (3. * np.ma.median(self.data_sigclip, axis=2) -
                           2. * np.ma.mean(self.data_sigclip, axis=2))
        elif self.method == 'custom':
            bkg_low_res = self.backfunc(self.data_sigclip)
            if not isinstance(bkg_low_res, np.ndarray):   # np.ma will pass
                raise ValueError('"backfunc" must return a numpy.ndarray.')
            if isinstance(bkg_low_res, np.ma.MaskedArray):
                raise ValueError('"backfunc" must return a numpy.ndarray.')
            if bkg_low_res.shape != (self.data_sigclip.shape[0],
                                     self.data_sigclip.shape[1]):
                raise ValueError('The shape of the array returned by '
                                 '"backfunc" is not correct.')
        if self.method != 'custom':
            bkg_low_res = np.ma.filled(bkg_low_res,
                                       fill_value=np.ma.median(bkg_low_res))
        if self.filter_shape != (1, 1):
            bkg_low_res = self._filter_meshes(bkg_low_res)
        return bkg_low_res

    @lazyproperty
    def background_rms_low_res(self):
        """
        A 2D `~numpy.ndarray` containing the background rms estimate in
        each of the meshes of size ``box_shape``.

        This low-resolution background rms map is equivalent to the
        low-resolution "MINIBACK_RMS" background rms map in
        `SExtractor`_.
        """

        bkgrms_low_res = np.ma.std(self.data_sigclip, axis=2)
        bkgrms_low_res = np.ma.filled(bkgrms_low_res,
                                      fill_value=np.ma.median(bkgrms_low_res))
        if self.filter_shape != (1, 1):
            bkgrms_low_res = self._filter_meshes(bkgrms_low_res)
        return bkgrms_low_res

    @lazyproperty
    def background(self):
        """
        A 2D `~numpy.ndarray` containing the background estimate.

        This is equivalent to the low-resolution "BACKGROUND" background
        map in `SExtractor`_.
        """

        bkg = self._resize_meshes(self.background_low_res)
        if self.padded:
            bkg = bkg[self.data_region]
        return bkg

    @lazyproperty
    def background_rms(self):
        """
        A 2D `~numpy.ndarray` containing the background rms estimate.

        This is equivalent to the low-resolution "BACKGROUND_RMS"
        background rms map in `SExtractor`_.
        """

        bkgrms = self._resize_meshes(self.background_rms_low_res)
        if self.padded:
            bkgrms = bkgrms[self.data_region]
        return bkgrms

    @lazyproperty
    def background_median(self):
        """
        The median value of the low-resolution background map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) Background: <value>").
        """

        return np.median(self.background_low_res)

    @lazyproperty
    def background_rms_median(self):
        """
        The median value of the low-resolution background rms map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) RMS: <value>").
        """

        return np.median(self.background_rms_low_res)
