# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
from itertools import product
import warnings
import numpy as np
from numpy.lib.index_tricks import index_exp
from astropy.stats import sigma_clip
from astropy.utils import lazyproperty
from .utils import ShepardIDWInterpolator

import astropy
if LooseVersion(astropy.__version__) < LooseVersion('1.1'):
    ASTROPY_LT_1P1 = True
else:
    ASTROPY_LT_1P1 = False


__all__ = ['BackgroundBase', 'Background']

__doctest_requires__ = {('Background'): ['scipy']}


class BackgroundBase(object):
    """
    Base class for background classes.

    The background classes estimate the 2D background and background rms
    noise in an image.

    The background is estimated using sigma-clipped statistics in each
    mesh of a grid that covers the input ``data`` to create a
    low-resolution, and possibly irregularly-gridded, background map.

    The final background map is calculated by interpolating the
    low-resolution background map.

    Parameters
    ----------
    data : array_like
        The 2D array from which to estimate the background and/or
        background rms map.

    box_size : int or array_like (int)
        The box size along each axis.  If ``box_size`` is a scalar then
        a square box of size ``box_size`` will be used.  If ``box_size``
        has two elements, they should be in ``(ny, nx)`` order.  For
        best results, the box shape should be chosen such that the
        ``data`` are covered by an integer number of boxes in both
        dimensions.  When this is not the case, see the ``pad`` keyword
        for more options.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    remove_masked : {'all', 'any'}, optional
        Determines whether to include a particular mesh in the
        background interpolation based on the number of masked pixels it
        contains:

            * 'all':  exclude meshes that contain all masked pixels
            * 'any':  exclude meshes that contain any masked pixels

        The default is 'all'.  Note that this applies only to pixels in
        the input ``mask``.  It does not include pixels that are
        rejected via sigma clipping.

    filter_size : int or array_like (int), optional
        The window size of the 2D median filter to apply to the
        low-resolution background map.  If ``filter_size`` is a scalar
        then a square box of size ``filter_size`` will be used.  If
        ``filter_size`` has two elements, they should be in ``(ny, nx)``
        order.  A filter size of ``1`` (or ``(1, 1)``) means no
        filtering.

    filter_threshold : int, optional
        The threshold value for used for selective median filtering of
        the low-resolution 2D background map.  If not `None`, then the
        median filter will be applied to only the background meshes with
        values larger than ``filter_threshold``.

    method : {'mean', 'median', 'mode_estimator', 'sextractor'}, optional
        The method used to estimate the background in each of the
        meshes.  For all methods, the statistics are calculated from the
        sigma-clipped ``data`` values in each mesh.

        * 'mean':  Mean.
        * 'median':  Median.
        * 'mode_estimator':  A mode estimator of the form
          ``(3 * median) - (2 * mean)``.
        * 'sextractor':  The mode estimator used by `SExtractor`_:
          ``(2.5 * median) - (1.5 * mean)``.  If ``(mean - median) / std >
          0.3`` then the median is used instead.  Despite what the
          `SExtractor`_ User's Manual says, this is the method it *always*
          uses.
        * 'custom': Use this method in combination with the
          ``backfunc`` parameter to specific a custom function to
          calculate the background in each mesh.

    backfunc : callable
        The function to compute the background in each mesh.  Must be a
        callable that takes in a 2D `~numpy.ma.MaskedArray` of size
        ``NxZ``, where the ``Z`` axis (axis=1) contains the
        sigma-clipped pixels in each background mesh, and outputs a 1D
        `~numpy.ndarray` low-resolution background map of length ``N``.
        ``backfunc`` is used only if ``method='custom'``.

    edge_method : {'crop', 'pad'}, optional
        The method used to determine how to handle the case where the
        image size is not an integer multiple of the ``box_size`` in
        either dimension.  Both options will resize the image to give an
        exact multiple of ``box_size`` in both dimensions.

        * ``'crop'``: crop the image along the top and/or right edges.
        * ``'pad'``: pad the image along the top and/or right edges

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when sigma-clipping the data in each mesh.

    sigclip_iters : int, optional
       The number of iterations to use when sigma-clipping the data in
       each mesh.  A value of `None` means clipping will continue until
       convergence is achieved (i.e., continue until the last iteration
       clips nothing).  The default is 10.

    Notes
    -----
    If there is only one background mesh element (i.e., ``box_size`` is
    the same size as the ``data``), then the background map will simply
    be a constant image.

    Reducing ``sigclip_iters`` will speed up the calculations,
    especially for large images, at the cost of some precision.

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    def __init__(self, data, box_size, mask=None, remove_masked='all',
                 filter_size=(3, 3), filter_threshold=None,
                 method='sextractor', backfunc=None, edge_method='crop',
                 sigclip_sigma=3., sigclip_iters=10):

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask shape must match data shape')

        valid_methods = ['mean', 'median', 'mode_estimator', 'sextractor',
                         'custom']
        if method not in valid_methods:
            raise ValueError('method "{0}" is not valid'.format(method))

        box_size = np.atleast_1d(box_size)
        if len(box_size) == 1:
            box_size = np.repeat(box_size, 2)
        self.box_size = (min(box_size[0], data.shape[0]),
                         min(box_size[1], data.shape[1]))
        self.box_npts = self.box_size[0] * self.box_size[1]

        self.data = data
        self.mask = mask
        self.remove_masked = remove_masked

        if len(filter_size) == 1:
            filter_size = np.repeat(filter_size, 2)
        self.filter_size = filter_size
        self.filter_threshold = filter_threshold

        self.method = method
        self.backfunc = backfunc
        self.edge_method = edge_method
        self.sigclip_sigma = sigclip_sigma
        self.sigclip_iters = sigclip_iters

        self._resize_data()
        self._sigclip_data()
        self._calc_bkg_meshes1d()
        self._calc_bkg_meshes2d()

        # the data coordinates to use when calling an interpolator
        nx, ny = self.data.shape
        self.data_coords = np.array(list(product(range(ny), range(nx))))

    def _resize_data(self):
        """
        Pad or crop the 2D data array so that there are an integer
        number of meshes in both dimensions.
        """

        self.nyboxes = self.data.shape[0] // self.box_size[0]
        self.nxboxes = self.data.shape[1] // self.box_size[1]
        self.yextra = self.data.shape[0] % self.box_size[0]
        self.xextra = self.data.shape[1] % self.box_size[1]

        if (self.xextra + self.yextra) == 0:
            self.data_ma = self.data
        else:
            if self.edge_method == 'pad':
                self.data_ma = self._pad_data()
                self.nyboxes += 1
                self.nxboxes += 1
            elif self.edge_method == 'crop':
                self.data_ma = self._crop_data()
            else:
                raise ValueError('edge_method must be "pad" or "crop"')
        return

    def _pad_data(self):
        """
        Pad the ``data`` and ``mask`` on the top and/or right to have a
        integer number of background meshes of size ``box_size`` in both
        dimensions.
        """

        ypad, xpad = 0, 0
        if self.yextra > 0:
            ypad = self.box_size[0] - self.yextra
        if self.xextra > 0:
            xpad = self.box_size[1] - self.xextra
        pad_width = ((0, ypad), (0, xpad))
        # mode must be a string for numpy < 0.11
        # (see https://github.com/numpy/numpy/issues/7112)
        mode = str('constant')
        padded_data = np.pad(self.data, pad_width, mode=mode,
                             constant_values=[np.nan])
        padded_data_mask = np.isnan(padded_data)

        # must handle the mask separately (no np.ma.pad type function)
        if self.mask is not None:
            padded_mask = np.pad(self.mask, pad_width, mode=mode,
                                 constant_values=[False])
            padded_data_mask = np.logical_or(padded_data_mask, padded_mask)

        return np.ma.masked_array(padded_data, mask=padded_data_mask)

    def _crop_data(self):
        """
        Crop the ``data`` and ``mask`` on the top and/or right to have a
        integer number of background meshes of size ``box_size`` in both
        dimensions.
        """

        ny_crop = self.nyboxes * self.box_size[0]
        nx_crop = self.nxboxes * self.box_size[1]
        crop_slc = index_exp[0:ny_crop, 0:nx_crop]
        if self.mask is not None:
            mask = self.mask[crop_slc]
        else:
            mask = False
        return np.ma.masked_array(self.data[crop_slc], mask=mask)

    def _define_mesh_indices(self, nmasked_img):
        """
        Define the x and y indices with respect to the low-resolution
        mesh of the meshes to use for the background interpolation.

        The ``remove_masked`` option determines which meshes are used
        for the background interpolation.

        Parameters
        ----------
        nmasked_img : 2D `~numpy.ndarray` of int
            A 2D array containing the number of masked pixels in each
            mesh.

        Returns
        -------
        yidx, xidx : `~numpy.ndarray`
            The ``y`` and ``x`` mesh indices.
        """

        if self.remove_masked == 'any':
            # remove meshes that have any masked pixels
            yidx, xidx = np.where(nmasked_img == 0)
            if len(yidx) == 0:
                raise ValueError('All meshes contain at least one masked '
                                 'pixel.')
        elif self.remove_masked == 'all':
            # remove meshes where all pixels are masked
            yidx, xidx = np.where((self.box_npts - nmasked_img) != 0)
            if len(yidx) == 0:
                raise ValueError('All meshes are completely masked.')
        elif self.remove_masked == '_none':
            # include all meshes (effectively same as 'all' because
            # mask will be True where all pixels are masked)
            yidx, xidx = np.mgrid[0:self.nyboxes, 0:self.nxboxes]
            yidx = yidx.ravel()
            xidx = xidx.ravel()
        else:
            raise ValueError('remove_masked must be "any", or "all".')
        return yidx, xidx

    def _convert_1d_to_2d_mesh(self, data):
        """
        Convert a 1D array of mesh values to a masked 2D mesh array
        given the 2D mesh indices ``mesh_yidx`` and ``mesh_xidx``.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            The 1D array.

        Returns
        -------
        result : 2D `~numpy.ma.MaskedArray`
            A 2D masked array.  Pixels not defined in ``mesh_yidx`` and
            ``mesh_xidx`` are masked.
        """

        if len(data) != len(self.mesh_yidx):
            raise ValueError('data and yidx must have the same length')

        if self.remove_masked == '_none':
            data2d = data.reshape((self.nyboxes, self.nxboxes))
            mask2d = False
        else:
            data2d = np.zeros((self.nyboxes, self.nxboxes))
            data2d[self.mesh_yidx, self.mesh_xidx] = data
            mask2d = np.ones_like(data2d).astype(np.bool)
            mask2d[self.mesh_yidx, self.mesh_xidx] = False
        return np.ma.masked_array(data2d, mask=mask2d)

    def _sigclip_data(self):
        """
        Perform sigma clipping on the (masked) data in regions of size
        ``box_size``.
        """

        data3d = np.ma.swapaxes(self.data_ma.reshape(
            self.nyboxes, self.box_size[0], self.nxboxes, self.box_size[1]),
            1, 2).reshape(self.nyboxes, self.nxboxes, self.box_npts)
        del self.data_ma

        # the number of masked pixels in each mesh including *only* the
        # input (and padding) mask
        self._nmasked_img_orig = np.ma.count_masked(data3d, axis=2)

        self.mesh_yidx, self.mesh_xidx = self._define_mesh_indices(
            self._nmasked_img_orig)
        data2d = data3d[self.mesh_yidx, self.mesh_xidx, :]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if ASTROPY_LT_1P1:
                self.data_sigclip = sigma_clip(
                    data2d, sig=self.sigclip_sigma, axis=1,
                    iters=self.sigclip_iters, cenfunc=np.ma.median,
                    varfunc=np.ma.var)
            else:
                self.data_sigclip = sigma_clip(
                    data2d, sigma=self.sigclip_sigma, axis=1,
                    iters=self.sigclip_iters, cenfunc=np.ma.median,
                    stdfunc=np.std)

        # the number of masked and unmasked pixels in each mesh
        # including *both* the input (and padding) mask and pixels masked
        # via sigma clipping
        nmasked = np.ma.count_masked(self.data_sigclip, axis=1)
        self.nmasked_img = self._convert_1d_to_2d_mesh(nmasked)
        self.nunmasked_img = self.box_npts - self.nmasked_img
        return

    def _calc_bkg_meshes1d(self):
        """
        Calculate 1D arrays containing the background and background rms
        estimate in each of the meshes.  The 1D x and y indices of the
        meshes values with respect to the input ``data`` are also
        calculated.

        These values can be used with interpolators that require 1D
        positions and array values (e.g. `ShepardIDWInterpolator').
        """

        if self.method == 'mean':
            bkg_mesh1d = np.ma.mean(self.data_sigclip, axis=1)
        elif self.method == 'median':
            bkg_mesh1d = np.ma.median(self.data_sigclip, axis=1)
        elif self.method == 'mode_estimator':
            bkg_mesh1d = (3. * np.ma.median(self.data_sigclip, axis=1) -
                          2. * np.ma.mean(self.data_sigclip, axis=1))
        elif self.method == 'sextractor':
            box_mean = np.ma.mean(self.data_sigclip, axis=1)
            box_median = np.ma.median(self.data_sigclip, axis=1)
            box_std = np.ma.std(self.data_sigclip, axis=1)
            condition = (np.abs(box_mean - box_median) / box_std) < 0.3
            bkg_est = (2.5 * box_median) - (1.5 * box_mean)
            bkg_mesh1d = np.ma.where(condition, bkg_est, box_median)
            bkg_mesh1d = np.ma.where(box_std == 0, box_mean, bkg_mesh1d)
        elif self.method == 'custom':
            bkg_mesh1d = self.backfunc(self.data_sigclip)
            if not isinstance(bkg_mesh1d, np.ndarray):   # np.ma will pass
                raise ValueError('"backfunc" must return a numpy.ndarray.')
            if isinstance(bkg_mesh1d, np.ma.MaskedArray):
                raise ValueError('"backfunc" must return a numpy.ndarray, '
                                 'not a masked array.')
            if bkg_mesh1d.shape != (self.data_sigclip.shape[0], ):
                raise ValueError('The shape of the array returned by '
                                 '"backfunc" is not correct.')

        bkgrms_mesh1d = np.ma.std(self.data_sigclip, axis=1)
        # NOTE:  remove_masked='_none' will return 1D arrays
        # that are masked for meshes that are completely masked
        # self.bkg_mesh1d = np.ma.filled(bkg_mesh1d, fill_value=np.nan)
        # self.bkgrms_mesh1d = np.ma.filled(bkgrms_mesh1d, fill_value=np.nan)
        self.bkg_mesh1d = bkg_mesh1d
        self.bkgrms_mesh1d = bkgrms_mesh1d

        # define the position arrays used to initialize an interpolator
        self.y = (self.mesh_yidx * self.box_size[0] +
                  (self.box_size[0] - 1) / 2.)
        self.x = (self.mesh_xidx * self.box_size[1] +
                  (self.box_size[1] - 1) / 2.)
        self.yx = np.column_stack([self.y, self.x])
        return

    def _interpolate_meshes(self, mesh1d):
        """
        Use IDW interpolation to fill in any masked pixels in the
        low-resolution 2D mesh background and background rms images.

        This is required to use a regular-grid interpolator to expand
        the low-resolution image to the full size image.

        Parameters
        ----------
        mesh1d : 1D `~numpy.ndarray`
            A 1D array of the mesh values.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            A 2D array of the mesh values where masked pixels have been
            filled by IDW interpolation.
        """

        yx = np.column_stack([self.mesh_yidx, self.mesh_xidx])
        coords = np.array(list(product(range(self.nyboxes),
                                       range(self.nxboxes))))
        f = ShepardIDWInterpolator(yx, mesh1d)
        img1d = f(coords, n_neighbors=10, power=1, eps=0.)
        return img1d.reshape((self.nyboxes, self.nxboxes))

    def _filter_meshes(self, mesh2d):
        """
        Apply a 2D median filter to the low-resolution 2D meshes,
        including only pixels inside the image at the borders.
        """

        from scipy.ndimage import generic_filter
        try:
            nanmedian_func = np.nanmedian    # numpy >= 1.9
        except AttributeError:
            from scipy.stats import nanmedian
            nanmedian_func = nanmedian

        if self.filter_threshold is None:
            return generic_filter(mesh2d, nanmedian_func,
                                  size=self.filter_size, mode='constant',
                                  cval=np.nan)
        else:
            # selectively filter only pixels above ``filter_threshold``
            data_out = np.copy(mesh2d)
            for i, j in zip(*np.nonzero(mesh2d > self.filter_threshold)):
                yfs, xfs = self.filter_size
                hyfs, hxfs = yfs // 2, xfs // 2
                y0, y1 = max(i - hyfs, 0), min(i - hyfs + yfs,
                                               mesh2d.shape[0])
                x0, x1 = max(j - hxfs, 0), min(j - hxfs + xfs,
                                               mesh2d.shape[1])
                data_out[i, j] = np.median(mesh2d[y0:y1, x0:x1])
            return data_out

    def _calc_bkg_meshes2d(self):
        """
        Calculate 2D arrays containing the background and background rms
        estimate in each of the meshes.  The strictly ascending 1D x and
        y indices of the meshes with respect to the input ``data`` are
        also calculated.

        These images are equivalent to the low-resolution
        "MINIBACKGROUND" and "MINIBACK_RMS" background maps in
        `SExtractor`_.

        Regular-grid interpolators require a 2D array of values.  Some
        require a 2D meshgrid of x and y.  Other require a strictly
        increasing 1D array of the x and y spans.
        """

        self.bkg_mesh2d = self._interpolate_meshes(self.bkg_mesh1d)
        self.bkgrms_mesh2d = self._interpolate_meshes(self.bkgrms_mesh1d)
        if self.filter_size != (1, 1):
            self.bkg_mesh2d = self._filter_meshes(self.bkg_mesh2d)
            self.bkgrms_mesh2d = self._filter_meshes(self.bkgrms_mesh2d)

        self.bkg_mesh2d_ma = self._convert_1d_to_2d_mesh(self.bkg_mesh1d)
        self.bkgrms_mesh2d_ma = self._convert_1d_to_2d_mesh(
            self.bkgrms_mesh1d)

    def plot_meshes(self, ax=None, marker='+', color='blue', outlines=False,
                    **kwargs):
        """
        Plot the low-resolution mesh boxes on a matplotlib Axes
        instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current ``Axes`` instance is used.

        marker : str, optional
            The marker to use to mark the center of the boxes.  Default
            is '+'.

        color : str, optional
            The color for the markers and the box outlines.  Default is
            'blue'.

        outlines : bool, optional
            Whether or not to plot the box outlines in addition to the
            box centers.

        kwargs
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.  Used only if ``outlines`` is
            True.
        """

        import matplotlib.pyplot as plt

        kwargs['color'] = color
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.x, self.y, marker=marker, color=color)
        if outlines:
            from .aperture_core import RectangularAperture
            xy = np.column_stack([self.x, self.y])
            apers = RectangularAperture(xy, self.box_size[1],
                                        self.box_size[0], 0.)
            apers.plot(ax=ax, **kwargs)
        return


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
        # mode must be a string for numpy < 0.11
        # (see https://github.com/numpy/numpy/issues/7112)
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
