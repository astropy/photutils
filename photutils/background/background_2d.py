# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate the 2D background and
background rms in a 2D image.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
from itertools import product
import warnings
import numpy as np
from numpy.lib.index_tricks import index_exp
from astropy.stats import sigma_clip
from astropy.utils import lazyproperty
from .core import SExtractorBackground
from ..utils import ShepardIDWInterpolator

import astropy
if LooseVersion(astropy.__version__) < LooseVersion('1.1'):
    ASTROPY_LT_1P1 = True
else:
    ASTROPY_LT_1P1 = False


__all__ = ['BackgroundBase2D', 'Background2D', 'BackgroundIDW2D',
           'std_blocksum']

__doctest_requires__ = {('Background2D'): ['scipy']}


class BackgroundBase2D(object):
    """
    Base class for 2D background classes.

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

    remove_masked : {'threshold', 'all', 'any'}, optional
        Determines whether to include a particular mesh in the
        background interpolation based on the number of masked pixels it
        contains:

            * ``'threshold'``:  exclude meshes that contain less than
              ``meshpix_threshold`` unmasked pixels after the
              sigma-clipping step.
            * ``'all'``:  exclude meshes that contain all masked pixels.
            * ``'any'``:  exclude meshes that contain any masked pixels.

        Note that ``'all'`` and ``'any'`` apply only to pixels in the
        input or padding ``mask``, while ``'threshold'`` also applies to
        pixels that are rejected via sigma clipping.

    meshpix_threshold : int, optional
        The number of unmasked pixels required in a mesh after the
        sigma-clipping step in order to use it in estimating the
        background and background rms.  This parameter is used only if
        ``remove_masked='threshold'``.

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

    bkg : callable
        A callable object (a function or e.g., an instance of any
        `~photutils.background.BackgroundBase` subclass) used to
        estimate the background in each of the meshes.  The callable
        must take in a 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        and have an ``axis`` keyword (internally, the background will be
        calculated along ``axis=1``).  The callable object must return a
        1D `~numpy.ma.MaskedArray`.  The default is an instance of
        `~photutils.background.SExtractorBackground` with ``sigma=3.``
        and ``iters=10``.

    edge_method : {'crop', 'pad'}, optional
        The method used to determine how to handle the case where the
        image size is not an integer multiple of the ``box_size`` in
        either dimension.  Both options will resize the image to give an
        exact multiple of ``box_size`` in both dimensions.

        * ``'crop'``: crop the image along the top and/or right edges.
        * ``'pad'``: pad the image along the top and/or right edges

    Notes
    -----
    If there is only one background mesh element (i.e., ``box_size`` is
    the same size as the ``data``), then the background map will simply
    be a constant image.
    """

    def __init__(self, data, box_size, mask=None, remove_masked='threshold',
                 meshpix_threshold=50, filter_size=(3, 3),
                 filter_threshold=None,
                 bkg=SExtractorBackground(sigma=3., iters=10),
                 edge_method='crop'):

        box_size = np.atleast_1d(box_size)
        if len(box_size) == 1:
            box_size = np.repeat(box_size, 2)
        self.box_size = (min(box_size[0], data.shape[0]),
                         min(box_size[1], data.shape[1]))
        self.box_npts = self.box_size[0] * self.box_size[1]

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask shape must match data shape')

        if remove_masked == 'threshold' and meshpix_threshold > self.box_npts:
            raise ValueError('meshpix_threshold must be smaller than the '
                             'number of pixels in mesh box.')

        self.data = data
        self.mask = mask
        self.remove_masked = remove_masked
        self.meshpix_threshold = meshpix_threshold

        filter_size = np.atleast_1d(filter_size)
        if len(filter_size) == 1:
            filter_size = np.repeat(filter_size, 2)
        self.filter_size = filter_size
        self.filter_threshold = filter_threshold
        self.bkg = bkg
        self.edge_method = edge_method

        self._resize_data()

        self._sigclip_data()

        self._calc_meshes1d()
        self._calc_meshes2d()

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
            self.data_ma = np.ma.masked_array(self.data, mask=self.mask)
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
        dimensions.  Returns a numpy masked array.
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
        dimensions.  Returns a numpy masked array.
        """

        ny_crop = self.nyboxes * self.box_size[0]
        nx_crop = self.nxboxes * self.box_size[1]
        crop_slc = index_exp[0:ny_crop, 0:nx_crop]
        if self.mask is not None:
            mask = self.mask[crop_slc]
        else:
            mask = False
        return np.ma.masked_array(self.data[crop_slc], mask=mask)

    def _define_mesh2d_indices(self, nmasked_mesh):
        """
        Define the x and y indices with respect to the low-resolution
        mesh image of the meshes to use for the background
        interpolation.

        The ``remove_masked`` option determines which meshes are used
        for the background interpolation.

        Parameters
        ----------
        nmasked_mesh : 2D `~numpy.ndarray` of int
            A 2D array containing the number of masked pixels in each
            mesh.

        Returns
        -------
        yidx, xidx : `~numpy.ndarray`
            The ``y`` and ``x`` mesh indices.
        """

        # np.where is used, not np.ma.where
        if isinstance(nmasked_mesh, np.ma.MaskedArray):
            raise ValueError('nmasked_mesh must not be a masked array.')

        if self.remove_masked == 'any':
            # include meshes that do not have any masked pixels
            yidx, xidx = np.where(nmasked_mesh == 0)
            if len(yidx) == 0:
                raise ValueError('All meshes contain at least one masked '
                                 'pixel.')
        elif self.remove_masked == 'all':
            # include meshes that are not completely masked
            yidx, xidx = np.where((self.box_npts - nmasked_mesh) != 0)
            if len(yidx) == 0:
                raise ValueError('All meshes are completely masked.')
        elif self.remove_masked == 'threshold':
            # include meshes only with at least meshpix_threshold
            # unmasked pixels
            yidx, xidx = np.where((self.box_npts - nmasked_mesh) >=
                                  self.meshpix_threshold)
            if len(yidx) == 0:
                raise ValueError('There are no valid meshes available.')
        elif self.remove_masked == '_none':
            # include all meshes (effectively same as 'all' because
            # mask will be True where all pixels are masked)
            yidx, xidx = np.mgrid[0:self.nyboxes, 0:self.nxboxes]
            yidx = yidx.ravel()
            xidx = xidx.ravel()
        else:
            raise ValueError('remove_masked must be "any", "all", or '
                             '"threshold".')
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
        self._nmasked_mesh_orig = np.ma.count_masked(data3d, axis=2)

        self.mesh_yidx, self.mesh_xidx = self._define_mesh2d_indices(
            self._nmasked_mesh_orig)
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

        if self.remove_masked == 'threshold':
            idx1d = np.where((self.box_npts - nmasked) >=
                             self.meshpix_threshold)
            self.data_sigclip = self.data_sigclip[idx1d]
            nmasked = nmasked[idx1d]
            self.mesh_yidx = self.mesh_yidx[idx1d]
            self.mesh_xidx = self.mesh_xidx[idx1d]
            if len(self.mesh_yidx) == 0:
                raise ValueError('There are no valid meshes available.')

        self.nmasked_mesh = self._convert_1d_to_2d_mesh(nmasked)
        self.nunmasked_mesh = self.box_npts - self.nmasked_mesh
        return

    def _calc_meshes1d(self):
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
            if bkg_mesh1d.shape != (self.data_sigclip.shape[0], ):
                raise ValueError('The shape of the array returned by '
                                 '"backfunc" is not correct.')

        # NOTE: remove_masked='_none' will return 1D arrays that are
        # masked for meshes that are completely masked
        self.background_mesh1d = bkg_mesh1d
        self.background_rms_mesh1d = np.ma.std(self.data_sigclip, axis=1)

        # define the position arrays used to initialize the final IDW
        # interpolation
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

    def _calc_meshes2d(self):
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

        self.background_mesh2d = self._interpolate_meshes(
            self.background_mesh1d)
        self.background_rms_mesh2d = self._interpolate_meshes(
            self.background_rms_mesh1d)
        if not np.array_equal(self.filter_size, [1, 1]):
            self.background_mesh2d = self._filter_meshes(
                self.background_mesh2d)
            self.background_rms_mesh2d = self._filter_meshes(
                self.background_rms_mesh2d)
            self.background_mesh1d = self.background_mesh2d[
                self.mesh_yidx, self.mesh_xidx]
            self.background_rms_mesh1d = self.background_rms_mesh2d[
                self.mesh_yidx, self.mesh_xidx]
        self.background_mesh2d_ma = self._convert_1d_to_2d_mesh(
            self.background_mesh1d)
        self.background_rms_mesh2d_ma = self._convert_1d_to_2d_mesh(
            self.background_rms_mesh1d)

    @lazyproperty
    def background_median(self):
        """
        The median value of the 2D low-resolution background map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) Background: <value>").
        """

        return np.median(self.background_mesh2d)

    @lazyproperty
    def background_rms_median(self):
        """
        The median value of the low-resolution background rms map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) RMS: <value>").
        """

        return np.median(self.background_rms_mesh2d)

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
            from ..aperture import RectangularAperture
            xy = np.column_stack([self.x, self.y])
            apers = RectangularAperture(xy, self.box_size[1],
                                        self.box_size[0], 0.)
            apers.plot(ax=ax, **kwargs)
        return


class Background2D(BackgroundBase2D):
    """
    Class to estimate a 2D background and background rms noise in an
    image.

    The background and background rms are estimated using sigma-clipped
    statistics in each mesh of a grid that covers the input ``data`` to
    create a low-resolution background map.

    The exact method used to estimate the background in each mesh can be
    set with the ``method`` parameter.  The background rms in each mesh
    is estimated by the sigma-clipped standard deviation.

    This class generates the full-sized background and background rms
    maps from the lower-resolution maps using bicubic spline
    interpolation.

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

    interp_order : int, optional
        The order of the spline interpolation used to resize the
        low-resolution background and background rms maps.  The value
        must be an integer in the range 0-5.  The default is 3 (bicubic
        interpolation).

    pad_crop : bool, optional
        When resizing the low-resolution map derived from padded input
        ``data`` (``edge_method='pad'``), this keywords determines
        whether to resize to the padded-data size and then crop back to
        the original data size (`True`; default) or if the
        low-resolution maps are resized directly to the original data
        size (`False`).

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    remove_masked : {'threshold', 'all', 'any'}, optional
        Determines whether to include a particular mesh in the
        background interpolation based on the number of masked pixels it
        contains:

            * ``'threshold'``:  exclude meshes that contain less than
              ``meshpix_threshold`` unmasked pixels after the
              sigma-clipping step.
            * ``'all'``:  exclude meshes that contain all masked pixels.
            * ``'any'``:  exclude meshes that contain any masked pixels.

        Note that ``'all'`` and ``'any'`` apply only to pixels in the
        input or padding ``mask``, while ``'threshold'`` also applies to
        pixels that are rejected via sigma clipping.

    meshpix_threshold : int, optional
        The number of unmasked pixels required in a mesh after the
        sigma-clipping step in order to use it in estimating the
        background and background rms.  This parameter is used only if
        ``remove_masked='threshold'``.

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

    def __init__(self, data, box_size, interp_order=3, pad_crop=True,
                 **kwargs):

        super(Background2D, self).__init__(data, box_size, **kwargs)
        self.interp_order = interp_order
        self.pad_crop = pad_crop
        self.data_slc = index_exp[0:data.shape[0], 0:data.shape[1]]
        self.background = self._resize_mesh(self.background_mesh2d)
        self.background_rms = self._resize_mesh(self.background_rms_mesh2d)

    def _resize_mesh(self, mesh2d):
        if np.ptp(mesh2d) == 0:
            return np.zeros_like(self.data) + np.min(mesh2d)
        else:
            from scipy.ndimage import zoom
            if self.edge_method == 'pad' and self.pad_crop:
                # matches photutils <= 0.2 Background class
                zoom_factor = (int(self.nyboxes * self.box_size[0] /
                                   mesh2d.shape[0]),
                               int(self.nxboxes * self.box_size[1] /
                                   mesh2d.shape[1]))
                result = zoom(mesh2d, zoom_factor, order=self.interp_order,
                              mode='reflect')
                return result[self.data_slc]
            else:
                zoom_factor = (float(self.data.shape[0] / mesh2d.shape[0]),
                               float(self.data.shape[1] / mesh2d.shape[1]))
                return zoom(mesh2d, zoom_factor, order=self.interp_order,
                            mode='reflect')


class BackgroundIDW2D(BackgroundBase2D):
    """
    Class to estimate a 2D background and background rms noise in an
    image.

    The background and background rms are estimated using sigma-clipped
    statistics in each mesh of a grid that covers the input ``data`` to
    create a low-resolution background map.

    The exact method used to estimate the background in each mesh can be
    set with the ``method`` parameter.  The background rms in each mesh
    is estimated by the sigma-clipped standard deviation.

    This class generates the full-sized background and background rms
    maps from the lower-resolution maps using inverse-distance weighting
    (IDW) interpolation.

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

    n_neighbors : int, optional
        The maximum number of nearest neighbors to use during the
        interpolation.

    power : float, optional
        The power of the inverse distance used for the interpolation
        weights.  See the Notes section for more details.

    reg : float, optional
        The regularization parameter. It may be used to control the
        smoothness of the interpolator. See the Notes section for more
        details.

    leafsize : float, optional
        The number of points at which the k-d tree algorithm switches
        over to brute-force. ``leafsize`` must be positive.  See
        `scipy.spatial.cKDTree` for further information.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    remove_masked : {'threshold', 'all', 'any'}, optional
        Determines whether to include a particular mesh in the
        background interpolation based on the number of masked pixels it
        contains:

            * ``'threshold'``:  exclude meshes that contain less than
              ``meshpix_threshold`` unmasked pixels after the
              sigma-clipping step.
            * ``'all'``:  exclude meshes that contain all masked pixels.
            * ``'any'``:  exclude meshes that contain any masked pixels.

        Note that ``'all'`` and ``'any'`` apply only to pixels in the
        input or padding ``mask``, while ``'threshold'`` also applies to
        pixels that are rejected via sigma clipping.

    meshpix_threshold : int, optional
        The number of unmasked pixels required in a mesh after the
        sigma-clipping step in order to use it in estimating the
        background and background rms.  This parameter is used only if
        ``remove_masked='threshold'``.

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

    def __init__(self, data, box_size, n_neighbors=8, power=1.0, reg=0.0,
                 leafsize=10, **kwargs):

        super(BackgroundIDW2D, self).__init__(data, box_size, **kwargs)

        f_bkg = ShepardIDWInterpolator(self.yx, self.background_mesh1d,
                                       leafsize=leafsize)
        bkg_1d = f_bkg(self.data_coords, n_neighbors=n_neighbors, power=power,
                       reg=reg)
        self.background = bkg_1d.reshape(data.shape)

        f_bkg_rms = ShepardIDWInterpolator(self.yx,
                                           self.background_rms_mesh1d,
                                           leafsize=leafsize)
        bkg_rms_1d = f_bkg_rms(self.data_coords, n_neighbors=n_neighbors,
                               power=power, reg=reg)
        self.background_rms = bkg_rms_1d.reshape(data.shape)


def _mesh_values(data, box_size, mask=None):
    """
    Extract all the data values in boxes of size box_size.
    """

    if mask is not None:
        data = np.ma.masked_array(data, mask=mask)

    ny, nx = data.shape
    nyboxes = ny // box_size[0]
    nxboxes = nx // box_size[1]
    ny_crop = nyboxes * box_size[0]
    nx_crop = nxboxes * box_size[1]
    crop_slc = index_exp[0:ny_crop, 0:nx_crop]
    data = data[crop_slc]

    data = np.ma.swapaxes(data.reshape(
        nyboxes, box_size[0], nxboxes, box_size[1]), 1, 2).reshape(
            nyboxes, nxboxes, box_size[0]*box_size[1])
    mesh_yidx, mesh_xidx = np.where(np.ma.count_masked(data, axis=2) == 0)
    mesh_values = data[mesh_yidx, mesh_xidx, :]

    # y = (mesh_yidx * box_size[0]) + (box_size[0] - 1) / 2.
    # x = (mesh_xidx * box_size[1]) + (box_size[1] - 1) / 2.
    return (mesh_yidx, mesh_xidx), mesh_values


def std_blocksum(data, block_sizes, mask=None):
    """
    Calculate the standard deviation of block-summed data values at
    sizes of ``block_sizes``.

    Parameters
    ----------
    data : array-like
        The 2D array to block sum.

    block_sizes : int, array-like of int
        An array of integer block sizes.

    mask : array-like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Meshes that contain *any* masked data are excluded from
        calculations.

    Returns
    -------
    result : `~numpy.ndarray`
        An array of the standard deviations of the block-summed array
        for the input ``block_sizes``.
    """

    stds = []
    block_sizes = np.atleast_1d(block_sizes)
    for block_size in block_sizes:
        mesh_idx, mesh_values = _mesh_values(data, (block_size, block_size),
                                             mask=mask)
        block_sums = np.sum(mesh_values, axis=1)
        stds.append(np.std(block_sums))
    return np.array(stds)
