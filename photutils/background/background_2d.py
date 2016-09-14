# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate the 2D background and
background RMS in a 2D image.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from itertools import product
import numpy as np
from numpy.lib.index_tricks import index_exp
from astropy.utils import lazyproperty
from .core import SigmaClip, SExtractorBackground, StdBackgroundRMS
from ..utils import ShepardIDWInterpolator


__all__ = ['BackgroundBase2D', 'Background2D', 'BackgroundIDW2D',
           'std_blocksum']

__doctest_requires__ = {('Background2D'): ['scipy']}


class BackgroundBase2D(object):
    """
    Base class for 2D background classes.

    The background classes estimate the 2D background and background RMS
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
        background RMS map.

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

    remove_mesh : {'threshold', 'all', 'any'}, optional
        Determines whether to remove a particular mesh in the background
        interpolation based on the number of masked pixels it contains:

            * ``'threshold'``:  exclude meshes that contain less than
              ``remove_mesh_percentile`` percent unmasked pixels.  This
              is the default.
            * ``'all'``:  exclude meshes that contain all masked pixels
            * ``'any'``:  exclude meshes that contain any masked pixels

    remove_mesh_percentile : int, optional
        The percent of unmasked pixels required in a mesh after the
        sigma clipping step in order to use it in estimating the
        background and background RMS.  This parameter is used only if
        ``remove_mesh='threshold'``.

    filter_size : int or array_like (int), optional
        The window size of the 2D median filter to apply to the
        low-resolution background map.  If ``filter_size`` is a scalar
        then a square box of size ``filter_size`` will be used.  If
        ``filter_size`` has two elements, they should be in ``(ny, nx)``
        order.  A filter size of ``1`` (or ``(1, 1)``) means no
        filtering.

    filter_threshold : int, optional
        The threshold value for used for selective median filtering of
        the low-resolution 2D background map.  The median filter will be
        applied to only the background meshes with values larger than
        ``filter_threshold``.  Set to `None` to filter all meshes
        (default).

    sigma_clip : `~photutils.background.SigmaClip` instance, optional
        A `~photutils.background.SigmaClip` object that defines the
        sigma clipping parameters.  If `None` then no sigma clipping
        will be performed.  The default is to perform sigma clipping
        with ``sigma=3.`` and ``iters=5``.

    bkg_estimator : callable, optional
        A callable object (a function or e.g., an instance of any
        `~photutils.background.BackgroundBase` subclass) used to
        estimate the background in each of the meshes.  The callable
        object must take in a 2D `~numpy.ndarray` or
        `~numpy.ma.MaskedArray` and have an ``axis`` keyword
        (internally, the background will be calculated along
        ``axis=1``).  The callable object must return a 1D
        `~numpy.ma.MaskedArray`.  If ``bkg_estimator`` includes sigma
        clipping, it will be ignored (use the ``sigma_clip`` keyword to
        define sigma clipping).  The default is an instance of
        `~photutils.background.SExtractorBackground`.

    bkgrms_estimator : callable, optional
        A callable object (a function or e.g., an instance of any
        `~photutils.background.BackgroundRMSBase` subclass) used to
        estimate the background RMS in each of the meshes.  The callable
        object must take in a 2D `~numpy.ndarray` or
        `~numpy.ma.MaskedArray` and have an ``axis`` keyword
        (internally, the background RMS will be calculated along
        ``axis=1``).  The callable object must return a 1D
        `~numpy.ma.MaskedArray`.  If ``bkgrms_estimator`` includes sigma
        clipping, it will be ignored (use the ``sigma_clip`` keyword to
        define sigma clipping).  The default is an instance of
        `~photutils.background.StdBackgroundRMS`.

    edge_method : {'crop', 'pad'}, optional
        The method used to determine how to handle the case where the
        image size is not an integer multiple of the ``box_size`` in
        either dimension.  Both options will resize the image to give an
        exact multiple of ``box_size`` in both dimensions.

        * ``'crop'``:  crop the image, roughly equally along all edges
        * ``'pad'``:  pad the image, roughly equally along all edges

    Notes
    -----
    If there is only one background mesh element (i.e., ``box_size`` is
    the same size as the ``data``), then the background map will simply
    be a constant image.
    """

    def __init__(self, data, box_size, mask=None, remove_mesh='threshold',
                 remove_mesh_percentile=0.5, filter_size=(3, 3),
                 filter_threshold=None,
                 sigma_clip=SigmaClip(sigma=3., iters=5),
                 bkg_estimator=SExtractorBackground(sigma_clip=None),
                 bkgrms_estimator=StdBackgroundRMS(sigma_clip=None),
                 edge_method='crop'):

        box_size = np.atleast_1d(box_size)
        if len(box_size) == 1:
            box_size = np.repeat(box_size, 2)
        self.box_size = (min(box_size[0], data.shape[0]),
                         min(box_size[1], data.shape[1]))
        self.box_npixels = self.box_size[0] * self.box_size[1]

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape')

        if remove_mesh_percentile < 0 or remove_mesh_percentile > 100:
            raise ValueError('remove_mesh_percentile must be between 0 and '
                             '1 (inclusive).')

        self.data = data
        self.mask = mask
        self.remove_mesh = remove_mesh
        self.remove_mesh_percentile = remove_mesh_percentile

        filter_size = np.atleast_1d(filter_size)
        if len(filter_size) == 1:
            filter_size = np.repeat(filter_size, 2)
        self.filter_size = filter_size
        self.filter_threshold = filter_threshold
        self.edge_method = edge_method

        self.sigma_clip = sigma_clip
        bkg_estimator.sigma_clip = None
        bkgrms_estimator.sigma_clip = None
        self.bkg_estimator = bkg_estimator
        self.bkgrms_estimator = bkgrms_estimator

        self._prepare_data()
        self._calc_bkg_bkgrms()
        self._calc_coordinates()

    def _pad_data(self, xextra, yextra):
        """
        Pad the ``data`` and ``mask`` to have an integer number of
        background meshes of size ``box_size`` in both dimensions.  The
        padding is added in (approximately) equal amounts to all edges
        such that the original data is (approximately) centered in the
        padded array.

        Parameters
        ----------
        xextra, yextra : int
            The modulus of the data size and the box size in both the
            ``x`` and ``y`` dimensions.  This is the number of extra
            pixels beyond a multiple of the box size in the ``x`` and
            ``y`` dimensions.

        Returns
        -------
        result : `~numpy.ma.MaskedArray`
            The padded data and mask as a masked array.
        """

        ypad = 0
        xpad = 0
        if yextra > 0:
            ypad = self.box_size[0] - yextra
        if xextra > 0:
            xpad = self.box_size[1] - xextra

        # pad all edges in approximately equal amounts
        hy0 = ypad // 2
        hy1 = ypad - hy0
        hx0 = xpad // 2
        hx1 = xpad - hx0
        pad_width = ((hy0, hy1), (hx0, hx1))
        self.data_origin = (-hy0, -hx0)

        # mode must be a string for numpy < 0.11
        # (see https://github.com/numpy/numpy/issues/7112)
        mode = str('constant')
        data = np.pad(self.data, pad_width, mode=mode,
                      constant_values=[1.e10])

        # mask the padded regions
        pad_mask = np.zeros_like(data)
        pad_mask[:pad_width[0][0], :] = True
        pad_mask[-pad_width[0][1]:, :] = True
        pad_mask[:, :pad_width[1][0]] = True
        pad_mask[:, -pad_width[1][1]:] = True

        # pad the input mask separately (there is no np.ma.pad function)
        if self.mask is not None:
            mask = np.pad(self.mask, pad_width, mode=mode,
                          constant_values=[False])
            mask = np.logical_or(mask, pad_mask)
        else:
            mask = pad_mask

        return np.ma.masked_array(data, mask=mask)

    def _crop_data(self):
        """
        Crop the ``data`` and ``mask`` to have an integer number of
        background meshes of size ``box_size`` in both dimensions.  The
        data are cropped in (approximately) equal amounts along all
        edges.

        Returns
        -------
        result : `~numpy.ma.MaskedArray`
            The cropped data and mask as a masked array.
        """

        ny_crop = self.nyboxes * self.box_size[1]
        nx_crop = self.nxboxes * self.box_size[0]
        hy0 = (self.data.shape[0] - ny_crop) // 2
        hy1 = hy0 + ny_crop
        hx0 = (self.data.shape[1] - nx_crop) // 2
        hx1 = hx0 + nx_crop
        self.data_origin = (hy0, hx0)

        crop_slc = index_exp[hy0:hy1, hx0:hx1]
        if self.mask is not None:
            mask = self.mask[crop_slc]
        else:
            mask = False

        return np.ma.masked_array(self.data[crop_slc], mask=mask)

    def _select_meshes(self, data):
        """
        Define the x and y indices with respect to the low-resolution
        mesh image of the meshes to use for the background
        interpolation.

        The ``remove_mesh`` and ``remove_mesh_percentile`` keywords
        determines which meshes are not used for the background
        interpolation.

        Parameters
        ----------
        data : 2D `~numpy.ma.MaskedArray`
            A 2D array where the y dimension represents each mesh and
            the x dimension represents the data in each mesh.

        Returns
        -------
        mesh_idx : 1D `~numpy.ndarray`
            The 1D mesh indices.
        """

        # the number of masked pixels in each mesh
        nmasked = np.ma.count_masked(data, axis=1)

        if self.remove_mesh == 'any':
            # keep meshes that do not have any masked pixels
            mesh_idx = np.where(nmasked == 0)[0]
            if len(mesh_idx) == 0:
                raise ValueError('All meshes contain at least one masked '
                                 'pixel.  Please check your data or try '
                                 'an alternate remove_mesh option.')

        elif self.remove_mesh == 'all':
            # keep meshes that are not completely masked
            mesh_idx = np.where((self.box_npixels - nmasked) != 0)[0]
            if len(mesh_idx) == 0:
                raise ValueError('All meshes are completely masked.  '
                                 'Please check your data or try an '
                                 'alternate remove_mesh option.')

        elif self.remove_mesh == 'threshold':
            # keep meshes only with at least ``remove_mesh_percentile``
            # unmasked pixels
            threshold_npixels = self.remove_mesh_percentile * self.box_npixels
            mesh_idx = np.where((self.box_npixels - nmasked) >=
                                threshold_npixels)[0]
            if len(mesh_idx) == 0:
                raise ValueError('There are no valid meshes available with '
                                 'at least remove_mesh_percentile ({0} '
                                 'percent) unmasked pixels.'
                                 .format(threshold_npixels))

        else:
            raise ValueError('remove_mesh must be "any", "all", or '
                             '"threshold".')

        return mesh_idx

    def _prepare_data(self):
        """
        Prepare the data.

        First, pad or crop the 2D data array so that there are an
        integer number of meshes in both dimensions, creating a masked
        array.

        Then reshape into a different 2D masked array where each row
        represents the data in a single mesh.  This method also performs
        a first cut at rejecting certain meshes as specified by the
        input keywords.
        """

        self.nyboxes = self.data.shape[0] // self.box_size[0]
        self.nxboxes = self.data.shape[1] // self.box_size[1]
        yextra = self.data.shape[0] % self.box_size[0]
        xextra = self.data.shape[1] % self.box_size[1]

        if (xextra + yextra) == 0:
            # no resizing of the data is necessary
            data_ma = np.ma.masked_array(self.data, mask=self.mask)
            self.data_origin = (0, 0)
        else:
            # pad or crop the data
            if self.edge_method == 'pad':
                data_ma = self._pad_data(yextra, xextra)
                self.nyboxes += 1
                self.nxboxes += 1
            elif self.edge_method == 'crop':
                data_ma = self._crop_data()
            else:
                raise ValueError('edge_method must be "pad" or "crop"')

        # a reshaped 2D array with mesh data along the x axis
        mesh_data = np.ma.swapaxes(data_ma.reshape(
            self.nyboxes, self.box_size[0], self.nxboxes, self.box_size[1]),
            1, 2).reshape(self.nyboxes * self.nxboxes, self.box_npixels)

        # first cut on rejecting meshes
        self.mesh_idx = self._select_meshes(mesh_data)
        self.mesh_data = mesh_data[self.mesh_idx, :]

        return

    def _make_2d_array(self, data):
        """
        Convert a 1D array of mesh values to a masked 2D mesh array
        given the 1D mesh indices ``mesh_idx``.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            A 1D array of mesh values.

        Returns
        -------
        result : 2D `~numpy.ma.MaskedArray`
            A 2D masked array.  Pixels not defined in ``mesh_idx`` are
            masked.
        """

        if len(data) != len(self.mesh_idx):
            raise ValueError('data and mesh_idx must have the same length')

        data2d = np.zeros(self._mesh_shape)
        mask2d = np.ones(data2d.shape).astype(np.bool)
        data2d[self.mesh_yidx, self.mesh_xidx] = data
        mask2d[self.mesh_yidx, self.mesh_xidx] = False

        return np.ma.masked_array(data2d, mask=mask2d)

    def _interpolate_meshes(self, data, n_neighbors=8, eps=0., power=1.,
                            reg=0.):
        """
        Use IDW interpolation to fill in any masked pixels in the
        low-resolution 2D mesh background and background RMS images.

        This is required to use a regular-grid interpolator to expand
        the low-resolution image to the full size image.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            A 1D array of mesh values.

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
            weights.  See the Notes section for more details.

        reg : float, optional
            The regularization parameter. It may be used to control the
            smoothness of the interpolator. See the Notes section for
            more details.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            A 2D array of the mesh values where masked pixels have been
            filled by IDW interpolation.
       """

        yx = np.column_stack([self.mesh_yidx, self.mesh_xidx])
        coords = np.array(list(product(range(self.nyboxes),
                                       range(self.nxboxes))))
        f = ShepardIDWInterpolator(yx, data)
        img1d = f(coords, n_neighbors=n_neighbors, power=power, eps=eps,
                  reg=reg)

        return img1d.reshape(self._mesh_shape)

    def _selective_filter(self, data, indices):
        """
        Selectively filter only pixels above ``filter_threshold`` in the
        background mesh.

        The same pixels are filtered in both the background and
        background RMS meshes.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            A 2D array of mesh values.

        indices : 2 tuple of int
            A tuple of the ``y`` and ``x`` indices of the pixels to
            filter.

        Returns
        -------
        filtered_data : 2D `~numpy.ndarray`
            The filtered 2D array of mesh values.
        """

        data_out = np.copy(data)
        for i, j in zip(*indices):
            yfs, xfs = self.filter_size
            hyfs, hxfs = yfs // 2, xfs // 2
            y0, y1 = max(i - hyfs, 0), min(i - hyfs + yfs, data.shape[0])
            x0, x1 = max(j - hxfs, 0), min(j - hxfs + xfs, data.shape[1])
            data_out[i, j] = np.median(data[y0:y1, x0:x1])

        return data_out

    def _filter_meshes(self):
        """
        Apply a 2D median filter to the low-resolution 2D mesh,
        including only pixels inside the image at the borders.
        """

        from scipy.ndimage import generic_filter
        try:
            nanmedian_func = np.nanmedian    # numpy >= 1.9
        except AttributeError:
            from scipy.stats import nanmedian
            nanmedian_func = nanmedian

        if self.filter_threshold is None:
            # filter the entire arrays
            self.background_mesh = generic_filter(
                self.background_mesh, nanmedian_func, size=self.filter_size,
                mode='constant', cval=np.nan)
            self.background_rms_mesh = generic_filter(
                self.background_rms_mesh, nanmedian_func,
                size=self.filter_size, mode='constant', cval=np.nan)
        else:
            # selectively filter
            indices = np.nonzero(self.background_mesh > self.filter_threshold)
            self.background_mesh = self._selective_filter(
                self.background_mesh, indices)
            self.background_rms_mesh = self._selective_filter(
                self.background_rms_mesh, indices)

        return

    def _calc_bkg_bkgrms(self):
        """
        Calculate the background and background RMS estimate in each of
        the meshes.

        Calculate 2D arrays containing the background and background RMS
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

        data_sigclip = self.sigma_clip(self.mesh_data, axis=1)

        # final cut on rejecting meshes
        if self.remove_mesh == 'threshold':
            idx = self._select_meshes(data_sigclip)
            data_sigclip = data_sigclip[idx, :]
            self.mesh_idx = self.mesh_idx[idx]

        self._mesh_shape = (self.nyboxes, self.nxboxes)
        self.mesh_yidx, self.mesh_xidx = np.unravel_index(self.mesh_idx,
                                                          self._mesh_shape)

        bkg1d = self.bkg_estimator(data_sigclip, axis=1)
        bkgrms1d = self.bkgrms_estimator(data_sigclip, axis=1)

        # make the 2D mesh arrays
        if len(bkg1d) == (self.nxboxes * self.nyboxes):
            bkg = self._make_2d_array(bkg1d)
            bkgrms = self._make_2d_array(bkgrms1d)
        else:
            bkg = self._interpolate_meshes(bkg1d)
            bkgrms = self._interpolate_meshes(bkgrms1d)

        self.background_mesh = bkg
        self.background_rms_mesh = bkgrms

        # filter the 2D mesh arrays
        if not np.array_equal(self.filter_size, [1, 1]):
            self._filter_meshes()

        return

    def _calc_coordinates(self):
        """
        Calculate the coordinates to use when calling an interpolator.

        These are needed for `Background2D` and `BackgroundIDW2D`.
        """

        # the position coordinates used to initialize an interpolation
        self.y = (self.mesh_yidx * self.box_size[0] +
                  (self.box_size[0] - 1) / 2.) + self.data_origin[1]
        self.x = (self.mesh_xidx * self.box_size[1] +
                  (self.box_size[1] - 1) / 2.) + self.data_origin[0]
        self.yx = np.column_stack([self.y, self.x])

        # the position coordinates used when calling an interpolator
        nx, ny = self.data.shape
        self.data_coords = np.array(list(product(range(ny), range(nx))))

    @lazyproperty
    def background_median(self):
        """
        The median value of the 2D low-resolution background map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) Background: <value>").
        """

        return np.median(self.background_mesh)

    @lazyproperty
    def background_rms_median(self):
        """
        The median value of the low-resolution background RMS map.

        This is equivalent to the value `SExtractor`_ prints to stdout
        (i.e., "(M+D) RMS: <value>").
        """

        return np.median(self.background_rms_mesh)

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
    Class to estimate a 2D background and background RMS noise in an
    image.

    The background and background RMS are estimated using sigma-clipped
    statistics in each mesh of a grid that covers the input ``data`` to
    create a low-resolution background map.

    The exact method used to estimate the background in each mesh can be
    set with the ``method`` parameter.  The background RMS in each mesh
    is estimated by the sigma-clipped standard deviation.

    This class generates the full-sized background and background RMS
    maps from the lower-resolution maps using bicubic spline
    interpolation.

    Parameters
    ----------
    data : array_like
        The 2D array from which to estimate the background and/or
        background RMS map.

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
        low-resolution background and background RMS maps.  The value
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

    remove_mesh : {'threshold', 'all', 'any'}, optional
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
        background and background RMS.  This parameter is used only if
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
        self.background = self._resize_mesh(self.background_mesh)
        self.background_rms = self._resize_mesh(self.background_rms_mesh)

    def _resize_mesh(self, mesh):
        if np.ptp(mesh) == 0:
            return np.zeros_like(self.data) + np.min(mesh)
        else:
            from scipy.ndimage import zoom
            if self.edge_method == 'pad' and self.pad_crop:
                # matches photutils <= 0.2 Background class
                zoom_factor = (int(self.nyboxes * self.box_size[0] /
                                   mesh.shape[0]),
                               int(self.nxboxes * self.box_size[1] /
                                   mesh.shape[1]))
                result = zoom(mesh, zoom_factor, order=self.interp_order,
                              mode='reflect')
                return result[self.data_slc]
            else:
                zoom_factor = (float(self.data.shape[0] / mesh.shape[0]),
                               float(self.data.shape[1] / mesh.shape[1]))
                return zoom(mesh, zoom_factor, order=self.interp_order,
                            mode='reflect')


class BackgroundIDW2D(BackgroundBase2D):
    """
    Class to estimate a 2D background and background RMS noise in an
    image.

    The background and background RMS are estimated using sigma-clipped
    statistics in each mesh of a grid that covers the input ``data`` to
    create a low-resolution background map.

    The exact method used to estimate the background in each mesh can be
    set with the ``method`` parameter.  The background RMS in each mesh
    is estimated by the sigma-clipped standard deviation.

    This class generates the full-sized background and background RMS
    maps from the lower-resolution maps using inverse-distance weighting
    (IDW) interpolation.

    Parameters
    ----------
    data : array_like
        The 2D array from which to estimate the background and/or
        background RMS map.

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

    remove_mesh : {'threshold', 'all', 'any'}, optional
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
        background and background RMS.  This parameter is used only if
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
