# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate the 2D background and
background RMS in a 2D image.
"""

from itertools import product

import numpy as np
from numpy.lib.index_tricks import index_exp
from astropy.utils import lazyproperty

from .core import SExtractorBackground, StdBackgroundRMS
from ..utils import ShepardIDWInterpolator

from astropy.version import version as astropy_version
if astropy_version < '3.1':
    from astropy.stats import SigmaClip
    SIGMA_CLIP = SigmaClip(sigma=3., iters=10)
else:
    from ..extern import SigmaClip
    SIGMA_CLIP = SigmaClip(sigma=3., maxiters=10)


__all__ = ['BkgZoomInterpolator', 'BkgIDWInterpolator', 'Background2D']

__doctest_requires__ = {('BkgZoomInterpolator', 'Background2D'): ['scipy']}


class BkgZoomInterpolator:
    """
    This class generates full-sized background and background RMS images
    from lower-resolution mesh images using the `~scipy.ndimage.zoom`
    (spline) interpolator.

    This class must be used in concert with the `Background2D` class.

    Parameters
    ----------
    order : int, optional
        The order of the spline interpolation used to resize the
        low-resolution background and background RMS mesh images.  The
        value must be an integer in the range 0-5.  The default is 3
        (bicubic interpolation).

    mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Default is 'reflect'.

    cval : float, optional
        The value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    """

    def __init__(self, order=3, mode='reflect', cval=0.0):
        self.order = order
        self.mode = mode
        self.cval = cval

    def __call__(self, mesh, bkg2d_obj):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        mesh : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        bkg2d_obj : `Background2D` object
            The `Background2D` object that prepared the ``mesh`` array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.
        """

        mesh = np.asanyarray(mesh)
        if np.ptp(mesh) == 0:
            return np.zeros_like(bkg2d_obj.data) + np.min(mesh)

        from scipy.ndimage import zoom

        if bkg2d_obj.edge_method == 'pad':
            # The mesh is first resized to the larger padded-data size
            # (i.e. zoom_factor should be an integer) and then cropped
            # back to the final data size.
            zoom_factor = (int(bkg2d_obj.nyboxes * bkg2d_obj.box_size[0] /
                               mesh.shape[0]),
                           int(bkg2d_obj.nxboxes * bkg2d_obj.box_size[1] /
                               mesh.shape[1]))
            result = zoom(mesh, zoom_factor, order=self.order, mode=self.mode,
                          cval=self.cval)

            return result[0:bkg2d_obj.data.shape[0],
                          0:bkg2d_obj.data.shape[1]]
        else:
            # The mesh is resized directly to the final data size.
            zoom_factor = (float(bkg2d_obj.data.shape[0] / mesh.shape[0]),
                           float(bkg2d_obj.data.shape[1] / mesh.shape[1]))

            return zoom(mesh, zoom_factor, order=self.order, mode=self.mode,
                        cval=self.cval)


class BkgIDWInterpolator:
    """
    This class generates full-sized background and background RMS images
    from lower-resolution mesh images using inverse-distance weighting
    (IDW) interpolation (`~photutils.utils.ShepardIDWInterpolator`).

    This class must be used in concert with the `Background2D` class.

    Parameters
    ----------
    leafsize : float, optional
        The number of points at which the k-d tree algorithm switches
        over to brute-force. ``leafsize`` must be positive.  See
        `scipy.spatial.cKDTree` for further information.

    n_neighbors : int, optional
        The maximum number of nearest neighbors to use during the
        interpolation.

    power : float, optional
        The power of the inverse distance used for the interpolation
        weights.

    reg : float, optional
        The regularization parameter. It may be used to control the
        smoothness of the interpolator.
    """

    def __init__(self, leafsize=10, n_neighbors=10, power=1.0, reg=0.0):
        self.leafsize = leafsize
        self.n_neighbors = n_neighbors
        self.power = power
        self.reg = reg

    def __call__(self, mesh, bkg2d_obj):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        mesh : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        bkg2d_obj : `Background2D` object
            The `Background2D` object that prepared the ``mesh`` array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.
        """

        mesh = np.asanyarray(mesh)
        if np.ptp(mesh) == 0:
            return np.zeros_like(bkg2d_obj.data) + np.min(mesh)

        mesh1d = mesh[bkg2d_obj.mesh_yidx, bkg2d_obj.mesh_xidx]
        f = ShepardIDWInterpolator(bkg2d_obj.yx, mesh1d,
                                   leafsize=self.leafsize)
        data = f(bkg2d_obj.data_coords, n_neighbors=self.n_neighbors,
                 power=self.power, reg=self.reg)

        return data.reshape(bkg2d_obj.data.shape)


class Background2D:
    """
    Class to estimate a 2D background and background RMS noise in an
    image.

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
        dimensions.  When this is not the case, see the ``edge_method``
        keyword for more options.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from calculations.

    exclude_percentile : float in the range of [0, 100], optional
        The percentage of masked pixels in a mesh, used as a threshold
        for determining if the mesh is excluded.  If a mesh has more
        than ``exclude_percentile`` percent of its pixels masked then it
        will be excluded from the low-resolution map.  Masked pixels
        include those from the input ``mask``, those resulting from the
        data padding (i.e. if ``edge_method='pad'``), and those
        resulting from any sigma clipping (i.e. if ``sigma_clip`` is
        used).  Setting ``exclude_percentile=0`` will exclude meshes
        that have any masked pixels.  Setting ``exclude_percentile=100``
        will only exclude meshes that are completely masked.  Note that
        completely masked meshes are *always* excluded.  For best
        results, ``exclude_percentile`` should be kept as low as
        possible (as long as there are sufficient pixels for reasonable
        statistical estimates).  The default is 10.

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

    edge_method : {'pad', 'crop'}, optional
        The method used to determine how to handle the case where the
        image size is not an integer multiple of the ``box_size`` in
        either dimension.  Both options will resize the image to give an
        exact multiple of ``box_size`` in both dimensions.

        * ``'pad'``: pad the image along the top and/or right edges.
          This is the default and recommended method.
        * ``'crop'``: crop the image along the top and/or right edges.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=10``.

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

    interpolator : callable, optional
        A callable object (a function or object) used to interpolate the
        low-resolution background or background RMS mesh to the
        full-size background or background RMS maps.  The default is an
        instance of `BkgZoomInterpolator`.

    Notes
    -----
    If there is only one background mesh element (i.e., ``box_size`` is
    the same size as the ``data``), then the background map will simply
    be a constant image.
    """

    def __init__(self, data, box_size, mask=None,
                 exclude_percentile=10., filter_size=(3, 3),
                 filter_threshold=None, edge_method='pad',
                 sigma_clip=SIGMA_CLIP,
                 bkg_estimator=SExtractorBackground(sigma_clip=None),
                 bkgrms_estimator=StdBackgroundRMS(sigma_clip=None),
                 interpolator=BkgZoomInterpolator()):

        data = np.asanyarray(data)

        box_size = np.atleast_1d(box_size)
        if len(box_size) == 1:
            box_size = np.repeat(box_size, 2)
        self.box_size = (min(box_size[0], data.shape[0]),
                         min(box_size[1], data.shape[1]))
        self.box_npixels = self.box_size[0] * self.box_size[1]

        if mask is not None:
            mask = np.asanyarray(mask)
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape')

        if exclude_percentile < 0 or exclude_percentile > 100:
            raise ValueError('exclude_percentile must be between 0 and 100 '
                             '(inclusive).')

        self.data = data
        self.mask = mask
        self.exclude_percentile = exclude_percentile

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
        self.interpolator = interpolator

        self._prepare_data()
        self._calc_bkg_bkgrms()
        self._calc_coordinates()

    def _pad_data(self, yextra, xextra):
        """
        Pad the ``data`` and ``mask`` to have an integer number of
        background meshes of size ``box_size`` in both dimensions.  The
        padding is added on the top and/or right edges (this is the best
        option for the "zoom" interpolator).

        Parameters
        ----------
        yextra, xextra : int
            The modulus of the data size and the box size in both the
            ``y`` and ``x`` dimensions.  This is the number of extra
            pixels beyond a multiple of the box size in the ``y`` and
            ``x`` dimensions.

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
        pad_width = ((0, ypad), (0, xpad))

        # mode must be a string for numpy < 0.11
        # (see https://github.com/numpy/numpy/issues/7112)
        mode = str('constant')
        data = np.pad(self.data, pad_width, mode=mode,
                      constant_values=[1.e10])

        # mask the padded regions
        pad_mask = np.zeros_like(data)
        y0 = data.shape[0] - ypad
        x0 = data.shape[1] - xpad
        pad_mask[y0:, :] = True
        pad_mask[:, x0:] = True

        # pad the input mask separately (there is no np.ma.pad function)
        if self.mask is not None:
            mask = np.pad(self.mask, pad_width, mode=mode,
                          constant_values=[True])
            mask = np.logical_or(mask, pad_mask)
        else:
            mask = pad_mask

        return np.ma.masked_array(data, mask=mask)

    def _crop_data(self):
        """
        Crop the ``data`` and ``mask`` to have an integer number of
        background meshes of size ``box_size`` in both dimensions.  The
        data are cropped on the top and/or right edges (this is the best
        option for the "zoom" interpolator).

        Returns
        -------
        result : `~numpy.ma.MaskedArray`
            The cropped data and mask as a masked array.
        """

        ny_crop = self.nyboxes * self.box_size[1]
        nx_crop = self.nxboxes * self.box_size[0]
        crop_slc = index_exp[0:ny_crop, 0:nx_crop]
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

        The ``exclude_percentile`` keyword determines which meshes are
        not used for the background interpolation.

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

        # meshes that contain more than ``exclude_percentile`` percent
        # masked pixels are excluded:
        #   - for exclude_percentile=0, good meshes will be only where
        #     nmasked=0
        #   - meshes where nmasked=self.box_npixels are *always* excluded
        #     (second conditional needed for exclude_percentile=100)
        threshold_npixels = self.exclude_percentile / 100. * self.box_npixels
        mesh_idx = np.where((nmasked <= threshold_npixels) &
                            (nmasked != self.box_npixels))[0]    # good meshes

        if len(mesh_idx) == 0:
            raise ValueError('All meshes contain > {0} ({1} percent per '
                             'mesh) masked pixels.  Please check your data '
                             'or decrease "exclude_percentile".'
                             .format(threshold_npixels,
                                     self.exclude_percentile))

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
        else:
            # pad or crop the data
            if self.edge_method == 'pad':
                data_ma = self._pad_data(yextra, xextra)
                self.nyboxes = data_ma.shape[0] // self.box_size[0]
                self.nxboxes = data_ma.shape[1] // self.box_size[1]
            elif self.edge_method == 'crop':
                data_ma = self._crop_data()
            else:
                raise ValueError('edge_method must be "pad" or "crop"')

        self.nboxes = self.nxboxes * self.nyboxes

        # a reshaped 2D masked array with mesh data along the x axis
        mesh_data = np.ma.swapaxes(data_ma.reshape(
            self.nyboxes, self.box_size[0], self.nxboxes, self.box_size[1]),
            1, 2).reshape(self.nyboxes * self.nxboxes, self.box_npixels)

        # first cut on rejecting meshes
        self.mesh_idx = self._select_meshes(mesh_data)
        self._mesh_data = mesh_data[self.mesh_idx, :]

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

        if data.shape != self.mesh_idx.shape:
            raise ValueError('data and mesh_idx must have the same shape')

        if np.ma.is_masked(data):
            raise ValueError('data must not be a masked array')

        data2d = np.zeros(self._mesh_shape).astype(data.dtype)
        data2d[self.mesh_yidx, self.mesh_xidx] = data

        if len(self.mesh_idx) == self.nboxes:
            # no meshes were masked
            return data2d
        else:
            # some meshes were masked
            mask2d = np.ones(data2d.shape).astype(np.bool)
            mask2d[self.mesh_yidx, self.mesh_xidx] = False

            return np.ma.masked_array(data2d, mask=mask2d)

    def _interpolate_meshes(self, data, n_neighbors=10, eps=0., power=1.,
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
        except AttributeError:    # pragma: no cover
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

        Both meshes are computed at the same time here method because
        the filtering of both depends on the background mesh.

        The ``background_mesh`` and ``background_rms_mesh`` images are
        equivalent to the low-resolution "MINIBACKGROUND" and
        "MINIBACK_RMS" background maps in SExtractor, respectively.
        """

        if self.sigma_clip is not None:
            data_sigclip = self.sigma_clip(self._mesh_data, axis=1)
        else:
            data_sigclip = self._mesh_data
        del self._mesh_data

        # preform mesh rejection on sigma-clipped data (i.e. for any
        # newly-masked pixels)
        idx = self._select_meshes(data_sigclip)
        self.mesh_idx = self.mesh_idx[idx]    # indices for the output mesh
        self._data_sigclip = data_sigclip[idx]    # always a 2D masked array

        self._mesh_shape = (self.nyboxes, self.nxboxes)
        self.mesh_yidx, self.mesh_xidx = np.unravel_index(self.mesh_idx,
                                                          self._mesh_shape)

        # These properties are needed later to calculate
        # background_mesh_ma and background_rms_mesh_ma.  Note that _bkg1d
        # and _bkgrms1d are masked arrays, but the mask should always be
        # False.
        self._bkg1d = self.bkg_estimator(self._data_sigclip, axis=1)
        self._bkgrms1d = self.bkgrms_estimator(self._data_sigclip, axis=1)

        # make the unfiltered 2D mesh arrays (these are not masked)
        if len(self._bkg1d) == self.nboxes:
            bkg = self._make_2d_array(self._bkg1d)
            bkgrms = self._make_2d_array(self._bkgrms1d)
        else:
            bkg = self._interpolate_meshes(self._bkg1d)
            bkgrms = self._interpolate_meshes(self._bkgrms1d)

        self._background_mesh_unfiltered = bkg
        self._background_rms_mesh_unfiltered = bkgrms
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

        Regular-grid interpolators require a 2D array of values.  Some
        require a 2D meshgrid of x and y.  Other require a strictly
        increasing 1D array of the x and y ranges.
        """

        # the position coordinates used to initialize an interpolation
        self.y = (self.mesh_yidx * self.box_size[0] +
                  (self.box_size[0] - 1) / 2.)
        self.x = (self.mesh_xidx * self.box_size[1] +
                  (self.box_size[1] - 1) / 2.)
        self.yx = np.column_stack([self.y, self.x])

        # the position coordinates used when calling an interpolator
        nx, ny = self.data.shape
        self.data_coords = np.array(list(product(range(ny), range(nx))))

    @lazyproperty
    def mesh_nmasked(self):
        """
        A 2D (masked) array of the number of masked pixels in each mesh.
        Only meshes included in the background estimation are included.
        The array is masked only if meshes were excluded.
        """

        return self._make_2d_array(
            np.ma.count_masked(self._data_sigclip, axis=1))

    @lazyproperty
    def background_mesh_ma(self):
        """
        The background 2D (masked) array mesh prior to any
        interpolation.  The array is masked only if meshes were
        excluded.
        """

        if len(self._bkg1d) == self.nboxes:
            return self.background_mesh
        else:
            return self._make_2d_array(self._bkg1d)    # masked array

    @lazyproperty
    def background_rms_mesh_ma(self):
        """
        The background RMS 2D (masked) array mesh prior to any
        interpolation.  The array is masked only if meshes were
        excluded.
        """

        if len(self._bkgrms1d) == self.nboxes:
            return self.background_rms_mesh
        else:
            return self._make_2d_array(self._bkgrms1d)    # masked array

    @lazyproperty
    def background_median(self):
        """
        The median value of the 2D low-resolution background map.

        This is equivalent to the value SExtractor prints to stdout
        (i.e., "(M+D) Background: <value>").
        """

        return np.median(self.background_mesh)

    @lazyproperty
    def background_rms_median(self):
        """
        The median value of the low-resolution background RMS map.

        This is equivalent to the value SExtractor prints to stdout
        (i.e., "(M+D) RMS: <value>").
        """

        return np.median(self.background_rms_mesh)

    @lazyproperty
    def background(self):
        """A 2D `~numpy.ndarray` containing the background image."""

        return self.interpolator(self.background_mesh, self)

    @lazyproperty
    def background_rms(self):
        """A 2D `~numpy.ndarray` containing the background RMS image."""

        return self.interpolator(self.background_rms_mesh, self)

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
