# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for upsampling images for Background2D using interpolation.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.units import Quantity
from astropy.utils import minversion
from astropy.utils.decorators import deprecated
from scipy.ndimage import affine_transform, spline_filter, zoom

from photutils.utils import ShepardIDWInterpolator
from photutils.utils._repr import make_repr

__all__ = ['BkgIDWInterpolator', 'BkgZoomInterpolator']

# scipy < 1.16 emits a UserWarning on the 1-D matrix code path of
# affine_transform used by the threaded zoom. Suppressing the warning
# with warnings.catch_warnings is not thread-safe, so the threaded
# zoom requires scipy 1.16 or later.
SCIPY_GE_1_16 = minversion('scipy', '1.16')


class _BkgZoomInterpolator:
    """
    Class to generate a full-sized background and background RMS images
    from lower-resolution mesh images using the `~scipy.ndimage.zoom`
    (spline) interpolator.

    This class must be used in concert with the `Background2D` class.

    Parameters
    ----------
    order : int, optional
        The order of the spline interpolation used to resize the
        low-resolution background and background RMS mesh images. The
        value must be an integer in the range 0-5. The default is 3
        (bicubic interpolation).

    mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode. Default is 'reflect'.

    cval : float, optional
        The value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0.

    clip : bool, optional
        Whether to clip the output to the range of values in the
        input image. This is enabled by default, since higher order
        interpolation may produce values outside the given input range.

    Notes
    -----
    When resizing the mesh to the full image size, the samples are
    considered as the centers of regularly-spaced grid elements (i.e.,
    `~scipy.ndimage.zoom` ``grid_mode`` is True). This makes
    zoom's behavior consistent with `scipy.ndimage.map_coordinates` and
    `skimage.transform.resize`

    When called with an ``n_threads`` keyword larger than 1 (e.g., by
    `~photutils.background.Background2D`), the ``mode`` is 'reflect'
    or 'mirror', and scipy 1.16 or later is installed, the output
    is computed concurrently over bands of rows. The multithreaded
    result is identical to the single `~scipy.ndimage.zoom` call up to
    floating-point rounding.
    """

    def __init__(self, *, order=3, mode='reflect', cval=0.0, clip=True):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.clip = clip

    def __repr__(self):
        params = ('order', 'mode', 'cval', 'clip')
        return make_repr(self, params)

    def _threaded_zoom(self, data, zoom_factor, n_threads):
        """
        Compute `~scipy.ndimage.zoom` (``grid_mode=True``) of a 2D array
        using multiple threads over bands of output rows.

        The spline prefilter is applied once to the (small) input mesh
        exactly as `~scipy.ndimage.zoom` does. The output resampling,
        which dominates the cost, is then evaluated concurrently
        over row bands with `~scipy.ndimage.affine_transform` (which
        releases the GIL and, for a diagonal transform, uses the
        same fast separable code path as `~scipy.ndimage.zoom`)
        using the same coordinate mapping as `~scipy.ndimage.zoom`
        with ``grid_mode=True``. For the 'reflect' and 'mirror'
        boundary modes, the result is identical to the single
        `~scipy.ndimage.zoom` call up to floating-point rounding. Other
        boundary modes have ``grid_mode``-specific edge handling that
        `~scipy.ndimage.affine_transform` does not reproduce, so the
        caller must not use this method for them.

        scipy < 1.16 emits a UserWarning on the 1-D matrix code path of
        `~scipy.ndimage.affine_transform` used here, and suppressing it
        with `warnings.catch_warnings` is not thread-safe, so the caller
        must also require scipy 1.16 or later.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        zoom_factor : tuple of int
            The integer zoom factor along each axis.

        n_threads : int
            The number of threads to use.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized 2D array, with the same dtype as the input.
        """
        out_shape = (data.shape[0] * zoom_factor[0],
                     data.shape[1] * zoom_factor[1])

        # The same coordinate mapping as scipy.ndimage.zoom with
        # grid_mode=True: input_coord = output_coord * ratio + offset
        zoom_ratio = np.array(data.shape) / np.array(out_shape)
        offset = 0.5 * zoom_ratio - 0.5

        if self.order > 1:
            filtered = spline_filter(data, self.order, output=np.float64,
                                     mode=self.mode)
        else:
            filtered = data

        result = np.empty(out_shape, dtype=data.dtype)

        def resample_band(y0, y1):
            band_offset = (y0 * zoom_ratio[0] + offset[0], offset[1])
            affine_transform(filtered, zoom_ratio, offset=band_offset,
                             output_shape=(y1 - y0, out_shape[1]),
                             output=result[y0:y1], order=self.order,
                             mode=self.mode, cval=self.cval,
                             prefilter=False)

        n_bands = min(n_threads, out_shape[0])
        band_edges = np.linspace(0, out_shape[0], n_bands + 1).astype(int)
        with ThreadPoolExecutor(max_workers=n_bands) as executor:
            list(executor.map(resample_band, band_edges[:-1],
                              band_edges[1:]))

        return result

    def __call__(self, data, **kwargs):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        **kwargs : dict
            Additional keyword arguments passed to the interpolator,
            including an optional ``n_threads`` value (see the class
            Notes).

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.

        Notes
        -----
        If ``data`` is an `~astropy.units.Quantity`, units are stripped
        before interpolation. Unit re-assignment is the caller's
        responsibility.
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        # The mesh is first resized to the larger padded-data size
        # (i.e., zoom_factor should be an integer) and then cropped
        # back to the final data size.
        zoom_factor = kwargs['box_size']
        n_threads = kwargs.get('n_threads', 1)
        if (n_threads > 1 and SCIPY_GE_1_16
                and self.mode in ('reflect', 'mirror')):
            result = self._threaded_zoom(data, zoom_factor, n_threads)
        else:
            result = zoom(data, zoom_factor, order=self.order,
                          mode=self.mode, cval=self.cval, grid_mode=True)
        result = result[0:kwargs['shape'][0], 0:kwargs['shape'][1]]

        if self.clip:
            minval = np.min(data)
            maxval = np.max(data)
            np.clip(result, minval, maxval, out=result)  # clip in place

        return result


@deprecated(since='3.0', message=('BkgZoomInterpolator is deprecated and will '
                                  'be removed in version 4.0.'))
class BkgZoomInterpolator(_BkgZoomInterpolator):
    """
    Class to generate a full-sized background and background RMS images
    from lower-resolution mesh images using the `~scipy.ndimage.zoom`
    (spline) interpolator.

    This class must be used in concert with the `Background2D` class.

    Parameters
    ----------
    order : int, optional
        The order of the spline interpolation used to resize the
        low-resolution background and background RMS mesh images. The
        value must be an integer in the range 0-5. The default is 3
        (bicubic interpolation).

    mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode. Default is 'reflect'.

    cval : float, optional
        The value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0.

    clip : bool, optional
        Whether to clip the output to the range of values in the
        input image. This is enabled by default, since higher order
        interpolation may produce values outside the given input range.

    Notes
    -----
    When resizing the mesh to the full image size, the samples are
    considered as the centers of regularly-spaced grid elements (i.e.,
    `~scipy.ndimage.zoom` ``grid_mode`` is True). This makes
    zoom's behavior consistent with `scipy.ndimage.map_coordinates` and
    `skimage.transform.resize`
    """

    def __init__(self, *, order=3, mode='reflect', cval=0.0, clip=True):
        super().__init__(order=order, mode=mode, cval=cval, clip=clip)


@deprecated(since='3.0', message=('BkgIDWInterpolator is deprecated and will '
                                  'be removed in version 4.0.'))
class BkgIDWInterpolator:
    """
    Class to generate a full-sized background and background RMS images
    from lower-resolution mesh images using inverse-distance weighting
    (IDW) interpolation (`~photutils.utils.ShepardIDWInterpolator`).

    This class must be used in concert with the `Background2D` class.

    Parameters
    ----------
    leafsize : float, optional
        The number of points at which the k-d tree algorithm switches
        over to brute-force. ``leafsize`` must be positive. See
        `scipy.spatial.cKDTree` for further information.

    n_neighbors : int, optional
        The maximum number of nearest neighbors to use during the
        interpolation.

    power : float, optional
        The power of the inverse distance used for the interpolation
        weights.

    regularization : float, optional
        The regularization parameter. It may be used to control the
        smoothness of the interpolator.
    """

    def __init__(self, *, leafsize=10, n_neighbors=10, power=1.0,
                 regularization=0.0):
        self.leafsize = leafsize
        self.n_neighbors = n_neighbors
        self.power = power
        self.regularization = regularization

    def __repr__(self):
        params = ('leafsize', 'n_neighbors', 'power', 'regularization')
        return make_repr(self, params)

    def __call__(self, data, **kwargs):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The low-resolution 2D mesh array.

        **kwargs : dict
            Additional keyword arguments passed to the interpolator.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The resized background or background RMS image.

        Notes
        -----
        If ``data`` is an `~astropy.units.Quantity`, units are stripped
        before interpolation. Unit re-assignment is the caller's
        responsibility.
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        # Create the interpolator from only the good mesh points
        yxcen = np.column_stack(kwargs['mesh_yxcen'])
        good_idx = np.where(~kwargs['mesh_nan_mask'])
        data = data[good_idx]
        interp_func = ShepardIDWInterpolator(yxcen, data,
                                             leafsize=self.leafsize)

        # Define the position coordinates used when calling the
        # interpolator
        yi, xi = np.mgrid[0:kwargs['shape'][0], 0:kwargs['shape'][1]]
        yx_indices = np.column_stack((yi.ravel(), xi.ravel()))
        data = interp_func(yx_indices, n_neighbors=self.n_neighbors,
                           power=self.power,
                           regularization=self.regularization)

        return data.reshape(kwargs['shape'])
