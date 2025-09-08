# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define interpolator classes for Background2D.
"""

import numpy as np
from astropy.units import Quantity
from astropy.utils.decorators import deprecated_renamed_argument
from scipy.ndimage import zoom

from photutils.utils import ShepardIDWInterpolator
from photutils.utils._repr import make_repr

__all__ = ['BkgIDWInterpolator', 'BkgZoomInterpolator']


class BkgZoomInterpolator:
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

    grid_mode : bool, optional
        If `True` (default), the samples are considered as the centers
        of regularly-spaced grid elements. If `False`, the samples
        are treated as isolated points. For zooming 2D images,
        this keyword should be set to `True`, which makes zoom's
        behavior consistent with `scipy.ndimage.map_coordinates` and
        `skimage.transform.resize`. The `False` option is provided only
        for backwards-compatibility.

        .. deprecated:: 2.0.0
           When this keyword is removed, the behavior will be
           ``grid_mode=True``.
    """

    @deprecated_renamed_argument('grid_mode', None, '2.0.0')
    def __init__(self, *, order=3, mode='reflect', cval=0.0, clip=True,
                 grid_mode=True):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.grid_mode = grid_mode
        self.clip = clip

    def __repr__(self):
        params = ('order', 'mode', 'cval', 'clip', 'grid_mode')
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
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        if kwargs['edge_method'] == 'pad':
            # The mesh is first resized to the larger padded-data size
            # (i.e., zoom_factor should be an integer) and then cropped
            # back to the final data size.
            zoom_factor = kwargs['box_size']
            result = zoom(data, zoom_factor, order=self.order, mode=self.mode,
                          cval=self.cval, grid_mode=self.grid_mode)
            result = result[0:kwargs['shape'][0], 0:kwargs['shape'][1]]
        else:
            # The mesh is resized directly to the final data size.
            zoom_factor = np.array(kwargs['shape']) / data.shape
            result = zoom(data, zoom_factor, order=self.order, mode=self.mode,
                          cval=self.cval)

        if self.clip:
            minval = np.min(data)
            maxval = np.max(data)
            np.clip(result, minval, maxval, out=result)  # clip in place

        return result


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

    reg : float, optional
        The regularization parameter. It may be used to control the
        smoothness of the interpolator.
    """

    def __init__(self, *, leafsize=10, n_neighbors=10, power=1.0, reg=0.0):
        self.leafsize = leafsize
        self.n_neighbors = n_neighbors
        self.power = power
        self.reg = reg

    def __repr__(self):
        params = ('leafsize', 'n_neighbors', 'power', 'reg')
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
        """
        data = np.asanyarray(data)
        if isinstance(data, Quantity):
            data = data.value
        if np.ptp(data) == 0:
            return np.full(kwargs['shape'], np.min(data),
                           dtype=kwargs['dtype'])

        # we create the interpolator from only the good mesh points
        yxcen = np.column_stack(kwargs['mesh_yxcen'])
        good_idx = np.where(~kwargs['mesh_nan_mask'])
        data = data[good_idx]
        interp_func = ShepardIDWInterpolator(yxcen, data,
                                             leafsize=self.leafsize)

        # the position coordinates used when calling the interpolator
        yi, xi = np.mgrid[0:kwargs['shape'][0], 0:kwargs['shape'][1]]
        yx_indices = np.column_stack((yi.ravel(), xi.ravel()))
        data = interp_func(yx_indices, n_neighbors=self.n_neighbors,
                           power=self.power, reg=self.reg)

        return data.reshape(kwargs['shape'])
