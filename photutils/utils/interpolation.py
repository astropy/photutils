# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['ShepardIDWInterpolator', 'interpolate_masked_data',
           'mask_to_mirrored_num']

__doctest_requires__ = {('ShepardIDWInterpolator'): ['scipy']}


class ShepardIDWInterpolator(object):
    """
    Class to perform Inverse Distance Weighted (IDW) interpolation.

    This interpolator uses a modified version of `Shepard's method
    <http://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ (see the
    Notes section for details).

    Parameters
    ----------
    coordinates : float, 1D array-like, or NxM-array-like
        Coordinates of the known data points. In general, it is expected
        that these coordinates are in a form of a NxM-like array where N
        is the number of points and M is dimension of the coordinate
        space. When M=1 (1D space), then the ``coordinates`` parameter
        may be entered as a 1D array or, if only one data point is
        available, ``coordinates`` can be a scalar number representing
        the 1D coordinate of the data point.

        .. note::
            If the dimensionality of ``coordinates`` is larger than 2,
            e.g., if it is of the form N1 x N2 x N3 x ... x Nn x M, then
            it will be flattened to form an array of size NxM where N =
            N1 * N2 * ... * Nn.

    values : float or 1D array-like
        Values of the data points corresponding to each coordinate
        provided in ``coordinates``. In general a 1D array is expected.
        When a single data point is available, then ``values`` can be a
        scalar number.

        .. note::
            If the dimensionality of ``values`` is larger than 1 then it
            will be flattened.

    weights : float or 1D array-like, optional
        Weights to be associated with each data value. These weights, if
        provided, will be combined with inverse distance weights (see
        the Notes section for details). When ``weights`` is `None`
        (default), then only inverse distance weights will be used. When
        provided, this input parameter must have the same form as
        ``values``.

    leafsize : float, optional
        The number of points at which the k-d tree algorithm switches
        over to brute-force. ``leafsize`` must be positive.  See
        `scipy.spatial.cKDTree` for further information.

    Notes
    -----
    This interpolator uses a slightly modified version of `Shepard's
    method <http://en.wikipedia.org/wiki/Inverse_distance_weighting>`_.
    The essential difference is the introduction of a "regularization"
    parameter (``reg``) that is used when computing the inverse distance
    weights:

    .. math::
        w_i = 1 / (d(x, x_i)^{power} + r)

    By supplying a positive regularization parameter one can avoid
    singularities at the locations of the data points as well as control
    the "smoothness" of the interpolation (e.g., make the weights of the
    neighbors less varied). The "smoothness" of interpolation can also
    be controlled by the power parameter (``power``).

    Examples
    --------
    This class can can be instantiated using the following syntax::

        >>> from photutils.utils import ShepardIDWInterpolator as idw

    Example of interpolating 1D data::

        >>> import numpy as np
        >>> np.random.seed(123)
        >>> x = np.random.random(100)
        >>> y = np.sin(x)
        >>> f = idw(x, y)
        >>> f(0.4)    # doctest: +FLOAT_CMP
        0.38862424043228855
        >>> np.sin(0.4)   # doctest: +FLOAT_CMP
        0.38941834230865052

        >>> xi = np.random.random(4)
        >>> xi
        array([ 0.51312815,  0.66662455,  0.10590849,  0.13089495])
        >>> f(xi)    # doctest: +FLOAT_CMP
        array([ 0.49086423,  0.62647862,  0.1056854 ,  0.13048335])
        >>> np.sin(xi)
        array([ 0.49090493,  0.6183367 ,  0.10571061,  0.13052149])

    NOTE: In the last example, ``xi`` may be a ``Nx1`` array instead of
    a 1D vector.

    Example of interpolating 2D data::

        >>> pos = np.random.rand(1000, 2)
        >>> val = np.sin(pos[:, 0] + pos[:, 1])
        >>> f = idw(pos, val)
        >>> f([0.5, 0.6])     # doctest: +FLOAT_CMP
        0.89312649587405657
        >>> np.sin(0.5 + 0.6)
        0.89120736006143542
    """

    def __init__(self, coordinates, values, weights=None, leafsize=10):
        from scipy.spatial import cKDTree

        coordinates = np.atleast_2d(coordinates)
        if coordinates.shape[0] == 1:
            coordinates = np.transpose(coordinates)
        if coordinates.ndim != 2:
            coordinates = np.reshape(coordinates, (-1, coordinates.shape[-1]))

        values = np.asanyarray(values).ravel()

        ncoords = coordinates.shape[0]
        if ncoords < 1:
            raise ValueError('You must enter at least one data point.')

        if values.shape[0] != ncoords:
            raise ValueError('The number of values must match the number '
                             'of coordinates.')

        if weights is not None:
            weights = np.asanyarray(weights).ravel()
            if weights.shape[0] != ncoords:
                raise ValueError('The number of weights must match the '
                                 'number of coordinates.')
            if np.any(weights < 0.0):
                raise ValueError('All weight values must be non-negative '
                                 'numbers.')

        self.coordinates = coordinates
        self.ncoords = ncoords
        self.coords_ndim = coordinates.shape[1]
        self.values = values
        self.weights = weights
        self.kdtree = cKDTree(coordinates, leafsize=leafsize)

    def __call__(self, positions, n_neighbors=8, eps=0.0, power=1.0, reg=0.0,
                 conf_dist=1e-12, dtype=np.float):

        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : float, 1D array-like, or NxM-array-like
            Coordinates of the position(s) at which the interpolator
            should be evaluated. In general, it is expected that these
            coordinates are in a form of a NxM-like array where N is the
            number of points and M is dimension of the coordinate space.
            When M=1 (1D space), then the ``positions`` parameter may be
            input as a 1D-like array or, if only one data point is
            available, ``positions`` can be a scalar number representing
            the 1D coordinate of the data point.

            .. note::
                If the dimensionality of the ``positions`` argument is
                larger than 2, e.g., if it is of the form N1 x N2 x N3 x
                ... x Nn x M, then it will be flattened to form an array
                of size NxM where N = N1 * N2 * ... * Nn.

            .. warning::
                The dimensionality of ``positions`` must match the
                dimensionality of the ``coordinates`` used during the
                initialization of the interpolator.

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

        conf_dist : float, optional
            The confusion distance below which the interpolator should
            use the value of the closest data point instead of
            attempting to interpolate. This is used to avoid
            singularities at the known data points, especially if
            ``reg`` is 0.0.

        dtype : data-type
            The data type of the output interpolated values. If `None`
            then the type will be inferred from the type of the
            ``values`` parameter used during the initialization of the
            interpolator.
        """

        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')

        if conf_dist is not None and conf_dist <= 0.0:
            conf_dist = None

        positions = np.asanyarray(positions)
        if positions.ndim == 0:
            # assume we have a single 1D coordinate
            if self.coords_ndim != 1:
                raise ValueError('The dimensionality of the input position '
                                 'does not match the dimensionality of the '
                                 'coordinates used to initialize the '
                                 'interpolator.')
        elif positions.ndim == 1:
            # assume we have a single point
            if (self.coords_ndim != 1 and
                    (positions.shape[-1] != self.coords_ndim)):
                raise ValueError('The input position was provided as a 1D '
                                 'array, but its length does not match the '
                                 'dimensionality of the coordinates used '
                                 'to initialize the interpolator.')
        elif positions.ndim != 2:
            raise ValueError('The input positions must be an array-like '
                             'object of dimensionality no larger than 2.')

        positions = np.reshape(positions, (-1, self.coords_ndim))
        npositions = positions.shape[0]

        distances, idx = self.kdtree.query(positions, k=n_neighbors, eps=eps)

        if n_neighbors == 1:
            return self.values[idx]

        if dtype is None:
            dtype = self.values.dtype

        interp_values = np.zeros(npositions, dtype=dtype)
        for k in range(npositions):
            valid_idx = np.isfinite(distances[k])
            idk = idx[k][valid_idx]
            dk = distances[k][valid_idx]

            if dk.shape[0] == 0:
                interp_values[k] = np.nan
                continue

            if conf_dist is not None:
                # check if we are close to a known data point
                confused = (dk <= conf_dist)
                if np.any(confused):
                    interp_values[k] = self.values[idk[confused][0]]
                    continue

            w = 1.0 / ((dk ** power) + reg)
            if self.weights is not None:
                w *= self.weights[idk]

            wtot = np.sum(w)
            if wtot > 0.0:
                interp_values[k] = np.dot(w, self.values[idk]) / wtot
            else:
                interp_values[k] = np.nan

        if len(interp_values) == 1:
            return interp_values[0]
        else:
            return interp_values


def interpolate_masked_data(data, mask, error=None, background=None):
    """
    Interpolate over masked pixels in data and optional error or
    background images.

    The value of masked pixels are replaced by the mean value of the
    connected neighboring non-masked pixels.  This function is intended
    for single, isolated masked pixels (e.g. hot/warm pixels).

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The data array.

    mask : array_like (bool)
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    error : array_like or `~astropy.units.Quantity`, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` must have the same shape as ``data``.

    background : array_like, or `~astropy.units.Quantity`, optional
        The pixel-wise background level of the input ``data``.
        ``background`` must have the same shape as ``data``.

    Returns
    -------
    data : `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``data`` with interpolated masked pixels.

    error : `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``error`` with interpolated masked pixels.  `None` if
        input ``error`` is not input.

    background : `~numpy.ndarray` or `~astropy.units.Quantity`
        Input ``background`` with interpolated masked pixels.  `None` if
        input ``background`` is not input.
    """

    if data.shape != mask.shape:
        raise ValueError('data and mask must have the same shape')

    data_out = np.copy(data)    # do not alter input data
    mask_idx = mask.nonzero()

    if mask_idx[0].size == 0:
        raise ValueError('All items in data are masked')

    for x in zip(*mask_idx):
        X = np.array([[max(x[i] - 1, 0), min(x[i] + 1, data.shape[i] - 1)]
                      for i in range(len(data.shape))])
        goodpix = ~mask[X]

        if not np.any(goodpix):
            warnings.warn('The masked pixel at "{}" is completely '
                          'surrounded by (connected) masked pixels, '
                          'thus unable to interpolate'.format(x,),
                          AstropyUserWarning)
            continue

        data_out[x] = np.mean(data[X][goodpix])

        if background is not None:
            if background.shape != data.shape:
                raise ValueError('background and data must have the same '
                                 'shape')
            background_out = np.copy(background)
            background_out[x] = np.mean(background[X][goodpix])
        else:
            background_out = None

        if error is not None:
            if error.shape != data.shape:
                raise ValueError('error and data must have the same '
                                 'shape')
            error_out = np.copy(error)
            error_out[x] = np.sqrt(np.mean(error[X][goodpix]**2))
        else:
            error_out = None

    return data_out, error_out, background_out


def mask_to_mirrored_num(image, mask_image, center_position, bbox=None):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given ``center_position``.  If the mirror pixel is unavailable (i.e.
    itself masked or outside of the image), then the masked pixel value
    is set to zero.

    Parameters
    ----------
    image : `numpy.ndarray`, 2D
        The 2D array of the image.

    mask_image : array-like, bool
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        considered bad.

    center_position : 2-tuple
        (x, y) center coordinates around which masked pixels will be
        mirrored.

    bbox : list, tuple, `numpy.ndarray`, optional
        The bounding box (x_min, x_max, y_min, y_max) over which to
        replace masked pixels.

    Returns
    -------
    result : `numpy.ndarray`, 2D
        A 2D array with replaced masked pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.utils import mask_to_mirrored_num
    >>> image = np.arange(16).reshape(4, 4)
    >>> mask = np.zeros_like(image, dtype=bool)
    >>> mask[0, 0] = True
    >>> mask[1, 1] = True
    >>> mask_to_mirrored_num(image, mask, (1.5, 1.5))
    array([[15,  1,  2,  3],
           [ 4, 10,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    """

    if bbox is None:
        ny, nx = image.shape
        bbox = [0, nx, 0, ny]
    subdata = np.copy(image[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1])
    submask = mask_image[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1]
    y_masked, x_masked = np.nonzero(submask)
    x_mirror = (2 * (center_position[0] - bbox[0]) -
                x_masked + 0.5).astype('int32')
    y_mirror = (2 * (center_position[1] - bbox[2]) -
                y_masked + 0.5).astype('int32')

    # Reset mirrored pixels that go out of the image.
    outofimage = ((x_mirror < 0) | (y_mirror < 0) |
                  (x_mirror >= subdata.shape[1]) |
                  (y_mirror >= subdata.shape[0]))
    if outofimage.any():
        x_mirror[outofimage] = x_masked[outofimage].astype('int32')
        y_mirror[outofimage] = y_masked[outofimage].astype('int32')

    subdata[y_masked, x_masked] = subdata[y_mirror, x_mirror]

    # Set pixels that mirrored to another masked pixel to zero.
    # This will also set to zero any pixels that mirrored out of
    # the image.
    mirror_is_masked = submask[y_mirror, x_mirror]
    x_bad = x_masked[mirror_is_masked]
    y_bad = y_masked[mirror_is_masked]
    subdata[y_bad, x_bad] = 0.0

    outimage = np.copy(image)
    outimage[bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1] = subdata
    return outimage
