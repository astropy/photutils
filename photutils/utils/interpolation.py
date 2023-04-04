# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for interpolating data.
"""

import numpy as np

__all__ = ['ShepardIDWInterpolator']

__doctest_requires__ = {('ShepardIDWInterpolator'): ['scipy']}


class ShepardIDWInterpolator:
    """
    Class to perform Inverse Distance Weighted (IDW) interpolation.

    This interpolator uses a modified version of `Shepard's method
    <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ (see
    the Notes section for details).

    Parameters
    ----------
    coordinates : float, 1D array_like, or NxM array_like
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

    values : float or 1D array_like
        Values of the data points corresponding to each coordinate
        provided in ``coordinates``. In general a 1D array is expected.
        When a single data point is available, then ``values`` can be a
        scalar number.

        .. note::
            If the dimensionality of ``values`` is larger than 1 then it
            will be flattened.

    weights : float or 1D array_like, optional
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
    method <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_.
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
        >>> rng = np.random.default_rng(0)
        >>> x = rng.random(100)  # 100 random values
        >>> y = np.sin(x)
        >>> f = idw(x, y)
        >>> f(0.4)  # doctest: +FLOAT_CMP
        0.38937843420912366
        >>> np.sin(0.4)  # doctest: +FLOAT_CMP
        0.3894183423086505

        >>> xi = rng.random(4)  # 4 random values
        >>> xi  # doctest: +FLOAT_CMP
        array([0.47998792, 0.23237292, 0.80188058, 0.92353016])
        >>> f(xi)  # doctest: +FLOAT_CMP
        array([0.46577097, 0.22837422, 0.71856662, 0.80125391])
        >>> np.sin(xi)  # doctest: +FLOAT_CMP
        array([0.46176846, 0.23028731, 0.71866503, 0.7977353 ])

    NOTE: In the last example, ``xi`` may be a ``Nx1`` array instead of
    a 1D vector.

    Example of interpolating 2D data::

        >>> rng = np.random.default_rng(0)
        >>> pos = rng.random((1000, 2))
        >>> val = np.sin(pos[:, 0] + pos[:, 1])
        >>> f = idw(pos, val)
        >>> f([0.5, 0.6])  # doctest: +FLOAT_CMP
        0.8948257014687874
        >>> np.sin(0.5 + 0.6)  # doctest: +FLOAT_CMP
        0.8912073600614354
    """

    def __init__(self, coordinates, values, weights=None, leafsize=10):
        from scipy.spatial import cKDTree

        coordinates = np.asarray(coordinates)
        if coordinates.ndim == 0:  # scalar coordinate
            coordinates = np.atleast_2d(coordinates)

        if coordinates.ndim == 1:
            coordinates = np.transpose(np.atleast_2d(coordinates))

        if coordinates.ndim > 2:
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
                 conf_dist=1.0e-12, dtype=float):
        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : float, 1D array_like, or NxM array_like
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

        dtype : data-type, optional
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
            if self.coords_ndim not in (1, positions.shape[-1]):
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

            w = 1.0 / ((dk**power) + reg)
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
