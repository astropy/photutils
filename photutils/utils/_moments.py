# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating image moments and the centroid, covariance, and
shape quantities derived from them.

The functions that take a moments array expect a leading source axis
and the layout ``moments[:, i, j]`` equal to the sum of ``y**i * x**j``.
The ``(0, 2)`` element is therefore the second moment along ``x`` and
the ``(2, 0)`` element is the second moment along ``y``. They return
plain `~numpy.ndarray` values without units.
"""

import warnings

import numpy as np

from photutils.utils._wcs_helpers import compute_pixel_to_sky_jacobians

# The variance of a uniform distribution across a single pixel. This is
# the smallest second moment a resolved source can have given finite
# pixel size.
PIXEL_VARIANCE = 1.0 / 12.0


def image_moments(data, *, center=(0, 0), order=1):
    """
    Calculate the image moments up to the specified order.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    center : tuple of two floats or `None`, optional
        The ``(x, y)`` center position. If `None` it will be calculated
        as the "center of mass" of the input ``data``. The default is
        ``(0, 0)``, which gives the raw image moments.

    order : int, optional
        The maximum order of the moments to calculate.

    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The image moments.
    """
    data = np.asarray(data).astype(float)

    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if order < 0:
        msg = 'order must be non-negative'
        raise ValueError(msg)

    if center is None:
        from photutils.centroids import centroid_com

        center = centroid_com(data)

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]
    ypowers = (indices[0] - center[1]) ** np.arange(order + 1)
    xpowers = np.transpose(indices[1] - center[0]) ** np.arange(order + 1)

    return np.dot(np.dot(np.transpose(ypowers), data), xpowers)


def centroid_from_moments(moments):
    """
    Compute the center-of-mass centroid of each source from its raw
    moments.

    Parameters
    ----------
    moments : `~numpy.ndarray`
        The raw moments with shape ``(N, 2, 2)`` or larger in the
        trailing axes. Only the elements up to first order are used.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``(N, 2)`` centroids in ``(x, y)`` order, relative to the
        origin of the moments. Sources with zero total flux have
        non-finite values.
    """
    # Ignore divide-by-zero RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        y_centroid = moments[:, 1, 0] / moments[:, 0, 0]
        x_centroid = moments[:, 0, 1] / moments[:, 0, 0]
    return np.transpose((x_centroid, y_centroid))


def inertia_tensor_from_moments(moments_central):
    """
    Compute the inertia tensor of each source for rotation around its
    center of mass.

    Parameters
    ----------
    moments_central : `~numpy.ndarray`
        The central moments with shape ``(N, 3, 3)`` or larger in the
        trailing axes. Only the second-order elements are used.

    Returns
    -------
    tensor : `~numpy.ndarray`
        The ``(N, 2, 2)`` inertia tensors. They are not normalized by
        the total flux.
    """
    mu_02 = moments_central[:, 0, 2]
    mu_11 = -moments_central[:, 1, 1]
    mu_20 = moments_central[:, 2, 0]
    tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
    return tensor.reshape((tensor.shape[0], 2, 2))


def covariance_from_moments(moments_central):
    """
    Compute the raw covariance matrix of the 2D Gaussian function that
    has the same normalized second-order central moments as each source.

    Parameters
    ----------
    moments_central : `~numpy.ndarray`
        The central moments with shape ``(N, 3, 3)`` or larger in the
        trailing axes. Only the elements up to second order are used.

    Returns
    -------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices, before any
        regularization. Sources with zero total flux have non-finite
        elements.
    """
    # Ignore divide-by-zero RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu_norm = (moments_central
                   / moments_central[:, 0, 0][:, np.newaxis, np.newaxis])
    covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                      mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
    return covar.reshape((covar.shape[0], 2, 2))


def pixel_cov_to_sky_cov(wcs, pix_cov, xycen):
    """
    Transport pixel covariance matrices to the local tangent plane.

    The pixel covariance of each source is mapped to the local tangent
    plane with the forward WCS Jacobian ``F`` evaluated at the source
    position (``sky_cov = F pix_cov F^T``).

    Parameters
    ----------
    wcs : WCS object
        A WCS object that implements the `astropy shared interface for
        WCS <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_.

    pix_cov : `~numpy.ndarray`
        The ``(N, 2, 2)`` pixel covariance matrices.

    xycen : `~numpy.ndarray`
        The ``(N, 2)`` pixel ``(x, y)`` centroid positions at which to
        evaluate the WCS Jacobians.

    Returns
    -------
    sky_cov : `~numpy.ndarray`
        The ``(N, 2, 2)`` tangent-plane covariance matrices in
        arcsec**2, with rows and columns ordered ``(East, North)``.
        Matrices are NaN where the position or covariance is not finite.
    """
    sky_cov = np.full(pix_cov.shape, np.nan)
    good = (np.all(np.isfinite(xycen), axis=1)
            & np.all(np.isfinite(pix_cov), axis=(1, 2)))
    if np.any(good):
        jac = compute_pixel_to_sky_jacobians(wcs, xycen[good, 0],
                                             xycen[good, 1])
        # The Einstein summation computes the matrix product of the
        # Jacobian, the pixel covariance, and the Jacobian transpose
        # for each source. The result is the sky covariance matrix in
        # the local tangent plane.
        sky_cov[good] = np.einsum('nij,njk,nlk->nil', jac, pix_cov[good], jac)

    return sky_cov


def sky_orientation_from_cov(sky_cov):
    """
    Compute the sky position angle of the major axis from tangent-plane
    covariance matrices.

    Parameters
    ----------
    sky_cov : `~numpy.ndarray`
        The ``(N, 2, 2)`` tangent-plane covariance matrices, with rows
        and columns ordered ``(East, North)``.

    Returns
    -------
    sky_orientation : `~numpy.ndarray`
        The ``(N,)`` position angles in degrees, measured from North
        toward East and in the range (-90, 90].
    """
    # The tangent-plane axes are ordered (East, North). Measuring the
    # angle from North toward East makes North play the role of the x
    # axis and East the role of the y axis in the pixel orientation
    # formula.
    orient_radians = 0.5 * np.arctan2(2.0 * sky_cov[:, 0, 1],
                                      sky_cov[:, 1, 1] - sky_cov[:, 0, 0])
    return np.rad2deg(orient_radians)
