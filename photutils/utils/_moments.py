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
    # Ignore divide-by-zero floating-point errors
    with np.errstate(all='ignore'):
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
    # Ignore divide-by-zero floating-point errors
    with np.errstate(all='ignore'):
        mu_norm = (moments_central
                   / moments_central[:, 0, 0][:, np.newaxis, np.newaxis])
    covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                      mu_norm[:, 1, 1], mu_norm[:, 2, 0]]).swapaxes(0, 1)
    return covar.reshape((covar.shape[0], 2, 2))


def covariance_determinant(covariance):
    """
    Compute the determinant of each ``(2, 2)`` covariance matrix.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    Returns
    -------
    determinant : `~numpy.ndarray`
        The ``(N,)`` determinants. Matrices with NaN elements give NaN.
    """
    # Ignore floating-point errors from NaN values in the covariance
    with np.errstate(all='ignore'):
        return np.linalg.det(covariance)


def covariance_min_eigval(covariance, *, determinant=None):
    """
    Compute the smaller eigenvalue of each symmetric ``(2, 2)``
    covariance matrix.

    The closed form ``lambda = tr/2 - sqrt((tr/2)**2 - det)`` is used.
    The discriminant ``((lambda1 - lambda2) / 2)**2`` is non-negative
    for a real symmetric matrix, so tiny negative rounding is clipped to
    zero.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    determinant : `~numpy.ndarray` or `None`, optional
        The precomputed ``(N,)`` determinants. If `None`, they are
        computed from ``covariance``.

    Returns
    -------
    min_eigval : `~numpy.ndarray`
        The ``(N,)`` smaller eigenvalues (the minor-axis variances).
        Matrices with NaN elements give NaN.
    """
    if determinant is None:
        determinant = covariance_determinant(covariance)
    # Ignore floating-point errors from NaN values in the covariance
    with np.errstate(all='ignore'):
        half_trace = 0.5 * (covariance[:, 0, 0] + covariance[:, 1, 1])
        disc = np.maximum(half_trace**2 - determinant, 0.0)
        return half_trace - np.sqrt(disc)


def is_singular_covariance(covariance, *, include_degenerate):
    """
    Return a mask of sources whose raw covariance matrix is singular or
    nearly singular.

    A source is flagged when the determinant of its raw (unregularized)
    covariance matrix is less than ``PIXEL_VARIANCE**2``. This is the
    isotropic point-like case where both axes are unresolved. It is also
    true for a negative determinant.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` raw covariance matrices.

    include_degenerate : bool
        If `True`, also flag sources whose minor-axis variance (the
        smaller eigenvalue) is less than ``PIXEL_VARIANCE``. This
        catches rank-1 degenerate sources that are unresolved along only
        one axis, which the determinant test alone misses.

    Returns
    -------
    mask : `~numpy.ndarray`
        The ``(N,)`` boolean mask. Sources with a NaN determinant are
        never flagged. If ``include_degenerate`` is `True`, sources
        with any non-finite determinant or minor-axis variance are
        never flagged.
    """
    determinant = covariance_determinant(covariance)
    point_like = determinant < PIXEL_VARIANCE**2
    if not include_degenerate:
        return point_like

    min_eigval = covariance_min_eigval(covariance, determinant=determinant)
    finite = np.isfinite(determinant) & np.isfinite(min_eigval)
    return finite & (point_like | (min_eigval < PIXEL_VARIANCE))


def regularize_covariance(covariance):
    """
    Regularize the raw covariance matrices of undefined and "infinitely"
    thin sources.

    A valid covariance is positive semidefinite (determinant and trace
    both non-negative). Any matrix that is not (e.g., from net-negative
    flux weighting) has an undefined shape and is set to NaN.

    Sources whose determinant is less than ``PIXEL_VARIANCE**2`` then
    have ``PIXEL_VARIANCE`` added to each diagonal element. A single
    bump is sufficient. For a positive semidefinite matrix the bumped
    determinant exceeds the raw determinant by ``PIXEL_VARIANCE`` times
    the trace plus ``PIXEL_VARIANCE**2``. Since the raw determinant and
    trace are both non-negative, the result is at least
    ``PIXEL_VARIANCE**2``, so it clears the threshold in one step.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` raw covariance matrices. This array is not
        modified.

    Returns
    -------
    regularized : `~numpy.ndarray`
        A new ``(N, 2, 2)`` array of regularized covariance matrices.
    """
    covar = covariance.copy()
    covar_det = covariance_determinant(covar)
    # Ignore floating-point errors from NaN values in the covariance
    with np.errstate(all='ignore'):
        covar_trace = covar[:, 0, 0] + covar[:, 1, 1]
        bad = (covar_det < 0) | (covar_trace < 0)
        covar[bad] = np.nan

        idx = np.where(covar_det < PIXEL_VARIANCE**2)[0]
        covar[idx, 0, 0] += PIXEL_VARIANCE
        covar[idx, 1, 1] += PIXEL_VARIANCE
    return covar


def eigvals_from_covariance(covariance):
    """
    Compute the two eigenvalues of each covariance matrix in decreasing
    order.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    Returns
    -------
    eigvals : `~numpy.ndarray`
        The ``(N, 2)`` eigenvalues, largest first. Both eigenvalues are
        NaN for a matrix with any non-finite element or with a negative
        eigenvalue (a matrix that is not positive semidefinite).
    """
    eigvals = np.full((covariance.shape[0], 2), np.nan)

    # The np.linalg.eigvalsh function requires that every element of a
    # covariance matrix be finite, so select only the wholly finite
    # matrices.
    idx = np.flatnonzero(np.isfinite(covariance).all(axis=(1, 2)))
    eigvals[idx] = np.linalg.eigvalsh(covariance[idx])

    # Check for negative variance (in case a covariance matrix is not
    # positive semidefinite).
    idx2 = np.unique(np.where(eigvals < 0)[0])
    eigvals[idx2] = (np.nan, np.nan)

    # Sort each eigenvalue pair in descending order (eigvalsh returns
    # values in ascending order).
    return np.fliplr(eigvals)


def orientation_from_covariance(covariance):
    """
    Compute the angle between the ``x`` axis and the major axis of the
    2D Gaussian function described by each covariance matrix.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` pixel covariance matrices.

    Returns
    -------
    orientation : `~numpy.ndarray`
        The ``(N,)`` orientation angles in degrees. The angle increases
        in the counter-clockwise direction and is in the range (-90,
        90].
    """
    # Ignore floating-point errors from non-finite values in the
    # covariance (e.g., the difference of two infinite variances)
    with np.errstate(all='ignore'):
        orient_radians = 0.5 * np.arctan2(2.0 * covariance[:, 0, 1],
                                          (covariance[:, 0, 0]
                                           - covariance[:, 1, 1]))
    return np.rad2deg(orient_radians)


def pixel_to_sky_covariance(wcs, pixel_covariance, xycen):
    """
    Transport pixel covariance matrices to the local tangent plane.

    The pixel covariance of each source is mapped to the local tangent
    plane with the forward WCS Jacobian ``F`` evaluated at the source
    position (``sky_covariance = F pixel_covariance F^T``).

    Parameters
    ----------
    wcs : WCS object
        A WCS object that implements the `astropy shared interface for
        WCS <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_.

    pixel_covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` pixel covariance matrices.

    xycen : `~numpy.ndarray`
        The ``(N, 2)`` pixel ``(x, y)`` centroid positions at which to
        evaluate the WCS Jacobians.

    Returns
    -------
    sky_covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` tangent-plane covariance matrices in
        arcsec**2, with rows and columns ordered ``(East, North)``.
        Matrices are NaN where the position or covariance is not finite.
    """
    sky_covariance = np.full(pixel_covariance.shape, np.nan)
    good = (np.all(np.isfinite(xycen), axis=1)
            & np.all(np.isfinite(pixel_covariance), axis=(1, 2)))
    if np.any(good):
        jac = compute_pixel_to_sky_jacobians(wcs, xycen[good, 0],
                                             xycen[good, 1])
        # The Einstein summation computes the matrix product of the
        # Jacobian, the pixel covariance, and the Jacobian transpose
        # for each source. The result is the sky covariance matrix in
        # the local tangent plane.
        sky_covariance[good] = np.einsum('nij,njk,nlk->nil', jac,
                                         pixel_covariance[good], jac)

    return sky_covariance


def sky_orientation_from_covariance(sky_covariance):
    """
    Compute the sky position angle of the major axis from tangent-plane
    covariance matrices.

    Parameters
    ----------
    sky_covariance : `~numpy.ndarray`
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
    orient_radians = 0.5 * np.arctan2(2.0 * sky_covariance[:, 0, 1],
                                      (sky_covariance[:, 1, 1]
                                       - sky_covariance[:, 0, 0]))
    return np.rad2deg(orient_radians)
