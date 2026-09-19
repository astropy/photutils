# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating image moments and the centroid, covariance, and
shape quantities derived from them.

`image_moments` computes the moments of a single 2D image. The
``*_from_moments`` functions take moments arrays with a leading source
axis and the layout ``moments[:, i, j]`` equal to the sum of ``y**i *
x**j``. The ``(0, 2)`` element is therefore the second moment along
``x`` and the ``(2, 0)`` element is the second moment along ``y``. The
remaining functions take ``(N, 2, 2)`` covariance matrices.

Array inputs with a leading source axis must have a floating-point
dtype. They are not validated or converted. Every function returns a
plain `~numpy.ndarray` without units.
"""

import numpy as np

from photutils.utils._wcs_helpers import compute_pixel_to_sky_jacobians

# The variance of a uniform distribution across a single pixel. This is
# the smallest second moment a resolved source can have given finite
# pixel size.
PIXEL_VARIANCE = 1.0 / 12.0

# The relative tolerance of the positive semidefinite test of a
# covariance matrix. The determinant of a rank-1 matrix is computed with
# a rounding error of a few times the machine epsilon relative to its
# squared trace.
PSD_RTOL = 1000.0 * np.finfo(float).eps


def image_moments(data, *, center=(0, 0), order=1):
    """
    Calculate the image moments up to the specified order.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    center : tuple of two floats, optional
        The ``(x, y)`` center position. The default is ``(0, 0)``, which
        gives the raw image moments. Passing the centroid gives the
        central moments.

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

    The closed form ``a * d - b * c`` is used instead of
    `numpy.linalg.det`. The LAPACK pivot search does not handle NaN
    consistently across platforms, so a matrix with a NaN element can
    give a determinant of zero instead of NaN. The closed form always
    propagates NaN.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    Returns
    -------
    determinant : `~numpy.ndarray`
        The ``(N,)`` determinants. Matrices with NaN elements give NaN.
    """
    # Ignore floating-point errors from non-finite values in the
    # covariance
    with np.errstate(all='ignore'):
        return (covariance[:, 0, 0] * covariance[:, 1, 1]
                - covariance[:, 0, 1] * covariance[:, 1, 0])


def covariance_min_eigval(covariance, *, determinant):
    """
    Compute the smaller eigenvalue of each symmetric ``(2, 2)``
    covariance matrix.

    The smaller eigenvalue has two equivalent closed forms, ``tr/2 -
    root`` and ``det / (tr/2 + root)``, where ``root`` is half of the
    difference of the two eigenvalues. For a symmetric matrix ``[[a, b],
    [b, d]]`` it is ``hypot((a - d) / 2, b)``, which is never negative
    and does not overflow for large variances.

    The second form is used for a positive trace. The subtraction
    in the first form cancels catastrophically when the matrix is
    highly elongated. The first form is used otherwise, where it has no
    cancellation and the second form can divide by zero. The accuracy is
    limited by the accuracy of the input ``determinant``, which has its
    own cancellation for an elongated matrix with a large off-diagonal
    element.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    determinant : `~numpy.ndarray`
        The ``(N,)`` determinants of ``covariance`` (see
        `covariance_determinant`).

    Returns
    -------
    min_eigval : `~numpy.ndarray`
        The ``(N,)`` smaller eigenvalues (the minor-axis variances).
        Matrices with NaN elements give NaN.
    """
    # Ignore floating-point errors from NaN values in the covariance
    with np.errstate(all='ignore'):
        var_x = covariance[:, 0, 0]
        var_y = covariance[:, 1, 1]
        half_trace = 0.5 * (var_x + var_y)
        root = np.hypot(0.5 * (var_x - var_y), covariance[:, 0, 1])
        return np.where(half_trace > 0, determinant / (half_trace + root),
                        half_trace - root)


def is_singular_covariance(covariance, *, determinant):
    """
    Return a mask of sources whose raw covariance matrix is singular or
    nearly singular.

    A source is flagged when its minor-axis variance (the smaller
    eigenvalue of its raw, unregularized covariance matrix) is less than
    ``PIXEL_VARIANCE``, meaning that at least one axis is unresolved.
    This covers point-like sources, where both axes are unresolved,
    and thin sources that are unresolved along only one axis. A test
    of the determinant against ``PIXEL_VARIANCE**2`` is implied for a
    symmetric matrix, because a determinant below that value requires
    an eigenvalue below ``PIXEL_VARIANCE``. The mask is also true for a
    covariance matrix that is not positive semidefinite.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` raw covariance matrices.

    determinant : `~numpy.ndarray`
        The ``(N,)`` determinants of ``covariance`` (see
        `covariance_determinant`).

    Returns
    -------
    mask : `~numpy.ndarray`
        The ``(N,)`` boolean mask. Sources with a NaN minor-axis
        variance are never flagged.
    """
    min_eigval = covariance_min_eigval(covariance, determinant=determinant)
    return min_eigval < PIXEL_VARIANCE


def floor_covariance_eigvals(covariance, *, minimum):
    """
    Raise the eigenvalues of each symmetric ``(2, 2)`` matrix to at
    least a minimum value while keeping its eigenvectors.

    The variance along each principal axis is floored independently,
    so the orientation is unchanged and an axis whose variance already
    exceeds the minimum is not modified. If both eigenvalues are below
    the minimum, the result is exactly the minimum times the identity
    matrix, which has no preferred axis. The result is a continuous
    function of the input.

    A floored eigenvalue equals the minimum only to within the rounding
    error of the matrix elements, which is of order the machine epsilon
    times the larger eigenvalue.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` symmetric matrices. Every element must be
        finite, and the array must have a floating-point dtype. This
        array is not modified.

    minimum : float or `~numpy.ndarray`
        The minimum eigenvalue, either a scalar or an ``(N,)`` array
        with one finite value per matrix.

    Returns
    -------
    floored : `~numpy.ndarray`
        A new ``(N, 2, 2)`` array. Matrices whose eigenvalues all reach
        the minimum are returned unchanged.
    """
    floored = covariance.copy()

    # Eigenvalues of each symmetric 2x2 matrix in closed form, which
    # avoids the per-matrix overhead of a general solver. The smaller
    # eigenvalue comes from `covariance_min_eigval`, which does not
    # lose precision for a highly elongated matrix. The larger one does
    # not use the determinant, which can overflow for huge variances.
    # Ignore floating-point errors from such an overflow. It gives a
    # smaller eigenvalue that is infinite or NaN, which compares false
    # below, so the matrix is returned unchanged.
    with np.errstate(all='ignore'):
        var_x = covariance[:, 0, 0]
        var_y = covariance[:, 1, 1]
        determinant = covariance_determinant(covariance)
        eig_min = covariance_min_eigval(covariance,
                                        determinant=determinant)
        eig_max = (0.5 * (var_x + var_y)
                   + np.hypot(0.5 * (var_x - var_y), covariance[:, 0, 1]))
    minimum = np.broadcast_to(minimum, eig_min.shape)

    # A matrix with both eigenvalues below the minimum becomes exactly
    # isotropic.
    isotropic = eig_max < minimum
    floored[isotropic] = (minimum[isotropic, np.newaxis, np.newaxis]
                          * np.eye(2))

    # Otherwise raise only the smaller eigenvalue. The projector onto the
    # minor axis is (eig_max * I - C) / (eig_max - eig_min), so adding
    # (minimum - eig_min) times the projector changes that eigenvalue
    # and keeps both eigenvectors. The result is exactly symmetric.
    idx = np.flatnonzero((eig_min < minimum) & ~isotropic)
    factor = (minimum[idx] - eig_min[idx]) / (eig_max[idx] - eig_min[idx])
    minor_axis = (eig_max[idx, np.newaxis, np.newaxis] * np.eye(2)
                  - covariance[idx])
    floored[idx] += factor[:, np.newaxis, np.newaxis] * minor_axis
    return floored


def regularize_covariance(covariance, *, determinant):
    """
    Regularize the raw covariance matrices of undefined and unresolved
    sources.

    A valid covariance is positive semidefinite (determinant and trace
    both non-negative). Any matrix that is not (e.g., from net-negative
    flux weighting) has an undefined shape and is set to NaN. The
    determinant of an exactly thin source (a rank-1 matrix) is zero, but
    its computed value can be slightly negative from rounding. A
    determinant is therefore treated as negative only if it is below
    ``-PSD_RTOL`` times the squared trace.

    The remaining finite matrices have each eigenvalue (the variance
    along a principal axis) raised to at least ``PIXEL_VARIANCE``, the
    smallest second moment a source can have given finite pixel size.
    The eigenvectors are kept, so the orientation is unchanged and a
    resolved axis is not modified. Only the sources selected by
    `is_singular_covariance` have an eigenvalue below that value, so
    every other matrix is returned unchanged. The regularized matrix is
    a continuous function of the raw matrix.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` raw covariance matrices. This array is not
        modified. It must have a floating-point dtype because invalid
        matrices are set to NaN.

    determinant : `~numpy.ndarray`
        The ``(N,)`` determinants of ``covariance`` (see
        `covariance_determinant`).

    Returns
    -------
    regularized : `~numpy.ndarray`
        A new ``(N, 2, 2)`` array of regularized covariance matrices.
    """
    covar = covariance.copy()
    # Ignore floating-point errors from NaN values in the covariance
    with np.errstate(all='ignore'):
        covar_trace = covar[:, 0, 0] + covar[:, 1, 1]
        bad = ((determinant < -PSD_RTOL * covar_trace**2)
               | (covar_trace < 0))
    covar[bad] = np.nan

    idx = np.flatnonzero(np.isfinite(covar).all(axis=(1, 2)))
    covar[idx] = floor_covariance_eigvals(covar[idx],
                                          minimum=PIXEL_VARIANCE)
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
    # The `sky_orientation_from_covariance` function applies the same
    # formula with the roles of the two axes swapped. Ignore
    # floating-point errors from non-finite values in the covariance
    # (e.g., the difference of two infinite variances).
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
    # formula (see `orientation_from_covariance`). Ignore floating-point
    # errors from non-finite values in the covariance.
    with np.errstate(all='ignore'):
        orient_radians = 0.5 * np.arctan2(2.0 * sky_covariance[:, 0, 1],
                                          (sky_covariance[:, 1, 1]
                                           - sky_covariance[:, 0, 0]))
    return np.rad2deg(orient_radians)
