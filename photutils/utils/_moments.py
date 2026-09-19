# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating image moments and the centroid, covariance, and
shape quantities derived from them.

`image_moments` computes the moments of a single 2D image. The
``*_from_moments`` functions take moments arrays with a leading source
axis and the layout ``moments[:, i, j]`` equal to the sum of ``y**i *
x**j``. The ``(0, 2)`` element is therefore the second moment along
``x`` and the ``(2, 0)`` element is the second moment along ``y``. The
covariance functions take ``(N, 2, 2)`` covariance matrices, and
`major_axis_angle` takes their ``(N,)`` elements.

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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    # covariance and from the overflow of huge variances
    with np.errstate(over='ignore', invalid='ignore'):
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
    # Ignore floating-point errors from non-finite values in the
    # covariance and from the unused branch of the np.where call
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        var_x = covariance[:, 0, 0]
        var_y = covariance[:, 1, 1]
        half_trace = 0.5 * (var_x + var_y)
        root = np.hypot(0.5 * (var_x - var_y), covariance[:, 0, 1])
        return np.where(half_trace > 0, determinant / (half_trace + root),
                        half_trace - root)


def covariance_max_eigval(covariance):
    """
    Compute the larger eigenvalue of each symmetric ``(2, 2)``
    covariance matrix.

    The closed form ``tr/2 + root`` is used (see
    `covariance_min_eigval`). It has no cancellation and does not use
    the determinant, which can overflow for huge variances.

    Parameters
    ----------
    covariance : `~numpy.ndarray`
        The ``(N, 2, 2)`` covariance matrices.

    Returns
    -------
    max_eigval : `~numpy.ndarray`
        The ``(N,)`` larger eigenvalues (the major-axis variances).
        Matrices with NaN elements give NaN.
    """
    # Ignore floating-point errors from non-finite values in the
    # covariance
    with np.errstate(over='ignore', invalid='ignore'):
        var_x = covariance[:, 0, 0]
        var_y = covariance[:, 1, 1]
        return (0.5 * (var_x + var_y)
                + np.hypot(0.5 * (var_x - var_y), covariance[:, 0, 1]))


def is_invalid_covariance(covariance, *, determinant):
    """
    Return a mask of sources whose raw covariance matrix is not positive
    semidefinite.

    A valid covariance is positive semidefinite (determinant and trace
    both non-negative). A matrix that is not (e.g., from net-negative
    flux weighting) does not describe a shape. The determinant of an
    exactly thin source (a rank-1 matrix) is zero, but its computed
    value can be slightly negative from rounding. A determinant is
    therefore treated as negative only if it is below ``-PSD_RTOL``
    times the squared trace.

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
        The ``(N,)`` boolean mask. Matrices with NaN elements are never
        flagged. A matrix with an infinite element is flagged only if
        its trace is negative or if its determinant is negative infinity
        while its trace is finite.
    """
    # Ignore floating-point errors from non-finite values in the
    # covariance and from the overflow of huge variances
    with np.errstate(over='ignore', invalid='ignore'):
        covar_trace = covariance[:, 0, 0] + covariance[:, 1, 1]
        return ((determinant < -PSD_RTOL * covar_trace**2)
                | (covar_trace < 0))


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
    symmetric matrix, because a determinant below that value requires an
    eigenvalue below ``PIXEL_VARIANCE``. A covariance matrix that is not
    positive semidefinite (see `is_invalid_covariance`) is not flagged.
    It does not describe a shape, so it is set to NaN instead of being
    regularized. The flagged sources are therefore exactly the ones that
    `regularize_covariance` modifies and keeps finite.

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
    invalid = is_invalid_covariance(covariance, determinant=determinant)
    return (min_eigval < PIXEL_VARIANCE) & ~invalid


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
        The ``(N, 2, 2)`` symmetric matrices. The array must have a
        floating-point dtype. This array is not modified.

    minimum : float or `~numpy.ndarray`
        The minimum eigenvalue, either a scalar or an ``(N,)`` array
        with one finite value per matrix.

    Returns
    -------
    floored : `~numpy.ndarray`
        A new ``(N, 2, 2)`` array. Matrices whose eigenvalues all reach
        the minimum are returned unchanged, as are matrices with any
        non-finite element.
    """
    floored = covariance.copy()

    # Eigenvalues of each symmetric 2x2 matrix in closed form, which
    # avoids the per-matrix overhead of a general solver. The smaller
    # eigenvalue comes from `covariance_min_eigval`, which does not
    # lose precision for a highly elongated matrix. Its determinant can
    # overflow for huge variances. That gives a smaller eigenvalue that
    # is infinite or NaN, which compares false below, so the matrix is
    # returned unchanged. The same holds for a matrix with a non-finite
    # element.
    determinant = covariance_determinant(covariance)
    eig_min = covariance_min_eigval(covariance, determinant=determinant)
    eig_max = covariance_max_eigval(covariance)
    minimum = np.broadcast_to(minimum, eig_min.shape)

    # A matrix with both eigenvalues below the minimum becomes exactly
    # isotropic.
    idx = np.flatnonzero(eig_max < minimum)
    floored[idx] = minimum[idx, np.newaxis, np.newaxis] * np.eye(2)

    # Otherwise raise only the smaller eigenvalue. The projector onto the
    # minor axis is (eig_max * I - C) / (eig_max - eig_min), so adding
    # (minimum - eig_min) times the projector changes that eigenvalue
    # and keeps both eigenvectors. The three unique elements are updated
    # directly, so the result is exactly symmetric.
    idx = np.flatnonzero((eig_min < minimum) & (eig_max >= minimum))
    eig_max = eig_max[idx]
    # Ignore floating-point errors from the overflow of a huge matrix
    # that is not positive semidefinite, whose eigenvalue difference can
    # exceed the largest finite value
    with np.errstate(over='ignore', invalid='ignore'):
        factor = (minimum[idx] - eig_min[idx]) / (eig_max - eig_min[idx])
        covar_xy = factor * covariance[idx, 0, 1]
        floored[idx, 0, 0] += factor * (eig_max - covariance[idx, 0, 0])
        floored[idx, 1, 1] += factor * (eig_max - covariance[idx, 1, 1])
        floored[idx, 0, 1] -= covar_xy
        floored[idx, 1, 0] -= covar_xy
    return floored


def regularize_covariance(covariance, *, determinant):
    """
    Regularize the raw covariance matrices of undefined and unresolved
    sources.

    A matrix that is not positive semidefinite (see
    `is_invalid_covariance`) has an undefined shape and is set to NaN.

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
    # Matrices with a non-finite element are not changed by the floor
    covar = floor_covariance_eigvals(covariance, minimum=PIXEL_VARIANCE)
    invalid = is_invalid_covariance(covariance, determinant=determinant)
    covar[invalid] = np.nan
    return covar


def eigvals_from_covariance(covariance):
    """
    Compute the two eigenvalues of each covariance matrix in decreasing
    order.

    The input is expected to be regularized (see
    `regularize_covariance`). The smaller eigenvalue of a raw matrix
    that is exactly singular can be slightly negative from rounding,
    which gives NaN eigenvalues.

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
    # Eigenvalues of each symmetric 2x2 matrix in closed form, which
    # avoids the per-matrix overhead of a general solver. The
    # determinant form of the smaller eigenvalue is accurate for a
    # highly elongated matrix, but the determinant overflows for huge
    # variances. In that case the determinant is divided by the larger
    # eigenvalue one factor at a time, which cannot overflow.
    determinant = covariance_determinant(covariance)
    eig_min = covariance_min_eigval(covariance, determinant=determinant)
    eig_max = covariance_max_eigval(covariance)
    # Ignore floating-point errors from non-finite values in the
    # covariance and from a zero larger eigenvalue
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        covar_xy = covariance[:, 0, 1]
        scaled = (covariance[:, 0, 0] * (covariance[:, 1, 1] / eig_max)
                  - covar_xy * (covar_xy / eig_max))
    eig_min = np.where(np.isfinite(determinant), eig_min, scaled)
    eigvals = np.transpose((eig_max, eig_min))

    # A non-finite element gives a NaN or infinite eigenvalue, and a
    # negative variance means that the matrix is not positive
    # semidefinite
    good = np.isfinite(eigvals).all(axis=1) & (eig_min >= 0)
    eigvals[~good] = np.nan
    return eigvals


def major_axis_angle(var_a, var_b, covar_ab):
    """
    Compute the angle from the ``a`` axis toward the ``b`` axis of the
    major axis of a 2D Gaussian function.

    Parameters
    ----------
    var_a, var_b : `~numpy.ndarray`
        The ``(N,)`` variances along the two axes.

    covar_ab : `~numpy.ndarray`
        The ``(N,)`` covariances of the two axes.

    Returns
    -------
    angle : `~numpy.ndarray`
        The ``(N,)`` angles in degrees, in the range (-90, 90]. The
        angle is zero for an isotropic matrix and NaN where the
        difference of the variances is undefined.
    """
    # Ignore floating-point errors from non-finite values (e.g., the
    # difference of two infinite variances)
    with np.errstate(over='ignore', invalid='ignore'):
        angle = 0.5 * np.arctan2(2.0 * covar_ab, var_a - var_b)
    return np.rad2deg(angle)


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
    return major_axis_angle(covariance[:, 0, 0], covariance[:, 1, 1],
                            covariance[:, 0, 1])


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
    # angle from North toward East makes North play the role of the
    # first axis and East the role of the second axis.
    return major_axis_angle(sky_covariance[:, 1, 1], sky_covariance[:, 0, 0],
                            sky_covariance[:, 0, 1])
