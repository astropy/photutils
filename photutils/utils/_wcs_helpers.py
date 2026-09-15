# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for WCS helpers.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, UnitSphericalRepresentation
from astropy.wcs.wcsapi import (high_level_objects_to_values,
                                values_to_high_level_objects)


def _is_gwcs(wcs):
    """
    Return True if the WCS looks like a gwcs object.

    A gwcs object is callable and carries a bounding box. Neither is
    true of `astropy.wcs.WCS`, which has no bounding box to bypass.
    """
    return callable(wcs) and hasattr(wcs, 'bounding_box')


def _pixel_to_world(wcs, x, y):
    """
    Convert pixel coordinates to a sky coordinate, ignoring any gwcs
    bounding box.

    The finite differences used here step half a pixel beyond the source
    position. For a source in the outermost pixel of the array those
    steps fall outside a gwcs bounding box, where the gwcs high-level
    API returns NaN. The forward transform is well defined there, so it
    is evaluated with the bounding box disabled.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    x, y : float or `~numpy.ndarray`
        The pixel coordinates.

    Returns
    -------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate(s).
    """
    if _is_gwcs(wcs):
        values = wcs(x, y, with_bounding_box=False)
        return values_to_high_level_objects(*values, low_level_wcs=wcs)[0]
    return wcs.pixel_to_world(x, y)


def _world_to_pixel(wcs, skycoord):
    """
    Convert a sky coordinate to pixel coordinates, ignoring any gwcs
    bounding box.

    See `_pixel_to_world` for why the bounding box is bypassed. The
    sky offsets used to find the direction of North can likewise fall
    outside the box for a source at the edge of the array.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate(s).

    Returns
    -------
    x, y : float or `~numpy.ndarray`
        The pixel coordinates.
    """
    if _is_gwcs(wcs):
        values = high_level_objects_to_values(skycoord, low_level_wcs=wcs)
        return wcs.invert(*values, with_bounding_box=False)
    return wcs.world_to_pixel(skycoord)


def _sky_to_pixel_jacobian(skycoord, wcs, *, pixcoord=None):
    """
    Compute the pixel center and the local Jacobian for a sky-to-pixel
    conversion.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixcoord : tuple of float, optional
        The ``(x, y)`` pixel position of ``skycoord``, if already known.
        When given, the WCS is not inverted to find it.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    jacobian : 2x2 `~numpy.ndarray`
        The Jacobian matrix ``d(pixel)/d(sky_arcsec)``.
    """
    if pixcoord is None:
        x0, y0 = _world_to_pixel(wcs, skycoord)
    else:
        x0, y0 = pixcoord
    center = (float(x0), float(y0))
    forward = compute_pixel_to_sky_jacobians(x0, y0, wcs)[0]
    return center, np.linalg.inv(forward)


def _svd_ellipse_from_composite(m_comp, *, width_col_idx=0, sky_angle=False,
                                input_circular=False):
    """
    Extract ellipse width, height, and angle from a composite matrix
    using SVD.

    Given a 2x2 matrix ``m_comp`` whose columns represent the mapped
    semi-axis vectors of an ellipse, perform SVD and return the full
    widths, heights, and rotation angle, preserving the width/height
    assignment of the input ellipse.

    Parameters
    ----------
    m_comp : 2x2 `~numpy.ndarray`
        The composite matrix whose SVD gives the output ellipse axes.

    width_col_idx : int, optional
        The column index (0 or 1) of ``m_comp`` that corresponds to the
        width semi-axis. Default is 0.

    sky_angle : bool, optional
        If True, the composite matrix columns are tangent-plane (``xi``
        = East, ``eta`` = North) vectors, so the returned angle is a sky
        position angle (PA) measured from North toward East. If False
        (the default), the columns are pixel (``x``, ``y``) vectors
        and the returned angle is measured counterclockwise from the
        positive x-axis.

    input_circular : bool, optional
        If True, the input ellipse is known to be circular (width ==
        height), so the SVD's principal axis is meaningless and the
        rotation angle is taken directly from the mapped width semi-
        axis. Default is False.

    Returns
    -------
    out_width : float
        The full width of the output ellipse.

    out_height : float
        The full height of the output ellipse.

    angle : `~astropy.coordinates.Angle`
        The rotation angle of the width axis, wrapped to [0, 360)
        degrees.
    """
    u_mat, s_vals, _vt = np.linalg.svd(m_comp)

    # SVD returns singular values in descending order, so s_vals[0]
    # corresponds to the major axis. Determine whether the major axis
    # corresponds to the width or height by checking alignment with the
    # mapped width semi-axis.
    width_col = m_comp[:, width_col_idx]
    if (np.abs(np.dot(u_mat[:, 0], width_col))
            >= np.abs(np.dot(u_mat[:, 1], width_col))):
        # Major axis aligns with width
        out_width = 2 * s_vals[0]
        out_height = 2 * s_vals[1]
        angle_col = u_mat[:, 0]
    else:
        # Major axis aligns with height, so swap
        out_width = 2 * s_vals[1]
        out_height = 2 * s_vals[0]
        angle_col = u_mat[:, 1]

    # Fix SVD sign ambiguity: ensure the angle direction aligns with the
    # mapped width semi-axis
    if np.dot(angle_col, width_col) < 0:
        angle_col = -angle_col

    # When the input ellipse is circular (width == height), the SVD's
    # principal direction has no physical meaning. Any rotation of a
    # circle yields an identical shape, so the SVD picks an arbitrary
    # principal axis derived from the Jacobian. To preserve the input
    # rotation angle, fall back to the mapped width semi-axis direction
    # (which carries the input theta through the Jacobian).
    if input_circular:
        width_norm = np.linalg.norm(width_col)
        if width_norm > 0:
            angle_col = width_col / width_norm

    # Compute the rotation angle
    if sky_angle:
        # Sky position angle (PA) measured from North (eta/Dec) toward
        # East. The composite-matrix columns are tangent-plane vectors
        # with components (xi=East, eta=North), so the PA is simply
        # arctan2(xi, eta). The local Jacobian (or its inverse) used to
        # build the composite matrix already encodes the WCS parity, so
        # no additional parity correction is needed here.
        angle = Angle(
            np.rad2deg(np.arctan2(angle_col[0],
                                  angle_col[1])) * u.deg,
        ).wrap_at(360 * u.deg)
    else:
        # Pixel angle: measured from +x toward +y
        angle = Angle(
            np.rad2deg(np.arctan2(angle_col[1],
                                  angle_col[0])) * u.deg,
        ).wrap_at(360 * u.deg)

    return out_width, out_height, angle


def _mean_singular_values(matrices):
    """
    Return the mean of the two singular values of 2x2 matrices.

    For a 2x2 matrix the sum of the singular values satisfies ``(s1 +
    s2)**2 = |M|_F**2 + 2 |det M|``, where ``|M|_F`` is the Frobenius
    norm, so the mean needs no singular value decomposition.

    Parameters
    ----------
    matrices : `~numpy.ndarray`
        An array of shape ``(..., 2, 2)``.

    Returns
    -------
    result : `~numpy.ndarray`
        The mean singular value of each matrix, with shape ``(...)``.
    """
    frobenius_sq = np.einsum('...ij,...ij->...', matrices, matrices)
    det = np.abs(np.linalg.det(matrices))
    return 0.5 * np.sqrt(frobenius_sq + 2.0 * det)


def compute_local_wcs_jacobian(skycoord, wcs):
    """
    Compute the local 2x2 Jacobian matrix d(pixel)/d(tangent-plane) at
    the given sky coordinate using central finite differences.

    The Jacobian matrix ``J`` linearizes the WCS transformation in the
    neighborhood of ``skycoord``. It maps infinitesimal offsets in the
    tangent-plane coordinate system (in arcsec) to pixel coordinate
    offsets (in pixels)::

        [dx, dy]^T ~ J @ [d_xi, d_eta]^T

    The tangent-plane coordinate system has two orthogonal axes:

        * ``xi`` (RA direction): offset along Right Ascension,
          increasing to the East.

        * ``eta`` (Dec direction): offset along Declination,
          increasing to the North.

    The Jacobian is computed by offsetting half a pixel either side of
    the position in x and y, converting the resulting pixel positions to
    sky coordinates, and differencing the tangent-plane displacements
    in arcsec. The (xi, eta) offsets are computed from the great-circle
    separation and position angle between the center and each offset
    point, so the formula is well-defined at the celestial poles and
    across the longitude wraparound (RA = 0 / 360). This gives the
    forward Jacobian ``F = d(sky_arcsec)/d(pixel)``, which is then
    inverted to obtain ``J = F^{-1} = d(pixel)/d(sky_arcsec)``. The
    central difference over one pixel is exact for a distortion that is
    locally quadratic, unlike a one-sided difference, which is biased by
    half the curvature. Using 1-pixel steps ensures numerical stability
    across all pixel scales.

    This function works with any WCS that supports
    the `astropy shared interface for WCS
    <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
    `astropy.wcs.WCS`, `gwcs.wcs.WCS`), because it relies only on the
    ``world_to_pixel`` and ``pixel_to_world`` methods.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate at which to evaluate the Jacobian.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    jacobian : 2x2 `~numpy.ndarray`
        The Jacobian matrix ``J`` such that ``[dx, dy]^T ≈ J @ [d_xi,
        d_eta]^T``, with units of pixels/arcsec.
    """
    # Reference pixel position
    x0, y0 = _world_to_pixel(wcs, skycoord)

    # Forward Jacobian F = d(sky_arcsec)/d(pixel), shape (2, 2).
    # Rows are (xi, eta), columns are (px_x, px_y).
    forward = compute_pixel_to_sky_jacobians(x0, y0, wcs)[0]

    # Invert to get J = d(pixel)/d(sky_arcsec)
    return np.linalg.inv(forward)


def compute_pixel_to_sky_jacobians(x, y, wcs):
    """
    Compute local forward WCS Jacobians for an array of pixel positions
    using central finite differences.

    Each 2x2 Jacobian ``F`` maps pixel-coordinate offsets to
    tangent-plane offsets in arcsec::

        [d_xi, d_eta]^T ~ F @ [dx, dy]^T

    where ``xi`` is the offset along East (the Right Ascension
    direction, as a great-circle angle) and ``eta`` is the offset along
    North (the Declination direction). Each derivative is the central
    difference between the sky positions half a pixel either side of the
    center, so it is unbiased where the distortion has curvature. The
    sky positions are handled as unit vectors and the differences are
    projected onto the local East and North directions, so the formula
    is well-defined at the celestial poles and across the longitude
    wraparound (RA = 0 / 360). All five positions per pixel are
    evaluated in a single WCS call.

    Parameters
    ----------
    x, y : float or array_like
        The pixel coordinates. Arrays are flattened, so the Jacobians
        are returned in the flattened order.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    jacobians : `~numpy.ndarray`
        The (N, 2, 2) array of forward Jacobians in arcsec / pixel, with
        rows ``(xi, eta)`` and columns ``(x, y)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        msg = 'x and y must have the same size'
        raise ValueError(msg)
    n = x.size

    # Evaluate the pixel centers and the four edge points half a pixel
    # either side of each center in a single WCS call. The blocks are
    # ordered center, -x, +x, -y, +y.
    xx = np.concatenate((x, x - 0.5, x + 0.5, x, x))
    yy = np.concatenate((y, y, y, y - 0.5, y + 0.5))
    sky = _pixel_to_world(wcs, xx, yy).represent_as(
        UnitSphericalRepresentation)
    lon = sky.lon.rad
    lat = sky.lat.rad

    # Unit vectors pointing at each position
    cos_lat = np.cos(lat)
    xyz = np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon),
                    np.sin(lat)), axis=-1).reshape(5, n, 3)

    # Local East (+longitude) and North (+latitude) unit vectors at the
    # centers. The closed forms stay defined at the poles, where they
    # follow the nominal longitude of the center.
    lon0 = lon[:n]
    lat0 = lat[:n]
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)
    sin_lat0 = np.sin(lat0)
    east = np.stack((-sin_lon0, cos_lon0, np.zeros(n)), axis=-1)
    north = np.stack((-sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0,
                      np.cos(lat0)), axis=-1)

    # The central difference of the edge unit vectors, projected onto
    # East and North, gives the tangent-plane (xi, eta) displacement per
    # pixel in radians. This formulation has no longitude subtraction
    # and no division by cos(lat), so it is wrap-safe and pole-safe.
    arcsec_per_rad = 3600.0 * np.degrees(1)
    jacobians = np.empty((n, 2, 2))
    for col, (lo, hi) in enumerate(((1, 2), (3, 4))):
        step = xyz[hi] - xyz[lo]
        jacobians[:, 0, col] = np.einsum('ij,ij->i', step, east)
        jacobians[:, 1, col] = np.einsum('ij,ij->i', step, north)
    return jacobians * arcsec_per_rad


def compute_pixel_to_sky_mean_scales(x, y, wcs):
    """
    Compute the isotropic (mean) pixel scale at an array of pixel
    positions.

    This is the vectorized counterpart of `pixel_to_sky_mean_scale`. The
    scale at each position is the mean of the two singular values of
    the local forward Jacobian ``F = d(sky_arcsec)/d(pixel)``, which is
    the best isotropic approximation to the (potentially anisotropic)
    mapping. It uses only the forward WCS transform, so it is fast for a
    gwcs whose inverse must be found numerically.

    Parameters
    ----------
    x, y : float or array_like
        The pixel coordinates. Arrays are flattened.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    mean_scales : `~numpy.ndarray`
        The 1D array of mean scale factors (arcsec per pixel).
    """
    jacobians = compute_pixel_to_sky_jacobians(x, y, wcs)
    return _mean_singular_values(jacobians)


def compute_pixel_scale_angles(x, y, wcs):
    """
    Compute the pixel scale and the pixel-frame angle of North at an
    array of pixel positions.

    This is the vectorized counterpart of `wcs_pixel_scale_angle`. The
    scale is the geometric mean of the scales along the x and y pixel
    axes. The angle is found by solving the local forward Jacobian for
    the pixel step that moves due North on the sky, so it needs only the
    forward WCS transform and no per-source inverse.

    Parameters
    ----------
    x, y : float or array_like
        The pixel coordinates. Arrays are flattened.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    scales : `~numpy.ndarray`
        The 1D array of pixel scales in arcsec/pixel.

    angles : `~astropy.coordinates.Angle`
        The angles measured counterclockwise from the positive x axis to
        the "North" axis of the celestial coordinate system, wrapped to
        [0, 360) degrees.
    """
    jacobians = compute_pixel_to_sky_jacobians(x, y, wcs)

    # The columns of F are the sky displacements per pixel step along
    # x and y, so their norms are the directional pixel scales
    axis_scales = np.linalg.norm(jacobians, axis=1)
    scales = np.sqrt(axis_scales[:, 0] * axis_scales[:, 1])

    # Solve F @ step = (0, 1) for the pixel step that moves North.
    # For F = [[a, b], [c, d]] the solution is (-b, a) / det(F). The
    # division keeps the sign of the determinant, which matters for a
    # flipped-parity WCS.
    det = np.linalg.det(jacobians)
    step_x = -jacobians[:, 0, 1] / det
    step_y = jacobians[:, 0, 0] / det
    angles = Angle(np.degrees(np.arctan2(step_y, step_x)) * u.deg)

    return scales, angles.wrap_at(360 * u.deg)


def sky_to_pixel_mean_scale(skycoord, wcs, *, pixcoord=None):
    """
    Compute the pixel center and isotropic (mean) scale factor for a
    sky-to-pixel conversion.

    This function is used for circular regions (circles and circle
    annuli) where a single isotropic scale factor is needed to preserve
    the circular shape. The scale factor is the mean of the two singular
    values of the local Jacobian ``J = d(pixel)/d(sky_arcsec)``, which
    are the maximum and minimum stretch factors of the mapping. Their
    mean is the best isotropic approximation to the (potentially
    anisotropic) Jacobian, in the sense that it minimizes the sum of
    squared residuals between the true (elliptical) mapping and the
    isotropic (circular) approximation.

    For a WCS without distortion and with equal pixel scales in x and y,
    the two singular values are equal and the mean is exact. For
    distorted WCS or non-square pixels, the two singular values differ
    and the mean provides a balanced compromise.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixcoord : tuple of float, optional
        The ``(x, y)`` pixel position of ``skycoord``, if already known.
        When given, the WCS is not inverted to find it.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    mean_scale : float
        The mean scale factor (pixels per arcsec).
    """
    center, jacobian = _sky_to_pixel_jacobian(skycoord, wcs,
                                              pixcoord=pixcoord)
    return center, float(_mean_singular_values(jacobian))


def pixel_to_sky_mean_scale(pixcoord, wcs):
    """
    Compute the sky center and isotropic (mean) scale factor for a
    pixel-to-sky conversion.

    This is the inverse of `sky_to_pixel_mean_scale`. It is used for
    circular pixel regions (circles and circle annuli) where a single
    isotropic scale factor is needed to preserve the circular shape. The
    scale factor is the mean of the two singular values of the local
    forward Jacobian ``F = d(sky_arcsec)/d(pixel)``, which are the
    maximum and minimum angular extents per pixel.

    Parameters
    ----------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        The sky center position.

    mean_scale : float
        The mean scale factor (arcsec per pixel).
    """
    center = _pixel_to_world(wcs, pixcoord[0], pixcoord[1])
    mean_scale = compute_pixel_to_sky_mean_scales(pixcoord[0], pixcoord[1],
                                                  wcs)[0]
    return center, float(mean_scale)


def pixel_shape_to_sky_svd(pixcoord, wcs, width, height, pixel_angle_rad):
    """
    Convert a pixel ellipse to a sky ellipse using SVD.

    This builds the composite matrix ``M_sky = J^{-1} @ M_pix`` where
    ``M_pix`` encodes the pixel ellipse semi-axes and rotation, and
    ``J^{-1}`` is the local inverse Jacobian. The SVD of ``M_sky`` gives
    the exact sky ellipse semi-axes and orientation.

    This handles WCS shear correctly. The sky image of a pixel ellipse
    is always an ellipse, and SVD extracts its true principal axes,
    regardless of whether the Jacobian's mapped width and height
    directions are orthogonal.

    Parameters
    ----------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate of the ellipse center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    width : float
        The full width of the pixel ellipse (before rotation) in pixels.

    height : float
        The full height of the pixel ellipse (before rotation) in
        pixels.

    pixel_angle_rad : float
        The pixel rotation angle in radians. This is the angle of the
        ellipse's width axis measured counterclockwise from the positive
        x-axis.

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        The sky center position.

    sky_width : float
        The full width of the sky ellipse in arcsec.

    sky_height : float
        The full height of the sky ellipse in arcsec.

    sky_angle : `~astropy.coordinates.Angle`
        The sky position angle (PA) of the width axis, measured
        counterclockwise from North (the latitude/Dec axis), wrapped to
        [0, 360) degrees.
    """
    center = _pixel_to_world(wcs, pixcoord[0], pixcoord[1])
    jacobian_inv = compute_pixel_to_sky_jacobians(pixcoord[0], pixcoord[1],
                                                  wcs)[0]

    # Build M_pix: columns are pixel semi-axis vectors
    cos_a = np.cos(pixel_angle_rad)
    sin_a = np.sin(pixel_angle_rad)
    half_w = 0.5 * width
    half_h = 0.5 * height
    m_pix = np.array([[half_w * cos_a, -half_h * sin_a],
                      [half_w * sin_a, half_h * cos_a]])

    # M_sky = J^{-1} @ M_pix: columns are sky semi-axis vectors
    m_sky = jacobian_inv @ m_pix

    sky_width, sky_height, sky_angle = _svd_ellipse_from_composite(
        m_sky, sky_angle=True,
        input_circular=np.isclose(width, height))

    return center, sky_width, sky_height, sky_angle


def sky_shape_to_pixel_svd(skycoord, wcs, width_arcsec, height_arcsec,
                           sky_angle_rad, *, pixcoord=None):
    """
    Convert a sky ellipse to a pixel ellipse using SVD.

    This builds the composite matrix ``M_pix = J @ M_sky`` where
    ``M_sky`` encodes the sky ellipse semi-axes and rotation, and ``J``
    is the local Jacobian. The SVD of ``M_pix`` gives the exact pixel
    ellipse semi-axes and orientation.

    This handles WCS shear correctly. The pixel image of a sky ellipse
    is always an ellipse, and SVD extracts its true principal axes,
    regardless of whether the Jacobian's mapped width and height
    directions are orthogonal.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the ellipse center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    width_arcsec : float
        The full width of the sky ellipse in arcsec.

    height_arcsec : float
        The full height of the sky ellipse in arcsec.

    sky_angle_rad : float
        The sky rotation angle in radians as a position angle (PA).
        This is the angle of the ellipse's width axis measured
        counterclockwise from North (the latitude/Dec axis).

    pixcoord : tuple of float, optional
        The ``(x, y)`` pixel position of ``skycoord``, if already known.
        When given, the WCS is not inverted to find it.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    pixel_width : float
        The full width of the pixel ellipse in pixels.

    pixel_height : float
        The full height of the pixel ellipse in pixels.

    pixel_angle : `~astropy.coordinates.Angle`
        The pixel rotation angle of the width axis, measured
        counterclockwise from the positive x-axis, wrapped to [0, 360)
        degrees.
    """
    center, jacobian = _sky_to_pixel_jacobian(skycoord, wcs,
                                              pixcoord=pixcoord)

    # Build M_sky: columns are sky semi-axis vectors in tangent-plane
    # coordinates (xi=East, eta=North). The width axis is at the given
    # PA from North (toward East), so its tangent-plane components are
    # (sin(PA), cos(PA)). The height axis is perpendicular, at PA+90.
    # The local Jacobian already encodes the WCS parity, so no manual
    # parity factor is applied here.
    cos_pa = np.cos(sky_angle_rad)
    sin_pa = np.sin(sky_angle_rad)
    half_w = 0.5 * width_arcsec
    half_h = 0.5 * height_arcsec
    m_sky = np.array([[half_w * sin_pa, half_h * cos_pa],
                      [half_w * cos_pa, -half_h * sin_pa]])

    # M_pix = J @ M_sky: columns are pixel semi-axis vectors
    m_pix = jacobian @ m_sky

    pixel_width, pixel_height, pixel_angle = _svd_ellipse_from_composite(
        m_pix, input_circular=np.isclose(width_arcsec, height_arcsec))

    return center, pixel_width, pixel_height, pixel_angle


def sky_to_pixel_svd_scales(skycoord, wcs, *, pixcoord=None):
    """
    Compute the pixel center, principal-axis scale factors, and pixel
    angle for a sky-to-pixel conversion using SVD of the local Jacobian.

    Uses the singular value decomposition (SVD) of the local Jacobian
    ``J = d(pixel)/d(sky_arcsec)`` to find the natural principal axes
    of the WCS transformation at the given sky position. The singular
    values give the scale factors along the major and minor axes of the
    ellipse that a unit circle in sky space maps to in pixel space. The
    left singular vectors give the directions of those axes in pixel
    coordinates.

    This is the appropriate method for converting a circular sky region
    to a pixel ellipse, as the resulting ellipse accurately represents
    the true shape of the WCS mapping (i.e., the tightest-fitting pixel
    ellipse that contains the sky circle).

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixcoord : tuple of float, optional
        The ``(x, y)`` pixel position of ``skycoord``, if already known.
        When given, the WCS is not inverted to find it.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    scale_major : float
        The scale factor along the major (maximum-stretch) axis
        (pixels per arcsec).

    scale_minor : float
        The scale factor along the minor (minimum-stretch) axis
        (pixels per arcsec).

    pixel_angle : `~astropy.coordinates.Angle`
        The pixel rotation angle of the major axis, measured
        counterclockwise from the positive x-axis, wrapped to
        [0, 360) degrees.
    """
    center, jacobian = _sky_to_pixel_jacobian(skycoord, wcs,
                                              pixcoord=pixcoord)
    u_mat, s_vals, _vt = np.linalg.svd(jacobian)

    # Pixel angle of the major axis: direction of u_mat[:, 0] in pixel
    # space. No parity correction is needed because pixel space has no
    # axis reflection.
    pixel_angle = Angle(
        np.rad2deg(np.arctan2(u_mat[1, 0], u_mat[0, 0])) * u.deg,
    ).wrap_at(360 * u.deg)

    return center, s_vals[0], s_vals[1], pixel_angle


def pixel_to_sky_svd_scales(pixcoord, wcs):
    """
    Compute the sky center, principal-axis scale factors, and sky angle
    for a pixel-to-sky conversion using SVD of the inverse Jacobian.

    Uses the singular value decomposition (SVD) of the local inverse
    Jacobian ``J^{-1} = d(sky)/d(pixel)`` to find the natural principal
    axes of the WCS transformation at the given pixel position. The
    singular values give the scale factors along the major and minor
    axes of the ellipse that a unit circle in pixel space maps to in sky
    space. The left singular vectors give the directions of those axes
    in tangent-plane coordinates.

    This is the appropriate method for converting a circular pixel
    region to a sky ellipse, as the resulting ellipse accurately
    represents the true shape of the WCS mapping (i.e., the
    tightest-fitting sky ellipse that contains the pixel circle).

    Parameters
    ----------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        The sky center position.

    scale_major : float
        The scale factor along the major (maximum-stretch) axis
        (arcsec per pixel).

    scale_minor : float
        The scale factor along the minor (minimum-stretch) axis
        (arcsec per pixel).

    sky_angle : `~astropy.coordinates.Angle`
        The sky position angle (PA) of the major axis, measured
        counterclockwise from North (the latitude/Dec axis), wrapped to
        [0, 360) degrees.
    """
    center = _pixel_to_world(wcs, pixcoord[0], pixcoord[1])
    jacobian_inv = compute_pixel_to_sky_jacobians(pixcoord[0], pixcoord[1],
                                                  wcs)[0]
    u_mat, s_vals, _vt = np.linalg.svd(jacobian_inv)

    # Sky position angle (PA) of the major axis, measured from North
    # (eta/Dec) toward East (xi/RA). The columns of u_mat are
    # tangent-plane vectors with components (xi=East, eta=North), so the
    # PA is arctan2(xi, eta). The inverse Jacobian already encodes the
    # WCS parity, so no manual parity correction is needed here.
    sky_angle = Angle(
        np.rad2deg(np.arctan2(u_mat[0, 0], u_mat[1, 0])) * u.deg,
    ).wrap_at(360 * u.deg)

    return center, s_vals[0], s_vals[1], sky_angle


def wcs_pixel_scale_angle(skycoord, wcs, *, pixcoord=None):
    """
    Calculate the pixel coordinate, scale, and WCS rotation angle at the
    position of a sky coordinate.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The SkyCoord coordinate.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixcoord : tuple of float, optional
        The ``(x, y)`` pixel position of ``skycoord``, if already known.
        When given, the WCS is not inverted to find it.

    Returns
    -------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate.

    scale : float
        The pixel scale in arcsec/pixel.

    angle : `~astropy.coordinates.Angle`
        The angle measured counterclockwise from the positive x axis to
        the "North" axis of the celestial coordinate system, wrapped to
        [0, 360) degrees.

    Notes
    -----
    This is the scalar counterpart of `compute_pixel_scale_angles`,
    which it calls after converting the sky coordinate to pixels. If
    distortions are present in the WCS, the x and y pixel scales likely
    differ. The returned scale is the geometric mean of the two.
    """
    # Convert to pixel coordinates unless they are already known
    if pixcoord is None:
        x, y = _world_to_pixel(wcs, skycoord)
    else:
        x, y = pixcoord
    pixcoord = (float(x), float(y))
    scales, angles = compute_pixel_scale_angles(x, y, wcs)

    return pixcoord, float(scales[0]), angles[0]
