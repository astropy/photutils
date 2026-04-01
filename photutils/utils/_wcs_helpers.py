# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for WCS helpers.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle


def _has_distortion(wcs):
    """
    Return True if the WCS has distortions or is non-FITS.
    """
    return getattr(wcs, 'has_distortion', True)


def _sky_to_pixel_jacobian(skycoord, wcs):
    """
    Common setup for sky-to-pixel Jacobian-based conversions.

    Returns the pixel center, the local Jacobian matrix, and the WCS
    parity.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    jacobian : 2x2 `~numpy.ndarray`
        The Jacobian matrix ``d(pixel)/d(sky_arcsec)``.

    parity : float
        The sign of ``det(jacobian)`` (+1 or -1).
    """
    x0, y0 = wcs.world_to_pixel(skycoord)
    center = (float(x0), float(y0))
    jacobian = compute_local_wcs_jacobian(skycoord, wcs)
    parity = np.sign(np.linalg.det(jacobian))
    return center, jacobian, parity


def _pixel_to_sky_jacobian(pixcoord, wcs):
    """
    Common setup for pixel-to-sky Jacobian-based conversions.

    Returns the sky center, the local Jacobian matrix, its inverse, and
    the WCS parity.

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

    jacobian : 2x2 `~numpy.ndarray`
        The Jacobian matrix ``d(pixel)/d(sky_arcsec)``.

    jacobian_inv : 2x2 `~numpy.ndarray`
        The inverse Jacobian ``d(sky_arcsec)/d(pixel)``.

    parity : float
        The sign of ``det(jacobian)`` (+1 or -1).
    """
    center = wcs.pixel_to_world(pixcoord[0], pixcoord[1])
    jacobian = compute_local_wcs_jacobian(center, wcs)
    jacobian_inv = np.linalg.inv(jacobian)
    parity = np.sign(np.linalg.det(jacobian))
    return center, jacobian, jacobian_inv, parity


def _svd_ellipse_from_composite(m_comp, width_col_idx=0,
                                use_parity_for_angle=False, parity=1):
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

    use_parity_for_angle : bool, optional
        If True, apply ``parity`` to the RA (x) component when computing
        the sky rotation angle. Default is False (for pixel angles).

    parity : float, optional
        The WCS parity (+1 or -1). Only used if ``use_parity_for_angle``
        is True.

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
        # Major axis aligns with height; swap
        out_width = 2 * s_vals[1]
        out_height = 2 * s_vals[0]
        angle_col = u_mat[:, 1]

    # Fix SVD sign ambiguity: ensure the angle direction aligns with the
    # mapped width semi-axis
    if np.dot(angle_col, width_col) < 0:
        angle_col = -angle_col

    # Compute the rotation angle
    if use_parity_for_angle:
        # Sky position angle (PA) measured from North (eta/Dec) toward
        # East. The xi (RA) component in the composite matrix has
        # -parity baked in, so we multiply by -parity to recover the
        # physical East direction.
        angle = Angle(
            np.rad2deg(np.arctan2(-parity * angle_col[0],
                                  angle_col[1])) * u.deg,
        ).wrap_at(360 * u.deg)
    else:
        # Pixel angle: measured from +x toward +y
        angle = Angle(
            np.rad2deg(np.arctan2(angle_col[1],
                                  angle_col[0])) * u.deg,
        ).wrap_at(360 * u.deg)

    return out_width, out_height, angle


def jacobian_sky_to_pixel_scales(skycoord, wcs, sky_angle_rad):
    """
    Compute the pixel center, directional scale factors, and pixel angle
    for a sky-to-pixel conversion using the local WCS Jacobian.

    This function is used for directed (non-circular) regions such as
    ellipses, rectangles, and asymmetric annuli that have independent
    width and height axes and a rotation angle. Unlike simpler methods
    that use a single scalar pixel scale, this function uses the local
    2x2 Jacobian matrix of the WCS transformation to compute directional
    scale factors along the width and height axes of the region. This
    better handles WCS distortions (e.g., SIP polynomial corrections)
    where the pixel scale varies along different directions.

    The algorithm works as follows:

    1. Compute the 2x2 Jacobian matrix ``J = d(pixel)/d(sky)``
       at the region center using finite differences (see
      `compute_local_wcs_jacobian`).

    2. Construct unit tangent-plane direction vectors ``d_w`` and
       ``d_h`` for the region's width and height axes, respectively,
       using the sky rotation angle. The WCS parity (sign of det(J)) is
       applied to account for the reflected RA axis (RA increases to the
       left in standard projections, giving a negative determinant).

    3. Map these direction vectors through the Jacobian to get the
       corresponding pixel-plane vectors: ``v_w = J @ d_w`` and ``v_h =
       J @ d_h``.

    4. The directional scale factors are the magnitudes (norms) of
       ``v_w`` and ``v_h``, in units of pixels per arcsec.

    5. The pixel rotation angle is the angle of ``v_w`` measured
       counterclockwise from the positive x-axis: ``arctan2(v_w[1],
       v_w[0])``.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    sky_angle_rad : float
        The sky rotation angle in radians as a position angle (PA).
        This is the angle of the region's width axis measured
        counterclockwise from North (the latitude/Dec axis) in the
        tangent-plane coordinate system.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    scale_w : float
        The scale factor along the width direction (pixels per arcsec).

    scale_h : float
        The scale factor along the height direction (pixels per arcsec).

    pixel_angle : `~astropy.coordinates.Angle`
        The pixel rotation angle of the width axis, measured
        counterclockwise from the positive x-axis, in degrees.
    """
    center, jacobian, parity = _sky_to_pixel_jacobian(skycoord, wcs)

    # Construct unit direction vectors in the tangent-plane coordinate
    # system for the region's width and height axes.
    # d_w points along the width axis at the given PA from North;
    # d_h is perpendicular to it.
    d_w = np.array([-parity * np.sin(sky_angle_rad),
                    np.cos(sky_angle_rad)])
    d_h = np.array([-parity * np.cos(sky_angle_rad),
                    -np.sin(sky_angle_rad)])

    # Map sky directions to pixel-plane vectors via the Jacobian
    v_w = jacobian @ d_w
    v_h = jacobian @ d_h

    # Directional scale factors: magnitudes of the mapped vectors
    # (pixels per arcsec along each axis)
    scale_w = np.hypot(v_w[0], v_w[1])
    scale_h = np.hypot(v_h[0], v_h[1])

    # Pixel rotation angle of the width axis
    pixel_angle = Angle(
        np.rad2deg(np.arctan2(v_w[1], v_w[0])) * u.deg).wrap_at(360 * u.deg)

    return center, scale_w, scale_h, pixel_angle


def jacobian_pixel_to_sky_scales(pixcoord, wcs, pixel_angle_rad):
    """
    Compute the sky center, directional scale factors, and sky angle for
    a pixel-to-sky conversion using the local WCS Jacobian.

    This is the inverse of `jacobian_sky_to_pixel_scales`. It is
    used for directed (non-circular) pixel regions such as ellipses,
    rectangles, and asymmetric annuli that have independent width and
    height axes and a rotation angle. The inverse of the local 2x2
    Jacobian matrix is used to map pixel-plane direction vectors back to
    the tangent-plane coordinate system.

    The algorithm works as follows:

    1. Compute the 2x2 Jacobian matrix ``J = d(pixel)/d(sky)`` at
       the region center, then invert it to get
       ``J^{-1} = d(sky)/d(pixel)``.

    2. Construct unit pixel-plane direction vectors ``e_w`` and
       ``e_h`` for the region's width and height axes using the pixel
       rotation angle.

    3. Map these through the inverse Jacobian to get the
       corresponding tangent-plane direction vectors:
       ``d_w = J^{-1} @ e_w`` and ``d_h = J^{-1} @ e_h``.

    4. The directional scale factors are the magnitudes of ``d_w``
       and ``d_h``, in units of arcsec per pixel.

    5. The sky rotation angle is derived from the direction of
       ``d_w`` in the tangent-plane coordinate system as a position
       angle (PA) measured from North.

    Parameters
    ----------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixel_angle_rad : float
        The pixel rotation angle in radians. This is the angle of the
        region's width axis measured counterclockwise from the positive
        x-axis in the pixel coordinate system.

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        The sky center position.

    scale_w : float
        The scale factor along the width direction (arcsec per pixel).

    scale_h : float
        The scale factor along the height direction (arcsec per pixel).

    sky_angle : `~astropy.coordinates.Angle`
        The sky position angle (PA) of the width axis, measured
        counterclockwise from North (the latitude/Dec axis), wrapped to
        [0, 360) degrees.
    """
    center, _, jacobian_inv, _ = _pixel_to_sky_jacobian(pixcoord, wcs)

    # Unit direction vectors in the pixel plane for width and height
    e_w = np.array([np.cos(pixel_angle_rad), np.sin(pixel_angle_rad)])
    e_h = np.array([-np.sin(pixel_angle_rad), np.cos(pixel_angle_rad)])

    # Map pixel directions to tangent-plane vectors via inverse Jacobian
    d_w = jacobian_inv @ e_w
    d_h = jacobian_inv @ e_h

    # Directional scale factors: magnitudes of the mapped vectors
    # (arcsec per pixel along each axis)
    scale_w = np.hypot(d_w[0], d_w[1])
    scale_h = np.hypot(d_h[0], d_h[1])

    # Sky position angle (PA) of the width axis: d_w is in raw
    # tangent-plane coordinates (xi=East, eta=North), so PA is simply
    # arctan2(xi, eta).
    sky_angle = Angle(np.rad2deg(np.arctan2(
        d_w[0], d_w[1])) * u.deg).wrap_at(360 * u.deg)

    return center, scale_w, scale_h, sky_angle


def jacobian_sky_to_pixel_mean_scale(skycoord, wcs):
    """
    Compute the pixel center and isotropic (mean) scale factor for a
    sky-to-pixel conversion using SVD of the local WCS Jacobian.

    This function is used for circular regions (circles and circle
    annuli) where a single isotropic scale factor is needed to preserve
    the circular shape. The scale factor is the mean of the two singular
    values of the Jacobian matrix.

    Singular Value Decomposition (SVD) of the 2x2 Jacobian ``J``
    yields ``J = U @ diag(s1, s2) @ V^T``, where ``s1`` and ``s2`` are
    the singular values representing the maximum and minimum stretch
    factors of the linear transformation. Using their mean as the scale
    factor is the best isotropic approximation to the (potentially
    anisotropic) Jacobian, in the sense that it minimizes the sum of
    squared residuals between the true (elliptical) mapping and the
    isotropic (circular) approximation.

    For a WCS without distortion and with equal pixel scales in x and y,
    ``s1 == s2`` and the mean is exact. For distorted WCS or non-square
    pixels, the two singular values may differ, and the mean provides a
    balanced compromise.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    mean_scale : float
        The mean scale factor (pixels per arcsec), computed as the mean
        of the two singular values of the Jacobian.
    """
    center, jacobian, _ = _sky_to_pixel_jacobian(skycoord, wcs)
    scales = np.linalg.svd(jacobian, compute_uv=False)

    # Mean of singular values gives the best isotropic approximation
    return center, np.mean(scales)


def jacobian_pixel_to_sky_mean_scale(pixcoord, wcs):
    """
    Compute the sky center and isotropic (mean) scale factor for a
    pixel-to-sky conversion using SVD of the inverse Jacobian.

    This is the inverse of `jacobian_sky_to_pixel_mean_scale`. It is
    used for circular pixel regions (circles and circle annuli) where
    a single isotropic scale factor is needed to preserve the circular
    shape.

    The inverse Jacobian ``J^{-1} = d(sky)/d(pixel)`` maps pixel offsets
    to tangent-plane offsets. Its singular values represent the maximum
    and minimum angular extents per pixel. The mean of these singular
    values provides the best isotropic approximation for converting
    pixel radii to sky angular radii.

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
        The mean scale factor (arcsec per pixel), computed as the mean
        of the two singular values of the inverse Jacobian.
    """
    center, _, jacobian_inv, _ = _pixel_to_sky_jacobian(pixcoord, wcs)
    scales = np.linalg.svd(jacobian_inv, compute_uv=False)

    # Mean of singular values gives the best isotropic approximation
    return center, np.mean(scales)


def compute_local_wcs_jacobian(skycoord, wcs):
    """
    Compute the local 2x2 Jacobian matrix d(pixel)/d(tangent-plane) at
    the given sky coordinate using 1-pixel finite differences.

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

    The Jacobian is computed by making 1-pixel offsets in x and y,
    converting the resulting pixel positions to sky coordinates, and
    measuring the tangent-plane displacements in arcsec. This gives
    the forward Jacobian ``F = d(sky_arcsec)/d(pixel)``, which is then
    inverted to obtain ``J = F^{-1} = d(pixel)/d(sky_arcsec)``. Using
    1-pixel steps ensures numerical stability across all pixel scales.

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
    x0, y0 = wcs.world_to_pixel(skycoord)

    # Sky positions at 1-pixel offsets in x and y
    sky0 = wcs.pixel_to_world(x0, y0)
    sky_x = wcs.pixel_to_world(x0 + 1, y0)
    sky_y = wcs.pixel_to_world(x0, y0 + 1)

    ra0 = sky0.spherical.lon.rad
    dec0 = sky0.spherical.lat.rad
    cos_dec = np.cos(dec0)

    # Tangent-plane offsets (xi, eta) in arcsec for a +1 pixel step
    # in x. xi = dRA * cos(dec), eta = dDec, both converted to arcsec.
    dra_x = sky_x.spherical.lon.rad - ra0
    ddec_x = sky_x.spherical.lat.rad - dec0
    dxi_x = dra_x * cos_dec * 3600.0 * np.degrees(1)
    deta_x = ddec_x * 3600.0 * np.degrees(1)

    # Same for a +1 pixel step in y
    dra_y = sky_y.spherical.lon.rad - ra0
    ddec_y = sky_y.spherical.lat.rad - dec0
    dxi_y = dra_y * cos_dec * 3600.0 * np.degrees(1)
    deta_y = ddec_y * 3600.0 * np.degrees(1)

    # Forward Jacobian F = d(sky_arcsec)/d(pixel), shape (2, 2)
    # Rows are (xi, eta), columns are (px_x, px_y).
    forward = np.array([[dxi_x, dxi_y],
                        [deta_x, deta_y]])

    # Invert to get J = d(pixel)/d(sky_arcsec)
    return np.linalg.inv(forward)


def sky_to_pixel_scales(skycoord, wcs, sky_angle_rad):
    """
    Convert sky region parameters (center, directional scales, angle) to
    pixel region parameters.

    For a WCS without distortion, this uses the `wcs_pixel_scale_angle`
    offset method (isotropic pixel scale with a North-based
    rotation angle). For a WCS with distortion (or a non-astropy
    WCS like GWCS), this uses the local Jacobian matrix via
    `jacobian_sky_to_pixel_scales` to compute directional scale factors.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    sky_angle_rad : float
        The sky rotation angle in radians as a position angle (PA).
        This is the angle of the region's width axis measured
        counterclockwise from North (the latitude/Dec axis) in the
        tangent-plane coordinate system.

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    scale_w : float
        The scale factor along the width direction (pixels per arcsec).

    scale_h : float
        The scale factor along the height direction (pixels per arcsec).

    pixel_angle : `~astropy.coordinates.Angle`
        The pixel rotation angle of the width axis, measured
        counterclockwise from the positive x-axis, in degrees.
    """
    # Non-FITS WCS (e.g., GWCS) and astropy.wcs.WCS with distortions
    # should use the Jacobian method to compute the pixel scales and
    # angle.
    if not _has_distortion(wcs):
        center, pixscale, north_angle = wcs_pixel_scale_angle(skycoord, wcs)

        scale = 1.0 / pixscale
        pixel_angle = Angle(np.rad2deg(sky_angle_rad) * u.deg
                            + north_angle,
                            ).wrap_at(360 * u.deg)
        return center, scale, scale, pixel_angle

    return jacobian_sky_to_pixel_scales(skycoord, wcs, sky_angle_rad)


def pixel_to_sky_scales(pixcoord, wcs, pixel_angle_rad):
    """
    Convert pixel region parameters (center, directional scales, angle)
    to sky region parameters.

    For a WCS without distortion, this uses the `wcs_pixel_scale_angle`
    offset method (isotropic pixel scale with a North-based
    rotation angle). For a WCS with distortion (or a non-astropy
    WCS like GWCS), this uses the local Jacobian matrix via
    `jacobian_pixel_to_sky_scales` to compute directional scale factors.

    Parameters
    ----------
    pixcoord : tuple of float
        The ``(x, y)`` pixel coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    pixel_angle_rad : float
        The pixel rotation angle in radians. This is the angle of the
        region's width axis measured counterclockwise from the positive
        x-axis in the pixel coordinate system.

    Returns
    -------
    center : `~astropy.coordinates.SkyCoord`
        The sky center position.

    scale_w : float
        The scale factor along the width direction (arcsec per pixel).

    scale_h : float
        The scale factor along the height direction (arcsec per pixel).

    sky_angle : `~astropy.coordinates.Angle`
        The sky position angle (PA) of the width axis, measured
        counterclockwise from North (the latitude/Dec axis), wrapped to
        [0, 360) degrees.
    """
    # Non-FITS WCS (e.g., GWCS) and astropy.wcs.WCS with distortions
    # should use the Jacobian method to compute the pixel scales and
    # angle.
    if not _has_distortion(wcs):
        center = wcs.pixel_to_world(pixcoord[0], pixcoord[1])
        _, pixscale, north_angle = wcs_pixel_scale_angle(center, wcs)
        sky_angle = Angle(np.rad2deg(pixel_angle_rad) * u.deg
                          - north_angle,
                          ).wrap_at(360 * u.deg)
        return center, pixscale, pixscale, sky_angle

    return jacobian_pixel_to_sky_scales(pixcoord, wcs, pixel_angle_rad)


def sky_to_pixel_mean_scale(skycoord, wcs):
    """
    Convert a sky region center to pixel coordinates with an isotropic
    scale factor.

    For a WCS without distortion, this uses the `wcs_pixel_scale_angle`
    offset method. For a WCS with distortion (or a non-astropy WCS
    like GWCS), this uses the SVD of the local Jacobian matrix via
    `jacobian_sky_to_pixel_mean_scale`.

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        The sky coordinate of the region center.

    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    Returns
    -------
    center : tuple of float
        The ``(x, y)`` pixel center position.

    mean_scale : float
        The mean scale factor (pixels per arcsec).
    """
    # Non-FITS WCS (e.g., GWCS) and astropy.wcs.WCS with distortions
    # should use the Jacobian method to compute the pixel scales and
    # angle.
    if not _has_distortion(wcs):
        center, pixscale, _ = wcs_pixel_scale_angle(skycoord, wcs)
        return center, 1.0 / pixscale

    return jacobian_sky_to_pixel_mean_scale(skycoord, wcs)


def pixel_to_sky_mean_scale(pixcoord, wcs):
    """
    Convert a pixel region center to sky coordinates with an isotropic
    scale factor.

    For a WCS without distortion, this uses the `wcs_pixel_scale_angle`
    offset method. For a WCS with distortion (or a non-astropy WCS
    like GWCS), this uses the SVD of the inverse Jacobian matrix via
    `jacobian_pixel_to_sky_mean_scale`.

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
    # Non-FITS WCS (e.g., GWCS) and astropy.wcs.WCS with distortions
    # should use the Jacobian method to compute the pixel scales and
    # angle.
    if not _has_distortion(wcs):
        center = wcs.pixel_to_world(pixcoord[0], pixcoord[1])
        _, pixscale, _ = wcs_pixel_scale_angle(center, wcs)
        return center, pixscale

    return jacobian_pixel_to_sky_mean_scale(pixcoord, wcs)


def pixel_ellipse_to_sky_svd(pixcoord, wcs, width, height, pixel_angle_rad):
    """
    Convert a pixel ellipse to a sky ellipse using SVD.

    This builds the composite matrix ``M_sky = J^{-1} @ M_pix`` where
    ``M_pix`` encodes the pixel ellipse semi-axes and rotation, and
    ``J^{-1}`` is the local inverse Jacobian. The SVD of ``M_sky`` gives
    the exact sky ellipse semi-axes and orientation.

    This handles WCS shear correctly: the sky image of a pixel ellipse
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
    center, _, jacobian_inv, parity = _pixel_to_sky_jacobian(pixcoord, wcs)

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
        m_sky, use_parity_for_angle=True, parity=parity)

    return center, sky_width, sky_height, sky_angle


def sky_ellipse_to_pixel_svd(skycoord, wcs, width_arcsec, height_arcsec,
                             sky_angle_rad):
    """
    Convert a sky ellipse to a pixel ellipse using SVD.

    This builds the composite matrix ``M_pix = J @ M_sky`` where
    ``M_sky`` encodes the sky ellipse semi-axes and rotation, and ``J``
    is the local Jacobian. The SVD of ``M_pix`` gives the exact pixel
    ellipse semi-axes and orientation.

    This handles WCS shear correctly: the pixel image of a sky ellipse
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
    center, jacobian, parity = _sky_to_pixel_jacobian(skycoord, wcs)

    # Build M_sky: columns are sky semi-axis vectors in tangent-plane
    # coordinates (xi=RA, eta=Dec). The width axis is at the given PA
    # from North. Apply parity to the RA (xi) component.
    cos_pa = np.cos(sky_angle_rad)
    sin_pa = np.sin(sky_angle_rad)
    half_w = 0.5 * width_arcsec
    half_h = 0.5 * height_arcsec
    m_sky = np.array([[-parity * half_w * sin_pa,
                       -parity * half_h * cos_pa],
                      [half_w * cos_pa, -half_h * sin_pa]])

    # M_pix = J @ M_sky: columns are pixel semi-axis vectors
    m_pix = jacobian @ m_sky

    pixel_width, pixel_height, pixel_angle = _svd_ellipse_from_composite(
        m_pix)

    return center, pixel_width, pixel_height, pixel_angle


def sky_to_pixel_svd_scales(skycoord, wcs):
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
    center, jacobian, _ = _sky_to_pixel_jacobian(skycoord, wcs)
    u_mat, s_vals, _vt = np.linalg.svd(jacobian)

    # Pixel angle of the major axis: direction of u_mat[:,0] in pixel
    # space. No parity correction needed — pixel space has no axis
    # reflection.
    pixel_angle = Angle(
        np.rad2deg(np.arctan2(u_mat[1, 0], u_mat[0, 0])) * u.deg).wrap_at(
            360 * u.deg)

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
    center, _, jacobian_inv, _ = _pixel_to_sky_jacobian(pixcoord, wcs)

    u_mat, s_vals, _vt = np.linalg.svd(jacobian_inv)

    # Sky position angle (PA) of the major axis: u_mat columns are in
    # raw tangent-plane coordinates (xi=East, eta=North), so PA is
    # simply arctan2(xi, eta).
    sky_angle = Angle(
        np.rad2deg(np.arctan2(u_mat[0, 0], u_mat[1, 0])) * u.deg,
    ).wrap_at(360 * u.deg)

    return center, s_vals[0], s_vals[1], sky_angle


def wcs_pixel_scale_angle(skycoord, wcs):
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
    If distortions are present in the WCS, the x and y pixel scales
    likely differ. This function computes independent x and y scales and
    takes their geometric mean.
    """
    # Convert to pixel coordinates
    x, y = wcs.world_to_pixel(skycoord)
    pixcoord = (float(x), float(y))

    # Position-dependent scale using 1-pixel offsets in x and y.
    # The pixel scale is the geometric mean of the two directional
    # scales.
    sky0 = wcs.pixel_to_world(x, y)
    sky_x = wcs.pixel_to_world(x + 1, y)
    sky_y = wcs.pixel_to_world(x, y + 1)
    cdelt_x = sky0.separation(sky_x).arcsec
    cdelt_y = sky0.separation(sky_y).arcsec
    scale = np.sqrt(cdelt_x * cdelt_y)

    # Compute the angle by offsetting in latitude by exactly the local
    # cdelt (geometric-mean pixel scale in degrees). This ensures
    # the finite-difference derivative probes the same scale of the
    # distortion field.
    cdelt_deg = scale / 3600  # arcsec -> deg
    skycoord_offset = skycoord.directional_offset_by(
        0.0, cdelt_deg * u.deg)
    x_offset, y_offset = wcs.world_to_pixel(skycoord_offset)
    dx = x_offset - x
    dy = y_offset - y

    angle_rad = np.arctan2(dy, dx)
    angle = Angle(np.rad2deg(angle_rad) * u.deg).wrap_at(360 * u.deg)

    return pixcoord, scale, angle
