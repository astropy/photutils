# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for computing the on-sky area of image pixels from a WCS.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import RectBivariateSpline

from photutils.utils._wcs_helpers import compute_pixel_to_sky_jacobians

__all__ = ['compute_pixel_area_map', 'compute_pixel_areas']


def compute_pixel_areas(wcs, x, y):
    """
    Compute the on-sky area of the pixels at the given positions.

    The area of a pixel is the area of the parallelogram spanned by
    its two edge vectors on the sky, which is the absolute determinant
    of the local forward WCS Jacobian ``d(sky_arcsec)/d(pixel)``. The
    Jacobian is evaluated by central finite differences half a pixel
    either side of each position, so the area is exact for a locally
    quadratic distortion in the small-angle limit. The finite pixel
    size adds a relative error of about ``h**2 / 12`` for a pixel that
    subtends an angle ``h`` in radians, which is 3e-7 at 0.1 deg per
    pixel and negligible for imaging data. The area is well defined at
    the celestial poles and across the longitude wraparound.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    x, y : float or array_like
        The pixel coordinates. Both must have the same shape.

    Returns
    -------
    areas : float or `~numpy.ndarray`
        The pixel areas in arcsec\\ :sup:`2`, with the same shape as
        ``x`` and ``y``. A float is returned for scalar input. The area
        is NaN for a non-finite position or a position where the WCS is
        undefined (e.g., outside the valid region of the projection).

    See Also
    --------
    compute_pixel_area_map

    Notes
    -----
    The pixel area of a distorted image typically varies by a few
    percent across the field. Multiplying a surface brightness image
    (e.g., in MJy/sr) by a single nominal pixel area therefore leaves
    a position-dependent error in the resulting fluxes. This function
    gives the area of each pixel from the WCS instead.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_wcs
    >>> from photutils.utils import compute_pixel_areas
    >>> wcs = make_wcs((100, 100))
    >>> areas = compute_pixel_areas(wcs, [10.0, 50.0], [20.0, 50.0])
    >>> print(np.round(areas, 6))  # arcsec**2
    [0.01 0.01]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        msg = 'x and y must have the same shape'
        raise ValueError(msg)

    jacobians = compute_pixel_to_sky_jacobians(wcs, x, y)

    # The explicit 2x2 determinant is much faster than np.linalg.det on
    # a stack of small matrices and propagates NaN without raising a
    # warning.
    areas = np.abs(jacobians[:, 0, 0] * jacobians[:, 1, 1]
                   - jacobians[:, 0, 1] * jacobians[:, 1, 0])

    # An infinite position gives coincident offset points and thus a
    # zero area. Report NaN for any non-finite position instead.
    finite = (np.isfinite(x) & np.isfinite(y)).ravel()
    areas = np.where(finite, areas, np.nan).reshape(x.shape)
    if areas.ndim == 0:
        return float(areas)
    return areas


def _is_positive_int(value):
    """
    Return whether ``value`` is a positive integer (and not a bool).
    """
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and value > 0)


def compute_pixel_area_map(wcs, shape, *, step=None):
    """
    Compute the on-sky area of every pixel in an image.

    The areas are computed with `compute_pixel_areas` on a coarse grid
    of positions and then interpolated onto the full image grid with a
    bicubic spline. Distortions vary smoothly on scales far larger than
    the grid spacing, so the interpolation error is negligible compared
    with the variation in area that the map captures.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    shape : 2-tuple of int
        The ``(ny, nx)`` shape of the image.

    step : int or None, optional
        The spacing in pixels of the coarse grid on which the areas are
        evaluated. The grid is padded by two steps beyond the image
        edges, so the interpolation never extrapolates and the spline
        is well defined for any image shape. Steps from about 32 to 128
        give equivalent results, since smaller steps only add runtime
        and larger steps lose accuracy on strongly distorted or very
        wide fields. The largest allowed step is ``min(shape) // 8``
        (and at least 1), so that the coarse grid always has at least
        eight intervals across the image. If `None`, the step is 64
        or the largest allowed step, whichever is smaller. A larger
        step than allowed is reduced to the largest allowed step with a
        warning.

    Returns
    -------
    areas : `~numpy.ndarray`
        The 2D array of pixel areas in arcsec\\ :sup:`2`, with the same
        shape as the image. The area is NaN for pixels where the WCS is
        undefined (e.g., outside the valid region of the projection).

    See Also
    --------
    compute_pixel_areas

    Notes
    -----
    If any node of the padded coarse grid lies outside the valid region
    of the projection, which can happen for an all-sky map even when
    every image pixel is valid, the areas are instead evaluated directly
    at every pixel with `compute_pixel_areas`. That is exact but slower
    (a few seconds for a 4096 x 4096 image).

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_wcs
    >>> from photutils.utils import compute_pixel_area_map
    >>> shape = (100, 100)
    >>> wcs = make_wcs(shape)
    >>> areas = compute_pixel_area_map(wcs, shape)
    >>> areas.shape
    (100, 100)
    >>> print(np.round(areas[50, 50], 6))  # arcsec**2
    0.01
    """
    msg = 'shape must be two positive integers'
    try:
        ny, nx = shape
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not (_is_positive_int(ny) and _is_positive_int(nx)):
        raise ValueError(msg)

    # Ensure the image spans at least eight grid intervals so that large
    # steps on smaller images do not leave the spline with too few
    # knots.
    max_step = max(1, min(ny, nx) // 8)
    if step is None:
        step = min(64, max_step)
    elif not _is_positive_int(step):
        msg = 'step must be a positive integer or None'
        raise ValueError(msg)
    elif step > max_step:
        msg = (f'step={step} was reduced to {max_step} so that the coarse '
               'grid has at least eight intervals across the image')
        warnings.warn(msg, AstropyUserWarning)
        step = max_step

    # Coarse grid padded by two steps beyond each edge. This gives the
    # spline at least five knots per axis and keeps the interpolation
    # inside the sampled region for every pixel.
    grid_y = np.arange(-2 * step, ny + 2 * step, step, dtype=float)
    grid_x = np.arange(-2 * step, nx + 2 * step, step, dtype=float)
    xx, yy = np.meshgrid(grid_x, grid_y)
    coarse = compute_pixel_areas(wcs, xx, yy)

    # A NaN anywhere on the coarse grid would make the spline NaN at
    # every pixel. That happens when a node lies outside the valid
    # region of the projection, which the padding can do even when every
    # image pixel is valid (e.g., an all-sky map). Evaluate the areas
    # directly at every pixel in that case.
    if not np.all(np.isfinite(coarse)):
        yy, xx = np.mgrid[:ny, :nx]
        return compute_pixel_areas(wcs, xx, yy)

    spline = RectBivariateSpline(grid_y, grid_x, coarse)
    return spline(np.arange(ny, dtype=float), np.arange(nx, dtype=float))
