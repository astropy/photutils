# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functional PSF models.
"""

from functools import lru_cache

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.utils import ellipse_extent
from astropy.units import UnitsError
from scipy.special import erf, j0, j1, jn_zeros, ndtr, owens_t

__all__ = [
    'AiryDiskPRF',
    'AiryDiskPSF',
    'CircularGaussianPRF',
    'CircularGaussianPSF',
    'CircularGaussianSigmaPRF',
    'GaussianPRF',
    'GaussianPSF',
    'MoffatPRF',
    'MoffatPSF',
]

# The smallest positive normal float32 value (~1.2e-38), used as the
# lower bound of the strictly-positive model parameters.
_FLOAT_TINY = float(np.finfo(np.float32).tiny)

# Conversion factor from a Gaussian FWHM to its standard deviation.
_GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _dimensionless_value(value):
    """
    Return the value of a dimensionless `~astropy.units.Quantity` as
    a plain array, and any other input unchanged.
    """
    if isinstance(value, u.Quantity):
        return value.to_value(u.dimensionless_unscaled)
    return value


def _rotated_gaussian_moments(x_sigma, y_sigma, theta):
    """
    Calculate the moments along the x and y axes of a rotated 2D
    Gaussian.

    Parameters
    ----------
    x_sigma, y_sigma : float or `~numpy.ndarray`
        The standard deviations along the principal axes of the
        Gaussian.

    theta : float or `~astropy.units.Quantity`
        The counterclockwise rotation angle, either as a float in
        radians or as an angular `~astropy.units.Quantity`.

    Returns
    -------
    cost, sint : float or `~numpy.ndarray`
        The cosine and sine of the rotation angle.

    x_std, y_std : float or `~numpy.ndarray`
        The standard deviations along the x and y axes.

    rho : float or `~numpy.ndarray`
        The correlation coefficient.

    sqrt_one_minus_rho2 : float or `~numpy.ndarray`
        The square root of ``1 - rho**2``.
    """
    cost = _dimensionless_value(np.cos(theta))
    sint = _dimensionless_value(np.sin(theta))
    x_std = np.hypot(cost * x_sigma, sint * y_sigma)
    y_std = np.hypot(sint * x_sigma, cost * y_sigma)
    std_product = x_std * y_std
    rho = _dimensionless_value(
        cost * sint * (x_sigma**2 - y_sigma**2) / std_product)
    # The determinant of the covariance matrix is invariant under
    # rotation, which gives sqrt(1 - rho**2) without loss of precision
    # for a highly elongated Gaussian.
    sqrt_one_minus_rho2 = _dimensionless_value(
        x_sigma * y_sigma / std_product)

    return cost, sint, x_std, y_std, rho, sqrt_one_minus_rho2


def _bivariate_normal_corner_term(h, k, rho, sqrt_one_minus_rho2):
    """
    Calculate the part of the standard bivariate normal distribution
    function that does not cancel in the probability of a rectangle.

    The distribution function with a correlation coefficient ``rho`` is
    ``(ndtr(h) + ndtr(k)) / 2`` minus the returned term, where the term
    is written with Owen's T function [1]_. The ``ndtr`` terms cancel
    when the distribution function is differenced over the four corners
    of a rectangle.

    Parameters
    ----------
    h, k : float or `~numpy.ndarray`
        The standardized coordinates.

    rho : float or `~numpy.ndarray`
        The correlation coefficient.

    sqrt_one_minus_rho2 : float or `~numpy.ndarray`
        The square root of ``1 - rho**2``.

    Returns
    -------
    result : `~numpy.ndarray`
        The term to subtract from ``(ndtr(h) + ndtr(k)) / 2`` to get the
        distribution function.

    References
    ----------
    .. [1] Owen, D. B. 1956, Annals of Mathematical Statistics, 27, 1075
    """
    h, k, rho, sqrt_one_minus_rho2 = np.broadcast_arrays(
        h, k, rho, sqrt_one_minus_rho2)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_h = (k - rho * h) / (h * sqrt_one_minus_rho2)
        slope_k = (h - rho * k) / (k * sqrt_one_minus_rho2)
    # The slopes take their limiting values where a coordinate is zero.
    slope_h = np.where(h == 0, np.copysign(np.inf, k), slope_h)
    slope_k = np.where(k == 0, np.copysign(np.inf, h), slope_k)

    product = h * k
    opposite_signs = (product < 0) | ((product == 0) & (h + k < 0))
    term = owens_t(h, slope_h) + owens_t(k, slope_k) + 0.5 * opposite_signs

    # Both slopes are undefined at the origin, where the distribution
    # function is 1/4 + arcsin(rho) / (2 pi).
    origin = (h == 0) & (k == 0)
    return np.where(origin, 0.25 - np.arcsin(rho) / (2.0 * np.pi), term)


def _is_separable(rho):
    """
    Return whether a 2D Gaussian with the input correlation coefficient
    is separable along the x and y axes.
    """
    return np.all(np.abs(rho) < 1.0e-12)


def _pixel_lattice(dx, dy):
    """
    Find the grid of pixel corners shared by pixels that lie on a
    lattice with a spacing of one pixel.

    Parameters
    ----------
    dx, dy : float or `~numpy.ndarray`
        The x and y pixel centers.

    Returns
    -------
    result : tuple or `None`
        The x and y positions of the pixel corners along each axis of
        the corner grid and the x and y integer indices in that grid of
        the lower corner of each pixel. `None` is returned if the pixels
        are not on a lattice with a spacing of one pixel or if the grid
        would have more corners than twice the number of pixels, in
        which case the grid saves no evaluations.
    """
    dx, dy = np.broadcast_arrays(dx, dy)
    indices = []
    edges = []
    n_corners = 1.0
    for offset in (dx, dy):
        lowest = offset.min()
        highest = offset.max()
        if not (np.isfinite(lowest) and np.isfinite(highest)):
            return None
        steps = offset - lowest
        index = np.rint(steps)
        # The offsets are differences from the source position, so
        # they are on the lattice only to within their rounding
        tolerance = 8.0 * np.spacing(max(abs(lowest), abs(highest), 1.0))
        if not np.all(np.abs(steps - index) <= tolerance):
            return None
        n_edges = index.max() + 2.0
        n_corners *= n_edges
        if n_corners > 2.0 * offset.size:
            return None
        indices.append(index.astype(np.intp))
        edges.append(lowest - 0.5 + np.arange(n_edges))

    return (*edges, *indices)


def _gaussian_pixel_fraction(dx, dy, x_std, y_std, rho, sqrt_one_minus_rho2):
    """
    Calculate the fraction of the flux of a 2D Gaussian that falls in a
    pixel.

    Parameters
    ----------
    dx, dy : float or `~numpy.ndarray`
        The x and y pixel centers relative to the Gaussian center.

    x_std, y_std : float or `~numpy.ndarray`
        The standard deviations of the Gaussian along the x and y axes.

    rho : float or `~numpy.ndarray`
        The correlation coefficient of the Gaussian.

    sqrt_one_minus_rho2 : float or `~numpy.ndarray`
        The square root of ``1 - rho**2``.

    Returns
    -------
    result : `~numpy.ndarray`
        The fraction of the flux in each pixel.
    """
    x_lo = (dx - 0.5) / x_std
    x_hi = (dx + 0.5) / x_std
    y_lo = (dy - 0.5) / y_std
    y_hi = (dy + 0.5) / y_std
    if _is_separable(rho):
        sqrt2 = np.sqrt(2.0)
        return (0.25 * (erf(x_hi / sqrt2) - erf(x_lo / sqrt2))
                * (erf(y_hi / sqrt2) - erf(y_lo / sqrt2)))

    args = (rho, sqrt_one_minus_rho2)
    single_gaussian = all(np.size(value) == 1
                          for value in (x_std, y_std, rho))
    if single_gaussian and (lattice := _pixel_lattice(dx, dy)) is not None:
        # Neighboring pixels share their corners, so each corner of
        # the grid is evaluated once instead of once for each of the
        # pixels that touch it.
        x_edges, y_edges, x_index, y_index = lattice
        corners = _bivariate_normal_corner_term(
            x_edges / np.ravel(x_std),
            y_edges[:, np.newaxis] / np.ravel(y_std),
            np.ravel(rho), np.ravel(sqrt_one_minus_rho2))
        fraction = (corners[y_index + 1, x_index]
                    + corners[y_index, x_index + 1]
                    - corners[y_index + 1, x_index + 1]
                    - corners[y_index, x_index])
        fraction = fraction.reshape(np.shape(x_lo))
    else:
        fraction = (_bivariate_normal_corner_term(x_lo, y_hi, *args)
                    + _bivariate_normal_corner_term(x_hi, y_lo, *args)
                    - _bivariate_normal_corner_term(x_hi, y_hi, *args)
                    - _bivariate_normal_corner_term(x_lo, y_lo, *args))
    # Rounding in the difference can give values of about -1e-16 far
    # from the peak
    return np.maximum(fraction, 0.0)


def _gaussian_edge_integrals(edge, lo, hi, std, rho, sqrt_one_minus_rho2):
    """
    Integrate a 2D Gaussian of unit flux, and its derivative across a
    pixel edge, along that pixel edge.

    Parameters
    ----------
    edge : float or `~numpy.ndarray`
        The position of the pixel edge along the axis perpendicular to
        it, relative to the Gaussian center and in units of the standard
        deviation along that axis.

    lo, hi : float or `~numpy.ndarray`
        The limits of the pixel edge along the other axis, relative to
        the Gaussian center and in units of the standard deviation along
        that axis.

    std : float or `~numpy.ndarray`
        The standard deviation of the Gaussian along the axis
        perpendicular to the edge.

    rho : float or `~numpy.ndarray`
        The correlation coefficient of the Gaussian.

    sqrt_one_minus_rho2 : float or `~numpy.ndarray`
        The square root of ``1 - rho**2``.

    Returns
    -------
    integral, deriv_integral : `~numpy.ndarray`
        The integral along the edge of the Gaussian and of the
        derivative of the Gaussian with respect to the coordinate
        perpendicular to the edge.
    """
    # Along the edge the Gaussian is a 1D Gaussian with a mean of
    # rho * edge and a standard deviation of sqrt(1 - rho**2).
    t_lo = (lo - rho * edge) / sqrt_one_minus_rho2
    t_hi = (hi - rho * edge) / sqrt_one_minus_rho2
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    density = norm * np.exp(-0.5 * edge**2)
    enclosed = ndtr(t_hi) - ndtr(t_lo)
    density_diff = norm * (np.exp(-0.5 * t_hi**2) - np.exp(-0.5 * t_lo**2))

    integral = density * enclosed / std
    deriv_integral = (-density * (edge * enclosed
                                  + rho / sqrt_one_minus_rho2 * density_diff)
                      / std**2)
    return integral, deriv_integral


def _separable_axis_terms(lo, hi, std):
    """
    Calculate the terms along one axis of the derivatives of the pixel
    fraction of a 2D Gaussian that is separable along the x and y axes.

    Parameters
    ----------
    lo, hi : float or `~numpy.ndarray`
        The pixel edges along the axis, relative to the Gaussian center
        and in units of the standard deviation along that axis.

    std : float or `~numpy.ndarray`
        The standard deviation of the Gaussian along the axis.

    Returns
    -------
    fraction, d_center, d_var, density_diff : `~numpy.ndarray`
        The fraction of the 1D Gaussian between the pixel edges, its
        partial derivatives with respect to the center and the variance,
        and the difference of the 1D Gaussian between the pixel edges.
    """
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    density_lo = norm * np.exp(-0.5 * lo**2) / std
    density_hi = norm * np.exp(-0.5 * hi**2) / std
    fraction = ndtr(hi) - ndtr(lo)
    d_center = density_lo - density_hi
    d_var = 0.5 * (lo * density_lo - hi * density_hi) / std

    return fraction, d_center, d_var, density_hi - density_lo


def _gaussian_pixel_fraction_derivs(x_lo, x_hi, y_lo, y_hi, x_std, y_std,
                                    rho, sqrt_one_minus_rho2):
    """
    Calculate the partial derivatives of the fraction of the flux of a
    2D Gaussian that falls in a pixel.

    The derivatives with respect to the elements of the covariance
    matrix follow from the Gaussian being a solution of the diffusion
    equation. The derivative of the Gaussian with respect to a variance
    is half of its second derivative along that axis, and the derivative
    with respect to the covariance is its mixed second derivative. The
    integrals of those derivatives over the pixel reduce to integrals
    along the pixel edges and to the values of the Gaussian at the pixel
    corners.

    Parameters
    ----------
    x_lo, x_hi, y_lo, y_hi : float or `~numpy.ndarray`
        The pixel edges relative to the Gaussian center, in units of the
        standard deviations of the Gaussian along the x and y axes.

    x_std, y_std : float or `~numpy.ndarray`
        The standard deviations of the Gaussian along the x and y axes.

    rho : float or `~numpy.ndarray`
        The correlation coefficient of the Gaussian.

    sqrt_one_minus_rho2 : float or `~numpy.ndarray`
        The square root of ``1 - rho**2``.

    Returns
    -------
    d_x_0, d_y_0, d_var_x, d_var_y, d_cov_xy : `~numpy.ndarray`
        The partial derivatives of the pixel fraction with respect to
        the x and y positions of the Gaussian center, the variances
        along the x and y axes, and the covariance.
    """
    if _is_separable(rho):
        # Each derivative is a product of terms along the two axes,
        # which is much faster to evaluate than the edge integrals.
        x_frac, x_d_center, x_d_var, x_diff = _separable_axis_terms(
            x_lo, x_hi, x_std)
        y_frac, y_d_center, y_d_var, y_diff = _separable_axis_terms(
            y_lo, y_hi, y_std)
        return (x_d_center * y_frac, x_frac * y_d_center, x_d_var * y_frac,
                x_frac * y_d_var, x_diff * y_diff)

    args = (rho, sqrt_one_minus_rho2)
    x_lo_int, x_lo_deriv = _gaussian_edge_integrals(x_lo, y_lo, y_hi, x_std,
                                                    *args)
    x_hi_int, x_hi_deriv = _gaussian_edge_integrals(x_hi, y_lo, y_hi, x_std,
                                                    *args)
    y_lo_int, y_lo_deriv = _gaussian_edge_integrals(y_lo, x_lo, x_hi, y_std,
                                                    *args)
    y_hi_int, y_hi_deriv = _gaussian_edge_integrals(y_hi, x_lo, x_hi, y_std,
                                                    *args)

    d_x_0 = x_lo_int - x_hi_int
    d_y_0 = y_lo_int - y_hi_int
    d_var_x = 0.5 * (x_hi_deriv - x_lo_deriv)
    d_var_y = 0.5 * (y_hi_deriv - y_lo_deriv)

    def corner_density(h, k):
        exponent = (h**2 - 2.0 * rho * h * k + k**2) / sqrt_one_minus_rho2**2
        return np.exp(-0.5 * exponent)

    d_cov_xy = ((corner_density(x_hi, y_hi) - corner_density(x_lo, y_hi)
                 - corner_density(x_hi, y_lo) + corner_density(x_lo, y_lo))
                / (2.0 * np.pi * x_std * y_std * sqrt_one_minus_rho2))

    return d_x_0, d_y_0, d_var_x, d_var_y, d_cov_xy


def _circular_gaussian_prf_derivs(x, y, flux, x_0, y_0, sigma):
    """
    Calculate the partial derivatives of a circular 2D Gaussian
    integrated over pixels.

    Parameters
    ----------
    x, y : float or array_like
        The x and y coordinates at which to evaluate the model.

    flux : float
        Total integrated flux over the entire PSF.

    x_0, y_0 : float
        Position of the peak along the x and y axes.

    sigma : float
        The standard deviation of the Gaussian.

    Returns
    -------
    result : list of `~numpy.ndarray`
        The partial derivatives with respect to the flux, the x and y
        positions, and the standard deviation.
    """
    # The Gaussian is separable, so the pixel fraction is the product
    # of the fractions along each axis.
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    sqrt2 = np.sqrt(2.0)
    fractions = []
    for offset in (x - x_0, y - y_0):
        lo = (offset - 0.5) / sigma
        hi = (offset + 0.5) / sigma
        density_lo = norm * np.exp(-0.5 * lo**2)
        density_hi = norm * np.exp(-0.5 * hi**2)
        fraction = 0.5 * (erf(hi / sqrt2) - erf(lo / sqrt2))
        d_center = (density_lo - density_hi) / sigma
        d_sigma = (lo * density_lo - hi * density_hi) / sigma
        fractions.append((fraction, d_center, d_sigma))
    (x_frac, x_d_center, x_d_sigma), (y_frac, y_d_center, y_d_sigma) = (
        fractions)

    return [x_frac * y_frac,
            flux * x_d_center * y_frac,
            flux * x_frac * y_d_center,
            flux * (x_d_sigma * y_frac + x_frac * y_d_sigma)]


@lru_cache
def _pixel_quadrature_nodes(n_nodes):
    """
    Return the Gauss-Legendre nodes and weights for the integral over
    a pixel of unit width along one axis.

    Parameters
    ----------
    n_nodes : int
        The number of nodes.

    Returns
    -------
    offsets, weights : `~numpy.ndarray`
        The offsets of the nodes from the pixel center and their
        weights, which sum to 1. The arrays are read-only because they
        are cached and shared.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    offsets = 0.5 * nodes
    weights = 0.5 * weights
    offsets.setflags(write=False)
    weights.setflags(write=False)
    return offsets, weights


# The largest number of elements in the arrays of quadrature node
# coordinates for which all of the nodes are evaluated in a single call
_MAX_QUADRATURE_ELEMENTS = 2**20


def _integrate_over_pixels(func, x, y, params, n_nodes):
    """
    Integrate a function over the pixels of unit area centered at the
    input positions using Gauss-Legendre quadrature.

    Parameters
    ----------
    func : callable
        The function to integrate. It is called as ``func(x, y)`` with
        arrays that broadcast to two leading axes for the quadrature
        nodes along the y and x axes. It must return an array, or a
        list of arrays, with those leading axes.

    x, y : float or array_like
        The x and y coordinates of the pixel centers.

    params : sequence
        The model parameter values used by ``func``. Their shapes
        are needed to place the quadrature axis ahead of every axis
        of a parameter array, e.g., the model axis of a model set.

    n_nodes : int
        The number of quadrature nodes along each axis of a pixel.

    Returns
    -------
    result : `~numpy.ndarray` or list of `~numpy.ndarray`
        The integral of each array returned by ``func`` over each
        pixel.
    """
    shape = np.broadcast_shapes(np.shape(x), np.shape(y),
                                *(np.shape(param) for param in params))
    x = np.broadcast_to(x, shape, subok=True)
    y = np.broadcast_to(y, shape, subok=True)
    offsets, weights = _pixel_quadrature_nodes(n_nodes)
    trailing = (1,) * len(shape)
    x_offsets = offsets.reshape((n_nodes, *trailing))
    y_offsets = offsets.reshape((n_nodes, 1, *trailing))
    if isinstance(x, u.Quantity):
        x_offsets = x_offsets << x.unit
        y_offsets = y_offsets << y.unit
    xsub = x + x_offsets
    weights_2d = np.outer(weights, weights).reshape((n_nodes, n_nodes,
                                                     *trailing))

    # Small inputs, e.g., the cutouts used for fitting, are evaluated at
    # all of the nodes in one call, which is the fastest. Large inputs
    # are evaluated one row of nodes at a time so that the temporary
    # arrays are n_nodes, and not n_nodes**2, times the size of the
    # input.
    step = n_nodes if xsub.size * n_nodes <= _MAX_QUADRATURE_ELEMENTS else 1
    result = None
    for start in range(0, n_nodes, step):
        rows = slice(start, start + step)
        values = func(xsub, y + y_offsets[rows])
        is_list = isinstance(values, list)
        if not is_list:
            values = [values]
        terms = [np.sum(value * weights_2d[rows], axis=(0, 1))
                 for value in values]
        if result is None:
            result = terms
        else:
            for total, term in zip(result, terms, strict=True):
                total += term

    return result if is_list else result[0]


def _validate_n_nodes(n_nodes):
    """
    Validate the number of quadrature nodes along each axis of a pixel.
    """
    if (isinstance(n_nodes, bool)
            or not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1):
        msg = 'n_nodes must be a positive integer'
        raise ValueError(msg)
    return int(n_nodes)


def _gaussian_amplitude(flux, xsigma, ysigma):
    # Output units should match the input flux units
    if isinstance(xsigma, u.Quantity):
        xsigma = xsigma.value
        ysigma = ysigma.value

    return flux / (2.0 * np.pi * xsigma * ysigma)


class GaussianPSF(Fittable2DModel):
    r"""
    A 2D Gaussian PSF model.

    This model is evaluated by sampling the 2D Gaussian at the input
    coordinates. The Gaussian is normalized such that the analytical
    integral over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    x_fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian along the
        x axis.

    y_fwhm : float, optional
        FWHM of the Gaussian along the y axis.

    theta : float, optional
        The counterclockwise rotation angle either as a float (in
        degrees) or a `~astropy.units.Quantity` angle (optional).

    bbox_factor : float, optional
        The multiple of the x and y standard deviations (sigma) used to
        define the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    CircularGaussianPSF, GaussianPRF, CircularGaussianPRF,
    CircularGaussianSigmaPRF, MoffatPSF, AiryDiskPSF

    Notes
    -----
    The Gaussian function is defined as:

    .. math::

        f(x, y) = \frac{F}{2 \pi \sigma_{x} \sigma_{y}}
                  \exp \left( -a\left(x - x_{0}\right)^{2}
                  - b \left(x - x_{0}\right) \left(y - y_{0}\right)
                  - c \left(y - y_{0}\right)^{2} \right)

    where :math:`F` denotes the total integrated flux, :math:`(x_{0},
    y_{0})` denotes the position of the peak, and :math:`\sigma_{x}` and
    :math:`\sigma_{y}` denote the standard deviations along the x and y
    axes, respectively.

    .. math::

        a = \frac{\cos^{2}{\theta}}{2 \sigma_{x}^{2}}
            + \frac{\sin^{2}{\theta}}{2 \sigma_{y}^{2}}

        b = \frac{\sin{2 \theta}}{2 \sigma_{x}^{2}}
            - \frac{\sin{2 \theta}}{2 \sigma_{y}^{2}}

        c = \frac{\sin^{2}{\theta}} {2 \sigma_{x}^{2}}
            + \frac{\cos^{2}{\theta}}{2 \sigma_{y}^{2}}

    where :math:`\theta` is the rotation angle of the Gaussian.

    The FWHMs of the Gaussian along the x and y axes are given by:

    .. math::

        \rm{FWHM}_{x} = 2 \sigma_{x} \sqrt{2 \ln{2}}

        \rm{FWHM}_{y} = 2 \sigma_{y} \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \,dx \,dy = F

    The ``x_fwhm``, ``y_fwhm``, and ``theta`` parameters are fixed by
    default. If you wish to fit these parameters, set the ``fixed``
    attribute to `False`, e.g.,::

        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF()
        >>> model.x_fwhm.fixed = False
        >>> model.y_fwhm.fixed = False
        >>> model.theta.fixed = False

    By default, the ``x_fwhm`` and ``y_fwhm`` parameters are bounded
    to be strictly positive. These bounds apply only during fitting.
    Directly setting a non-positive width produces unphysical model
    values.

    This model is evaluated at the input coordinates and is not
    integrated over the detector pixels. Its values on a grid of
    detector pixels are the values of the PSF at the pixel centers,
    not the fluxes in the pixels. For a PSF that is undersampled by the
    detector pixels, the model is sharper than a source in the data and
    the sum of its values over the pixels depends on the subpixel
    position of the source.

    This model should therefore not be used to fit the pixel values of
    an image, e.g., for PSF photometry. If its shape parameters are
    fixed to those of the PSF before the integration over the pixels,
    the fitted flux is biased by a few percent for a FWHM of 2 to 3
    pixels, and by much more, with an error that depends on the subpixel
    position of the source, for a FWHM of less than about 1.5 pixels.
    Use `GaussianPRF` instead, which is integrated over the pixels.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import GaussianPSF

        model = GaussianPSF(flux=71.4, x_0=24.3, y_0=25.2, x_fwhm=10.1,
                            y_fwhm=5.82, theta=21.7)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian along the x axis')
    y_fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian along the y axis')
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.x_sigma, self.y_sigma)

    @property
    def x_sigma(self):
        """
        Gaussian sigma (standard deviation) along the x-axis.
        """
        return self.x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    @property
    def y_sigma(self):
        """
        Gaussian sigma (standard deviation) along the y-axis.
        """
        return self.y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations (sigma) used
            to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        # A float theta is in degrees, but ellipse_extent expect radians
        if self.theta.unit is None:
            theta = np.deg2rad(self.theta.value)
        else:
            theta = self.theta.quantity
        dx, dy = ellipse_extent(a, b, theta)
        return ((self.y_0 - dy, self.y_0 + dy), (self.x_0 - dx, self.x_0 + dx))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPSF
        >>> model = GaussianPSF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-7.006904852376157, upper=7.006904852376157)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-8.9178789030242, upper=8.9178789030242)
            }
            model=GaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2.0 * theta)
        xstd = x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        ystd = y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        xstd2 = xstd ** 2
        ystd2 = ystd ** 2
        xdiff = x - x_0
        ydiff = y - y_0
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        # Output units should match the input flux units
        if isinstance(xstd, u.Quantity):
            xstd = xstd.value
            ystd = ystd.value

        amplitude = flux / (2 * np.pi * xstd * ystd)
        return amplitude * np.exp(
            -(a * xdiff**2) - (b * xdiff * ydiff) - (c * ydiff**2))

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the partial derivatives of the 2D Gaussian function
        with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter. The derivative with respect to ``theta`` is
            always per degree, even when ``theta`` is input as an
            angular `~astropy.units.Quantity`.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        cost = np.cos(theta)
        sint = np.sin(theta)
        cost2 = cost ** 2
        sint2 = sint ** 2
        cos2t = np.cos(2.0 * theta)
        sin2t = np.sin(2.0 * theta)
        xstd = x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        ystd = y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        xstd2 = xstd ** 2
        ystd2 = ystd ** 2
        xstd3 = xstd ** 3
        ystd3 = ystd ** 3
        xdiff = x - x_0
        ydiff = y - y_0
        xdiff2 = xdiff ** 2
        ydiff2 = ydiff ** 2
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        amplitude = flux / (2 * np.pi * xstd * ystd)
        exp = np.exp(-(a * xdiff2) - (b * xdiff * ydiff) - (c * ydiff2))
        # Compute the flux derivative directly from the exponential
        # term so that it is finite even at flux = 0
        dg_dflux = exp / (2 * np.pi * xstd * ystd)
        g = flux * dg_dflux

        da_dtheta = sint * cost * ((1.0 / ystd2) - (1.0 / xstd2))
        db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
        dc_dtheta = -da_dtheta

        da_dxstd = -cost2 / xstd3
        db_dxstd = -sin2t / xstd3
        dc_dxstd = -sint2 / xstd3

        da_dystd = -sint2 / ystd3
        db_dystd = sin2t / ystd3
        dc_dystd = -cost2 / ystd3

        dg_dx_0 = g * ((2.0 * a * xdiff) + (b * ydiff))
        dg_dy_0 = g * ((b * xdiff) + (2.0 * c * ydiff))

        damp_dxstd = -amplitude / xstd
        damp_dystd = -amplitude / ystd
        dexp_dxstd = -exp * (da_dxstd * xdiff2
                             + db_dxstd * xdiff * ydiff
                             + dc_dxstd * ydiff2)
        dexp_dystd = -exp * (da_dystd * xdiff2
                             + db_dystd * xdiff * ydiff
                             + dc_dystd * ydiff2)
        dg_dxstd = damp_dxstd * exp + amplitude * dexp_dxstd
        dg_dystd = damp_dystd * exp + amplitude * dexp_dystd

        # Chain rule for change of variables from sigma to fwhm
        # std => fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        # dstd/dfwhm => _GAUSSIAN_FWHM_TO_SIGMA
        dg_dxfwhm = dg_dxstd * _GAUSSIAN_FWHM_TO_SIGMA
        dg_dyfwhm = dg_dystd * _GAUSSIAN_FWHM_TO_SIGMA

        dg_dtheta = g * (-(da_dtheta * xdiff2 + db_dtheta * xdiff * ydiff
                           + dc_dtheta * ydiff2))
        # Chain rule for unit change
        # theta[rad] => theta[deg] * pi / 180, so drad/dtheta = pi / 180
        dg_dtheta *= np.pi / 180.0

        return [dg_dflux, dg_dx_0, dg_dy_0, dg_dxfwhm, dg_dyfwhm, dg_dtheta]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # We need to make sure that x and y are in the same units
        # otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            msg = "Units of 'x' and 'y' inputs should match"
            raise UnitsError(msg)

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'x_fwhm': inputs_unit[self.inputs[0]],
                'y_fwhm': inputs_unit[self.inputs[0]],
                'theta': u.deg,
                'flux': outputs_unit[self.outputs[0]]}


class CircularGaussianPSF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model.

    This model is evaluated by sampling the 2D Gaussian at the input
    coordinates. The Gaussian is normalized such that the analytical
    integral over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian.

    bbox_factor : float, optional
        The multiple of the standard deviation (sigma) used to define
        the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, GaussianPRF, CircularGaussianPRF,
    CircularGaussianSigmaPRF, MoffatPSF, AiryDiskPSF

    Notes
    -----
    The circular Gaussian function is defined as:

    .. math::

        f(x, y) = \frac{F}{2 \pi \sigma^{2}}
                  \exp \left( {\frac{-(x - x_{0})^{2} - (y - y_{0})^{2}}
                             {2 \sigma^{2}}} \right)

    where :math:`F` is the total integrated flux, :math:`(x_{0}, y_{0})`
    is the position of the peak, and :math:`\sigma` is the standard
    deviation, respectively.

    The FWHM of the Gaussian is given by:

    .. math::

        \rm{FWHM} = 2 \sigma \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \,dx \,dy = F

    The ``fwhm`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF()
        >>> model.fwhm.fixed = False

    By default, the ``fwhm`` parameter is bounded to be strictly
    positive. This bound applies only during fitting. Directly setting a
    non-positive width produces unphysical model values.

    This model is evaluated at the input coordinates and is not
    integrated over the detector pixels. Its values on a grid of
    detector pixels are the values of the PSF at the pixel centers, not
    the fluxes in the pixels. For a PSF that is undersampled by the
    detector pixels, the model is sharper than a source in the data
    and the sum of its values over the pixels depends on the subpixel
    position of the source.

    This model should therefore not be used to fit the pixel values of
    an image, e.g., for PSF photometry. If its shape parameters are
    fixed to those of the PSF before the integration over the pixels,
    the fitted flux is biased by a few percent for a FWHM of 2 to 3
    pixels, and by much more, with an error that depends on the subpixel
    position of the source, for a FWHM of less than about 1.5 pixels.
    Use `CircularGaussianPRF` instead, which is integrated over the
    pixels.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPSF

        model = CircularGaussianPSF(flux=71.4, x_0=24.3, y_0=25.2, fwhm=10.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPSF
        >>> model = CircularGaussianPSF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-4.671269901584105, upper=4.671269901584105)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-5.945252602016134, upper=5.945252602016134)
            }
            model=CircularGaussianPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        sigma2 = (fwhm * _GAUSSIAN_FWHM_TO_SIGMA) ** 2

        # Output units should match the input flux units
        sigma2_norm = sigma2
        if isinstance(sigma2, u.Quantity):
            sigma2_norm = sigma2.value

        amplitude = flux / (2 * np.pi * sigma2_norm)
        return amplitude * np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2)
                                  / sigma2)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the partial derivatives of the 2D Gaussian function
        with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        derivs = GaussianPSF.fit_deriv(x, y, flux, x_0, y_0, fwhm, fwhm, 0.0)

        # The x and y FWHMs are the same variable for a circular
        # Gaussian, so the chain rule gives the sum of the two partial
        # derivatives. The theta derivative is dropped because theta is
        # not a parameter of this model.
        dg_dfwhm = derivs[3] + derivs[4]

        return [*derivs[:3], dg_dfwhm]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # The radial distance requires x and y to have the same unit.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            msg = "Units of 'x' and 'y' inputs should match"
            raise UnitsError(msg)

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'fwhm': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class GaussianPRF(Fittable2DModel):
    r"""
    A 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    area of a pixel centered at each input position (see Notes), and is
    equivalent to assuming the PSF is a 2D Gaussian at a *sub-pixel*
    level. The integral is exact for any rotation angle. Because it is
    integrated over pixels, this model is considered a PRF instead of a
    PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    x_fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian along the
        x axis.

    y_fwhm : float, optional
        FWHM of the Gaussian along the y axis.

    theta : float, optional
        The counterclockwise rotation angle either as a float (in
        degrees) or a `~astropy.units.Quantity` angle (optional).

    bbox_factor : float, optional
        The multiple of the x and y standard deviations (sigma) used to
        define the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, CircularGaussianPSF, CircularGaussianPRF,
    CircularGaussianSigmaPRF, MoffatPSF, AiryDiskPSF

    Notes
    -----
    The model is the integral of a 2D Gaussian over a pixel of unit area
    that is aligned with the x and y axes and centered at :math:`(x,
    y)`:

    .. math::

        f(x, y) = \int_{y - 0.5}^{y + 0.5} \int_{x - 0.5}^{x + 0.5}
            g(u, v) \,du \,dv

    where the Gaussian is:

    .. math::

        g(u, v) = \frac{F}{2 \pi \sigma_{x} \sigma_{y}}
            \exp \left( -\frac{u^{\prime 2}}{2 \sigma_{x}^{2}}
            - \frac{v^{\prime 2}}{2 \sigma_{y}^{2}} \right)

    .. math::

        u^\prime = (u - x_0) \cos(\theta) + (v - y_0) \sin(\theta)

        v^\prime = -(u - x_0) \sin(\theta) + (v - y_0) \cos(\theta)

    :math:`F` is the total integrated flux, :math:`(x_{0}, y_{0})` is
    the position of the peak, :math:`\sigma_{x}` and :math:`\sigma_{y}`
    are the standard deviations along the principal axes of the
    Gaussian, and :math:`\theta` is the rotation angle of the Gaussian.

    For :math:`\theta = 0` the integral is a product of error functions:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(
                    \frac{x - x_0 + 0.5}{\sqrt{2} \sigma_{x}} \right) -
                {\rm erf} \left(
                    \frac{x - x_0 - 0.5}{\sqrt{2} \sigma_{x}} \right)
            \right]
            \left[
                {\rm erf} \left(
                    \frac{y - y_0 + 0.5}{\sqrt{2} \sigma_{y}} \right) -
                {\rm erf} \left(
                    \frac{y - y_0 - 0.5}{\sqrt{2} \sigma_{y}} \right)
            \right]

    A rotated Gaussian is not separable along the x and y axes.
    Its integral over the pixel is the probability of a rectangle
    for a bivariate normal distribution with a nonzero correlation
    coefficient, which is evaluated with Owen's T function [2]_. That
    evaluation is about four times slower than the product of error
    functions (roughly 0.2 ms instead of 0.05 ms for a 25 x 25 pixel
    stamp). The product of error functions is used whenever the
    correlation coefficient is zero, which is the case when ``theta``
    is a multiple of 90 degrees or the x and y widths are equal.

    The rotated evaluation needs the bivariate normal distribution
    function at the four corners of each pixel. Input positions on a
    grid with a spacing of one pixel share those corners between
    neighboring pixels, so each corner is evaluated only once. Other
    input positions are about ten times slower than the product of
    error functions.

    The FWHMs of the Gaussian along the x and y axes are given by:

    .. math::

        \rm{FWHM}_{x} = 2 \sigma_{x} \sqrt{2 \ln{2}}

        \rm{FWHM}_{y} = 2 \sigma_{y} \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \,dx \,dy = F

    Because the model is integrated over the pixels, its values on a
    grid with a spacing of one pixel also sum to the total flux, for any
    subpixel position of the source:

    .. math::

        \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty}
            f(x + i, y + j) = F

    The ``x_fwhm``, ``y_fwhm``, and ``theta`` parameters are fixed by
    default. If you wish to fit these parameters, set the ``fixed``
    attribute to `False`, e.g.,::

        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF()
        >>> model.x_fwhm.fixed = False
        >>> model.y_fwhm.fixed = False
        >>> model.theta.fixed = False

    By default, the ``x_fwhm`` and ``y_fwhm`` parameters are bounded
    to be strictly positive. These bounds apply only during fitting.
    Directly setting a non-positive width produces unphysical model
    values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    .. [2] Owen, D. B. 1956, Annals of Mathematical Statistics, 27, 1075

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import GaussianPRF

        model = GaussianPRF(flux=71.4, x_0=24.3, y_0=25.2, x_fwhm=10.1,
                            y_fwhm=5.82, theta=21.7)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    x_fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian along the x axis')
    y_fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian along the y axis')
    theta = Parameter(
        default=0.0, description=('CCW rotation angle either as a float (in '
                                  'degrees) or a Quantity angle (optional)'),
        fixed=True)

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 x_fwhm=x_fwhm.default, y_fwhm=y_fwhm.default,
                 theta=theta.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, x_fwhm=x_fwhm,
                         y_fwhm=y_fwhm, theta=theta, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.x_sigma, self.y_sigma)

    @property
    def x_sigma(self):
        """
        Gaussian sigma (standard deviation) along the x-axis.
        """
        return self.x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    @property
    def y_sigma(self):
        """
        Gaussian sigma (standard deviation) along the y-axis.
        """
        return self.y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of the x and y standard deviations (sigma) used
            to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        a = factor * self.x_sigma
        b = factor * self.y_sigma
        # A float theta is in degrees, but ellipse_extent expect radians
        if self.theta.unit is None:
            theta = np.deg2rad(self.theta.value)
        else:
            theta = self.theta.quantity
        dx, dy = ellipse_extent(a, b, theta)
        return ((self.y_0 - dy, self.y_0 + dy), (self.x_0 - dx, self.x_0 + dx))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import GaussianPRF
        >>> model = GaussianPRF(x_0=0, y_0=0, x_fwhm=2, y_fwhm=3)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-7.006904852376157, upper=7.006904852376157)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-8.9178789030242, upper=8.9178789030242)
            }
            model=GaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        x_sigma = x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        y_sigma = y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        _, _, x_std, y_std, rho, sqrt_one_minus_rho2 = (
            _rotated_gaussian_moments(x_sigma, y_sigma, theta))

        dx = x - x_0
        dy = y - y_0
        if has_units := isinstance(dx, u.Quantity):
            # A pixel has a size of one in the units of the positions
            x_std = x_std.to_value(dx.unit)
            y_std = y_std.to_value(dy.unit)
            dx = dx.value
            dy = dy.value

        fraction = _gaussian_pixel_fraction(dx, dy, x_std, y_std, rho,
                                            sqrt_one_minus_rho2)

        if has_units:
            # Inputs with units give an output with units
            fraction <<= u.dimensionless_unscaled

        return flux * fraction

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, x_fwhm, y_fwhm, theta):
        """
        Calculate the partial derivatives of the pixel-integrated 2D
        Gaussian function with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        x_fwhm, y_fwhm : float
            FWHM of the Gaussian along the x and y axes.

        theta : float
            The counterclockwise rotation angle either as a float (in
            degrees) or a `~astropy.units.Quantity` angle (optional).

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter. The derivative with respect to ``theta`` is
            always per degree, even when ``theta`` is input as an
            angular `~astropy.units.Quantity`.
        """
        if not isinstance(theta, u.Quantity):
            theta = np.deg2rad(theta)

        x_sigma = x_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        y_sigma = y_fwhm * _GAUSSIAN_FWHM_TO_SIGMA
        cost, sint, x_std, y_std, rho, sqrt_one_minus_rho2 = (
            _rotated_gaussian_moments(x_sigma, y_sigma, theta))
        cov_xy = cost * sint * (x_sigma**2 - y_sigma**2)

        dx = x - x_0
        dy = y - y_0
        edges = ((dx - 0.5) / x_std, (dx + 0.5) / x_std,
                 (dy - 0.5) / y_std, (dy + 0.5) / y_std)
        fraction = _gaussian_pixel_fraction(dx, dy, x_std, y_std, rho,
                                            sqrt_one_minus_rho2)
        d_x_0, d_y_0, d_var_x, d_var_y, d_cov_xy = (
            _gaussian_pixel_fraction_derivs(*edges, x_std, y_std, rho,
                                            sqrt_one_minus_rho2))

        # Chain rule from the covariance matrix elements to the standard
        # deviations along the principal axes and the angle
        d_x_sigma = 2.0 * x_sigma * (cost**2 * d_var_x + sint**2 * d_var_y
                                     + cost * sint * d_cov_xy)
        d_y_sigma = 2.0 * y_sigma * (sint**2 * d_var_x + cost**2 * d_var_y
                                     - cost * sint * d_cov_xy)
        d_theta = (2.0 * cov_xy * (d_var_y - d_var_x)
                   + ((cost**2 - sint**2) * (x_sigma**2 - y_sigma**2)
                      * d_cov_xy))

        # Chain rule for change of variables from sigma to fwhm
        d_x_fwhm = flux * d_x_sigma * _GAUSSIAN_FWHM_TO_SIGMA
        d_y_fwhm = flux * d_y_sigma * _GAUSSIAN_FWHM_TO_SIGMA
        # Chain rule for unit change
        # theta[rad] => theta[deg] * pi / 180, so drad/dtheta = pi / 180
        d_theta = flux * d_theta * np.pi / 180.0

        return [fraction, flux * d_x_0, flux * d_y_0, d_x_fwhm, d_y_fwhm,
                d_theta]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # We need to make sure that x and y are in the same units
        # otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            msg = "Units of 'x' and 'y' inputs should match"
            raise UnitsError(msg)

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'x_fwhm': inputs_unit[self.inputs[0]],
                'y_fwhm': inputs_unit[self.inputs[0]],
                'theta': u.deg,
                'flux': outputs_unit[self.outputs[0]]}


class CircularGaussianPRF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    input coordinate pixels, and is equivalent to assuming the PSF is a
    2D Gaussian at a *sub-pixel* level. Because it is integrated over
    pixels, this model is considered a PRF instead of a PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    fwhm : float, optional
        The full width at half maximum (FWHM) of the Gaussian.

    bbox_factor : float, optional
        The multiple of the standard deviation (sigma) used to define
        the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    GaussianPSF, CircularGaussianPSF, GaussianPRF,
    CircularGaussianSigmaPRF, MoffatPSF, AiryDiskPSF

    Notes
    -----
    The circular Gaussian function is defined as:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(
                    \frac{x - x_0 + 0.5}{\sqrt{2} \sigma} \right) -
                {\rm erf} \left(
                    \frac{x - x_0 - 0.5}{\sqrt{2} \sigma} \right)
            \right]
            \left[
                {\rm erf} \left(
                    \frac{y - y_0 + 0.5}{\sqrt{2} \sigma} \right) -
                {\rm erf} \left(
                    \frac{y - y_0 - 0.5}{\sqrt{2} \sigma} \right)
            \right]

    where :math:`F` is the total integrated flux, :math:`(x_{0},
    y_{0})` is the position of the peak, :math:`\sigma` is the standard
    deviation of the Gaussian, and :math:`{\rm erf}` denotes the error
    function.

    The FWHM of the Gaussian is given by:

    .. math::

        \rm{FWHM} = 2 \sigma \sqrt{2 \ln{2}}

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \,dx \,dy = F

    Because the model is integrated over the pixels, its values on a
    grid with a spacing of one pixel also sum to the total flux, for
    any subpixel position of the source:

    .. math::

        \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty}
            f(x + i, y + j) = F

    The ``fwhm`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF()
        >>> model.fwhm.fixed = False

    By default, the ``fwhm`` parameter is bounded to be strictly
    positive. This bound applies only during fitting. Directly setting a
    non-positive width produces unphysical model values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianPRF

        model = CircularGaussianPRF(flux=71.4, x_0=24.3, y_0=25.2, fwhm=10.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    fwhm = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='FWHM of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 fwhm=fwhm.default, bbox_factor=5.5, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, fwhm=fwhm, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def sigma(self):
        """
        Gaussian sigma (standard deviation).
        """
        return self.fwhm * _GAUSSIAN_FWHM_TO_SIGMA

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianPRF
        >>> model = CircularGaussianPRF(x_0=0, y_0=0, fwhm=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-4.671269901584105, upper=4.671269901584105)
                y: Interval(lower=-4.671269901584105, upper=4.671269901584105)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.945252602016134, upper=5.945252602016134)
                y: Interval(lower=-5.945252602016134, upper=5.945252602016134)
            }
            model=CircularGaussianPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        x0 = x - x_0
        y0 = y - y_0
        sigma = fwhm * _GAUSSIAN_FWHM_TO_SIGMA

        dpix = 0.5
        if isinstance(x0, u.Quantity):
            dpix <<= x0.unit

        return (flux / 4.0
                * ((erf((x0 + dpix) / (np.sqrt(2) * sigma))
                    - erf((x0 - dpix) / (np.sqrt(2) * sigma)))
                   * (erf((y0 + dpix) / (np.sqrt(2) * sigma))
                      - erf((y0 - dpix) / (np.sqrt(2) * sigma)))))

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, fwhm):
        """
        Calculate the partial derivatives of the pixel-integrated 2D
        Gaussian function with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        fwhm : float
            FWHM of the Gaussian.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        derivs = _circular_gaussian_prf_derivs(
            x, y, flux, x_0, y_0, fwhm * _GAUSSIAN_FWHM_TO_SIGMA)
        # Chain rule for change of variables from sigma to fwhm
        derivs[3] = derivs[3] * _GAUSSIAN_FWHM_TO_SIGMA
        return derivs

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # The radial distance requires x and y to have the same unit.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            msg = "Units of 'x' and 'y' inputs should match"
            raise UnitsError(msg)

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'fwhm': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class CircularGaussianSigmaPRF(Fittable2DModel):
    r"""
    A circular 2D Gaussian PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Gaussian over the
    input coordinate pixels, and is equivalent to assuming the PSF is a
    2D Gaussian at a *sub-pixel* level. Because it is integrated over
    pixels, this model is considered a PRF instead of a PSF.

    The Gaussian is normalized such that the analytical integral over
    the entire 2D plane is equal to the total flux.

    This model is equivalent to `CircularGaussianPRF`, but it is
    parameterized in terms of the standard deviation (sigma) instead of
    the full width at half maximum (FWHM).

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    sigma : float, optional
        The standard deviation of the Gaussian.

    bbox_factor : float, optional
        The multiple of the standard deviation (sigma) used to define
        the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` parent class.

    See Also
    --------
    GaussianPSF, CircularGaussianPSF, GaussianPRF, CircularGaussianPRF,
    MoffatPSF, AiryDiskPSF

    Notes
    -----
    The circular Gaussian function is defined as:

    .. math::

        f(x, y) =
            \frac{F}{4}
            \left[
                {\rm erf} \left(\frac{x - x_0 + 0.5}
                                     {\sqrt{2} \sigma} \right)  -
                {\rm erf} \left(\frac{x - x_0 - 0.5}
                                     {\sqrt{2} \sigma} \right)
            \right]
            \left[
                {\rm erf} \left(\frac{y - y_0 + 0.5}
                                     {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{y - y_0 - 0.5}
                                     {\sqrt{2} \sigma} \right)
            \right]

    where :math:`F` is the total integrated flux, :math:`(x_{0},
    y_{0})` is the position of the peak, :math:`\sigma` is the standard
    deviation of the Gaussian, and :math:`{\rm erf}` denotes the error
    function.

    The model is normalized such that:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \,dx \,dy = F

    Because the model is integrated over the pixels, its values on a
    grid with a spacing of one pixel also sum to the total flux, for
    any subpixel position of the source:

    .. math::

        \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty}
            f(x + i, y + j) = F

    The ``sigma`` parameter is fixed by default. If you wish to fit this
    parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import CircularGaussianSigmaPRF
        >>> model = CircularGaussianSigmaPRF()
        >>> model.sigma.fixed = False

    By default, the ``sigma`` parameter is bounded to be strictly
    positive. This bound applies only during fitting. Directly setting a
    non-positive width produces unphysical model values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import CircularGaussianSigmaPRF

        model = CircularGaussianSigmaPRF(flux=71.4, x_0=24.3, y_0=25.2,
                                         sigma=5.1)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    sigma = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='Sigma (standard deviation) of the Gaussian')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 sigma=sigma.default, bbox_factor=5.5, **kwargs):
        super().__init__(sigma=sigma, x_0=x_0, y_0=y_0, flux=flux, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def amplitude(self):
        """
        The peak amplitude of the Gaussian.
        """
        return _gaussian_amplitude(self.flux, self.sigma, self.sigma)

    @property
    def fwhm(self):
        """
        Gaussian FWHM.
        """
        return self.sigma / _GAUSSIAN_FWHM_TO_SIGMA

    def _calc_bounding_box(self, *, factor=5.5):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the standard deviations (sigma) used to
            define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.sigma
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import CircularGaussianSigmaPRF
        >>> model = CircularGaussianSigmaPRF(x_0=0, y_0=0, sigma=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-11.0, upper=11.0)
                y: Interval(lower=-11.0, upper=11.0)
            }
            model=CircularGaussianSigmaPRF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-14.0, upper=14.0)
                y: Interval(lower=-14.0, upper=14.0)
            }
            model=CircularGaussianSigmaPRF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """
        Calculate the value of the 2D Gaussian model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        sigma : float
            The standard deviation of the Gaussian.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        x0 = x - x_0
        y0 = y - y_0

        dpix = 0.5
        if isinstance(x0, u.Quantity):
            dpix <<= x0.unit

        return (flux / 4.0
                * ((erf((x0 + dpix) / (np.sqrt(2) * sigma))
                    - erf((x0 - dpix) / (np.sqrt(2) * sigma)))
                   * (erf((y0 + dpix) / (np.sqrt(2) * sigma))
                      - erf((y0 - dpix) / (np.sqrt(2) * sigma)))))

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, sigma):
        """
        Calculate the partial derivatives of the pixel-integrated 2D
        Gaussian function with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        sigma : float
            The standard deviation of the Gaussian.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        return _circular_gaussian_prf_derivs(x, y, flux, x_0, y_0, sigma)

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # The radial distance requires x and y to have the same unit.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            msg = "Units of 'x' and 'y' inputs should match"
            raise UnitsError(msg)

        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'sigma': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class MoffatPSF(Fittable2DModel):
    r"""
    A 2D Moffat PSF model.

    This model is evaluated by sampling the 2D Moffat function at the
    input coordinates. The Moffat profile is normalized such that the
    analytical integral over the entire 2D plane is equal to the total
    flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    alpha : float, optional
        The characteristic radius of the Moffat profile.

    beta : float, optional
        The asymptotic power-law slope of the Moffat profile wings at
        large radial distances. Larger values provide less flux in the
        profile wings. For large ``beta``, this profile approaches a
        Gaussian profile. ``beta`` must be greater than 1. If ``beta``
        is set to 1, then the Moffat profile is a Lorentz function,
        whose integral is infinite. For this normalized model, if
        ``beta`` is set to 1, then the profile will be zero everywhere.

    bbox_factor : float, optional
        The multiple of the FWHM used to define the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    MoffatPRF, GaussianPSF, CircularGaussianPSF, GaussianPRF,
    CircularGaussianPRF, CircularGaussianSigmaPRF, AiryDiskPSF

    Notes
    -----
    The Moffat profile is defined as:

    .. math::

       f(x, y) = F \frac{\beta - 1}{\pi \alpha^2}
           \left(1 + \frac{\left(x - x_{0}\right)^{2}
               + \left(y - y_{0}\right)^{2}}{\alpha^{2}}\right)^{-\beta}

    where :math:`F` is the total integrated flux and :math:`(x_{0},
    y_{0})` is the position of the peak. Note that :math:`\beta` must be
    greater than 1.

    The FWHM of the Moffat profile is given by:

    .. math::

        \rm{FWHM} = 2 \alpha \sqrt{2^{1 / \beta} - 1}

    The model is normalized such that, for :math:`\beta > 1`:

    .. math::

        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y)
            \,dx \,dy = F

    The ``alpha`` and ``beta`` parameters are fixed by default. If
    you wish to fit these parameters, set the ``fixed`` attribute to
    `False`, e.g.,::

        >>> from photutils.psf import MoffatPSF
        >>> model = MoffatPSF()
        >>> model.alpha.fixed = False
        >>> model.beta.fixed = False

    By default, the ``alpha`` parameter is bounded to be strictly
    positive and the ``beta`` parameter is bounded to be greater than 1.

    This model is evaluated at the input coordinates and is not
    integrated over the detector pixels. Its values on a grid of
    detector pixels are the values of the PSF at the pixel centers, not
    the fluxes in the pixels. For a PSF that is undersampled by the
    detector pixels, the model is sharper than a source in the data
    and the sum of its values over the pixels depends on the subpixel
    position of the source.

    This model should therefore not be used to fit the pixel values of
    an image, e.g., for PSF photometry. If its shape parameters are
    fixed to those of the PSF before the integration over the pixels,
    the fitted flux is biased by a few percent for a FWHM of 2 to 3
    pixels, and by much more, with an error that depends on the subpixel
    position of the source, for a FWHM of less than about 1.5 pixels.
    Use `MoffatPRF` instead, which is integrated over the pixels.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moffat_distribution

    .. [2] https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract

    .. [3] https://ned.ipac.caltech.edu/level5/Stetson/Stetson2_2_1.html

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.psf import MoffatPSF

        model = MoffatPSF(flux=71.4, x_0=24.3, y_0=25.2, alpha=5.1, beta=3.2)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    alpha = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='Characteristic radius of the Moffat profile')
    beta = Parameter(
        default=2,
        bounds=(np.nextafter(1.0, np.inf), None),
        fixed=True,
        description='Power-law index of the Moffat profile')

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 alpha=alpha.default, beta=beta.default, bbox_factor=10.0,
                 **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, alpha=alpha, beta=beta,
                         **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def fwhm(self):
        """
        The FWHM of the Moffat profile.
        """
        return 2.0 * self.alpha * np.sqrt(2 ** (1.0 / self.beta) - 1)

    def _calc_bounding_box(self, *, factor=10.0):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the FWHM used to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.fwhm
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import MoffatPSF
        >>> model = MoffatPSF(x_0=0, y_0=0, alpha=2, beta=3)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-20.39298114135835, upper=20.39298114135835)
                y: Interval(lower=-20.39298114135835, upper=20.39298114135835)
            }
            model=MoffatPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-14.27508679895084, upper=14.27508679895084)
                y: Interval(lower=-14.27508679895084, upper=14.27508679895084)
            }
            model=MoffatPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, alpha, beta):
        """
        Calculate the value of the 2D Moffat model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        alpha : float
            The characteristic radius of the Moffat profile.

        beta : float
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances. Larger values provide less flux
            in the profile wings. For large ``beta``, this profile
            approaches a Gaussian profile. ``beta`` must be greater
            than 1. If ``beta`` is set to 1, then the Moffat profile is
            a Lorentz function, whose integral is infinite. For this
            normalized model, if ``beta`` is set to 1, then the profile
            will be zero everywhere.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        # Output units should match the input flux units
        alpha_norm = alpha
        if isinstance(alpha, u.Quantity):
            alpha_norm = alpha.value

        amp = flux * (beta - 1) / (np.pi * alpha_norm ** 2)
        r2 = (x - x_0) ** 2 + (y - y_0) ** 2
        return amp * (1 + (r2 / alpha**2)) ** (-beta)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, alpha, beta):
        """
        Calculate the partial derivatives of the 2D Moffat function with
        respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        alpha : float
            The characteristic radius of the Moffat profile.

        beta : float
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        dx = x - x_0
        dy = y - y_0
        r2_scaled = (dx**2 + dy**2) / alpha**2
        base = 1.0 + r2_scaled
        # The profile for unit flux without the (beta - 1) factor of
        # the normalization, which is kept apart so that the beta
        # derivative is finite at beta = 1
        profile = base ** (-beta) / (np.pi * alpha**2)
        model = flux * (beta - 1.0) * profile

        d_flux = (beta - 1.0) * profile
        position_factor = 2.0 * beta * model / (alpha**2 * base)
        d_x_0 = position_factor * dx
        d_y_0 = position_factor * dy
        d_alpha = 2.0 * model / alpha * (beta * r2_scaled / base - 1.0)
        d_beta = flux * profile * (1.0 - (beta - 1.0) * np.log(base))

        return [d_flux, d_x_0, d_y_0, d_alpha, d_beta]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'alpha': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class MoffatPRF(MoffatPSF):
    r"""
    A 2D Moffat PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Moffat function over
    the area of a pixel centered at each input position. Because it is
    integrated over pixels, this model is considered a PRF instead of a
    PSF. The response is assumed to be uniform across a pixel.

    The Moffat profile is normalized such that the analytical integral
    over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    alpha : float, optional
        The characteristic radius of the Moffat profile.

    beta : float, optional
        The asymptotic power-law slope of the Moffat profile wings at
        large radial distances. Larger values provide less flux in the
        profile wings. ``beta`` must be greater than 1.

    bbox_factor : float, optional
        The multiple of the FWHM used to define the bounding box limits.

    n_nodes : int, optional
        The number of Gauss-Legendre quadrature nodes along each axis of
        a pixel that are used to integrate the profile over the pixel.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    MoffatPSF, AiryDiskPRF, GaussianPRF, CircularGaussianPRF,
    CircularGaussianSigmaPRF

    Notes
    -----
    The model is the integral of the Moffat profile over a pixel of unit
    area that is centered at :math:`(x, y)`:

    .. math::

        f(x, y) = \int_{y - 0.5}^{y + 0.5} \int_{x - 0.5}^{x + 0.5}
            g(u, v) \,du \,dv

    where the Moffat profile is:

    .. math::

       g(u, v) = F \frac{\beta - 1}{\pi \alpha^2}
           \left(1 + \frac{\left(u - x_{0}\right)^{2}
               + \left(v - y_{0}\right)^{2}}{\alpha^{2}}\right)^{-\beta}

    :math:`F` is the total integrated flux and :math:`(x_{0}, y_{0})` is
    the position of the peak. Note that :math:`\beta` must be greater
    than 1.

    The integral over each pixel is computed with Gauss-Legendre
    quadrature using ``n_nodes`` nodes along each axis of the pixel.
    For ``beta`` of at least 1.5, the default of 9 nodes gives values
    that are accurate to better than :math:`10^{-4}` of the peak for
    a FWHM of 0.5 pixels and to better than :math:`10^{-7}` of the
    peak for a FWHM of at least 1 pixel. The errors are about twice as
    large for ``beta`` close to 1. The partial derivatives used for
    fitting are computed with the same quadrature.

    The profile is evaluated at ``n_nodes**2`` points in each pixel, so
    this model is much slower than `MoffatPSF`. With the default of 9
    nodes, it is about 10 times slower on the small cutouts used for
    fitting and about 80 times slower on a large image. A smaller
    ``n_nodes`` is faster and less accurate.

    Because the model is integrated over the pixels, its values on a
    grid with a spacing of one pixel sum to the total flux, for any
    subpixel position of the source:

    .. math::

        \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty}
            f(x + i, y + j) = F

    The FWHM of the Moffat profile before the integration over the
    pixels is given by:

    .. math::

        \rm{FWHM} = 2 \alpha \sqrt{2^{1 / \beta} - 1}

    The ``alpha`` and ``beta`` parameters are fixed by default. If
    you wish to fit these parameters, set the ``fixed`` attribute to
    `False`, e.g.,::

        >>> from photutils.psf import MoffatPRF
        >>> model = MoffatPRF()
        >>> model.alpha.fixed = False
        >>> model.beta.fixed = False

    By default, the ``alpha`` parameter is bounded to be strictly
    positive and the ``beta`` parameter is bounded to be greater than 1.

    Examples
    --------
    The values of an undersampled model on a grid of pixels sum to the
    flux for any subpixel position of the source:

    >>> import numpy as np
    >>> from photutils.psf import MoffatPRF
    >>> model = MoffatPRF(flux=71.4, x_0=50.3, y_0=49.8, alpha=0.9,
    ...                   beta=3.5)
    >>> yy, xx = np.mgrid[0:101, 0:101]
    >>> print(f'{model(xx, yy).sum():.2f}')
    71.40
    """

    def __init__(self, *, flux=MoffatPSF.flux.default,
                 x_0=MoffatPSF.x_0.default, y_0=MoffatPSF.y_0.default,
                 alpha=MoffatPSF.alpha.default, beta=MoffatPSF.beta.default,
                 bbox_factor=10.0, n_nodes=9, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, alpha=alpha, beta=beta,
                         bbox_factor=bbox_factor, **kwargs)
        self.n_nodes = n_nodes

    @property
    def n_nodes(self):
        """
        The number of Gauss-Legendre quadrature nodes along each axis
        of a pixel.
        """
        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, value):
        """
        Set the number of quadrature nodes along each axis of a pixel.

        Parameters
        ----------
        value : int
            The number of nodes, which must be a positive integer.
        """
        self._n_nodes = _validate_n_nodes(value)

    def evaluate(self, x, y, flux, x_0, y_0, alpha, beta):
        """
        Calculate the value of the pixel-integrated 2D Moffat model at
        the input coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        alpha : float
            The characteristic radius of the Moffat profile.

        beta : float
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        return _integrate_over_pixels(
            lambda xsub, ysub: MoffatPSF.evaluate(self, xsub, ysub, flux,
                                                  x_0, y_0, alpha, beta),
            x, y, (flux, x_0, y_0, alpha, beta), self.n_nodes)

    def fit_deriv(self, x, y, flux, x_0, y_0, alpha, beta):
        """
        Calculate the partial derivatives of the pixel-integrated 2D
        Moffat function with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        alpha : float
            The characteristic radius of the Moffat profile.

        beta : float
            The asymptotic power-law slope of the Moffat profile wings
            at large radial distances.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        return _integrate_over_pixels(
            lambda xsub, ysub: MoffatPSF.fit_deriv(xsub, ysub, flux, x_0,
                                                   y_0, alpha, beta),
            x, y, (flux, x_0, y_0, alpha, beta), self.n_nodes)


class AiryDiskPSF(Fittable2DModel):
    r"""
    A 2D Airy disk PSF model.

    This model is evaluated by sampling the 2D Airy disk function at the
    input coordinates. The Airy disk profile is normalized such that the
    analytical integral over the entire 2D plane is equal to the total
    flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    radius : float, optional
        The radius of the Airy disk at the first zero.

    bbox_factor : float, optional
        The multiple of the FWHM used to define the bounding box limits.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    AiryDiskPRF, GaussianPSF, CircularGaussianPSF, GaussianPRF,
    CircularGaussianPRF, CircularGaussianSigmaPRF, MoffatPSF

    Notes
    -----
    The Airy disk profile is defined as:

    .. math::

        f(r) = \frac{\pi F}{4 (R / R_z)^{2}}
               \left[ \frac{2 J_1\left(\frac{\pi r}{R / R_z}\right)}
                      {\frac{\pi r}{R / R_z}} \right]^2

    where :math:`r` is radial distance from the peak

    .. math::

        r = \sqrt{(x - x_0)^2 + (y - y_0)^2}

    :math:`F` is the total integrated flux,
    :math:`J_1` is the first order `Bessel function
    <https://en.wikipedia.org/wiki/Bessel_function>`_ of the first
    kind, :math:`R` is the input ``radius`` parameter, and :math:`R_z =
    1.2196698912665045` is the solution to the equation :math:`J_1(\pi
    R_z) = 0`.

    For an optical system, the radius of the first zero represents
    the limiting angular resolution. The limiting angular resolution
    is :math:`R_z \, \lambda / D \approx 1.22 \, \lambda / D`, where
    :math:`\lambda` is the wavelength of the light and :math:`D` is the
    diameter of the aperture.

    The full width at half maximum (FWHM) of the Airy disk profile is
    given by:

    .. math::

        \rm{FWHM} = 1.028993969962188 \, \frac{R}{R_z}
                  = 0.8436659602162364 \, R

    The model is normalized such that:

    .. math::

        \int_{0}^{2 \pi} \int_{0}^{\infty} f(r) \,r \,dr \,d\theta =
        \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y)
            \,dx \,dy = F

    The ``radius`` parameter is fixed by default. If you wish to fit
    this parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import AiryDiskPSF
        >>> model = AiryDiskPSF()
        >>> model.radius.fixed = False

    By default, the ``radius`` parameter is bounded to be strictly
    positive.

    This model is evaluated at the input coordinates and is not
    integrated over the detector pixels. Its values on a grid of
    detector pixels are the values of the PSF at the pixel centers, not
    the fluxes in the pixels. For a PSF that is undersampled by the
    detector pixels, the model is sharper than a source in the data
    and the sum of its values over the pixels depends on the subpixel
    position of the source.

    This model should therefore not be used to fit the pixel values of
    an image, e.g., for PSF photometry. If its shape parameters are
    fixed to those of the PSF before the integration over the pixels,
    the fitted flux is biased by a few percent for a FWHM of 2 to 3
    pixels, and by much more, with an error that depends on the subpixel
    position of the source, for a FWHM of less than about 1.5 pixels.
    Use `AiryDiskPRF` instead, which is integrated over the pixels.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Airy_disk

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.visualization import simple_norm
        from photutils.psf import AiryDiskPSF

        model = AiryDiskPSF(flux=71.4, x_0=24.3, y_0=25.2, radius=5)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots()
        ax.imshow(data, norm=norm, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the entire PSF.')
    x_0 = Parameter(
        default=0, description='Position of the peak along the x axis')
    y_0 = Parameter(
        default=0, description='Position of the peak along the y axis')
    radius = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='Radius of the Airy disk at the first zero')

    # The radius of the first zero of the Airy disk profile, in units of
    # the profile scale radius, i.e., the solution of J1(pi * Rz) = 0.
    _rz = jn_zeros(1, 1)[0] / np.pi

    # The half width at half maximum of the Airy disk profile, i.e., the
    # solution of (2 * J1(x) / x)**2 = 1/2.
    _hwhm = 1.616339948310703

    def __init__(self, *, flux=flux.default, x_0=x_0.default, y_0=y_0.default,
                 radius=radius.default, bbox_factor=10.0, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, radius=radius, **kwargs)
        self.bbox_factor = bbox_factor

    @property
    def fwhm(self):
        """
        The FWHM of the Airy disk profile.
        """
        return 2.0 * self._hwhm * self.radius / self._rz / np.pi

    def _calc_bounding_box(self, *, factor=10.0):
        """
        Calculate a bounding box defining the limits of the model.

        Parameters
        ----------
        factor : float, optional
            The multiple of the FWHM used to define the limits.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        delta = factor * self.fwhm
        return ((self.y_0 - delta, self.y_0 + delta),
                (self.x_0 - delta, self.x_0 + delta))

    @property
    def bounding_box(self):
        """
        The bounding box of the model.

        Examples
        --------
        >>> from photutils.psf import AiryDiskPSF
        >>> model = AiryDiskPSF(x_0=0, y_0=0, radius=3)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-25.30997880648709, upper=25.30997880648709)
                y: Interval(lower=-25.30997880648709, upper=25.30997880648709)
            }
            model=AiryDiskPSF(inputs=('x', 'y'))
            order='C'
        )
        >>> model.bbox_factor = 7
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-17.71698516454096, upper=17.71698516454096)
                y: Interval(lower=-17.71698516454096, upper=17.71698516454096)
            }
            model=AiryDiskPSF(inputs=('x', 'y'))
            order='C'
        )
        """
        return self._calc_bounding_box(factor=self.bbox_factor)

    def evaluate(self, x, y, flux, x_0, y_0, radius):
        """
        Calculate the value of the 2D Airy disk model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        radius : float
            The radius of the Airy disk at the first zero.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / self._rz)

        if isinstance(r, u.Quantity):
            # Convert to dimensionless_unscaled to avoid unit conversion
            # issues in scipy functions, since they expect dimensionless
            # inputs.
            r = r.to_value(u.dimensionless_unscaled)
        r = np.asarray(r, dtype=float)

        # Since r can be zero, we have to take care to treat that case
        # separately so as not to raise a numpy warning. The limit of
        # (2 * J1(x) / x)**2 as x approaches zero is 1.
        rt = np.pi * np.atleast_1d(r)
        z = np.ones(rt.shape)
        nonzero = rt > 0
        z[nonzero] = (2.0 * j1(rt[nonzero]) / rt[nonzero]) ** 2
        # The model tends to zero at infinite radial distance, but
        # j1(inf) is NaN, so set the limit explicitly
        z[np.isinf(rt)] = 0.0
        z[np.isnan(rt)] = np.nan
        z = z.reshape(r.shape)

        normalization = (4.0 / np.pi) * (radius / self._rz) ** 2
        if isinstance(normalization, u.Quantity):
            normalization = normalization.value

        # Not an in-place multiplication because a flux array can have
        # more dimensions than the input coordinates
        return z * (flux / normalization)

    @staticmethod
    def fit_deriv(x, y, flux, x_0, y_0, radius):
        """
        Calculate the partial derivatives of the 2D Airy disk function
        with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        radius : float
            The radius of the Airy disk at the first zero.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        scale = radius / AiryDiskPSF._rz
        dx, dy = np.broadcast_arrays(x - x_0, y - y_0)
        rt = np.pi * np.hypot(dx, dy) / scale
        rt = np.atleast_1d(np.asarray(rt, dtype=float))

        # The profile is (2 J1(t) / t)**2 and its derivative with
        # respect to t is -8 J1(t) J2(t) / t**2. Both are computed
        # from J1(t) / t and J2(t) / t**2, whose limits as t approaches
        # zero are 1/2 and 1/8.
        j1_ratio = np.full(rt.shape, 0.5)
        nonzero = rt > 0
        j1_ratio[nonzero] = j1(rt[nonzero]) / rt[nonzero]

        # J2(t) comes from the recurrence J2(t) = 2 J1(t) / t - J0(t),
        # which is much faster than evaluating it directly. The
        # recurrence loses precision as t approaches zero, where the
        # power series of J2(t) / t**2 is used instead.
        j2_ratio = np.empty(rt.shape)
        small = rt < 0.3
        rt2 = rt[small] ** 2
        j2_ratio[small] = 0.125 * (1.0 - rt2 / 12.0 * (
            1.0 - rt2 / 32.0 * (1.0 - rt2 / 60.0 * (1.0 - rt2 / 96.0))))
        large = ~small
        j2_ratio[large] = ((2.0 * j1_ratio[large] - j0(rt[large]))
                           / rt[large] ** 2)
        shape = np.shape(dx)
        j1_ratio = j1_ratio.reshape(shape)
        j2_ratio = j2_ratio.reshape(shape)
        rt = rt.reshape(shape)

        amplitude = np.pi / (4.0 * scale**2)
        profile = 4.0 * j1_ratio**2
        # The derivative of the profile divided by t, which is finite
        # at t = 0
        d_profile_over_t = -8.0 * j1_ratio * j2_ratio

        d_flux = amplitude * profile
        position_factor = (-flux * amplitude * d_profile_over_t
                           * (np.pi / scale) ** 2)
        d_x_0 = position_factor * dx
        d_y_0 = position_factor * dy
        d_scale = (-flux * amplitude / scale
                   * (2.0 * profile + d_profile_over_t * rt**2))
        d_radius = d_scale / AiryDiskPSF._rz

        return [d_flux, d_x_0, d_y_0, d_radius]

    @property
    def input_units(self):
        """
        The input units of the model.
        """
        x_unit = self.x_0.input_unit
        y_unit = self.y_0.input_unit
        if x_unit is None and y_unit is None:
            return None

        return {self.inputs[0]: x_unit, self.inputs[1]: y_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'x_0': inputs_unit[self.inputs[0]],
                'y_0': inputs_unit[self.inputs[0]],
                'radius': inputs_unit[self.inputs[0]],
                'flux': outputs_unit[self.outputs[0]]}


class AiryDiskPRF(AiryDiskPSF):
    r"""
    A 2D Airy disk PSF model integrated over pixels.

    This model is evaluated by integrating the 2D Airy disk function
    over the area of a pixel centered at each input position. Because it
    is integrated over pixels, this model is considered a PRF instead of
    a PSF. The response is assumed to be uniform across a pixel.

    The Airy disk profile is normalized such that the analytical
    integral over the entire 2D plane is equal to the total flux.

    Parameters
    ----------
    flux : float, optional
        Total integrated flux over the entire PSF.

    x_0 : float, optional
        Position of the peak along the x-axis.

    y_0 : float, optional
        Position of the peak along the y-axis.

    radius : float, optional
        The radius of the Airy disk at the first zero.

    bbox_factor : float, optional
        The multiple of the FWHM used to define the bounding box limits.

    n_nodes : int, optional
        The number of Gauss-Legendre quadrature nodes along each axis of
        a pixel that are used to integrate the profile over the pixel.

    **kwargs : dict, optional
        Additional optional keyword arguments to be passed to the
        `astropy.modeling.Model` base class.

    See Also
    --------
    AiryDiskPSF, MoffatPRF, GaussianPRF, CircularGaussianPRF,
    CircularGaussianSigmaPRF

    Notes
    -----
    The model is the integral of the Airy disk profile over a pixel of
    unit area that is centered at :math:`(x, y)`:

    .. math::

        f(x, y) = \int_{y - 0.5}^{y + 0.5} \int_{x - 0.5}^{x + 0.5}
            g(u, v) \,du \,dv

    where the Airy disk profile is:

    .. math::

        g(u, v) = \frac{\pi F}{4 (R / R_z)^{2}}
               \left[ \frac{2 J_1\left(\frac{\pi r}{R / R_z}\right)}
                      {\frac{\pi r}{R / R_z}} \right]^2

    .. math::

        r = \sqrt{(u - x_0)^2 + (v - y_0)^2}

    :math:`F` is the total integrated flux, :math:`(x_{0}, y_{0})` is
    the position of the peak, :math:`J_1` is the first order `Bessel
    function <https://en.wikipedia.org/wiki/Bessel_function>`_ of
    the first kind, :math:`R` is the input ``radius`` parameter, and
    :math:`R_z = 1.2196698912665045` is the solution to the equation
    :math:`J_1(\pi R_z) = 0`.

    The integral over each pixel is computed with Gauss-Legendre
    quadrature using ``n_nodes`` nodes along each axis of the pixel. The
    default of 9 nodes gives values that are accurate to better than
    :math:`10^{-7}` of the peak for a FWHM of at least 0.5 pixels.
    The partial derivatives used for fitting are computed with the same
    quadrature.

    The profile is evaluated at ``n_nodes**2`` points in each pixel, so
    this model is much slower than `AiryDiskPSF`. With the default of 9
    nodes, it is about 10 times slower on the small cutouts used for
    fitting and about 80 times slower on a large image. A smaller
    ``n_nodes`` is faster and less accurate.

    Because the model is integrated over the pixels, its values on a
    grid with a spacing of one pixel sum to the total flux, for any
    subpixel position of the source:

    .. math::

        \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty}
            f(x + i, y + j) = F

    The flux in the wings of an Airy disk decreases slowly with the
    distance from the peak, so the sum over a finite grid of pixels is
    smaller than the total flux.

    The FWHM of the Airy disk profile before the integration over the
    pixels is given by:

    .. math::

        \rm{FWHM} = 1.028993969962188 \, \frac{R}{R_z}
                  = 0.8436659602162364 \, R

    The ``radius`` parameter is fixed by default. If you wish to fit
    this parameter, set the ``fixed`` attribute to `False`, e.g.,::

        >>> from photutils.psf import AiryDiskPRF
        >>> model = AiryDiskPRF()
        >>> model.radius.fixed = False

    By default, the ``radius`` parameter is bounded to be strictly
    positive.

    Examples
    --------
    The values of an undersampled model on a grid of pixels sum to the
    same fraction of the flux for any subpixel position of the source:

    >>> import numpy as np
    >>> from photutils.psf import AiryDiskPRF
    >>> yy, xx = np.mgrid[0:101, 0:101]
    >>> for x_0, y_0 in ((50.0, 50.0), (50.5, 50.5), (50.3, 49.8)):
    ...     model = AiryDiskPRF(flux=100.0, x_0=x_0, y_0=y_0, radius=1.0)
    ...     print(f'{model(xx, yy).sum():.2f}')
    99.70
    99.70
    99.70
    """

    def __init__(self, *, flux=AiryDiskPSF.flux.default,
                 x_0=AiryDiskPSF.x_0.default, y_0=AiryDiskPSF.y_0.default,
                 radius=AiryDiskPSF.radius.default, bbox_factor=10.0,
                 n_nodes=9, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, radius=radius,
                         bbox_factor=bbox_factor, **kwargs)
        self.n_nodes = n_nodes

    @property
    def n_nodes(self):
        """
        The number of Gauss-Legendre quadrature nodes along each axis
        of a pixel.
        """
        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, value):
        """
        Set the number of quadrature nodes along each axis of a pixel.

        Parameters
        ----------
        value : int
            The number of nodes, which must be a positive integer.
        """
        self._n_nodes = _validate_n_nodes(value)

    def evaluate(self, x, y, flux, x_0, y_0, radius):
        """
        Calculate the value of the pixel-integrated 2D Airy disk model
        at the input coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        radius : float
            The radius of the Airy disk at the first zero.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
        """
        return _integrate_over_pixels(
            lambda xsub, ysub: AiryDiskPSF.evaluate(self, xsub, ysub, flux,
                                                    x_0, y_0, radius),
            x, y, (flux, x_0, y_0, radius), self.n_nodes)

    def fit_deriv(self, x, y, flux, x_0, y_0, radius):
        """
        Calculate the partial derivatives of the pixel-integrated 2D
        Airy disk function with respect to the parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float
            Total integrated flux over the entire PSF.

        x_0, y_0 : float
            Position of the peak along the x and y axes.

        radius : float
            The radius of the Airy disk at the first zero.

        Returns
        -------
        result : list of `~numpy.ndarray`
            The list of partial derivatives with respect to each
            parameter.
        """
        return _integrate_over_pixels(
            lambda xsub, ysub: AiryDiskPSF.fit_deriv(xsub, ysub, flux, x_0,
                                                     y_0, radius),
            x, y, (flux, x_0, y_0, radius), self.n_nodes)
