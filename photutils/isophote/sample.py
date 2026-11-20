# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for sampling data along an elliptical path.
"""

import copy

import astropy.units as u
import numpy as np

from photutils.isophote._polar_path import compute_polar_path
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import INTEGRATORS
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

# Remove in 4.0
_DEPRECATED_SAMPLE_ATTRIBUTES = {
    'gradient_error': 'gradient_err',
    'gradient_relative_error': 'gradient_rel_err',
    'nclip': 'n_clip',
}

__all__ = ['EllipseSample']

# The relative distances (as fractions of the semimajor axis length)
# of the samples used to measure the local radial gradient, as in the
# STSDAS ellipse task. The second distance is used only if the first one
# does not give a meaningful gradient.
_GRADIENT_RADIUS = 0.1
_GRADIENT_RADIUS_RETRY = 0.2


def _validate_image(image):
    """
    Validate the input image for the isophote fitting tools.

    Parameters
    ----------
    image : 2D array_like
        The input image.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        The input image as an array. A `~numpy.ma.MaskedArray` is
        returned unchanged.

    Raises
    ------
    TypeError
        If ``image`` is a `~astropy.units.Quantity` array.

    ValueError
        If ``image`` is not a 2D array.
    """
    if isinstance(image, u.Quantity):
        msg = ('image must not be a Quantity array. Input the image '
               'values without units, e.g., image.value.')
        raise TypeError(msg)
    image = np.asanyarray(image)
    if image.ndim != 2:
        msg = 'image must be a 2D array'
        raise ValueError(msg)

    return image


class EllipseSample:
    """
    Class to sample image data along an elliptical path.

    The image intensities along the elliptical path can be extracted
    using a selection of integration algorithms.

    The ``geometry`` attribute describes the geometry of the elliptical
    path.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The input image. Masked pixels (if ``image`` is a
        `~numpy.ma.MaskedArray`) and nonfinite (NaN or infinity) values
        are excluded from the sample.
    sma : float
        The semimajor axis length in pixels.
    x0, y0 : float, optional
        The (x, y) coordinate of the ellipse center. The coordinates
        must be finite. A coordinate that is `None` (the default) is
        set to the image center.
    astep : float, optional
        The step value for growing/shrinking the semimajor axis. It can
        be expressed either in pixels (when ``linear_growth=True``) or
        as a relative value (when ``linear_growth=False``). The default
        is 0.1.
    eps : float, optional
        The ellipticity of the ellipse. It must be less than 1. The
        default is 0.2.
    position_angle : float, optional
        The position angle of ellipse in relation to the positive x-axis
        of the image array (rotating towards the positive y-axis). The
        default is 0.
    sclip : float, optional
        The sigma-clip sigma value. The default is 3.0.
    n_clip : int, optional
        The number of sigma-clip iterations. Set to zero to skip
        sigma-clipping. The default is 0.

        .. deprecated:: 3.0
            The ``nclip`` keyword is deprecated. Use ``n_clip`` instead.
    linear_growth : bool, optional
        The semimajor axis growing/shrinking mode. The default is
        `False`.
    integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, optional
        The area integration mode. The default is 'bilinear'.
    geometry : `~photutils.isophote.EllipseGeometry` instance or `None`
        The geometry that describes the ellipse. This can be
        used in lieu of the explicit specification of parameters
        ``sma``, ``x0``, ``y0``, ``eps``, etc. In any case, the
        `~photutils.isophote.EllipseGeometry` instance becomes an
        attribute of the `~photutils.isophote.EllipseSample` object. The
        default is `None`.

    Attributes
    ----------
    values : 2D `~numpy.ndarray`
        The sampled values as a 2D array, where the rows contain the
        angles, radii, and extracted intensity values, respectively.
    mean : float
        The mean intensity along the elliptical path.
    geometry : `~photutils.isophote.EllipseGeometry` instance
        The geometry of the elliptical path.
    gradient : float
        The local radial intensity gradient.
    gradient_err : float
        The error associated with the local radial intensity gradient.
    gradient_rel_err : float
        The relative error associated with the local radial intensity
        gradient.
    sector_area : float
        The average area of the sectors along the elliptical path from
        which the sample values were integrated.
    total_points : int
        The total number of sample values that would cover the entire
        elliptical path.
    actual_points : int
        The actual number of sample values that were taken from the
        image. It can be smaller than ``total_points`` when the ellipse
        encompasses regions outside the image, when it encompasses
        masked or nonfinite pixels, or when sigma-clipping removed some
        of the points.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2,
                 position_angle=0.0, sclip=3.0, n_clip=0,
                 linear_growth=False,
                 integrmode='bilinear', geometry=None):
        image = _validate_image(image)
        self.image = image
        self.integrmode = integrmode

        if geometry:
            # When the geometry is inherited from somewhere else, its
            # sma attribute must be replaced by the value explicitly
            # passed to the constructor.
            self.geometry = copy.deepcopy(geometry)
            self.geometry.sma = sma
            self.geometry._reset_sma_dependent_attributes()
        else:
            # A center coordinate that was not specified is assumed to
            # be roughly coincident with the image center.
            _x0 = image.shape[1] / 2 if x0 is None else x0
            _y0 = image.shape[0] / 2 if y0 is None else y0

            self.geometry = EllipseGeometry(_x0, _y0, sma, eps,
                                            position_angle, astep=astep,
                                            linear_growth=linear_growth)

        # Sigma-clip parameters
        self.sclip = sclip
        self.n_clip = n_clip

        # Extracted values associated with this sample.
        self.values = None
        self._updated_values = None
        self.mean = None
        self.gradient = None
        self.gradient_err = None
        self.gradient_rel_err = None
        self.sector_area = None

        # The gradient that tells whether a measured gradient is
        # meaningful. It is the gradient from the last update of this
        # sample, or the gradient of the previous iteration set by the
        # fitter.
        self._reference_gradient = None

        # The total_points attribute is the number of angle-radius pairs
        # that were attempted. The actual_points attribute is the number
        # of sampled pairs that resulted in valid values.
        self.total_points = 0
        self.actual_points = 0

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _DEPRECATED_SAMPLE_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def extract(self):
        """
        Extract sample data by scanning an elliptical path over the
        image array.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The rows of the array contain the angles, radii, and
            extracted intensity values, respectively.

        Raises
        ------
        ValueError
            If the ellipticity of the ellipse is 1 or larger or if its
            center is not finite.
        """
        # The sample values themselves are kept cached to prevent
        # multiple calls to the integrator code.
        if self.values is not None:
            return self.values

        # An ellipse with an ellipticity of 1 or larger or a nonfinite
        # center has no elliptical path. The geometry is checked here
        # instead of in the constructor because the fitter makes trial
        # samples with any geometry and discards the invalid ones
        # without extracting them.
        geometry = self.geometry
        if geometry.eps >= 1.0:
            msg = f'The ellipticity must be less than 1, got {geometry.eps}'
            raise ValueError(msg)
        if not (np.isfinite(geometry.x0) and np.isfinite(geometry.y0)):
            msg = ('The center of the ellipse must be finite, got '
                   f'({geometry.x0}, {geometry.y0})')
            raise ValueError(msg)

        s = self._extract()
        self.values = s
        return s

    def _extract(self, *, phi_min=0.05):
        # Here the actual sampling takes place. This is called only once
        # during the life of an EllipseSample instance, because it's
        # an expensive calculation. This method should not be called
        # from external code. To force it to rerun, set "sample.values =
        # None" before calling sample.extract().

        # Individual extracted sample points will be stored in here.
        angles = []
        radii = []
        intensities = []
        sector_areas = []

        # Reset counters.
        self.total_points = 0
        self.actual_points = 0

        # Build integrator.
        integrator = INTEGRATORS[self.integrmode](self.image, self.geometry,
                                                  angles, radii, intensities)

        # Initialize walk along elliptical path.
        radius = self.geometry.initial_polar_radius
        phi = self.geometry.initial_polar_angle

        # In case of an area integrator, ask the integrator to deliver
        # a hint of how much area the sectors will have. In case of too
        # small areas, tests showed that the area integrators (mean,
        # median) won't perform properly. In that case, we override the
        # caller's selection and use the bilinear integrator regardless.
        if integrator.is_area():
            integrator.integrate(radius, phi)
            area = integrator.get_sector_area()
            # This integration that just took place messes up with the
            # storage arrays and the constructors. We have to build a
            # new integrator instance from scratch, even if it is the
            # same kind as originally selected by the caller.
            angles = []
            radii = []
            intensities = []
            if area < 1.0:
                integrator = INTEGRATORS['bilinear'](
                    self.image, self.geometry, angles, radii, intensities)
            else:
                integrator = INTEGRATORS[self.integrmode](self.image,
                                                          self.geometry,
                                                          angles, radii,
                                                          intensities)

        # Walk along elliptical path, integrating at specified places
        # defined by polar vector. Need to go a bit beyond full circle
        # to ensure full coverage.
        phi_max = np.pi * 2.0 + phi_min
        if integrator.is_area():
            while phi <= phi_max:
                # Do the integration at phi-radius position, and append
                # results to the angles, radii, and intensities lists.
                integrator.integrate(radius, phi)

                # Store sector area locally.
                sector_areas.append(integrator.get_sector_area())

                # Update total number of points.
                self.total_points += 1

                # Update angle and radius to be used to define next
                # polar vector along the elliptical path.
                phistep_ = integrator.get_polar_angle_step()
                phi += min(phistep_, 0.5)
                radius = self.geometry.radius(phi)
        else:
            # The pixel integrators step along the path by about one
            # pixel (an angle step of 1 / radius, but at most 0.5
            # radians), so the path can be computed first and then
            # sampled at once.
            phis, path_radii = compute_polar_path(
                phi, radius, phi_max, self.geometry.sma, self.geometry.eps)
            angles, radii, intensities = integrator.integrate_path(
                path_radii, phis)
            sector_areas.append(integrator.get_sector_area())
            self.total_points = len(phis)

        # Average sector area is calculated after the integrator had
        # the opportunity to step over the entire elliptical path.
        self.sector_area = np.mean(np.array(sector_areas))

        # Apply sigma-clipping.
        angles, radii, intensities = self._sigma_clip(angles, radii,
                                                      intensities)

        # Actual number of sampled points, after sigma-clip removed outliers
        self.actual_points = len(angles)

        # Pack results in 2D array.
        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])

    def _sigma_clip(self, angles, radii, intensities):
        # Can't use scipy or astropy tools because they use masked
        # arrays. Also, they operate on a single array, and we need to
        # operate on three arrays simultaneously. We need something
        # that physically removes the clipped points from the arrays,
        # since that is what the remaining of the `ellipse` code
        # expects.
        angles = np.asarray(angles)
        radii = np.asarray(radii)
        intensities = np.asarray(intensities)

        for _ in range(self.n_clip):
            if intensities.size == 0:
                break
            mean = np.mean(intensities)
            sig = np.std(intensities)
            lower = mean - self.sclip * sig
            upper = mean + self.sclip * sig
            keep = (intensities >= lower) & (intensities <= upper)
            angles = angles[keep]
            radii = radii[keep]
            intensities = intensities[keep]

        return angles, radii, intensities

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def update(self, fixed_parameters=None):
        """
        Update this `~photutils.isophote.EllipseSample` instance.

        This method calls the
        :meth:`~photutils.isophote.EllipseSample.extract` method to get
        the values that match the current ``geometry`` attribute, and
        then computes the mean intensity, local gradient, and other
        associated quantities.

        The local radial gradient is measured from a second sample on
        an ellipse with a 10% larger semimajor axis length (or 20%
        larger if the first estimate is not meaningful), as in the
        STSDAS ``ellipse`` task, regardless of the ``astep`` value.

        The mean intensity and the gradient are computed only if the
        sample values changed since the last call.

        Parameters
        ----------
        fixed_parameters : `None` or array_like, optional
            An array of the fixed parameters. Must have 4 elements,
            corresponding to x center, y center, PA, and EPS. If
            `None`, the fixed parameters of the ``geometry`` attribute
            are not changed.
        """
        if fixed_parameters is not None:
            self.geometry.fix = fixed_parameters

        # Update the mean value first, using extraction from main sample.
        s = self.extract()

        # The mean and the gradient depend only on the sample values, so
        # they are not computed again if the values did not change since
        # the last update. This avoids extracting the gradient sample
        # again.
        if s is self._updated_values:
            return
        self._updated_values = s

        # If the elliptical path falls entirely outside the image (e.g.,
        # a trial geometry that diverged during the fit), no points are
        # sampled. Short-circuit here to avoid emitting numpy warnings
        # (e.g., "Mean of empty slice") from operating on empty arrays.
        # The fitter detects this condition separately via its own
        # zero-length check and marks the isophote as invalid.
        if len(s[2]) == 0:
            self.mean = np.nan
            self.gradient = np.nan
            self.gradient_err = np.nan
            self.gradient_rel_err = None
            return

        self.mean = np.mean(s[2])

        # Get sample with same geometry but at a different distance from
        # center. Estimate gradient from there.
        gradient, gradient_err = self._get_gradient(_GRADIENT_RADIUS)

        # Check for meaningful gradient. If no meaningful gradient, try
        # another sample, this time using larger radius. Meaningful
        # gradient means something  shallower, but still close to within
        # a factor 3 from previous gradient estimate. If no previous
        # estimate is available, guess it by adding the error to the
        # current gradient.
        previous_gradient = self._reference_gradient
        if not previous_gradient:
            previous_gradient = gradient + gradient_err

        # A meaningful gradient must also be negative (the isophote
        # intensity decreases outward). A gradient of exactly zero can
        # occur with the nearest-neighbor integrator when the samples at
        # both radii extract identical pixel sets, and it would cause
        # divisions by zero in the fitter's parameter correctors.
        if (gradient >= (previous_gradient / 3.0)  # gradient is negative!
                or gradient >= 0.0):
            gradient, gradient_err = self._get_gradient(
                _GRADIENT_RADIUS_RETRY)

        # If still no meaningful gradient can be measured, try with
        # previous one, slightly shallower. A factor 0.8 is not too far
        # from what is expected from geometrical sampling steps of 10-20%
        # and a deVaucouleurs law or an exponential disk (at least at its
        # inner parts, r <~ 5 req). Gradient error is meaningless in this
        # case.
        if gradient >= (previous_gradient / 3.0) or gradient >= 0.0:
            gradient = previous_gradient * 0.8
            gradient_err = None

        self.gradient = gradient
        self._reference_gradient = gradient
        self.gradient_err = gradient_err
        if gradient_err and gradient < 0.0:
            self.gradient_rel_err = gradient_err / np.abs(gradient)
        else:
            self.gradient_rel_err = None

    def _make_sample(self, *, sma=None, x0=None, y0=None, eps=None,
                     pa=None):
        """
        Make a new sample of the same image, with the same settings and
        with the given ellipse parameters changed.

        The new sample is not extracted. The ellipse parameters that
        are `None` are taken from this sample.
        """
        geometry = self.geometry
        return EllipseSample(
            self.image, geometry.sma if sma is None else sma,
            x0=geometry.x0 if x0 is None else x0,
            y0=geometry.y0 if y0 is None else y0,
            astep=geometry.astep, sclip=self.sclip, n_clip=self.n_clip,
            eps=geometry.eps if eps is None else eps,
            position_angle=geometry.pa if pa is None else pa,
            linear_growth=geometry.linear_growth,
            integrmode=self.integrmode)

    def _get_gradient(self, radius):
        # The gradient sample is at a relative distance ``radius`` from
        # this ellipse.
        gradient_sma = (1.0 + radius) * self.geometry.sma

        gradient_sample = self._make_sample(sma=gradient_sma)

        sg = gradient_sample.extract()

        # If the gradient sample (taken at a slightly larger semimajor
        # axis) falls entirely outside the image, no points are sampled.
        # Return a nonmeaningful gradient without operating on empty
        # arrays (which would emit numpy warnings). NaN (rather than
        # None) is returned to match the previous numeric behavior so
        # that downstream arithmetic continues to work.
        if len(sg[2]) == 0:
            return np.nan, np.nan

        mean_g = np.mean(sg[2])
        gradient = (mean_g - self.mean) / self.geometry.sma / radius

        s = self.extract()
        sigma = np.std(s[2])
        sigma_g = np.std(sg[2])

        gradient_err = (np.sqrt(sigma**2 / len(s[2])
                                + sigma_g**2 / len(sg[2]))
                        / self.geometry.sma / radius)

        return gradient, gradient_err

    def coordinates(self):
        """
        Return the (x, y) coordinates associated with each sampled
        point.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinate arrays.
        """
        angles = self.values[0]
        radii = self.values[1]
        x = radii * np.cos(angles + self.geometry.pa) + self.geometry.x0
        y = radii * np.sin(angles + self.geometry.pa) + self.geometry.y0
        return x, y


class CentralEllipseSample(EllipseSample):
    """
    An `~photutils.isophote.EllipseSample` subclass designed to handle
    the special case of the central pixel in the galaxy image.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def update(self, fixed_parameters=None):  # noqa: ARG002
        """
        Update this `~photutils.isophote.EllipseSample` instance with
        the intensity integrated at the (x0, y0) center position using
        bilinear integration. The local gradient is set to `None`.

        Parameters
        ----------
        fixed_parameters : `None` or array_like, optional
            An array of the fixed parameters. Must have 4 elements,
            corresponding to x center, y center, PA, and EPS. This
            keyword is ignored in this subclass.
        """
        s = self.extract()

        # The central pixel may be masked or nonfinite.
        self.mean = s[2][0] if self.actual_points > 0 else np.nan

        self.gradient = None
        self.gradient_err = None
        self.gradient_rel_err = None

    def _extract(self):
        angles = []
        radii = []
        intensities = []

        integrator = INTEGRATORS['bilinear'](self.image, self.geometry,
                                             angles, radii, intensities)
        integrator.integrate(0.0, 0.0)

        self.total_points = 1
        self.actual_points = len(intensities)

        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])
