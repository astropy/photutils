# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for sampling data along an elliptical path.
"""

import copy
import warnings

import astropy.units as u
import numpy as np

from photutils.isophote._polar_path import compute_polar_path
from photutils.isophote.geometry import (EllipseGeometry,
                                         _legacy_orientation_rad,
                                         _orientation_argument)
from photutils.isophote.integrator import _INTEGRATORS
from photutils.utils._deprecation import (deprecated_attribute,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils.exceptions import PhotutilsDeprecationWarning

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

# The largest relative error of a gradient that is meaningful by itself,
# without being compared to a reference gradient.
_MAX_GRADIENT_REL_ERR = 0.2


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
    image : 2D array_like
        The input image. Masked pixels (if ``image`` is a
        `~numpy.ma.MaskedArray`) and nonfinite (NaN or infinity) values
        are excluded from the sample.
    sma : float
        The semimajor axis length in pixels.
    x0, y0 : float, optional
        The (x, y) coordinate of the ellipse center. The coordinates
        must be finite. A coordinate that is `None` (the default) is
        set to half the image size along that axis, i.e.,
        ``image.shape[1] / 2`` for ``x0`` and ``image.shape[0] / 2``
        for ``y0``. This is half a pixel larger than the image center
        in the pixel-center convention, ``(n - 1) / 2``.
    step : float, optional
        The step value for growing/shrinking the semimajor axis. It can
        be expressed either in pixels (when ``linear_growth=True``) or
        as a relative value (when ``linear_growth=False``). The default
        is 0.1.

        .. deprecated:: 3.2
            The ``astep`` keyword is deprecated. Use ``step`` instead.
    ellipticity : float, optional
        The ellipticity of the ellipse. It must be less than 1. The
        default is 0.2.

        .. deprecated:: 3.2
            The ``eps`` keyword is deprecated. Use ``ellipticity``
            instead.
    orientation : `~astropy.units.Quantity`, optional
        The orientation of the semimajor axis of the ellipse as an
        angular `~astropy.units.Quantity` (e.g., ``30 * u.deg``),
        measured from the positive x-axis of the image array towards
        the positive y-axis. The default is 0 degrees.

        .. deprecated:: 3.2
            The ``position_angle`` keyword is deprecated. Use
            ``orientation`` instead. A number without units is
            accepted as radians with the ``position_angle`` keyword or
            as a positional argument.
    clip_sigma : float, optional
        The sigma-clip sigma value. The default is 3.0.

        .. deprecated:: 3.2
            The ``sclip`` keyword is deprecated. Use ``clip_sigma``
            instead.
    n_clip : int, optional
        The number of sigma-clip iterations. Set to zero to skip
        sigma clipping. The default is 0.

        .. deprecated:: 3.0
            The ``nclip`` keyword is deprecated. Use ``n_clip`` instead.
    linear_growth : bool, optional
        The semimajor axis growing/shrinking mode. The default is
        `False`.
    integration_mode : {'bilinear', 'nearest_neighbor', 'mean', \
            'median'}, optional
        The integration mode. The 'mean' and 'median' modes integrate
        over elliptical sectors and fall back to 'bilinear' for sectors
        smaller than 1 pixel. The default is 'bilinear'.

        .. deprecated:: 3.2
            The ``integrmode`` keyword is deprecated. Use
            ``integration_mode`` instead.
    geometry : `~photutils.isophote.EllipseGeometry` or `None`, optional
        The geometry that describes the ellipse. If input, a copy of it
        is used with its semimajor axis length set to ``sma``, and the
        ``x0``, ``y0``, ``step``, ``ellipticity``, ``orientation``, and
        ``linear_growth`` keywords are ignored. The input geometry is
        not modified. The default is `None`.

    Attributes
    ----------
    image : 2D `~numpy.ndarray`
        The input image as an array. A `~numpy.ma.MaskedArray` input is
        kept as is.
    integration_mode : str
        The integration mode.
    integrmode : str
        .. deprecated:: 3.2
            Use ``integration_mode`` instead.
    clip_sigma : float
        The sigma-clip sigma value.
    sclip : float
        .. deprecated:: 3.2
            Use ``clip_sigma`` instead.
    n_clip : int
        The number of sigma-clip iterations.
    nclip : int
        .. deprecated:: 3.0
            Use ``n_clip`` instead.
    angles : 1D `~numpy.ndarray` or `None`
        The angles of the valid sampled points. It is `None` until
        :meth:`~photutils.isophote.EllipseSample.extract` is called.
    radii : 1D `~numpy.ndarray` or `None`
        The radii of the valid sampled points. It is `None` until
        :meth:`~photutils.isophote.EllipseSample.extract` is called.
    intensities : 1D `~numpy.ndarray` or `None`
        The extracted intensity values of the valid sampled points. It
        is `None` until
        :meth:`~photutils.isophote.EllipseSample.extract` is called.
    values : 2D `~numpy.ndarray` or `None`
        .. deprecated:: 3.2
            Use ``angles``, ``radii``, and ``intensities`` instead.
    intensity : float or `None`
        The mean intensity along the elliptical path. It is `None`
        until :meth:`~photutils.isophote.EllipseSample.update` is
        called and NaN if there are no valid sampled points.
    mean : float or `None`
        .. deprecated:: 3.2
            Use ``intensity`` instead.
    geometry : `~photutils.isophote.EllipseGeometry`
        The geometry of the elliptical path.
    gradient : float or `None`
        The local radial intensity gradient. It is `None` until
        :meth:`~photutils.isophote.EllipseSample.update` is called and
        NaN if there are no valid sampled points.
    gradient_err : float or `None`
        The error associated with the local radial intensity gradient.
        It is `None` until
        :meth:`~photutils.isophote.EllipseSample.update` is called or
        if no meaningful gradient could be measured, and NaN if there
        are no valid sampled points.
    gradient_error : float or `None`
        .. deprecated:: 3.0
            Use ``gradient_err`` instead.
    gradient_rel_err : float or `None`
        The relative error associated with the local radial intensity
        gradient. It is `None` until
        :meth:`~photutils.isophote.EllipseSample.update` is called, if
        no meaningful gradient could be measured, or if there are no
        valid sampled points.
    gradient_relative_error : float or `None`
        .. deprecated:: 3.0
            Use ``gradient_rel_err`` instead.
    sector_area : float or `None`
        The average area of the sectors along the elliptical path from
        which the sample values were integrated. For the 'bilinear'
        and 'nearest_neighbor' modes it is a nominal value of 2 or 1,
        respectively. It is `None` until
        :meth:`~photutils.isophote.EllipseSample.extract` is called.
    n_total : int
        The total number of sample values that would cover the entire
        elliptical path.
    total_points : int
        .. deprecated:: 3.2
            Use ``n_total`` instead.
    n_valid : int
        The actual number of sample values that were taken from the
        image. It can be smaller than ``n_total`` when the ellipse
        encompasses regions outside the image, when it encompasses
        masked or nonfinite pixels, or when sigma clipping removed some
        of the points.
    actual_points : int
        .. deprecated:: 3.2
            Use ``n_valid`` instead.

    Raises
    ------
    TypeError
        If ``image`` is a `~astropy.units.Quantity` array or if
        ``orientation`` is given with the ``orientation`` keyword as a
        number without units.

    ValueError
        If ``image`` is not a 2D array or ``integration_mode`` is not a
        valid integration mode.
    """

    @deprecated_renamed_argument(
        ['nclip', 'astep', 'eps', 'sclip', 'integrmode'],
        ['n_clip', 'step', 'ellipticity', 'clip_sigma', 'integration_mode'],
        ['3.0', '3.2', '3.2', '3.2', '3.2'], until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0', stacklevel=3)
    @_orientation_argument('position_angle', warn_number=False)
    def __init__(self, image, sma, x0=None, y0=None, step=0.1,
                 ellipticity=0.2, orientation=0 * u.deg, clip_sigma=3.0,
                 n_clip=0, linear_growth=False,
                 integration_mode='bilinear', geometry=None):
        # The orientation is ignored if a geometry is input.
        if geometry is None:
            orientation = _legacy_orientation_rad(orientation)
        self._initialize(image, sma, x0=x0, y0=y0, step=step,
                         ellipticity=ellipticity, orientation=orientation,
                         clip_sigma=clip_sigma, n_clip=n_clip,
                         linear_growth=linear_growth,
                         integration_mode=integration_mode,
                         geometry=geometry)

    @classmethod
    def _from_radians(cls, image, sma, **kwargs):
        """
        Make a sample with the orientation given as a number in
        radians.

        This is the constructor for the internal code. It issues no
        warning and does not convert the orientation.
        """
        sample = cls.__new__(cls)
        sample._initialize(image, sma, **kwargs)
        return sample

    def _initialize(self, image, sma, *, x0=None, y0=None, step=0.1,
                    ellipticity=0.2, orientation=0.0, clip_sigma=3.0,
                    n_clip=0, linear_growth=False,
                    integration_mode='bilinear', geometry=None):
        """
        Set the attributes of a new sample, with the orientation in
        radians.
        """
        image = _validate_image(image)
        if integration_mode not in _INTEGRATORS:
            msg = ('integration_mode must be one of '
                   f'{sorted(_INTEGRATORS)}')
            raise ValueError(msg)
        self.image = image
        self.integration_mode = integration_mode

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

            self.geometry = EllipseGeometry._from_radians(
                _x0, _y0, sma, ellipticity, orientation, step=step,
                linear_growth=linear_growth)

        # Sigma-clip parameters
        self.clip_sigma = clip_sigma
        self.n_clip = n_clip

        # Extracted values associated with this sample.
        self._values = None
        self._updated_values = None
        self.intensity = None
        self.gradient = None
        self.gradient_err = None
        self.gradient_rel_err = None
        self.sector_area = None

        # The gradient that tells whether a measured gradient is
        # meaningful. It is the gradient from the last update of this
        # sample, or the gradient of the previous iteration set by the
        # fitter.
        self._reference_gradient = None

        # The n_total attribute is the number of angle-radius pairs
        # that were attempted. The n_valid attribute is the number
        # of sampled pairs that resulted in valid values.
        self.n_total = 0
        self.n_valid = 0

    # Remove in 4.0
    mean = deprecated_attribute('mean', 'intensity', '3.2', until='4.0')
    actual_points = deprecated_attribute('actual_points', 'n_valid', '3.2',
                                         until='4.0')
    total_points = deprecated_attribute('total_points', 'n_total', '3.2',
                                        until='4.0')
    sclip = deprecated_attribute('sclip', 'clip_sigma', '3.2', until='4.0')
    integrmode = deprecated_attribute('integrmode', 'integration_mode',
                                      '3.2', until='4.0')

    # Remove in 4.0
    @property
    def values(self):
        """
        The sampled values as an array of shape (3, N).

        .. deprecated:: 3.2
            Use ``angles``, ``radii``, and ``intensities`` instead.
        """
        self._warn_values()
        return self._values

    @values.setter
    def values(self, value):  # numpydoc ignore=GL08
        self._warn_values()
        self._values = value

    @staticmethod
    def _warn_values():
        msg = ("The 'values' attribute was deprecated in version 3.2 and "
               'will be removed in version 4.0. Use the angles, radii, '
               'and intensities attributes instead.')
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=3)

    @property
    def angles(self):
        """
        The angles of the valid sampled points, or `None` before
        extraction.
        """
        return None if self._values is None else self._values[0]

    @property
    def radii(self):
        """
        The radii of the valid sampled points, or `None` before
        extraction.
        """
        return None if self._values is None else self._values[1]

    @property
    def intensities(self):
        """
        The intensities of the valid sampled points, or `None` before
        extraction.
        """
        return None if self._values is None else self._values[2]

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _DEPRECATED_SAMPLE_ATTRIBUTES,
                                  since='3.0', until='4.0')

    def extract(self):
        """
        Extract sample data by scanning an elliptical path over the
        image array.

        The extracted values are cached in the ``angles``, ``radii``,
        and ``intensities`` attributes, and later calls return them
        without scanning the image again, even if the ``geometry``
        attribute has changed. Make a new sample to extract again.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            An array of shape (3, N), where the rows contain the angles,
            radii, and extracted intensity values, respectively, of the
            N valid sampled points after sigma clipping.

        Raises
        ------
        ValueError
            If the ellipticity of the ellipse is 1 or larger or if its
            center is not finite.
        """
        # The sample values themselves are kept cached to prevent
        # multiple calls to the integrator code.
        if self._values is not None:
            return self._values

        # An ellipse with an ellipticity of 1 or larger or a nonfinite
        # center has no elliptical path. The geometry is checked here
        # instead of in the constructor because the fitter makes trial
        # samples with any geometry and discards the invalid ones
        # without extracting them.
        geometry = self.geometry
        if geometry.ellipticity >= 1.0:
            msg = ('The ellipticity must be less than 1, got '
                   f'{geometry.ellipticity}')
            raise ValueError(msg)
        if not (np.isfinite(geometry.x0) and np.isfinite(geometry.y0)):
            msg = ('The center of the ellipse must be finite, got '
                   f'({geometry.x0}, {geometry.y0})')
            raise ValueError(msg)

        s = self._extract()
        self._values = s
        return s

    def _extract(self, *, phi_min=0.05):
        if self.geometry.sma == 0:
            return self._extract_center()

        # Here the actual sampling takes place. This is called only once
        # during the life of an EllipseSample instance, because it's
        # an expensive calculation. This method should not be called
        # from external code. To force it to rerun, set
        # "sample._values = None" before calling sample.extract().

        # Individual extracted sample points will be stored in here.
        angles = []
        radii = []
        intensities = []
        sector_areas = []

        # Reset counters.
        self.n_total = 0
        self.n_valid = 0

        # Build integrator.
        integrator = _INTEGRATORS[self.integration_mode](
            self.image, self.geometry, angles, radii, intensities)

        # Initialize walk along elliptical path.
        radius = self.geometry._initial_polar_radius
        phi = self.geometry._initial_polar_angle

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
                integrator = _INTEGRATORS['bilinear'](
                    self.image, self.geometry, angles, radii, intensities)
            else:
                integrator = _INTEGRATORS[self.integration_mode](
                    self.image, self.geometry, angles, radii, intensities)

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
                self.n_total += 1

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
                phi, radius, phi_max, self.geometry.sma,
                self.geometry.ellipticity)
            angles, radii, intensities = integrator.integrate_path(
                path_radii, phis)
            sector_areas.append(integrator.get_sector_area())
            self.n_total = len(phis)

        # Average sector area is calculated after the integrator had
        # the opportunity to step over the entire elliptical path.
        self.sector_area = np.mean(np.array(sector_areas))

        # Apply sigma clipping.
        angles, radii, intensities = self._sigma_clip(angles, radii,
                                                      intensities)

        # Actual number of sampled points, after sigma clipping
        # removed outliers
        self.n_valid = len(angles)

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
            lower = mean - self.clip_sigma * sig
            upper = mean + self.clip_sigma * sig
            keep = (intensities >= lower) & (intensities <= upper)
            angles = angles[keep]
            radii = radii[keep]
            intensities = intensities[keep]

        return angles, radii, intensities

    def _extract_center(self):
        # The central pixel is a single bilinear sample at (x0, y0).
        angles = []
        radii = []
        intensities = []

        integrator = _INTEGRATORS['bilinear'](self.image, self.geometry,
                                              angles, radii, intensities)
        integrator.integrate(0.0, 0.0)

        self.n_total = 1
        self.n_valid = len(intensities)

        return np.array([np.array(angles), np.array(radii),
                         np.array(intensities)])

    # The fixed_parameters argument is deprecated, so passing it
    # positionally emits only the warning of this decorator, which
    # handles both the positional and the keyword forms.
    @deprecated_renamed_argument('fixed_parameters', None, '3.2',
                                 until='4.0')
    def update(self, fixed_parameters=None):
        """
        Update this `~photutils.isophote.EllipseSample` instance.

        This method calls the
        :meth:`~photutils.isophote.EllipseSample.extract` method to get
        the sample values, and then computes the mean intensity, local
        gradient, and other associated quantities. The sample values
        are those cached from the last extraction, if any, which may
        not match the current ``geometry`` attribute.

        The local radial gradient is measured from a second sample on
        an ellipse with a 10% larger semimajor axis length (or 20%
        larger if the first estimate is not meaningful), as in the
        STSDAS ``ellipse`` task, regardless of the ``step`` value.

        A meaningful gradient is negative and is either steeper than
        one third of a reference gradient or measured with a relative
        error smaller than 20%. The STSDAS task does not use the
        relative error. The reference is the gradient from the last
        update of this sample.
        `~photutils.isophote.Ellipse.fit_isophote` sets it to the
        gradient of its previous iteration. If there is
        no reference, the first estimate plus its error is used. If
        neither estimate is meaningful, the gradient is set to 0.8
        times the reference and its error is set to `None`.

        The intensity and the gradient are computed only if the
        sample values changed since the last call.

        If there are no valid sampled points, ``intensity``,
        ``gradient``, and ``gradient_err`` are set to NaN and
        ``gradient_rel_err`` is set to `None`.

        Parameters
        ----------
        fixed_parameters : `None` or array_like, optional
            An array of the fixed parameters. Must have 4 elements, the
            flags for the center (the first two elements), the
            orientation, and the ellipticity, in the order of the
            ``fix`` attribute of `~photutils.isophote.EllipseGeometry`.
            If `None`, the fixed parameters of the ``geometry``
            attribute are not changed.

            .. deprecated:: 3.2
                The ``fixed_parameters`` keyword is deprecated.
        """
        self._update(fixed_parameters=fixed_parameters)

    def _update(self, *, fixed_parameters=None):
        # See the update method. The fitter uses this method to set the
        # fixed parameters of the geometry without a warning.
        if self.geometry.sma == 0:
            # The central pixel has a single sample and no gradient.
            # The fixed parameters are ignored.
            s = self.extract()

            # The central pixel may be masked or nonfinite.
            self.intensity = s[2][0] if self.n_valid > 0 else np.nan
            self.gradient = None
            self.gradient_err = None
            self.gradient_rel_err = None
            return

        if fixed_parameters is not None:
            self.geometry._fix = fixed_parameters

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
            self.intensity = np.nan
            self.gradient = np.nan
            self.gradient_err = np.nan
            self.gradient_rel_err = None
            return

        self.intensity = np.mean(s[2])

        # Get sample with same geometry but at a different distance from
        # center. Estimate gradient from there.
        gradient, gradient_err = self._get_gradient(_GRADIENT_RADIUS)

        # Check for meaningful gradient. If no meaningful gradient, try
        # another sample, this time using larger radius. If no previous
        # estimate is available as the reference, guess it by adding
        # the error to the current gradient.
        previous_gradient = self._reference_gradient
        if not previous_gradient:
            previous_gradient = gradient + gradient_err

        if not self._is_meaningful_gradient(gradient, gradient_err,
                                            previous_gradient):
            gradient, gradient_err = self._get_gradient(
                _GRADIENT_RADIUS_RETRY)

        # If still no meaningful gradient can be measured, use the
        # reference, slightly shallower. The factor of 0.8 comes from
        # the STSDAS ellipse task, where the reference can also be the
        # gradient of the previous isophote (see the isophote FAQ in
        # the documentation). The gradient error is meaningless in
        # this case.
        if not self._is_meaningful_gradient(gradient, gradient_err,
                                            previous_gradient):
            gradient = previous_gradient * 0.8
            gradient_err = None

        self.gradient = gradient
        self._reference_gradient = gradient
        self.gradient_err = gradient_err
        if gradient_err and gradient < 0.0:
            self.gradient_rel_err = gradient_err / np.abs(gradient)
        else:
            self.gradient_rel_err = None

    @staticmethod
    def _is_meaningful_gradient(gradient, gradient_err, reference):
        """
        Return whether a measured gradient is meaningful.

        A meaningful gradient is negative and is either steeper than a
        third of the ``reference`` gradient or measured with a relative
        error smaller than 20%. The STSDAS ellipse task has only the
        first of these two alternatives. With only that one, a single
        gradient that is much steeper than the following ones would
        make them not meaningful for several iterations.
        """
        # The isophote intensity must decrease outward. A gradient of
        # exactly zero can occur with the nearest-neighbor integrator
        # when the samples at both radii extract identical pixel sets,
        # and it would cause divisions by zero in the fitter's parameter
        # correctors.
        if gradient >= 0.0:
            return False

        # This is true only for a negative reference. The gradient is
        # then shallower than a third of the reference.
        if gradient >= reference / 3.0:
            return gradient_err < _MAX_GRADIENT_REL_ERR * abs(gradient)

        # A NaN gradient (e.g., from a gradient sample without valid
        # points) also gets here, because all the comparisons above
        # are false. It is kept, and the fitter stops on it.
        return True

    def _make_sample(self, *, sma=None, x0=None, y0=None, ellipticity=None,
                     orientation=None):
        """
        Make a new sample of the same image, with the same settings and
        with the given ellipse parameters changed.

        The new sample is not extracted. The ellipse parameters that
        are `None` are taken from this sample.
        """
        geometry = self.geometry
        return EllipseSample._from_radians(
            self.image, geometry.sma if sma is None else sma,
            x0=geometry.x0 if x0 is None else x0,
            y0=geometry.y0 if y0 is None else y0,
            step=geometry.step, clip_sigma=self.clip_sigma,
            n_clip=self.n_clip,
            ellipticity=(geometry.ellipticity if ellipticity is None
                         else ellipticity),
            orientation=(geometry._orientation_rad if orientation is None
                         else orientation),
            linear_growth=geometry.linear_growth,
            integration_mode=self.integration_mode)

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
        gradient = (mean_g - self.intensity) / self.geometry.sma / radius

        s = self.extract()
        sigma = np.std(s[2])
        sigma_g = np.std(sg[2])

        gradient_err = (np.sqrt(sigma**2 / len(s[2])
                                + sigma_g**2 / len(sg[2]))
                        / self.geometry.sma / radius)

        return gradient, gradient_err

    def coordinates(self):
        """
        Return the (x, y) pixel coordinates of the valid sampled
        points.

        :meth:`~photutils.isophote.EllipseSample.extract` must be called
        first.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinate arrays.
        """
        angles = self.angles
        radii = self.radii
        orientation = self.geometry._orientation_rad
        x = radii * np.cos(angles + orientation) + self.geometry.x0
        y = radii * np.sin(angles + orientation) + self.geometry.y0
        return x, y
