# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for storing parameters for the geometry of an ellipse.
"""

import inspect
import math
import warnings
from functools import wraps

import astropy.units as u
import numpy as np
from astropy import log

from photutils.utils._deprecation import (deprecated, deprecated_attribute,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument,
                                          renamed_argument_message,
                                          warn_deprecated_attribute)
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['EllipseGeometry']


# The first and second harmonic coefficients of the isophote fit, in
# the order returned by _fit_first_and_second_harmonics after the mean,
# and the ellipse parameter that each one corrects. The ``fix`` array of
# EllipseGeometry and the parameter correctors of the isophote fitter
# follow this order.
_HARMONIC_PARAMETERS = {'a1': 'center', 'b1': 'center',
                        'a2': 'orientation', 'b2': 'ellipticity'}


# The indices of the center, orientation, and ellipticity flags in the
# ``fix`` array.
_CENTER_INDICES = tuple(index for index, name
                        in enumerate(_HARMONIC_PARAMETERS.values())
                        if name == 'center')
_ORIENTATION_INDEX = list(_HARMONIC_PARAMETERS.values()).index('orientation')
_ELLIPTICITY_INDEX = list(_HARMONIC_PARAMETERS.values()).index('ellipticity')


def _make_fix_array(*, fix_center, fix_orientation, fix_ellipticity):
    """
    Make the array of fixed-parameter flags, in the order of the
    harmonic coefficients that correct the parameters.
    """
    fixed = {'center': fix_center, 'orientation': fix_orientation,
             'ellipticity': fix_ellipticity}
    return np.array([fixed[name] for name in _HARMONIC_PARAMETERS.values()])


# The radii of the two circular samples of the object centerer, as in
# the STSDAS ellipse task.
_CENTERER_RADII = (4.0, 8.0)


def _eccentric_anomaly(phi, eps):
    """
    Compute the eccentric anomaly of the point at polar angle ``phi``
    (radians) on an ellipse with ellipticity ``eps``.

    The area of the elliptical sector from polar angle 0 to ``phi``
    is the product of the eccentric anomaly and half the product of
    the semimajor and semiminor axis lengths. The result is continuous
    in ``phi`` and differs from it by less than pi / 2, so it can be
    used for any polar angle, including angles outside of [0, 2 pi].
    """
    anomaly = math.atan2(math.sin(phi), (1.0 - eps) * math.cos(phi))
    return phi + (anomaly - phi + math.pi) % (2.0 * math.pi) - math.pi


def _validate_orientation(value):
    """
    Raise a `TypeError` if an orientation is not a
    `~astropy.units.Quantity`.
    """
    if not isinstance(value, u.Quantity):
        msg = ('orientation must be an angular Quantity, e.g., '
               '30 * u.deg')
        raise TypeError(msg)


def _validate_sma(sma):
    """
    Raise a `ValueError` if a semimajor axis length is negative or not
    finite.
    """
    if not 0 <= sma < np.inf:  # also rejects NaN
        msg = f'sma must be finite and nonnegative, got {sma}'
        raise ValueError(msg)


# Remove in 4.0
def _legacy_orientation_rad(value):
    """
    Return an orientation in radians.

    The orientation is an angular `~astropy.units.Quantity` or a number
    in radians.
    """
    if isinstance(value, u.Quantity):
        return value.to_value(u.rad)
    return value


# Remove in 4.0
def _orientation_argument(old_name, *, warn_number):
    """
    Make a decorator that handles the ``orientation`` argument of a
    constructor.

    A number without units is accepted as radians only from the
    deprecated ``old_name`` keyword or as a positional argument, with a
    deprecation warning for the positional argument if ``warn_number``
    is `True`. A number given with the ``orientation`` keyword raises a
    `TypeError`.

    The ``old_name`` keyword is renamed here, rather than by the
    renamed-argument decorator, because that decorator passes it on
    as the ``orientation`` keyword, which can then no longer be told
    apart from a number given with the ``orientation`` keyword. This
    decorator must be placed inside both the renamed-argument and the
    positional-argument decorators so that its warnings point to the
    caller.
    """
    rename_msg = renamed_argument_message(old_name, 'orientation', '3.2',
                                          '4.0')
    number_msg = ('Passing the orientation as a number without units was '
                  'deprecated in version 3.2 and will be removed in '
                  'version 4.0. The number is interpreted as radians. Pass '
                  'an angular Quantity instead, e.g., value * u.rad.')

    def decorator(func):
        position = list(inspect.signature(func).parameters).index(
            'orientation')

        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                warnings.warn(rename_msg, PhotutilsDeprecationWarning,
                              stacklevel=4)
                if 'orientation' in kwargs or len(args) > position:
                    msg = (f'cannot specify both "{old_name}" and '
                           '"orientation".')
                    raise TypeError(msg)
                kwargs['orientation'] = kwargs.pop(old_name)
            elif 'orientation' in kwargs:
                _validate_orientation(kwargs['orientation'])
            elif (warn_number and len(args) > position
                  and not isinstance(args[position], u.Quantity)):
                warnings.warn(number_msg, PhotutilsDeprecationWarning,
                              stacklevel=4)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Remove in 4.0
def _warn_deprecated_fix():
    msg = ("The 'fix' attribute was deprecated in version 3.2. Use "
           "'fix_center', 'fix_orientation', and 'fix_ellipticity' "
           'instead. It will be removed in version 4.0.')
    warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=3)


# Remove in 4.0
def _deprecated_internal_attribute(name, description):
    """
    Make a property for a deprecated attribute that is internal to the
    fitter and has no replacement.

    Getting or setting the property issues a deprecation warning and
    forwards to the private attribute with the same name.
    """
    private_name = f'_{name}'
    msg = (f'The {name!r} attribute was deprecated in version 3.2. It is '
           'internal to the fitter and has no replacement. It will be '
           'removed in version 4.0.')

    def getter(self):
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=2)
        # The private attribute is not defined when sma is zero.
        try:
            return getattr(self, private_name)
        except AttributeError:
            error_msg = (f'{type(self).__name__!r} object has no attribute '
                         f'{name!r}')
            raise AttributeError(error_msg) from None

    def setter(self, value):
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=2)
        setattr(self, private_name, value)

    doc = f"""
    {description}

    .. deprecated:: 3.2
        This attribute is internal to the fitter and has no
        replacement.
    """
    return property(getter, setter, doc=doc)


class EllipseGeometry:
    r"""
    Container class to store parameters for the geometry of an ellipse.

    Parameters that describe the relationship of a given ellipse with
    other associated ellipses are also encapsulated in this container.
    These associated ellipses may include, e.g., the two (inner and
    outer) bounding ellipses that are used to build sectors along the
    elliptical path. These sectors are used as areas for integrating
    pixel values, when the area integration mode (mean or median) is
    used.

    This class also keeps track of where in the ellipse we are when
    performing an 'extract' operation. This is mostly relevant when
    using an area integration mode (as opposed to a pixel integration
    mode).

    Parameters
    ----------
    x0, y0 : float
        The center pixel coordinate of the ellipse.

    sma : float
        The semimajor axis of the ellipse in pixels. It must be finite
        and nonnegative.

    ellipticity : float
        The ellipticity of the ellipse. The ellipticity is defined as

        .. math::

            \epsilon =  1  -  \frac{b}{a}

        where a and b are the lengths of the semimajor and semiminor
        axes, respectively. The ellipticity must be less than 1.

        .. deprecated:: 3.2
            The ``eps`` keyword is deprecated. Use ``ellipticity``
            instead.

    orientation : `~astropy.units.Quantity`
        The orientation of the semimajor axis as an angular
        `~astropy.units.Quantity` (e.g., ``30 * u.deg`` or an
        `~astropy.coordinates.Angle`), measured counterclockwise from
        the positive x axis of the image array. The ``orientation``
        attribute is a `~astropy.units.Quantity` in degrees. The fitted
        orientations are in the range of 0 to 180 degrees. When the
        orientations are near either end of the range, noise can make
        the solution jump back and forth between successive isophotes,
        by amounts close to 180 degrees.

        .. deprecated:: 3.2
            The ``pa`` keyword is deprecated. Use ``orientation``
            instead. A number without units is accepted as radians
            with the ``pa`` keyword.

        .. deprecated:: 3.2
            Passing the orientation as the fifth positional argument
            as a number without units is deprecated. The number is
            interpreted as radians. Pass an angular
            `~astropy.units.Quantity` instead (e.g., ``value * u.rad``).
            A number without units given with the ``orientation``
            keyword raises a `TypeError`.

    step : float, optional
        The step value for growing/shrinking the semimajor axis. It can
        be expressed either in pixels (when ``linear_growth=True``) or
        as a relative value (when ``linear_growth=False``). The default
        is 0.1. When the geometry is input to
        `~photutils.isophote.Ellipse`, this value is replaced by the
        ``step`` keyword of its ``fit_image`` and ``fit_isophote``
        methods.

        .. deprecated:: 3.2
            The ``astep`` keyword is deprecated. Use ``step`` instead.

    linear_growth : bool, optional
        The semimajor axis growing/shrinking mode. The default is
        `False`. When the geometry is input to
        `~photutils.isophote.Ellipse`, this value is used unless the
        ``linear_growth`` keyword of its ``fit_image`` and
        ``fit_isophote`` methods is set.

    fix_center : bool, optional
        Keep center of ellipse fixed during fit? The default is
        `False`.

    fix_orientation : bool, optional
        Keep the orientation of the semimajor axis of the ellipse fixed
        during the fit? The default is `False`.

        .. deprecated:: 3.2
            The ``fix_pa`` keyword is deprecated. Use
            ``fix_orientation`` instead.

    fix_ellipticity : bool, optional
        Keep ellipticity of ellipse fixed during fit? The default is
        `False`.

        .. deprecated:: 3.2
            The ``fix_eps`` keyword is deprecated. Use
            ``fix_ellipticity`` instead.

    Attributes
    ----------
    x0, y0 : float
        The center pixel coordinate of the ellipse.

    sma : float
        The semimajor axis of the ellipse in pixels.

    ellipticity : float
        The ellipticity of the ellipse.

    eps : float
        .. deprecated:: 3.2
            Use ``ellipticity`` instead.

    orientation : `~astropy.units.Quantity`
        The orientation of the semimajor axis in degrees, measured
        counterclockwise from the positive x axis of the image array.
        It must be set with an angular `~astropy.units.Quantity`.

    pa : float
        .. deprecated:: 3.2
            Use ``orientation`` instead. It is a float in radians.

    step : float
        The step value for growing/shrinking the semimajor axis.

    astep : float
        .. deprecated:: 3.2
            Use ``step`` instead.

    linear_growth : bool
        The semimajor axis growing/shrinking mode.

    fix_center : bool
        Whether the center is kept fixed during the fit.

    fix_orientation : bool
        Whether the orientation is kept fixed during the fit.

    fix_ellipticity : bool
        Whether the ellipticity is kept fixed during the fit.

    fix : 1D `~numpy.ndarray`
        .. deprecated:: 3.2
            Use ``fix_center``, ``fix_orientation``, and
            ``fix_ellipticity`` instead.

    sector_angular_width : float
        .. deprecated:: 3.2
            This attribute was internal to the fitter and has no
            replacement.

    initial_polar_angle : float
        .. deprecated:: 3.2
            This attribute was internal to the fitter and has no
            replacement.

    initial_polar_radius : float
        .. deprecated:: 3.2
            This attribute was internal to the fitter and has no
            replacement.

    Raises
    ------
    TypeError
        If ``orientation`` is given with the ``orientation`` keyword as
        a number without units, or if a number without units is
        assigned to the ``orientation`` attribute.

    ValueError
        If ``sma`` is negative or not finite.
    """

    @deprecated_renamed_argument(
        ['eps', 'astep', 'fix_pa', 'fix_eps'],
        ['ellipticity', 'step', 'fix_orientation', 'fix_ellipticity'],
        '3.2', until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0', stacklevel=3)
    @_orientation_argument('pa', warn_number=True)
    def __init__(self, x0, y0, sma, ellipticity, orientation, step=0.1,
                 linear_growth=False, fix_center=False,
                 fix_orientation=False, fix_ellipticity=False):
        self._initialize(x0, y0, sma, ellipticity,
                         _legacy_orientation_rad(orientation), step=step,
                         linear_growth=linear_growth, fix_center=fix_center,
                         fix_orientation=fix_orientation,
                         fix_ellipticity=fix_ellipticity)

    @classmethod
    def _from_radians(cls, x0, y0, sma, ellipticity, orientation,
                      **kwargs):
        """
        Make a geometry with the orientation given as a number in
        radians.

        This is the constructor for the internal code. It issues no
        warning and does not convert the orientation.
        """
        geometry = cls.__new__(cls)
        geometry._initialize(x0, y0, sma, ellipticity, orientation,
                             **kwargs)
        return geometry

    def _initialize(self, x0, y0, sma, ellipticity, orientation, *,
                    step=0.1, linear_growth=False, fix_center=False,
                    fix_orientation=False, fix_ellipticity=False):
        """
        Set the attributes of a new geometry, with the orientation in
        radians.
        """
        _validate_sma(sma)
        self.x0 = x0
        self.y0 = y0
        self.sma = sma
        self.ellipticity = ellipticity
        self._orientation_rad = orientation

        self.step = step
        self.linear_growth = linear_growth

        # Fixed parameters are flagged in here, in the order of the
        # harmonic coefficients that correct them.
        self._fix = _make_fix_array(fix_center=fix_center,
                                    fix_orientation=fix_orientation,
                                    fix_ellipticity=fix_ellipticity)

        # Limits for sector angular width
        self._phi_min = 0.05
        self._phi_max = 0.2

        self._reset_sma_dependent_attributes()

    @property
    def orientation(self):
        """
        The orientation of the semimajor axis as an angular
        `~astropy.units.Quantity` in degrees.

        The angle is measured counterclockwise from the positive x axis
        of the image array. It must be set with an angular
        `~astropy.units.Quantity`.
        """
        return np.rad2deg(self._orientation_rad) << u.deg

    @orientation.setter
    def orientation(self, value):  # numpydoc ignore=GL08
        # The internal code uses the orientation as a float in radians.
        _validate_orientation(value)
        self._orientation_rad = value.to_value(u.rad)

    @property
    def fix_center(self):
        """
        Whether the center is kept fixed during the fit.
        """
        return all(self._fix[index] for index in _CENTER_INDICES)

    @property
    def fix_orientation(self):
        """
        Whether the orientation is kept fixed during the fit.
        """
        return bool(self._fix[_ORIENTATION_INDEX])

    @property
    def fix_ellipticity(self):
        """
        Whether the ellipticity is kept fixed during the fit.
        """
        return bool(self._fix[_ELLIPTICITY_INDEX])

    # Remove in 4.0
    @property
    def fix(self):
        """
        The array of fixed-parameter flags for the center (the first
        two elements), the orientation, and the ellipticity.

        .. deprecated:: 3.2
            Use ``fix_center``, ``fix_orientation``, and
            ``fix_ellipticity`` instead.
        """
        _warn_deprecated_fix()
        return self._fix

    @fix.setter
    def fix(self, value):  # numpydoc ignore=GL08
        _warn_deprecated_fix()
        self._fix = value

    # Remove in 4.0
    sector_angular_width = _deprecated_internal_attribute(
        'sector_angular_width',
        'The angular width (radians) of the integration sectors.')
    initial_polar_angle = _deprecated_internal_attribute(
        'initial_polar_angle',
        'The polar angle (radians) of the first sample point.')
    initial_polar_radius = _deprecated_internal_attribute(
        'initial_polar_radius',
        'The radius (pixels) of the ellipse at the initial polar angle.')

    # Remove in 4.0
    eps = deprecated_attribute('eps', 'ellipticity', '3.2', until='4.0')
    astep = deprecated_attribute('astep', 'step', '3.2', until='4.0')

    # Remove in 4.0
    @property
    def pa(self):
        """
        The orientation of the semimajor axis as a float in radians.

        .. deprecated:: 3.2
            Use ``orientation`` instead.
        """
        warn_deprecated_attribute('pa', 'orientation', since='3.2',
                                  until='4.0')
        return self._orientation_rad

    @pa.setter
    def pa(self, value):  # numpydoc ignore=GL08
        warn_deprecated_attribute('pa', 'orientation', since='3.2',
                                  until='4.0')
        self._orientation_rad = _legacy_orientation_rad(value)

    def _reset_sma_dependent_attributes(self):
        """
        Reset attributes derived from the current semimajor axis length.
        """
        # Variables used in the calculation of the sector angular width
        sma1, sma2 = self._bounding_ellipses()
        inner_sma = min((sma2 - sma1), 3.0)
        self._area_factor = (sma2 - sma1) * inner_sma

        for attr in ('_sector_angular_width', '_initial_polar_angle',
                     '_initial_polar_radius'):
            if hasattr(self, attr):
                delattr(self, attr)

        # The sma can eventually be zero.
        if self.sma > 0.0:
            self._sector_angular_width = max(
                min((inner_sma / self.sma), self._phi_max), self._phi_min)
            self._initial_polar_angle = self._sector_angular_width / 2.0
            self._initial_polar_radius = self.radius(
                self._initial_polar_angle)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def find_center(self, image, threshold=1.0, verbose=True):
        """
        Find the center of a galaxy.

        If the algorithm is successful, the (x, y) coordinates of this
        `~photutils.isophote.EllipseGeometry` instance (i.e., the ``x0``
        and ``y0`` attributes) will be modified.

        The isophote fit algorithm requires an initial guess for the
        galaxy center (x, y) coordinates and these coordinates must be
        close to the actual galaxy center for the isophote fit to work.
        This method can provide an initial guess for the galaxy center
        coordinates. See the **Notes** section below for more details.

        Parameters
        ----------
        image : 2D `~numpy.ndarray`
            The image array. Masked pixels (if ``image`` is a
            `~numpy.ma.MaskedArray`) and nonfinite (NaN or infinity)
            values are excluded.

        threshold : float, optional
            The centerer threshold. To turn off the centerer, set this
            to `numpy.inf`. The default is 1.0, as in the STSDAS
            ``ellipse`` task.

        verbose : bool, optional
            Whether to log object centering information at the INFO
            level. The default is `True`.

        Notes
        -----
        The centerer function scans an 11x11 window centered on the
        ``x0`` and ``y0`` coordinates of this instance (truncated to
        whole pixels). If any of the ``x0`` and ``y0`` coordinates are
        `None` or outside of the image, the window is centered on the
        center of the input image frame instead, at x = ``shape[1] / 2``
        and y = ``shape[0] / 2``, where ``shape`` is the image shape.
        If the center acquisition is successful, the
        `~photutils.isophote.EllipseGeometry` instance is modified in
        place to reflect the solution of the object centerer algorithm.
        Otherwise, the (x, y) coordinates are not changed, including
        coordinates that are `None` or outside of the image.

        As in the STSDAS ``ellipse`` task, the image is sampled with
        bilinear interpolation along two concentric circles, with radii
        of 4 and 8 pixels, at each position. The figure of merit is the
        difference between the mean intensities along the inner and
        outer circles, divided by the quadrature sum of their sample
        standard deviations. The position with the largest figure of
        merit is accepted if that value exceeds ``threshold``. The new
        center is a whole-pixel position.

        Only the window positions inside the image are scanned. Near
        the image edges, only the sampled points inside the image are
        used to compute the figure of merit.

        In some cases the object centerer algorithm may fail even though
        there is enough signal-to-noise to start a fit (e.g., objects
        with very high ellipticity). In those cases, the centerer can
        be made to accept a weaker detection by decreasing the value
        of the ``threshold`` parameter. The figure of merit is akin
        to a signal-to-noise ratio. It has no upper limit, so the
        centerer is shut off entirely only by setting the threshold to
        `numpy.inf`.
        """
        self.centerer_threshold = threshold

        # The sample module imports this module, so it is imported here
        # to avoid a circular import.
        from photutils.isophote.sample import EllipseSample, _sample_std

        # Check if center coordinates point to somewhere inside the frame.
        # If not, set them to frame center.
        shape = image.shape
        _x0 = self.x0
        _y0 = self.y0
        if (_x0 is None or _x0 < 0 or _x0 >= shape[1] or _y0 is None
                or _y0 < 0 or _y0 >= shape[0]):
            _x0 = shape[1] / 2
            _y0 = shape[0] / 2

        max_fom = -np.inf
        max_i = 0
        max_j = 0

        # Scan all positions inside window that are inside the image.
        window_half_size = 5
        i_range = range(max(int(_x0 - window_half_size), 0),
                        min(int(_x0 + window_half_size), shape[1] - 1) + 1)
        j_range = range(max(int(_y0 - window_half_size), 0),
                        min(int(_y0 + window_half_size), shape[0] - 1) + 1)
        for i in i_range:
            for j in j_range:
                # The mean and standard deviation of the bilinear
                # samples along two concentric circles. Points that
                # are outside the image or near masked or nonfinite
                # pixels are not sampled.
                means = []
                stds = []
                for radius in _CENTERER_RADII:
                    sample = EllipseSample._from_radians(
                        image, radius, x0=float(i), y0=float(j),
                        ellipticity=0.0, orientation=0.0)
                    intensities = sample.extract()[2]
                    if len(intensities) == 0:
                        break
                    means.append(np.mean(intensities))
                    stds.append(np.nan_to_num(_sample_std(intensities)))
                else:
                    stddev = np.hypot(*stds)

                    # As in the STSDAS ellipse task, a position without
                    # any scatter (e.g., in a flat image), where the
                    # figure of merit is not defined, is skipped. A
                    # scatter at the level of rounding errors of the
                    # bilinear interpolation counts as no scatter.
                    tolerance = 8 * np.finfo(float).eps * max(
                        abs(means[0]), abs(means[1]))
                    if not stddev > tolerance:
                        continue
                    fom = (means[0] - means[1]) / stddev
                    if fom > max_fom:
                        max_fom = fom
                        max_i = i
                        max_j = j

        # Figure of merit > threshold: update geometry with new coordinates.
        if max_fom > threshold:
            self.x0 = float(max_i)
            self.y0 = float(max_j)

            if verbose:
                log.info(f'Found center at x0 = {self.x0:5.1f}, '
                         f'y0 = {self.y0:5.1f}')
        elif verbose:
            log.info('Result is below the threshold. Keeping the '
                     'original coordinates.')

    def radius(self, angle):
        """
        Calculate the polar radius for a given polar angle.

        Parameters
        ----------
        angle : float or `~numpy.ndarray`
            The polar angle(s) (radians), measured from the semimajor
            axis.

        Returns
        -------
        radius : float or `~numpy.ndarray`
            The polar radius (pixels).
        """
        return (self.sma * (1.0 - self.ellipticity)
                / np.sqrt(((1.0 - self.ellipticity) * np.cos(angle))**2
                          + (np.sin(angle))**2))

    def _initialize_sector_geometry(self, phi):
        """
        Initialize geometry attributes associated with an elliptical
        sector at the given polar angle ``phi``.

        The sector spans the angular width stored in the
        ``_sector_angular_width`` attribute, centered on ``phi``. This
        method computes:

        * the four vertices that define the elliptical sector on the
          pixel array

        * the sector area (saved in the ``sector_area`` attribute)

        * the angular width of the next sector, computed from this
          sector's radii so that the sectors have roughly constant area
          (saved in the ``_sector_angular_width`` attribute)

        Parameters
        ----------
        phi : float
            The polar angle (radians) where the sector is located.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinates of each vertex as 1D arrays.
        """
        # These polar radii bound the region between the inner
        # and outer ellipses that define the sector.
        sma1, sma2 = self._bounding_ellipses()
        eps_ = 1.0 - self.ellipticity

        # Polar vector at one side of the elliptical sector
        self._phi1 = phi - self._sector_angular_width / 2.0
        r1 = (sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2
                                      + (math.sin(self._phi1))**2))
        r2 = (sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi1))**2
                                      + (math.sin(self._phi1))**2))

        # Polar vector at the other side of the elliptical sector
        self._phi2 = phi + self._sector_angular_width / 2.0
        r3 = (sma2 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2
                                      + (math.sin(self._phi2))**2))

        r4 = (sma1 * eps_ / math.sqrt((eps_ * math.cos(self._phi2))**2
                                      + (math.sin(self._phi2))**2))

        # Sector area. The two bounding ellipses have the same
        # eccentric anomalies at the sector limits.
        delta_anomaly = (
            _eccentric_anomaly(self._phi2, self.ellipticity)
            - _eccentric_anomaly(self._phi1, self.ellipticity))
        self.sector_area = (eps_ * (sma2**2 - sma1**2) / 2.0
                            * delta_anomaly)

        # Angular width of sector. It is calculated such that the sectors
        # come out with roughly constant area along the ellipse. The
        # width tends to infinity as the inner radius tends to zero.
        if r4 == 0:
            self._sector_angular_width = self._phi_max
        else:
            self._sector_angular_width = max(
                min((self._area_factor / (r3 - r4) / r4), self._phi_max),
                self._phi_min)

        # Compute the 4 vertices that define the elliptical sector.
        vertex_x = np.zeros(shape=4, dtype=float)
        vertex_y = np.zeros(shape=4, dtype=float)

        # Vertices are labeled in counterclockwise sequence.
        angle1 = self._phi1 + self._orientation_rad
        angle2 = self._phi2 + self._orientation_rad
        vertex_x[0:2] = np.array([r1, r2]) * math.cos(angle1)
        vertex_x[2:4] = np.array([r4, r3]) * math.cos(angle2)
        vertex_y[0:2] = np.array([r1, r2]) * math.sin(angle1)
        vertex_y[2:4] = np.array([r4, r3]) * math.sin(angle2)
        vertex_x += self.x0
        vertex_y += self.y0

        return vertex_x, vertex_y

    def _bounding_ellipses(self):
        """
        Compute the semimajor axis lengths of the two ellipses that
        bound the annulus where integrations take place.

        The two lengths are ``sma`` minus and plus half of the annulus
        width, which is ``step`` pixels if ``linear_growth`` is `True`
        or ``sma * step`` pixels otherwise.

        Returns
        -------
        sma1, sma2 : float
            The smaller and larger values of semimajor axis length that
            define the annulus bounding ellipses. The smaller value is
            never negative.
        """
        if self.linear_growth:
            a1 = self.sma - self.step / 2.0
            a2 = self.sma + self.step / 2.0
        else:
            a1 = self.sma * (1.0 - self.step / 2.0)
            a2 = self.sma * (1.0 + self.step / 2.0)

        # The annulus extends to the center if the step is larger than
        # twice the semimajor axis length.
        return max(a1, 0.0), a2

    def _polar_angle_sector_limits(self):
        """
        Return the two polar angles that bound the sector.

        The two bounding polar angles become available only after
        calling the ``_initialize_sector_geometry`` method.

        Returns
        -------
        phi1, phi2 : float
            The smaller and larger values of polar angle that bound the
            current sector.
        """
        return self._phi1, self._phi2

    def to_aperture(self):
        """
        Return the ellipse as an
        `~photutils.aperture.EllipticalAperture`.

        The aperture can be used to plot the ellipse with its
        :meth:`~photutils.aperture.EllipticalAperture.plot` method.

        Returns
        -------
        aperture : `~photutils.aperture.EllipticalAperture`
            The aperture with the center, semimajor axis length,
            ellipticity, and orientation of this geometry.

        Raises
        ------
        ValueError
            If the semimajor axis length is not positive or the
            ellipticity is 1 or larger.

        Examples
        --------
        >>> import astropy.units as u
        >>> from photutils.isophote import EllipseGeometry
        >>> geometry = EllipseGeometry(x0=75, y0=75, sma=20,
        ...                            ellipticity=0.5,
        ...                            orientation=20.0 * u.deg)
        >>> geometry.to_aperture()
        <EllipticalAperture([75., 75.], a=20.0, b=10.0, theta=20.0 deg)>
        """
        # This import is slow because it loads the aperture subpackage
        from photutils.aperture import EllipticalAperture

        if not self.sma > 0 or not self.ellipticity < 1:
            msg = ('An aperture can be made only for an ellipse with a '
                   'positive semimajor axis length and an ellipticity '
                   'smaller than 1')
            raise ValueError(msg)

        return EllipticalAperture((self.x0, self.y0), self.sma,
                                  self.sma * (1.0 - self.ellipticity),
                                  theta=self.orientation)

    def to_polar(self, x, y):
        r"""
        Return the radius and polar angle in the ellipse coordinate
        system given (x, y) pixel image coordinates.

        This method takes care of the different definitions for
        orientation (:math:`\theta`) and polar angle (:math:`\phi`):

        .. math::

            -2 \pi < \theta < 2 \pi

            0 \leq \phi < 2 \pi

        Note that radius can be anything. The solution is not tied to
        the semimajor axis length, but to the center position and
        orientation.

        Parameters
        ----------
        x, y : float or array_like
            The (x, y) image coordinates.

        Returns
        -------
        radius, angle : float or 2D `~numpy.ndarray`
            The distance (pixels) from the ellipse center and the polar
            angle (radians) measured counterclockwise from the semimajor
            axis. Python int and float inputs (including
            `numpy.float64`) return floats. Other inputs return arrays
            of at least two dimensions.
        """
        # We split in between a scalar version and a
        # vectorized version. This is necessary for
        # now so we don't pay a heavy speed penalty
        # that is incurred when using vectorized code.

        # The split in two separate functions helps in
        # the profiling analysis. Most of the time is
        # spent in the scalar function.
        if isinstance(x, (int, float)):
            return self._to_polar_scalar(x, y)

        return self._to_polar_vectorized(x, y)

    def _to_polar_scalar(self, x, y):
        x1 = x - self.x0
        y1 = y - self.y0

        radius = x1**2 + y1**2
        if radius > 0.0:
            radius = math.sqrt(radius)
            angle = math.asin(abs(y1) / radius)
        else:
            radius = 0.0
            angle = 1.0

        if x1 >= 0.0 and y1 < 0.0:
            angle = 2 * np.pi - angle
        elif x1 < 0.0 and y1 >= 0.0:
            angle = np.pi - angle
        elif x1 < 0.0 and y1 < 0.0:
            angle = np.pi + angle

        orientation = self._orientation_rad
        if orientation < 0.0:
            orientation += 2 * np.pi
        angle = angle - orientation
        if angle < 0.0:
            angle = angle + 2 * np.pi

        return radius, angle

    def _to_polar_vectorized(self, x, y):
        # Integer coordinates and centers would truncate the radii.
        x1 = np.atleast_2d(x).astype(float) - self.x0
        y1 = np.atleast_2d(y).astype(float) - self.y0

        radius = x1**2 + y1**2
        angle = np.ones(radius.shape)

        imask = (radius > 0.0)
        radius[imask] = np.sqrt(radius[imask])
        angle[imask] = np.arcsin(np.abs(y1[imask]) / radius[imask])
        radius[~imask] = 0.0
        angle[~imask] = 1.0

        idx = (x1 >= 0.0) & (y1 < 0.0)
        angle[idx] = 2 * np.pi - angle[idx]
        idx = (x1 < 0.0) & (y1 >= 0.0)
        angle[idx] = np.pi - angle[idx]
        idx = (x1 < 0.0) & (y1 < 0.0)
        angle[idx] = np.pi + angle[idx]

        orientation = self._orientation_rad
        if orientation < 0.0:
            orientation += 2 * np.pi
        angle = angle - orientation
        angle[angle < 0] += 2 * np.pi

        return radius, angle

    def _update_sma(self, step):
        """
        Calculate an updated value for the semimajor axis, given the
        current value and the step value.

        The step value must be managed by the caller to support both
        modes: grow outwards and shrink inwards.

        The new value is ``sma + step`` if ``linear_growth`` is `True`
        or ``sma * (1 + step)`` otherwise. The ``sma`` attribute is not
        modified.

        Parameters
        ----------
        step : float
            The step value.

        Returns
        -------
        sma : float
            The new semimajor axis length.
        """
        if self.linear_growth:
            sma = self.sma + step
        else:
            sma = self.sma * (1.0 + step)
        return sma

    def _reset_sma(self, step):
        """
        Compute the semimajor axis length and the step value that
        reverse the direction of growth, from outwards to inwards.

        The new values are ``sma - step`` and ``-step`` if
        ``linear_growth`` is `True`, or ``sma / (1 + step)`` and
        ``1 / (1 + step) - 1`` otherwise. The geometry is not modified.

        Parameters
        ----------
        step : float
            The current step value.

        Returns
        -------
        sma, new_step : float
            The new semimajor axis length and the new step value to
            initiate the shrinking of the semimajor axis length. This is
            the step value that should be used when calling the
            ``_update_sma`` method.
        """
        if self.linear_growth:
            sma = self.sma - step
            step = -step
        else:
            aux = 1.0 / (1.0 + step)
            sma = self.sma * aux
            step = aux - 1.0

        return sma, step

    # Remove in 4.0
    @deprecated('3.2', until='4.0')
    def initialize_sector_geometry(self, phi):
        """
        Initialize the geometry of an elliptical sector.

        Parameters
        ----------
        phi : float
            The polar angle (radians) where the sector is located.

        Returns
        -------
        x, y : 1D `~numpy.ndarray`
            The x and y coordinates of each vertex as 1D arrays.
        """
        return self._initialize_sector_geometry(phi)

    # Remove in 4.0
    @deprecated('3.2', until='4.0')
    def polar_angle_sector_limits(self):
        """
        Return the two polar angles that bound the sector.

        Returns
        -------
        phi1, phi2 : float
            The smaller and larger values of polar angle.
        """
        return self._polar_angle_sector_limits()

    # Remove in 4.0
    @deprecated('3.2', until='4.0')
    def bounding_ellipses(self):
        """
        Compute the semimajor axis lengths of the annulus bounds.

        Returns
        -------
        sma1, sma2 : float
            The smaller and larger semimajor axis lengths.
        """
        return self._bounding_ellipses()

    # Remove in 4.0
    @deprecated('3.2', until='4.0')
    def update_sma(self, step):
        """
        Calculate an updated value for the semimajor axis.

        Parameters
        ----------
        step : float
            The step value.

        Returns
        -------
        sma : float
            The new semimajor axis length.
        """
        return self._update_sma(step)

    # Remove in 4.0
    @deprecated('3.2', until='4.0')
    def reset_sma(self, step):
        """
        Compute the semimajor axis length and step that reverse the
        direction of growth.

        Parameters
        ----------
        step : float
            The current step value.

        Returns
        -------
        sma, new_step : float
            The new semimajor axis length and the new step value.
        """
        return self._reset_sma(step)
