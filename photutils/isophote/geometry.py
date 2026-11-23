# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for storing parameters for the geometry of an ellipse.
"""

import math

import astropy.units as u
import numpy as np
from astropy import log

from photutils.utils._deprecation import (deprecated, deprecated_attribute,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument,
                                          warn_deprecated_attribute)

__all__ = ['EllipseGeometry']


# The first and second harmonic coefficients of the isophote fit, in
# the order returned by _fit_first_and_second_harmonics after the mean,
# and the ellipse parameter that each one corrects. The ``fix`` array of
# EllipseGeometry and the parameter correctors of the isophote fitter
# follow this order.
_HARMONIC_PARAMETERS = {'a1': 'center', 'b1': 'center',
                        'a2': 'orientation', 'b2': 'ellipticity'}


# The indices of the orientation and ellipticity in the ``fix`` array.
_FIX_INDEX = {name: index for index, name
              in enumerate(_HARMONIC_PARAMETERS.values())
              if name != 'center'}


def _make_fix_array(*, fix_center, fix_orientation, fix_ellipticity):
    """
    Make the array of fixed-parameter flags, in the order of the
    harmonic coefficients that correct the parameters.
    """
    fixed = {'center': fix_center, 'orientation': fix_orientation,
             'ellipticity': fix_ellipticity}
    return np.array([fixed[name] for name in _HARMONIC_PARAMETERS.values()])


def _make_centerer_masks():
    """
    Make the masks of the object centerer.

    The masks are centered on the central pixel of a 17x17 pixel box.
    As in a masked array, a value of `False` marks the pixels that are
    used. The inner mask selects a disk with a radius of 3.5 pixels and
    the outer mask selects a ring with radii from 7.5 to 8.5 pixels.

    Returns
    -------
    in_mask, out_mask : 2D bool `~numpy.ndarray`
        The inner and outer masks.
    """
    half_size = 8
    y, x = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    radius = np.hypot(x, y)
    in_mask = radius > 3.5
    out_mask = (radius < 7.5) | (radius >= 8.5)

    return in_mask, out_mask


IN_MASK, OUT_MASK = _make_centerer_masks()


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
        The semimajor axis of the ellipse in pixels.
    ellipticity : float
        The ellipticity of the ellipse. The ellipticity is defined as

        .. math::

            \epsilon =  1  -  \frac{b}{a}

        where a and b are the lengths of the semimajor and semiminor
        axes, respectively. The ellipticity must be less than 1.

        .. deprecated:: 3.2
            The ``eps`` keyword is deprecated. Use ``ellipticity``
            instead.
    orientation : float or `~astropy.units.Quantity`
        The orientation of the semimajor axis, measured
        counterclockwise from the positive x axis of the image array.
        It is an angular `~astropy.units.Quantity` (e.g., an
        `~astropy.coordinates.Angle`) or a float in radians. The
        ``orientation`` attribute is a `~astropy.units.Quantity` in
        degrees. The fitted orientations are in the range of 0 to 180
        degrees. When the orientations are near either end of the
        range, noise can make the solution jump back and forth between
        successive isophotes, by amounts close to 180 degrees.

        .. deprecated:: 3.2
            The ``pa`` keyword is deprecated. Use ``orientation``
            instead.
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
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('eps', 'ellipticity', '3.2', until='4.0')
    @deprecated_renamed_argument('pa', 'orientation', '3.2', until='4.0')
    @deprecated_renamed_argument('astep', 'step', '3.2', until='4.0')
    @deprecated_renamed_argument('fix_pa', 'fix_orientation', '3.2',
                                 until='4.0')
    @deprecated_renamed_argument('fix_eps', 'fix_ellipticity', '3.2',
                                 until='4.0')
    def __init__(self, x0, y0, sma, ellipticity, orientation, step=0.1,
                 linear_growth=False, fix_center=False,
                 fix_orientation=False, fix_ellipticity=False):
        self.x0 = x0
        self.y0 = y0
        self.sma = sma
        self.ellipticity = ellipticity
        self.orientation = orientation

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
        of the image array. It can be set with an angular
        `~astropy.units.Quantity` or with a float in radians.
        """
        return np.rad2deg(self._orientation_rad) << u.deg

    @orientation.setter
    def orientation(self, value):  # numpydoc ignore=GL08
        # The internal code uses the orientation as a float in radians.
        if isinstance(value, u.Quantity):
            value = value.to_value(u.rad)
        self._orientation_rad = value

    @property
    def fix_center(self):
        """
        Whether the center is kept fixed during the fit.
        """
        return bool(self._fix[0] and self._fix[1])

    @property
    def fix_orientation(self):
        """
        Whether the orientation is kept fixed during the fit.
        """
        return bool(self._fix[_FIX_INDEX['orientation']])

    @property
    def fix_ellipticity(self):
        """
        Whether the ellipticity is kept fixed during the fit.
        """
        return bool(self._fix[_FIX_INDEX['ellipticity']])

    # Remove in 4.0
    fix = deprecated_attribute('fix', '_fix', '3.2', until='4.0')
    sector_angular_width = deprecated_attribute(
        'sector_angular_width', '_sector_angular_width', '3.2',
        until='4.0')
    initial_polar_angle = deprecated_attribute(
        'initial_polar_angle', '_initial_polar_angle', '3.2', until='4.0')
    initial_polar_radius = deprecated_attribute(
        'initial_polar_radius', '_initial_polar_radius', '3.2',
        until='4.0')

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
        self.orientation = value

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
    def find_center(self, image, threshold=0.1, verbose=True):
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
            to `numpy.inf`. The default is 0.1.

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

        At each position, the figure of merit is the difference between
        the mean of the pixels within 3.5 pixels and the mean of the
        pixels in a ring with radii from 7.5 to 8.5 pixels, divided by
        the quadrature sum of their standard deviations. The position
        with the largest figure of merit is accepted if that value
        exceeds ``threshold``. The new center is a whole-pixel position.

        Only the window positions inside the image are scanned. Near
        the image edges, only the pixels inside the image are used to
        compute the figure of merit.

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
        half_size = len(IN_MASK) // 2

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

                # Clip the box at the image edges. The masks are clipped
                # by the same amount so that they stay aligned with the
                # image pixels.
                i1 = i - half_size
                j1 = j - half_size
                i1_clip = max(i1, 0)
                j1_clip = max(j1, 0)
                i2_clip = min(i + half_size + 1, shape[1])
                j2_clip = min(j + half_size + 1, shape[0])
                window = np.ma.masked_invalid(
                    image[j1_clip:j2_clip, i1_clip:i2_clip])
                mask_slices = (slice(j1_clip - j1, j2_clip - j1),
                               slice(i1_clip - i1, i2_clip - i1))

                # Averages in inner and outer regions.
                inner = np.ma.masked_array(window, mask=IN_MASK[mask_slices])
                outer = np.ma.masked_array(window,
                                           mask=OUT_MASK[mask_slices])
                inner_avg = np.ma.mean(inner)
                outer_avg = np.ma.mean(outer)

                # Standard deviation and figure of merit
                inner_std = np.std(inner)
                outer_std = np.std(outer)
                stddev = np.sqrt(inner_std**2 + outer_std**2)

                # As in the STSDAS ellipse task, a position without any
                # scatter (e.g., in a flat image), where the figure of
                # merit is not defined, is skipped.
                if not stddev > 0:
                    continue

                fom = (inner_avg - outer_avg) / stddev

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
