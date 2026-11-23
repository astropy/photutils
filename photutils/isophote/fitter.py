# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for fitting ellipses.
"""

import copy
import math
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.geometry import (_CENTER_INDICES, _ELLIPTICITY_INDEX,
                                         _HARMONIC_PARAMETERS,
                                         _ORIENTATION_INDEX)
from photutils.isophote.harmonics import (_first_and_second_harmonic_function,
                                          _fit_first_and_second_harmonics)
from photutils.isophote.isophote import Isophote
from photutils.utils._deprecation import deprecated_renamed_argument
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['EllipseFitter']


PI2 = np.pi / 2
MAX_EPS = 0.95
MIN_EPS = 0.05

DEFAULT_CONVERGENCE = 0.05
DEFAULT_MINITERS = 10
DEFAULT_MAXITERS = 50
DEFAULT_MIN_VALID_FRACTION = 0.7
DEFAULT_MAX_GRADIENT_REL_ERR = 0.5


class _EllipseFitter:
    """
    Class to fit ellipses.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample data to be fitted.
    """

    def __init__(self, sample):
        self._sample = sample

    @deprecated_renamed_argument(
        ['maxgerr', 'going_inwards', 'conver', 'minit', 'maxit', 'fflag'],
        [None, None, 'convergence', 'miniters', 'maxiters',
         'min_valid_fraction'],
        '3.2', until='4.0')
    def fit(self, *, convergence=DEFAULT_CONVERGENCE,
            miniters=DEFAULT_MINITERS, maxiters=DEFAULT_MAXITERS,
            min_valid_fraction=DEFAULT_MIN_VALID_FRACTION,
            maxgerr=None, going_inwards=None,  # noqa: ARG002
            max_center_shift=None, reference_center=None):
        """
        Fit an elliptical isophote.

        The input sample is not modified.

        If all the ellipse parameters are fixed by the geometry of
        the sample, there is nothing to fit. The isophote is then
        extracted with that geometry and its stop code is 4, or 3
        (invalid) if it has no valid data points. If the ellipticity
        is fixed at zero, the orientation is also treated as fixed.

        Parameters
        ----------
        convergence : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``convergence`` times the harmonic fit rms
            times the square root of the mean sector area of the sample
            points (see the ``sector_area`` attribute of
            `~photutils.isophote.EllipseSample`). The default is 0.05.

        miniters : int, optional
            The minimum number of iterations to perform. A minimum of
            10 (the default) iterations guarantees that, on average, 2
            iterations will be available for fitting each independent
            parameter (the four harmonic amplitudes and the intensity
            level).

        maxiters : int, optional
            The maximum number of iterations to perform. The default is
            50.

        min_valid_fraction : float, optional
            The minimum acceptable fraction of valid (nonflagged)
            data points in the sample. If the fraction of valid data
            points is smaller than this, the iterations stop and the
            isophote with the smallest harmonic amplitude found so far
            is returned with a stop code of 1. Flagged data points are
            those that lie outside the image, are masked or nonfinite,
            or were rejected by sigma clipping. The default is 0.7.

        maxgerr : float, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0. The ``max_gradient_rel_err`` criterion
               compares consecutive isophotes. It is applied by
               :meth:`~photutils.isophote.Ellipse.fit_image`, so it has
               no effect when fitting a single isophote.

        going_inwards : bool, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0.

        max_center_shift : float or `None`, optional
            The maximum distance (in pixels) that the ellipse center
            may move from ``reference_center``. If an iteration would
            move the center farther than this, the center is placed on
            the boundary at this distance and held fixed for the rest
            of the fit, while the other free parameters continue to be
            fitted. If holding the center fixed leaves no free
            parameters, the fit stops with ``stop_code=0``. If `None`
            (default), the center is not limited.

        reference_center : tuple of 2 floats or `None`, optional
            The ``(x, y)`` position from which ``max_center_shift``
            is measured. If `None` (default), the center of the input
            sample is used. This parameter is ignored if
            ``max_center_shift`` is `None`.

        Returns
        -------
        result : `~photutils.isophote.Isophote` instance
            The fitted isophote, which also contains fit status
            information.

        Raises
        ------
        ValueError
            If ``max_center_shift`` is negative or NaN.

        Warns
        -----
        AstropyUserWarning
            If the sample has fewer than 5 valid data points or the
            harmonic fit fails. The returned isophote is then invalid
            and has a stop code of 3.

        Examples
        --------
        The `~photutils.isophote.EllipseFitter` class is deprecated. Use
        :meth:`~photutils.isophote.Ellipse.fit_isophote` instead to fit
        a single isophote:

        >>> import numpy as np
        >>> from astropy.modeling.models import Gaussian2D
        >>> from photutils.datasets import make_noise_image
        >>> from photutils.isophote import Ellipse, EllipseGeometry
        >>> model = Gaussian2D(100.0, 75, 75, 20, 12,
        ...                    theta=np.deg2rad(40))
        >>> y, x = np.mgrid[0:150, 0:150]
        >>> data = model(x, y) + make_noise_image((150, 150), mean=0.0,
        ...                                       stddev=2.0, seed=0)
        >>> geometry = EllipseGeometry(x0=75, y0=75, sma=20,
        ...                            ellipticity=0.5,
        ...                            orientation=np.deg2rad(30))
        >>> ellipse = Ellipse(data, geometry=geometry)
        >>> isophote = ellipse.fit_isophote(20)
        >>> isophote.stop_code
        0
        """
        if max_center_shift is not None and not max_center_shift >= 0:
            msg = 'max_center_shift must be a non-negative number'
            raise ValueError(msg)

        # The fit works on a copy of the input sample with its own
        # geometry, because a round starting ellipse and the fixed
        # parameters change the geometry. The copy keeps any values
        # already extracted from the image.
        sample = copy.copy(self._sample)
        sample.geometry = copy.deepcopy(self._sample.geometry)
        if reference_center is None:
            reference_center = (sample.geometry.x0, sample.geometry.y0)

        # Here we keep track of the sample that caused the minimum harmonic
        # amplitude(in absolute value). This will eventually be used to
        # build the resulting Isophote in cases where iterations run to
        # the maximum allowed (maxiters), or the maximum number of
        # flagged data points (min_valid_fraction) is reached.
        minimum_amplitude_value = np.inf
        minimum_amplitude_sample = None

        # These must be passed throughout the execution chain.
        fixed_parameters = sample.geometry._fix

        # The orientation of a circle is undefined and its correction
        # diverges, so it is not fit if the ellipticity is fixed at
        # zero.
        if (fixed_parameters[_ELLIPTICITY_INDEX]
                and sample.geometry.ellipticity == 0.0):
            fixed_parameters = np.array(fixed_parameters)
            fixed_parameters[_ORIENTATION_INDEX] = True

        # If all the ellipse parameters are fixed, there is nothing to
        # fit. As in the STSDAS ellipse task, the isophote is extracted
        # with the input geometry, as in the noniterative mode. An
        # isophote without any valid data points is invalid.
        if np.all(fixed_parameters):
            sample._update(fixed_parameters=fixed_parameters)
            valid = sample.n_valid > 0
            return Isophote(sample, 0, valid=valid,
                            stop_code=4 if valid else 3)

        # A starting ellipse that is round or has a negative ellipticity
        # must be fixed before it is sampled, like the corrected
        # ellipses below.
        if (sample.geometry.ellipticity <= 0.0
                and not fixed_parameters[_ELLIPTICITY_INDEX]):
            self._fix_round_ellipse(sample.geometry, fixed_parameters)
            sample._values = None

        # The parameters held fixed during this fit. These include the
        # center once it reaches the ``max_center_shift`` boundary. The
        # isophote geometry keeps ``fixed_parameters`` so that the next
        # isophote can move its center again.
        fit_fixed = np.array(fixed_parameters, dtype=bool)
        center_at_bound = False

        for i in range(maxiters):
            # Force the sample to compute its gradient and associated values.
            sample._update(fixed_parameters=fixed_parameters)

            # The extract() method returns sampled values as a 2D numpy
            # array with the following structure:
            # values[0] = 1-d array with angles
            # values[1] = 1-d array with radii
            # values[2] = 1-d array with intensity
            values = sample.extract()

            # Fit harmonic coefficients. A sample with fewer data points
            # than the 5 harmonic coefficients or a failure in the fit
            # is a fatal error, so terminate immediately with the sample
            # marked as invalid.
            msg = None
            if len(values[2]) < 5:
                msg = ('Too small sample to warrant a fit. SMA is '
                       f'{sample.geometry.sma}')
            else:
                try:
                    coeffs = _fit_first_and_second_harmonics(values[0],
                                                             values[2])[0]
                except Exception as exc:
                    msg = ('The harmonic fit failed at SMA '
                           f'{sample.geometry.sma}: {exc}')
            if msg is not None:
                warnings.warn(msg, AstropyUserWarning)
                sample.geometry._fix = fixed_parameters
                return Isophote(sample, i + 1, valid=False, stop_code=3)

            # Mask out coefficients that control fixed ellipse parameters.
            free_coeffs = np.ma.masked_array(coeffs[1:], mask=fit_fixed)

            # Largest nonmasked harmonic in absolute value drives the
            # correction.
            largest_harmonic_index = np.argmax(np.abs(free_coeffs))
            largest_harmonic = free_coeffs[largest_harmonic_index]

            # See if the amplitude decreased. If yes, keep the
            # corresponding sample for eventual later use.
            if abs(largest_harmonic) < minimum_amplitude_value:
                minimum_amplitude_value = abs(largest_harmonic)
                minimum_amplitude_sample = sample

            # Check if converged. As in the STSDAS ellipse task, the
            # residual rms is scaled by the square root of the sector
            # area to account for the smoothing from area averaging.
            model = _first_and_second_harmonic_function(values[0], coeffs)
            residual = values[2] - model

            if ((convergence * np.sqrt(sample.sector_area) * np.std(residual))
                    > np.abs(largest_harmonic)) and (i >= miniters - 1):
                # Got a valid solution and a minimum number of
                # iterations has run.
                sample._update(fixed_parameters=fixed_parameters)
                return Isophote(sample, i + 1, valid=True, stop_code=0)

            # It may not have converged yet, but the sample contains too
            # many invalid data points, so return the best-fit sample
            # instead of the current one.
            if sample.n_valid < (sample.n_total * min_valid_fraction):
                # When too many data points were flagged, return the
                # best-fit sample instead of the current one.
                minimum_amplitude_sample._update(
                    fixed_parameters=fixed_parameters)
                return Isophote(minimum_amplitude_sample, i + 1, valid=True,
                                stop_code=1)

            # The parameter correctors divide by the sample's local
            # gradient, which must be negative (the intensity decreases
            # outward) for the corrections to have the right sign. If
            # the gradient is zero, positive, or nonfinite (e.g., from
            # a degenerate sample where no gradient could be measured),
            # no correction can be computed and iterations cannot
            # proceed.
            proceed = (sample.gradient < 0.0
                       and np.isfinite(sample.gradient))

            if proceed:
                # Pick appropriate corrector code.
                corrector = _CORRECTORS[largest_harmonic_index]

                # Generate a new EllipseSample instance with corrected
                # parameter. Note that this instance is still devoid of
                # other information besides its geometry. It needs to
                # be explicitly updated for computations to proceed.
                # We have to build a new EllipseSample instance every
                # time because of the lazy extraction process used
                # by EllipseSample code. To minimize the number of
                # calls to the area integrators, we pay a (hopefully
                # smaller) price here, by having multiple calls to the
                # EllipseSample constructor.
                previous_gradient = sample.gradient
                sample = corrector(sample, largest_harmonic)

                # Only the position correctors move the center.
                if (max_center_shift is not None
                        and largest_harmonic_index in _CENTER_INDICES):
                    sample, center_at_bound = _limit_center_shift(
                        sample, reference_center, max_center_shift)
                    if center_at_bound:
                        fit_fixed[list(_CENTER_INDICES)] = True

                # As in the STSDAS ellipse task, the gradient of this
                # iteration is the reference that tells whether the
                # gradient measured for the corrected ellipse is
                # meaningful. It is set after the center is limited,
                # which can make a new sample.
                sample._reference_gradient = previous_gradient

                # If the corrected ellipse diverged, stop without
                # sampling it. Otherwise, fix a round ellipse before the
                # new sample is extracted so that its values match its
                # geometry.
                if self._is_diverged(sample):
                    proceed = False
                else:
                    self._fix_round_ellipse(sample.geometry,
                                            fixed_parameters)
                    sample._update(fixed_parameters=fixed_parameters)

            # If iterations cannot proceed, return the best-fit sample
            # instead of the current one. Holding the center at the
            # boundary can leave no free parameters, in which case the
            # fit is complete and the current sample is returned.
            complete = proceed and center_at_bound and fit_fixed.all()
            if complete or not proceed:
                final_sample = sample if complete else minimum_amplitude_sample
                final_sample._update(fixed_parameters=fixed_parameters)
                return Isophote(final_sample, i + 1, valid=True,
                                stop_code=0 if complete else -1)

        # Got to the maximum number of iterations. Return with
        # code 2, and handle it as a valid isophote. Use the
        # best-fit sample instead of the current one.
        minimum_amplitude_sample._update(fixed_parameters=fixed_parameters)
        return Isophote(minimum_amplitude_sample, maxiters, valid=True,
                        stop_code=2)

    @staticmethod
    def _fix_round_ellipse(geometry, fixed_parameters):
        """
        Adjust an ellipse geometry whose ellipticity crossed or
        reached zero.

        This must be called before the sample with this geometry is
        extracted. A fixed ellipticity is not changed. A fixed
        orientation is not changed either.
        """
        if fixed_parameters[_ELLIPTICITY_INDEX]:
            return

        # See if eps == 0 (round isophote) was crossed. If so, fix it
        # but still proceed. An ellipse with a negative ellipticity is
        # the same as an ellipse rotated by 90 degrees with a positive
        # ellipticity. A fixed orientation cannot be rotated, so the
        # ellipse is then made nearly round below.
        if geometry.ellipticity < 0.0:
            if fixed_parameters[_ORIENTATION_INDEX]:
                geometry.ellipticity = 0.0
            else:
                geometry.ellipticity = min(-geometry.ellipticity, MAX_EPS)
                if geometry._orientation_rad < PI2:
                    geometry._orientation_rad += PI2
                else:
                    geometry._orientation_rad -= PI2

        # If ellipse is an exact circle, computations will diverge.
        # Make it slightly flat, but still proceed.
        if geometry.ellipticity == 0.0:
            geometry.ellipticity = MIN_EPS

        geometry._reset_sma_dependent_attributes()

    @staticmethod
    def _is_diverged(sample):
        """
        Return whether the ellipticity is too large or the ellipse
        center lies outside of the image.

        The center limits are the centers of the first and last
        pixels.
        """
        geometry = sample.geometry
        ny, nx = sample.image.shape
        return (abs(geometry.ellipticity) > MAX_EPS
                or not 0.0 <= geometry.x0 <= nx - 1
                or not 0.0 <= geometry.y0 <= ny - 1)


def _shift_center(sample, dx, dy):
    """
    Return a new sample with the center shifted by ``(dx, dy)``.
    """
    return sample._make_sample(x0=sample.geometry.x0 + dx,
                               y0=sample.geometry.y0 + dy)


def _limit_center_shift(sample, reference_center, max_shift):
    """
    Move the sample center onto the ``max_shift`` boundary if it is
    beyond it.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample whose center is checked.

    reference_center : tuple of 2 floats
        The ``(x, y)`` position from which the shift is measured.

    max_shift : float
        The maximum allowed distance (in pixels) from
        ``reference_center``.

    Returns
    -------
    sample : `~photutils.isophote.EllipseSample` instance
        The input sample if its center is within ``max_shift``.
        Otherwise, a new sample whose center is on the boundary, along
        the direction from ``reference_center`` to the input center.

    at_bound : bool
        Whether the center was moved onto the boundary.
    """
    dx = sample.geometry.x0 - reference_center[0]
    dy = sample.geometry.y0 - reference_center[1]
    shift = math.hypot(dx, dy)
    if shift <= max_shift:
        return sample, False

    scale = max_shift / shift - 1.0
    sample = _shift_center(sample, dx * scale, dy * scale)
    return sample, True


def _correct_center_a1(sample, harmonic):
    """
    Return a new sample with the center corrected using the ``a1``
    harmonic amplitude.
    """
    aux = -harmonic * (1.0 - sample.geometry.ellipticity) / sample.gradient

    dx = -aux * math.sin(sample.geometry._orientation_rad)
    dy = aux * math.cos(sample.geometry._orientation_rad)

    return _shift_center(sample, dx, dy)


def _correct_center_b1(sample, harmonic):
    """
    Return a new sample with the center corrected using the ``b1``
    harmonic amplitude.
    """
    aux = -harmonic / sample.gradient

    dx = aux * math.cos(sample.geometry._orientation_rad)
    dy = aux * math.sin(sample.geometry._orientation_rad)

    return _shift_center(sample, dx, dy)


def _correct_orientation(sample, harmonic):
    """
    Return a new sample with the orientation corrected using the
    ``a2`` harmonic amplitude.
    """
    eps = sample.geometry.ellipticity
    sma = sample.geometry.sma
    gradient = sample.gradient

    correction = (harmonic * 2.0 * (1.0 - eps) / sma / gradient
                  / ((1.0 - eps)**2 - 1.0))

    # Make the angle lie between 0 and np.pi radians.
    new_orientation = (sample.geometry._orientation_rad + correction) % np.pi

    return sample._make_sample(orientation=new_orientation)


def _correct_ellipticity(sample, harmonic):
    """
    Return a new sample with the ellipticity corrected using the ``b2``
    harmonic amplitude.
    """
    eps = sample.geometry.ellipticity
    sma = sample.geometry.sma
    gradient = sample.gradient

    correction = harmonic * 2.0 * (1.0 - eps) / sma / gradient

    new_eps = sample.geometry.ellipticity - correction

    return sample._make_sample(ellipticity=new_eps)


# The parameter corrector for each harmonic coefficient. The correctors
# are listed in the order of the coefficients returned by
# _fit_first_and_second_harmonics, which is also the order of the
# ``fix`` array of EllipseGeometry.
_CORRECTORS_BY_COEFFICIENT = {
    'a1': _correct_center_a1,
    'b1': _correct_center_b1,
    'a2': _correct_orientation,
    'b2': _correct_ellipticity,
}
_CORRECTORS = [_CORRECTORS_BY_COEFFICIENT[name]
               for name in _HARMONIC_PARAMETERS]


class EllipseFitter(_EllipseFitter):
    """
    Class to fit ellipses.

    .. deprecated:: 3.2
        This class is deprecated and will be removed in version 4.0.
        Use `~photutils.isophote.Ellipse.fit_isophote` instead.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample data to be fitted.
    """

    def __init__(self, sample):
        msg = ('EllipseFitter was deprecated in version 3.2 and will be '
               'removed in version 4.0. Use Ellipse.fit_isophote instead.')
        warnings.warn(msg, PhotutilsDeprecationWarning, stacklevel=2)
        super().__init__(sample)
