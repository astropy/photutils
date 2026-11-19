# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for fitting ellipses.
"""

import math
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.harmonics import (first_and_second_harmonic_function,
                                          fit_first_and_second_harmonics)
from photutils.isophote.isophote import CentralPixel, Isophote
from photutils.utils._deprecation import deprecated_renamed_argument

__all__ = ['EllipseFitter']

__doctest_skip__ = ['EllipseFitter.fit']


PI2 = np.pi / 2
MAX_EPS = 0.95
MIN_EPS = 0.05

DEFAULT_CONVERGENCE = 0.05
DEFAULT_MINIT = 10
DEFAULT_MAXIT = 50
DEFAULT_FFLAG = 0.7
DEFAULT_MAXGERR = 0.5


class EllipseFitter:
    """
    Class to fit ellipses.

    Parameters
    ----------
    sample : `~photutils.isophote.EllipseSample` instance
        The sample data to be fitted.
    """

    def __init__(self, sample):
        self._sample = sample

    @deprecated_renamed_argument('maxgerr', None, '3.2', until='4.0')
    @deprecated_renamed_argument('going_inwards', None, '3.2', until='4.0')
    def fit(self, *, conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
            maxgerr=None, going_inwards=None):  # noqa: ARG002
        """
        Fit an elliptical isophote.

        Parameters
        ----------
        conver : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``conver`` times the harmonic fit rms times the
            square root of the mean sector area of the sample points
            (see the ``sector_area`` attribute of
            `~photutils.isophote.EllipseSample`). The default is 0.05.

        minit : int, optional
            The minimum number of iterations to perform. A minimum of
            10 (the default) iterations guarantees that, on average, 2
            iterations will be available for fitting each independent
            parameter (the four harmonic amplitudes and the intensity
            level). For the first isophote, the minimum number of
            iterations is 2 * ``minit`` to ensure that, even departing
            from not-so-good initial values, the algorithm has a better
            chance to converge to a sensible solution.

        maxit : int, optional
            The maximum number of iterations to perform. The default is
            50.

        fflag : float, optional
            The acceptable fraction of flagged data points in the
            sample. If the actual fraction of valid data points is
            smaller than this, the iterations will stop and the current
            `~photutils.isophote.Isophote` will be returned. Flagged
            data points are points that either lie outside the image
            frame, are masked or nonfinite, or were rejected by
            sigma-clipping. The default is 0.7.

        maxgerr : float, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0. The ``maxgerr`` criterion compares
               consecutive isophotes. It is applied by
               :meth:`~photutils.isophote.Ellipse.fit_image`, so it has
               no effect when fitting a single isophote.

        going_inwards : bool, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0.

        Returns
        -------
        result : `~photutils.isophote.Isophote` instance
            The fitted isophote, which also contains fit status
            information.

        Examples
        --------
        >>> from photutils.isophote import EllipseSample, EllipseFitter
        >>> sample = EllipseSample(data, sma=10.0)
        >>> fitter = EllipseFitter(sample)
        >>> isophote = fitter.fit()
        """
        sample = self._sample

        # Here we keep track of the sample that caused the minimum harmonic
        # amplitude(in absolute value). This will eventually be used to
        # build the resulting Isophote in cases where iterations run to
        # the maximum allowed (maxit), or the maximum number of flagged
        # data points (fflag) is reached.
        minimum_amplitude_value = np.inf
        minimum_amplitude_sample = None

        # These must be passed throughout the execution chain.
        fixed_parameters = self._sample.geometry.fix

        # If all the ellipse parameters are fixed, there is nothing to
        # fit. As in the STSDAS ellipse task, the isophote is extracted
        # with the input geometry, as in the noniterative mode.
        if np.all(fixed_parameters):
            sample.update(fixed_parameters=fixed_parameters)
            return Isophote(sample, 0, valid=True, stop_code=4)

        for i in range(maxit):
            # Force the sample to compute its gradient and associated values.
            sample.update(fixed_parameters=fixed_parameters)

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
                    coeffs = fit_first_and_second_harmonics(values[0],
                                                            values[2])[0]
                except Exception as exc:
                    msg = ('The harmonic fit failed at SMA '
                           f'{sample.geometry.sma}: {exc}')
            if msg is not None:
                warnings.warn(msg, AstropyUserWarning)
                sample.geometry.fix = fixed_parameters
                return Isophote(sample, i + 1, valid=False, stop_code=3)

            # Mask out coefficients that control fixed ellipse parameters.
            free_coeffs = np.ma.masked_array(coeffs[1:], mask=fixed_parameters)

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
            model = first_and_second_harmonic_function(values[0], coeffs)
            residual = values[2] - model

            if ((conver * np.sqrt(sample.sector_area) * np.std(residual))
                    > np.abs(largest_harmonic)) and (i >= minit - 1):
                # Got a valid solution and a minimum number of
                # iterations has run.
                sample.update(fixed_parameters=fixed_parameters)
                return Isophote(sample, i + 1, valid=True, stop_code=0)

            # It may not have converged yet, but the sample contains too
            # many invalid data points, so return the best-fit sample
            # instead of the current one.
            if sample.actual_points < (sample.total_points * fflag):
                # When too many data points were flagged, return the
                # best-fit sample instead of the current one.
                minimum_amplitude_sample.update(
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
                sample = corrector.correct(sample, largest_harmonic)

                # If the corrected ellipse diverged, stop without
                # sampling it. Otherwise, fix a round ellipse before the
                # new sample is extracted so that its values match its
                # geometry.
                if self._is_diverged(sample):
                    proceed = False
                else:
                    self._fix_round_ellipse(sample.geometry)
                    sample.update(fixed_parameters=fixed_parameters)

            # If iterations cannot proceed, return the best-fit sample
            # instead of the current one.
            if not proceed:
                minimum_amplitude_sample.update(
                    fixed_parameters=fixed_parameters)
                return Isophote(minimum_amplitude_sample, i + 1, valid=True,
                                stop_code=-1)

        # Got to the maximum number of iterations. Return with
        # code 2, and handle it as a valid isophote. Use the
        # best-fit sample instead of the current one.
        minimum_amplitude_sample.update(fixed_parameters=fixed_parameters)
        return Isophote(minimum_amplitude_sample, maxit, valid=True,
                        stop_code=2)

    @staticmethod
    def _fix_round_ellipse(geometry):
        """
        Fix an ellipse geometry whose ellipticity crossed or reached
        zero.

        This must be called before the sample with this geometry is
        extracted.
        """
        # See if eps == 0 (round isophote) was crossed.
        # If so, fix it but still proceed.
        if geometry.eps < 0.0:
            geometry.eps = min(-geometry.eps, MAX_EPS)
            if geometry.pa < PI2:
                geometry.pa += PI2
            else:
                geometry.pa -= PI2

        # If ellipse is an exact circle, computations will diverge.
        # Make it slightly flat, but still proceed.
        if geometry.eps == 0.0:
            geometry.eps = MIN_EPS

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
        return (abs(geometry.eps) > MAX_EPS
                or not 0.0 <= geometry.x0 <= nx - 1
                or not 0.0 <= geometry.y0 <= ny - 1)


class _ParameterCorrector:
    def correct(self, sample, harmonic):
        raise NotImplementedError


class _PositionCorrector(_ParameterCorrector):
    @staticmethod
    def finalize_correction(dx, dy, sample):
        return sample._make_sample(x0=sample.geometry.x0 + dx,
                                   y0=sample.geometry.y0 + dy)


class _PositionCorrector0(_PositionCorrector):
    def correct(self, sample, harmonic):
        aux = -harmonic * (1.0 - sample.geometry.eps) / sample.gradient

        dx = -aux * math.sin(sample.geometry.pa)
        dy = aux * math.cos(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class _PositionCorrector1(_PositionCorrector):
    def correct(self, sample, harmonic):
        aux = -harmonic / sample.gradient

        dx = aux * math.cos(sample.geometry.pa)
        dy = aux * math.sin(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class _AngleCorrector(_ParameterCorrector):
    def correct(self, sample, harmonic):
        eps = sample.geometry.eps
        sma = sample.geometry.sma
        gradient = sample.gradient

        correction = (harmonic * 2.0 * (1.0 - eps) / sma / gradient
                      / ((1.0 - eps)**2 - 1.0))

        # Make the angle lie between 0 and np.pi radians.
        new_pa = (sample.geometry.pa + correction) % np.pi

        return sample._make_sample(pa=new_pa)


class _EllipticityCorrector(_ParameterCorrector):
    def correct(self, sample, harmonic):
        eps = sample.geometry.eps
        sma = sample.geometry.sma
        gradient = sample.gradient

        correction = harmonic * 2.0 * (1.0 - eps) / sma / gradient

        new_eps = sample.geometry.eps - correction

        return sample._make_sample(eps=new_eps)


# Instances of corrector code
_CORRECTORS = [_PositionCorrector0(), _PositionCorrector1(),
               _AngleCorrector(), _EllipticityCorrector()]


class CentralEllipseFitter(EllipseFitter):
    """
    A special Fitter class to handle the case of the central pixel in
    the galaxy image.
    """

    def fit(self, **kwargs):  # noqa: ARG002
        """
        Perform just a simple 1-pixel extraction at the current (x0, y0)
        position using bilinear interpolation.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are ignored, but allowed to match
            the calling signature of the parent class.

        Returns
        -------
        result : `~photutils.isophote.CentralEllipsePixel` instance
            The central pixel value. For convenience, the
            `~photutils.isophote.CentralEllipsePixel` class inherits
            from the `~photutils.isophote.Isophote` class, although it's
            not really a true isophote but just a single intensity value
            at the central position. Thus, most of its attributes are
            hardcoded to `None` or other default value when appropriate.
        """
        # Default values
        fixed_parameters = np.array([False, False, False, False])

        self._sample.update(fixed_parameters=fixed_parameters)
        return CentralPixel(self._sample)
