# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides a class to fit ellipses.
"""

import math

from astropy import log
import numpy as np
import numpy.ma as ma

from .harmonics import (first_and_second_harmonic_function,
                        fit_first_and_second_harmonics)
from .isophote import CentralPixel, Isophote
from .sample import EllipseSample

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

    def fit(self, conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
            going_inwards=False):
        """
        Fit an elliptical isophote.

        Parameters
        ----------
        conver : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``conver`` times the harmonic fit rms.  The
            default is 0.05.
        minit : int, optional
            The minimum number of iterations to perform. A minimum of 10
            (the default) iterations guarantees that, on average, 2
            iterations will be available for fitting each independent
            parameter (the four harmonic amplitudes and the intensity
            level). For the first isophote, the minimum number of
            iterations is 2 * ``minit`` to ensure that, even departing
            from not-so-good initial values, the algorithm has a better
            chance to converge to a sensible solution.
        maxit : int, optional
            The maximum number of iterations to perform.  The default is
            50.
        fflag : float, optional
            The acceptable fraction of flagged data points in the
            sample.  If the actual fraction of valid data points is
            smaller than this, the iterations will stop and the current
            `~photutils.isophote.Isophote` will be returned.  Flagged
            data points are points that either lie outside the image
            frame, are masked, or were rejected by sigma-clipping.  The
            default is 0.7.
        maxgerr : float, optional
            The maximum acceptable relative error in the local radial
            intensity gradient. This is the main control for preventing
            ellipses to grow to regions of too low signal-to-noise
            ratio.  It specifies the maximum acceptable relative error
            in the local radial intensity gradient.  `Busko (1996; ASPC
            101, 139)
            <https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_
            showed that the fitting precision relates to that relative
            error.  The usual behavior of the gradient relative error is
            to increase with semimajor axis, being larger in outer,
            fainter regions of a galaxy image.  In the current
            implementation, the ``maxgerr`` criterion is triggered only
            when two consecutive isophotes exceed the value specified by
            the parameter. This prevents premature stopping caused by
            contamination such as stars and HII regions.

            A number of actions may happen when the gradient error
            exceeds ``maxgerr`` (or becomes non-significant and is set
            to `None`).  If the maximum semimajor axis specified by
            ``maxsma`` is set to `None`, semimajor axis growth is
            stopped and the algorithm proceeds inwards to the galaxy
            center. If ``maxsma`` is set to some finite value, and this
            value is larger than the current semimajor axis length, the
            algorithm enters non-iterative mode and proceeds outwards
            until reaching ``maxsma``.  The default is 0.5.
        going_inwards : bool, optional
            Parameter to define the sense of SMA growth. When fitting
            just one isophote, this parameter is used only by the code
            that defines the details of how elliptical arc segments
            ("sectors") are extracted from the image, when using area
            extraction modes (see the ``integrmode`` parameter in the
            `~photutils.isophote.EllipseSample` class).  The default is
            `False`.

        Returns
        -------
        result : `~photutils.isophote.Isophote` instance
            The fitted isophote, which also contains fit status
            information.

        Examples
        --------
        >>> from photutils.isophote import EllipseSample, EllipseFitter
        >>> sample = EllipseSample(data, sma=10.)
        >>> fitter = EllipseFitter(sample)
        >>> isophote = fitter.fit()
        """

        sample = self._sample

        # this flag signals that limiting gradient error (`maxgerr`)
        # wasn't exceeded yet.
        lexceed = False

        # here we keep track of the sample that caused the minimum harmonic
        # amplitude(in absolute value). This will eventually be used to
        # build the resulting Isophote in cases where iterations run to
        # the maximum allowed (maxit), or the maximum number of flagged
        # data points (fflag) is reached.
        minimum_amplitude_value = np.Inf
        minimum_amplitude_sample = None

        # these must be passed throughout the execution chain.
        fixed_parameters = self._sample.geometry.fix

        for i in range(maxit):
            # Force the sample to compute its gradient and associated values.
            sample.update(fixed_parameters)

            # The extract() method returns sampled values as a 2-d numpy array
            # with the following structure:
            # values[0] = 1-d array with angles
            # values[1] = 1-d array with radii
            # values[2] = 1-d array with intensity
            values = sample.extract()

            # We have to check for a zero-length condition here, and bail out
            # in case it is detected. The scipy fitter won't raise an exception
            # for zero-length input arrays, but just prints an "INFO" message.
            # This may result in an infinite loop.
            if len(values[2]) < 1:
                s = str(sample.geometry.sma)
                log.warning("Too small sample to warrant a fit. SMA is " + s)
                sample.geometry.fix = fixed_parameters
                return Isophote(sample, i + 1, False, 3)

            # Fit harmonic coefficients. Failure in fitting is
            # a fatal error; terminate immediately with sample
            # marked as invalid.
            try:
                coeffs = fit_first_and_second_harmonics(values[0], values[2])
                coeffs = coeffs[0]
            except Exception as e:
                log.warning(e)
                sample.geometry.fix = fixed_parameters
                return Isophote(sample, i + 1, False, 3)

            # Mask out coefficients that control fixed ellipse parameters.
            free_coeffs = ma.masked_array(coeffs[1:], mask=fixed_parameters)

            # Largest non-masked harmonic in absolute value drives the
            # correction.
            largest_harmonic_index = np.argmax(np.abs(free_coeffs))
            largest_harmonic = free_coeffs[largest_harmonic_index]

            # see if the amplitude decreased; if yes, keep the
            # corresponding sample for eventual later use.
            if abs(largest_harmonic) < minimum_amplitude_value:
                minimum_amplitude_value = abs(largest_harmonic)
                minimum_amplitude_sample = sample

            # check if converged
            model = first_and_second_harmonic_function(values[0], coeffs)
            residual = values[2] - model

            if ((conver * sample.sector_area * np.std(residual))
                    > np.abs(largest_harmonic)):
                # Got a valid solution. But before returning, ensure
                # that a minimum of iterations has run.
                if i >= minit - 1:
                    sample.update(fixed_parameters)
                    return Isophote(sample, i + 1, True, 0)

            # it may not have converged yet, but the sample contains too
            # many invalid data points: return.
            if sample.actual_points < (sample.total_points * fflag):
                # when too many data points were flagged, return the
                # best fit sample instead of the current one.
                minimum_amplitude_sample.update(fixed_parameters)
                return Isophote(minimum_amplitude_sample, i + 1, True, 1)

            # pick appropriate corrector code.
            corrector = _CORRECTORS[largest_harmonic_index]

            # generate *NEW* EllipseSample instance with corrected
            # parameter.  Note that this instance is still devoid of other
            # information besides its geometry.  It needs to be explicitly
            # updated for computations to proceed.  We have to build a new
            # EllipseSample instance every time because of the lazy
            # extraction process used by EllipseSample code. To minimize
            # the number of calls to the area integrators, we pay a
            # (hopefully smaller) price here, by having multiple calls to
            # the EllipseSample constructor.
            sample = corrector.correct(sample, largest_harmonic)
            sample.update(fixed_parameters)

            # see if any abnormal (or unusual) conditions warrant
            # the change to non-iterative mode, or go-inwards mode.
            proceed, lexceed = self._check_conditions(
                sample, maxgerr, going_inwards, lexceed)

            if not proceed:
                sample.update(fixed_parameters)
                return Isophote(sample, i + 1, True, -1)

        # Got to the maximum number of iterations. Return with
        # code 2, and handle it as a valid isophote. Use the
        # best fit sample instead of the current one.
        minimum_amplitude_sample.update(fixed_parameters)
        return Isophote(minimum_amplitude_sample, maxit, True, 2)

    @staticmethod
    def _check_conditions(sample, maxgerr, going_inwards, lexceed):
        proceed = True

        # If center wandered more than allowed, put it back
        # in place and signal the end of iterative mode.
        # if wander:
        #     if abs(dx) > WANDER(al)) or abs(dy) > WANDER(al):
        #         sample.geometry.x0 -= dx
        #         sample.geometry.y0 -= dy
        #         STOP(al) = ST_NONITERATE
        #         proceed = False

        # check if an acceptable gradient value could be computed.
        if sample.gradient_error:
            if not going_inwards and (
                    sample.gradient_relative_error > maxgerr
                    or sample.gradient >= 0.0):
                if lexceed:
                    proceed = False
                else:
                    lexceed = True
        else:
            proceed = False

        # check if ellipse geometry diverged.
        if abs(sample.geometry.eps > MAX_EPS):
            proceed = False
        if (sample.geometry.x0 < 1. or
                sample.geometry.x0 > sample.image.shape[1] or
                sample.geometry.y0 < 1. or
                sample.geometry.y0 > sample.image.shape[0]):
            proceed = False

        # See if eps == 0 (round isophote) was crossed.
        # If so, fix it but still proceed
        if sample.geometry.eps < 0.:
            sample.geometry.eps = min(-sample.geometry.eps, MAX_EPS)
            if sample.geometry.pa < PI2:
                sample.geometry.pa += PI2
            else:
                sample.geometry.pa -= PI2

        # If ellipse is an exact circle, computations will diverge.
        # Make it slightly flat, but still proceed
        if sample.geometry.eps == 0.0:
            sample.geometry.eps = MIN_EPS

        return proceed, lexceed


class _ParameterCorrector:

    def correct(self, sample, harmonic):
        raise NotImplementedError


class _PositionCorrector(_ParameterCorrector):

    @staticmethod
    def finalize_correction(dx, dy, sample):
        new_x0 = sample.geometry.x0 + dx
        new_y0 = sample.geometry.y0 + dy

        return EllipseSample(sample.image, sample.geometry.sma, x0=new_x0,
                             y0=new_y0, astep=sample.geometry.astep,
                             sclip=sample.sclip, nclip=sample.nclip,
                             eps=sample.geometry.eps,
                             position_angle=sample.geometry.pa,
                             linear_growth=sample.geometry.linear_growth,
                             integrmode=sample.integrmode)


class _PositionCorrector0(_PositionCorrector):

    def correct(self, sample, harmonic):
        aux = -harmonic * (1. - sample.geometry.eps) / sample.gradient

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

        correction = (harmonic * 2. * (1. - eps) / sma / gradient /
                      ((1. - eps)**2 - 1.))

        # '% np.pi' to make angle lie between 0 and np.pi radians
        new_pa = (sample.geometry.pa + correction) % np.pi

        return EllipseSample(sample.image, sample.geometry.sma,
                             x0=sample.geometry.x0, y0=sample.geometry.y0,
                             astep=sample.geometry.astep, sclip=sample.sclip,
                             nclip=sample.nclip, eps=sample.geometry.eps,
                             position_angle=new_pa,
                             linear_growth=sample.geometry.linear_growth,
                             integrmode=sample.integrmode)


class _EllipticityCorrector(_ParameterCorrector):

    def correct(self, sample, harmonic):
        eps = sample.geometry.eps
        sma = sample.geometry.sma
        gradient = sample.gradient

        correction = harmonic * 2. * (1. - eps) / sma / gradient

        new_eps = min((sample.geometry.eps - correction), MAX_EPS)

        return EllipseSample(sample.image, sample.geometry.sma,
                             x0=sample.geometry.x0, y0=sample.geometry.y0,
                             astep=sample.geometry.astep, sclip=sample.sclip,
                             nclip=sample.nclip, eps=new_eps,
                             position_angle=sample.geometry.pa,
                             linear_growth=sample.geometry.linear_growth,
                             integrmode=sample.integrmode)


# instances of corrector code live here:
_CORRECTORS = [_PositionCorrector0(), _PositionCorrector1(),
               _AngleCorrector(), _EllipticityCorrector()]


class CentralEllipseFitter(EllipseFitter):
    """
    A special Fitter class to handle the case of the central pixel in
    the galaxy image.
    """

    def fit(self, conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
            going_inwards=False):
        """
        Perform just a simple 1-pixel extraction at the current (x0, y0)
        position using bilinear interpolation.

        The input parameters are ignored, but included simple to match
        the calling signature of the parent class.

        Returns
        -------
        result : `~photutils.isophote.CentralEllipsePixel` instance
            The central pixel value.  For convenience, the
            `~photutils.isophote.CentralEllipsePixel` class inherits
            from the `~photutils.isophote.Isophote` class, although it's
            not really a true isophote but just a single intensity value
            at the central position.  Thus, most of its attributes are
            hardcoded to `None` or other default value when appropriate.
        """
        # default values
        fixed_parameters = np.array([False, False, False, False])

        self._sample.update(fixed_parameters)
        return CentralPixel(self._sample)
