# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import numpy as np

from .geometry import normalize_angle
from .harmonics import (fit_1st_and_2nd_harmonics,
                        first_and_2nd_harmonic_function)
from .isophote import Isophote, CentralPixel
from .sample import Sample


__all__ = ['Fitter']


PI2 = np.pi / 2
MAX_EPS = 0.95
MIN_EPS = 0.05
TOO_MANY_FLAGGED = 1
NORMAL_FIT = 0

DEFAULT_CONVERGENCY = 0.05
DEFAULT_MINIT = 10
DEFAULT_MAXIT = 50
DEFAULT_FFLAG = 0.7
DEFAULT_MAXGERR = 0.5


class Fitter(object):
    """
    The main fitter class.

    Parameters
    ----------
    sample : instance of Sample
        the sample to be fitted
    """

    def __init__(self, sample):
        self._sample = sample

    def fit(self, conver=DEFAULT_CONVERGENCY, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
            going_inwards=False):
        """
        Perform the actual fit, returning an Isophote instance:

            fitter = Fitter(sample)
            isophote = fitter.fit()

        Parameters
        ----------
        conver : float, default = 0.05
            main convergency criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than `conver` times the harmonic fit rms.
        minit : int, default = 10
            minimum number of iterations to perform. A minimum of 10
            iterations guarantees that, on average, 2 iterations will
            be available for fitting each independent parameter (the
            four harmonic amplitudes and the intensity level). In the
            first isophote, the minimum number of iterations is 2 * `minit`,
            to ensure that, even departing from not-so-good initial values,
            the algorithm has a better chance to converge to a sensible
            solution.
        maxit : int, default = 50
            maximum number of iterations to perform
        fflag : float, default = 0.7
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            Flagged data points are points that either lie outside
            the image frame, are masked, or were rejected by
            sigma-clipping.
        maxgerr : float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient. This is the main control for preventing
            ellipses to grow to regions of too low signal-to-noise ratio.
            It specifies the maximum acceptable relative error in the
            local radial intensity gradient. Experiments (see paper
            [2] quoted in the FAQ) showed that the fitting precision
            relates to that relative error. The usual behavior of the
            gradient relative error is to increase with semi-major axis,
            being larger in outer, fainter regions of a galaxy image.
            In the current implementation, the `maxgerr` criterion is
            triggered only when two consecutive isophotes exceed the
            value specified in the parameter. This prevents premature
            stopping caused by contamination such as stars and HII
            regions.
            A number of actions may happen when the current gradient
            error exceeds `maxgerr` (or becomes non-significant and is
            set to None) in the process of increasing semi-major axis
            length. If the maximum semi-major axis specified by parameter
            `maxsma` is set to None, semi-major axis grow is stopped and
            the algorithm proceeds inwards to the galaxy image center. If
            `maxsma` is set to some finite value, and this value is larger
            than the current semi-major axis length, the algorithm enters
            non-iterative mode and proceeds outwards until reaching `.maxsma`.
        going_inwards : boolean, default = False
            defines the sense of SMA growth. This is used by the Ellipse
            class for defining stopping criteria that depend on the gradient
            relative error. When fitting just one isophote, this parameter
            is used only by the code that defines the details of how
            elliptical arc segments ("sectors") are extracted from the image,
            when using area extraction modes (see parameter `integrmode` in
            the Sample class).

        Returns
        -------
        instance of Isophote
            isophote with the fitted sample plus additional fit status
            information
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

        for iter in range(maxit):
            # Force the sample to compute its gradient and associated values.
            sample.update()

            # The extract() method returns sampled values as a 2-d numpy array
            # with the following structure:
            # values[0] = 1-d array with angles
            # values[1] = 1-d array with radii
            # values[2] = 1-d array with intensity
            values = sample.extract()

            # Fit harmonic coefficients. Failure in fitting is
            # a fatal error; terminate immediately with sample
            # marked as invalid.
            try:
                coeffs = fit_1st_and_2nd_harmonics(values[0], values[2])
            except Exception as e:
                print(e)
                return Isophote(sample, iter+1, False, 3)

            coeffs = coeffs[0]

            # largest harmonic in absolute value drives the correction.
            largest_harmonic_index = np.argmax(np.abs(coeffs[1:]))
            largest_harmonic = coeffs[1:][largest_harmonic_index]

            # see if the amplitude decreased; if yes, keep the
            # corresponding sample for eventual later use.
            if abs(largest_harmonic) < minimum_amplitude_value:
                minimum_amplitude_value = abs(largest_harmonic)
                minimum_amplitude_sample = sample

            # check if converged
            model = first_and_2nd_harmonic_function(values[0], coeffs)
            residual = values[2] - model

            if ((conver * sample.sector_area * np.std(residual))
                    > np.abs(largest_harmonic)):
                # Got a valid solution. But before returning, ensure
                # that a minimum of iterations has run.
                if iter >= minit-1:
                    sample.update()
                    return Isophote(sample, iter+1, True, NORMAL_FIT)

            # it may not have converged yet, but the sample contains too
            # many invalid data points: return.
            if sample.actual_points < (sample.total_points * fflag):
                # when too many data points were flagged, return the
                # best fit sample instead of the current one.
                minimum_amplitude_sample.update()
                return Isophote(minimum_amplitude_sample, iter+1, True,
                                TOO_MANY_FLAGGED)

            # pick appropriate corrector code.
            corrector = _correctors[largest_harmonic_index]

            # generate *NEW* Sample instance with corrected parameter.
            # Note that this instance is still devoid of other information
            # besides its geometry.  It needs to be explicitly updated for
            # computations to proceed.  We have to build a new Sample
            # instance every time because of the lazy extraction process
            # used by Sample code. To minimize the number of calls to the
            # area integrators, we pay a (hopefully smaller) price here,
            # by having multiple calls to the Sample constructor.
            sample = corrector.correct(sample, largest_harmonic)
            sample.update()

            # see if any abnormal (or unusual) conditions warrant
            # the change to non-iterative mode, or go inwards mode.
            good_to_go, lexceed = self._is_good_to_go(sample, maxgerr,
                                                      going_inwards, lexceed)
            if not good_to_go:
                sample.update()
                return Isophote(sample, iter+1, True, -1)

        # Got to the maximum number of iterations. Return with
        # code 2, and handle it as a valid isophote. Use the
        # best fit sample instead of the current one.
        minimum_amplitude_sample.update()
        return Isophote(minimum_amplitude_sample, maxit, True, 2)

    def _is_good_to_go(self, sample, maxgerr, going_inwards, lexceed):
        good_to_go = True

        # If center wandered more than allowed, put it back
        # in place and signal the end of iterative mode.
        # if wander:
        #     if abs(dx) > WANDER(al)) or abs(dy) > WANDER(al):
        #         sample.geometry.x0 -= dx
        #         sample.geometry.y0 -= dy
        #         STOP(al) = ST_NONITERATE
        #         good_to_go = False

        # check if an acceptable gradient value could be computed.
        if sample.gradient_error:
            if (not going_inwards and
                (sample.gradient_relative_error > maxgerr or
                 sample.gradient >= 0.)):
                if lexceed:
                    good_to_go = False
                else:
                    lexceed = True
        else:
            good_to_go = False

        # check if ellipse geometry diverged.
        if abs(sample.geometry.eps > MAX_EPS):
            good_to_go = False
        if (sample.geometry.x0 < 1. or
                sample.geometry.x0 > sample.image.shape[0] or
                sample.geometry.y0 < 1. or
                sample.geometry.y0 > sample.image.shape[1]):
            good_to_go = False

        # See if eps == 0 (round isophote) was crossed.
        # If so, fix it but still good_to_go to go.
        if sample.geometry.eps < 0.:
            sample.geometry.eps = min(-sample.geometry.eps, MAX_EPS)
            if sample.geometry.pa < PI2:
                sample.geometry.pa += PI2
            else:
                sample.geometry.pa -= PI2

        # If ellipse is an exact circle, computations will diverge.
        # Make it slightly flat, but still good to go.
        if sample.geometry.eps == 0.0:
            sample.geometry.eps = MIN_EPS

        return good_to_go, lexceed


class _ParameterCorrector(object):

    def correct(self, sample, harmonic):
        raise NotImplementedError


class _PositionCorrector(_ParameterCorrector):

    def finalize_correction(self, dx, dy, sample):
        new_x0 = sample.geometry.x0 + dx
        new_y0 = sample.geometry.y0 + dy

        return Sample(sample.image, sample.geometry.sma, x0=new_x0, y0=new_y0,
                      astep=sample.geometry.astep, sclip=sample.sclip,
                      nclip=sample.nclip, eps=sample.geometry.eps,
                      position_angle=sample.geometry.pa,
                      linear_growth=sample.geometry.linear_growth,
                      integrmode=sample.integrmode)


class _PositionCorrector_0(_PositionCorrector):

    def correct(self, sample, harmonic):
        aux = -harmonic * (1. - sample.geometry.eps) / sample.gradient

        dx = -aux * math.sin(sample.geometry.pa)
        dy = aux * math.cos(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class _PositionCorrector_1(_PositionCorrector):

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

        new_pa = normalize_angle(sample.geometry.pa + correction)

        return Sample(sample.image, sample.geometry.sma,
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

        return Sample(sample.image, sample.geometry.sma,
                      x0=sample.geometry.x0, y0=sample.geometry.y0,
                      astep=sample.geometry.astep, sclip=sample.sclip,
                      nclip=sample.nclip, eps=new_eps,
                      position_angle=sample.geometry.pa,
                      linear_growth=sample.geometry.linear_growth,
                      integrmode=sample.integrmode)


# instances of corrector code live here:
_correctors = [_PositionCorrector_0(), _PositionCorrector_1(),
               _AngleCorrector(), _EllipticityCorrector()]


class CentralFitter(Fitter):
    """
    Derived Fitter class, designed specifically to handle the
    case of the central pixel in the galaxy image.
    """

    def fit(self, conver=DEFAULT_CONVERGENCY, minit=DEFAULT_MINIT,
            maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
            going_inwards=False):
        """
        Overrides the base class to perform just a simple 1-pixel
        extraction at the current x0,y0 position, using bilinear
        interpolation.

        Parameters are ignored. They were added just so the method's
        signature matches the superclass' and my IDE doesn't complain.

        :return: instance of the CentralPixel class.
            For convenience, the CentralPixel class inherits from
            the Isophote class, although it's not really a true
            isophote but just a single intensity value at the central
            position. Thus, most of its attributes are hardcoded to
            None, or other default value when appropriate.
        """

        self._sample.update()
        return CentralPixel(self._sample)
