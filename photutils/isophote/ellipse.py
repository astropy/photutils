from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from .geometry import Geometry, DEFAULT_STEP, DEFAULT_EPS
from .integrator import BI_LINEAR
from .sample import Sample, CentralSample, DEFAULT_SCLIP
from .fitter import Fitter, CentralFitter, TOO_MANY_FLAGGED, \
    DEFAULT_CONVERGENCY, DEFAULT_MINIT, DEFAULT_MAXIT, DEFAULT_FFLAG, DEFAULT_MAXGERR
from .isophote import Isophote, IsophoteList, print_header
from .centerer import Centerer, DEFAULT_THRESHOLD

__all__ = ['Ellipse']


FIXED_ELLIPSE = 4
FAILED_FIT = 5


class Ellipse(object):
    """
    This class provides the main access point to the isophote fitting algorithm.

    This algorithm is designed to fit elliptical isophotes to galaxy images. Its
    main input is a 2-dimensional numpy array with the image. The output is a
    list of instances of class Isophote. See the documentation for this class
    for details. For convenience, this list is packaged in an instance of class
    IsophoteList, which augments the list with isophote-specific features.

    During the fitting process, some of the isophote parameters are displayed
    in tabular form at the standard output. These parameters allow the user to
    monitor the fitting process.

    The image is measured using an iterative method described by Jedrzejewski
    (Mon.Not.R.Astr.Soc., 226, 747, 1987). Each isophote is fitted at a pre-defined,
    fixed semi-major axis length. The algorithm starts from a first guess elliptical
    isophote defined by approximate values for the X and Y center coordinates,
    ellipticity and position angle. Using these values, the image is sampled along
    an elliptical path, producing a 1-dimensional function that describes the
    dependency of intensity (pixel value) with angle (E). The function is stored
    as a set of 1-D numpy arrays. The harmonic content of this function is analyzed
    by least-squares fitting to the function:

    y  =  y0 + A1 * sin(E) + B1 * cos(E) + A2 * sin(2 * E) + B2 * cos (2 * E)

    Each one of the harmonic amplitudes A1, B1, A2, B2 is related to a specific
    ellipse geometric parameter, in the sense that it conveys information regarding
    how much the parameter's current value deviates from the "true" one. To compute
    this deviation, the image's local radial gradient has to be taken into account
    too. The algorithm picks up the largest amplitude among the four, estimates the
    local gradient and computes the corresponding increment in the associated ellipse
    parameter. That parameter is updated, and the image is resampled. This process
    is repeated until any one of the following criteria are met:

    1 - the largest harmonic amplitude is less than a given fraction of the rms
        residual of the intensity data around the harmonic fit.

    2 - a user-specified maximum number of iterations is reached.

    3 - more than a given fraction of the elliptical sample points have no valid
        data in then, either because they lie outside the image boundaries or
        because they where flagged out from the fit by sigma-clipping.

    In any case, a minimum number of iterations is always performed. If iterations
    stop because of reasons 2 or 3 above, then those ellipse parameters that
    generated the lowest absolute values for harmonic amplitudes will be used. At
    this point, the image data sample coming from the best fit ellipse is fitted
    by the following function:

    y  =  y0  +  An * sin(n * E)  +  Bn * cos(n * E)

    with n = 3 and n = 4. The corresponding amplitudes (A3, B3, A4, B4), divided
    by the semi-major axis length and local intensity gradient, measure the isophote's
    deviations from perfect ellipticity (these amplitudes, divided by semi-major axis
    and gradient, are the actual quantities stored in the output Isophote instance).

    The algorithm then measures the integrated intensity and the number of
    non-flagged pixels inside the elliptical isophote, and also inside the
    corresponding circle with same center and radius equal to the semi-major
    axis length. These parameters, their errors, other associated parameters, and
    auxiliary information, are stored in the Isophote instance.

    Errors in intensity and local gradient are obtained directly from the rms
    scatter of intensity data along the fitted ellipse. Ellipse geometry errors
    are obtained from the errors in the coefficients of the 1st and 2nd simultaneous
    harmonic fit. 3rd and 4th harmonic amplitude errors are obtained in the same
    way, but only after the 1st and 2nd harmonics are subtracted from the raw data.
    See error analysis in Busko, I., 1996, Proceedings of the Fifth Astronomical Data
    Analysis Software and Systems Conference, Tucson, PASP Conference Series v.101,
    ed. G.H. Jacoby and J. Barnes, p.139-142.

    After fitting the ellipse that corresponds to a given value of the semi-major
    axis (by the process described above), the axis length is incremented/decremented
    following a pre-defined rule. At each step, the starting, first guess ellipse
    parameters are taken from the previously fitted ellipse that has the closest
    semi-major axis length to the current one. On low surface brightness regions
    (those having large radii), the small values of the image radial gradient can
    induce large corrections and meaningless values for the ellipse parameters.
    The algorithm has the ability to stop increasing semi-major axis based on several
    criteria, including signal-to-noise ratio.

    See documentation of class Isophote for the meaning of the stop code reported
    after each fit.

    The fit algorithm provides a k-sigma clipping algorithm for cleaning deviant
    sample points at each isophote, thus improving convergency stability against
    any non-elliptical structure such as stars, spiral arms, HII regions, defects, etc.

    The fit algorithm has no way of finding where, in the input image frame, the
    galaxy to be measured sits in. The center X,Y coordinates need to be close to
    the actual center for the fit to work. An "object centerer" function helps to
    verify that the selected position can be used as starting point. This function
    scans a 10 X 10 window centered either on the X,Y coordinates in the Geometry
    instance passed to the constructor of the Ellipse class, or, if any one of them,
    or both, are set to None, on the input image frame center. In case a successful
    acquisition takes place, the Geometry instance is modified in place to reflect
    the solution of the object centerer algorithm.

    In some cases the object centerer algorithm may fail, even though there is enough
    signal-to-noise to start a fit (e.g. in objects with very high ellipticity).
    In those cases the sensitivity of the algorithm can be decreased by decreasing
    the value of the object centerer threshold parameter. The centerer works by
    looking to where a quantity akin to a signal-to-noise ratio is maximized within
    the 10 X 10 window. The centerer can thus be shut off entirely by setting the
    threshold to a large value >> 1 (meaning, no location inside the search window
    will achieve that signal-to-noise ratio).

    A note of caution: the `ellipse` fitting algorithm was designed explicitly
    with a (elliptical) galaxy brightness distribution in mind. In particular, a
    well defined negative radial intensity gradient across the region being fitted
    is paramount for the achievement of stable solutions. Use of the algorithm in
    other types of images (e.g., planetary nebulae) may lead to inability to
    converge to any acceptable solution.

    Parameters
    ----------
    image : np 2-D array
        image array
    geometry : instance of Geometry
        the optional geometry that describes the first ellipse to be fitted.
        If None, a default Geometry instance centered on the image frame and
        with ellipticity 0.2 and position angle 90 deg. is created.
    threshold : float, default = 0.1
        Threshold for the object centerer algorithm. By lowering this value
        the object centerer becomes less strict, in the sense that it will
        accept lower signal-to-noise data. If set to a very large value, the
        centerer is effectively shut off. In this case, either the geometry
        information supplied by the `geometry` parameter is used as is, or the
        fit algorithm will terminate prematurely. Note that, once the object
        centerer runs successfully, the X and Y coordinates in the geometry
        instance are modified in place.
    verbose : boolean, default True
        print object centering info
    """
    def __init__(self, image, geometry=None, threshold=DEFAULT_THRESHOLD, verbose=True):
        self.image = image

        if geometry:
            self._geometry = geometry
        else:
            _x0 = image.shape[0] / 2
            _y0 = image.shape[1] / 2

            self._geometry = Geometry(_x0, _y0, 10., DEFAULT_EPS, np.pi/2)

        # run object centerer
        self._centerer = Centerer(image, self._geometry, verbose)
        self._centerer.center(threshold)

    def set_threshold(self, threshold):
        """
        Modify the threshold value used by the centerer.

        Parameters
        ----------
        threshold : float
            the new threshold value to use
        """
        self._centerer.threshold = threshold

    def fit_image(self, sma0 = 10.,
                          minsma      = 0.,
                          maxsma      = None,
                          step        = DEFAULT_STEP,
                          conver      = DEFAULT_CONVERGENCY,
                          minit       = DEFAULT_MINIT,
                          maxit       = DEFAULT_MAXIT,
                          fflag       = DEFAULT_FFLAG,
                          maxgerr     = DEFAULT_MAXGERR,
                          sclip       = DEFAULT_SCLIP,
                          nclip       = 0,
                          integrmode  = BI_LINEAR,
                          linear      = False,
                          maxrit      = None,
                          verbose     = True):
        # This parameter list is quite large and should in principle be simplified
        # by re-distributing these controls to somewhere else. We keep this design
        # though because it better mimics the flat architecture used in the original
        # STSDAS task `ellipse`.
        """
        Main fitting method. Fits multiple isophotes on the image array passed
        to the constructor. This method basically loops over each one of the
        values of semi-major axis length (sma) constructed from the input parameters,
        and fits one isophote at each sma, returning the entire set of isophotes in
        a sorted IsophoteList instance.

        Parameters
        ----------
        sma0 : float, default = 10.
            starting value for the semi-major axis length (pixels). This can't be
            neither the minimum or the maximum, but something in between. The
            algorithm can't start from the very center of the galaxy image because
            the modelling of elliptical isophotes on that region is poor and it will
            diverge very easily if not tied to other previously fit isophotes. It can't
            start from the maximum value either because the maximum is not known
            beforehand, depending on signal-to-noise. The sma0 value should be selected
            such that the corresponding isophote has a good signal-to-noise ratio and
            a clearly defined geometry.
        minsma : float, default = 0.
            minimum value for the semi-major axis length (pixels).
        maxsma : float, default = None.
            maximum value for the semi-major axis length (pixels).
            When set to None, the algorithm will increase the semi
            major axis until one of several conditions will cause
            it to stop and revert to fit ellipses with sma < sma0.
        step : float, default = 0.1
            the step value being used to grow/shrink the semi-major
            axis length (pixels if `linear=True`, or relative value
            if `linear=False`). See `linear` parameter.
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
            the image frame, or where rejected by sigma-clipping.
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
            non-iterative mode and proceeds outwards until reaching `maxsma`.
        sclip : float, default = 3.0
            sigma-cliping criterion
        nclip : int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        integrmode : string, default = `bi-linear`
            area integration mode, as defined in module integrator.py
        linear : boolean, default False
            semi-major axis growing/shrinking mode. If False, geometric
            growing mode is chosen, thus the semi-major axis length is
            increased by a factor of (1.+`step`), and the process is repeated
            until either the semi-major axis value reaches the value of
            parameter `maxsma`, or the last fitted ellipse has more than a
            given fraction of its sampled points flagged out (see `fflag`).
            The process then resumes from the first fitted ellipse (at `sma0`)
            inwards, in steps of (1./(1.+`step`)), until the semi- major axis
            length reaches the value `minsma`. In case of linear growing, the
            increment or decrement value is given directly by `step` in pixels.
            If `maxsma` is set to None, the semi-major axis will grow until a
            low signal-to-noise criterion is met. See `maxgerr`.
        maxrit : float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophotes wil be just extracted using the current
            geometry, without being fitted. Ignored if None.
            This non-iterative mode may be useful for sampling regions
            of very low surface brightness, where the algorithm may become
            unstable and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically whenever
            the ellipticity exceeds 1.0 or the ellipse center crosses the
            image boundaries.
        verbose : boolean, default True
            print iteration info

        Returns
        -------
        IsophoteList instance
            this list stores fitted Isophote instances, sorted according
            to the semi-major axis length value.
        """
        # multiple fitted isophotes will be stored here
        isophote_list = []

        if verbose:
            print_header()

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        noiter = False
        first_isophote = True
        while True:

            # first isophote runs longer
            minit_a = 2 * minit if first_isophote else minit
            first_isophote = False

            isophote = self.fit_isophote(sma, step, conver, minit_a, maxit, fflag, maxgerr,
                                         sclip, nclip, integrmode,
                                         linear, maxrit, noniterate=noiter,
                                         isophote_list=isophote_list)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == TOO_MANY_FLAGGED:

                # in case the fit failed right at the outset, return an empty
                # list. This is the usual case when the user provides initial
                # guesses that are too way off to enable the fitting algorithm
                # to find any meaningful solution.
                if len(isophote_list) == 1:
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, -1)

                # get last isophote from the actual list, since the last
                # `isophote` instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # if two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if len(isophote_list) > 1:
                    if (isophote.stop_code == FAILED_FIT and isophote_list[-2].stop_code == FAILED_FIT) \
                            or \
                        isophote.stop_code == TOO_MANY_FLAGGED:
                        if maxsma and maxsma > isophote.sma:
                            # if a maximum sma value was provided by user, and the
                            # current sma is smaller than maxsma, keep growing sma
                            # in non-iterative mode until reaching it.
                            noiter = True
                        else:
                            # if no maximum sma, stop growing and change
                            # to go inwards. Print from last kept isophote.
                            if verbose:
                                print(isophote)
                            break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]
            if verbose:
                print(isophote)

            # update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if maxsma and sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # now, go from initial sma inwards towards center.
        while True:
            isophote = self.fit_isophote(sma, step, conver, minit, maxit, fflag, maxgerr,
                                         sclip, nclip,
                                         integrmode, linear, maxrit,
                                         going_inwards=True,
                                         isophote_list=isophote_list)

            # if abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, 0)

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]
            if verbose:
                print(isophote)

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            isophote = self.fit_isophote(0.0, isophote_list=isophote_list)
            if verbose:
                print(isophote)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return IsophoteList(isophote_list)

    def fit_isophote(self, sma,
                            step          = DEFAULT_STEP,
                            conver        = DEFAULT_CONVERGENCY,
                            minit         = DEFAULT_MINIT,
                            maxit         = DEFAULT_MAXIT,
                            fflag         = DEFAULT_FFLAG,
                            maxgerr       = DEFAULT_MAXGERR,
                            sclip         = DEFAULT_SCLIP,
                            nclip         = 0,
                            integrmode    = BI_LINEAR,
                            linear        = False,
                            maxrit        = None,
                            noniterate    = False,
                            going_inwards = False,
                            isophote_list = None):
        """
        Fit one isophote with a given semi-major axis length.

        The `step` and `linear` parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        Parameters
        ----------
        sma : float
            the semi-major axis length (pixels)
        step : float, default = 0.1
            the step value being used to grow/shrink the semi-major
            axis length (pixels)
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
            the image frame, or where rejected by sigma-clipping.
        maxgerr : float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient. When fitting one isophote by itself,
            this parameter doesn't have any effect on the outcome.
        sclip : float, default = 3.0
            sigma-cliping criterion
        nclip : int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        integrmode : string, default = `bi-linear`
            area integration mode, as defined in module integrator.py
        linear : boolean, default = False
            semi-major axis growing/shrinking mode. When fitting just
            one isophote, this parameter is used only by the code that
            defines the details of how elliptical arc segments ("sectors")
            are extracted from the image, when using area extraction modes
            (see parameter `integrmode`)
        maxrit : float, default None
            maximum value of semi-major axis to perform an actual fit.
            If the passed `sma` value is larger than `maxrit`, the
            isophote wil be just extracted using the current geometry,
            without being fitted. Ignored if None.
            This non-iterative mode may be useful for sampling regions
            of very low surface brightness, where the algorithm may become
            unstable and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically whenever
            the ellipticity exceeds 1.0 or the ellipse center crosses the
            image boundaries.
        noniterate : boolean, default False
            signals that the fitting algorithm should be bypassed and an
            isophote should be extracted with the geometry taken directly
            from the most recent Isophote instance stored in the
            `isophote_list` parameter.
            This parameter is mainly used when running the method in a loop
            over different values of semi-major axis length, and we want
            to change from iterative to non-iterative mode somewhere
            along the sequence of isophotes. When set to True, this
            parameter overrides the behavior associated with parameter
            `maxrit`.
        going_inwards : boolean, default False
            defines the sense of SMA growth. When fitting just one isophote,
            this parameter is used only by the code that defines the details
            of how elliptical arc segments ("sectors") are extracted from
            the image, when using area extraction modes (see parameter
            `integrmode`)
        isophote_list : list, default = None
            fitted Isophote instance is appended to this list. Must
            be created and managed by the caller.

        Returns
        -------
        Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to the input list passed via parameter `isophote_list`.
        """
        geometry = self._geometry

        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        if isophote_list is not None and len(isophote_list) > 0:
            geometry = isophote_list[-1].sample.geometry

        # do the fit.
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry,
                                           sclip, nclip)
        else:
            isophote = self._iterative(sma, step, linear, geometry, sclip, nclip,
                                       integrmode, conver, minit, maxit, fflag,
                                       maxgerr, going_inwards)

        # store result in list
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def _iterative(self, sma, step, linear, geometry, sclip, nclip, integrmode,
                   conver, minit, maxit, fflag, maxgerr, going_inwards=False):
        if sma > 0.:
            # iterative fitter
            sample = Sample(self.image, sma,
                            astep=step,
                            sclip=sclip,
                            nclip=nclip,
                            linear_growth=linear,
                            geometry=geometry,
                            integrmode=integrmode)
            fitter = Fitter(sample)
        else:
            # sma == 0 requires special handling.
            sample = CentralSample(self.image, 0.0, geometry=geometry)
            fitter = CentralFitter(sample)

        isophote = fitter.fit(conver, minit, maxit, fflag, maxgerr, going_inwards)

        return isophote

    def _non_iterative(self, sma, step, linear, geometry, sclip, nclip):
        sample = Sample(self.image, sma,
                        astep=step,
                        sclip=sclip,
                        nclip=nclip,
                        linear_growth=linear,
                        geometry=geometry)
        sample.update()

        # build isophote without iterating with a Fitter
        isophote = Isophote(sample, 0, True, FIXED_ELLIPSE)

        return isophote

    def _fix_last_isophote(self, isophote_list, index):
        if len(isophote_list) > 0:
            isophote = isophote_list.pop()

            # check if isophote is bad; if so, fix its geometry
            # to be like the geometry of the index-th isophote
            # in list.
            isophote.fix_geometry(isophote_list[index])

            # force new extraction of raw data, since
            # geometry changed.
            isophote.sample.values = None
            isophote.sample.update()

            # we take the opportunity to change an eventual
            # negative stop code to its' positive equivalent.
            code = FAILED_FIT if isophote.stop_code < 0 else isophote.stop_code

            # build new instance so it can have its attributes
            # populated from the updated sample attributes.
            new_isophote = Isophote(isophote.sample, isophote.niter, isophote.valid, code)

            # add new isophote to list
            isophote_list.append(new_isophote)


