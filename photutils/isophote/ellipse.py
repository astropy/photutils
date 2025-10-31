# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define a class to fit elliptical isophotes.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.fitter import (DEFAULT_CONVERGENCE, DEFAULT_FFLAG,
                                       DEFAULT_MAXGERR, DEFAULT_MAXIT,
                                       DEFAULT_MINIT, CentralEllipseFitter,
                                       EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import BILINEAR
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import CentralEllipseSample, EllipseSample

__all__ = ['Ellipse']


class Ellipse:
    """
    Class to fit elliptical isophotes to a galaxy image.

    The isophotes in the image are measured using an iterative method
    described by `Jedrzejewski (1987; MNRAS  226, 747)
    <https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
    See the **Notes** section below for details about the algorithm.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image array.
    geometry : `~photutils.isophote.EllipseGeometry` instance or `None`, \
            optional
        The optional geometry that describes the first
        ellipse to be fitted. If `None`, a default
        `~photutils.isophote.EllipseGeometry` instance is created
        centered on the image frame with ellipticity of 0.2 and a
        position angle of 90 degrees.

    threshold : float, optional
        The threshold for the object centerer algorithm. By lowering
        this value the object centerer becomes less strict, in the
        sense that it will accept lower signal-to-noise data. If set
        to a very large value, the centerer is effectively shut off.
        In this case, either the geometry information supplied by the
        ``geometry`` parameter is used as is, or the fit algorithm
        will terminate prematurely. Note that once the object centerer
        runs successfully, the (x, y) coordinates in the ``geometry``
        attribute (an `~photutils.isophote.EllipseGeometry` instance)
        are modified in place. The default is 0.1.

    Notes
    -----
    The image is measured using an iterative method
    described by `Jedrzejewski (1987; MNRAS 226, 747)
    <https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
    Each isophote is fitted at a predefined, fixed semimajor axis
    length. The algorithm starts from a first-guess elliptical isophote
    defined by approximate values for the (x, y) center coordinates,
    ellipticity, and position angle. Using these values, the image
    is sampled along an elliptical path, producing a 1-dimensional
    function that describes the dependence of intensity (pixel value)
    with angle (E). The function is stored as a set of 1D numpy arrays.
    The harmonic content of this function is analyzed by least-squares
    fitting to the function:

    .. math::

        y  =  y0 + (A1 * \\sin(E)) + (B1 * \\cos(E)) + (A2 * \\sin(2 * E))
              + (B2 * \\cos(2 * E))

    Each one of the harmonic amplitudes (A1, B1, A2, and B2) is related
    to a specific ellipse geometric parameter in the sense that it
    conveys information regarding how much the parameter's current
    value deviates from the "true" one. To compute this deviation, the
    image's local radial gradient has to be taken into account too. The
    algorithm picks up the largest amplitude among the four, estimates
    the local gradient, and computes the corresponding increment in the
    associated ellipse parameter. That parameter is updated, and the
    image is resampled. This process is repeated until any one of the
    following criteria are met:

    1. the largest harmonic amplitude is less than a given fraction of
       the rms residual of the intensity data around the harmonic fit.

    2. a user-specified maximum number of iterations is reached.

    3. more than a given fraction of the elliptical sample points have no
       valid data in them, either because they lie outside the image
       boundaries or because they were flagged out from the fit by
       sigma-clipping.

    In any case, a minimum number of iterations is always performed. If
    iterations stop because of reasons 2 or 3 above, then those ellipse
    parameters that generated the lowest absolute values for harmonic
    amplitudes will be used. At this point, the image data sample coming
    from the best fit ellipse is fitted by the following function:

    .. math::

        y  =  y0 + (An * sin(n * E)) + (Bn * cos(n * E))

    with :math:`n = 3` and :math:`n = 4`. The corresponding amplitudes
    (A3, B3, A4, and B4), divided by the semimajor axis length and
    local intensity gradient, measure the isophote's deviations from
    perfect ellipticity (these amplitudes, divided by semimajor axis
    and gradient, are the actual quantities stored in the output
    `~photutils.isophote.Isophote` instance).

    The algorithm then measures the integrated intensity and the number
    of non-flagged pixels inside the elliptical isophote, and also
    inside the corresponding circle with same center and radius equal
    to the semimajor axis length. These parameters, their errors, other
    associated parameters, and auxiliary information, are stored in the
    `~photutils.isophote.Isophote` instance.

    Errors in intensity and local gradient are obtained directly
    from the rms scatter of intensity data along the fitted
    ellipse. Ellipse geometry errors are obtained from the errors
    in the coefficients of the first and second simultaneous
    harmonic fit. Third and fourth harmonic amplitude errors
    are obtained in the same way, but only after the first and
    second harmonics are subtracted from the raw data. For more
    details, see the error analysis in `Busko (1996; ASPC 101, 139)
    <https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_.

    After fitting the ellipse that corresponds to a given value of the
    semimajor axis (by the process described above), the axis length
    is incremented/decremented following a predefined rule. At each
    step, the starting, first-guess, ellipse parameters are taken
    from the previously fitted ellipse that has the closest semimajor
    axis length to the current one. On low surface brightness regions
    (those having large radii), the small values of the image radial
    gradient can induce large corrections and meaningless values for the
    ellipse parameters. The algorithm has the ability to stop increasing
    semimajor axis based on several criteria, including signal-to-noise
    ratio.

    See the `~photutils.isophote.Isophote` documentation for the meaning
    of the stop code reported after each fit.

    The fit algorithm provides a k-sigma clipping algorithm for cleaning
    deviant sample points at each isophote, thus improving convergence
    stability against any non-elliptical structure such as stars, spiral
    arms, HII regions, defects, etc.

    The fit algorithm has no way of finding where, in the input image
    frame, the galaxy to be measured is located. The center (x, y)
    coordinates need to be close to the actual center for the fit
    to work. An "object centerer" function helps to verify that the
    selected position can be used as starting point. This function
    scans a 10x10 window centered either on the (x, y) coordinates in
    the `~photutils.isophote.EllipseGeometry` instance passed to the
    constructor of the `~photutils.isophote.Ellipse` class, or, if
    any one of them, or both, are set to `None`, on the input image
    frame center. In case a successful acquisition takes place, the
    `~photutils.isophote.EllipseGeometry` instance is modified in place
    to reflect the solution of the object centerer algorithm.

    In some cases the object centerer algorithm may fail, even though
    there is enough signal-to-noise to start a fit (e.g., in objects
    with very high ellipticity). In those cases the sensitivity of the
    algorithm can be decreased by decreasing the value of the object
    centerer threshold parameter. The centerer works by looking to where
    a quantity akin to a signal-to-noise ratio is maximized within the
    10x10 window. The centerer can thus be shut off entirely by setting
    the threshold to a large value >> 1 (meaning, no location inside the
    search window will achieve that signal-to-noise ratio).

    A note of caution: the ellipse fitting algorithm was designed
    explicitly with an elliptical galaxy brightness distribution in
    mind. In particular, a well-defined negative radial intensity
    gradient across the region being fitted is paramount for the
    achievement of stable solutions. Use of the algorithm in other
    types of images (e.g., planetary nebulae) may lead to inability to
    converge to any acceptable solution.
    """

    def __init__(self, image, geometry=None, threshold=0.1):
        self.image = image

        if geometry is not None:
            self._geometry = geometry
        else:
            _x0 = image.shape[1] / 2
            _y0 = image.shape[0] / 2
            self._geometry = EllipseGeometry(_x0, _y0, 10.0, eps=0.2,
                                             pa=np.pi / 2)
        self.set_threshold(threshold)

    def set_threshold(self, threshold):
        """
        Modify the threshold value used by the centerer.

        Parameters
        ----------
        threshold : float
            The new threshold value to use.
        """
        self._geometry.centerer_threshold = threshold

    def fit_image(self, sma0=None, minsma=0.0, maxsma=None, step=0.1,
                  conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
                  maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
                  maxgerr=DEFAULT_MAXGERR, sclip=3.0, nclip=0,
                  integrmode=BILINEAR, linear=None, maxrit=None,
                  fix_center=False, fix_pa=False, fix_eps=False):
        # This parameter list is quite large and should in principle be
        # simplified by redistributing these controls to somewhere else.
        # We keep this design though because it better mimics the flat
        # architecture used in the original STSDAS task `ellipse`.
        """
        Fit multiple isophotes to the image array.

        This method loops over each value of the semimajor axis (sma)
        length (constructed from the input parameters), fitting a single
        isophote at each sma. The entire set of isophotes is returned in
        an `~photutils.isophote.IsophoteList` instance.

        Note that the fix_XXX parameters act in unison. Meaning,
        if one of them is set via this call, the others will
        assume their default (False) values. This effectively
        overrides any settings that are present in the internal
        `~photutils.isophote.EllipseGeometry` instance that is
        carried along as a property of this class. If an instance of
        `~photutils.isophote.EllipseGeometry` was passed to this class'
        constructor, that instance will be effectively overridden by the
        fix_XXX parameters in this call.

        Parameters
        ----------
        sma0 : float, optional
            The starting value for the semimajor axis length (pixels).
            This value must not be the minimum or maximum semimajor
            axis length, but something in between. The algorithm can't
            start from the very center of the galaxy image because
            the modelling of elliptical isophotes on that region is
            poor and it will diverge very easily if not tied to other
            previously fit isophotes. It can't start from the maximum
            value either because the maximum is not known beforehand,
            depending on signal-to-noise. The ``sma0`` value should be
            selected such that the corresponding isophote has a good
            signal-to-noise ratio and a clearly defined geometry. If set
            to `None` (the default), one of two actions will be taken:
            if a `~photutils.isophote.EllipseGeometry` instance was
            input to the `~photutils.isophote.Ellipse` constructor, its
            ``sma`` value will be used. Otherwise, a default value of
            10. will be used.

        minsma : float, optional
            The minimum value for the semimajor axis length (pixels).
            The default is 0.

        maxsma : float or `None`, optional
            The maximum value for the semimajor axis length (pixels).
            When set to `None` (default), the algorithm will increase
            the semimajor axis until one of several conditions will
            cause it to stop and revert to fit ellipses with sma <
            ``sma0``.

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear=True``, or a relative value if
            ``linear=False``). See the ``linear`` parameter. The default
            is 0.1.

        conver : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``conver`` times the harmonic fit rms. The
            default is 0.05.

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
            frame, are masked, or were rejected by sigma-clipping. The
            default is 0.7.

        maxgerr : float, optional
            The maximum acceptable relative error in the local
            radial intensity gradient. This is the main control
            for preventing ellipses to grow to regions of too
            low signal-to-noise ratio. It specifies the maximum
            acceptable relative error in the local radial
            intensity gradient. `Busko (1996; ASPC 101, 139)
            <https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstr
            act>`_ showed that the fitting precision relates to that
            relative error. The usual behavior of the gradient relative
            error is to increase with semimajor axis, being larger in
            outer, fainter regions of a galaxy image. In the current
            implementation, the ``maxgerr`` criterion is triggered only
            when two consecutive isophotes exceed the value specified by
            the parameter. This prevents premature stopping caused by
            contamination such as stars and HII regions.

            A number of actions may happen when the gradient error
            exceeds ``maxgerr`` (or becomes non-significant and is
            set to `None`). If the maximum semimajor axis specified
            by ``maxsma`` is set to `None`, semimajor axis growth is
            stopped and the algorithm proceeds inwards to the galaxy
            center. If ``maxsma`` is set to some finite value, and this
            value is larger than the current semimajor axis length, the
            algorithm enters non-iterative mode and proceeds outwards
            until reaching ``maxsma``. The default is 0.5.

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        nclip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

        integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, \
                optional
            The area integration mode. The default is 'bilinear'.

        linear : bool, optional
            The semimajor axis growing/shrinking mode. If `False`
            (default), the geometric growing mode is chosen, thus the
            semimajor axis length is increased by a factor of (1.
            + ``step``), and the process is repeated until either
            the semimajor axis value reaches the value of parameter
            ``maxsma``, or the last fitted ellipse has more than a given
            fraction of its sampled points flagged out (see ``fflag``).
            The process then resumes from the first fitted ellipse (at
            ``sma0``) inwards, in steps of (1./(1. + ``step``)), until
            the semimajor axis length reaches the value ``minsma``. In
            case of linear growing, the increment or decrement value
            is given directly by ``step`` in pixels. If ``maxsma`` is
            set to `None`, the semimajor axis will grow until a low
            signal-to-noise criterion is met. See ``maxgerr``.

        maxrit : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``maxrit``, the isophotes will be extracted using the
            current geometry, without being fitted. This non-iterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically
            whenever the ellipticity exceeds 1.0 or the ellipse center
            crosses the image boundaries. If `None` (default), then no
            maximum value is used.

        fix_center : bool, optional
            Keep center of ellipse fixed during fit? The default is
            False.

        fix_pa : bool, optional
            Keep position angle of semi-major axis of ellipse fixed
            during fit? The default is False.

        fix_eps : bool, optional
            Keep ellipticity of ellipse fixed during fit? The default is
            False.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList` instance
            A list-like object of `~photutils.isophote.Isophote`
            instances, sorted by increasing semimajor axis length.
        """
        # multiple fitted isophotes will be stored here
        isophote_list = []

        # get starting sma from appropriate source: keyword parameter,
        # internal EllipseGeometry instance, or fixed default value.
        if not sma0:
            sma = self._geometry.sma if self._geometry else 10.0
        else:
            sma = sma0

        # Override geometry instance with parameters set at the call.
        if isinstance(linear, bool):
            self._geometry.linear_growth = linear
        else:
            linear = self._geometry.linear_growth
        if fix_center and fix_pa and fix_eps:
            warnings.warn(': Everything is fixed. Fit not possible.',
                          AstropyUserWarning)
            return IsophoteList([])
        if fix_center or fix_pa or fix_eps:
            # Note that this overrides the geometry instance for good.
            self._geometry.fix = np.array([fix_center, fix_center, fix_pa,
                                           fix_eps])

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        noiter = False
        first_isophote = True
        while True:
            # first isophote runs longer
            minit_a = 2 * minit if first_isophote else minit
            first_isophote = False

            isophote = self.fit_isophote(sma, step, conver, minit_a, maxit,
                                         fflag, maxgerr, sclip, nclip,
                                         integrmode, linear, maxrit,
                                         noniterate=noiter,
                                         isophote_list=isophote_list)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == 1:
                # in case the fit failed right at the outset, return an
                # empty list. This is the usual case when the user
                # provides initial guesses that are too way off to enable
                # the fitting algorithm to find any meaningful solution.

                if len(isophote_list) == 1:
                    warnings.warn('No meaningful fit was possible.',
                                  AstropyUserWarning)
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, -1)

                # get last isophote from the actual list, since the last
                # `isophote` instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # if two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if (len(isophote_list) > 2
                    and ((isophote.stop_code == 5
                          and isophote_list[-2].stop_code == 5)
                         or isophote.stop_code == 1)):
                    if maxsma and maxsma > isophote.sma:
                        # if a maximum sma value was provided by
                        # user, and the current sma is smaller than
                        # maxsma, keep growing sma in non-iterative
                        # mode until reaching it.
                        noiter = True
                    else:
                        # if no maximum sma, stop growing and change
                        # to go inwards.
                        break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

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
            isophote = self.fit_isophote(sma, step, conver, minit, maxit,
                                         fflag, maxgerr, sclip, nclip,
                                         integrmode, linear, maxrit,
                                         going_inwards=True,
                                         isophote_list=isophote_list)

            # if abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, 0)

            # but if we get an error from the scipy fitter, bail out
            # immediately. This usually happens at very small radii
            # when the number of data points is too small.
            if isophote.stop_code == 3:
                break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            # isophote is appended to isophote_list
            _ = self.fit_isophote(0.0, isophote_list=isophote_list)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return IsophoteList(isophote_list)

    def fit_isophote(self, sma, step=0.1, conver=DEFAULT_CONVERGENCE,
                     minit=DEFAULT_MINIT, maxit=DEFAULT_MAXIT,
                     fflag=DEFAULT_FFLAG, maxgerr=DEFAULT_MAXGERR,
                     sclip=3.0, nclip=0, integrmode=BILINEAR,
                     linear=False, maxrit=None, noniterate=False,
                     going_inwards=False, isophote_list=None):
        """
        Fit a single isophote with a given semimajor axis length.

        The ``step`` and ``linear`` parameters do not directly control
        the growth or reduction of the current fitting semimajor axis
        length. Instead, they are used by the sampling algorithm to
        determine the starting point for gradient computation and
        to calculate the areas of the elliptical sectors (when area
        integration mode is enabled).

        Parameters
        ----------
        sma : float
            The semimajor axis length (pixels).

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear=True``, or a relative value if
            ``linear=False``). See the ``linear`` parameter. The default
            is 0.1.

        conver : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``conver`` times the harmonic fit rms. The
            default is 0.05.

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
            frame, are masked, or were rejected by sigma-clipping. The
            default is 0.7.

        maxgerr : float, optional
            The maximum acceptable relative error in the local radial
            intensity gradient. When fitting a single isophote by itself
            this parameter doesn't have any effect on the outcome.

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        nclip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

        integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, \
                optional
            The area integration mode. The default is 'bilinear'.

        linear : bool, optional
            The semimajor axis growing/shrinking mode. When fitting
            just one isophote, this parameter is used only by the code
            that define the details of how elliptical arc segments
            ("sectors") are extracted from the image when using area
            extraction modes (see the ``integrmode`` parameter).

        maxrit : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``maxrit``, the isophotes will be extracted using the
            current geometry, without being fitted. This non-iterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically
            whenever the ellipticity exceeds 1.0 or the ellipse center
            crosses the image boundaries. If `None` (default), then no
            maximum value is used.

        noniterate : bool, optional
            Whether the fitting algorithm should be bypassed and an
            isophote should be extracted with the geometry taken
            directly from the most recent `~photutils.isophote.Isophote`
            instance stored in the ``isophote_list`` parameter. This
            parameter is mainly used when running the method in a loop
            over different values of semimajor axis length, and we want
            to change from iterative to non-iterative mode somewhere
            along the sequence of isophotes. When set to `True`, this
            parameter overrides the behavior associated with parameter
            ``maxrit``. The default is `False`.

        going_inwards : bool, optional
            Parameter to define the sense of SMA growth. When fitting
            just one isophote, this parameter is used only by the code
            that defines the details of how elliptical arc segments
            ("sectors") are extracted from the image, when using area
            extraction modes (see the ``integrmode`` parameter). The
            default is `False`.

        isophote_list : list or `None`, optional
            If not `None` (the default), the fitted
            `~photutils.isophote.Isophote` instance is appended to this
            list. It must be created and managed by the caller.

        Returns
        -------
        result : `~photutils.isophote.Isophote` instance
            The fitted isophote. The fitted isophote is also appended to
            the input list input to the ``isophote_list`` parameter.
        """
        geometry = self._geometry

        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        if isophote_list:
            geometry = isophote_list[-1].sample.geometry

        # do the fit
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry,
                                           sclip, nclip, integrmode)
        else:
            isophote = self._iterative(sma, step, linear, geometry, sclip,
                                       nclip, integrmode, conver, minit,
                                       maxit, fflag, maxgerr, going_inwards)

        # store result in list
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def _iterative(self, sma, step, linear, geometry, sclip, nclip,
                   integrmode, conver, minit, maxit, fflag, maxgerr,
                   going_inwards=False):
        if sma > 0.0:
            # iterative fitter
            sample = EllipseSample(self.image, sma, astep=step, sclip=sclip,
                                   nclip=nclip, linear_growth=linear,
                                   geometry=geometry, integrmode=integrmode)
            fitter = EllipseFitter(sample)
        else:
            # sma == 0 requires special handling
            sample = CentralEllipseSample(self.image, 0.0, geometry=geometry)
            fitter = CentralEllipseFitter(sample)

        return fitter.fit(conver=conver, minit=minit, maxit=maxit,
                          fflag=fflag, maxgerr=maxgerr,
                          going_inwards=going_inwards)

    def _non_iterative(self, sma, step, linear, geometry, sclip, nclip,
                       integrmode):
        sample = EllipseSample(self.image, sma, astep=step, sclip=sclip,
                               nclip=nclip, linear_growth=linear,
                               geometry=geometry, integrmode=integrmode)
        sample.update(geometry.fix)

        # build isophote without iterating with an EllipseFitter
        return Isophote(sample, 0, valid=True, stop_code=4)

    @staticmethod
    def _fix_last_isophote(isophote_list, index):
        if isophote_list:
            isophote = isophote_list.pop()

            # check if isophote is bad; if so, fix its geometry
            # to be like the geometry of the index-th isophote
            # in list.
            isophote.fix_geometry(isophote_list[index])

            # force new extraction of raw data, since
            # geometry changed.
            isophote.sample.values = None
            isophote.sample.update(isophote.sample.geometry.fix)

            # we take the opportunity to change an eventual
            # negative stop code to its' positive equivalent.
            code = 5 if isophote.stop_code < 0 else isophote.stop_code

            # build new instance so it can have its attributes
            # populated from the updated sample attributes.
            new_isophote = Isophote(isophote.sample, isophote.niter,
                                    isophote.valid, code)

            # add new isophote to list
            isophote_list.append(new_isophote)
