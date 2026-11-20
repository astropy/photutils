# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for fitting elliptical isophotes.
"""

import copy
import functools
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.fitter import (DEFAULT_CONVERGENCE, DEFAULT_FFLAG,
                                       DEFAULT_MAXGERR, DEFAULT_MAXIT,
                                       DEFAULT_MINIT, CentralEllipseFitter,
                                       EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry, _make_fix_array
from photutils.isophote.integrator import BILINEAR
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import (CentralEllipseSample, EllipseSample,
                                       _validate_image)
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

__all__ = ['Ellipse']


class Ellipse:
    """
    Class to fit elliptical isophotes to a galaxy image.

    The isophotes in the image are measured using an iterative method
    described by `Jedrzejewski (1987, MNRAS 226, 747)
    <https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
    See the **Notes** section below for details about the algorithm.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image array. Masked pixels (if ``image`` is a
        `~numpy.ma.MaskedArray`) and nonfinite (NaN or infinity)
        values are excluded from the fits.
    geometry : `~photutils.isophote.EllipseGeometry` instance or `None`, \
            optional
        The optional geometry that describes the first
        ellipse to be fitted. If `None`, a default
        `~photutils.isophote.EllipseGeometry` instance is created
        centered on the image frame with ellipticity of 0.2 and a
        position angle of 90 degrees.

    threshold : float, optional
        This keyword has no effect.

        .. deprecated:: 3.2
           This keyword argument is deprecated and will be removed in
           version 4.0. The object centerer is not run by this class.
           Use :meth:`~photutils.isophote.EllipseGeometry.find_center`
           to refine the initial center before creating this class.

    Notes
    -----
    The image is measured using an iterative method
    described by `Jedrzejewski (1987, MNRAS 226, 747)
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
    from the best-fit ellipse is fitted by the following function:

    .. math::

        y  =  y0 + (An * sin(n * E)) + (Bn * cos(n * E))

    with :math:`n = 3` and :math:`n = 4`. The corresponding amplitudes
    (A3, B3, A4, and B4), divided by the semimajor axis length and
    local intensity gradient, measure the isophote's deviations from
    perfect ellipticity (these amplitudes, divided by semimajor axis
    and gradient, are the actual quantities stored in the output
    `~photutils.isophote.Isophote` instance).

    The algorithm then measures the integrated intensity and the number
    of nonflagged pixels inside the elliptical isophote, and also
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
    details, see the error analysis in `Busko (1996, ASPC 101, 139)
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
    stability against any nonelliptical structure such as stars, spiral
    arms, HII regions, defects, etc.

    The fit algorithm has no way of finding where, in the input image
    frame, the galaxy to be measured is located. The center (x, y)
    coordinates need to be close to the actual center for the fit
    to work. The "object centerer", the
    :meth:`~photutils.isophote.EllipseGeometry.find_center` method,
    can be used to refine the initial center before the
    `~photutils.isophote.EllipseGeometry` instance is passed to the
    constructor of this class. It scans an 11x11 window centered on
    the (x, y) coordinates of the instance, and it modifies them in
    place if it finds the galaxy center. This class does not run the
    object centerer.

    A note of caution: the ellipse fitting algorithm was designed
    explicitly with an elliptical galaxy brightness distribution in
    mind. In particular, a well-defined negative radial intensity
    gradient across the region being fitted is paramount for the
    achievement of stable solutions. Use of the algorithm in other
    types of images (e.g., planetary nebulae) may lead to inability to
    converge to any acceptable solution.
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('threshold', None, '3.2', until='4.0')
    def __init__(self, image, geometry=None,
                 threshold=None):  # noqa: ARG002
        image = _validate_image(image)
        self.image = image

        data = np.ma.getdata(image)
        if np.any(~np.isfinite(data) & ~np.ma.getmaskarray(image)):
            msg = ('Input image contains nonfinite (NaN or infinity) '
                   'values, which are excluded from the fits.')
            warnings.warn(msg, AstropyUserWarning)

        if geometry is not None:
            self._geometry = geometry
        else:
            _x0 = image.shape[1] / 2
            _y0 = image.shape[0] / 2
            self._geometry = EllipseGeometry(_x0, _y0, 10.0, eps=0.2,
                                             pa=np.pi / 2)

    @deprecated('3.2', alternative='EllipseGeometry.find_center',
                until='4.0')
    def set_threshold(self, threshold):
        """
        Modify the threshold value used by the centerer.

        This method has no effect because this class does not run the
        object centerer. Use
        :meth:`~photutils.isophote.EllipseGeometry.find_center`
        instead.

        Parameters
        ----------
        threshold : float
            The new threshold value to use.
        """
        self._geometry.centerer_threshold = threshold

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    def fit_image(self, sma0=None, minsma=0.0, maxsma=None, step=0.1,
                  conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
                  maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
                  maxgerr=DEFAULT_MAXGERR, sclip=3.0, n_clip=0,
                  integrmode=BILINEAR, linear=None, maxrit=None,
                  fix_center=False, fix_pa=False, fix_eps=False, *,
                  max_center_shift=None):
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

        Note that the fix_XXX parameters act in unison. If any of
        them is set in this call, the others assume their default
        (False) values, and together they override the fixed
        parameters of the `~photutils.isophote.EllipseGeometry`
        instance used by this class. If none of them is set, the
        fixed parameters of that instance are used. Likewise, the
        ``linear`` parameter overrides the ``linear_growth`` attribute
        of that instance. These overrides apply only to this call. The
        `~photutils.isophote.EllipseGeometry` instance is not modified.

        If the center, position angle, and ellipticity are all fixed,
        by the fix_XXX parameters or by the
        `~photutils.isophote.EllipseGeometry` instance, there is
        nothing to fit. A warning is emitted and an empty
        `~photutils.isophote.IsophoteList` is returned.

        Parameters
        ----------
        sma0 : float, optional
            The starting value for the semimajor axis length (pixels).
            This value must not be the minimum or maximum semimajor
            axis length, but something in between. The algorithm can't
            start from the very center of the galaxy image because
            the modeling of elliptical isophotes on that region is
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
            10. will be used. The starting value must be positive.

        minsma : float, optional
            The minimum value for the semimajor axis length (pixels).
            The default is 0, which also adds the central pixel to the
            returned list as an isophote with a semimajor axis length
            of 0. Its intensity is NaN if it cannot be measured (e.g.,
            the central pixel is masked or nonfinite).

        maxsma : float or `None`, optional
            The maximum value for the semimajor axis length (pixels).
            When set to `None` (default), the algorithm will increase
            the semimajor axis until one of several conditions will
            cause it to stop and revert to fit ellipses with sma <
            ``sma0``.

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear=True``, or a relative value if
            ``linear=False``). See the ``linear`` parameter. It also
            defines the elliptical annulus used by the sampling
            algorithm, which sets the areas of the elliptical sectors
            for the area integration modes (see the ``integrmode``
            parameter). It replaces the ``astep`` attribute of the
            `~photutils.isophote.EllipseGeometry` instance used by this
            class. The step must be positive. The default is 0.1.

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
            The maximum acceptable relative error in the local
            radial intensity gradient. This is the main control
            for preventing ellipses to grow to regions of too
            low signal-to-noise ratio. It specifies the maximum
            acceptable relative error in the local radial
            intensity gradient. `Busko (1996, ASPC 101, 139)
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
            exceeds ``maxgerr`` (or becomes nonsignificant and is
            set to `None`). If the maximum semimajor axis specified
            by ``maxsma`` is set to `None`, semimajor axis growth is
            stopped and the algorithm proceeds inwards to the galaxy
            center. If ``maxsma`` is set to some finite value, and this
            value is larger than the current semimajor axis length, the
            algorithm enters noniterative mode and proceeds outwards
            until reaching ``maxsma``. The default is 0.5.

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

            .. deprecated:: 3.0
                The ``nclip`` keyword is deprecated. Use ``n_clip``
                instead.

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
            current geometry, without being fitted. This noniterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            Noniterative mode can also be entered automatically
            whenever the ellipticity exceeds 1.0 or the ellipse center
            crosses the image boundaries. If `None` (default), then no
            maximum value is used.

        fix_center : bool, optional
            Keep center of ellipse fixed during fit? The default is
            False.

        fix_pa : bool, optional
            Keep position angle of semimajor axis of ellipse fixed
            during fit? The default is False.

        fix_eps : bool, optional
            Keep ellipticity of ellipse fixed during fit? The default is
            False.

        max_center_shift : float or `None`, optional
            The maximum distance (in pixels) that the center of any
            fitted isophote may move from the center of the ``geometry``
            input to `~photutils.isophote.Ellipse`. Unlike
            ``fix_center``, this lets the center move freely within a
            circle of this radius. If a fit would move the center
            outside the circle, the center of that isophote is placed
            on the circle and held fixed for the rest of its fit, while
            ellipticity and position angle continue to be fitted. If
            `None` (default), the center is not limited. This parameter
            has no effect if ``fix_center`` is `True`.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList` instance
            A list-like object of `~photutils.isophote.Isophote`
            instances, sorted by increasing semimajor axis length.
        """
        # Multiple fitted isophotes will be stored here.
        isophote_list = []

        # Get starting sma from appropriate source: keyword parameter or
        # internal EllipseGeometry instance.
        sma = self._geometry.sma if sma0 is None else sma0
        if not sma > 0:  # also rejects NaN
            msg = ('The starting semimajor axis length (sma0 or the sma of '
                   'the input geometry) must be positive')
            raise ValueError(msg)
        if not step > 0:  # also rejects NaN
            msg = 'step must be positive'
            raise ValueError(msg)

        # Override the geometry parameters set in this call on a copy
        # so that they do not persist in the input geometry instance.
        geometry = copy.deepcopy(self._geometry)
        geometry.astep = step
        if linear is not None:
            geometry.linear_growth = bool(linear)
        if fix_center or fix_pa or fix_eps:
            geometry.fix = _make_fix_array(fix_center=fix_center,
                                           fix_pa=fix_pa, fix_eps=fix_eps)
        if np.all(geometry.fix):
            msg = 'Everything is fixed. Fit not possible.'
            warnings.warn(msg, AstropyUserWarning)
            return IsophoteList([])

        # The fit parameters that are the same for all isophotes. The
        # fitted isophotes are appended to isophote_list.
        fit = functools.partial(self._fit_isophote, conver=conver,
                                maxit=maxit, fflag=fflag, sclip=sclip,
                                n_clip=n_clip, integrmode=integrmode,
                                maxrit=maxrit, isophote_list=isophote_list,
                                max_center_shift=max_center_shift)

        # First, go from initial sma outwards until
        # hitting one of several stopping criteria.
        noiter = False
        previous_exceeded = False
        while True:
            # The first isophote runs longer. The geometry of the last
            # fitted isophote is the initial guess for the next one.
            if isophote_list:
                minit_a = minit
                geometry = isophote_list[-1].sample.geometry
            else:
                minit_a = 2 * minit
            isophote = fit(geometry, sma, minit=minit_a, noniterate=noiter)

            # An invalid isophote (e.g., no valid data points) is not
            # stored, so an empty list means that the first isophote
            # could not be fit at all.
            if not isophote_list:
                msg = self._first_isophote_failure_message(isophote,
                                                           fflag)
                warnings.warn(msg, AstropyUserWarning)
                return IsophoteList([])

            # If a later isophote could not be fit or extracted at all
            # (e.g., no valid data points), it is not stored. Stop
            # growing outwards, as in the inward loop below. Otherwise,
            # the next sma would be computed again from the last stored
            # isophote, repeating the same failed fit forever.
            if isophote.stop_code == 3:
                break

            # Check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == 1:
                # In case the fit failed right at the outset, return an
                # empty list. This is the usual case when the user
                # provides initial guesses that are too way off to enable
                # the fitting algorithm to find any meaningful solution.

                if len(isophote_list) == 1:
                    msg = self._first_isophote_failure_message(isophote,
                                                               fflag)
                    warnings.warn(msg, AstropyUserWarning)
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, isophote_list[-2])

                # Get last isophote from the actual list, since the last
                # `isophote` instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # If two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if (len(isophote_list) > 2
                    and ((isophote.stop_code == 5
                          and isophote_list[-2].stop_code == 5)
                         or isophote.stop_code == 1)):
                    if maxsma is not None and maxsma > isophote.sma:
                        # If a maximum sma value was provided by
                        # user, and the current sma is smaller than
                        # maxsma, keep growing sma in noniterative
                        # mode until reaching it.
                        noiter = True
                    else:
                        # If no maximum sma, stop growing and change
                        # to go inwards.
                        break

            # Reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # Check the gradient relative error criterion. As in the
            # STSDAS ellipse task with soft stopping, it is triggered
            # only when two consecutive isophotes exceed ``maxgerr``.
            # An isophote whose gradient error could not be measured
            # counts as exceeding it. The first isophote cannot
            # trigger it, but it counts as the first of the two.
            gradient_rel_err = isophote.gradient_rel_err
            exceeded = gradient_rel_err is None or gradient_rel_err > maxgerr
            if exceeded and previous_exceeded:
                if maxsma is not None and maxsma > isophote.sma:
                    noiter = True
                else:
                    break
            previous_exceeded = exceeded

            # Update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if maxsma is not None and sma >= maxsma:
                break

        # Reset sma so as to go inwards. As in the STSDAS ellipse task,
        # the first inward isophote starts from the geometry of the
        # first isophote, which is the closest one in sma.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)
        geometry = first_isophote.sample.geometry

        # Now, go from initial sma inwards towards center, until the
        # sma reaches the user-defined minimum or is too small.
        while sma > max(minsma, 0.5):
            isophote = fit(geometry, sma, minit=minit)

            # If abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, first_isophote)

            # But if the harmonic fit failed, bail out immediately.
            # This usually happens at very small radii when the number
            # of data points is too small.
            if isophote.stop_code == 3:
                break

            # Reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]
            geometry = isophote.sample.geometry

            # Figure out the next sma.
            sma = isophote.sample.geometry.update_sma(step)

        # If user asked for minsma=0, extract special isophote there.
        # It is centered on the innermost fitted isophote, whose
        # geometry is the last one set above, and it is appended to
        # isophote_list.
        if minsma == 0.0:
            fit(geometry, 0.0, minit=minit)

        # Sort list of isophotes according to sma.
        isophote_list.sort()

        return IsophoteList(isophote_list)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    @deprecated_renamed_argument('nclip', 'n_clip', '3.0', until='4.0')
    @deprecated_renamed_argument('maxgerr', None, '3.2', until='4.0')
    @deprecated_renamed_argument('going_inwards', None, '3.2', until='4.0')
    def fit_isophote(self, sma, step=0.1, conver=DEFAULT_CONVERGENCE,
                     minit=DEFAULT_MINIT, maxit=DEFAULT_MAXIT,
                     fflag=DEFAULT_FFLAG, maxgerr=None,  # noqa: ARG002
                     sclip=3.0, n_clip=0, integrmode=BILINEAR,
                     linear=None, maxrit=None, noniterate=False,
                     going_inwards=None,  # noqa: ARG002
                     isophote_list=None, *, max_center_shift=None):
        """
        Fit a single isophote with a given semimajor axis length.

        The ``step`` and ``linear`` parameters do not directly control
        the growth or reduction of the current fitting semimajor axis
        length. Instead, they define the elliptical annulus used by
        the sampling algorithm, which sets the areas of the elliptical
        sectors for the area integration modes (see the ``integrmode``
        parameter). The ``step`` parameter replaces the ``astep``
        attribute of the starting geometry. The ``linear`` parameter,
        if set, replaces its ``linear_growth`` attribute.

        If the center, position angle, and ellipticity are all fixed
        by the `~photutils.isophote.EllipseGeometry` instance, there
        is nothing to fit. The isophote is then extracted with that
        geometry and its stop code is 4.

        Parameters
        ----------
        sma : float
            The semimajor axis length (pixels).

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear=True``, or a relative value if
            ``linear=False``). See the ``linear`` parameter. The step
            must be positive. The default is 0.1.

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

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

            .. deprecated:: 3.0
                The ``nclip`` keyword is deprecated. Use ``n_clip``
                instead.

        integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, \
                optional
            The area integration mode. The default is 'bilinear'.

        linear : bool or `None`, optional
            The semimajor axis growing/shrinking mode. When fitting
            just one isophote, this parameter is used only by the code
            that define the details of how elliptical arc segments
            ("sectors") are extracted from the image when using area
            extraction modes (see the ``integrmode`` parameter). If
            `None` (default), the ``linear_growth`` attribute of the
            starting geometry is used.

        maxrit : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``maxrit``, the isophotes will be extracted using the
            current geometry, without being fitted. This noniterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            Noniterative mode can also be entered automatically
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
            to change from iterative to noniterative mode somewhere
            along the sequence of isophotes. When set to `True`, this
            parameter overrides the behavior associated with parameter
            ``maxrit``. The default is `False`.

        going_inwards : bool, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0.

        isophote_list : list or `None`, optional
            If not `None` (the default), the fitted
            `~photutils.isophote.Isophote` instance is appended to this
            list. It must be created and managed by the caller.

        max_center_shift : float or `None`, optional
            The maximum distance (in pixels) that the ellipse center may
            move from the center of the ``geometry`` input to
            `~photutils.isophote.Ellipse`. If the fit
            would move the center farther than this, the center is
            placed on the boundary at this distance and held fixed for
            the rest of the fit, while the other free parameters
            continue to be fitted. If `None` (default), the center is
            not limited.

        Returns
        -------
        result : `~photutils.isophote.Isophote` instance
            The fitted isophote. The fitted isophote is also appended to
            the input list input to the ``isophote_list`` parameter.
        """
        if not step > 0:  # also rejects NaN
            msg = 'step must be positive'
            raise ValueError(msg)

        # If available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        geometry = self._geometry
        if isophote_list:
            geometry = isophote_list[-1].sample.geometry

        # The step and linear values define the sampling annulus. They
        # are set on a copy so that the input geometry is not modified.
        geometry = copy.deepcopy(geometry)
        geometry.astep = step
        if linear is not None:
            geometry.linear_growth = bool(linear)

        return self._fit_isophote(geometry, sma, conver=conver, minit=minit,
                                  maxit=maxit, fflag=fflag, sclip=sclip,
                                  n_clip=n_clip, integrmode=integrmode,
                                  maxrit=maxrit, noniterate=noniterate,
                                  isophote_list=isophote_list,
                                  max_center_shift=max_center_shift)

    def _fit_isophote(self, geometry, sma, *, conver, minit, maxit, fflag,
                      sclip, n_clip, integrmode, maxrit, noniterate=False,
                      isophote_list=None, max_center_shift=None):
        """
        Fit a single isophote, starting from the input geometry.

        See `fit_isophote` for a description of the parameters. The
        input ``geometry`` is the initial guess. Its ``astep`` and
        ``linear_growth`` attributes define the sampling annulus. If
        ``isophote_list`` is not `None`, a valid fitted isophote is
        appended to it.
        """
        # Do the fit.
        if noniterate or (maxrit is not None and sma > maxrit):
            isophote = self._non_iterative(sma, geometry, sclip, n_clip,
                                           integrmode)
        else:
            isophote = self._iterative(sma, geometry, sclip, n_clip,
                                       integrmode, conver, minit, maxit,
                                       fflag,
                                       max_center_shift=max_center_shift)

        # Store result in list.
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def measure_isophotes(self, isophotes, *, sclip=3.0, n_clip=0,
                          integrmode=BILINEAR):
        """
        Measure the image along a set of ellipses without fitting.

        The image is sampled along each input ellipse with its geometry
        held fixed. This is equivalent to running the IRAF ``ellipse``
        task with an input ellipse table (the ``inellip`` parameter).
        A common use is to measure an image in a second band along the
        isophotes fit to a reference image with
        :meth:`~photutils.isophote.Ellipse.fit_image`, so that both
        intensity profiles (e.g., for a color profile) are measured
        along the same elliptical paths.

        Parameters
        ----------
        isophotes : `~photutils.isophote.IsophoteList` or iterable
            The ellipses along which to measure the image. It can be an
            `~photutils.isophote.IsophoteList` (e.g., the output of
            :meth:`~photutils.isophote.Ellipse.fit_image`) or an
            iterable of
            `~photutils.isophote.Isophote` and
            `~photutils.isophote.EllipseGeometry` instances. The center,
            semimajor axis length, ellipticity, and position angle of
            each ellipse are used. The ``astep`` and ``linear_growth``
            values of each geometry define the sampling used to compute
            the local radial gradient and the sector areas. The input
            isophotes and geometries are not modified.

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

        integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, \
                optional
            The area integration mode. The default is 'bilinear'.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList` instance
            The measured isophotes, sorted by semimajor axis length.
            There is one isophote for each input ellipse, each with
            ``n_iter=0`` and ``stop_code=4``. An ellipse with a
            semimajor axis length of zero gives the intensity at its
            center, with ``stop_code=0``. An ellipse with no valid
            sampled points (e.g., one that lies entirely outside the
            image) has a NaN intensity and ``stop_code=3``.
        """
        geometries = []
        for isophote in isophotes:
            if isinstance(isophote, Isophote):
                geometries.append(isophote.sample.geometry)
            elif isinstance(isophote, EllipseGeometry):
                geometries.append(isophote)
            else:
                msg = ('isophotes must contain only Isophote or '
                       f'EllipseGeometry instances, got {type(isophote)}')
                raise TypeError(msg)

        # The samples make deep copies of the input geometries, so the
        # inputs are not modified.
        isophote_list = []
        for geometry in geometries:
            if geometry.sma == 0:
                sample = CentralEllipseSample(self.image, 0.0,
                                              geometry=geometry)
                isophote = CentralEllipseFitter(sample).fit()
            else:
                isophote = self._non_iterative(geometry.sma, geometry, sclip,
                                               n_clip, integrmode)
            isophote_list.append(isophote)

        isophote_list.sort()

        return IsophoteList(isophote_list)

    def _iterative(self, sma, geometry, sclip, n_clip, integrmode, conver,
                   minit, maxit, fflag, *, max_center_shift=None):
        if sma > 0.0:
            # Iterative fitter
            sample = EllipseSample(self.image, sma, sclip=sclip,
                                   n_clip=n_clip, geometry=geometry,
                                   integrmode=integrmode)
            fitter = EllipseFitter(sample)
        else:
            # A zero sma requires special handling.
            sample = CentralEllipseSample(self.image, 0.0, geometry=geometry)
            fitter = CentralEllipseFitter(sample)

        # The center shift is measured from the input geometry, which
        # fitting does not modify (each sample uses a copy).
        reference_center = (self._geometry.x0, self._geometry.y0)
        return fitter.fit(conver=conver, minit=minit, maxit=maxit,
                          fflag=fflag, max_center_shift=max_center_shift,
                          reference_center=reference_center)

    def _non_iterative(self, sma, geometry, sclip, n_clip, integrmode):
        sample = EllipseSample(self.image, sma, sclip=sclip, n_clip=n_clip,
                               geometry=geometry, integrmode=integrmode)
        sample.update(fixed_parameters=geometry.fix)

        # An isophote without any valid data points (e.g., an ellipse
        # outside of the image) is invalid.
        if sample.actual_points == 0:
            return Isophote(sample, 0, valid=False, stop_code=3)

        # Build isophote without iterating with an EllipseFitter.
        return Isophote(sample, 0, valid=True, stop_code=4)

    @staticmethod
    def _first_isophote_failure_message(isophote, fflag):
        """
        Return the warning message for a first isophote that could not
        be fit, with the reason given by its stop code.
        """
        check_data = ('Check that the starting ellipse is inside the image '
                      'and that its data are not masked.')
        if not isophote.valid:
            reason = ('it has too few valid data points or its harmonic '
                      'fit failed')
            advice = check_data
        elif isophote.stop_code == 1:
            reason = ('the fraction of valid data points is smaller than '
                      f'fflag ({fflag})')
            advice = check_data
        else:
            reason = ('the ellipse diverged or no meaningful intensity '
                      'gradient could be measured')
            advice = ('Check the starting ellipse geometry, or try a '
                      'slightly different starting semimajor axis length '
                      '(sma0). For an image with a low signal-to-noise '
                      'ratio, see the isophote FAQ '
                      '(https://photutils.readthedocs.io/en/latest/'
                      'user_guide/isophote_faq.html).')

        return ('No meaningful fit was possible. The first isophote '
                f'(sma={isophote.sma:g}) failed with stop code '
                f'{isophote.stop_code} because {reason}. {advice}')

    @staticmethod
    def _fix_last_isophote(isophote_list, reference):
        """
        Replace the last isophote in the list, whose fit failed, with
        an isophote extracted with the geometry of the ``reference``
        isophote.
        """
        isophote = isophote_list[-1]
        isophote.fix_geometry(reference)

        # Force new extraction of raw data, since
        # geometry changed.
        isophote.sample.values = None
        isophote.sample.update(
            fixed_parameters=isophote.sample.geometry.fix)

        # We take the opportunity to change an eventual
        # negative stop code to its positive equivalent.
        code = 5 if isophote.stop_code < 0 else isophote.stop_code

        # Build new instance so it can have its attributes
        # populated from the updated sample attributes.
        isophote_list[-1] = Isophote(isophote.sample, isophote.n_iter,
                                     isophote.valid, code)
