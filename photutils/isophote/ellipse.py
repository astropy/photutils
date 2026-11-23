# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for fitting elliptical isophotes.
"""

import copy
import functools
import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.isophote.fitter import (DEFAULT_CONVERGENCE,
                                       DEFAULT_MAX_GRADIENT_REL_ERR,
                                       DEFAULT_MAXITERS,
                                       DEFAULT_MIN_VALID_FRACTION,
                                       DEFAULT_MINITERS, _EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry, _make_fix_array
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import EllipseSample, _validate_image
from photutils.utils._deprecation import (deprecated,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)

__all__ = ['Ellipse']


def _central_isophote(image, geometry):
    """
    Make the isophote of the central pixel.

    The input geometry is not modified. The central isophote has
    ``sma``, ``ellipticity``, and ``orientation`` of zero and holds a
    single bilinear sample at the center.
    """
    geometry = copy.deepcopy(geometry)
    geometry.sma = 0.0
    geometry.ellipticity = 0.0
    geometry._orientation_rad = 0.0
    sample = EllipseSample(image, 0.0, geometry=geometry)
    sample.update()
    return Isophote(sample, 0, valid=True, stop_code=0)


class Ellipse:
    """
    Class to fit elliptical isophotes to a galaxy image.

    The isophotes in the image are measured using an iterative method
    described by `Jedrzejewski (1987, MNRAS 226, 747)
    <https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
    See the **Notes** section below for details about the algorithm.

    Parameters
    ----------
    image : 2D array_like
        The image array. Masked pixels (if ``image`` is a
        `~numpy.ma.MaskedArray`) and nonfinite (NaN or infinity)
        values are excluded from the fits.

    geometry : `~photutils.isophote.EllipseGeometry` instance or `None`, \
            optional
        The optional geometry that describes the first
        ellipse to be fitted. If `None`, a default
        `~photutils.isophote.EllipseGeometry` instance is created
        with a semimajor axis length of 10 pixels, an ellipticity of
        0.2, and an orientation of 90 degrees. Its center is at
        ``x0 = nx / 2`` and ``y0 = ny / 2``, where ``ny`` and ``nx``
        are the image dimensions.

    threshold : float, optional
        This keyword has no effect.

        .. deprecated:: 3.2
           This keyword argument is deprecated and will be removed in
           version 4.0. The object centerer is not run by this class.
           Use :meth:`~photutils.isophote.EllipseGeometry.find_center`
           to refine the initial center before creating this class.

    Raises
    ------
    TypeError
        If ``image`` is a `~astropy.units.Quantity` array.

    ValueError
        If ``image`` is not a 2D array.

    Warns
    -----
    AstropyUserWarning
        If ``image`` contains nonfinite values that are not masked.
        These values are excluded from the fits.

    Notes
    -----
    The image is measured using an iterative method
    described by `Jedrzejewski (1987, MNRAS 226, 747)
    <https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
    Each isophote is fitted at a predefined, fixed semimajor axis
    length. The algorithm starts from a first-guess elliptical isophote
    defined by approximate values for the (x, y) center coordinates,
    ellipticity, and orientation. Using these values, the image
    is sampled along an elliptical path, producing a 1D
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
    following criteria is met:

    1. the largest harmonic amplitude is less than a given fraction of
       the rms residual of the intensity data around the harmonic fit,
       scaled by the square root of the mean sector area.

    2. a user-specified maximum number of iterations is reached.

    3. more than a given fraction of the elliptical sample points have no
       valid data in them, either because they lie outside the image
       boundaries or because they were flagged out from the fit by
       sigma clipping.

    Criterion 1 is applied only after a minimum number of iterations.
    The other criteria, a diverging ellipse, or a gradient that cannot
    be measured can stop the iterations earlier. If iterations stop
    because of reasons 2 or 3 above, then those ellipse parameters
    that generated the lowest absolute values for harmonic amplitudes
    will be used. At this point, the image data sample coming from the
    best-fit ellipse is fitted by the following function:

    .. math::

        y  =  y0 + (An * \\sin(n * E)) + (Bn * \\cos(n * E))

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
    harmonic fit. Third and fourth harmonic amplitudes and their
    errors are obtained from a separate fit of each harmonic to the
    sampled intensities. The amplitude errors are combined in
    quadrature with the relative error of the local gradient. For
    more details, see the error analysis in `Busko (1996, ASPC 101, 139)
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

    @deprecated_renamed_argument('threshold', None, '3.2', until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0', stacklevel=3)
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
            self._geometry = EllipseGeometry(_x0, _y0, 10.0, 0.2,
                                             np.pi / 2)

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

    @deprecated_renamed_argument(
        ['nclip', 'sma0', 'minsma', 'maxsma', 'conver', 'minit', 'maxit',
         'fflag', 'maxgerr', 'sclip', 'integrmode', 'linear', 'maxrit',
         'fix_pa', 'fix_eps'],
        ['n_clip', 'sma_start', 'min_sma', 'max_sma', 'convergence',
         'miniters', 'maxiters', 'min_valid_fraction',
         'max_gradient_rel_err', 'clip_sigma', 'integration_mode',
         'linear_growth', 'max_iterative_sma', 'fix_orientation',
         'fix_ellipticity'],
        ['3.0'] + ['3.2'] * 14, until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0', stacklevel=3)
    def fit_image(self, sma_start=None, min_sma=0.0, max_sma=None, step=0.1,
                  convergence=DEFAULT_CONVERGENCE,
                  miniters=DEFAULT_MINITERS, maxiters=DEFAULT_MAXITERS,
                  min_valid_fraction=DEFAULT_MIN_VALID_FRACTION,
                  max_gradient_rel_err=DEFAULT_MAX_GRADIENT_REL_ERR,
                  clip_sigma=3.0, n_clip=0, integration_mode='bilinear',
                  linear_growth=None, max_iterative_sma=None,
                  fix_center=False, fix_orientation=False,
                  fix_ellipticity=False, *, max_center_shift=None):
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

        Note that the fix_XXX parameters act in unison. If any of them
        is set to `True` in this call, the others assume their default
        (False) values, and together they override the fixed
        parameters of the `~photutils.isophote.EllipseGeometry`
        instance used by this class. If none of them is set, the
        fixed parameters of that instance are used. Likewise, the
        ``linear_growth`` parameter overrides the ``linear_growth`` attribute
        of that instance. These overrides apply only to this call. The
        `~photutils.isophote.EllipseGeometry` instance is not modified.

        If the center, orientation, and ellipticity are all fixed,
        by the fix_XXX parameters or by the
        `~photutils.isophote.EllipseGeometry` instance, there is
        nothing to fit. A warning is emitted and an empty
        `~photutils.isophote.IsophoteList` is returned.

        Parameters
        ----------
        sma_start : float or `None`, optional
            The starting value for the semimajor axis length (pixels).
            This value must not be the minimum or maximum semimajor
            axis length, but something in between. The algorithm can't
            start from the very center of the galaxy image because
            the modeling of elliptical isophotes on that region is
            poor and it will diverge very easily if not tied to other
            previously fit isophotes. It can't start from the maximum
            value either because the maximum is not known beforehand,
            depending on signal-to-noise. The ``sma_start`` value should be
            selected such that the corresponding isophote has a good
            signal-to-noise ratio and a clearly defined geometry. If set
            to `None` (the default), one of two actions will be taken:
            if a `~photutils.isophote.EllipseGeometry` instance was
            input to the `~photutils.isophote.Ellipse` constructor, its
            ``sma`` value will be used. Otherwise, a default value of
            10 will be used. The starting value must be positive.

        min_sma : float, optional
            The minimum value for the semimajor axis length (pixels).
            The default is 0, which also adds the central pixel to the
            returned list as an isophote with a semimajor axis length
            of 0. Its intensity is NaN if it cannot be measured (e.g.,
            the central pixel is masked or nonfinite). Isophotes with
            a semimajor axis length smaller than 0.5 pixels are not
            fitted.

        max_sma : float or `None`, optional
            The maximum value for the semimajor axis length (pixels).
            When set to `None` (default), the algorithm will increase
            the semimajor axis until one of several conditions will
            cause it to stop and revert to fit ellipses with sma <
            ``sma_start``.

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear_growth=True``, or a relative
            value if ``linear_growth=False``). See the ``linear_growth``
            parameter. It also defines the elliptical annulus used by
            the sampling algorithm, which sets the areas of the
            elliptical sectors for the area integration modes (see the
            ``integration_mode`` parameter). It replaces the ``step``
            attribute of the `~photutils.isophote.EllipseGeometry`
            instance used by this class. The step must be positive. The
            default is 0.1.

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
            level). For the first isophote, the minimum number of
            iterations is 2 * ``miniters`` to ensure that, even departing
            from not-so-good initial values, the algorithm has a better
            chance to converge to a sensible solution.

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

        max_gradient_rel_err : float, optional
            The maximum acceptable relative error in the local
            radial intensity gradient. This is the main control
            for preventing ellipses from growing to regions of too
            low signal-to-noise ratio. `Busko (1996, ASPC 101, 139)
            <https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstr
            act>`_ showed that the fitting precision relates to that
            relative error. The usual behavior of the gradient relative
            error is to increase with semimajor axis, being larger in
            outer, fainter regions of a galaxy image. In the current
            implementation, the ``max_gradient_rel_err`` criterion is
            triggered only when two consecutive isophotes exceed the
            value specified by the parameter. This prevents premature
            stopping caused by contamination such as stars and HII
            regions.

            A number of actions may happen when the gradient error
            exceeds ``max_gradient_rel_err`` (or becomes nonsignificant
            and is set to `None`). If the maximum semimajor axis specified
            by ``max_sma`` is set to `None`, semimajor axis growth is
            stopped and the algorithm proceeds inwards to the galaxy
            center. If ``max_sma`` is set to some finite value, and this
            value is larger than the current semimajor axis length, the
            algorithm enters noniterative mode and proceeds outwards
            until reaching ``max_sma``. The default is 0.5.

        clip_sigma : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma clipping is skipped.

            .. deprecated:: 3.0
                The ``nclip`` keyword is deprecated. Use ``n_clip``
                instead.

        integration_mode : {'bilinear', 'nearest_neighbor', 'mean', \
                'median'}, optional
            The integration mode. The 'mean' and 'median' modes
            integrate over elliptical sectors and fall back to
            'bilinear' for sectors smaller than 1 pixel. The default is
            'bilinear'.

        linear_growth : bool or `None`, optional
            The semimajor axis growing/shrinking mode. If `False`,
            the geometric growing mode is chosen, thus the
            semimajor axis length is increased by a factor of (1.
            + ``step``), and the process is repeated until either
            the semimajor axis value reaches the value of parameter
            ``max_sma``, or the last fitted ellipse has more than a given
            fraction of its sampled points flagged out (see
            ``min_valid_fraction``). The process then resumes from the
            first fitted ellipse (at ``sma_start``) inwards, in steps
            of (1./(1. + ``step``)), until the semimajor axis length
            reaches the value ``min_sma``. In case of linear growing,
            the increment or decrement value is given directly by
            ``step`` in pixels. If ``max_sma`` is set to `None`, the
            semimajor axis will grow until a low signal-to-noise
            criterion is met. See ``max_gradient_rel_err``. If `None`
            (default), the ``linear_growth`` attribute of the input
            geometry is used (`False` for the default geometry).

        max_iterative_sma : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``max_iterative_sma``, the isophotes will be extracted
            using the current geometry, without being fitted. This
            noniterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            If ``max_sma`` is set, noniterative mode is also entered
            when two consecutive isophotes fail to fit (e.g., the
            ellipticity exceeds 0.95 or the center leaves the image),
            when an isophote has too many flagged data points (see
            ``min_valid_fraction``), or when the
            ``max_gradient_rel_err`` criterion is met. If `None`
            (default), then no maximum value is used.

        fix_center : bool, optional
            Keep center of ellipse fixed during fit? The default is
            False.

        fix_orientation : bool, optional
            Keep orientation of semimajor axis of ellipse fixed
            during fit? The default is False.

        fix_ellipticity : bool, optional
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
            ellipticity and orientation continue to be fitted. If
            `None` (default), the center is not limited. This parameter
            has no effect if ``fix_center`` is `True`.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList` instance
            A list-like object of `~photutils.isophote.Isophote`
            instances, sorted by increasing semimajor axis length.

        Raises
        ------
        ValueError
            If the starting semimajor axis length or ``step`` is not
            positive, if ``max_center_shift`` is negative or NaN, if
            the ellipticity of the starting geometry is 1 or larger, or
            if ``integration_mode`` is not a valid integration mode.

        Warns
        -----
        AstropyUserWarning
            If the center, orientation, and ellipticity are all
            fixed, or if the first isophote cannot be fit. In both
            cases, an empty `~photutils.isophote.IsophoteList` is
            returned. A warning is also emitted for an isophote whose
            sample has too few valid data points or whose harmonic fit
            fails.
        """
        # Multiple fitted isophotes will be stored here.
        isophote_list = []

        # Get starting sma from appropriate source: keyword parameter or
        # internal EllipseGeometry instance.
        sma = self._geometry.sma if sma_start is None else sma_start
        if not sma > 0:  # also rejects NaN
            msg = ('The starting semimajor axis length (sma_start or the '
                   'sma of the input geometry) must be positive')
            raise ValueError(msg)
        if not step > 0:  # also rejects NaN
            msg = 'step must be positive'
            raise ValueError(msg)

        # Override the geometry parameters set in this call on a copy
        # so that they do not persist in the input geometry instance.
        geometry = copy.deepcopy(self._geometry)
        geometry.step = step
        if linear_growth is not None:
            geometry.linear_growth = bool(linear_growth)
        if fix_center or fix_orientation or fix_ellipticity:
            geometry._fix = _make_fix_array(fix_center=fix_center,
                                            fix_orientation=fix_orientation,
                                            fix_ellipticity=fix_ellipticity)
        if np.all(geometry._fix):
            msg = 'Everything is fixed. Fit not possible.'
            warnings.warn(msg, AstropyUserWarning)
            return IsophoteList([])

        # The fit parameters that are the same for all isophotes. The
        # fitted isophotes are appended to isophote_list.
        fit = functools.partial(
            self._fit_isophote, convergence=convergence, maxiters=maxiters,
            min_valid_fraction=min_valid_fraction, clip_sigma=clip_sigma,
            n_clip=n_clip, integration_mode=integration_mode,
            max_iterative_sma=max_iterative_sma,
            isophote_list=isophote_list, max_center_shift=max_center_shift)

        # First, go from initial sma outwards until
        # hitting one of several stopping criteria.
        noiter = False
        previous_exceeded = False
        while True:
            # The first isophote runs longer. The geometry of the last
            # fitted isophote is the initial guess for the next one.
            if isophote_list:
                miniters_a = miniters
                geometry = isophote_list[-1].geometry
            else:
                miniters_a = 2 * miniters
            isophote = fit(geometry, sma, miniters=miniters_a,
                           noniterate=noiter)

            # An invalid isophote (e.g., no valid data points) is not
            # stored, so an empty list means that the first isophote
            # could not be fit at all.
            if not isophote_list:
                msg = self._first_isophote_failure_message(isophote,
                                                           min_valid_fraction)
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
                    msg = self._first_isophote_failure_message(
                        isophote, min_valid_fraction)
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
                    if max_sma is not None and max_sma > isophote.sma:
                        # If a maximum sma value was provided by
                        # user, and the current sma is smaller than
                        # max_sma, keep growing sma in noniterative
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
            # only when two consecutive isophotes exceed
            # ``max_gradient_rel_err``. An isophote whose gradient error
            # could not be measured counts as exceeding it. The first
            # isophote cannot trigger it, but it counts as the first of
            # the two.
            # The sample value is tested because a gradient error that
            # could not be measured is `None` there, while a measured
            # relative error can also be NaN.
            gradient_rel_err = isophote.sample.gradient_rel_err
            exceeded = (gradient_rel_err is None
                        or gradient_rel_err > max_gradient_rel_err)
            if exceeded and previous_exceeded:
                if max_sma is not None and max_sma > isophote.sma:
                    noiter = True
                else:
                    break
            previous_exceeded = exceeded

            # Update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry._update_sma(step)
            if max_sma is not None and sma >= max_sma:
                break

        # Reset sma so as to go inwards. As in the STSDAS ellipse task,
        # the first inward isophote starts from the geometry of the
        # first isophote, which is the closest one in sma.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry._reset_sma(step)
        geometry = first_isophote.sample.geometry

        # Now, go from initial sma inwards towards center, until the
        # sma reaches the user-defined minimum or is too small.
        while sma > max(min_sma, 0.5):
            isophote = fit(geometry, sma, miniters=miniters)

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
            sma = isophote.sample.geometry._update_sma(step)

        # If user asked for min_sma=0, extract special isophote there.
        # It is centered on the innermost fitted isophote, whose
        # geometry is the last one set above, and it is appended to
        # isophote_list.
        if min_sma == 0.0:
            fit(geometry, 0.0, miniters=miniters)

        # Sort list of isophotes according to sma.
        isophote_list.sort(key=lambda iso: iso.sma)

        return IsophoteList(isophote_list)

    @deprecated_renamed_argument(
        ['nclip', 'maxgerr', 'going_inwards', 'conver', 'minit', 'maxit',
         'fflag', 'sclip', 'integrmode', 'linear', 'maxrit'],
        ['n_clip', None, None, 'convergence', 'miniters', 'maxiters',
         'min_valid_fraction', 'clip_sigma', 'integration_mode',
         'linear_growth', 'max_iterative_sma'],
        ['3.0'] + ['3.2'] * 10, until='4.0')
    @deprecated_positional_kwargs(since='3.0', until='4.0', stacklevel=3)
    def fit_isophote(self, sma, step=0.1, convergence=DEFAULT_CONVERGENCE,
                     miniters=DEFAULT_MINITERS, maxiters=DEFAULT_MAXITERS,
                     min_valid_fraction=DEFAULT_MIN_VALID_FRACTION,
                     maxgerr=None,  # noqa: ARG002
                     clip_sigma=3.0, n_clip=0, integration_mode='bilinear',
                     linear_growth=None, max_iterative_sma=None,
                     noniterate=False,
                     going_inwards=None,  # noqa: ARG002
                     isophote_list=None, *, max_center_shift=None):
        """
        Fit a single isophote with a given semimajor axis length.

        The ``step`` and ``linear_growth`` parameters do not directly
        control the growth or reduction of the current fitting
        semimajor axis length. Instead, they define the elliptical
        annulus used by the sampling algorithm, which sets the areas of
        the elliptical sectors for the area integration modes (see the
        ``integration_mode`` parameter). The ``step`` parameter sets
        the ``step`` attribute of the starting geometry. The
        ``linear_growth`` parameter, if set, sets its ``linear_growth``
        attribute.

        If the center, orientation, and ellipticity are all fixed
        by the `~photutils.isophote.EllipseGeometry` instance, there
        is nothing to fit. The isophote is then extracted with that
        geometry and its stop code is 4.

        Parameters
        ----------
        sma : float
            The semimajor axis length (pixels).

        step : float, optional
            The step value that defines the sampling annulus (pixels
            if linear growth is used, otherwise a relative value). See
            the ``linear_growth`` parameter. The step must be positive. The
            default is 0.1.

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
               in version 4.0. The ``max_gradient_rel_err`` criterion compares
               consecutive isophotes. It is applied by
               :meth:`~photutils.isophote.Ellipse.fit_image`, so it has
               no effect when fitting a single isophote.

        clip_sigma : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma clipping is skipped.

            .. deprecated:: 3.0
                The ``nclip`` keyword is deprecated. Use ``n_clip``
                instead.

        integration_mode : {'bilinear', 'nearest_neighbor', 'mean', \
                'median'}, optional
            The integration mode. The 'mean' and 'median' modes
            integrate over elliptical sectors and fall back to
            'bilinear' for sectors smaller than 1 pixel. The default is
            'bilinear'.

        linear_growth : bool or `None`, optional
            The semimajor axis growing/shrinking mode. When fitting
            just one isophote, this parameter is used only by the code
            that defines the details of how elliptical arc segments
            ("sectors") are extracted from the image when using area
            extraction modes (see the ``integration_mode`` parameter). If
            `None` (default), the ``linear_growth`` attribute of the
            starting geometry is used.

        max_iterative_sma : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``max_iterative_sma``, the isophotes will be extracted
            using the current geometry, without being fitted. This
            noniterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            If the ellipticity exceeds 0.95 or the center leaves the
            image during the fit, the fit stops with a stop code of -1.
            If `None` (default), then no maximum value is used.

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
            ``max_iterative_sma``. The default is `False`.

        going_inwards : bool, optional
            This keyword has no effect.

            .. deprecated:: 3.2
               This keyword argument is deprecated and will be removed
               in version 4.0.

        isophote_list : list or `None`, optional
            If not `None` (the default), a valid fitted
            `~photutils.isophote.Isophote` instance is appended to this
            list. It must be created and managed by the caller. If the
            list is not empty, the geometry of its last isophote is the
            starting geometry of the fit. Otherwise, the geometry input
            to `~photutils.isophote.Ellipse` is used.

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
            The fitted isophote. If ``isophote_list`` is not `None` and
            the isophote is valid, it is also appended to that list.

        Raises
        ------
        ValueError
            If ``step`` is not positive, if ``max_center_shift`` is
            negative or NaN, if the ellipticity of the starting geometry
            is 1 or larger, or if ``integration_mode`` is not a valid
            integration mode.

        Warns
        -----
        AstropyUserWarning
            If the sample has too few valid data points or the harmonic
            fit fails. The returned isophote is then invalid with a stop
            code of 3.
        """
        if not step > 0:  # also rejects NaN
            msg = 'step must be positive'
            raise ValueError(msg)

        # If available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        geometry = self._geometry
        if isophote_list:
            geometry = isophote_list[-1].geometry

        # The step and linear values define the sampling annulus. They
        # are set on a copy so that the input geometry is not modified.
        geometry = copy.deepcopy(geometry)
        geometry.step = step
        if linear_growth is not None:
            geometry.linear_growth = bool(linear_growth)

        return self._fit_isophote(
            geometry, sma, convergence=convergence, miniters=miniters,
            maxiters=maxiters, min_valid_fraction=min_valid_fraction,
            clip_sigma=clip_sigma, n_clip=n_clip,
            integration_mode=integration_mode,
            max_iterative_sma=max_iterative_sma, noniterate=noniterate,
            isophote_list=isophote_list, max_center_shift=max_center_shift)

    def _fit_isophote(self, geometry, sma, *, convergence, miniters,
                      maxiters, min_valid_fraction, clip_sigma, n_clip,
                      integration_mode, max_iterative_sma,
                      noniterate=False, isophote_list=None,
                      max_center_shift=None):
        """
        Fit a single isophote, starting from the input geometry.

        See `fit_isophote` for a description of the parameters. The
        input ``geometry`` is the initial guess. Its ``step`` and
        ``linear_growth`` attributes define the sampling annulus. If
        ``isophote_list`` is not `None`, a valid fitted isophote is
        appended to it.
        """
        # Do the fit.
        if noniterate or (max_iterative_sma is not None
                          and sma > max_iterative_sma):
            isophote = self._non_iterative(sma, geometry, clip_sigma, n_clip,
                                           integration_mode)
        else:
            isophote = self._iterative(sma, geometry, clip_sigma, n_clip,
                                       integration_mode, convergence,
                                       miniters, maxiters,
                                       min_valid_fraction,
                                       max_center_shift=max_center_shift)

        # Store result in list.
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def measure_isophotes(self, isophotes, *, clip_sigma=3.0, n_clip=0,
                          integration_mode='bilinear'):
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
            semimajor axis length, ellipticity, and orientation of
            each ellipse are used. The ``step`` and ``linear_growth``
            values of each geometry define the sampling used to compute
            the local radial gradient and the sector areas. The input
            isophotes and geometries are not modified.

        clip_sigma : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma clipping is skipped.

        integration_mode : {'bilinear', 'nearest_neighbor', 'mean', \
                'median'}, optional
            The integration mode. The 'mean' and 'median' modes
            integrate over elliptical sectors and fall back to
            'bilinear' for sectors smaller than 1 pixel. The default is
            'bilinear'.

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

        Raises
        ------
        TypeError
            If ``isophotes`` contains an element that is neither an
            `~photutils.isophote.Isophote` nor an
            `~photutils.isophote.EllipseGeometry` instance.

        ValueError
            If an ellipse has an ellipticity of 1 or larger or a
            nonfinite center, or if ``integration_mode`` is not a valid
            integration mode.
        """
        geometries = []
        for isophote in isophotes:
            if isinstance(isophote, Isophote):
                geometries.append(isophote.geometry)
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
                isophote = _central_isophote(self.image, geometry)
            else:
                isophote = self._non_iterative(geometry.sma, geometry,
                                               clip_sigma, n_clip,
                                               integration_mode)
            isophote_list.append(isophote)

        isophote_list.sort(key=lambda iso: iso.sma)

        return IsophoteList(isophote_list)

    def _iterative(self, sma, geometry, clip_sigma, n_clip,
                   integration_mode, convergence, miniters, maxiters,
                   min_valid_fraction, *, max_center_shift=None):
        if sma <= 0.0:
            # A zero sma requires special handling.
            return _central_isophote(self.image, geometry)

        sample = EllipseSample(self.image, sma, clip_sigma=clip_sigma,
                               n_clip=n_clip, geometry=geometry,
                               integration_mode=integration_mode)
        fitter = _EllipseFitter(sample)

        # The center shift is measured from the input geometry, which
        # fitting does not modify (each sample uses a copy).
        reference_center = (self._geometry.x0, self._geometry.y0)
        return fitter.fit(convergence=convergence, miniters=miniters,
                          maxiters=maxiters,
                          min_valid_fraction=min_valid_fraction,
                          max_center_shift=max_center_shift,
                          reference_center=reference_center)

    def _non_iterative(self, sma, geometry, clip_sigma, n_clip,
                       integration_mode):
        sample = EllipseSample(self.image, sma, clip_sigma=clip_sigma,
                               n_clip=n_clip, geometry=geometry,
                               integration_mode=integration_mode)
        sample._update(fixed_parameters=geometry._fix)

        # An isophote without any valid data points (e.g., an ellipse
        # outside of the image) is invalid.
        if sample.n_valid == 0:
            return Isophote(sample, 0, valid=False, stop_code=3)

        # Build isophote without iterating with a fitter.
        return Isophote(sample, 0, valid=True, stop_code=4)

    @staticmethod
    def _first_isophote_failure_message(isophote, min_valid_fraction):
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
                      f'min_valid_fraction ({min_valid_fraction})')
            advice = check_data
        else:
            reason = ('the ellipse diverged or no meaningful intensity '
                      'gradient could be measured')
            advice = ('Check the starting ellipse geometry, or try a '
                      'slightly different starting semimajor axis length '
                      '(sma_start). For an image with a low signal-to-noise '
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
        isophote._fix_geometry(reference)

        # Force new extraction of raw data, since
        # geometry changed.
        isophote.sample._values = None
        isophote.sample._update(
            fixed_parameters=isophote.sample.geometry._fix)

        # We take the opportunity to change an eventual
        # negative stop code to its positive equivalent.
        code = 5 if isophote.stop_code < 0 else isophote.stop_code

        # Build new instance so it can have its attributes
        # populated from the updated sample attributes.
        isophote_list[-1] = Isophote(isophote.sample, isophote.n_iter,
                                     isophote.valid, code)
