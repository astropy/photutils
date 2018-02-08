"""
This module provides tools for creation and fitting of empirical PSFs (ePSF)
to stars.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import warnings
import copy
import numpy as np

from astropy.convolution import convolve, Kernel
from astropy.modeling import fitting
from astropy.modeling.parameters import Parameter

from .centroid import find_peak
from .models import FittableImageModel2D, NonNormalizable
from .catalogs import Star
from .utils import py2round, interpolate_missing_data

__all__ = ['PSF2DModel', 'build_psf', 'iter_build_psf', 'fit_stars',
           'compute_residuals']

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler(level=logging.INFO))


_kernel_quar = np.array(
    [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
     [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
     [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
     [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
     [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]]
)

_kernel_quad = np.array(
    [[-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311],
     [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
     [+0.03999952, 0.12571449, 0.15428215, 0.12571449, +0.03999952],
     [+0.01142786, 0.09714283, 0.12571449, 0.09714283, +0.01142786],
     [-0.07428311, 0.01142786, 0.03999952, 0.01142786, -0.07428311]]
)


class PSF2DModel(FittableImageModel2D):
    """ A subclass of `~psfutils.models.FittableImageModel2D` that adds the
    ``pixel_scale`` attribute so that oversampling factor can be computed from
    scales of stars.

    .. note::

        If this class is subclassed in order to override the
        :py:func:`~psfutils.FittableImageModel2D.compute_interpolator()`
        method of the :py:class:`psfutils.FittableImageModel2D` class, then,
        depending on the properties of the new interpolator, one may need
        to override the :py:func:`~psfutils.FittableImageModel2D.evaluate`
        method as well. In particular, it is important that a custom
        interpolator's evaluation function accepts 2D `~numpy.ndarray`
        for coordinate arguments.

    """
    def __init__(self, data, flux=1.0, x_0=0, y_0=0, origin=None,
                 normalize=True, correction_factor=1.0, fill_value=0.0,
                 ikwargs={}, pixel_scale=1, peak_fit_box=5,
                 peak_search_box=None, recenter_accuracy=1.0e-4,
                 recenter_nmax=1000):
        """
        :py:class:`PSF2DModel`'s initializer has almost the same parameters
        as the :py:class:`psfutils.FittableImageModel2D` initializer.
        Therefore here we document only the differences.

        Parameters
        ----------
        pixel_scale : float, optional
            Pixel scale (in arbitrary units) of model's image. Either a single
            floating point value or a 1D iterable of length at least 2
            (x-scale, y-scale) can be provided.

        """
        super(PSF2DModel, self).__init__(
            data=data,
            flux=flux,
            x_0=x_0,
            y_0=y_0,
            origin=origin,
            normalize=normalize,
            correction_factor=correction_factor,
            fill_value=fill_value,
            ikwargs=ikwargs
        )
        self._pixscale = pixel_scale

    @property
    def pixel_scale(self):
        """ Set/Get pixel scale (in arbitrary units). Either a single floating
        point value or a 1D iterable of length at least 2 (x-scale, y-scale)
        can be provided. When getting pixel scale, a tuple of two values is
        returned with a pixel scale for each axis.

        """
        return self._pixscale

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale):
        if hasattr(pixel_scale, '__iter__'):
            if len(pixel_scale) != 2:
                raise TypeError("Parameter 'pixel_scale' must be either a "
                                "scalar or an iterable with two elements.")
            self._pixscale = (float(pixel_scale[0]), float(pixel_scale[1]))
        else:
            self._pixscale = (float(pixel_scale), float(pixel_scale))

    def make_similar_from_data(self, data, origin=None):
        """
        This method creates a new model initialized with the same
        settings as the current object except for the ``origin`` and
        model parameters from input data.

        **IMPORTANT:** Model parameters of the new model will be set to default
        values.

        This method may need to be overriden in a subclass to take into account
        additional settings that may be present in subclass' initializer.


        Parameters
        ----------
        data : numpy.ndarray
            Array containing 2D image.

        origin : tuple, None, optional
            A reference point in the input image ``data`` array. See
            :py:class:`~PSF2DModel` for more details.

            The *only difference* here is the behavior when ``origin``
            is `None`. In this case:

            * if input ``data`` has the same shape as the shape as the
              current object, then origin will be set to the same value as in
              the current object;

            * if input ``data`` has a shape different from the shape of the
              current object, then the origin of the new object will be set
              at the center of the image array.


        Returns
        -------
        new_model : PSF2DModel
            A new `PSF2DModel` constructed from new data using
            same settings as the current object except for ``origin``
            and model parameters.

        """
        data = np.asarray(data, dtype=np.float64)

        if data.shape == self.shape:
            origin = self.origin

        new_model = PSF2DModel(
            data=data,
            flux=self.flux.default,
            x_0=self.x_0.default,
            y_0=self.y_0.default,
            normalize=self.normalization_status != 2,
            correction_factor=self.correction_factor,
            origin=origin,
            fill_value=self.fill_value,
            ikwargs=self.interpolator_kwargs,
            pixel_scale=self._pixscale
        )

        return new_model


def init_psf(stars, shape=None, oversampling=4.0, pixel_scale=None,
             psf_cls=PSF2DModel, **kwargs):
    """
    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts that are used to "build" a PSF.

    psf : PSF2DModel, type, None, optional
        An existing approximation to the PSF which needs to be recomputed
        using new ``stars`` (with new parameters for center and flux)
        or a class indicating the type of the ``psf`` object to be created
        (preferebly a subclass of :py:class:`PSF2DModel`).
        If `psf` is `None`, a new ``psf`` will be computed using
        :py:class:`PSF2DModel`.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models. This is ignored if `psf` is not `None`.

    oversampling : float, tuple of float, list of (float, tuple of float), \
optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    """
    # check parameters:
    if pixel_scale is None and oversampling is None:
        raise ValueError(
            "'oversampling' and 'pixel_scale' cannot be 'None' together. "
            "At least one of these two parameters must be provided."
        )

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # find pixel scale:
    if pixel_scale is None:
        ovx, ovy = _parse_tuple_pars(oversampling, name='oversampling',
                                     dtype=float)

        # compute PSF's pixel scale as the smallest scale of the stars
        # divided by the requested oversampling factor:
        pscale_x, pscale_y = all_stars[0].pixel_scale
        for s in all_stars[1:]:
            px, py = s.pixel_scale
            if px < pscale_x:
                pscale_x = px
            if py < pscale_y:
                pscale_y = py

        pscale_x /= ovx
        pscale_y /= ovy

    else:
        pscale_x, pscale_y = _parse_tuple_pars(pixel_scale, name='pixel_scale',
                                               dtype=float)

    # if shape is None, find the minimal shape that will include input star's
    # data:
    if shape is None:
        w = np.array([((s.x_center + 0.5) * s.pixel_scale[0] / pscale_x,
                       (s.nx - s.x_center - 0.5) * s.pixel_scale[0] / pscale_x)
                      for s in all_stars])
        h = np.array([((s.y_center + 0.5) * s.pixel_scale[1] / pscale_y,
                       (s.ny - s.y_center - 0.5) * s.pixel_scale[1] / pscale_y)
                      for s in all_stars])

        # size of the PSF in the input image pixels
        # (the image with min(pixel_scale)):
        nx = int(np.ceil(np.amax(w / ovx + 0.5)))
        ny = int(np.ceil(np.amax(h / ovy + 0.5)))

        # account for a maximum error of 1 pix in the initial star coordinates:
        nx += 2
        ny += 2

        # convert to oversampled pixels:
        nx = int(np.ceil(np.amax(nx * ovx + 0.5)))
        ny = int(np.ceil(np.amax(ny * ovy + 0.5)))

        # we prefer odd sized images
        nx += 1 - nx % 2
        ny += 1 - ny % 2
        shape = (ny, nx)

    else:
        (ny, nx) = _parse_tuple_pars(shape, name='shape', dtype=int)

    # center of the output grid:
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore", NonNormalizable)

        # filter out parameters set by us:
        kwargs = copy.deepcopy(kwargs)
        for kw in ['data', 'origin', 'normalize', 'pixel_scale']:
            if kw in kwargs:
                del kwargs[kw]

        data = np.zeros((ny, nx), dtype=np.float)
        psf = psf_cls(
            data=data, origin=(cx, cy), normalize=True,
            pixel_scale=(pscale_x, pscale_y), **kwargs
        )

    return psf


def build_psf(stars, psf=None, peak_fit_box=5, peak_search_box='fitbox',
              recenter_accuracy=1.0e-4, recenter_nmax=1000,
              ignore_badfit_stars=True,
              stat='median', nclip=10, lsig=3.0, usig=3.0, ker='quar'):
    """
    Register multiple stars into a single oversampled grid to create an ePSF.

    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts that are used to "build" a PSF.

    psf : PSF2DModel, type, None, optional
        An existing approximation to the PSF which needs to be recomputed
        using new ``stars`` (with new parameters for center and flux)
        or a class indicating the type of the ``psf`` object to be created
        (preferebly a subclass of :py:class:`PSF2DModel`).
        If ``psf`` is `None`, a new ``psf`` will be computed using
        :py:class:`PSF2DModel`.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models. This is ignored if ``psf`` is not `None`.

    oversampling : float, tuple of float, list of (float, tuple of float), \
optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    peak_fit_box : int, tuple of int, optional
        Size (in pixels) of the box around the ``origin`` (or around PSF's peak
        if peak was searched for - see ``peak_search_box`` for more details)
        to be used for quadratic fitting from which peak location is computed.
        If a single integer number is provided, then it is assumed that fitting
        box is a square with sides of length given by ``peak_fit_box``. If a
        tuple of two values is provided, then first value indicates the width
        of the box and the second value indicates the height of the box.

    peak_search_box :  str {'all', 'off', 'fitbox'}, int, tuple of int, None, \
optional
        Size (in pixels) of the box around the ``origin`` of the input ``psf``
        to be used for brute-force search of the maximum value pixel. This
        search is performed before quadratic fitting in order to improve
        the original estimate of the peak location. If a single integer
        number is provided, then it is assumed that search box is a square
        with sides of length given by ``peak_fit_box``. If a tuple of two
        values is provided, then first value indicates the width of the box
        and the second value indicates the height of the box. ``'off'`` or
        `None` turns off brute-force search of the maximum. When
        ``peak_search_box`` is ``'all'`` then the entire data array of the
        ``psf`` is searched for maximum and when it is set to ``'fitbox'`` then
        the brute-force search is performed in the same box as
        ``peak_fit_box``.

    recenter_accuracy : float, optional
        Accuracy of PSF re-centering. When a PSF is assembled from input stars,
        often its center is not at the location specified by
        :py:meth:`~psfutils.models.FittableImageModel2D.origin` due to errors
        is input star positions. Therefore, a re-centering of the PSF
        must be performed.

    recenter_nmax : int, optional
        Maximum number of PSF re-centering iterations.

    ignore_badfit_stars : bool, optional
        This parameter indicates whether or not to prevent stars that have
        `~psfutils.catalogs.Star.fit_error_status` larger than zero from being
        used for building a PSF.

    stat : str {'pmode1', 'pmode2', 'mean', 'median'}, optional
        When multiple stars contribute to the same pixel in the PSF this
        parameter indicates how the value of that pixel in the PSF is computed
        (i.e., which statistics is to be used):

        * 'pmode1' - SEXTRACTOR-like mode estimate based on a
          modified `Pearson's rule <http://en.wikipedia.org/wiki/\
Nonparametric_skew#Pearson.27s_rule>`_:
          ``2.5*median-1.5*mean``;
        * 'pmode2' - mode estimate based on
          `Pearson's rule <http://en.wikipedia.org/wiki/Nonparametric_skew#\
Pearson.27s_rule>`_:
          ``3*median-2*mean``;
        * 'mean' - the mean of the distribution of the "good" pixels (after
          clipping);
        * 'median' - the median of the distribution of the "good" pixels.

    nclip : int, optional
        A non-negative number of clipping iterations to use when computing
        the sky value.

    lsig : float, optional
        Lower clipping limit, in sigma, used when computing PSF pixel value.

    usig : float, optional
        Upper clipping limit, in sigma, used when computing PSF pixel value.

    ker : str {'quad', 'quar'}, numpy.ndarray, Kernel, None, optional
        PSF is to be convolved with the indicated kernel. Predefined smoothing
        kernels ``'quad'`` and ``'quar'`` have been optimized for oversampling
        factor 4. Besides `~numpy.ndarray`, ``ker`` also accepts
        `~astropy.convolution.Kernel`.

    Returns
    -------
    ePSF : PSF2DModel
        A 2D normalized fittable image model of the ePSF.

    """

    # process parameters:
    if len(stars) < 1:
        raise ValueError("'stars' must be a list containing at least one "
                         "'Star' object.")

    if psf is None:
        psf = init_psf(stars)
    elif isinstance(psf, type):
        psf = init_psf(stars, psf_cls=psf)
    else:
        psf = copy.deepcopy(psf)

    recenter_accuracy = float(recenter_accuracy)
    if recenter_accuracy <= 0.0:
        raise ValueError("Re-center accuracy must be a strictly positive "
                         "number.")

    recenter_nmax = int(recenter_nmax)
    if recenter_nmax < 0:
        raise ValueError("Maximum number of re-recntering iterations must be "
                         "a non-negative integer number.")

    cx, cy = psf.origin
    ny, nx = psf.shape
    pscale_x, pscale_y = psf.pixel_scale

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # allocate "accumulator" array (to store transformed PSFs):
    apsf = [[[] for k in range(nx)] for k in range(ny)]

    norm_psf_data = psf.normalized_data

    for s in all_stars:
        if s.ignore:
            continue

        if ignore_badfit_stars and s.fit_error_status is not None and \
           s.fit_error_status > 0:
            continue

        pixlist = s.centered_plist(normalized=True)

        # evaluate previous PSF model at star pixel location in the PSF grid:
        ovx = s.pixel_scale[0] / pscale_x
        ovy = s.pixel_scale[1] / pscale_y
        x = ovx * (pixlist[:, 0])
        y = ovy * (pixlist[:, 1])
        old_model_vals = psf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0)

        # find integer location of star pixels in the PSF grid
        # and compute residuals
        ix = py2round(x + cx).astype(np.int)
        iy = py2round(y + cy).astype(np.int)
        pv = pixlist[:, 2] / (ovx * ovy) - old_model_vals
        m = np.logical_and(
            np.logical_and(ix >= 0, ix < nx),
            np.logical_and(iy >= 0, iy < ny)
        )

        # add all pixel values to the corresponding accumulator:
        for i, j, v in zip(ix[m], iy[m], pv[m]):
            apsf[j][i].append(v)

    psfdata = np.empty((ny, nx), dtype=np.float)
    psfdata.fill(np.nan)

    for i in range(nx):
        for j in range(ny):
            psfdata[j, i] = _pixstat(
                apsf[j][i], stat=stat, nclip=nclip, lsig=lsig, usig=usig,
                default=np.nan
            )

    mask = np.isfinite(psfdata)
    if not np.all(mask):
        # fill in the "holes" (=np.nan) using interpolation
        # I. using cubic spline for inner holes
        psfdata = interpolate_missing_data(psfdata, method='cubic',
                                           mask=mask)

        # II. we fill outer points with zeros
        mask = np.isfinite(psfdata)
        psfdata[np.logical_not(mask)] = 0.0

    # add residuals to old PSF data:
    psfdata += norm_psf_data

    # apply a smoothing kernel to the PSF:
    psfdata = _smoothPSF(psfdata, ker)

    shift_x = 0
    shift_y = 0
    peak_eps2 = recenter_accuracy**2
    eps2_prev = None
    y, x = np.indices(psfdata.shape, dtype=np.float)
    ePSF = psf.make_similar_from_data(psfdata)
    ePSF.fill_value = 0.0

    for iteration in range(recenter_nmax):
        # find peak location:
        peak_x, peak_y = find_peak(
            psfdata, xmax=cx, ymax=cy, peak_fit_box=peak_fit_box,
            peak_search_box=peak_search_box, mask=None
        )

        dx = cx - peak_x
        dy = cy - peak_y

        eps2 = dx**2 + dy**2
        if (eps2_prev is not None and eps2 > eps2_prev) or eps2 < peak_eps2:
            break
        eps2_prev = eps2

        shift_x += dx
        shift_y += dy

        # Resample PSF data to a shifted grid such that the pick of the PSF is
        # at expected position.
        psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                x_0=shift_x + cx, y_0=shift_y + cy)

    # apply final shifts and fill in any missing data:
    if shift_x != 0.0 or shift_y != 0.0:
        ePSF.fill_value = np.nan
        psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                x_0=shift_x + cx, y_0=shift_y + cy)

        # fill in the "holes" (=np.nan) using 0 (no contribution to the flux):
        mask = np.isfinite(psfdata)
        psfdata[np.logical_not(mask)] = 0.0

    norm = np.abs(np.sum(psfdata, dtype=np.float64))
    psfdata /= norm

    # Create ePSF model and return:
    ePSF = psf.make_similar_from_data(psfdata)

    return ePSF


def fit_stars(stars, psf, psf_fit_box=5, fitter=fitting.LevMarLSQFitter,
              fitter_kwargs={}, residuals=False):
    """
    Fit a PSF model to stars.

    .. note::
        When models in ``stars`` contain weights, a weighted fit of the PSF to
        the stars will be performed.

    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts to which the PSF must be fitted.
        Fitting procedure relies on correct coordinates of the center of the
        PSF and as close as possible to the correct center positions of stars.
        Star positions are derived from ``x_0`` and ``y_0`` parameters of the
        `PSF2DModel` model.

    psf : PSF2DModel
        A PSF model to be fitted to the stars.

    psf_fit_box : int, tuple of int, None, optional
        The size of the innermost box centered on stars center to be used for
        PSF fitting. This allows using only a small number of central pixels
        of the star for fitting processed thus ignoring wings. A tuple of
        two integers can be used to indicate separate sizes of the fitting
        box for ``X-`` and ``Y-`` axes. When ``psf_fit_box`` is `None`, the
        entire star's image will be used for fitting.

    fitter : astropy.modeling.fitting.Fitter, optional
        A :py:class:`~astropy.modeling.fitting.Fitter`-subclassed fitter
        class or initialized object.

    fitter_kwargs : dict-like, optional
        Additional optional keyword arguments to be passed directly to
        fitter's ``__call__()`` method.

    residuals : bool, optional
        Enable/disable computation of residuals between star's data and fitted
        PSF model. Residual image can be retrieved using
        :py:attr:`~psfutils.catalogs.Star.fit_residual` of returned star(s).


    Returns
    -------
    fitted_stars : list of FittableImageModel2D
        A list of `~psfutils.models.FittableImageModel2D` of stars with model
        parameters `~psfutils.models.FittableImageModel2D.x_0` and
        `~psfutils.models.FittableImageModel2D.y_0` set to 0 and
        `~psfutils.models.FittableImageModel2D.origin` will show fitted
        center of the star. If `update_flux` was `True`, the
        `~psfutils.models.FittableImageModel2D.flux`
        model parameter will contain fitted flux and the original star's
        flux otherwise.

    """
    if not hasattr(stars, '__iter__'):
        stars = [stars]

    if len(stars) == 0:
        return []

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # analize psf_fit_box:
    snx = [s.nx for s in all_stars]
    sny = [s.ny for s in all_stars]
    minfbx = min(snx)
    minfby = min(sny)

    if psf_fit_box is None:
        # use full grid defined by stars' data size:
        psf_fit_box = (minfbx, minfby)

    elif hasattr(psf_fit_box, '__iter__'):
        if len(psf_fit_box) != 2:
            raise ValueError("'psf_fit_box' must be a tuple of two integers, "
                             "a single integer, or None")

        psf_fit_box = (min(minfbx, psf_fit_box[0]),
                       min(minfby, psf_fit_box[0]))

    else:
        psf_fit_box = min(minfbx, minfby, psf_fit_box)
        psf_fit_box = (psf_fit_box, psf_fit_box)

    # create grid for fitting box (in stars' grid units):
    width, height = psf_fit_box
    width = int(py2round(width))
    height = int(py2round(height))
    igy, igx = np.indices((height, width), dtype=np.float)

    # perform fitting for each star:
    fitted_stars = []

    # initialize fitter (if needed):
    if isinstance(fitter, type):
        fit = fitter()
    else:
        fit = fitter

    # remove fitter's keyword arguments that we set ourselves:
    rem_kwd = ['x', 'y', 'z', 'weights']
    fitter_kwargs = copy.deepcopy(fitter_kwargs)
    for k in rem_kwd:
        if k in fitter_kwargs:
            del fitter_kwargs[k]

    fitter_has_fit_info = hasattr(fit, 'fit_info')

    # make a copy of the original PSF:
    psf = psf.copy()

    for st in stars:
        cst = _fit_star(st, psf, fit, fitter_kwargs,
                        fitter_has_fit_info, residuals,
                        width, height, igx, igy)

        # also fit stars linked to the left:
        lnks = st.prev
        while lnks is not None:
            lnkcst = _fit_star(lnks, psf, fit, fitter_kwargs,
                               fitter_has_fit_info, residuals,
                               width, height, igx, igy)
            cst.append_first(lnkcst)
            lnks = lnks.prev

        # ... and fit stars linked to the right:
        lnks = st.next
        while lnks is not None:
            lnkcst = _fit_star(lnks, psf, fit, fitter_kwargs,
                               fitter_has_fit_info, residuals,
                               width, height, igx, igy)
            cst.append_last(lnkcst)
            lnks = lnks.next

        cst.constrain_linked_centers(ignore_badfit_stars=True)
        fitted_stars.append(cst)

    return fitted_stars


def _fit_star(star, psf, fit, fit_kwargs, fitter_has_fit_info, residuals,
              width, height, igx, igy):
    # NOTE: input PSF may be modified by this function. Make a copy if
    #       it is important to preserve input model.

    err = 0
    ovx = star.pixel_scale[0] / psf.pixel_scale[0]
    ovy = star.pixel_scale[1] / psf.pixel_scale[1]
    ny, nx = star.shape

    rxc = int(py2round(star.x_center))
    ryc = int(py2round(star.y_center))

    x1 = rxc - (width - 1) // 2
    x2 = x1 + width
    y1 = ryc - (height - 1) // 2
    y2 = y1 + height

    # check boundaries of the fitting box:
    if x1 < 0:
        i1 = -x1
        x1 = 0

    else:
        i1 = 0

    if x2 > nx:
        i2 = width - (x2 - nx)
        x2 = nx

    else:
        i2 = width

    if y1 < 0:
        j1 = -y1
        y1 = 0

    else:
        j1 = 0

    if y2 > ny:
        j2 = height - (y2 - ny)
        y2 = ny

    else:
        j2 = height

    # initial guess for fitted flux and shifts:
    psf.flux = star.flux
    psf.x_0 = 0.0
    psf.y_0 = 0.0

    if rxc < 0 or rxc > (nx - 1) or ryc < 0 or ryc > (ny - 1):
        # star's center is outside the extraction box
        err = 1
        fit_info = None
        fitted_psf = psf
        warnings.warn("Source with coordinates ({}, {}) is being ignored "
                      "because its center is outside the image."
                      .format(star.x_center, star.y_center))

    elif (i2 - i1) < 3 or (j2 - j1) < 3:
        # star's center is too close to the edge of the star's image:
        err = 2
        fit_info = None
        fitted_psf = psf
        warnings.warn("Source with coordinates ({}, {}) is being ignored "
                      "because there are too few pixels available around "
                      "its center pixel.".format(star.x_center, star.y_center))

    else:
        # define PSF sampling grid:
        gx = (igx[j1:j2, i1:i2] - (star.x_center - x1)) * ovx
        gy = (igy[j1:j2, i1:i2] - (star.y_center - y1)) * ovy

        # fit PSF to the star:
        scaled_data = star.data[y1:y2, x1:x2] / (ovx * ovy)
        if star.weights is None:
            # a separate treatment for the case when fitters
            # do not support weights (star's models must not have
            # weights set in such cases)
            fitted_psf = fit(model=psf, x=gx, y=gy, z=scaled_data,
                             **fit_kwargs)

        else:
            wght = star.weights[y1:y2, x1:x2]
            fitted_psf = fit(model=psf, x=gx, y=gy, z=scaled_data,
                             weights=wght, **fit_kwargs)

        if fitter_has_fit_info:
            # TODO: this treatment of fit info (fit error info) may not be
            # compatible with other fitters. This code may need revising.
            fit_info = fit.fit_info
            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                err = 3

        else:
            fit_info = None

    # compute correction to the star's position and flux:
    cst = copy.deepcopy(star)
    cst.x_center += fitted_psf.x_0.value / ovx
    cst.y_center += fitted_psf.y_0.value / ovy

    # set "measured" star's flux based on fitted ePSF:
    cst.flux = fitted_psf.flux.value

    if residuals:
        cst.fit_residual = _calc_res(fitted_psf, cst)

    cst.fit_info = fit_info
    cst.fit_error_status = err
    return cst


def iter_build_psf(stars, psf=None, peak_fit_box=5, peak_search_box='fitbox',
                   recenter_accuracy=1.0e-4, recenter_nmax=1000,
                   ignore_badfit_stars=True,
                   stat='median', nclip=10, lsig=3.0, usig=3.0, ker='quar',
                   psf_fit_box=5, fitter=fitting.LevMarLSQFitter,
                   fitter_kwargs={}, residuals=True,
                   max_iter=50, accuracy=1e-4):

    """
    Iteratively build the empirical PSF (ePSF) using input stars and then
    improve star position estimates by fitting this ePSF to stars. The process
    is repeated until stop conditions are met.

    Fundamentally, each iteration consists of first calling
    :py:func:`build_psf` to build an improved (after each iteration) PSF model
    followed by a call to :py:func:`fit_stars` to improve estimates of stars'
    centers and fluxes. Thisprocess is repeated until either estimates of the
    centers of the stars do not change by more than the value specified by
    parameter `accuracy` or until maximum number of iterations specified by
    `max_iter` is achieved.

    Thus, this function most of its parameters in common with
    :py:func:`build_psf` and :py:func:`fit_stars`. Below we describe only new
    parameters.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of PSF build / star fitting iterations to be performed.

    accuracy : float, optional
        Stop iterations when change of stars' centers between two iterations
        is smalled that indicated.


    Returns
    -------

    ePSF : PSF2DModel
        A 2D normalized fittable image model of the ePSF.

    fitted_stars : list of FittableImageModel2D
        A list of `~psfutils.models.FittableImageModel2D` of stars with model
        parameters `~psfutils.models.FittableImageModel2D.x_0` and
        `~psfutils.models.FittableImageModel2D.y_0` set to 0 and
        `~psfutils.models.FittableImageModel2D.origin` will show fitted
        center of the star. If `update_flux` was `True`, the
        `~psfutils.models.FittableImageModel2D.flux`
        model parameter will contain fitted flux and the original star's
        flux otherwise.

    niter : int
        Number of performed iterations.

    """
    nmax = int(max_iter)
    if nmax < 0:
        raise ValueError("'max_iter' must be non-negative")
    if accuracy <= 0.0:
        raise ValueError("'accuracy' must be a positive number")
    acc2 = accuracy**2

    # initialize fitter (if needed):
    if isinstance(fitter, type):
        fitter = fitter()

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # create an array of star centers:
    prev_centers = np.asarray([s.center for s in stars], dtype=np.float)

    # initialize array for detection of scillatory behavior:
    oscillatory = np.zeros((len(all_stars), ), dtype=np.bool)
    prev_failed = np.zeros((len(all_stars), ), dtype=np.bool)

    niter = -1
    eps2 = 2.0 * acc2
    dxy = np.zeros((len(all_stars), 2), dtype=np.float)

    while niter < nmax and np.amax(eps2) >= acc2 and not np.all(oscillatory):
        niter += 1

        # improved PSF:
        psf = build_psf(
            stars=stars,
            psf=psf,
            peak_fit_box=peak_fit_box,
            peak_search_box=peak_search_box,
            recenter_accuracy=recenter_accuracy,
            recenter_nmax=recenter_nmax,
            ignore_badfit_stars=ignore_badfit_stars,
            stat=stat,
            nclip=nclip,
            lsig=lsig,
            usig=usig,
            ker=ker
        )

        # improved fit of the PSF to stars:
        stars = fit_stars(
            stars=stars,
            psf=psf,
            psf_fit_box=psf_fit_box,
            fitter=fitter,
            fitter_kwargs=fitter_kwargs,
            residuals=False
        )

        # get all stars including linked stars as a flat list:
        all_stars = []
        for s in stars:
            all_stars += s.get_linked_list()

        # create an array of star centers at this iteration:
        centers = np.asarray([s.center for s in stars], dtype=np.float)

        # detect oscillatory behavior
        failed = np.array([s.fit_error_status > 0 for s in stars],
                          dtype=np.bool)
        if niter > 2:  # allow a few iterations at the beginning
            oscillatory = np.logical_and(prev_failed, np.logical_not(failed))
            for s, osc in zip(stars, oscillatory):
                s.ignore = bool(osc)
            prev_failed = failed

        # check termination criterion:
        good_mask = np.logical_not(np.logical_or(failed, oscillatory))
        dxy = centers - prev_centers
        mdxy = dxy[good_mask]
        eps2 = np.sum(mdxy * mdxy, axis=1, dtype=np.float64)
        prev_centers = centers

    # compute residuals:
    if residuals:
        res = compute_residuals(psf, all_stars)
    else:
        res = len(all_stars) * [None]

    for s, r in zip(all_stars, res):
        s.fit_residual = r

    # assign coordinate residuals of the iterative process:
    for s, (dx, dy) in zip(all_stars, dxy):
        s.iter_fit_eps = (float(dx), float(dy))

    return (psf, stars, niter)


def _pixstat(data, stat='mean', nclip=0, lsig=3.0, usig=3.0, default=np.nan):
    if nclip > 0:
        if lsig is None or usig is None:
            raise ValueError("When 'nclip' > 0 neither 'lsig' nor 'usig' "
                             "may be None")
    data = np.ravel(data)
    nd, = data.shape

    if nd == 0:
        return default

    m = np.mean(data, dtype=np.float64)

    if nd == 1:
        return m

    need_std = (stat != 'mean' or nclip > 0)
    if need_std:
        s = np.std(data, dtype=np.float64)

    i = np.ones(nd, dtype=np.bool)

    for x in range(nclip):
        m_prev = m
        s_prev = s
        nd_prev = nd

        # sigma clipping:
        lval = m - lsig * s
        uval = m + usig * s
        i = ((data >= lval) & (data <= uval))
        d = data[i]
        nd, = d.shape
        if nd < 1:
            # return statistics based on previous iteration
            break

        m = np.mean(d, dtype=np.float64)
        s = np.std(d, dtype=np.float64)

        if nd == nd_prev:
            # NOTE: we could also add m == m_prev and s == s_prev
            # NOTE: a more rigurous check would be needed to see if
            #       index array 'i' did not change but that would be too slow
            #       and the current check is very likely good enough.
            break

    if stat == 'mean':
        return m
    elif stat == 'median':
        return np.median(data[i])
    elif stat == 'pmode1':
        return (2.5 * np.median(data[i]) - 1.5 * m)
    elif stat == 'pmode2':
        return (3.0 * np.median(data[i]) - 2.0 * m)
    else:
        raise ValueError("Unsupported 'stat' value")


def _smoothPSF(psf, kernel):
    if kernel is None:
        return psf
    if kernel == 'quad':
        ker = _kernel_quad
    elif kernel == 'quar':
        ker = _kernel_quar
    elif isinstance(kernel, numpy.ndarray) or isinstance(kernel, Kernel):
        ker = kernel
    else:
        raise TypeError("Unsupported kernel.")

    spsf = convolve(psf, ker)

    return spsf


def _calc_res(psf, star):
    ovx = star.pixel_scale[0] / psf.pixel_scale[0]
    ovy = star.pixel_scale[1] / psf.pixel_scale[1]
    gy, gx = np.indices((star.ny, star.nx), dtype=np.float)
    gx = ovx * (gx - star.x_center)
    gy = ovy * (gy - star.y_center)
    psfval = psf.evaluate(gx, gy, flux=1.0, x_0=0.0, y_0=0.0)
    return (star.data - star.flux * (ovx * ovy) * psfval)


def compute_residuals(psf, stars):
    """
    Register the ``psf`` to intput ``stars`` and compute the difference.

    Parameters
    ----------
    psf : FittableImageModel2D
        Model of the PSF.

    stars : Star, list of Star
        A single :py:class:`~psfutils.catalogs.Star` object or a list of stars
        for which resuduals need to be computed.

    Returns
    -------
    res : numpy.ndarray, list of numpy.ndarray
        A list of `numpy.ndarray` of residuals when input is a list of
        :py:class:`~psfutils.catalogs.Star` objects and a single
        `numpy.ndarray` when input is a single
        :py:class:`~psfutils.catalogs.Star` object.

    """
    if isinstance(stars, Star):
        res = _calc_res(psf, star)
        return res

    else:
        res = []
        for s in stars:
            res.append(_calc_res(psf, s))

    return res


def _parse_tuple_pars(par, default=None, name='', dtype=None,
                      check_positive=True):
    if par is None:
        par = default

    if hasattr(par, '__iter__'):
        if len(par) != 2:
            raise TypeError("Parameter '{:s}' must be either a scalar or an "
                            "iterable with two elements.".format(name))
        px = par[0]
        py = par[1]
    elif par is None:
        return (None, None)
    else:
        px = par
        py = par

    if dtype is not None or check_positive:
        try:
            pxf = dtype(px)
            pyf = dtype(px)
        except TypeError:
            raise TypeError("Parameter '{:s}' must be a number or a tuple of "
                            "numbers.".format(name))

        if dtype is not None:
            px = pxf
            py = pyf

        if check_positive and (pxf <= 0 or pyf <= 0):
            raise TypeError("Parameter '{:s}' must be a strictly positive "
                            "number or a tuple of strictly positive numbers."
                            .format(name))

    return (px, py)
