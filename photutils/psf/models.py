# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for doing PSF/PRF fitting photometry on image data.
"""

from __future__ import division
import warnings
import numpy as np
from astropy.modeling import models, Parameter, Fittable2DModel

__all__ = ['FittableImageModel2D', 'IntegratedGaussianPRF', 'PRFAdapter',
           'prepare_psf_model']

__all__ = ['IntegratedGaussianPRF', 'PRFAdapter', 'prepare_psf_model',
           'get_grouped_psf_model']

class FittableImageModel2D(Fittable2DModel):
    """
    A fittable 2D model of an image allowing for image intensity scaling
    and image translations.

    This class stores normalized 2D image data and computes the values of
    the model at arbitrary locations (including at intra-pixel,
    fractional positions) within this image using spline interpolation
    provided by :py:class:`~scipy.interpolate.RectBivariateSpline`.

    The fittable model provided by this class has three model parameters:
    an image intensity scaling factor (`flux`) which is applied to
    normalized image, and two positional parameters (`x_0` and `y_0`)
    indicating the location of a feature in the coordinate grid on which
    the model is to be evaluated.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing 2D image.

    origin : tuple, None, optional
        A reference point in the input image `data` array.

        If `origin` represents the location of a feature (e.g., the position
        of an intensity peak) in the input `data`, then model parameters
        `x_0` and `y_0` show the location of this peak in an another target
        image to which this model was fitted.

        Alternatively, when `origin` is set to ``(0,0)``, then model parameters
        `x_0` and `y_0` are shifts by which model's image should be translated
        in order to match a target image.

    normalization_const : float, None, optional
        Normalization constant by which to normalize input image data. If it is
        `None`, then the normalization constant will be computed such that
        ``sum(normalization_const * data)`` is 1.

        Assuming model's data represent a PSF to be fitted to some star,
        `normalization_const` could be used to account for aperture correction.
        In this case, best fitted value of the `flux` model parameter will
        represent an aperture-corrected flux of the star.

    fillval : float, optional
        The value to be returned by the `evaluate` or
        ``astropy.modeling.Model.__call__`` methods
        when evaluation is performed outside the definition domain of the
        model.

    kwargs : dict, optional

        Additional optional keyword arguments to be passed directly to the
        `compute_interpolator` method. See `compute_interpolator` for more
        details.

    """
    flux = Parameter(description='Intensity scaling factor of normalized '
                     'image data.', default=1.0)
    x_0 = Parameter(description='X-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)
    y_0 = Parameter(description='Y-position of a feature in the image in '
                    'the output coordinate grid on which the model is '
                    'evaluated.', default=0.0)

    def __init__(self, data, flux=flux.default,
                 x_0=x_0.default, y_0=y_0.default,
                 origin=None, normalization_const=None,
                 fillval=0.0, kwargs={}):
        self._fillval = fillval

        data = np.asarray(data)

        if not np.all(np.isfinite(data)):
            raise ValueError("All elements of input 'data' must be finite.")

        if normalization_const is None:
            # compute normalizization constant so that
            # sum(normalized_data) = 1:
            tflux = np.sum(data, dtype=np.float64)
            if tflux == 0.0 or not np.isfinite(tflux):
                warnings.warn("Overflow encountered while computing "
                              "normalization constant. Normalization constant "
                              "will be set to 1.", RuntimeWarning)
                tflux = 1.0

            normalization_const = 1.0 / tflux

        else:
            if normalization_const == 0.0:
                raise ValueError("'normalizing_constant' must be non-zero.")

        self._normalized_data = normalization_const * data

        # set input image related parameters:
        self._ny, self._nx = data.shape
        self._shape = data.shape

        # set the origin of the coordinate system in image's pixel grid:
        self.origin = origin

        super(FittableImageModel2D, self).__init__(flux, x_0, y_0)

        # define interpolating spline:
        self.compute_interpolator(kwargs)

    @property
    def normalized_data(self):
        """ Normalized image data defined as ``normalization_const * data``."""
        return self._normalized_data

    @property
    def intensity_scaled_data(self):
        """ Intensity-scaled by a factor of `flux` normalized image data."""
        return (self.flux.value * self._normalized_data)

    @property
    def shape(self):
        """A tuple of dimensions of the data array in numpy style (ny, nx)."""
        return self._shape

    @property
    def nx(self):
        """Number of columns in the data array."""
        return self._nx

    @property
    def ny(self):
        """Number of rows in the data array."""
        return self._ny

    @property
    def origin(self):
        """
        A tuple of ``x`` and ``y`` coordinates of the origin of the coordinate
        system in terms of pixels of model's image.

        When setting the coordinate system origin, a tuple of two `int` or
        `float` may be used. If origin is set to `None`, the origin of the
        coordinate system will be set to the middle of the data array
        (``(npix-1)/2.0``).

        .. warning::
            Modifying `origin` will not adjust (modify) model's parameters
            `x_0` and `y_0`.
        """
        return (self._x_origin, self._y_origin)

    @origin.setter
    def origin(self, origin):
        if origin is None:
            self._x_origin = (self._nx - 1) / 2.0
            self._y_origin = (self._ny - 1) / 2.0
        elif hasattr(origin, '__iter__') and len(origin) == 2:
            self._x_origin, self._y_origin = origin
        else:
            raise TypeError("Parameter 'origin' must be either None or an "
                            "iterable with two elements.")

    @property
    def x_origin(self):
        """X-coordinate of the origin of the coordinate system."""
        return self._x_origin

    @property
    def y_origin(self):
        """Y-coordinate of the origin of the coordinate system."""
        return self._y_origin

    @property
    def fillval(self):
        """Fill value to be returned for coordinates outside of the domain of
        definition of the interpolator.

        """
        return self._fillval

    @fillval.setter
    def fillval(self, fillval):
        self._fillval = fillval

    def compute_interpolator(self, kwargs={}):
        """
        Compute/define the interpolating spline. This function can be overriden
        in a subclass to define custom interpolators.

        .. note::
            When subclassing :py:class:`FittableImageModel2D` for the purpose of
            overriding :py:func:`compute_interpolator`, the :py:func:`evaluate`
            may need to overriden as well depending on the behavior of the
            new interpolator.

        .. note::
            Use caution when modifying interpolator's degree or smoothness in a
            computationally intensive part of the code as it may decrease code
            performance due to the need to recompute interpolator.

        Parameters
        ----------
        kwargs : dict, optional

            Additional optional keyword arguments. Possible values are:

            - **degree** : int, tuple, optional
                Degree of the interpolating spline. A tuple can be used to
                provide different degrees for the X- and Y-axes.
                Default value is degree=3.

            - **s** : float, optional
                Non-negative smoothing factor. Default value s=0 corresponds to
                interpolation.
                See :py:class:`~scipy.interpolate.RectBivariateSpline` for more
                details.

        """
        from scipy.interpolate import RectBivariateSpline

        if 'degree' in kwargs:
            degree = kwargs['degree']
            if hasattr(degree, '__iter__') and len(degree) == 2:
                degx = int(degree[0])
                degy = int(degree[1])
            else:
                degx = int(degree)
                degy = int(degree)
            if degx < 0 or degy < 0:
                raise ValueError("Interpolator degree must be a non-negative "
                                 "integer")
        else:
            degx = 3
            degy = 3

        if 's' in kwargs:
            smoothness = kwargs['s']
        else:
            smoothness = 0

        x = np.arange(self._nx, dtype=np.float)
        y = np.arange(self._ny, dtype=np.float)
        self.interpolator = RectBivariateSpline(
            x, y, self._normalized_data.T, kx=degx, ky=degx, s=smoothness
        )

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the model on some input variables and provided model
        parameters.

        """
        xi = np.asarray(x, dtype=np.float) + (self._x_origin - x_0)
        yi = np.asarray(y, dtype=np.float) + (self._y_origin - y_0)

        evaluated_model = flux * self.interpolator.ev(xi, yi)

        if self._fillval is not None:
            # find indices of pixels that are outside the input pixel grid and
            # set these pixels to the 'fillval':
            invalid = (((xi < 0) | (xi > self._nx - 1)) |
                       ((yi < 0) | (yi > self._ny - 1)))
            evaluated_model[invalid] = self._fillval

        return evaluated_model


class IntegratedGaussianPRF(Fittable2DModel):
    r"""
    Circular Gaussian model integrated over pixels. Because it is
    integrated, this model is considered a PRF, *not* a PSF (see
    :ref:`psf-terminology` for more about the terminology used here.)

    This model is a Gaussian *integrated* over an area of ``1`` (in
    units of the model input coordinates, e.g. 1 pixel).  This is in
    contrast to the apparently similar
    `astropy.modeling.functional_models.Gaussian2D`, which is the value
    of a 2D Gaussian *at* the input coordinates, with no integration.
    So this model is equivalent to assuming the PSF is Gaussian at a
    *sub-pixel* level.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian PSF.
    flux : float (default 1)
        Total integrated flux over the entire PSF
    x_0 : float (default 0)
        Position of the peak in x direction.
    y_0 : float (default 0)
        Position of the peak in y direction.

    Notes
    -----
    This model is evaluated according to the following formula:

        .. math::

            f(x, y) =
                \frac{F}{4}
                \left[
                {\rm erf} \left(\frac{x - x_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{x - x_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]
                \left[
                {\rm erf} \left(\frac{y - y_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{y - y_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]

    where ``erf`` denotes the error function and ``F`` the total
    integrated flux.
    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    sigma = Parameter(default=1, fixed=True)

    _erf = None
    fit_deriv = None

    @property
    def bounding_box(self):
        halfwidth = 4 * self.sigma
        return ((int(self.y_0 - halfwidth), int(self.y_0 + halfwidth)),
                (int(self.x_0 - halfwidth), int(self.x_0 + halfwidth)))

    def __init__(self, sigma=sigma.default,
                 x_0=x_0.default, y_0=y_0.default, flux=flux.default,
                 **kwargs):
        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super(IntegratedGaussianPRF, self).__init__(n_models=1, sigma=sigma,
                                                    x_0=x_0, y_0=y_0,
                                                    flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """Model function Gaussian PSF model."""

        return (flux / 4 *
                ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) *
                 (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))


class PRFAdapter(Fittable2DModel):
    """
    A model that adapts a supplied PSF model to act as a PRF. It
    integrates the PSF model over pixel "boxes".  A critical built-in
    assumption is that the PSF model scale and location parameters are
    in *pixel* units.

    Parameters
    ----------
    psfmodel : a 2D model
        The model to assume as representative of the PSF
    renormalize_psf : bool
        If True, the model will be integrated from -inf to inf and
        re-scaled so that the total integrates to 1.  Note that this
        renormalization only occurs *once*, so if the total flux of
        ``psfmodel`` depends on position, this will *not* be correct.
    xname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        x-axis center of the PSF.  If None, the model will be assumed to
        be centered at x=0.
    yname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        y-axis center of the PSF.  If None, the model will be assumed to
        be centered at y=0.
    fluxname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        total flux of the star.  If None, a scaling factor will be
        applied by the ``PRFAdapter`` instead of modifying the
        ``psfmodel``.

    Notes
    -----
    This current implementation of this class (using numerical
    integration for each pixel) is extremely slow, and only suited for
    experimentation over relatively few small regions.
    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)

    def __init__(self, psfmodel, renormalize_psf=True, flux=flux.default,
                 x_0=x_0.default, y_0=y_0.default, xname=None, yname=None,
                 fluxname=None, **kwargs):

        self.psfmodel = psfmodel.copy()

        if renormalize_psf:
            from scipy.integrate import dblquad
            self._psf_scale_factor = 1. / dblquad(self.psfmodel,
                                                  -np.inf, np.inf,
                                                  lambda x: -np.inf,
                                                  lambda x: np.inf)[0]
        else:
            self._psf_scale_factor = 1

        self.xname = xname
        self.yname = yname
        self.fluxname = fluxname

        # these can be used to adjust the integration behavior. Might be
        # used in the future to expose how the integration happens
        self._dblquadkwargs = {}

        super(PRFAdapter, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                         flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0):
        """The evaluation function for PRFAdapter."""

        if self.xname is None:
            dx = x - x_0
        else:
            dx = x
            setattr(self.psfmodel, self.xname, x_0)

        if self.xname is None:
            dy = y - y_0
        else:
            dy = y
            setattr(self.psfmodel, self.yname, y_0)

        if self.fluxname is None:
            return (flux * self._psf_scale_factor *
                    self._integrated_psfmodel(dx, dy))
        else:
            setattr(self.psfmodel, self.yname, flux * self._psf_scale_factor)
            return self._integrated_psfmodel(dx, dy)

    def _integrated_psfmodel(self, dx, dy):
        from scipy.integrate import dblquad

        # infer type/shape from the PSF model.  Seems wasteful, but the
        # integration step is a *lot* more expensive so its just peanuts
        out = np.empty_like(self.psfmodel(dx, dy))
        outravel = out.ravel()
        for i, (xi, yi) in enumerate(zip(dx.ravel(), dy.ravel())):
            outravel[i] = dblquad(self.psfmodel,
                                  xi-0.5, xi+0.5,
                                  lambda x: yi-0.5, lambda x: yi+0.5,
                                  **self._dblquadkwargs)[0]
        return out


def prepare_psf_model(psfmodel, xname=None, yname=None, fluxname=None,
                      renormalize_psf=True):
    """
    Convert a 2D PSF model to one suitable for use with
    `psf_photometry`.

    The resulting model may be a composite model, but should have only
    the x, y, and flux related parameters un-fixed.

    Parameters
    ----------
    psfmodel : a 2D model
        The model to assume as representative of the PSF.
    xname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        x-axis center of the PSF.  If None, the model will be assumed to
        be centered at x=0, and a new parameter will be added for the
        offset.
    yname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        y-axis center of the PSF.  If None, the model will be assumed to
        be centered at x=0, and a new parameter will be added for the
        offset.
    fluxname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        total flux of the star.  If None, a scaling factor will be added
        to the model.
    renormalize_psf : bool
        If True, the model will be integrated from -inf to inf and
        re-scaled so that the total integrates to 1.  Note that this
        renormalization only occurs *once*, so if the total flux of
        ``psfmodel`` depends on position, this will *not* be correct.

    Returns
    -------
    outmod : a model
        A new model ready to be passed into `psf_photometry`.
    """

    if xname is None:
        xinmod = models.Shift(0, name='x_offset')
        xname = 'offset_0'
    else:
        xinmod = models.Identity(1)
        xname = xname + '_2'
    xinmod.fittable = True

    if yname is None:
        yinmod = models.Shift(0, name='y_offset')
        yname = 'offset_1'
    else:
        yinmod = models.Identity(1)
        yname = yname + '_2'
    yinmod.fittable = True

    outmod = (xinmod & yinmod) | psfmodel

    if fluxname is None:
        outmod = outmod * models.Const2D(1, name='flux_scaling')
        fluxname = 'amplitude_3'
    else:
        fluxname = fluxname + '_2'

    if renormalize_psf:
        # we do the import here because other machinery works w/o scipy
        from scipy import integrate

        integrand = integrate.dblquad(psfmodel, -np.inf, np.inf,
                                      lambda x: -np.inf, lambda x: np.inf)[0]
        normmod = models.Const2D(1./integrand, name='renormalize_scaling')
        outmod = outmod * normmod

    # final setup of the output model - fix all the non-offset/scale
    # parameters
    for pnm in outmod.param_names:
        outmod.fixed[pnm] = pnm not in (xname, yname, fluxname)

    # and set the names so that psf_photometry knows what to do
    outmod.xname = xname
    outmod.yname = yname
    outmod.fluxname = fluxname

    # now some convenience aliases if reasonable
    outmod.psfmodel = outmod[2]
    if 'x_0' not in outmod.param_names and 'y_0' not in outmod.param_names:
        outmod.x_0 = getattr(outmod, xname)
        outmod.y_0 = getattr(outmod, yname)
    if 'flux' not in outmod.param_names:
        outmod.flux = getattr(outmod, fluxname)

    return outmod


def get_grouped_psf_model(template_psf_model, star_group):
    """
    Construct a joint PSF model which consists of a sum of PSF's templated on
    a specific model, but whose parameters are given by a table of objects.

    Parameters
    ----------
    template_psf_model : `astropy.modeling.Fittable2DModel` instance
        The model to use for *individual* objects.  Must have parameters named
        ``x_0``, ``y_0``, and ``flux``.
    star_group : `~astropy.table.Table`
        Table of stars for which the compound PSF will be constructed.  It
        must have columns named ``x_0``, ``y_0``, and ``flux_0``.

    Returns
    -------
    group_psf
        An `astropy.modeling` ``CompoundModel`` instance which is a sum of the
        given PSF models.
    """

    group_psf = None
    for i in range(len(star_group)):
        psf_to_add = template_psf_model.copy()
        psf_to_add.flux = star_group['flux_0'][i]
        psf_to_add.x_0 = star_group['x_0'][i]
        psf_to_add.y_0 = star_group['y_0'][i]

        if group_psf is None:
            # this is the first one only
            group_psf = psf_to_add
        else:
            group_psf += psf_to_add

    return group_psf
