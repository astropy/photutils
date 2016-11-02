# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for doing PSF/PRF fitting photometry on image data.
"""

from __future__ import division
import numpy as np
from astropy.modeling import models, Parameter, Fittable2DModel


__all__ = ['IntegratedGaussianPRF', 'PRFAdapter', 'prepare_psf_model',
           'get_grouped_psf_model']


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
