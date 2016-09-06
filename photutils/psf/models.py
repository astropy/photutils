# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for doing PSF/PRF fitting photometry on image data.
"""

from __future__ import division
import numpy as np
from astropy.table import Table
from astropy.modeling import models, Parameter, Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import subpixel_indices
from ..utils import mask_to_mirrored_num
from ..extern.nddata_compat import extract_array


__all__ = ['IntegratedGaussianPRF', 'PRFAdapter', 'prepare_psf_model']


class DiscretePRF(Fittable2DModel):
    """
    A discrete Pixel Response Function (PRF) model.

    The discrete PRF model stores images of the PRF at different
    subpixel positions or offsets as a lookup table. The resolution is
    given by the subsampling parameter, which states in how many
    subpixels a pixel is divided.

    In the typical case of wanting to create a PRF from an image with
    many point sources, use the `~DiscretePRF.create_from_image` method,
    rather than directly initializing this class.

    The discrete PRF model class in initialized with a 4 dimensional
    array, that contains the PRF images at different subpixel positions.
    The definition of the axes is as following:

        1. Axis: y subpixel position
        2. Axis: x subpixel position
        3. Axis: y direction of the PRF image
        4. Axis: x direction of the PRF image

    The total array therefore has the following shape
    (subsampling, subsampling, prf_size, prf_size)

    Parameters
    ----------
    prf_array : ndarray
        Array containing PRF images.
    normalize : bool
        Normalize PRF images to unity.  Equivalent to saying there is
        *no* flux outside the bounds of the PRF images.
    subsampling : int, optional
        Factor of subsampling. Default = 1.

    Notes
    -----
    See :ref:`psf-terminology` for more details on the distinction
    between PSF and PRF as used in this module.
    """

    flux = Parameter('flux')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, prf_array, normalize=True, subsampling=1):
        # Array shape and dimension check
        if subsampling == 1:
            if prf_array.ndim == 2:
                prf_array = np.array([[prf_array]])
        if prf_array.ndim != 4:
            raise TypeError('Array must have 4 dimensions.')
        if prf_array.shape[:2] != (subsampling, subsampling):
            raise TypeError('Incompatible subsampling and array size')
        if np.isnan(prf_array).any():
            raise Exception("Array contains NaN values. Can't create PRF.")

        # Normalize if requested
        if normalize:
            for i in range(prf_array.shape[0]):
                for j in range(prf_array.shape[1]):
                    prf_array[i, j] /= prf_array[i, j].sum()

        # Set PRF asttributes
        self._prf_array = prf_array
        self.subsampling = subsampling

        constraints = {'fixed': {'x_0': True, 'y_0': True}}
        x_0 = 0
        y_0 = 0
        flux = 1
        super(DiscretePRF, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                          flux=flux, **constraints)
        self.fitter = LevMarLSQFitter()

    @property
    def prf_shape(self):
        """Shape of the PRF image."""

        return self._prf_array.shape[-2:]

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Discrete PRF model evaluation.

        Given a certain position and flux the corresponding image of the
        PSF is chosen and scaled to the flux. If x and y are outside the
        boundaries of the image, zero will be returned.

        Parameters
        ----------
        x : float
            x coordinate array in pixel coordinates.
        y : float
            y coordinate array in pixel coordinates.
        flux : float
            Model flux.
        x_0 : float
            x position of the center of the PRF.
        y_0 : float
            y position of the center of the PRF.
        """

        # Convert x and y to index arrays
        x = (x - x_0 + 0.5 + self.prf_shape[1] // 2).astype('int')
        y = (y - y_0 + 0.5 + self.prf_shape[0] // 2).astype('int')

        # Get subpixel indices
        y_sub, x_sub = subpixel_indices((y_0, x_0), self.subsampling)

        # Out of boundary masks
        x_bound = np.logical_or(x < 0, x >= self.prf_shape[1])
        y_bound = np.logical_or(y < 0, y >= self.prf_shape[0])
        out_of_bounds = np.logical_or(x_bound, y_bound)

        # Set out of boundary indices to zero
        x[x_bound] = 0
        y[y_bound] = 0
        result = flux * self._prf_array[int(y_sub), int(x_sub)][y, x]

        # Set out of boundary values to zero
        result[out_of_bounds] = 0
        return result

    @classmethod
    def create_from_image(cls, imdata, positions, size, fluxes=None,
                          mask=None, mode='mean', subsampling=1,
                          fix_nan=False):
        """
        Create a discrete point response function (PRF) from image data.

        Given a list of positions and size this function estimates an
        image of the PRF by extracting and combining the individual PRFs
        from the given positions.

        NaN values are either ignored by passing a mask or can be
        replaced by the mirrored value with respect to the center of the
        PRF.

        Note that if fluxes are *not* specified explicitly, it will be
        flux estimated from an aperture of the same size as the PRF
        image. This does *not* account for aperture corrections so often
        will *not* be what you want for anything other than quick-look
        needs.

        Parameters
        ----------
        imdata : array
            Data array with the image to extract the PRF from
        positions : List or array or `~astropy.table.Table`
            List of pixel coordinate source positions to use in creating
            the PRF.  If this is a `~astropy.table.Table` it must have
            columns called ``x_0`` and ``y_0``.
        size : odd int
            Size of the quadratic PRF image in pixels.
        mask : bool array, optional
            Boolean array to mask out bad values.
        fluxes : array, optional
            Object fluxes to normalize extracted PRFs. If not given (or
            None), the flux is estimated from an aperture of the same
            size as the PRF image.
        mode : {'mean', 'median'}
            One of the following modes to combine the extracted PRFs:
                * 'mean':  Take the pixelwise mean of the extracted PRFs.
                * 'median':  Take the pixelwise median of the extracted PRFs.
        subsampling : int
            Factor of subsampling of the PRF (default = 1).
        fix_nan : bool
            Fix NaN values in the data by replacing it with the mirrored
            value. Assuming that the PRF is symmetrical.

        Returns
        -------
        prf : `photutils.psf.sandbox.DiscretePRF`
            Discrete PRF model estimated from data.
        """

        # Check input array type and dimension.
        if np.iscomplexobj(imdata):
            raise TypeError('Complex type not supported')
        if imdata.ndim != 2:
            raise ValueError('{0}-d array not supported. '
                             'Only 2-d arrays supported.'.format(imdata.ndim))
        if size % 2 == 0:
            raise TypeError("Size must be odd.")

        if fluxes is not None and len(fluxes) != len(positions):
            raise TypeError('Position and flux arrays must be of equal '
                            'length.')

        if mask is None:
            mask = np.isnan(imdata)

        if isinstance(positions, (list, tuple)):
            positions = np.array(positions)

        if isinstance(positions, Table) or \
            (isinstance(positions, np.ndarray) and
             positions.dtype.names is not None):
            # One can do clever things like
            # positions['x_0', 'y_0'].as_array().view((positions['x_0'].dtype,
            #                                          2))
            # but that requires positions['x_0'].dtype is
            # positions['y_0'].dtype.
            # Better do something simple to allow type promotion if required.
            pos = np.empty((len(positions), 2))
            pos[:, 0] = positions['x_0']
            pos[:, 1] = positions['y_0']
            positions = pos

        if isinstance(fluxes, (list, tuple)):
            fluxes = np.array(fluxes)

        if mode == 'mean':
            combine = np.ma.mean
        elif mode == 'median':
            combine = np.ma.median
        else:
            raise Exception('Invalid mode to combine prfs.')

        data_internal = np.ma.array(data=imdata, mask=mask)
        prf_model = np.ndarray(shape=(subsampling, subsampling, size, size))
        positions_subpixel_indices = \
            np.array([subpixel_indices(_, subsampling) for _ in positions],
                     dtype=np.int)

        for i in range(subsampling):
            for j in range(subsampling):
                extracted_sub_prfs = []
                sub_prf_indices = np.all(positions_subpixel_indices == [j, i],
                                         axis=1)
                positions_sub_prfs = positions[sub_prf_indices]
                for k, position in enumerate(positions_sub_prfs):
                    x, y = position
                    extracted_prf = extract_array(data_internal, (size, size),
                                                  (y, x))
                    # Check shape to exclude incomplete PRFs at the boundaries
                    # of the image
                    if (extracted_prf.shape == (size, size) and
                            np.ma.sum(extracted_prf) != 0):
                        # Replace NaN values by mirrored value, with respect
                        # to the prf's center
                        if fix_nan:
                            prf_nan = extracted_prf.mask
                            if prf_nan.any():
                                if (prf_nan.sum() > 3 or
                                        prf_nan[size // 2, size // 2]):
                                    continue
                                else:
                                    extracted_prf = mask_to_mirrored_num(
                                        extracted_prf, prf_nan,
                                        (size // 2, size // 2))
                        # Normalize and add extracted PRF to data cube
                        if fluxes is None:
                            extracted_prf_norm = (np.ma.copy(extracted_prf) /
                                                  np.ma.sum(extracted_prf))
                        else:
                            fluxes_sub_prfs = fluxes[sub_prf_indices]
                            extracted_prf_norm = (np.ma.copy(extracted_prf) /
                                                  fluxes_sub_prfs[k])
                        extracted_sub_prfs.append(extracted_prf_norm)
                    else:
                        continue
                prf_model[i, j] = np.ma.getdata(
                    combine(np.ma.dstack(extracted_sub_prfs), axis=2))
        return cls(prf_model, subsampling=subsampling)


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
        be centered at x=0, and a new paramter will be added for the
        offset.
    yname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the
        y-axis center of the PSF.  If None, the model will be assumed to
        be centered at x=0, and a new paramter will be added for the
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
