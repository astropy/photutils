# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Models and functions for doing PSF/PRF fitting photometry on image data."""

from __future__ import division

import warnings
import copy

import numpy as np

from .utils import mask_to_mirrored_num
from .aperture_core import _prepare_photometry_input

from astropy.table import Table, Column
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Parameter, Fittable2DModel
from astropy.nddata.utils import add_array, subpixel_indices
from .extern.nddata_compat import extract_array
from astropy.nddata import support_nddata
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['DiscretePRF', 'IntegratedGaussianPRF', 'psf_photometry',
           'subtract_psf']


class DiscretePRF(Fittable2DModel):
    """
    A discrete Pixel Response Function (PRF) model.

    The discrete PRF model stores images of the PRF at different subpixel
    positions or offsets as a lookup table. The resolution is given by the
    subsampling parameter, which states in how many subpixels a pixel is
    divided.

    In the typical case of wanting to create a PRF from an image with many point
    sources, use the `~DiscretePRF.create_from_image` method, rather than
    directly initializing this class.

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
        Normalize PRF images to unity.  Equivalent to saying there is *no* flux
        outside the bounds of the PRF images.
    subsampling : int, optional
        Factor of subsampling. Default = 1.

    Notes
    -----
    See :ref:`psf-terminology` for more details on the distinction between PSF
    and PRF as used in this module.
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
        """
        Shape of the PRF image.
        """
        return self._prf_array.shape[-2:]

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Discrete PRF model evaluation.

        Given a certain position and flux the corresponding image of
        the PSF is chosen and scaled to the flux. If x and y are
        outside the boundaries of the image, zero will be returned.

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
    def create_from_image(cls, imdata, positions, size, fluxes=None, mask=None,
                          mode='mean', subsampling=1, fix_nan=False):
        """
        Create a discrete point response function (PRF) from image data.

        Given a list of positions and size this function estimates an image of
        the PRF by extracting and combining the individual PRFs from the given
        positions.

        NaN values are either ignored by passing a mask or can be replaced by
        the mirrored value with respect to the center of the PRF.

        Note that if fluxes are *not* specified explicitly, it will be flux
        estimated from an aperture of the same size as the PRF image. This does
        *not* account for aperture corrections so often will *not* be what you
        want for anything other than quick-look needs.

        Parameters
        ----------
        imdata : array
            Data array with the image to extract the PRF from
        positions : List or array or `~astropy.table.Table`
            List of pixel coordinate source positions to use in creating the PRF.
            If this is a `~astropy.table.Table` it must have columns called
            ``x_0`` and ``y_0``.
        size : odd int
            Size of the quadratic PRF image in pixels.
        mask : bool array, optional
            Boolean array to mask out bad values.
        fluxes : array, optional
            Object fluxes to normalize extracted PRFs. If not given (or None),
            the flux is estimated from an aperture of the same size as
            the PRF image.
        mode : {'mean', 'median'}
            One of the following modes to combine the extracted PRFs:
                * 'mean'
                    Take the pixelwise mean of the extracted PRFs.
                * 'median'
                    Take the pixelwise median of the extracted PRFs.
        subsampling : int
            Factor of subsampling of the PRF (default = 1).
        fix_nan : bool
            Fix NaN values in the data by replacing it with the
            mirrored value. Assuming that the PRF is symmetrical.

        Returns
        -------
        prf : `photutils.psf.DiscretePRF`
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
            raise TypeError("Position and flux arrays must be of equal length.")

        if mask is None:
            mask = np.isnan(imdata)

        if isinstance(positions, (list, tuple)):
            positions = np.array(positions)

        if isinstance(positions, Table) or \
            (isinstance(positions, np.ndarray) and positions.dtype.names is not None):
            # Can do clever things like
            # positions['x_0', 'y_0'].as_array().view((positions['x_0'].dtype, 2))
            # but that requires  positions['x_0'].dtype is positions['y_0'].dtype
            # better do something simple to allow type promotion if required.
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
        positions_subpixel_indices = np.array([subpixel_indices(_, subsampling)
                                               for _ in positions], dtype=np.int)

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
    Circular Gaussian model integrated over pixels. Because it is integrated,
    this model is considered a PRF, *not* a PSF (see :ref:`psf-terminology` for
    more about the terminology used here.)

    This model is a Gaussian *integrated* over an area of ``1`` (in units
    of the model input coordinates).  This is in contrast to the apparently
    similar `astropy.modeling.functional_models.Gaussian2D`, which is the value
    of a 2D Gaussian *at* the input coordinates, with no integration.  So this
    model is equivalent to assuming the *sub-pixel* PSF is Gaussian.

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
        """
        Model function Gaussian PSF model.
        """
        return (flux / 4 *
                ((self._erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) *
                 (self._erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) -
                  self._erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))


class PRFAdapter(Fittable2DModel):
    """
    A model that adapts a supplied PSF model to actas a PRF.  I.e., this
    integrates the PSF model over pixel "boxes".  A critical built-in assumption
    is that the PSF model scale and location parameters are in *pixel* units.

    Parameters
    ----------
    psfmodel : a 2D model
        The model to assume as representative of the PSF
    renormalize_psf : bool
        If True, the model will be integrated from -inf to inf and re-scaled
        so that the total integrates to 1.  Note that this renormalization only
        occurs *once*, so if the total flux of ``psfmodel`` depends on position,
        this will *not* be correct.
    xname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the x-axis
        center of the PSF.  If None, the model will be assumed to be centered at x=0.
    yname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the y-axis
        center of the PSF.  If None, the model will be assumed to be centered at y=0.
    fluxname : str or None
        The name of the ``psfmodel`` parameter that corresponds to the total
        flux of the star.  If None, a scaling factor will be applied by the
        ``PRFAdapter`` instead of modifying the ``psfmodel``.

    Notes
    -----
    This current implementation of this class (using numberical integration for
    each pixel) is extremely slow, and only suited for experimentation over
    relatively few small regions.
    """
    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)

    def __init__(self, psfmodel, renormalize_psf=True, flux=flux.default,
                 x_0=x_0.default, y_0=y_0.default, xname=None, yname=None,
                 fluxname=None, **kwargs):

        self.psfmodel = psfmodel.copy()

        if renormalize_psf:
            from scipy import integrate
            self._psf_scale_factor = 1/integrate.dblquad(self.psfmodel,
                                                         -np.inf, np.inf,
                                                         lambda x: -np.inf,
                                                         lambda x: np.inf)[0]
        else:
            self._psf_scale_factor = 1

        self.xname = xname
        self.yname = yname
        self.fluxname = fluxname

        # these can be used to adjust the integration behavior. Might be used
        # in the future to expose how the integration happens
        self._dblquadkwargs = {}

        super(PRFAdapter, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                         flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        The evaluation function for PRFAdapter.
        """
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
            return flux * self._psf_scale_factor * self._integrated_psfmodel(dx, dy)
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


def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from a
    model.  Returns (xname, yname, fluxname)
    """
    if hasattr(psf, 'psf_xname'):
        xname = psf.psf_xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for psf_photometry.')

    if hasattr(psf, 'psf_yname'):
        yname = psf.psf_yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for psf_photometry.')

    if hasattr(psf, 'psf_fluxname'):
        fluxname = psf.psf_fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname


def _call_fitter(fitter, psf, x, y, data, weights):
    '''Not all fitters have to support a weight array. This function includes
    the weight in the fitter call only if really needed.'''
    if np.all(weights == 1.):
        return fitter(psf, x, y, data)
    else:
        return fitter(psf, x, y, data, weights=weights)


@support_nddata
def psf_photometry(data, positions, psf, fitshape=None,
                   fitter=LevMarLSQFitter(),
                   unit=None, wcs=None, error=None, effective_gain=None,
                   mask=None, pixelwise_error=True,
                   mode='sequential',
                   store_fit_info=False):
    """
    Perform PSF/PRF photometry on the data.

    Given a PSF or PRF model, the model is fitted simultaneously or
    sequentially to the given positions to obtain an estimate of the
    flux. If required, coordinates are also tuned to match best the data.

    Parameters
    ----------
    data : array_like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        The 2-d array on which to perform photometry. ``data`` should be
        background-subtracted.  Units are used during the photometry,
        either provided along with the data array, or stored in the
        header keyword ``'BUNIT'``.
    positions : Array-like of shape (2 or 3, N) or `~astropy.table.Table`
        Positions at which to *start* the fit, in pixel coordinates. If
        array-like, it can be either (x_0, y_0) or (x_0, y_0, flux_0). If a
        table, the columns 'x_0' and 'y_0' must be present.  'flux_0' can also
        be provided to set initial fluxes.  Additional columns of the form
        '<parametername>_0' will be used to set the initial guess for any
        parameters of the ``psf`` model that are not fixed.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit to the data. Examples for such models are
        `photutils.psf.DiscretePRF` or `photutils.psf.GaussianPSF`.
    fitshape : length-2 or None
        The shape of the region around the center of the target location to do
        the fitting in.  If None, fit the whole image without windowing. (See
        notes)
    fitter : an `astropy.modeling.fitting.Fitter` object
        The fitter object used to actually derive the fits. See
        `astropy.modeling.fitting` for more details on fitters.
    unit : `~astropy.units.UnitBase` instance, str
        An object that represents the unit associated with ``data``.  Must
        be an `~astropy.units.UnitBase` object or a string parseable by the
        :mod:`~astropy.units` package. It overrides the ``data`` unit from
        the ``'BUNIT'`` header keyword and issues a warning if
        different. However an error is raised if ``data`` as an array
        already has a different unit.
    wcs : `~astropy.wcs.WCS`, optional
        Use this as the wcs transformation. It overrides any wcs transformation
        passed along with ``data`` either in the header or in an attribute.
    error : float or array_like, optional
        Error in each pixel, interpreted as Gaussian 1-sigma uncertainty.
    effective_gain : float or array_like, optional
        Ratio of counts (e.g., electrons or photons) to units of the
        data (e.g., ADU), for the purpose of calculating Poisson error
        from the object itself. If ``effective_gain`` is `None`
        (default), ``error`` is assumed to include all uncertainty in
        each pixel. If ``effective_gain`` is given, ``error`` is assumed
        to be the "background error" only (not accounting for Poisson
        error in the flux in the apertures).
    mask : array_like (bool), optional
        Mask to apply to the data.  Masked pixels are excluded/ignored.
    pixelwise_error : bool, optional
        For ``error`` and/or ``effective_gain`` arrays. If `True`,
        assume ``error`` and/or ``effective_gain`` vary significantly
        within an aperture: sum contribution from each pixel. If
        `False`, assume ``error`` and ``effective_gain`` do not vary
        significantly within an aperture. Use the single value of
        ``error`` and/or ``effective_gain`` at the center of each
        aperture as the value for the entire aperture.  Default is
        `True`.

    mode : {'sequential'}
        One of the following modes to do PSF/PRF photometry:
            * 'sequential' (default)
                Fit PSF/PRF separately for the given positions.
            * (No other modes are yet implemented)
    store_fit_info : bool or list
        If False, the fitting information is discarded.  If True, the output
        table will have an additional column 'fit_message' with the message that
        came from the fit.  If a list, it will be populated with the
        ``fit_info`` dictionary of the fitter for each fit.

    Returns
    -------
    result_tab : `~astropy.table.Table`
        The results of the fitting procedure.  The fitted flux is in the column
        'flux_fit', and the centroids are in 'x_fit' and 'y_fit'. If `positions`
        was a table, any columns in that table will be carried over to this
        table.  If any of the ``psf`` model parameters other than flux/x/y are
        not fixed, their results will be in the column '<parametername>_fit'.

    Notes
    -----
    Most fitters will not do well if ``fitshape`` is None because they will try
    to fit the whole image as just one star.

    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.

    Examples
    --------
    See `Spitzer PSF Photometry <http://nbviewer.ipython.org/gist/adonath/
    6550989/PSFPhotometrySpitzer.ipynb>`_ for a short tutorial.
    """
    data, wcs_transformation, mask, error, effective_gain, pixelwise_error \
        = _prepare_photometry_input(data, unit, wcs, mask, error, effective_gain,
                                    pixelwise_error)

    # As long as models don't support quantities, we'll break that apart
    fluxunit = data.unit
    data = np.array(data)

    if (error is not None) or (effective_gain is not None):
        warnings.warn('Uncertainties are not yet supported in PSF fitting.',
                      AstropyUserWarning)
    weights = np.ones_like(data)

    # determine the names of the model's relevant attributes
    xname, yname, fluxname = _extract_psf_fitting_names(psf)

    # Prep the index arrays and the table for output
    indices = np.indices(data.shape)
    if hasattr(positions, 'colnames'):
        # quacks like a table, so assume it's a table
        if 'x_0' not in positions.colnames:
            raise ValueError('Input table does not have x0 column')
        if 'y_0' not in positions.colnames:
            raise ValueError('Input table does not have y0 column')
        result_tab = positions.copy()
    else:
        positions = np.array(positions, copy=False)
        if positions.shape[0] < 2:
            raise ValueError('Positions should be a table or an array (2, N) or (3, N)')
        elif positions.shape[0] > 3:
            raise ValueError('Positions should be a table or an array (2, N) or (3, N)')

        result_tab = Table()
        result_tab['x_0'] = positions[0]
        result_tab['y_0'] = positions[1]
        if positions.shape[0] >= 3:
            result_tab['flux_0'] = positions[2]

    result_tab['x_fit'] = result_tab['x_0']
    result_tab['y_fit'] = result_tab['y_0']
    result_tab.add_column(Column(name='flux_fit', unit=fluxunit,
                                 data=np.empty(len(result_tab), dtype=data.dtype)))

    # prep for fitting
    psf = psf.copy()  # don't want to muck up whatever PSF the user gives us

    pars_to_set = {'x_0': xname, 'y_0': yname}  # maps input table name to parameter name
    if 'flux_0' in result_tab.colnames:
        pars_to_set['flux_0'] = fluxname

    pars_to_output = {'x_fit': xname,
                      'y_fit': yname,
                      'flux_fit': fluxname}  # maps output table name to parameter name

    for p, isfixed in psf.fixed.items():
        p0 = p + '_0'
        if p0 in result_tab.colnames and p not in (xname, yname, fluxname):
            pars_to_set[p0] = p
        pfit = p + '_fit'
        if not isfixed and p not in (xname, yname, fluxname):
                pars_to_output[pfit] = p

    fit_messages = None
    fit_infos = None
    if isinstance(store_fit_info, list):
        fit_infos = store_fit_info
    elif store_fit_info:
        fit_messages = []

    # Many fitters take a "weight" array, but no "mask".
    # Thus, we convert the mask to weights on 1 and 0. Unfortunately,
    # that only works if the values "behind the mask" are finite.
    if mask is not None:
        data = copy.deepcopy(data)
        data[mask] = 0
        weights[mask] = 0.

    if mode == 'sequential':
        for row in result_tab:
            for table_name, parameter_name in pars_to_set.items():
                setattr(psf, parameter_name, row[table_name])

            if fitshape is None:
                fitted = _call_fitter(fitter, psf, indices[1], indices[0],
                                      data, weights=weights)
            else:
                position = (row['y_0'], row['x_0'])
                y = extract_array(indices[0], fitshape, position)
                x = extract_array(indices[1], fitshape, position)
                sub_array_data = extract_array(data, fitshape, position,
                                               fill_value=0.)
                sub_array_weights = extract_array(weights, fitshape,
                                                  position, fill_value=0.)
                fitted = _call_fitter(fitter, psf, x, y, sub_array_data,
                                      weights=sub_array_weights)

            for table_name, parameter_name in pars_to_output.items():
                row[table_name] = getattr(fitted, parameter_name).value

            if fit_infos is not None:
                fit_infos.append(fitter.fit_info)
            if fit_messages is not None:
                fit_messages.append(fitter.fit_info['message'])
    else:
        raise ValueError('Invalid photometry mode.')

    if fit_messages is not None:
        result_tab['fit_messages'] = fit_messages

    return result_tab


def subtract_psf(data, psf, posflux, subshape=None):
    """
    Subtracts PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array, it is
        interpreted as ``(x, y, flux)``  If a table, the columns 'x_fit',
        'y_fit', and 'flux_fit' must be present.
    subshape : length-2 or None
        The shape of the region around the center of the location to subtract
        the PSF from.  If None, subtract from the whole image.

    Returns
    -------
    subdata : same as `data`
        The image with the PSF subtracted
    """
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. Only 2-d arrays can be '
                         'passed to subtract_psf.'.format(data.ndim))

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = Table(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up contstants across the loop
    psf = psf.copy()
    xname, yname, fluxname = _extract_psf_fitting_names(psf)
    indices = np.indices(data.shape)
    subbeddata = data.copy()

    if subshape is None:
        indicies_reversed = indices[::-1]

        for row in posflux:
            getattr(psf, xname).value = row['x_fit']
            getattr(psf, yname).value = row['y_fit']
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata -= psf(*indicies_reversed)
    else:
        for row in posflux:
            x_0, y_0 = row['x_fit'], row['y_fit']

            y = extract_array(indices[0], subshape, (y_0, x_0))
            x = extract_array(indices[1], subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))

    return subbeddata
