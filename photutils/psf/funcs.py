# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models and functions for doing PSF/PRF fitting photometry on image data.
"""

from __future__ import division
import warnings
import copy
import numpy as np
from astropy.table import Table, Column
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import add_array
from astropy.nddata import support_nddata
from astropy.utils.exceptions import AstropyUserWarning
from ..aperture.core import _prepare_photometry_input
from ..extern.nddata_compat import extract_array


__all__ = ['psf_photometry', 'subtract_psf']


def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from
    a model.  Returns (xname, yname, fluxname)
    """

    if hasattr(psf, 'psf_xname'):
        xname = psf.psf_xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'psf_yname'):
        yname = psf.psf_yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'psf_fluxname'):
        fluxname = psf.psf_fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname


def _call_fitter(fitter, psf, x, y, data, weights):
    """
    Not all fitters have to support a weight array. This function
    includes the weight in the fitter call only if really needed.
    """

    if np.all(weights == 1.):
        return fitter(psf, x, y, data)
    else:
        return fitter(psf, x, y, data, weights=weights)


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


@support_nddata
def psf_photometry(data, positions, psf, fitshape=None,
                   fitter=LevMarLSQFitter(), unit=None, wcs=None, error=None,
                   mask=None, pixelwise_error=True, mode='sequential',
                   store_fit_info=False, param_uncert=False):
    """
    Perform PSF/PRF photometry on the data.

    Given a PSF or PRF model, the model is fitted simultaneously or
    sequentially to the given positions to obtain an estimate of the
    flux. If required, coordinates are also tuned to match best the
    data.

    Parameters
    ----------
    data : array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        The 2-d array on which to perform photometry. ``data`` should be
        background-subtracted.  Units are used during the photometry,
        either provided along with the data array, or stored in the
        header keyword ``'BUNIT'``.
    positions : array-like of shape (2 or 3, N) or `~astropy.table.Table`
        Positions at which to *start* the fit for each object, in pixel
        coordinates. If array-like, it can be either (x_0, y_0) or (x_0,
        y_0, flux_0). If a table, the columns 'x_0' and 'y_0' must be
        present.  'flux_0' can also be provided to set initial fluxes.
        Additional columns of the form '<parametername>_0' will be used
        to set the initial guess for any parameters of the ``psf`` model
        that are not fixed.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This function needs to identify three parameters
        (position of center in x and y coordinates and the flux) in
        order to set them to suitable starting values for each fit. The
        names of these parameters can be given as follows:

        - Set ``psf.psf_xname``, ``psf.psf_yname`` and
          ``psf.psf_fluxname`` to strings with the names of the respective
          psf model parameter.
        - If those attributes are not found, the names ``x_0``, ``y_0``
          and ``flux`` are assumed.

        `~photutils.psf.prepare_psf_model` can be used to prepare any 2D
        model to match these assumptions.
    fitshape : length-2 or None
        The shape of the region around the center of the target location
        to do the fitting in.  If None, fit the whole image without
        windowing. (See notes)
    fitter : an `astropy.modeling.fitting.Fitter` object
        The fitter object used to actually derive the fits. See
        `astropy.modeling.fitting` for more details on fitters.
    unit : `~astropy.units.UnitBase` instance, str
        An object that represents the unit associated with ``data``.
        Must be an `~astropy.units.UnitBase` object or a string
        parseable by the :mod:`~astropy.units` package. It overrides the
        ``data`` unit from the ``'BUNIT'`` header keyword and issues a
        warning if different. However an error is raised if ``data`` as
        an array already has a different unit.
    wcs : `~astropy.wcs.WCS`, optional
        Use this as the wcs transformation. It overrides any wcs
        transformation passed along with ``data`` either in the header
        or in an attribute.
    error : float or array_like, optional
        The pixel-wise Gaussian 1-sigma errors of the input ``data``.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.
    mask : array_like (bool), optional
        Mask to apply to the data.  Masked pixels are excluded/ignored.
    pixelwise_error : bool, optional
        If `True`, assume ``error`` varies significantly across the PSF
        and sum contribution from each pixel. If `False`, assume
        ``error`` does not vary significantly across the PSF and use the
        single value of ``error`` at the center of each PSF.  Default is
        `True`.
    mode : {'sequential'}
        One of the following modes to do PSF/PRF photometry:
            * 'sequential' (default)
                Fit PSF/PRF separately for the given positions.
            * (No other modes are yet implemented)
    store_fit_info : bool or list
        If False, the fitting information is discarded.  If True, the
        output table will have an additional column 'fit_message' with
        the message that came from the fit.  If a list, it will be
        populated with the ``fit_info`` dictionary of the fitter for
        each fit.
    param_uncert : bool (default=False)
        If True, the uncertainties on each parameter estimate will be
        stored in the output table. This option assumes that the fitter
        has the 'param_cov' key in its 'fit_info' dictionary.  See
        'fit_info' in `~astropy.modeling.fitting.LevMarLSQFitter`.

    Returns
    -------
    result_tab : `~astropy.table.Table`
        The results of the fitting procedure.  The fitted flux is in the
        column 'flux_fit', and the centroids are in 'x_fit' and 'y_fit'.
        If ``positions`` was a table, any columns in that table will be
        carried over to this table.  If any of the ``psf`` model
        parameters other than flux/x/y are not fixed, their results will
        be in the column '<parametername>_fit'.

    Notes
    -----
    Most fitters will not do well if ``fitshape`` is None because they
    will try to fit the whole image as just one star.

    This function is decorated with `~astropy.nddata.support_nddata` and
    thus supports `~astropy.nddata.NDData` objects as input.
    """

    (data, wcs_transformation, mask, error, pixelwise_error) = (
        _prepare_photometry_input(data, unit, wcs, mask, error,
                                  pixelwise_error))

    # As long as models don't support quantities, we'll break that apart
    fluxunit = data.unit
    data = np.array(data)

    if (error is not None):
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
            raise ValueError('Positions should be a table or an array (2, N) '
                             'or (3, N)')
        elif positions.shape[0] > 3:
            raise ValueError('Positions should be a table or an array (2, N) '
                             'or (3, N)')

        result_tab = Table()
        result_tab['x_0'] = positions[0]
        result_tab['y_0'] = positions[1]
        if positions.shape[0] >= 3:
            result_tab['flux_0'] = positions[2]

    result_tab['x_fit'] = result_tab['x_0']
    result_tab['y_fit'] = result_tab['y_0']
    result_tab.add_column(Column(name='flux_fit', unit=fluxunit,
                                 data=np.empty(len(result_tab),
                                               dtype=data.dtype)))

    # prep for fitting
    psf = psf.copy()  # don't want to muck up whatever PSF the user gives us

    # maps input table name to parameter name
    pars_to_set = {'x_0': xname, 'y_0': yname}
    if 'flux_0' in result_tab.colnames:
        pars_to_set['flux_0'] = fluxname

    # maps output table name to parameter name
    pars_to_output = {'x_fit': xname,
                      'y_fit': yname,
                      'flux_fit': fluxname}

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
    if param_uncert:
        if 'param_cov' in fitter.fit_info:
            uncert = []
        else:
            warnings.warn('uncertainties on fitted parameters cannot be '
                          'computed because fitter does not contain '
                          '`param_cov` key in its `fit_info` dictionary.',
                          AstropyUserWarning)
            param_uncert = False

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
            if param_uncert:
                if fitter.fit_info['param_cov'] is not None:
                    uncert.append(np.sqrt(np.diag(
                        fitter.fit_info['param_cov'])))
                else:
                    warnings.warn('uncertainties on fitted parameters '
                                  'cannot be computed because the fit may '
                                  'be unsuccessful', AstropyUserWarning)
                    uncert.append((None, None, None))
    else:
        raise ValueError('Invalid photometry mode.')

    if fit_messages is not None:
        result_tab['fit_messages'] = fit_messages
    if param_uncert:
        uncert = np.array(uncert)
        i = 0
        for param in psf.param_names:
            if not getattr(psf, param).fixed:
                result_tab.add_column(Column(name=param + "_fit_uncertainty",
                                             unit=None, data=uncert[:, i]))
                i += 1

    return result_tab


def subtract_psf(data, psf, posflux, subshape=None):
    """
    Subtract PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array,
        it is interpreted as ``(x, y, flux)``  If a table, the columns
        'x_fit', 'y_fit', and 'flux_fit' must be present.
    subshape : length-2 or None
        The shape of the region around the center of the location to
        subtract the PSF from.  If None, subtract from the whole image.

    Returns
    -------
    subdata : same shape and type as ``data``
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
