# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides utilities for PSF-fitting photometry.
"""

import numpy as np
from astropy.modeling.models import Const2D, Identity, Shift
from astropy.nddata import NDData
from astropy.nddata.utils import add_array, extract_array
from astropy.table import QTable
from astropy.utils.decorators import deprecated

__all__ = ['prepare_psf_model', 'get_grouped_psf_model', 'subtract_psf',
           'grid_from_epsfs']


class _InverseShift(Shift):
    @staticmethod
    def evaluate(x, offset):
        return x - offset

    @staticmethod
    def fit_deriv(x, *params):
        """
        One dimensional Shift model derivative with respect to parameter.
        """
        d_offset = -np.ones_like(x)
        return [d_offset]


def prepare_psf_model(psfmodel, *, xname=None, yname=None, fluxname=None,
                      renormalize_psf=True):
    """
    Convert a 2D PSF model to one suitable for use with
    `BasicPSFPhotometry` or its subclasses.

    .. note::

        This function is needed only in special cases where the PSF
        model does not have ``x_0``, ``y_0``, and ``flux`` model
        parameters. In particular, it is not needed for any of the PSF
        models provided by photutils (e.g., `~photutils.psf.EPSFModel`,
        `~photutils.psf.IntegratedGaussianPRF`,
        `~photutils.psf.FittableImageModel`,
        `~photutils.psf.GriddedPSFModel`, etc).

    Parameters
    ----------
    psfmodel : `~astropy.modeling.Fittable2DModel`
        The model to assume as representative of the PSF.

    xname : `str` or `None`, optional
        The name of the ``psfmodel`` parameter that corresponds to the
        x-axis center of the PSF. If `None`, the model will be assumed
        to be centered at x=0, and a new parameter will be added for the
        offset.

    yname : `str` or `None`, optional
        The name of the ``psfmodel`` parameter that corresponds to the
        y-axis center of the PSF. If `None`, the model will be assumed
        to be centered at y=0, and a new parameter will be added for the
        offset.

    fluxname : `str` or `None`, optional
        The name of the ``psfmodel`` parameter that corresponds to the
        total flux of the star. If `None`, a scaling factor will be
        added to the model.

    renormalize_psf : bool, optional
        If `True`, the model will be integrated from -inf to inf and
        rescaled so that the total integrates to 1. Note that this
        renormalization only occurs *once*, so if the total flux of
        ``psfmodel`` depends on position, this will *not* be correct.

    Returns
    -------
    result : `~astropy.modeling.Fittable2DModel`
        A new model ready to be passed into `BasicPSFPhotometry` or its
        subclasses.
    """
    if xname is None:
        xinmod = _InverseShift(0, name='x_offset')
        xname = 'offset_0'
    else:
        xinmod = Identity(1)
        xname = xname + '_2'
    xinmod.fittable = True

    if yname is None:
        yinmod = _InverseShift(0, name='y_offset')
        yname = 'offset_1'
    else:
        yinmod = Identity(1)
        yname = yname + '_2'
    yinmod.fittable = True

    outmod = (xinmod & yinmod) | psfmodel.copy()

    if fluxname is None:
        outmod = outmod * Const2D(1, name='flux_scaling')
        fluxname = 'amplitude_3'
    else:
        fluxname = fluxname + '_2'

    if renormalize_psf:
        # we do the import here because other machinery works w/o scipy
        from scipy import integrate

        integrand = integrate.dblquad(psfmodel, -np.inf, np.inf,
                                      lambda x: -np.inf, lambda x: np.inf)[0]
        normmod = Const2D(1.0 / integrand, name='renormalize_scaling')
        outmod = outmod * normmod

    # final setup of the output model - fix all the non-offset/scale
    # parameters
    for pnm in outmod.param_names:
        outmod.fixed[pnm] = pnm not in (xname, yname, fluxname)

    # and set the names so that BasicPSFPhotometry knows what to do
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


def _interpolate_missing_data(data, mask, method='cubic'):
    """
    Interpolate missing data as identified by the ``mask`` keyword.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        An array containing the 2D image.

    mask : 2D bool `~numpy.ndarray`
        A 2D boolean mask array with the same shape as the input
        ``data``, where a `True` value indicates the corresponding
        element of ``data`` is masked.  The masked data points are
        those that will be interpolated.

    method : {'cubic', 'nearest'}, optional
        The method of used to interpolate the missing data:

        * ``'cubic'``:  Masked data are interpolated using 2D cubic
            splines.  This is the default.

        * ``'nearest'``:  Masked data are interpolated using
            nearest-neighbor interpolation.

    Returns
    -------
    data_interp : 2D `~numpy.ndarray`
        The interpolated 2D image.
    """
    from scipy import interpolate

    data_interp = np.array(data, copy=True)

    if len(data_interp.shape) != 2:
        raise ValueError("'data' must be a 2D array.")

    if mask.shape != data.shape:
        raise ValueError("'mask' and 'data' must have the same shape.")

    y, x = np.indices(data_interp.shape)
    xy = np.dstack((x[~mask].ravel(), y[~mask].ravel()))[0]
    z = data_interp[~mask].ravel()

    if method == 'nearest':
        interpol = interpolate.NearestNDInterpolator(xy, z)
    elif method == 'cubic':
        interpol = interpolate.CloughTocher2DInterpolator(xy, z)
    else:
        raise ValueError('Unsupported interpolation method.')

    xy_missing = np.dstack((x[mask].ravel(), y[mask].ravel()))[0]
    data_interp[mask] = interpol(xy_missing)

    return data_interp


@deprecated('1.9.0')
def get_grouped_psf_model(template_psf_model, star_group, pars_to_set):
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

    pars_to_set : `dict`
        A dictionary of parameter names and values to set.

    Returns
    -------
    group_psf
        An `astropy.modeling` ``CompoundModel`` instance which is a sum of the
        given PSF models.
    """
    group_psf = None

    for index, star in enumerate(star_group):
        psf_to_add = template_psf_model.copy()
        # we 'tag' the model here so that later we don't have to rely
        # on possibly mangled names of the compound model to find
        # the parameters again
        psf_to_add.name = index
        for param_tab_name, param_name in pars_to_set.items():
            setattr(psf_to_add, param_name, star[param_tab_name])

        if group_psf is None:
            # this is the first one only
            group_psf = psf_to_add
        else:
            group_psf = group_psf + psf_to_add

    return group_psf


@deprecated('1.9.0')
def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from
    a model.  Returns (xname, yname, fluxname)
    """
    if hasattr(psf, 'xname'):
        xname = psf.xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'yname'):
        yname = psf.yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'fluxname'):
        fluxname = psf.fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname


@deprecated('1.9.0')
def subtract_psf(data, psf, posflux, *, subshape=None):
    """
    Subtract PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.

    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be subtracted from the data.

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
        raise ValueError(f'{data.ndim}-d array not supported. Only 2-d '
                         'arrays can be passed to subtract_psf.')

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = QTable(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up constants across the loop
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

            # float dtype needed for fill_value=np.nan
            y = extract_array(indices[0].astype(float), subshape, (y_0, x_0))
            x = extract_array(indices[1].astype(float), subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))

    return subbeddata


def grid_from_epsfs(epsfs, grid_xypos=None, meta=None):
    """
    Create a GriddedPSFModel from a list of EPSFModels.

    Given a list of EPSFModels, this function will return a GriddedPSFModel.
    The fiducial points for each input EPSFModel can either be set on each
    individual model by setting the 'x_0' and 'y_0' attributes, or provided as
    a list of tuples (``grid_xypos``). If a ``grid_xypos`` list is provided, it
    must match the length of input EPSFs. In either case, the fiducial points
    must be on a grid.

    Optionally, a ``meta`` dictionary may be provided for the
    output GriddedPSFModel. If this dictionary contains the keys 'grid_xypos',
    'oversampling', or 'fill_value', they will be overridden.

    Note: If set on the input EPSFModel (x_0, y_0), then ``origin`` must be the
    same for each input EPSF. Additionally data units and dimensions must be
    for each input EPSF, and values for ``flux`` and ``oversampling``, and
    ``fill_value`` must match as well.

    Parameters
    ----------
    epsfs : list of `photutils.psf.models.EPSFModel`
        A list of EPSFModels representing the individual PSFs.
    grid_xypos : list, optional
        A list of fiducial points (x_0, y_0) for each PSF.
        If not provided, the x_0 and y_0 of each input EPSF will be considered
        the fiducial point for that PSF. Default is None.
    meta : dict, optional
        Additional metadata for the GriddedPSFModel. Note that, if they exist
        in the supplied ``meta``, any values under the keys ``grid_xypos`` ,
        ``oversampling``, or ``fill_value`` will be overridden. Default is None.

    Returns
    -------
    GriddedPSFModel: `photutils.psf.GriddedPSFModel`
        The gridded PSF model created from the input EPSFs.
    """
    # prevent circular imports
    from photutils.psf import EPSFModel, GriddedPSFModel

    x_0s = []  # optional, to store fiducial from input if `grid_xypos` is None
    y_0s = []
    data_arrs = []
    oversampling = None
    fill_value = None
    dat_unit = None
    origin = None
    flux = None

    # make sure, if provided, that ``grid_xypos`` is the same length as ``epsfs``
    if grid_xypos is not None:
        if len(grid_xypos) != len(epsfs):
            raise ValueError('``grid_xypos`` must be the same length as ``epsfs``.')

    # loop over input once
    for i, epsf in enumerate(epsfs):

        # check input type
        if not isinstance(epsf, EPSFModel):
            raise ValueError('All input `epsfs` must be of type '
                             '`photutils.psf.models.EPSFModel`.')

        # get data array from EPSF
        data_arrs.append(epsf.data)

        if i == 0:
            # EPSFModel allows a tuple for oversampling factor in x, y,
            # but GriddedPSFModel requires it to be a single scalar value.
            # Keep this condition for now by checking that x and y match
            if np.isscalar(epsf.oversampling):
                oversampling = epsf.oversampling
            else:
                if epsf.oversampling[0] != epsf.oversampling[1]:
                    raise ValueError('Oversampling must be the same in x and y.')
                oversampling = epsf.oversampling[0]

            # same for fill value and flux, grid will have a single value
            # so it should be the same for all input, and error if not.
            fill_value = epsf.fill_value

            # check that origins are the same
            if grid_xypos is None:
                origin = epsf.origin

            flux = epsf.flux

            # if theres a unit, those should also all be the same
            try:
                dat_unit = epsf.data.unit
            except AttributeError:
                pass  # just keep as None

        else:
            if np.isscalar(epsf.oversampling):
                if epsf.oversampling != oversampling:
                    raise ValueError('All input EPSFModels must have the same '
                                     'value for ``oversampling``.')
                else:
                    if epsf.oversampling[0] != epsf.oversampling[1] != oversampling:
                        raise ValueError('All input EPSFModels must have the '
                                         'same value for ``oversampling``.')

            if epsf.fill_value != fill_value:
                raise ValueError('All input EPSFModels must have the same value '
                                 'for ``fill_value``.')

            if epsf.data.ndim != data_arrs[0].ndim:
                raise ValueError('All input EPSFModels must have data with the '
                                 'same dimensions.')

            try:
                unitt = epsf.data_unit
                if unitt != dat_unit:
                    raise ValueError('All input data must have the same unit.')
            except AttributeError:
                if dat_unit is not None:
                    raise ValueError('All input data must have the same unit.')

            if epsf.flux != flux:
                raise ValueError('All input EPSFModels must have the same value '
                                 'for ``flux``.')

        if grid_xypos is None:  # get gridxy_pos from x_0, y_0 if not provided
            x_0s.append(epsf.x_0.value)
            y_0s.append(epsf.y_0.value)

            # also check that origin is the same, if using x_0s and y_0s from input
            if epsf.origin != origin:
                raise ValueError('If using ``x_0``, ``y_0`` as fiducial point,'
                                 '``origin`` must match for each input EPSF.')

    # if not supplied, use from x_0, y_0 of input EPSFs as fiducuals
    # these are checked when GriddedPSFModel is created to make sure they
    # are actually on a grid.
    if grid_xypos is None:
        grid_xypos = list(zip(x_0s, y_0s))

    data_cube = np.stack(data_arrs, axis=0)

    if meta is None:
        meta = {}
    # add required keywords to meta
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = oversampling
    meta['fill_value'] = fill_value

    data = NDData(data_cube, meta=meta)

    grid = GriddedPSFModel(data, fill_value=fill_value)

    return grid
