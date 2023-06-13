# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import inspect
import warnings
from collections import defaultdict
from itertools import chain

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import (NDData, NoOverlapError, StdDevUncertainty,
                            overlap_slices)
from astropy.table import QTable, Table, hstack
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.utils._misc import _get_meta
from photutils.utils._optional_deps import HAS_TQDM
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['PSFPhotometry']


class PSFPhotometry:
    """
    Class to perform PSF photometry.

    This class implements a flexible PSF photometry algorithm that can
    find sources in an image, group overlapping sources, fit the PSF
    model to the sources, and subtract the fit PSF models from the
    image.
    """

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=LevMarLSQFitter(), localbkg_estimator=None,
                 maxiters=100, aperture_radius=None, progress_bar=None):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_callable(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.localbkg_estimator = self._validate_callable(
            localbkg_estimator, 'localbkg_estimator')
        self.maxiters = self._validate_maxiters(maxiters)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self._init_colnames = self._define_init_colnames()
        self._xinit_name = self._init_colnames['x_valid'][0]
        self._yinit_name = self._init_colnames['y_valid'][0]
        self._fluxinit_name = self._init_colnames['flux_valid'][0]

        self._unfixed_params = self._get_unfixed_params()

        # reset these attributes for each __call__ (see _reset_results)
        self.finder_results = []
        self.fit_error_indices = []
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)
        self._ungroup_indices = []

    def _reset_results(self):
        self.finder_results = []
        self.fit_error_indices = []
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)
        self._ungroup_indices = []

    @staticmethod
    def _validate_model(psf_model):
        # TODO: also allow output of prepare_psf_model (CompoundModel)
        #       when prepare_psf_model is fixed
        if not isinstance(psf_model, Fittable2DModel):
            raise TypeError('psf_model must be an astropy Fittable2DModel')
        return psf_model

    @staticmethod
    def _validate_callable(obj, name):
        if obj is not None and not callable(obj):
            raise TypeError(f'{name!r} must be a callable object')
        return obj

    def _validate_maxiters(self, maxiters):
        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            warnings.warn('"maxiters" will be ignored because it is not '
                          'accepted by the input fitter __call__ method',
                          AstropyUserWarning)
            maxiters = None
        return maxiters

    @staticmethod
    def _validate_radius(radius):
        if radius is not None and (not np.isscalar(radius)
                                   or radius <= 0 or ~np.isfinite(radius)):
            raise ValueError('radius must be a strictly-positive scalar')
        return radius

    def _validate_array(self, array, name, data_shape=None):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                raise ValueError(f'{name} must be a 2D array.')
            if data_shape is not None and array.shape != data_shape:
                raise ValueError(f'data and {name} must have the same shape.')
        return array

    def _get_unfixed_params(self):
        unfixed_params = []
        for param in self.psf_model.param_names:
            if not self.psf_model.fixed[param]:
                # TODO: check for only x, y, flux
                unfixed_params.append(param)

        if len(unfixed_params) > 3:
            raise ValueError('psf_model must have only 3 unfixed parameters, '
                             'corresponding to (x, y, flux)')

        return unfixed_params

    @staticmethod
    def _define_init_colnames():
        xy_suffixes = ('_init', 'centroid', '_centroid', '_peak', '',
                       'cen', '_cen', 'pos', '_pos')
        x_valid = ['x' + i for i in xy_suffixes]
        y_valid = ['y' + i for i in xy_suffixes]

        init_colnames = {}
        init_colnames['x_valid'] = x_valid
        init_colnames['y_valid'] = y_valid
        init_colnames['flux_valid'] = ('flux_init', 'flux', 'source_sum',
                                       'segment_flux', 'kron_flux')

        return init_colnames

    def _find_column_name(self, key, colnames):
        name = ''
        valid_names = self._init_colnames[key]
        for valid_name in valid_names:
            if valid_name in colnames:
                name = valid_name
        return name

    def _validate_params(self, init_params, unit):
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            raise TypeError('init_params must be an astropy Table')

        xcolname = self._find_column_name('x_valid', init_params.colnames)
        ycolname = self._find_column_name('y_valid', init_params.colnames)
        if not xcolname or not ycolname:
            raise ValueError('init_param must contain valid column names '
                             'for the x and y source positions')

        init_params = init_params.copy()
        if xcolname != self._xinit_name:
            init_params.rename_column(xcolname, self._xinit_name)
        if ycolname != self._yinit_name:
            init_params.rename_column(ycolname, self._yinit_name)

        fluxcolname = self._find_column_name('flux_valid',
                                             init_params.colnames)

        if fluxcolname:
            if fluxcolname != self._fluxinit_name:
                init_params.rename_column(fluxcolname, self._fluxinit_name)

            init_flux = init_params[self._fluxinit_name]
            if isinstance(init_flux, u.Quantity):
                if unit is None:
                    raise ValueError('init_params flux column has '
                                     'units, but the input data does not '
                                     'have units.')
                try:
                    init_params[self._fluxinit_name] = init_flux.to(unit)
                except u.UnitConversionError as exc:
                    raise ValueError('init_params flux column has '
                                     'units that are incompatible with '
                                     'the input data units.') from exc
            else:
                if unit is not None:
                    raise ValueError('The input data has units, but the '
                                     'init_params flux column does not have '
                                     'units.')

        return init_params

    @staticmethod
    def _make_mask(image, mask):
        def warn_nonfinite():
            warnings.warn('Input data contains unmasked non-finite values '
                          '(NaN or inf), which were automatically ignored.',
                          AstropyUserWarning)

        # if NaNs are in the data, no actual fitting takes place
        # https://github.com/astropy/astropy/pull/12811
        finite_mask = ~np.isfinite(image)

        if mask is not None:
            mask |= finite_mask
            if np.any(finite_mask & ~mask):
                warn_nonfinite()
        else:
            mask = finite_mask
            if np.any(finite_mask):
                warn_nonfinite()

        return mask

    @staticmethod
    def _flatten(iterable):
        """
        Flatten a list of lists.
        """
        return list(chain.from_iterable(iterable))

    def _add_progress_bar(self, iterable, desc=None):
        if self.progress_bar and HAS_TQDM:
            try:
                from ipywidgets import FloatProgress  # noqa: F401
                from tqdm.auto import tqdm
            except ImportError:
                from tqdm import tqdm

            iterable = tqdm(iterable, desc=desc)  # pragma: no cover

        return iterable

    def _get_aper_fluxes(self, data, mask, init_params):
        xpos = init_params[self._xinit_name]
        ypos = init_params[self._yinit_name]
        apertures = CircularAperture(zip(xpos, ypos), r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)
        return flux

    def _prepare_init_params(self, data, unit, mask, init_params):
        if init_params is None:
            if self.finder is None:
                raise ValueError('finder must be defined if init_params '
                                 'is not input')

            sources = self.finder(data, mask=mask)
            self.finder_results.append(sources)
            if sources is None:
                return None

            init_params = QTable()
            init_params['id'] = np.arange(len(sources)) + 1
            init_params[self._xinit_name] = sources['xcentroid']
            init_params[self._yinit_name] = sources['ycentroid']

        else:
            colnames = init_params.colnames
            if 'id' not in colnames:
                init_params['id'] = np.arange(len(init_params)) + 1

            if 'group_id' in colnames:
                # grouper is ignored if group_id is input in init_params
                self.grouper = None

        if 'local_bkg' not in init_params.colnames:
            if self.localbkg_estimator is None:
                local_bkg = np.zeros(len(init_params))
            else:
                local_bkg = self.localbkg_estimator(
                    data, init_params[self._xinit_name],
                    init_params[self._yinit_name], mask=mask)
            init_params['local_bkg'] = local_bkg

        self.fit_results['local_bkg'] = init_params['local_bkg']

        if self._fluxinit_name not in init_params.colnames:
            flux = self._get_aper_fluxes(data, mask, init_params)
            flux -= init_params['local_bkg']
            if unit is not None:
                flux <<= unit
            init_params[self._fluxinit_name] = flux

        if self.grouper is not None:
            # TODO: change grouper API
            # init_params['group_id'] = self.grouper(init_params)
            init_params = self.grouper(init_params)

        # no grouping
        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        # order init_params columns
        colname_order = ('id', 'group_id', 'local_bkg', self._xinit_name,
                         self._yinit_name, self._fluxinit_name)
        init_params = init_params[colname_order]

        return init_params

    def _get_psf_param_names(self):
        """
        Get the names of the PSF model parameters corresponding to x, y,
        and flux.

        The PSF model must either define 'xname', 'yname', and
        'fluxname' attributes (checked first) or have parameters called
        'x_0', 'y_0', and 'flux'. Otherwise, a `ValueError` is raised.
        """
        keys = [('xname', 'x_0'), ('yname', 'y_0'), ('fluxname', 'flux')]
        names = []
        for key in keys:
            try:
                name = getattr(self.psf_model, key[0])
            except AttributeError as exc:
                if key[1] in self.psf_model.param_names:
                    name = key[1]
                else:
                    msg = 'Could not find PSF parameter names'
                    raise ValueError(msg) from exc

            names.append(name)

        return tuple(names)

    def _param_map(self):
        psf_param_names = self._get_psf_param_names()

        param_map = {}
        param_map[self._xinit_name] = psf_param_names[0]
        param_map[self._yinit_name] = psf_param_names[1]
        param_map[self._fluxinit_name] = psf_param_names[2]

        init_suffix = self._xinit_name[1:]
        fit_param_map = {val: key.replace(init_suffix, '_fit')
                         for key, val in param_map.items()}
        err_param_map = {val: key.replace(init_suffix, '_err')
                         for key, val in param_map.items()}

        return param_map, fit_param_map, err_param_map

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources within
        a group.
        """
        param_map = self._param_map()[0]

        for index, source in enumerate(sources):
            model = self.psf_model.copy()
            for param, model_param in param_map.items():
                value = source[param]
                if isinstance(value, u.Quantity):
                    value = value.value  # psf model cannot be fit with units
                setattr(model, model_param, value)
                model.name = source['id']

            if index == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def _define_fit_data(self, sources, data, mask):
        yi = []
        xi = []
        cutout = []
        npixfit = []
        cen_index = []
        for row in sources:
            xcen = int(row[self._xinit_name] + 0.5)
            ycen = int(row[self._yinit_name] + 0.5)

            try:
                slc_lg, _ = overlap_slices(data.shape, self.fit_shape,
                                           (ycen, xcen), mode='trim')
            except NoOverlapError as exc:
                msg = (f'Initial source at ({xcen}, {ycen}) does not '
                       'overlap with the input data.')
                raise ValueError(msg) from exc

            yy, xx = np.mgrid[slc_lg]

            if mask is not None:
                inv_mask = ~mask[yy, xx]
                if np.count_nonzero(inv_mask) == 0:
                    msg = (f'Source at ({xcen}, {ycen}) is completely masked. '
                           'Remove the source from init_params or correct '
                           'the input mask.')
                    raise ValueError(msg)

                yy = yy[inv_mask]
                xx = xx[inv_mask]
            else:
                xx = xx.ravel()
                yy = yy.ravel()

            xi.append(xx)
            yi.append(yy)
            cutout.append(data[yy, xx] - row['local_bkg'])
            npixfit.append(len(xx))

            idx = np.where((xx == xcen) & (yy == ycen))[0]
            if len(idx) == 0:
                idx = [np.nan]
            cen_index.append(idx[0])

        # flatten the lists, which may contain arrays of different lengths
        # due to masking
        xi = self._flatten(xi)
        yi = self._flatten(yi)
        cutout = self._flatten(cutout)

        self._group_results['npixfit'].append(npixfit)
        self._group_results['cen_res_idx'].append(cen_index)

        return yi, xi, cutout

    @staticmethod
    def _split_compound_model(model, chunk_size):
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i:i + chunk_size]

    @staticmethod
    def _split_param_errs(param_err, nparam):
        for i in range(0, len(param_err), nparam):
            yield param_err[i:i + nparam]

    def _order_by_id(self, iterable):
        """
        Reorder the list from group-id to source-id order.
        """
        return [iterable[i] for i in self._ungroup_indices]

    def _ungroup(self, iterable):
        """
        Expand a list of lists (groups) and reorder in source-id order.
        """
        iterable = self._flatten(iterable)
        return self._order_by_id(iterable)

    def _make_fit_results(self, models, infos):
        psf_nsub = self.psf_model.n_submodels

        fit_models = []
        fit_infos = []
        fit_param_errs = []
        nparam = len(self._unfixed_params)
        for model, info in zip(models, infos):
            model_nsub = model.n_submodels

            param_cov = info.get('param_cov', None)
            if param_cov is None:
                if nparam == 0:  # model params are all fixed
                    nparam = 3
                param_err = np.array([np.nan] * nparam * model_nsub)
            else:
                param_err = np.sqrt(np.diag(param_cov))

            # model is for a single source (which may be compound)
            if model_nsub == psf_nsub:
                fit_models.append(model)
                fit_infos.append(info)
                fit_param_errs.append(param_err)
                continue

            # model is a grouped model for multiple sources
            fit_models.extend(self._split_compound_model(model, psf_nsub))
            nsources = model_nsub // psf_nsub
            fit_infos.extend([info] * nsources)  # views
            fit_param_errs.extend(self._split_param_errs(param_err, nparam))

        if len(fit_models) != len(fit_infos):
            raise ValueError('fit_models and fit_infos have different lengths')

        # change the sorting from group_id to source id order
        fit_models = self._order_by_id(fit_models)
        fit_infos = self._order_by_id(fit_infos)
        fit_param_errs = np.array(self._order_by_id(fit_param_errs))

        self.fit_results['fit_models'] = fit_models
        self.fit_results['fit_infos'] = fit_infos
        self.fit_results['fit_param_errs'] = fit_param_errs

        return fit_models

    def _set_fit_error_indices(self):
        indices = []
        for index, fit_info in enumerate(self.fit_results['fit_infos']):
            ierr = fit_info.get('ierr', None)
            # check if in good flags defined by scipy
            if ierr is not None:
                # scipy.optimize.leastsq
                if ierr not in (1, 2, 3, 4):
                    indices.append(index)
            else:
                # scipy.optimize.least_squares
                status = fit_info.get('status', None)
                if status is not None and status in (-1, 0):
                    indices.append(index)

        self.fit_error_indices = np.array(indices, dtype=int)

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        if self.maxiters is not None:
            kwargs = {'maxiter': self.maxiters}
        else:
            kwargs = {}

        sources = init_params.group_by('group_id')
        self._ungroup_indices = np.argsort(sources['id'].value)
        sources = sources.groups
        sources = self._add_progress_bar(sources, desc='Fit source/group')

        fit_models = []
        fit_infos = []
        nmodels = []
        for sources_ in sources:  # fit in group_id order
            nsources = len(sources_)
            nmodels.append([nsources] * nsources)
            psf_model = self._make_psf_model(sources_)
            yi, xi, cutout = self._define_fit_data(sources_, data, mask)

            if error is not None:
                weights = 1.0 / error[yi, xi]
            else:
                weights = None

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyUserWarning)
                try:
                    fit_model = self.fitter(psf_model, xi, yi, cutout,
                                            weights=weights, **kwargs)
                except TypeError as exc:
                    msg = ('The number of data points is less than the '
                           'number of fit parameters. This is likely due '
                           'to overmasked data. Please check the input '
                           'mask.')
                    raise ValueError(msg) from exc

                fit_info = self.fitter.fit_info.copy()

            fit_models.append(fit_model)
            fit_infos.append(fit_info)

        self._group_results['fit_models'] = fit_models
        self._group_results['fit_infos'] = fit_infos
        self._group_results['nmodels'] = nmodels

        # split the groups and return objects in source-id order
        fit_models = self._make_fit_results(fit_models, fit_infos)
        self._set_fit_error_indices()

        return fit_models

    def _model_params_to_table(self, models):
        param_map = self._param_map()[1]

        params = []
        for model in models:
            mparams = []
            for model_param in param_map.keys():
                mparams.append(getattr(model, model_param).value)
            params.append(mparams)
        vals = np.transpose(params)

        colnames = param_map.values()
        table = QTable()
        for index, colname in enumerate(colnames):
            table[colname] = vals[index]

        return table

    def _param_errors_to_table(self):
        param_map = self._param_map()[2]
        table = QTable()
        for index, name in enumerate(self._unfixed_params):
            colname = param_map[name]
            table[colname] = self.fit_results['fit_param_errs'][:, index]

        # add missing error columns
        colnames = ('x_err', 'y_err', 'flux_err')
        nsources = len(self.fit_results['fit_models'])
        for colname in colnames:
            if colname not in table.colnames:
                table[colname] = [np.nan] * nsources

        # sort column names
        return table[colnames]

    def _calc_fit_metrics(self, data, source_tbl):
        cen_idx = self._ungroup(self._group_results['cen_res_idx'])
        self.fit_results['cen_res_idx'] = cen_idx

        split_index = []
        for npixfit in self._group_results['npixfit']:
            split_index.append(np.cumsum(npixfit)[:-1])

        fit_residuals = []
        for idx, fit_info in zip(split_index,
                                 self._group_results['fit_infos']):
            fit_residuals.extend(np.split(fit_info['fvec'], idx))
        fit_residuals = self._order_by_id(fit_residuals)
        self.fit_results['fit_residuals'] = fit_residuals

        for npixfit, residuals in zip(self.fit_results['npixfit'],
                                      fit_residuals):
            if len(residuals) != npixfit:
                raise ValueError('size of residuals does not match npixfit')

        if len(fit_residuals) != len(source_tbl):
            raise ValueError('fit_residuals does not match the source '
                             'table length')

        with warnings.catch_warnings():
            # ignore divide-by-zero if flux = 0
            warnings.simplefilter('ignore', RuntimeWarning)

            qfit = []
            cfit = []
            for index, (model, residual, cen_idx_) in enumerate(
                    zip(self.fit_results['fit_models'], fit_residuals,
                        cen_idx)):
                source = source_tbl[index]
                xcen = int(source[self._xinit_name] + 0.5)
                ycen = int(source[self._yinit_name] + 0.5)
                flux_fit = source['flux_fit']
                qfit.append(np.sum(np.abs(residual)) / flux_fit)

                if np.isnan(cen_idx_):
                    # calculate residual at central pixel
                    cen_residual = data[ycen, xcen] - model(xcen, ycen)
                else:
                    # find residual at (xcen, ycen)
                    cen_residual = -residual[cen_idx_]
                cfit.append(cen_residual / flux_fit)

        return qfit, cfit

    def _define_flags(self, source_tbl, shape):
        flags = np.zeros(len(self.fit_results['fit_infos']), dtype=int)

        for index, row in enumerate(source_tbl):
            if row['npixfit'] < np.prod(self.fit_shape):
                flags[index] += 1
            if (row['x_fit'] < 0 or row['y_fit'] < 0
                    or row['x_fit'] > shape[1] or row['y_fit'] > shape[0]):
                flags[index] += 2
            if row['flux_fit'] <= 0:
                flags[index] += 4

        flags[self.fit_error_indices] += 8

        for index, fit_info in enumerate(self.fit_results['fit_infos']):
            if fit_info['param_cov'] is None:
                flags[index] += 16

        return flags

    def __call__(self, data, *, mask=None, error=None, init_params=None):
        """
        Perform PSF photometry.
        """
        if isinstance(data, NDData):
            data_ = data.data
            if data.unit is not None:
                data_ <<= data.unit
            mask = data.mask
            unc = data.uncertainty
            if unc is not None:
                unc = unc.represent_as(StdDevUncertainty)
                error = unc.array
                if unc.unit is not None:
                    error <<= unc.unit
            return self.__call__(data_, mask=mask, error=error,
                                 init_params=init_params)

        # reset results from previous runs
        self._reset_results()

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        data = self._validate_array(data, 'data')
        mask = self._validate_array(mask, 'mask', data_shape=data.shape)
        mask = self._make_mask(data, mask)
        init_params = self._validate_params(init_params, unit)  # also copies

        if (self.aperture_radius is None
            and (init_params is None
                 or self._fluxinit_name not in init_params.colnames)):
            raise ValueError('aperture_radius must be defined if init_params '
                             'is not input or if a flux column is not in '
                             'init_params')

        init_params = self._prepare_init_params(data, unit, mask, init_params)

        fit_models = self._fit_sources(data, init_params, error=error,
                                       mask=mask)

        # create output table
        fit_sources = self._model_params_to_table(fit_models)  # ungrouped
        if len(init_params) != len(fit_sources):
            raise ValueError('init_params and fit_sources tables have '
                             'different lengths')
        source_tbl = hstack((init_params, fit_sources))

        param_errors = self._param_errors_to_table()
        if len(param_errors) > 0:
            if len(param_errors) != len(source_tbl):
                raise ValueError('param errors and fit sources tables have '
                                 'different lengths')
            source_tbl = hstack((source_tbl, param_errors))

        npixfit = self._ungroup(self._group_results['npixfit'])
        self.fit_results['npixfit'] = npixfit
        source_tbl['npixfit'] = npixfit

        nmodels = self._ungroup(self._group_results['nmodels'])
        self.fit_results['nmodels'] = nmodels
        source_tbl['group_size'] = nmodels

        qfit, cfit = self._calc_fit_metrics(data, source_tbl)
        source_tbl['qfit'] = qfit
        source_tbl['cfit'] = cfit

        source_tbl['flags'] = self._define_flags(source_tbl, data.shape)

        if unit is not None:
            source_tbl['local_bkg'] <<= unit
            source_tbl['flux_fit'] <<= unit
            source_tbl['flux_err'] <<= unit

        meta = _get_meta()
        attrs = ('fit_shape', 'maxiters', 'aperture_radius', 'progress_bar')
        for attr in attrs:
            meta[attr] = getattr(self, attr)
        source_tbl.meta = meta

        if len(self.fit_error_indices) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table.',
                          AstropyUserWarning)

        return source_tbl

    def make_model_image(self, shape, psf_shape):
        fit_models = self.fit_results['fit_models']

        data = np.zeros(shape)
        xname, yname = self._get_psf_param_names()[0:2]

        desc = 'Model image'
        fit_models = self._add_progress_bar(fit_models, desc=desc)

        # fit_models must be a list of individual, not grouped, PSF
        # models, i.e., there should be one PSF model (which may be
        # compound) for each source
        for fit_model, local_bkg in zip(fit_models,
                                        self.fit_results['local_bkg']):
            x0 = getattr(fit_model, xname).value
            y0 = getattr(fit_model, yname).value
            slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0), mode='trim')
            yy, xx = np.mgrid[slc_lg]
            data[slc_lg] += (fit_model(xx, yy) + local_bkg)

        return data

    def make_residual_image(self, data, psf_shape):
        return data - self.make_model_image(data.shape, psf_shape)
