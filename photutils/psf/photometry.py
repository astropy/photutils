# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

from itertools import chain
import inspect
import warnings

import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import overlap_slices
from astropy.table import QTable, Table, hstack
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
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
                 fitter=LevMarLSQFitter(), maxiters=100, aperture_radius=None,
                 progress_bar=None):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_callable(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.maxiters = self._validate_maxiters(maxiters)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self._unfixed_params = self._get_unfixed_params()

        self.finder_results = []
        self.fit_error_indices = []
        self.fit_models = []
        self.fit_infos = []

        self._fit_group_models = []
        self._fit_group_infos = []
        self._fit_group_nsources = []
        self._fit_param_errs = []
        self._ungroup_indices = []
        self._cen_residual_indices = []
        self._group_nfit = []
        self._group_index = []

        self._valid_x_colnames = ('x_init', 'xcentroid', 'x_centroid',
                                  'x_peak', 'x', 'xcen', 'x_cen', 'xpos',
                                  'x_pos')
        self._valid_y_colnames = ('y_init', 'ycentroid', 'y_centroid',
                                  'y_peak', 'y', 'ycen', 'y_cen', 'ypos',
                                  'y_pos')
        self._valid_flux_colnames = ('flux_init', 'flux', 'source_sum',
                                     'segment_flux', 'kron_flux')

        self._xinit_name = self._valid_x_colnames[0]
        self._yinit_name = self._valid_y_colnames[0]
        self._fluxinit_name = self._valid_flux_colnames[0]

    @staticmethod
    def _validate_model(psf_model):
        if not isinstance(psf_model, Fittable2DModel):
            raise TypeError('psf_model must be an astropy Fittable2DModel')
        return psf_model

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

    @staticmethod
    def _find_column_name(valid_names, colnames):
        name = ''
        for valid_name in valid_names:
            if valid_name in colnames:
                name = valid_name
        return name

    def _validate_params(self, init_params):
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            raise TypeError('init_params must be an astropy Table')

        xcolname = self._find_column_name(self._valid_x_colnames,
                                          init_params.colnames)
        ycolname = self._find_column_name(self._valid_y_colnames,
                                          init_params.colnames)
        if not xcolname or not ycolname:
            raise ValueError('init_param must contain valid column names '
                             'for the x and y source positions')

        init_params = init_params.copy()
        if xcolname != self._xinit_name:
            init_params.rename_column(xcolname, self._xinit_name)
        if ycolname != self._xinit_name:
            init_params.rename_column(ycolname, self._yinit_name)

        fluxcolname = self._find_column_name(self._valid_flux_colnames,
                                             init_params.colnames)
        if fluxcolname:
            if fluxcolname != self._fluxinit_name:
                init_params.rename_column(fluxcolname, self._fluxinit_name)

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

    def _make_init_params(self, data, mask, sources):
        """
        sources : `~astropy.table.Table`
            Output from star finder with 'xcentroid' and 'ycentroid'
            columns'.
        """
        init_params = QTable()
        init_params['id'] = np.arange(len(sources)) + 1
        init_params[self._xinit_name] = sources['xcentroid']
        init_params[self._yinit_name] = sources['ycentroid']
        init_params[self._fluxinit_name] = self._get_aper_fluxes(data, mask,
                                                                 init_params)

        return init_params

    def _param_map(self):
        # TODO: generalize this mapping based of self.psf_model
        param_map = {}
        param_map[self._xinit_name] = 'x_0'
        param_map[self._yinit_name] = 'y_0'
        param_map[self._fluxinit_name] = 'flux'

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
                setattr(model, model_param, source[param])
                model.name = source['id']

            if index == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def _define_fit_coords(self, sources, shape, mask):
        hshape = (self.fit_shape - 1) // 2
        yi = []
        xi = []
        group_nfit = []
        cen_index = []
        for row in sources:
            xcen = int(row[self._xinit_name] + 0.5)
            ycen = int(row[self._yinit_name] + 0.5)
            # max values are non-inclusive (slices)
            xmin = max((0, xcen - hshape[1]))
            xmax = min((shape[1], xcen + hshape[1] + 1))
            ymin = max((0, ycen - hshape[0]))
            ymax = min((shape[0], ycen + hshape[0] + 1))
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]

            if mask is not None:
                inv_mask = ~mask[yy, xx]
                if np.count_nonzero(inv_mask) == 0:
                    msg = (f'Source at {(xcen, ycen)} is completely masked. '
                           'Remove the source from init_params or correct '
                           'the input mask.')
                    raise ValueError(msg)

                yy = yy[inv_mask]
                xx = xx[inv_mask]
            else:
                xx = xx.ravel()
                yy = yy.ravel()

            idx = np.where((xx == xcen) & (yy == ycen))[0]
            if len(idx) == 0:
                idx = [np.nan]
            cen_index.append(idx[0])

            xi.append(xx)
            yi.append(yy)
            group_nfit.append(len(xx))

        # flatten the lists, which may contain arrays of different
        # lengths due to masking
        xi = self._flatten(xi)
        yi = self._flatten(yi)

        self._cen_residual_indices.append(cen_index)
        self._group_nfit.append(group_nfit)
        self._group_index.append(np.cumsum(group_nfit)[:-1])

        return yi, xi

    @staticmethod
    def _split_compound_model(model, chunk_size):
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i:i + chunk_size]

    @staticmethod
    def _split_param_errs(param_err, nparam):
        for i in range(0, len(param_err), nparam):
            yield param_err[i:i + nparam]

    def _split_groups(self, models, infos):
        psf_nsub = self.psf_model.n_submodels

        fit_models = []
        fit_infos = []
        fit_param_errs = []
        nparam = len(self._unfixed_params)
        for model, info in zip(models, infos):
            model_nsub = model.n_submodels

            param_cov = info.get('param_cov', None)
            if param_cov is None:
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
            if nparam != 0:
                fit_param_errs.extend(self._split_param_errs(param_err,
                                                             nparam))

        if len(fit_models) != len(fit_infos):
            raise ValueError('fit_models and fit_infos have different lengths')

        # change the sorting from group_id to source id order
        fit_models = [fit_models[i] for i in self._ungroup_indices]
        fit_infos = [fit_infos[i] for i in self._ungroup_indices]
        if nparam != 0:
            fit_param_errs = [fit_param_errs[i] for i in self._ungroup_indices]

        self.fit_models = fit_models
        self.fit_infos = fit_infos
        self._fit_param_errs = np.array(fit_param_errs)

        return fit_models

    def _set_fit_error_indices(self):
        indices = []
        for index, fit_info in enumerate(self.fit_infos):
            ierr = fit_info.get('ierr', None)
            if ierr not in (1, 2, 3, 4):  # all good flags defined by scipy
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
        for sources_ in sources:  # fit in group_id order
            psf_model = self._make_psf_model(sources_)
            yi, xi = self._define_fit_coords(sources_, data.shape, mask)
            cutout = data[yi, xi]

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

        # split the groups and return objects in source-id order
        self._fit_group_models = fit_models
        self._fit_group_infos = fit_infos
        fit_models = self._split_groups(fit_models, fit_infos)
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
            table[colname] = self._fit_param_errs[:, index]

        # order error columns
        colnames = ('x_err', 'y_err', 'flux_err')
        tmp = {val: i for i, val in enumerate(colnames)}
        sorted_colnames = sorted(table.colnames, key=lambda val: tmp[val])

        return table[sorted_colnames]

    def _calc_fit_metrics(self, data, source_tbl):

        psf_shape = self.fit_shape
        model_resid = []
        for i, fit_model in enumerate(self.fit_models):
            x0 = source_tbl['x_init'][i]
            y0 = source_tbl['y_init'][i]
            slc_lg, _ = overlap_slices(data.shape, psf_shape, (y0, x0),
                                       mode='trim')
            yy, xx = np.mgrid[slc_lg]
            res = fit_model(xx, yy)
            model_resid.append(data[slc_lg] - res)
        self.model_resid = model_resid

        fit_residuals = []
        for idx, fit_info in zip(self._group_index, self._fit_group_infos):
            fit_residuals.extend(np.split(fit_info['fvec'], idx))
        fit_residuals = [fit_residuals[i] for i in self._ungroup_indices]
        self.fit_res = fit_residuals

        if len(fit_residuals) != len(source_tbl):
            raise ValueError('fit_residuals does not match the source '
                             'table length')

        with warnings.catch_warnings():
            # ignore divide-by-zero if flux = 0
            warnings.simplefilter('ignore', RuntimeWarning)

            qfit = []
            qfit2 = []
            cfit = []
            cfit2 = []

            for index, (model, residual, res_cen_idx) in enumerate(
                    zip(self.fit_models, fit_residuals,
                        self._cen_residual_indices)):
                source = source_tbl[index]
                xcen = int(source[self._xinit_name] + 0.5)
                ycen = int(source[self._yinit_name] + 0.5)
                flux_fit = source['flux_fit']
                # flux_fit2 = model.flux.value  # identical to flux_fit
                qfit.append(np.sum(np.abs(residual)) / flux_fit)
                qfit2.append(np.sum(np.abs(model_resid[index])) / flux_fit)

                if np.isnan(res_cen_idx):
                    # need to calculate residual at central pixel
                    cen_residual = data[ycen, xcen] - model(xcen, ycen)
                else:
                    # find residual at (xcen, ycen)
                    cen_residual = -residual[res_cen_idx]

                cfit.append(cen_residual / flux_fit)
                cfit2.append((data[ycen, xcen] - model(xcen, ycen)) / flux_fit)

        return qfit, cfit, qfit2, cfit2

    def _define_flags(self):
        flags = np.zeros(len(self.fit_infos), dtype=int)
        flags[self.fit_error_indices] = 1

        idx = []
        for index, fit_info in enumerate(self.fit_infos):
            if 'completely masked' in fit_info['message']:
                idx.append(index)
        flags[idx] = 2

        return flags

    def __call__(self, data, *, mask=None, error=None, init_params=None):
        """
        Perform PSF photometry.
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        data = self._validate_array(data, 'data')
        mask = self._make_mask(data,
                               self._validate_array(mask, 'mask',
                                                    data_shape=data.shape))
        init_params = self._validate_params(init_params)  # also copies

        if (self.aperture_radius is None
            and (init_params is None
                 or self._fluxinit_name not in init_params.colnames)):
            raise ValueError('aperture_radius must be defined if init_params '
                             'is not input or if a flux column is not in '
                             'init_params')

        if init_params is None:
            if self.finder is None:
                raise ValueError('finder must be defined if init_params '
                                 'is not input')

            sources = self.finder(data, mask=mask)
            self.finder_results.append(sources)
            if sources is None:
                return None

            init_params = self._make_init_params(data, mask, sources)
        else:
            colnames = init_params.colnames
            if 'id' not in colnames:
                init_params['id'] = np.arange(len(init_params)) + 1

            if self._fluxinit_name not in colnames:
                init_params[self._fluxinit_name] = self._get_aper_fluxes(
                    data, mask, init_params)

            if 'group_id' in colnames:
                # grouper is ignored if group_id is input in init_params
                self.grouper = None

        if self.grouper is not None:
            # TODO: change grouper API
            # init_params['group_id'] = self.grouper(init_params)
            init_params = self.grouper(init_params)

        # no grouping
        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        # order init_params columns
        colnames = ('id', 'group_id', self._xinit_name, self._yinit_name,
                    self._fluxinit_name)
        init_params = init_params[colnames]

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

        # flatten the indices and put in source-id order
        nfit = self._flatten(self._group_nfit)
        nfit = [nfit[i] for i in self._ungroup_indices]
        self._group_nfit = nfit
        source_tbl['nfit'] = self._group_nfit

        indices = self._flatten(self._cen_residual_indices)
        indices = [indices[i] for i in self._ungroup_indices]
        self._cen_residual_indices = indices

        qfit, cfit, qfit2, cfit2 = self._calc_fit_metrics(data, source_tbl)
        source_tbl['qfit'] = qfit
        source_tbl['qfit2'] = qfit2
        source_tbl['cfit'] = cfit
        source_tbl['cfit2'] = cfit2

        source_tbl['flags'] = self._define_flags()

        if len(self.fit_error_indices) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table, and '
                          'the "fit_error_indices" and "fit_infos" attributes '
                          'for more information.', AstropyUserWarning)

        return source_tbl

    def _get_psf_param_names(self):
        """
        Get the names of the PSF model parameters corresponding to x, y,
        and flux.

        The PSF model must either define 'xname', 'yname', and
        'fluxname' attributes or have parameters called 'x_0', 'y_0',
        and 'flux'. Otherwise, a `ValueError` is raised.
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

    def make_model_image(self, shape, psf_shape):
        fit_models = self.fit_models

        data = np.zeros(shape)
        xname, yname, _ = self._get_psf_param_names()

        desc = 'Model image'
        fit_models = self._add_progress_bar(fit_models, desc=desc)

        # fit_models must be a list of individual, not grouped, PSF
        # models, i.e., there should be one PSF model (which may be
        # compound) for each source
        for fit_model in fit_models:
            x0 = getattr(fit_model, xname).value
            y0 = getattr(fit_model, yname).value
            slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0), mode='trim')
            yy, xx = np.mgrid[slc_lg]
            data[slc_lg] += fit_model(xx, yy)

        return data

    def make_residual_image(self, data, psf_shape):
        return data - self.make_model_image(data.shape, psf_shape)
