# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import inspect
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import astropy.units as u
import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import (NDData, NoOverlapError, StdDevUncertainty,
                            overlap_slices)
from astropy.table import QTable, Table, hstack, vstack
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.background import LocalBackground
from photutils.psf.groupstars import GroupStarsBase
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._round import py2intround
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['PSFPhotometry', 'IterativePSFPhotometry']


class PSFPhotometry:
    """
    Class to perform PSF photometry.

    This class implements a flexible PSF photometry algorithm that can
    find sources in an image, group overlapping sources, fit the PSF
    model to the sources, and subtract the fit PSF models from the
    image.

    Parameters
    ----------
    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model to fit to the data. The model needs to have
        three parameters named ``x_0``, ``y_0``, and ``flux``,
        corresponding to the center (x, y) position and flux.

    fit_shape : int or length-2 array_like
        The rectangular shape around the center of a star that will
        be used to define the PSF-fitting data. If ``fit_shape`` is a
        scalar then a square shape of size ``fit_shape`` will be used.
        If ``fit_shape`` has two elements, they must be in ``(ny,
        nx)`` order. Each element of ``fit_shape`` must be an odd
        number. In general, ``fit_shape`` should be set to a small size
        (e.g., ``(5, 5)``) that covers the region with the highest flux
        signal-to-noise.

    finder : callable or `~photutils.detection.StarFinderBase` or `None`, optional
        A callable used to identify stars in an image. The
        ``finder`` must accept a 2D image as input and return a
        `~astropy.table.Table` containing the x and y centroid
        positions. These positions are used as the starting points
        for the PSF fitting. The allowed ``x`` column names are (same
        suffix for ``y``): ``'x_init'``, ``'xinit'``, ``'xcentroid'``,
        ``'x_centroid'``, ``'x_peak'``, ``'x'``, ``'xcen'``,
        ``'x_cen'``, ``'xpos'``, ``'x_pos'``, ``'x_0'``, and ``'x0'``.
        If `None`, then the initial (x, y) model positions must be input
        using the ``init_params`` keyword when calling the class. The
        (x, y) values in ``init_params`` override this keyword.

    grouper : `~photutils.psf.SourceGrouper` or callable or `None`, optional
        A callable used to group stars. Typically, grouped stars are
        those that overlap with their neighbors. Stars that are grouped
        are fit simultaneously. The ``grouper`` must accept the x and
        y coordinates of the sources and return an integer array of
        the group id numbers (starting from 1) indicating the group
        in which a given source belongs. If `None`, then no grouping
        is performed, i.e. each source is fit independently. The
        ``group_id`` values in ``init_params`` override this keyword
        *only for the first iteration*. A warning is raised if any group
        size is larger than 25 sources.

    fitter : `~astropy.modeling.fitting.Fitter`, optional
        The fitter object used to perform the fit of the model to the
        data.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each source.

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, optional
        The object used to estimate the local background around each
        source.  If `None`, then no local background is subtracted.  The
        ``local_bkg`` values in ``init_params`` override this keyword.

    aperture_radius : float, optional
        The radius of the circular aperture used to estimate the initial
        flux of each source. The ``flux_init`` values in ``init_params``
        override this keyword.

    progress_bar : bool, optional
        Whether to display a progress bar when fitting the sources
        (or groups). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.
        Note that the progress bar does not currently work in the
        Jupyter console due to limitations in ``tqdm``.
    """

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=LevMarLSQFitter(), fitter_maxiters=100,
                 localbkg_estimator=None, aperture_radius=None,
                 progress_bar=False):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_grouper(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.localbkg_estimator = self._validate_localbkg(
            localbkg_estimator, 'localbkg_estimator')
        self.fitter_maxiters = self._validate_maxiters(fitter_maxiters)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self._init_colnames = self._define_init_colnames()
        self._xinit_name = self._init_colnames['x_valid'][0]
        self._yinit_name = self._init_colnames['y_valid'][0]
        self._fluxinit_name = self._init_colnames['flux_valid'][0]

        self._unfixed_params = self._get_unfixed_params()

        # reset these attributes for each __call__ (see _reset_results)
        self.finder_results = []
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)
        self._fit_models = None

    def _reset_results(self):
        self.finder_results = []
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)
        self._fit_models = None

    def _validate_grouper(self, grouper, name):
        # remove this check when GroupStarsBase subclasses are removed
        if isinstance(grouper, GroupStarsBase):
            raise ValueError('Invalid grouper class. Please use '
                             'SourceGrouper.')
        return self._validate_callable(grouper, name)

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

    def _validate_localbkg(self, value, name):
        if value is not None and not isinstance(value, LocalBackground):
            raise ValueError('localbkg_estimator must be a '
                             'LocalBackground instance.')
        return self._validate_callable(value, name)

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
            raise ValueError('aperture_radius must be a strictly-positive '
                             'scalar')
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
                unfixed_params.append(param)
        return unfixed_params

    @staticmethod
    def _define_init_colnames():
        xy_suffixes = ('_init', 'init', 'centroid', '_centroid', '_peak', '',
                       'cen', '_cen', 'pos', '_pos', '_0', '0')
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

        init_params = init_params.copy()  # preserve input init_params
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
            finite_mask |= mask
            if np.any(finite_mask & ~mask):
                warn_nonfinite()
        else:
            mask = finite_mask
            if np.any(finite_mask):
                warn_nonfinite()
            else:
                mask = None

        return mask

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

        self.fit_results['local_bkg'] = init_params['local_bkg'].value

        if self._fluxinit_name not in init_params.colnames:
            flux = self._get_aper_fluxes(data, mask, init_params)
            flux -= init_params['local_bkg']
            if unit is not None:
                flux <<= unit
            init_params[self._fluxinit_name] = flux

        if self.grouper is not None:
            init_params['group_id'] = self.grouper(
                init_params['x_init'], init_params['y_init'])

        # no grouping
        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        extra_params = self._get_psf_param_names()[1]
        param_map = self._param_map()[0]

        extra_param_cols = []
        for extra_param in extra_params:
            for key, val in param_map.items():
                if val == extra_param:
                    extra_param_cols.append(key)

        for extra_col in extra_param_cols:
            if extra_col not in init_params.colnames:
                init_params[extra_col] = getattr(self.psf_model,
                                                 param_map[extra_col])

        # order init_params columns
        colname_order = ['id', 'group_id', 'local_bkg', self._xinit_name,
                         self._yinit_name, self._fluxinit_name]
        colname_order.extend(extra_param_cols)
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

        extra_params = []
        for key in self._unfixed_params:
            if key not in names:
                extra_params.append(key)

        return tuple(names), tuple(extra_params)

    def _param_map(self):
        psf_param_names, extra_params = self._get_psf_param_names()
        xname, yname, fluxname = psf_param_names

        param_map = {}
        param_map[self._xinit_name] = xname
        param_map[self._yinit_name] = yname
        param_map[self._fluxinit_name] = fluxname

        for extra_param in extra_params:
            param_map[f'{extra_param}_init'] = extra_param

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
            xcen = py2intround(row[self._xinit_name])
            ycen = py2intround(row[self._yinit_name])

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
        xi = _flatten(xi)
        yi = _flatten(yi)
        cutout = _flatten(cutout)

        self._group_results['npixfit'].append(npixfit)
        self._group_results['psfcenter_indices'].append(cen_index)

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
        return [iterable[i] for i in self._group_results['ungroup_indices']]

    def _ungroup(self, iterable):
        """
        Expand a list of lists (groups) and reorder in source-id order.
        """
        iterable = _flatten(iterable)
        return self._order_by_id(iterable)

    def _get_fit_error_indices(self):
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

        return np.array(indices, dtype=int)

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

        self._fit_models = fit_models
        self.fit_results['fit_infos'] = fit_infos
        self.fit_results['fit_param_errs'] = fit_param_errs
        self.fit_results['fit_error_indices'] = self._get_fit_error_indices()

        return fit_models

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        if self.fitter_maxiters is not None:
            kwargs = {'maxiter': self.fitter_maxiters}
        else:
            kwargs = {}

        sources = init_params.group_by('group_id')
        ungroup_idx = np.argsort(sources['id'].value)
        self._group_results['ungroup_indices'] = ungroup_idx
        sources = sources.groups
        if self.progress_bar:
            desc = 'Fit source/group'
            sources = add_progress_bar(sources, desc=desc)  # pragma: no cover

        # Save the fit_info results for these keys if they are present.
        # Some of these keys are returned only by some fitters. These
        # keys contain the fit residuals (fvec or fun), the parameter
        # covariance matrix (param_cov), and the fit status (ierr,
        # message) or (status).
        fit_info_keys = ('fvec', 'fun', 'param_cov', 'ierr', 'message',
                         'status')

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
                    try:
                        fit_model.clear_cache()
                    except AttributeError:
                        pass
                except TypeError as exc:
                    msg = ('The number of data points is less than the '
                           'number of fit parameters. This is likely due '
                           'to overmasked data. Please check the input '
                           'mask.')
                    raise ValueError(msg) from exc

                fit_info = {}
                for key in fit_info_keys:
                    value = self.fitter.fit_info.get(key, None)
                    if value is not None:
                        fit_info[key] = value

            fit_models.append(fit_model)
            fit_infos.append(fit_info)

        self._group_results['fit_models'] = fit_models
        self._group_results['fit_infos'] = fit_infos
        self._group_results['nmodels'] = nmodels

        # split the groups and return objects in source-id order
        fit_models = self._make_fit_results(fit_models, fit_infos)

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

        colnames = list(param_map.values())

        # add missing error columns
        nsources = len(self._fit_models)
        for colname in colnames:
            if colname not in table.colnames:
                table[colname] = [np.nan] * nsources

        # sort column names
        return table[colnames]

    def _calc_fit_metrics(self, data, source_tbl):
        # Keep cen_idx as a list because it can have NaNs with the ints.
        # If NaNs are present, turning it into an array will convert the
        # ints to floats, which cannot be used as slices.
        cen_idx = self._ungroup(self._group_results['psfcenter_indices'])
        self.fit_results['psfcenter_indices'] = cen_idx

        split_index = []
        for npixfit in self._group_results['npixfit']:
            split_index.append(np.cumsum(npixfit)[:-1])

        # find the key with the fit residual (fitter dependent)
        finfo_keys = self._group_results['fit_infos'][0].keys()
        keys = ('fvec', 'fun')
        key = None
        for key_ in keys:
            if key_ in finfo_keys:
                key = key_

        # SimplexLSQFitter
        if key is None:
            qfit = cfit = np.array([[np.nan]] * len(source_tbl))
            return qfit, cfit

        fit_residuals = []
        for idx, fit_info in zip(split_index,
                                 self._group_results['fit_infos']):
            fit_residuals.extend(np.split(fit_info[key], idx))
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
                    zip(self._fit_models, fit_residuals, cen_idx)):
                source = source_tbl[index]
                xcen = py2intround(source[self._xinit_name])
                ycen = py2intround(source[self._yinit_name])
                flux_fit = source['flux_fit']
                qfit.append(np.sum(np.abs(residual)) / flux_fit)

                if np.isnan(cen_idx_):
                    # calculate residual at central pixel if the central pixel
                    # is within the bounds of the ``data``, otherwise mask it:
                    cen_in_data = (
                        0 <= ycen <= data.shape[0] - 1
                        and 0 <= xcen <= data.shape[1] - 1
                    )
                    if cen_in_data:
                        cen_residual = data[ycen, xcen] - model(xcen, ycen)
                    else:
                        cen_residual = np.nan
                else:
                    # find residual at (xcen, ycen)
                    cen_residual = -residual[cen_idx_]
                cfit.append(cen_residual / flux_fit)

        return qfit, cfit

    def _define_flags(self, source_tbl, shape):
        flags = np.zeros(len(source_tbl), dtype=int)

        for index, row in enumerate(source_tbl):
            if row['npixfit'] < np.prod(self.fit_shape):
                flags[index] += 1
            if (row['x_fit'] < 0 or row['y_fit'] < 0
                    or row['x_fit'] > shape[1] or row['y_fit'] > shape[0]):
                flags[index] += 2
            if row['flux_fit'] <= 0:
                flags[index] += 4

        flags[self.fit_results['fit_error_indices']] += 8

        try:
            for index, fit_info in enumerate(self.fit_results['fit_infos']):
                if fit_info['param_cov'] is None:
                    flags[index] += 16
        except KeyError:
            pass

        return flags

    def __call__(self, data, *, mask=None, error=None, init_params=None):
        """
        Perform PSF photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which to perform photometry. Invalid data
            values (i.e., NaN or inf) are automatically masked.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.

        error : 2D `~numpy.ndarray`, optional
            The pixel-wise 1-sigma errors of the input ``data``.
            ``error`` is assumed to include *all* sources of
            error, including the Poisson error of the sources
            (see `~photutils.utils.calc_total_error`) . ``error``
            must have the same shape as the input ``data``. If a
            `~astropy.units.Quantity` array, then ``data`` must also be
            a `~astropy.units.Quantity` array with the same units.

        init_params : `~astropy.table.Table` or `None`, optional
            A table containing the initial guesses of the (x, y, flux)
            model parameters for each source. If the x and y values are
            not input, then the ``finder`` keyword must be defined. If
            the flux values are not input, then the ``aperture_radius``
            keyword must be defined. Note that the initital flux
            values refer to the model flux parameters and are not
            corrected for local background values (computed using
            ``localbkg_estimator`` or input in a ``local_bkg`` column)
            The allowed column names are:

              * ``x_init``, ``xinit``, ``xcentroid``, ``x_centroid``,
                ``x_peak``, ``x``, ``xcen``, ``x_cen``, ``xpos``,
                ``x_pos``, ``x_0``, and ``x0``.

              * ``y_init``, ``yinit``, ``ycentroid``, ``y_centroid``,
                ``y_peak``, ``y``, ``ycen``, ``y_cen``, ``ypos``,
                ``y_pos``, ``y_0``, and ``y0``.

              * ``flux_init``, ``flux``, ``source_sum``,
                ``segment_flux``, and ``kron_flux``.

            The parameter names are searched in the input table in the
            above order, stopping at the first match.

            The table can also have ``group_id`` and ``local_bkg``
            columns. If ``group_id`` is input, the values will be used
            and ``grouper`` keyword will be ignored. If ``local_bkg`` is
            input, they will be used and the ``localbkg_estimator`` will
            be ignored.

        Returns
        -------
        table : `~astropy.table.QTable`
            An astropy table with the PSF-fitting results. The table
            will contain the following columns:

              * ``id`` : unique identification number for the source
              * ``group_id`` : unique identification number for the
                source group
              * ``x_init``, ``x_fit``, ``x_err`` : the initial, fit, and
                error of the source x center
              * ``y_init``, ``y_fit``, ``y_err`` : the initial, fit, and
                error of the soruce y center
              * ``flux_init``, ``flux_fit``, ``flux_err`` : the initial,
                fit, and error of the source flux
              * ``npixfit`` : the number of unmasked pixels used to fit
                the source
              * ``group_size`` : the total number of sources that were
                simultaneously fit along with the given source
              * ``qfit`` : a quality-of-fit metric defined as the
                absolute value of the sum of the fit residuals divided by
                the fit flux
              * ``cfit`` : a quality-of-fit metric defined as the
                fit residual in the central pixel divided by the fit flux
              * ``flags`` : bitwise flag values:
                  * 1 : one or more pixels in the ``fit_shape`` region
                    were masked
                  * 2 : the fit x and/or y position lies outside of the
                    input data
                  * 4 : the fit flux is less than or equal to zero
                  * 8 : the fitter may not have converged
                  * 16 : the fitter parameter covariance matrix was not
                    returned
        """
        if isinstance(data, NDData):
            data_ = data.data
            if data.unit is not None:
                data_ <<= data.unit
            mask = data.mask
            unc = data.uncertainty
            if unc is not None:
                error = unc.represent_as(StdDevUncertainty).quantity
                if error.unit is u.dimensionless_unscaled:
                    error = error.value
                else:
                    error = error.to(data.unit)
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

        if init_params is None:  # no sources detected
            # TODO: raise warning
            return None

        _, counts = np.unique(init_params['group_id'], return_counts=True)
        if max(counts) > 25:
            warnings.warn('Some groups have more than 25 sources. Fitting '
                          'such groups may take a long time and be '
                          'error-prone. You may want to consider using '
                          'different `SourceGrouper` parameters or '
                          'changing the "group_id" column in "init_params".',
                          AstropyUserWarning)

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

        npixfit = np.array(self._ungroup(self._group_results['npixfit']))
        self.fit_results['npixfit'] = npixfit
        source_tbl['npixfit'] = npixfit

        nmodels = np.array(self._ungroup(self._group_results['nmodels']))
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
        attrs = ('fit_shape', 'fitter_maxiters', 'aperture_radius',
                 'progress_bar')
        for attr in attrs:
            meta[attr] = getattr(self, attr)
        source_tbl.meta = meta

        if len(self.fit_results['fit_error_indices']) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table.',
                          AstropyUserWarning)

        return source_tbl

    def make_model_image(self, shape, psf_shape):
        """
        Create a 2D image from the fit PSF models and local background.

        Parameters
        ----------
        shape : 2 tuple of int
            The shape of the output array.

        psf_shape : 2 tuple of int
            The shape of region around the center of the fit model to
            render in the output image.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The rendered image from the fit PSF models.
        """
        fit_models = self._fit_models

        data = np.zeros(shape)
        xname, yname = self._get_psf_param_names()[0][0:2]

        if self.progress_bar:
            desc = 'Model image'
            fit_models = add_progress_bar(fit_models, desc=desc)  # pragma: no cover

        # fit_models must be a list of individual, not grouped, PSF
        # models, i.e., there should be one PSF model (which may be
        # compound) for each source
        for fit_model, local_bkg in zip(fit_models,
                                        self.fit_results['local_bkg']):
            x0 = getattr(fit_model, xname).value
            y0 = getattr(fit_model, yname).value
            try:
                slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0),
                                           mode='trim')
            except NoOverlapError:
                continue
            yy, xx = np.mgrid[slc_lg]
            data[slc_lg] += (fit_model(xx, yy) + local_bkg)

        return data

    def make_residual_image(self, data, psf_shape):
        """
        Create a 2D residual image from the fit PSF models and local
        background.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which photometry was performed. This should
            be the same array input when calling the PSF-photometry
            class.

        psf_shape : 2 tuple of int
            The shape of region around the center of the fit model to
            subtract.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The residual image of the ``data`` minus the ``local_bkg``
            minus the fit PSF models.
        """
        if isinstance(data, NDData):
            residual = deepcopy(data)
            residual.data[:] = self.make_residual_image(data.data, psf_shape)
        else:
            unit = None
            if isinstance(data, u.Quantity):
                unit = data.unit
                data = data.value
            residual = -self.make_model_image(data.shape, psf_shape)
            residual += data

            if unit is not None:
                residual <<= unit

        return residual


class IterativePSFPhotometry:
    """
    Class to iteratively perform PSF photometry.

    This class implements a flexible PSF photometry algorithm that can
    find sources in an image, group overlapping sources, fit the PSF
    model to the sources, subtract the fit PSF models from the image,
    and then repeat until no more stars are detected or a given number
    of maximum iterations is reached.

    Parameters
    ----------
    psf_model : `astropy.modeling.Fittable2DModel`
        The PSF model to fit to the data. The model needs to have
        three parameters named ``x_0``, ``y_0``, and ``flux``,
        corresponding to the center (x, y) position and flux.

    fit_shape : int or length-2 array_like
        The rectangular shape around the center of a star that will
        be used to define the PSF-fitting data. If ``fit_shape`` is a
        scalar then a square shape of size ``fit_shape`` will be used.
        If ``fit_shape`` has two elements, they must be in ``(ny,
        nx)`` order. Each element of ``fit_shape`` must be an odd
        number. In general, ``fit_shape`` should be set to a small size
        (e.g., ``(5, 5)``) that covers the region with the highest flux
        signal-to-noise.

    finder : callable or `~photutils.detection.StarFinderBase`
        A callable used to identify stars in an image. The
        ``finder`` must accept a 2D image as input and return a
        `~astropy.table.Table` containing the x and y centroid
        positions. These positions are used as the starting points
        for the PSF fitting. The allowed ``x`` column names are (same
        suffix for ``y``): ``'x_init'``, ``'xinit'``, ``'xcentroid'``,
        ``'x_centroid'``, ``'x_peak'``, ``'x'``, ``'xcen'``,
        ``'x_cen'``, ``'xpos'``, ``'x_pos'``, ``'x_0'``, and ``'x0'``.
        If `None`, then the initial (x, y) model positions must be input
        using the ``init_params`` keyword when calling the class. The
        (x, y) values in ``init_params`` override this keyword *only for
        the first iteration*.

    grouper : `~photutils.psf.SourceGrouper` or callable or `None`, optional
        A callable used to group stars. Typically, grouped stars are
        those that overlap with their neighbors. Stars that are grouped
        are fit simultaneously. The ``grouper`` must accept the x and
        y coordinates of the sources and return an integer array of
        the group id numbers (starting from 1) indicating the group
        in which a given source belongs. If `None`, then no grouping
        is performed, i.e. each source is fit independently. The
        ``group_id`` values in ``init_params`` override this keyword
        *only for the first iteration*. A warning is raised if any group
        size is larger than 25 sources.

    fitter : `~astropy.modeling.fitting.Fitter`, optional
        The fitter object used to perform the fit of the model to the
        data.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each source.

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, optional
        The object used to estimate the local background around each
        source.  If `None`, then no local background is subtracted.  The
        ``local_bkg`` values in ``init_params`` override this keyword.

    aperture_radius : float, optional
        The radius of the circular aperture used to estimate the initial
        flux of each source. The ``flux_init`` values in ``init_params``
        override this keyword *only for the first iteration*.

    sub_shape : `None`, int, or length-2 array_like
        The rectangular shape around the center of a star that will be
        used when subtracting the fitted PSF models. If ``sub_shape`` is
        a scalar then a square shape of size ``sub_shape`` will be used.
        If ``sub_shape`` has two elements, they must be in ``(ny, nx)``
        order. Each element of ``sub_shape`` must be an odd number. If
        `None`, then ``sub_shape`` is set to ``fit_shape``.

    progress_bar : bool, optional
        Whether to display a progress bar when fitting the sources
        (or groups). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.
        Note that the progress bar does not currently work in the
        Jupyter console due to limitations in ``tqdm``.
    """

    def __init__(self, psf_model, fit_shape, finder, *, grouper=None,
                 fitter=LevMarLSQFitter(), fitter_maxiters=100, maxiters=3,
                 localbkg_estimator=None, aperture_radius=None,
                 sub_shape=None, progress_bar=False):

        if finder is None:
            raise ValueError('finder cannot be None for '
                             'IterativePSFPhotometry.')

        if aperture_radius is None:
            raise ValueError('aperture_radius cannot be None for '
                             'IterativePSFPhotometry.')

        self.maxiters = self._validate_maxiters(maxiters)

        self.psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, fitter=fitter,
                                     fitter_maxiters=fitter_maxiters,
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=aperture_radius,
                                     progress_bar=progress_bar)

        self.fit_results = []
        if sub_shape is None:
            sub_shape = fit_shape
        self.sub_shape = as_pair('sub_shape', sub_shape, lower_bound=(0, 1),
                                 check_odd=True)

    @staticmethod
    def _validate_maxiters(maxiters):
        if (not np.isscalar(maxiters) or maxiters <= 0
                or ~np.isfinite(maxiters)):
            raise ValueError('maxiters must be a strictly-positive scalar')
        if maxiters != int(maxiters):
            raise ValueError('maxiters must be an integer')
        return maxiters

    def __call__(self, data, *, mask=None, error=None, init_params=None):
        """
        Perform PSF photometry.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which to perform photometry. Invalid data
            values (i.e., NaN or inf) are automatically masked.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.

        error : 2D `~numpy.ndarray`, optional
            The pixel-wise 1-sigma errors of the input ``data``.
            ``error`` is assumed to include *all* sources of
            error, including the Poisson error of the sources
            (see `~photutils.utils.calc_total_error`) . ``error``
            must have the same shape as the input ``data``. If a
            `~astropy.units.Quantity` array, then ``data`` must also be
            a `~astropy.units.Quantity` array with the same units.

        init_params : `~astropy.table.Table` or `None`, optional
            A table containing the initial guesses of the (x, y, flux)
            model parameters for each source *only for the first
            iteration*. If the x and y values are not input, then the
            ``finder`` will be used for all iterations. If the flux
            values are not input, then the initial fluxes will be
            measured in using the ``aperture_radius`` keyword. The input
            flux values will be used for the first iteration only. The
            allowed column names are:

              * ``x_init``, ``xinit``, ``xcentroid``, ``x_centroid``,
                ``x_peak``, ``x``, ``xcen``, ``x_cen``, ``xpos``,
                ``x_pos``, ``x_0``, and ``x0``.

              * ``y_init``, ``yinit``, ``ycentroid``, ``y_centroid``,
                ``y_peak``, ``y``, ``ycen``, ``y_cen``, ``ypos``,
                ``y_pos``, and ``y_0``, and ``y0``.

              * ``flux_init``, ``flux``, ``source_sum``,
                ``segment_flux``, and ``kron_flux``.

            The parameter names are searched in the input table in the
            above order, stopping at the first match.

        Returns
        -------
        table : `~astropy.table.QTable`
            An astropy table with the PSF-fitting results. The table
            will contain the following columns:

              * ``id`` : unique identification number for the source
              * ``group_id`` : unique identification number for the
                source group
              * ``iter_detected`` : the iteration number in which the
                source was detected
              * ``x_init``, ``x_fit``, ``x_err`` : the initial, fit, and
                error of the source x center
              * ``y_init``, ``y_fit``, ``y_err`` : the initial, fit, and
                error of the soruce y center
              * ``flux_init``, ``flux_fit``, ``flux_err`` : the initial,
                fit, and error of the source flux
              * ``npixfit`` : the number of unmasked pixels used to fit
                the source
              * ``group_size`` : the total number of sources that were
                simultaneously fit along with the given source
              * ``qfit`` : a quality-of-fit metric defined as the
                absolute value of the sum of the fit residuals divided by
                the fit flux
              * ``cfit`` : a quality-of-fit metric defined as the
                fit residual in the central pixel divided by the fit flux
              * ``flags`` : bitwise flag values:
                  * 1 : one or more pixels in the ``fit_shape`` region
                    were masked
                  * 2 : the fit x and/or y position lies outside of the
                    input data
                  * 4 : the fit flux is less than or equal to zero
                  * 8 : the fitter may not have converged
                  * 16 : the fitter parameter covariance matrix was not
                    returned
        """
        phot_tbl = self.psfphot(data, mask=mask, error=error,
                                init_params=init_params)
        if phot_tbl is None:
            return None
        self.fit_results.append(deepcopy(self.psfphot))

        phot_tbl['iter_detected'] = 1

        resid = []

        iter_num = 2
        while iter_num <= self.maxiters and phot_tbl is not None:
            if iter_num == 2:
                residual_data = self.psfphot.make_residual_image(
                    data, self.sub_shape)
            else:
                residual_data = self.psfphot.make_residual_image(
                    residual_data, self.sub_shape)

            resid.append(residual_data)

            # do not warn if no sources are found beyond the first iteration
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NoDetectionsWarning)
                new_tbl = self.psfphot(residual_data, mask=mask, error=error,
                                       init_params=None)

            if new_tbl is None:  # no new sources detected
                break

            self.fit_results.append(deepcopy(self.psfphot))

            new_tbl['iter_detected'] = iter_num
            new_tbl['id'] += np.max(phot_tbl['id'])
            new_tbl['group_id'] += np.max(phot_tbl['group_id'])
            new_tbl.meta = {}  # prevent merge conflicts

            # combine tables
            phot_tbl = vstack([phot_tbl, new_tbl])

            iter_num += 1

        # re-order 'iter_detected' column
        colnames = phot_tbl.colnames.copy()
        colnames.insert(2, 'iter_detected')
        colnames = colnames[:-1]
        phot_tbl = phot_tbl[colnames]

        return phot_tbl

    def make_model_image(self, shape, psf_shape):
        """
        Create a 2D image from the fit PSF models and local background.

        Parameters
        ----------
        shape : 2 tuple of int
            The shape of the output array.

        psf_shape : 2 tuple of int
            The shape of region around the center of the fit model to
            render in the output image.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The rendered image from the fit PSF models.
        """
        fit_models = []
        local_bkgs = []
        for psfphot in self.fit_results:
            fit_models.append(psfphot._fit_models)
            local_bkgs.append(psfphot.fit_results['local_bkg'])

        fit_models = _flatten(fit_models)
        local_bkgs = _flatten(local_bkgs)

        data = np.zeros(shape)
        xname, yname = self.fit_results[0]._get_psf_param_names()[0][0:2]

        if self.psfphot.progress_bar:
            desc = 'Model image'
            fit_models = add_progress_bar(fit_models, desc=desc)  # pragma: no cover

        # fit_models must be a list of individual, not grouped, PSF
        # models, i.e., there should be one PSF model (which may be
        # compound) for each source
        for fit_model, local_bkg in zip(fit_models, local_bkgs):
            x0 = getattr(fit_model, xname).value
            y0 = getattr(fit_model, yname).value
            try:
                slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0),
                                           mode='trim')
            except NoOverlapError:
                continue
            yy, xx = np.mgrid[slc_lg]
            data[slc_lg] += (fit_model(xx, yy) + local_bkg)

        return data

    def make_residual_image(self, data, psf_shape):
        """
        Create a 2D residual image from the fit PSF models and local
        background.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which photometry was performed. This should
            be the same array input when calling the PSF-photometry
            class.

        psf_shape : 2 tuple of int
            The shape of region around the center of the fit model to
            subtract.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The residual image of the ``data`` minus the ``local_bkg``
            minus the fit PSF models.
        """
        if isinstance(data, NDData):
            residual = deepcopy(data)
            residual.data[:] = self.make_residual_image(data.data, psf_shape)
        else:
            unit = None
            if isinstance(data, u.Quantity):
                unit = data.unit
                data = data.value
            residual = -self.make_model_image(data.shape, psf_shape)
            residual += data

            if unit is not None:
                residual <<= unit

        return residual


def _flatten(iterable):
    """
    Flatten a list of lists.
    """
    return list(chain.from_iterable(iterable))
