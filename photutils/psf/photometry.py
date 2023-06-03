# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import warnings

import numpy as np
from astropy.modeling import Fittable2DModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable, Table
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
                 fitter=LevMarLSQFitter(), aperture_radius=None,
                 progress_bar=None):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_callable(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self._fitted_group_models = None
        self._fitted_models = None

    @staticmethod
    def _validate_model(psf_model):
        if not isinstance(psf_model, Fittable2DModel):
            raise TypeError('psf_model must be an astropy Fittable2DModel')
        return psf_model

    @staticmethod
    def _validate_callable(obj, name):
        if obj is not None and not callable(obj):
            raise TypeError(f'{name!r} must be a callable object')
        return obj

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
    def _validate_params(init_params):
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            raise TypeError('init_params must be an astropy Table')

        columns = ('x_init', 'y_init')
        for column in columns:
            if column not in init_params.columns:
                raise ValueError(f'{column!r} must be a column in '
                                 'init_params')

        return init_params.copy()

    def _get_aper_fluxes(self, data, mask, init_params):
        # TODO: flexible input column names
        xpos = init_params['x_init']
        ypos = init_params['y_init']
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
        # TODO: flexible finder column names
        init_params['x_init'] = sources['xcentroid']
        init_params['y_init'] = sources['ycentroid']
        init_params['flux_init'] = self._get_aper_fluxes(data, mask,
                                                         init_params)

        return init_params

    def _add_progress_bar(self, sources, desc=None):
        if self.progress_bar and HAS_TQDM:
            from tqdm.auto import tqdm  # pragma: no cover

            sources = tqdm(sources, desc=desc)  # pragma: no cover

        return sources

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources within
        a group.
        """
        # TODO: generalize this mapping
        par_map = {}
        par_map['x_init'] = 'x_0'
        par_map['y_init'] = 'y_0'
        par_map['flux_init'] = 'flux'

        for i, source in enumerate(sources):
            model = self.psf_model.copy()
            for param, model_param in par_map.items():
                setattr(model, model_param, source[param])
                model.name = source['id']

            if i == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def _define_fit_coords(self, sources, mask):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf

        hshape = (self.fit_shape - 1) // 2
        yi = []
        xi = []
        for row in sources:
            # bbox "slice indices" (max is non-inclusive)
            xcen = int(row['x_init'] + 0.5)
            ycen = int(row['y_init'] + 0.5)
            xmin = min((xmin, xcen - hshape[1]))
            xmax = max((xmax, xcen + hshape[1] + 1))
            ymin = min((ymin, ycen - hshape[0]))
            ymax = max((ymax, ycen + hshape[0] + 1))
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            xi.append(xx)
            yi.append(yy)

        xi = np.array(xi).ravel()
        yi = np.array(yi).ravel()
        # find unique (x, y) pairs
        yi, xi = np.unique(np.column_stack((yi, xi)), axis=0).T
        # yi, xi = np.array(list(set(zip(yi, xi)))).T

        if mask is not None:
            inv_mask = ~mask[yi, xi]
            yi = yi[inv_mask]
            xi = xi[inv_mask]

        return yi, xi

    @staticmethod
    def _split_compound_model(model, chunk_size):
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i: i + chunk_size]

    def _split_grouped_models(self, models):
        psf_nsub = self.psf_model.n_submodels

        ungrouped_models = []
        for model in models:
            model_nsub = model.n_submodels

            if model_nsub == psf_nsub:
                # model for a single star (which may be compound)
                ungrouped_models.append(model)
                continue

            # model is a grouped model for multiple stars
            ungrouped_models.extend(self._split_compound_model(model,
                                                               psf_nsub))

        # sorted back into original source order
        return sorted(ungrouped_models, key=lambda model: model.name)

    def _fit_sources(self, data, init_params, *, mask=None):
        sources = init_params.group_by('group_id').groups
        sources = self._add_progress_bar(sources, desc='Fit star/group')

        fitted_models = []
        for sources_ in sources:
            psf_model = self._make_psf_model(sources_)
            yi, xi = self._define_fit_coords(sources_, data.shape, mask)
            cutout = data[yi, xi]
            fitted_models.append(self.fitter(psf_model, xi, yi, cutout))

        return fitted_models

    def __call__(self, data, *, mask=None, init_params=None):
        """
        Perform PSF photometry.
        """
        (data,), unit = process_quantities((data,), ('data',))
        data = self._validate_array(data, 'data')
        mask = self._make_mask(data,
                               self._validate_array(mask, 'mask',
                                                    data_shape=data.shape))
        init_params = self._validate_params(init_params)  # also copies

        if (self.aperture_radius is None
            and (init_params is None
                 or 'flux_init' not in init_params.colnames)):
            raise ValueError('aperture_radius must be defined if init_params '
                             'is not input or if a "flux_init" column is '
                             'not in init_params')

        if init_params is None:
            if self.finder is None:
                raise ValueError('finder must be defined if init_params '
                                 'is not input')

            # TODO: save sources table in dict
            sources = self.finder(data, mask=mask)
            if sources is None:
                return None

            init_params = self._make_init_params(data, mask, sources)
        else:
            colnames = init_params.colnames
            if 'id' not in colnames:
                init_params['id'] = np.arange(len(init_params)) + 1

            if 'flux_init' not in colnames:
                init_params['flux_init'] = self._get_aper_fluxes(data, mask,
                                                                 init_params)

            if 'group_id' in colnames:
                # grouper is ignored if group_id is input in init_params
                self.grouper = None

        if self.grouper is not None:
            # TODO: change grouper API
            # init_params['group_id'] = self.grouper(init_params)
            init_params = self.grouper(init_params)

        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        fitted_models = self._fit_sources(data, init_params, mask=mask)
        ungrouped_models = self._split_grouped_models(fitted_models)
        self._fitted_group_models = fitted_models
        self._fitted_models = ungrouped_models

        # TODO: create output table
        fitted_stars = QTable()

        return fitted_stars
