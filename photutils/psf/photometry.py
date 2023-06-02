# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable2DModel
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

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
                 fitter=LevMarLSQFitter(), aperture_radius=None):

        self.psf_model = self._validate_model(psf_model)
        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(0, 1),
                                 check_odd=True)
        self.grouper = self._validate_callable(grouper, 'grouper')
        self.finder = self._validate_callable(finder, 'finder')
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.aperture_radius = self._validate_radius(aperture_radius)

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

    def _make_init_params(self, data, mask, sources):
        xpos = sources['xcentroid']
        ypos = sources['ycentroid']
        apertures = CircularAperture(zip(xpos, ypos), r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)

        init_params = QTable()
        init_params['id'] = np.arange(len(xpos)) + 1
        init_params['x_init'] = xpos
        init_params['y_init'] = ypos
        init_params['flux_init'] = flux

        return init_params

    def _fit_stars(self, data, init_params, *, mask=None,
                   progress_bar=None):
        pass

    def __call__(self, data, *, mask=None, init_params=None,
                 progress_bar=False):
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

            init_params = self._make_init_params(sources)

        # TODO: group stars
        #if grouper is not None:
        #    init_params['group_id'] = ...

        fit_stars = self._fit_stars(data, init_params, mask=mask,
                                    progress_bar=progress_bar)

        # TODO: create output table

        return fit_stars
