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
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData, NoOverlapError, StdDevUncertainty
from astropy.table import QTable, Table, hstack, join, vstack
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.background import LocalBackground
from photutils.datasets import make_model_image as _make_model_image
from photutils.psf.groupers import SourceGrouper
from photutils.psf.utils import _get_psf_model_params, _validate_psf_model
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils.cutouts import _overlap_slices as overlap_slices
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['ModelImageMixin', 'PSFPhotometry', 'IterativePSFPhotometry']


class ModelImageMixin:
    """
    Mixin class to provide methods to calculate model images and
    residuals.
    """

    def make_model_image(self, shape, psf_shape, *, include_localbkg=False):
        """
        Create a 2D image from the fit PSF models and optional local
        background.

        Parameters
        ----------
        shape : 2 tuple of int
            The shape of the output array.

        psf_shape : 2 tuple of int
            The shape of region around the center of the fit model to
            render in the output image.

        include_localbkg : bool, optional
            Whether to include the local background in the rendered
            output image. Note that the local background level is
            included around each source over the region defined by
            ``psf_shape``. Thus, regions where the ``psf_shape`` of
            sources overlap will have the local background added
            multiple times.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The rendered image from the fit PSF models. This image will
            not have any units.
        """
        if isinstance(self, PSFPhotometry):
            progress_bar = self.progress_bar
            psf_model = self.psf_model
            fit_params = self._fit_params
            local_bkgs = self.init_params['local_bkg']
        else:
            psf_model = self._psfphot.psf_model
            progress_bar = self._psfphot.progress_bar

            if self.mode == 'new':
                # collect the fit params and local backgrounds from each
                # iteration
                local_bkgs = []
                for i, psfphot in enumerate(self.fit_results):
                    if i == 0:
                        fit_params = psfphot._fit_params
                    else:
                        fit_params = vstack((fit_params, psfphot._fit_params))
                    local_bkgs.append(psfphot.init_params['local_bkg'])

                local_bkgs = _flatten(local_bkgs)
            else:
                # use the fit params and local backgrounds only from the
                # final iteration, which includes all sources
                fit_params = self.fit_results[-1]._fit_params
                local_bkgs = self.fit_results[-1].init_params['local_bkg']

        model_params = fit_params

        if include_localbkg:
            if isinstance(local_bkgs, u.Quantity):
                local_bkgs = local_bkgs.value

            # add local_bkg
            model_params = model_params.copy()
            model_params['local_bkg'] = local_bkgs

        return _make_model_image(shape, psf_model, model_params,
                                 model_shape=psf_shape,
                                 progress_bar=progress_bar)

    def make_residual_image(self, data, psf_shape, *, include_localbkg=False):
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

        include_localbkg : bool, optional
            Whether to include the local background in the subtracted
            model. Note that the local background level is subtracted
            around each source over the region defined by ``psf_shape``.
            Thus, regions where the ``psf_shape`` of sources overlap
            will have the local background subtracted multiple times.

        Returns
        -------
        array : 2D `~numpy.ndarray`
            The residual image of the ``data`` minus the ``local_bkg``
            minus the fit PSF models.
        """
        if isinstance(data, NDData):
            residual = deepcopy(data)
            residual.data[:] = self.make_residual_image(
                data.data, psf_shape, include_localbkg=include_localbkg)
        else:
            unit = None
            if isinstance(data, u.Quantity):
                unit = data.unit
                data = data.value
            residual = self.make_model_image(data.shape, psf_shape,
                                             include_localbkg=include_localbkg)
            np.subtract(data, residual, out=residual)

            if unit is not None:
                residual <<= unit

        return residual


class PSFPhotometry(ModelImageMixin):
    """
    Class to perform PSF photometry.

    This class implements a flexible PSF photometry algorithm that can
    find sources in an image, group overlapping sources, fit the PSF
    model to the sources, and subtract the fit PSF models from the
    image.

    Parameters
    ----------
    psf_model : 2D `astropy.modeling.Model`
        The PSF model to fit to the data. The model must have parameters
        named ``x_0``, ``y_0``, and ``flux``, corresponding to the
        center (x, y) position and flux, or it must have 'x_name',
        'y_name', and 'flux_name' attributes that map to the x, y, and
        flux parameters (i.e., a model output from `make_psf_model`).
        The model must be two-dimensional such that it accepts 2 inputs
        (e.g., x and y) and provides 1 output.

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
        positions. These positions are used as the starting points for
        the PSF fitting. The allowed ``x`` column names are (same suffix
        for ``y``): ``'x_init'``, ``'xinit'``, ``'x'``, ``'x_0'``,
        ``'x0'``, ``'xcentroid'``, ``'x_centroid'``, ``'x_peak'``,
        ``'xcen'``, ``'x_cen'``, ``'xpos'``, ``'x_pos'``, ``'x_fit'``,
        and ``'xfit'``. If `None`, then the initial (x, y) model
        positions must be input using the ``init_params`` keyword
        when calling the class. The (x, y) values in ``init_params``
        override this keyword. If this class is run on an image that
        has units (i.e., a `~astropy.units.Quantity` array), then
        certain ``finder`` keywords (e.g., ``threshold``) must have the
        same units. Please see the the documentation for the specific
        ``finder`` class for more information.

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
        called for each source. This keyword can be increased if the fit
        is not converging for sources (e.g., the output ``flags`` value
        contains 8).

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, optional
        The object used to estimate the local background around each
        source. If `None`, then no local background is subtracted. The
        ``local_bkg`` values in ``init_params`` override this keyword.
        This option should be used with care, especially in crowded
        fields where the ``fit_shape`` of sources overlap (see Notes
        below).

    aperture_radius : float, optional
        The radius of the circular aperture used to estimate the initial
        flux of each source. If initial flux values are present in the
        ``init_params`` table, they will override this keyword.

    progress_bar : bool, optional
        Whether to display a progress bar when fitting the sources
        (or groups). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.
        Note that the progress bar does not currently work in the
        Jupyter console due to limitations in ``tqdm``.

    Notes
    -----
    The data that will be fit for each source is defined by the
    ``fit_shape`` parameter. A cutout will be made around the initial
    center of each source with a shape defined by ``fit_shape``. The PSF
    model will be fit to the data in this region. The cutout region that
    is fit does not shift if the source center shifts during the fit
    iterations. Therefore, the initial source positions should be close
    to the true source positions. One way to ensure this is to use a
    ``finder`` to identify sources in the data.

    If the fitted positions are significantly different from the initial
    positions, one can re-run the `PSFPhotometry` class using the fit
    results as the input ``init_params``, which will change the fitted
    cutout region for each source. After calling `PSFPhotometry` on the
    data, it will have a ``fit_params`` attribute containing the fitted
    model parameters. This table can be used as the ``init_params``
    input in a subsequent call to `PSFPhotometry`.

    If the returned model parameter errors are NaN, then either
    the fit did not converge, the model parameter was fixed, or
    the input ``fitter`` did not return parameter errors. For the
    later case, one can try a different fitter that may return
    parameter errors (e.g., `astropy.models.fitting.LMLSQFitter
    or `astropy.models.fitting.TRFLSQFitter`). Note that
    these fitters are typically slower than the default
    `astropy.models.fitting.LevMarLSQFitter`.

    The local background value around each source is optionally
    estimated using the ``localbkg_estimator`` or obtained from the
    ``local_bkg`` column in the input ``init_params`` table. This local
    background is then subtracted from the data over the ``fit_shape``
    region for each source before fitting the PSF model. For sources
    where their ``fit_shape`` regions overlap, the local background will
    effectively be subtracted twice in the overlapping ``fit_shape``
    regions, even if the source ``grouper`` is input. This is not an
    issue if the sources are well-separated. However, for crowded
    fields, please use the ``localbkg_estimator`` (or ``local_bkg``
    column in ``init_params``) with care.
    """

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=LevMarLSQFitter(), fitter_maxiters=100,
                 localbkg_estimator=None, aperture_radius=None,
                 progress_bar=False):

        self.psf_model = psf_model
        self._validate_psf_model()  # validate the PSF model

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

        # reset these attributes for each __call__ (see _reset_results)
        self.finder_results = None
        self.init_params = None
        self.fit_params = None
        self._fit_params = None
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)

    def _reset_results(self):
        self.finder_results = None
        self.init_params = None
        self.fit_params = None
        self._fit_params = None
        self.fit_results = defaultdict(list)
        self._group_results = defaultdict(list)

    def _validate_grouper(self, grouper, name):
        if grouper is not None and not isinstance(grouper, SourceGrouper):
            raise ValueError('grouper must be a SourceGrouper instance.')
        return grouper

    @lazyproperty
    def _psf_param_names(self):
        """
        The PSF model parameters corresponding to x, y, and flux.

        The parameter options are checked together as a unit in the
        following order:

            * ('x_0', 'y_0', 'flux') parameters
            * ('x_name', 'y_name', 'flux_name') attributes
        """
        return _get_psf_model_params(self.psf_model)

    @lazyproperty
    def _fitted_psf_param_names(self):
        """
        All PSF model parameters that are fit.
        """
        fitted_params = []
        for key, val in self.psf_model.fixed.items():
            if not val:
                fitted_params.append(key)
        return fitted_params

    @lazyproperty
    def _extra_psf_param_names(self):
        """
        PSF model parameters that are fit, but do not correspond to x,
        y, or flux.

        The order of the psf_model parameters is preserved.
        """
        extra_params = []
        for key in self._fitted_psf_param_names:
            if key not in self._psf_param_names:
                extra_params.append(key)
        return extra_params

    def _validate_psf_model(self):
        """
        Validate the input PSF model.

        The PSF model must be a subclass of `astropy.modeling.Model`. It
        must also be two-dimensional and have a single output.

        The PSF model must have parameters called 'x_0', 'y_0', and
        'flux' or it must have 'x_name', 'y_name', and 'flux_name'
        attributes (i.e., output from `make_psf_model`). Otherwise, a
        `ValueError` is raised.
        """
        _validate_psf_model(self.psf_model)
        self._psf_param_names  # validate the PSF model parameters

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

    @lazyproperty
    def _init_colnames(self):
        """
        A dictionary of column names for the initial x, y, and flux values
        reported in the output table.
        """
        suffix = '_init'
        init_colnames = {}
        init_colnames['suffix'] = suffix
        init_colnames['x'] = f'x{suffix}'
        init_colnames['y'] = f'y{suffix}'
        init_colnames['flux'] = f'flux{suffix}'
        return init_colnames

    @lazyproperty
    def _valid_colnames(self):
        """
        A dictionary of valid column names for the input ``init_params``
        table.

        These lists are searched in order.
        """
        xy_suffixes = ('_init', 'init', '', '_0', '0', 'centroid', '_centroid',
                       '_peak', 'cen', '_cen', 'pos', '_pos', '_fit', 'fit')
        x_valid = ['x' + i for i in xy_suffixes]
        y_valid = ['y' + i for i in xy_suffixes]

        valid_colnames = {}
        valid_colnames['x'] = x_valid
        valid_colnames['y'] = y_valid
        valid_colnames['flux'] = ('flux_init', 'fluxinit', 'flux', 'flux_0',
                                  'flux0', 'flux_fit', 'fluxfit', 'source_sum',
                                  'segment_flux', 'kron_flux')

        return valid_colnames

    def _find_column_name(self, key, colnames):
        """
        Find the first valid matching column name for x, y, or flux
        (defined by `_valid_colnames` in the input ``init_params``
        table).
        """
        name = ''
        valid_names = self._valid_colnames[key]
        for valid_name in valid_names:
            if valid_name in colnames:
                name = valid_name
                break
        return name

    def _check_init_units(self, init_params, colname, data_unit):
        values = init_params[colname]
        if isinstance(values, u.Quantity):
            if data_unit is None:
                raise ValueError(f'init_params {colname} column has '
                                 'units, but the input data does not '
                                 'have units.')
            try:
                init_params[colname] = values.to(data_unit)
            except u.UnitConversionError as exc:
                raise ValueError(f'init_params {colname} column has '
                                 'units that are incompatible with '
                                 'the input data units.') from exc
        else:
            if data_unit is not None:
                raise ValueError('The input data has units, but the '
                                 f'init_params {colname} column does not '
                                 'have units.')

        return init_params

    def _validate_init_params(self, init_params, data_unit):
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            raise TypeError('init_params must be an astropy Table')

        xcolname = self._find_column_name('x', init_params.colnames)
        ycolname = self._find_column_name('y', init_params.colnames)
        if not xcolname or not ycolname:
            raise ValueError('init_param must contain valid column names '
                             'for the x and y source positions')

        init_params = init_params.copy()  # preserve input init_params
        xinit_name = self._init_colnames['x']
        yinit_name = self._init_colnames['y']
        if xcolname != xinit_name:
            init_params.rename_column(xcolname, xinit_name)
        if ycolname != yinit_name:
            init_params.rename_column(ycolname, yinit_name)

        fluxcolname = self._find_column_name('flux', init_params.colnames)
        if fluxcolname:
            fluxinit_name = self._init_colnames['flux']
            if fluxcolname != fluxinit_name:
                init_params.rename_column(fluxcolname, fluxinit_name)

            init_params = self._check_init_units(init_params, fluxinit_name,
                                                 data_unit)

        if 'local_bkg' in init_params.colnames:
            if not np.all(np.isfinite(init_params['local_bkg'])):
                raise ValueError('init_params local_bkg column contains '
                                 'non-finite values.')
            init_params = self._check_init_units(init_params, 'local_bkg',
                                                 data_unit)

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
        xpos = init_params[self._init_colnames['x']]
        ypos = init_params[self._init_colnames['y']]
        apertures = CircularAperture(zip(xpos, ypos), r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)
        return flux

    def _prepare_init_params(self, data, unit, mask, init_params):
        if init_params is None:
            if self.finder is None:
                raise ValueError('finder must be defined if init_params '
                                 'is not input')

            if unit is not None:
                sources = self.finder(data << unit, mask=mask)
            else:
                sources = self.finder(data, mask=mask)
            self.finder_results = sources
            if sources is None:
                return None

            init_params = QTable()
            init_params['id'] = np.arange(len(sources)) + 1
            init_params[self._init_colnames['x']] = sources['xcentroid']
            init_params[self._init_colnames['y']] = sources['ycentroid']

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
                    data, init_params[self._init_colnames['x']],
                    init_params[self._init_colnames['y']], mask=mask)
            init_params['local_bkg'] = local_bkg

        if self._init_colnames['flux'] not in init_params.colnames:
            flux = self._get_aper_fluxes(data, mask, init_params)
            flux -= init_params['local_bkg']
            if unit is not None:
                flux <<= unit
            init_params[self._init_colnames['flux']] = flux

        if self.grouper is not None:
            init_params['group_id'] = self.grouper(
                init_params['x_init'], init_params['y_init'])

        # no grouping
        if 'group_id' not in init_params.colnames:
            init_params['group_id'] = init_params['id']

        extra_param_cols = []
        init_param_map = self._param_maps['init']
        for extra_param in self._extra_psf_param_names:
            for key, val in init_param_map.items():
                if val == extra_param:
                    extra_param_cols.append(key)

        for extra_col in extra_param_cols:
            if extra_col not in init_params.colnames:
                init_params[extra_col] = getattr(self.psf_model,
                                                 init_param_map[extra_col])

        # order init_params columns
        colname_order = ['id', 'group_id', 'local_bkg',
                         self._init_colnames['x'], self._init_colnames['y'],
                         self._init_colnames['flux']]
        colname_order.extend(extra_param_cols)
        init_params = init_params[colname_order]

        return init_params

    def _get_invalid_positions(self, init_params, shape):
        """
        Get a mask of sources with no overlap with the data.

        This code is based on astropy.nddata.overlap_slices.
        """
        x = init_params[self._init_colnames['x']]
        y = init_params[self._init_colnames['y']]
        positions = np.column_stack((y, x))
        delta = self.fit_shape / 2
        min_idx = np.ceil(positions - delta)
        max_idx = np.ceil(positions + delta)
        mask = np.any(max_idx <= 0, axis=1) | np.any(min_idx >= shape, axis=1)
        return mask

    def _check_init_positions(self, init_params, shape):
        """
        Check the initial source positions to ensure they are within the
        data shape.
        """
        if np.any(self._get_invalid_positions(init_params, shape)):
            raise ValueError('Some of the sources have no overlap with the '
                             'data. Check the initial source positions or '
                             'increase the fit_shape.')

    @lazyproperty
    def _param_maps(self):
        """
        Map x, y, and flux column names to the PSF model parameter names.

        Also include any extra PSF model parameters that are fit, but do
        not correspond to x, y, or flux.

        The column names include the ``_init``, ``_fit``, and ``_err``
        suffixes for each parameter.
        """
        init_param_map = {}
        init_param_map[self._init_colnames['x']] = self._psf_param_names[0]
        init_param_map[self._init_colnames['y']] = self._psf_param_names[1]
        init_param_map[self._init_colnames['flux']] = self._psf_param_names[2]

        for extra_param in self._extra_psf_param_names:
            init_param_map[f'{extra_param}_init'] = extra_param

        init_suffix = self._init_colnames['suffix']
        fit_param_map = {val: key.replace(init_suffix, '_fit')
                         for key, val in init_param_map.items()}
        err_param_map = {val: key.replace(init_suffix, '_err')
                         for key, val in init_param_map.items()}

        param_maps = {}
        param_maps['init'] = init_param_map
        param_maps['fit'] = fit_param_map
        param_maps['err'] = err_param_map

        return param_maps

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources within
        a group.
        """
        init_param_map = self._param_maps['init']

        for index, source in enumerate(sources):
            model = self.psf_model.copy()
            for param, model_param in init_param_map.items():
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

    @staticmethod
    def _move_column(table, colname, colname_after):
        """
        Move a column to a new position in a table.

        The table is modified in place.

        Parameters
        ----------
        colname : str
            The column name to move.

        colname_after : str
            The column name after which to place the moved column.

        table : `~astropy.table.Table`
            The input table.

        Returns
        -------
        table : `~astropy.table.Table`
            The input table with the column moved.
        """
        colnames = table.colnames
        if colname not in colnames or colname_after not in colnames:
            return table
        if colname == colname_after:
            return table

        old_index = colnames.index(colname)
        new_index = colnames.index(colname_after)
        if old_index > new_index:
            new_index += 1
        colnames.insert(new_index, colnames.pop(old_index))
        table = table[colnames]
        return table

    @staticmethod
    def _model_params_to_table(models):
        """
        Convert a list of PSF models to a table of model parameters.

        The inputs ``models`` are assumed to be instances of the same
        model class (i.e., the parameters names are the same for all
        models).
        """
        param_names = list(models[0].param_names)
        params = defaultdict(list)
        for model in models:
            for name in param_names:
                try:
                    value = getattr(model, name).value
                except AttributeError:
                    value = getattr(model, name)
                params[name].append(value)

        table = QTable(params)
        ids = np.arange(len(table)) + 1
        table.add_column(ids, index=0, name='id')

        return table

    def _param_errors_to_table(self):
        param_err = self.fit_results.pop('fit_param_errs')

        err_param_map = self._param_maps['err']
        table = QTable()
        for index, name in enumerate(self._fitted_psf_param_names):
            colname = err_param_map[name]
            table[colname] = param_err[:, index]

        colnames = list(err_param_map.values())

        # add missing error columns
        nsources = len(self.init_params)
        for colname in colnames:
            if colname not in table.colnames:
                table[colname] = [np.nan] * nsources

        # sort column names
        return table[colnames]

    def _prepare_fit_results(self, fit_params):
        """
        Prepare the output table of fit results.
        """
        # remove parameters that are not fit
        out_params = fit_params.copy()
        for column in out_params.colnames:
            if column == 'id':
                continue
            if column not in self._param_maps['fit'].keys():
                out_params.remove_column(column)

        # rename columns to have the "fit" suffix
        for key, val in self._param_maps['fit'].items():
            out_params.rename_column(key, val)

        # reorder columns to have "flux" come immediately after "y"
        y_col = self._param_maps['fit'][self._psf_param_names[1]]
        flux_col = self._param_maps['fit'][self._psf_param_names[2]]
        out_params = self._move_column(out_params, flux_col, y_col)

        # add parameter error columns
        param_errs = self._param_errors_to_table()
        out_params = hstack([out_params, param_errs])

        return out_params

    def _define_fit_data(self, sources, data, mask):
        yi = []
        xi = []
        cutout = []
        npixfit = []
        cen_index = []
        for row in sources:
            xcen = row[self._init_colnames['x']]
            ycen = row[self._init_colnames['y']]

            try:
                slc_lg, _ = overlap_slices(data.shape, self.fit_shape,
                                           (ycen, xcen), mode='trim')
            except NoOverlapError as exc:  # pragma: no cover
                # this should never happen because the initial positions
                # are checked in _prepare_fit_inputs
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
            local_bkg = row['local_bkg']
            if isinstance(local_bkg, u.Quantity):
                local_bkg = local_bkg.value
            cutout.append(data[yy, xx] - local_bkg)
            npixfit.append(len(xx))

            # this is overlap_slices center pixel index (before any trimming)
            xcen = np.ceil(xcen - 0.5).astype(int)
            ycen = np.ceil(ycen - 0.5).astype(int)

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

    def _parse_fit_results(self, group_models, group_fit_infos):
        """
        Parse the fit results for each source or group of sources.
        """
        psf_nsub = self.psf_model.n_submodels

        fit_models = []
        fit_infos = []
        fit_param_errs = []
        nfitparam = len(self._fitted_psf_param_names)
        for model, fit_info in zip(group_models, group_fit_infos):
            model_nsub = model.n_submodels
            npsf_models = model_nsub // psf_nsub

            param_cov = fit_info.get('param_cov', None)
            if param_cov is None:
                if nfitparam == 0:  # model params are all fixed
                    nfitparam = 3   # x_err, y_err, and flux_err are np.nan
                param_err = np.array([np.nan] * nfitparam * npsf_models)
            else:
                param_err = np.sqrt(np.diag(param_cov))

            # model is for a single source (which may be compound)
            if npsf_models == 1:
                fit_models.append(model)
                fit_infos.append(fit_info)
                fit_param_errs.append(param_err)
                continue

            # model is a grouped model for multiple sources
            fit_models.extend(self._split_compound_model(model, psf_nsub))
            fit_infos.extend([fit_info] * npsf_models)  # views
            fit_param_errs.extend(self._split_param_errs(param_err, nfitparam))

        if len(fit_models) != len(fit_infos):  # pragma: no cover
            raise ValueError('fit_models and fit_infos have different lengths')

        # change the sorting from group_id to source id order
        fit_models = self._order_by_id(fit_models)
        fit_infos = self._order_by_id(fit_infos)
        fit_param_errs = np.array(self._order_by_id(fit_param_errs))

        self.fit_results['fit_infos'] = fit_infos
        self.fit_results['fit_error_indices'] = self._get_fit_error_indices()
        self.fit_results['fit_param_errs'] = fit_param_errs

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
        if self.progress_bar:  # pragma: no cover
            desc = 'Fit source/group'
            sources = add_progress_bar(sources, desc=desc)

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
                    msg = ('For one or more sources, the number of data '
                           'points available to fit is less than the '
                           'number of fit parameters. This could be due to '
                           'a source(s) near the edge of the detector or '
                           'if it has few unmasked pixels. Please check the '
                           'input mask or source positions.')
                    raise ValueError(msg) from exc

                fit_info = {}
                for key in fit_info_keys:
                    value = self.fitter.fit_info.get(key, None)
                    if value is not None:
                        fit_info[key] = value

            fit_models.append(fit_model)
            fit_infos.append(fit_info)

        self._group_results['fit_infos'] = fit_infos
        self._group_results['nmodels'] = nmodels

        # split the groups and return objects in source-id order
        fit_models = self._parse_fit_results(fit_models, fit_infos)
        _fit_params = self._model_params_to_table(fit_models)  # ungrouped
        fit_params = self._prepare_fit_results(_fit_params)
        self._fit_params = _fit_params
        self.fit_params = fit_params

        return fit_params

    def _calc_fit_metrics(self, data, results_tbl):
        # Keep cen_idx as a list because it can have NaNs with the ints.
        # If NaNs are present, turning it into an array will convert the
        # ints to floats, which cannot be used as slices.
        cen_idx = self._ungroup(self._group_results['psfcenter_indices'])

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
            qfit = cfit = np.array([[np.nan]] * len(results_tbl))
            return qfit, cfit

        fit_residuals = []
        for idx, fit_info in zip(split_index,
                                 self._group_results['fit_infos']):
            fit_residuals.extend(np.split(fit_info[key], idx))
        fit_residuals = self._order_by_id(fit_residuals)

        with warnings.catch_warnings():
            # ignore divide-by-zero if flux = 0
            warnings.simplefilter('ignore', RuntimeWarning)

            qfit = []
            cfit = []
            for index, (residual, cen_idx_) in enumerate(
                    zip(fit_residuals, cen_idx)):

                flux_fit = results_tbl['flux_fit'][index]
                qfit.append(np.sum(np.abs(residual)) / flux_fit)

                if np.isnan(cen_idx_):  # masked central pixel
                    cen_residual = np.nan
                else:
                    # find residual at center pixel;
                    # astropy fitters compute residuals as
                    # (model - data), thus need to negate the residual
                    cen_residual = -residual[cen_idx_]

                cfit.append(cen_residual / flux_fit)

        return qfit, cfit

    def _define_flags(self, results_tbl, shape):
        flags = np.zeros(len(results_tbl), dtype=int)

        for index, row in enumerate(results_tbl):
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

    def _prepare_fit_inputs(self, data, *, mask=None, error=None,
                            init_params=None):
        """
        Prepare inputs for PSF fitting.

        Tasks:
            * Checks array input shapes and units.
            * Calculates a total mask
            * Validates inputs for init_params and aperture_radius
            * Prepares initial parameters table
              - Runs source finder if needed
              - Runs aperture photometry if needed
              - Runs local background estimation if needed
              - Groups sources if needed
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        data = self._validate_array(data, 'data')
        error = self._validate_array(error, 'error', data_shape=data.shape)
        mask = self._validate_array(mask, 'mask', data_shape=data.shape)
        mask = self._make_mask(data, mask)
        init_params = self._validate_init_params(init_params, unit)  # copies

        if (self.aperture_radius is None
            and (init_params is None
                 or self._init_colnames['flux'] not in init_params.colnames)):
            raise ValueError('aperture_radius must be defined if init_params '
                             'is not input or if a flux column is not in '
                             'init_params')

        init_params = self._prepare_init_params(data, unit, mask, init_params)
        if init_params is not None:
            self._check_init_positions(init_params, data.shape)
        self.init_params = init_params

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

        return data, mask, error, init_params, unit

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
            must have the same shape as the input ``data``. If ``data``
            is a `~astropy.units.Quantity` array, then ``error`` must
            also be a `~astropy.units.Quantity` array with the same
            units.

        init_params : `~astropy.table.Table` or `None`, optional
            A table containing the initial guesses of the (x, y, flux)
            model parameters for each source. If the x and y values are
            not input, then the ``finder`` keyword must be defined. If
            the flux values are not input, then the ``aperture_radius``
            keyword must be defined. Note that the initial flux
            values refer to the model flux parameters and are not
            corrected for local background values (computed using
            ``localbkg_estimator`` or input in a ``local_bkg`` column)
            The allowed column names are:

              * ``x_init``, ``xinit``, ``x``, ``x_0``, ``x0``,
                ``xcentroid``, ``x_centroid``, ``x_peak``, ``xcen``,
                ``x_cen``, ``xpos``, ``x_pos``, ``x_fit``, and ``xfit``.

              * ``y_init``, ``yinit``, ``y``, ``y_0``, ``y0``,
                ``ycentroid``, ``y_centroid``, ``y_peak``, ``ycen``,
                ``y_cen``, ``ypos``, ``y_pos``, ``y_fit``, and ``yfit``.

              * ``flux_init``, ``fluxinit``, ``flux``, ``flux_0``,
                ``flux0``, ``flux_fit``, ``fluxfit``, ``source_sum``,
                ``segment_flux``, and ``kron_flux``.

            The parameter names are searched in the input table in the
            above order, stopping at the first match.

            If ``data`` is a `~astropy.units.Quantity` array, then the
            initial flux values in this table must also must also have
            compatible units.

            The table can also have ``group_id`` and ``local_bkg``
            columns. If ``group_id`` is input, the values will
            be used and ``grouper`` keyword will be ignored. If
            ``local_bkg`` is input, those values will be used and the
            ``localbkg_estimator`` will be ignored. If ``data`` has
            units, then the ``local_bkg`` values must have the same
            units.

        Returns
        -------
        table : `~astropy.table.QTable`
            An astropy table with the PSF-fitting results. The table
            will contain the following columns:

              * ``id`` : unique identification number for the source
              * ``group_id`` : unique identification number for the
                source group
              * ``group_size`` : the total number of sources that were
                simultaneously fit along with the given source
              * ``x_init``, ``x_fit``, ``x_err`` : the initial, fit, and
                error of the source x center
              * ``y_init``, ``y_fit``, ``y_err`` : the initial, fit, and
                error of the source y center
              * ``flux_init``, ``flux_fit``, ``flux_err`` : the initial,
                fit, and error of the source flux
              * ``npixfit`` : the number of unmasked pixels used to fit
                the source
              * ``qfit`` : a quality-of-fit metric defined as the
                absolute value of the sum of the fit residuals divided by
                the fit flux
              * ``cfit`` : a quality-of-fit metric defined as the
                fit residual in the initial central pixel value divided by
                the fit flux. NaN values indicate that the central pixel
                was masked.
              * ``flags`` : bitwise flag values:
                  * 1 : one or more pixels in the ``fit_shape`` region
                    were masked
                  * 2 : the fit x and/or y position lies outside of the
                    input data
                  * 4 : the fit flux is less than or equal to zero
                  * 8 : the fitter may not have converged. In this case,
                    you can try increasing the maximum number of fit
                    iterations using the ``fitter_maxiters`` keyword.
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

        # Prepare fit inputs, including defining the initial source
        # parameters. This also runs the source finder, aperture
        # photometry, local background estimator, and source grouper, if
        # needed.
        fit_inputs = self._prepare_fit_inputs(data, mask=mask, error=error,
                                              init_params=init_params)
        if fit_inputs is None:
            return None

        # fit the sources
        data, mask, error, init_params, unit = fit_inputs
        fit_params = self._fit_sources(data, init_params, error=error,
                                       mask=mask)

        # stack initial and fit params to create output table
        results_tbl = join(init_params, fit_params)

        npixfit = np.array(self._ungroup(self._group_results['npixfit']))
        results_tbl['npixfit'] = npixfit

        nmodels = np.array(self._ungroup(self._group_results['nmodels']))
        index = results_tbl.index_column('group_id') + 1
        results_tbl.add_column(nmodels, name='group_size', index=index)

        qfit, cfit = self._calc_fit_metrics(data, results_tbl)
        results_tbl['qfit'] = qfit
        results_tbl['cfit'] = cfit

        results_tbl['flags'] = self._define_flags(results_tbl, data.shape)

        if unit is not None:
            results_tbl['local_bkg'] <<= unit
            results_tbl['flux_fit'] <<= unit
            results_tbl['flux_err'] <<= unit

        meta = _get_meta()
        attrs = ('fit_shape', 'fitter_maxiters', 'aperture_radius',
                 'progress_bar')
        for attr in attrs:
            meta[attr] = getattr(self, attr)
        results_tbl.meta = meta

        if len(self.fit_results['fit_error_indices']) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table.',
                          AstropyUserWarning)

        # convert results from defaultdict to dict
        self.fit_results = dict(self.fit_results)

        return results_tbl

    def make_model_image(self, shape, psf_shape, *, include_localbkg=False):
        return ModelImageMixin.make_model_image(
            self, shape, psf_shape, include_localbkg=include_localbkg)

    def make_residual_image(self, data, psf_shape, *, include_localbkg=False):
        return ModelImageMixin.make_residual_image(
            self, data, psf_shape, include_localbkg=include_localbkg)


class IterativePSFPhotometry(ModelImageMixin):
    """
    Class to iteratively perform PSF photometry.

    This is a convenience class that iteratively calls the
    `PSFPhotometry` class to perform PSF photometry on an input image.
    It can be useful for crowded fields where faint stars are very
    close to bright stars and are not detected in the first pass of
    PSF photometry. For complex cases, one may need to manually run
    `PSFPhotometry` in an iterative manner and inspect the residual
    image after each iteration.

    Parameters
    ----------
    psf_model : 2D `astropy.modeling.Model`
        The PSF model to fit to the data. The model must have parameters
        named ``x_0``, ``y_0``, and ``flux``, corresponding to the
        center (x, y) position and flux, or it must have 'x_name',
        'y_name', and 'flux_name' attributes that map to the x, y, and
        flux parameters (i.e., a model output from `make_psf_model`).
        The model must be two-dimensional such that it accepts 2 inputs
        (e.g., x and y) and provides 1 output.

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
        positions. These positions are used as the starting points for
        the PSF fitting. The allowed ``x`` column names are (same suffix
        for ``y``): ``'x_init'``, ``'xinit'``, ``'x'``, ``'x_0'``,
        ``'x0'``, ``'xcentroid'``, ``'x_centroid'``, ``'x_peak'``,
        ``'xcen'``, ``'x_cen'``, ``'xpos'``, ``'x_pos'``, ``'x_fit'``,
        and ``'xfit'``. If `None`, then the initial (x, y) model
        positions must be input using the ``init_params`` keyword
        when calling the class. The (x, y) values in ``init_params``
        override this keyword *only for the first iteration*. If
        this class is run on an image that has units (i.e., a
        `~astropy.units.Quantity` array), then certain ``finder``
        keywords (e.g., ``threshold``) must have the same units. Please
        see the the documentation for the specific ``finder`` class for
        more information.

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

    maxiters : int, optional
        The maximum number of PSF-fitting/subtraction iterations to
        perform.

    mode : {'new', 'all'}, optional
        For the 'new' mode, `PSFPhotometry` is run in each iteration
        only on the new sources detected in the residual image. For the
        'all' mode, `PSFPhotometry` is run in each iteration on all the
        detected sources (from all previous iterations) on the original,
        unsubtracted, data. For the 'all' mode, a source ``grouper``
        must be input. See the Notes section for more details.

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, optional
        The object used to estimate the local background around each
        source. If `None`, then no local background is subtracted. The
        ``local_bkg`` values in ``init_params`` override this keyword.
        This option should be used with care, especially in crowded
        fields where the ``fit_shape`` of sources overlap (see Notes
        below).

    aperture_radius : float, optional
        The radius of the circular aperture used to estimate the initial
        flux of each source. If initial flux values are present in the
        ``init_params`` table, they will override this keyword *only for
        the first iteration*.

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

    Notes
    -----
    The data that will be fit for each source is defined by the
    ``fit_shape`` parameter. A cutout will be made around the initial
    center of each source with a shape defined by ``fit_shape``. The PSF
    model will be fit to the data in this region. The cutout region that
    is fit does not shift if the source center shifts during the fit
    iterations. Therefore, the initial source positions should be close
    to the true source positions. One way to ensure this is to use a
    ``finder`` to identify sources in the data.

    If the fitted positions are significantly different from the initial
    positions, one can re-run the `PSFPhotometry` class using the fit
    results as the input ``init_params``, which will change the fitted
    cutout region for each source. After calling `PSFPhotometry` on the
    data, it will have a ``fit_params`` attribute containing the fitted
    model parameters. This table can be used as the ``init_params``
    input in a subsequent call to `PSFPhotometry`.

    If the returned model parameter errors are NaN, then either
    the fit did not converge, the model parameter was fixed, or
    the input ``fitter`` did not return parameter errors. For the
    later case, one can try a different fitter that may return
    parameter errors (e.g., `astropy.models.fitting.LMLSQFitter
    or `astropy.models.fitting.TRFLSQFitter`). Note that
    these fitters are typically slower than the default
    `astropy.models.fitting.LevMarLSQFitter`.

    The local background value around each source is optionally
    estimated using the ``localbkg_estimator`` or obtained from the
    ``local_bkg`` column in the input ``init_params`` table. This local
    background is then subtracted from the data over the ``fit_shape``
    region for each source before fitting the PSF model. For sources
    where their ``fit_shape`` regions overlap, the local background will
    effectively be subtracted twice in the overlapping ``fit_shape``
    regions, even if the source ``grouper`` is input. This is not an
    issue if the sources are well-separated. However, for crowded
    fields, please use the ``localbkg_estimator`` (or ``local_bkg``
    column in ``init_params``) with care.

    This class has two modes of operation: 'new' and 'all'. For both
    modes, `PSFPhotometry` is first run on the data, a residual image
    is created, and the source finder is run on the residual image to
    detect any new sources.

    In the 'new' mode, `PSFPhotometry` is then run on the residual image
    to fit the PSF model to the new sources. The process is repeated
    until no new sources are detected or a maximum number of iterations
    is reached.

    In the 'all' mode, a new source list combining the sources from
    first `PSFPhotometry` run and the new sources detected in the
    residual image is created. `PSFPhotometry` is then run on the
    original, unsubtracted, data with this combined source list. This
    allows the source ``grouper`` (which is required for the 'all'
    mode) to combine close sources to be fit simultaneously, improving
    the fit. Again, the process is repeated until no new sources are
    detected or a maximum number of iterations is reached.
    """

    def __init__(self, psf_model, fit_shape, finder, *, grouper=None,
                 fitter=LevMarLSQFitter(), fitter_maxiters=100, maxiters=3,
                 mode='new', localbkg_estimator=None, aperture_radius=None,
                 sub_shape=None, progress_bar=False):

        if finder is None:
            raise ValueError('finder cannot be None for '
                             'IterativePSFPhotometry.')

        if aperture_radius is None:
            raise ValueError('aperture_radius cannot be None for '
                             'IterativePSFPhotometry.')

        self._psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                      grouper=grouper, fitter=fitter,
                                      fitter_maxiters=fitter_maxiters,
                                      localbkg_estimator=localbkg_estimator,
                                      aperture_radius=aperture_radius,
                                      progress_bar=progress_bar)

        self.maxiters = self._validate_maxiters(maxiters)

        if mode not in ['new', 'all']:
            raise ValueError('mode must be "new" or "all".')
        if mode == 'all' and grouper is None:
            raise ValueError('grouper must be input for the "all" mode.')
        self.mode = mode

        if sub_shape is None:
            sub_shape = fit_shape
        self.sub_shape = as_pair('sub_shape', sub_shape, lower_bound=(0, 1),
                                 check_odd=True)

        self.fit_results = []

    @staticmethod
    def _validate_maxiters(maxiters):
        if (not np.isscalar(maxiters) or maxiters <= 0
                or ~np.isfinite(maxiters)):
            raise ValueError('maxiters must be a strictly-positive scalar')
        if maxiters != int(maxiters):
            raise ValueError('maxiters must be an integer')
        return maxiters

    @staticmethod
    def _emit_warnings(recorded_warnings):
        """
        Emit unique warnings from a list of recorded warnings.

        Parameters
        ----------
        recorded_warnings : list
            A list of recorded warnings.
        """
        msgs = []
        emit_warnings = []
        for warning in recorded_warnings:
            if str(warning.message) not in msgs:
                msgs.append(str(warning.message))
                emit_warnings.append(warning)
        for warning in emit_warnings:
            warnings.warn_explicit(warning.message, warning.category,
                                   warning.filename, warning.lineno)

    def _convert_finder_to_init(self, sources):
        """
        Convert the output of the finder to a table with initial (x, y)
        position column names
        """
        xcol = self._psfphot._init_colnames['x']
        ycol = self._psfphot._init_colnames['y']
        sources = sources['xcentroid', 'ycentroid']
        sources.rename_column('xcentroid', xcol)
        sources.rename_column('ycentroid', ycol)
        return sources

    def _measure_init_fluxes(self, data, mask, sources):
        """
        Measure initial fluxes for the new sources from the
        residual data.

        The fluxes are added in place to the input ``sources`` table.

        The fluxes are measured using aperture photometry with a
        circular aperture of radius ``aperture_radius``.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which to perform photometry.

        mask : 2D bool `~numpy.ndarray`
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked.

        sources : `~astropy.table.Table`
            A table containing the initial (x, y) positions of the
            sources.

        Returns
        -------
        sources : `~astropy.table.Table`
            The input ``sources`` table with the new flux column added.
        """
        flux = self._psfphot._get_aper_fluxes(data, mask, sources)
        unit = getattr(data, 'unit', None)
        if unit is not None:
            flux <<= unit
        fluxcol = self._psfphot._init_colnames['flux']
        sources[fluxcol] = flux
        return sources

    def _create_init_params(self, data, mask, new_sources, orig_sources):
        """
        Create the initial parameters table by combining the original
        and new sources.
        """
        # add initial fluxes for the new sources from the residual data
        new_sources = self._measure_init_fluxes(data, mask, new_sources)

        # use the init_params column names
        orig_sources = orig_sources['x_fit', 'y_fit', 'flux_fit']
        xcol = self._psfphot._init_colnames['x']
        ycol = self._psfphot._init_colnames['y']
        fluxcol = self._psfphot._init_colnames['flux']
        orig_sources.rename_column('x_fit', xcol)
        orig_sources.rename_column('y_fit', ycol)
        orig_sources.rename_column('flux_fit', fluxcol)

        # combine original and new source tables
        new_sources.meta.pop('date', None)  # prevent merge conflicts
        return vstack([orig_sources, new_sources])

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
            error, including the Poisson error of the sources (see
            `~photutils.utils.calc_total_error`) . ``error`` must have
            the same shape as the input ``data``. If ``data`` is a
            `~astropy.units.Quantity` array, then ``error`` must also be
            a `~astropy.units.Quantity` array with the same units.

        init_params : `~astropy.table.Table` or `None`, optional
            A table containing the initial guesses of the (x, y, flux)
            model parameters for each source *only for the first
            iteration*. If the x and y values are not input, then the
            ``finder`` will be used for all iterations. If the flux
            values are not input, then the initial fluxes will be
            measured using the ``aperture_radius`` keyword. The input
            flux values will be used for the first iteration only.
            Note that the initial flux values refer to the model flux
            parameters and are not corrected for local background
            values (computed using ``localbkg_estimator`` or input in a
            ``local_bkg`` column) The allowed column names are:

              * ``x_init``, ``xinit``, ``x``, ``x_0``, ``x0``,
                ``xcentroid``, ``x_centroid``, ``x_peak``, ``xcen``,
                ``x_cen``, ``xpos``, ``x_pos``, ``x_fit``, and ``xfit``.

              * ``y_init``, ``yinit``, ``y``, ``y_0``, ``y0``,
                ``ycentroid``, ``y_centroid``, ``y_peak``, ``ycen``,
                ``y_cen``, ``ypos``, ``y_pos``, ``y_fit``, and ``yfit``.

              * ``flux_init``, ``fluxinit``, ``flux``, ``flux_0``,
                ``flux0``, ``flux_fit``, ``fluxfit``, ``source_sum``,
                ``segment_flux``, and ``kron_flux``.

            The parameter names are searched in the input table in the
            above order, stopping at the first match.

            If ``data`` is a `~astropy.units.Quantity` array, then the
            initial flux values in this table must also must also have
            compatible units.

        Returns
        -------
        table : `~astropy.table.QTable`
            An astropy table with the PSF-fitting results. The table
            will contain the following columns:

              * ``id`` : unique identification number for the source
              * ``group_id`` : unique identification number for the
                source group
              * ``group_size`` : the total number of sources that were
                simultaneously fit along with the given source
              * ``iter_detected`` : the iteration number in which the
                source was detected
              * ``x_init``, ``x_fit``, ``x_err`` : the initial, fit, and
                error of the source x center
              * ``y_init``, ``y_fit``, ``y_err`` : the initial, fit, and
                error of the source y center
              * ``flux_init``, ``flux_fit``, ``flux_err`` : the initial,
                fit, and error of the source flux
              * ``npixfit`` : the number of unmasked pixels used to fit
                the source
              * ``qfit`` : a quality-of-fit metric defined as the
                absolute value of the sum of the fit residuals divided by
                the fit flux
              * ``cfit`` : a quality-of-fit metric defined as the
                fit residual in the initial central pixel value divided by
                the fit flux. NaN values indicate that the central pixel
                was masked.
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
        with warnings.catch_warnings(record=True) as rwarn0:
            phot_tbl = self._psfphot(data, mask=mask, error=error,
                                     init_params=init_params)
            self.fit_results.append(deepcopy(self._psfphot))

        # this needs to be run outside of the context manager to be able
        # to re-emit any warnings
        if phot_tbl is None:
            self._emit_warnings(rwarn0)
            return None

        residual_data = data
        with warnings.catch_warnings(record=True) as rwarn1:
            phot_tbl['iter_detected'] = 1
            if self.mode == 'all':
                iter_detected = np.ones(len(phot_tbl), dtype=int)

            iter_num = 2
            while iter_num <= self.maxiters and phot_tbl is not None:
                residual_data = self._psfphot.make_residual_image(
                    residual_data, self.sub_shape)

                # do not warn if no sources are found beyond the first
                # iteration
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', NoDetectionsWarning)

                    new_sources = self._psfphot.finder(residual_data,
                                                       mask=mask)
                    if new_sources is None:  # no new sources detected
                        break

                finder_results = new_sources.copy()
                new_sources = self._convert_finder_to_init(new_sources)
                if self.mode == 'all':
                    init_params = self._create_init_params(
                        residual_data, mask, new_sources, phot_tbl)
                    residual_data = data

                    # keep track of the iteration number in which the source
                    # was detected
                    current_iter = (np.ones(len(new_sources), dtype=int)
                                    * iter_num)
                    iter_detected = np.concatenate((iter_detected,
                                                    current_iter))
                elif self.mode == 'new':
                    # fit new sources on the residual data
                    init_params = new_sources

                # remove any sources that do not overlap the data
                imask = self._psfphot._get_invalid_positions(init_params,
                                                             data.shape)
                init_params = init_params[~imask]
                if self.mode == 'all':
                    iter_detected = iter_detected[~imask]

                new_tbl = self._psfphot(residual_data, mask=mask, error=error,
                                        init_params=init_params)
                self._psfphot.finder_results = finder_results
                self.fit_results.append(deepcopy(self._psfphot))

                if self.mode == 'all':
                    new_tbl['iter_detected'] = iter_detected
                    phot_tbl = new_tbl

                elif self.mode == 'new':
                    # combine tables
                    new_tbl['id'] += np.max(phot_tbl['id'])
                    new_tbl['group_id'] += np.max(phot_tbl['group_id'])
                    new_tbl['iter_detected'] = iter_num
                    new_tbl.meta = {}  # prevent merge conflicts on date
                    phot_tbl = vstack([phot_tbl, new_tbl])

                iter_num += 1

            # move 'iter_detected' column
            phot_tbl = self._psfphot._move_column(phot_tbl, 'iter_detected',
                                                  'group_size')

        # emit unique warnings
        recorded_warnings = rwarn0 + rwarn1
        self._emit_warnings(recorded_warnings)

        return phot_tbl

    def make_model_image(self, shape, psf_shape, *, include_localbkg=False):
        return ModelImageMixin.make_model_image(
            self, shape, psf_shape, include_localbkg=include_localbkg)

    def make_residual_image(self, data, psf_shape, *, include_localbkg=False):
        return ModelImageMixin.make_residual_image(
            self, data, psf_shape, include_localbkg=include_localbkg)


def _flatten(iterable):
    """
    Flatten a list of lists.
    """
    return list(chain.from_iterable(iterable))
