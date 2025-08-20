# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import contextlib
import inspect
import warnings
from collections import defaultdict
from itertools import chain

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData, NoOverlapError, StdDevUncertainty
from astropy.table import QTable, Table, hstack, join
from astropy.utils.decorators import deprecated_attribute
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.background import LocalBackground
from photutils.psf.groupers import SourceGrouper
from photutils.psf.utils import (ModelImageMixin, _get_psf_model_main_params,
                                 _make_mask, _validate_psf_model)
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['PSFPhotometry']


class _PSFParameterManager:
    """
    A helper class to manage the mapping between PSF model parameter
    names and table column names.

    Parameters
    ----------
    psf_model : `astropy.modeling.Model`
        The PSF model to be used for photometry. It must have parameters
        named ``x_0``, ``y_0``, and ``flux``, or it must have 'x_name',
        'y_name', and 'flux_name' attributes that map to the x, y, and
        flux parameters.
    """

    _VALID_INIT_COLNAMES = {  # noqa: RUF012
        'x': ('x_init', 'xinit', 'x', 'x_0', 'x0', 'xcentroid',
              'x_centroid', 'x_peak', 'xcen', 'x_cen', 'xpos', 'x_pos',
              'x_fit', 'xfit'),
        'y': ('y_init', 'yinit', 'y', 'y_0', 'y0', 'ycentroid',
              'y_centroid', 'y_peak', 'ycen', 'y_cen', 'ypos', 'y_pos',
              'y_fit', 'yfit'),
        'flux': ('flux_init', 'fluxinit', 'flux', 'flux_0', 'flux0',
                 'flux_fit', 'fluxfit', 'source_sum', 'segment_flux',
                 'kron_flux'),
    }

    def __init__(self, psf_model):
        self.psf_model = psf_model

        # store an ordered list of only the parameters that are being fit
        self.fitted_param_names = [p_name for p_name in psf_model.param_names
                                   if not psf_model.fixed[p_name]]

        # Map aliases of fitted parameters to the actual model parameter
        # names. The main parameters are 'x', 'y', and 'flux' are always
        # included, even if they are not fit. Other parameters are
        # included only if they are fit.
        self.alias_to_model_param = self._get_model_params_map()
        self.model_param_to_alias = {v: k
                                     for k, v in
                                     self.alias_to_model_param.items()}

        # create maps from alias to table column names
        self.init_colnames = {alias: f'{alias}_init'
                              for alias in self.alias_to_model_param}
        self.fit_colnames = {alias: f'{alias}_fit'
                             for alias in self.alias_to_model_param}
        self.err_colnames = {alias: f'{alias}_err'
                             for alias in self.alias_to_model_param}

    def _get_model_params_map(self):
        """
        Get the mapping of aliases ('x', 'y', 'flux', etc.) to the
        actual parameter names in the PSF model.

        Returns
        -------
        params_map : dict
            A dictionary mapping parameter aliases to their actual
            names in the PSF model. The keys are 'x', 'y', 'flux', and
            any additional parameters defined in the model.
        """
        # the order of the main parameters is important; it defines
        # the order of table outputs
        main_params = _get_psf_model_main_params(self.psf_model)
        params_map = dict(zip(('x', 'y', 'flux'), main_params, strict=True))

        # extra parameters that are not 'x', 'y', or 'flux', but
        # are free to be fit (fixed = False), are added to the map
        # with their own aliases
        extra_params = [param for param in self.fitted_param_names
                        if param not in main_params]

        params_map.update({key: key for key in extra_params})
        return params_map

    def find_column(self, table, param_alias):
        """
        Find the first valid column name in a table for a given
        parameter alias.

        Parameters
        ----------
        table : `~astropy.table.Table`
            The input table to search for the column.

        param_alias : str
            The alias for the parameter (e.g., 'x', 'y', 'flux').

        Returns
        -------
        result : str or `None`
            The first valid column name found in the table for the
            parameter alias, or `None` if no valid column is found.
        """
        try:
            valid_names = self._VALID_INIT_COLNAMES[param_alias]
        except KeyError:
            # valid names for extra parameters are more limited
            valid_names = (f'{param_alias}_init', param_alias,
                           f'{param_alias}_fit')

        for name in valid_names:
            if name in table.colnames:
                return name

        return None

    def rename_table_columns(self, table):
        """
        Rename columns in-place in an input table to the ``_init``
        format.

        Parameters
        ----------
        table : `~astropy.table.Table`
            The input table with columns to be renamed.

        Returns
        -------
        table : `~astropy.table.Table`
            The input table with columns renamed to the `_init` format
            based on the parameter aliases.
        """
        for param_alias in self.alias_to_model_param:
            found_col = self.find_column(table, param_alias)
            if found_col:
                target_col = self.init_colnames[param_alias]
                if found_col != target_col:
                    table.rename_column(found_col, target_col)
        return table


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

    finder : callable or `~photutils.detection.StarFinderBase` or `None`, \
            optional
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
        override this keyword. If this class is run on an image that has
        units (i.e., a `~astropy.units.Quantity` array), then certain
        ``finder`` keywords (e.g., ``threshold``) must have the same
        units. Please see the documentation for the specific ``finder``
        class for more information.

    grouper : `~photutils.psf.SourceGrouper` or callable or `None`, optional
        A callable used to group stars. Typically, grouped stars are
        those that overlap with their neighbors. Stars that are grouped
        are fit simultaneously. The ``grouper`` must accept the x and
        y coordinates of the sources and return an integer array of
        the group id numbers (starting from 1) indicating the group
        in which a given source belongs. If `None`, then no grouping
        is performed, i.e. each source is fit independently. The
        ``group_id`` values in ``init_params`` override this keyword. A
        warning is raised if any group size is larger than 25 sources.

    fitter : `~astropy.modeling.fitting.Fitter`, optional
        The fitter object used to perform the fit of the
        model to the data. If `None`, then the default
        `astropy.modeling.fitting.TRFLSQFitter` is used.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each source. This keyword can be increased if the fit
        is not converging for sources (e.g., the output ``flags`` value
        contains 8).

    xy_bounds : `None`, float, or 2-tuple of float, optional
        The maximum distance in pixels that a fitted source can be from
        the initial (x, y) position. If a single float, then the same
        maximum distance is used for both x and y. If a 2-tuple of
        floats, then the distances are in ``(x, y)`` order. If `None`,
        then no bounds are applied. Either value can also be `None` to
        indicate no bound in that direction.

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, \
        optional
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
    positions, one can rerun the `PSFPhotometry` class using the fit
    results as the input ``init_params``, which will change the fitted
    cutout region for each source. After calling `PSFPhotometry` on the
    data, it will have a ``fit_params`` attribute containing the fitted
    model parameters. This table can be used as the ``init_params``
    input in a subsequent call to `PSFPhotometry`.

    If the returned model parameter errors are NaN, then either the
    fit did not converge, the model parameter was fixed, or the
    input ``fitter`` did not return parameter errors. For the later
    case, one can try a different fitter that may return parameter
    errors (e.g., `astropy.modeling.fitting.DogBoxLSQFitter` or
    `astropy.modeling.fitting.LMLSQFitter`).

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

    Care should be taken in defining the star groups. Simultaneously
    fitting very large star groups is computationally expensive and
    error-prone. Internally, source grouping requires the creation of a
    compound Astropy model. Due to the way compound Astropy models are
    currently constructed, large groups also require excessively large
    amounts of memory; this will hopefully be fixed in a future Astropy
    version. A warning will be raised if the number of sources in a
    group exceeds 25.
    """

    fit_results = deprecated_attribute('fit_results', '2.0.0',
                                       alternative='fit_info')

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 localbkg_estimator=None, aperture_radius=None,
                 progress_bar=False):

        self.psf_model = _validate_psf_model(psf_model)
        self._param_mapper = _PSFParameterManager(self.psf_model)

        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 0),
                                 check_odd=True)
        self.grouper = self._validate_grouper(grouper)
        self.finder = self._validate_callable(finder, 'finder')
        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.localbkg_estimator = self._validate_localbkg(
            localbkg_estimator, 'localbkg_estimator')
        self.fitter_maxiters = self._validate_maxiters(fitter_maxiters)
        self.xy_bounds = self._validate_bounds(xy_bounds)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.progress_bar = progress_bar

        self.group_warning_threshold = 25

        self._reset_results()

    def _reset_results(self):
        """
        Reset these attributes for each __call__.
        """
        self.data_unit = None
        self.finder_results = None
        self.init_params = None
        self.fit_params = None
        self._fitted_models_table = None
        self.results = None
        self.fit_info = defaultdict(list)
        self._group_results = defaultdict(list)

    def __repr__(self):
        params = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                  'fitter_maxiters', 'xy_bounds', 'localbkg_estimator',
                  'aperture_radius', 'progress_bar')
        return make_repr(self, params)

    def _validate_grouper(self, grouper):
        if grouper is not None and not isinstance(grouper, SourceGrouper):
            msg = 'grouper must be a SourceGrouper instance'
            raise ValueError(msg)
        return grouper

    @staticmethod
    def _validate_callable(obj, name):
        if obj is not None and not callable(obj):
            msg = f'{name!r} must be a callable object'
            raise TypeError(msg)
        return obj

    def _validate_localbkg(self, value, name):
        if value is not None and not isinstance(value, LocalBackground):
            msg = 'localbkg_estimator must be a LocalBackground instance'
            raise ValueError(msg)
        return self._validate_callable(value, name)

    def _validate_maxiters(self, maxiters):
        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            warnings.warn('"maxiters" will be ignored because it is not '
                          'accepted by the input fitter __call__ method',
                          AstropyUserWarning)
            maxiters = None
        return maxiters

    def _validate_bounds(self, xy_bounds):
        if xy_bounds is None:
            return xy_bounds

        xy_bounds = np.atleast_1d(xy_bounds)
        if len(xy_bounds) == 1:
            xy_bounds = np.array((xy_bounds[0], xy_bounds[0]))
        if len(xy_bounds) != 2:
            msg = 'xy_bounds must have 1 or 2 elements'
            raise ValueError(msg)
        if xy_bounds.ndim != 1:
            msg = 'xy_bounds must be a 1D array'
            raise ValueError(msg)
        for bound in xy_bounds:
            if bound is not None and bound <= 0:
                msg = 'xy_bounds must be strictly positive'
                raise ValueError(msg)
        return xy_bounds

    @staticmethod
    def _validate_radius(radius):
        if radius is not None and (not np.isscalar(radius)
                                   or radius <= 0 or not np.isfinite(radius)):
            msg = 'aperture_radius must be a strictly-positive scalar'
            raise ValueError(msg)
        return radius

    def _validate_array(self, array, name, data_shape=None):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.ndim != 2:
                msg = f'{name} must be a 2D array'
                raise ValueError(msg)
            if data_shape is not None and array.shape != data_shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    def _normalize_init_units(self, init_params, colname):
        values = init_params[colname]
        if isinstance(values, u.Quantity):
            if self.data_unit is None:
                msg = (f'init_params {colname} column has units, but the '
                       'input data does not have units')
                raise ValueError(msg)
            try:
                init_params[colname] = values.to(self.data_unit)
            except u.UnitConversionError as exc:
                msg = (f'init_params {colname} column has units that are '
                       'incompatible with the input data units')
                raise ValueError(msg) from exc
        elif self.data_unit is not None:
            msg = ('The input data has units, but the init_params '
                   f'{colname} column does not have units.')
            raise ValueError(msg)

        return init_params

    def _validate_init_params(self, init_params):
        """
        Validate the input ``init_params`` table.

        Also rename the columns to the expected names with the "_init"
        suffix if necessary.
        """
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            msg = 'init_params must be an astropy Table'
            raise TypeError(msg)

        # copy is used to preserve the input init_params
        init_params = self._param_mapper.rename_table_columns(
            init_params.copy())

        if (self._param_mapper.init_colnames['x'] not in init_params.colnames
                or self._param_mapper.init_colnames['y'] not in
                init_params.colnames):
            msg = ('init_param must contain valid column names for the '
                   'x and y source positions')
            raise ValueError(msg)

        flux_col = self._param_mapper.init_colnames['flux']
        if flux_col in init_params.colnames:
            init_params = self._normalize_init_units(init_params, flux_col)

        if 'local_bkg' in init_params.colnames:
            if not np.all(np.isfinite(init_params['local_bkg'])):
                msg = ('init_params local_bkg column contains non-finite '
                       'values')
                raise ValueError(msg)
            init_params = self._normalize_init_units(init_params, 'local_bkg')

        return init_params

    def _get_aper_fluxes(self, data, mask, init_params):
        x_pos = init_params[self._param_mapper.init_colnames['x']]
        y_pos = init_params[self._param_mapper.init_colnames['y']]
        apertures = CircularAperture(zip(x_pos, y_pos, strict=True),
                                     r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)
        return flux

    @staticmethod
    def _convert_finder_to_init(param_mapper, sources):
        """
        Convert the output table of the finder to a table with initial
        (x, y) position column names.
        """
        # find the first valid column names for x and y
        x_name_found = param_mapper.find_column(sources, 'x')
        y_name_found = param_mapper.find_column(sources, 'y')
        if x_name_found is None or y_name_found is None:
            msg = ("The table returned by the 'finder' must contain columns "
                   'for x and y coordinates. Valid column names are: '
                   f"x: {param_mapper._VALID_INIT_COLNAMES['x']}, "
                   f"y: {param_mapper._VALID_INIT_COLNAMES['y']}")
            raise ValueError(msg)

        # create a new table with only the needed columns
        init_params = QTable()
        init_params['id'] = np.arange(len(sources)) + 1
        x_col = param_mapper.init_colnames['x']
        y_col = param_mapper.init_colnames['y']
        init_params[x_col] = sources[x_name_found]
        init_params[y_col] = sources[y_name_found]

        return init_params

    def _find_sources_if_needed(self, data, mask, init_params):
        """
        Find sources using the finder if initial positions are not
        provided.

        The finder must return a table with valid x and y column names,
        which will be used to initialize the source positions in the
        ``init_params`` table.
        """
        if init_params is not None:
            if 'id' not in init_params.colnames:
                init_params['id'] = np.arange(len(init_params)) + 1
            return init_params

        if self.finder is None:
            msg = 'finder must be defined if init_params is not input'
            raise ValueError(msg)

        if self.data_unit is not None:
            sources = self.finder(data << self.data_unit, mask=mask)
        else:
            sources = self.finder(data, mask=mask)

        self.finder_results = sources
        if sources is None:
            return None

        return self._convert_finder_to_init(self._param_mapper, sources)

    def _estimate_flux_and_bkg_if_needed(self, data, mask, init_params):
        """
        Estimate initial fluxes and backgrounds if not provided.
        """
        x_col = self._param_mapper.init_colnames['x']
        y_col = self._param_mapper.init_colnames['y']
        flux_col = self._param_mapper.init_colnames['flux']

        if 'local_bkg' not in init_params.colnames:
            if self.localbkg_estimator is None:
                local_bkg = np.zeros(len(init_params))
            else:
                local_bkg = self.localbkg_estimator(data, init_params[x_col],
                                                    init_params[y_col],
                                                    mask=mask)
            if self.data_unit is not None:
                local_bkg <<= self.data_unit
            init_params['local_bkg'] = local_bkg

        if flux_col not in init_params.colnames:
            # check for aperture_radius before attempting to use it
            if self.aperture_radius is None:
                msg = ('aperture_radius must be defined if a flux column is '
                       'not in init_params')
                raise ValueError(msg)

            flux = self._get_aper_fluxes(data, mask, init_params)
            if self.data_unit is not None:
                flux <<= self.data_unit
            flux -= init_params['local_bkg']
            init_params[flux_col] = flux

        return init_params

    def _group_sources(self, init_params):
        """
        Group sources using the grouper or the user-provided 'group_id'
        column.
        """
        if 'group_id' in init_params.colnames:
            # user-provided group_id takes precedence
            self.grouper = None

        elif self.grouper is not None:
            # use the grouper to group sources
            x_col = self._param_mapper.init_colnames['x']
            y_col = self._param_mapper.init_colnames['y']
            init_params['group_id'] = self.grouper(init_params[x_col],
                                                   init_params[y_col])

        else:
            # no grouper provided, so each source is its own group
            init_params['group_id'] = init_params['id'].copy()

        return init_params

    def _build_initial_parameters(self, data, mask, init_params):
        """
        Build the table of initial parameters for fitting.

        This method orchestrates finding sources, estimating initial
        fluxes and backgrounds, and grouping sources.
        """
        init_params = self._find_sources_if_needed(data, mask, init_params)
        if init_params is None:
            return None

        # strip any units from the x/y position columns
        for axis in ('x', 'y'):
            colname = self._param_mapper.init_colnames[axis]
            if isinstance(init_params[colname], u.Quantity):
                init_params[colname] = init_params[colname].value

        init_params = self._estimate_flux_and_bkg_if_needed(data, mask,
                                                            init_params)
        init_params = self._group_sources(init_params)

        # check for large group sizes after grouping is complete
        warn_size = self.group_warning_threshold
        if 'group_id' in init_params.colnames:
            _, counts = np.unique(init_params['group_id'], return_counts=True)
            if len(counts) > 0 and max(counts) > warn_size:
                msg = (f'Some groups have more than {warn_size} '
                       'sources. Fitting such groups may take a long time '
                       'and be error-prone. You may want to consider using '
                       'different `SourceGrouper` parameters or changing '
                       'the "group_id" column in "init_params".')
                warnings.warn(msg, AstropyUserWarning)

        # Add columns for any additional model parameters that are
        # fit, using the model's default value if not already present.
        for alias, col_name in self._param_mapper.init_colnames.items():
            if col_name not in init_params.colnames:
                alias_map = self._param_mapper.alias_to_model_param
                model_param_name = alias_map[alias]
                init_params[col_name] = getattr(self.psf_model,
                                                model_param_name)

        # define the final column order
        main_aliases = ('x', 'y', 'flux')
        # Extra aliases are those that are not in the main_aliases.
        # The alias and model_param names are the same for
        # extra parameters, so we can use the alias_to_model_param map
        # to get the extra aliases.
        extra_aliases = [param
                         for param in self._param_mapper.alias_to_model_param
                         if param not in main_aliases]

        main_cols = [self._param_mapper.init_colnames[alias]
                     for alias in main_aliases]
        extra_cols = [self._param_mapper.init_colnames[alias]
                      for alias in extra_aliases]
        col_order = ['id', 'group_id', 'local_bkg', *main_cols, *extra_cols]

        return init_params[col_order]

    def _get_no_overlap_mask(self, init_params, shape):
        """
        Get a mask of sources with no overlap with the data.

        This code is based on astropy.nddata.overlap_slices.
        """
        x_pos = init_params[self._param_mapper.init_colnames['x']]
        y_pos = init_params[self._param_mapper.init_colnames['y']]
        positions = np.column_stack((y_pos, x_pos))
        delta = self.fit_shape / 2
        min_idx = np.ceil(positions - delta)
        max_idx = np.ceil(positions + delta)
        return np.any(max_idx <= 0, axis=1) | np.any(min_idx >= shape, axis=1)

    def _validate_source_positions(self, init_params, shape):
        """
        Validate the initial source positions to ensure they are within
        the data shape.
        """
        if np.any(self._get_no_overlap_mask(init_params, shape)):
            msg = ('Some of the sources have no overlap with the data. '
                   'Check the initial source positions or increase the '
                   'fit_shape.')
            raise ValueError(msg)

    def _prepare_fit_inputs(self, data, *, mask=None, error=None,
                            init_params=None):
        """
        Prepare all inputs for the PSF fitting.

        This method handles data validation, unit processing, source
        finding, initial parameter estimation, and grouping. It returns
        the processed inputs ready for the `_fit_sources` method.
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        self.data_unit = unit
        data = self._validate_array(data, 'data')
        error = self._validate_array(error, 'error', data_shape=data.shape)
        mask = self._validate_array(mask, 'mask', data_shape=data.shape)
        mask = _make_mask(data, mask)

        init_params = self._validate_init_params(init_params)
        init_params = self._build_initial_parameters(data, mask, init_params)

        if init_params is None:
            # no sources found
            return None, None, None, None

        self._validate_source_positions(init_params, data.shape)

        return data, mask, error, init_params

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources
        within a group.
        """
        for index, source in enumerate(sources):
            model = self.psf_model.copy()
            for alias, col_name in self._param_mapper.init_colnames.items():
                model_param = self._param_mapper.alias_to_model_param[alias]
                value = source[col_name]
                if isinstance(value, u.Quantity):
                    value = value.value  # psf model cannot be fit with units
                setattr(model, model_param, value)
            model.name = source['id']

            if self.xy_bounds is not None:
                if self.xy_bounds[0] is not None:
                    x_param_name = self._param_mapper.alias_to_model_param['x']
                    x_param = getattr(model, x_param_name)
                    x_param.bounds = (x_param.value - self.xy_bounds[0],
                                      x_param.value + self.xy_bounds[0])
                if self.xy_bounds[1] is not None:
                    y_param_name = self._param_mapper.alias_to_model_param['y']
                    y_param = getattr(model, y_param_name)
                    y_param.bounds = (y_param.value - self.xy_bounds[1],
                                      y_param.value + self.xy_bounds[1])

            if index == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def _all_model_params_to_table(self, models):
        """
        Convert a list of PSF models to a table of model parameters.

        All model parameters are included, including those that are not
        fit (i.e., fixed parameters). The table also includes columns
        for the parameter "fixed" and "bounds" values.

        The input ``models`` must all be instances of the same model
        class (i.e., the parameters names are the same for all models).
        """
        model_params = list(models[0].param_names)
        params = defaultdict(list)
        for model in models:
            for model_param in model_params:
                param = getattr(model, model_param)
                value = param.value
                flux_param = self._param_mapper.alias_to_model_param['flux']
                if (self.data_unit is not None and model_param == flux_param):
                    value <<= self.data_unit  # add the flux units
                params[model_param].append(value)
                params[f'{model_param}_fixed'].append(param.fixed)
                params[f'{model_param}_bounds'].append(param.bounds)

        table = QTable(params)
        ids = np.arange(len(table)) + 1
        table.add_column(ids, index=0, name='id')

        return table

    def _param_errors_to_table(self):
        """
        Convert the fitter's parameter errors to an astropy Table.

        This method creates error columns for all fitted parameters. It
        also ensures that 'x_err', 'y_err', and 'flux_err' columns
        always exist, filling them with NaN if the corresponding
        parameter was fixed.
        """
        # param_err_rows are stacked in ID order and have
        # columns in the same order as _param_mapper.fitted_param_names.
        # Only fitted model parameters are included.
        param_err_rows = self.fit_info.pop('fit_param_errs')
        table = QTable()

        # create error columns for models parameters that were fit
        fitted_params = self._param_mapper.fitted_param_names
        for i, fitted_param in enumerate(fitted_params):
            alias = self._param_mapper.model_param_to_alias[fitted_param]
            col_name = self._param_mapper.err_colnames[alias]
            table[col_name] = param_err_rows[:, i]

        # define the required column names for the table
        col_names = list(self._param_mapper.err_colnames.values())
        for col_name in col_names:
            if col_name not in table.colnames:
                # if the column is not present, it means the parameter was
                # fixed, so we fill it with NaNs.
                table[col_name] = np.nan

        # apply data_unit to flux_err column if applicable
        if self.data_unit is not None:
            flux_err_col = self._param_mapper.err_colnames['flux']
            if flux_err_col in table.colnames:  # should always be True
                table[flux_err_col] <<= self.data_unit

        # sort the columns to match the expected order defined
        # in _param_mapper.err_colnames
        return table[col_names]

    def _prepare_fit_results(self, fit_model_all_params):
        """
        Prepare the output table of fit results.

        This method takes the raw table of all model parameters from the
        fitter, filters it to include only the main parameters and
        parameters that were actually fit (i.e., not fixed), renames the
        columns with a '_fit' suffix, and merges them with the parameter
        errors.
        """
        # alias_to_model_param always contains the main parameters
        # (i.e., 'x', 'y', 'flux') and any additional parameters that
        # were fit. The main parameters are always returned.
        mapper = self._param_mapper.alias_to_model_param
        col_names = ['id', *list(mapper.values())]
        fit_params = fit_model_all_params[col_names]

        # rename the fitted parameter columns to have the "_fit" suffix
        for col_name in fit_params.colnames:
            if col_name == 'id':
                continue

            alias = self._param_mapper.model_param_to_alias[col_name]
            new_name = self._param_mapper.fit_colnames[alias]
            fit_params.rename_column(col_name, new_name)

        # get the table of parameter errors
        param_errs = self._param_errors_to_table()

        # horizontally stack the fit parameters and errors
        fit_table = hstack([fit_params, param_errs])

        # sort columns to match the expected order defined
        # in _param_mapper.fit_colnames and _param_mapper.err_colnames
        col_order = ['id',
                     *list(self._param_mapper.fit_colnames.values()),
                     *list(self._param_mapper.err_colnames.values())]
        return fit_table[col_order]

    def _define_fit_data(self, sources, data, mask):
        yi = []
        xi = []
        cutout = []
        npixfit = []
        cen_index = []

        # Create a grid of pixel offsets for the fit_shape. Calling
        # mgrid once is more efficient than calling it in the loop
        # for each source. This is possible because the fit_shape is
        # constant for all sources.
        ny, nx = self.fit_shape
        y_offsets, x_offsets = np.mgrid[0:ny, 0:nx]

        for row in sources:
            # get the initial center position
            x_cen = row[self._param_mapper.init_colnames['x']]
            y_cen = row[self._param_mapper.init_colnames['y']]

            try:
                slc_lg, _ = overlap_slices(data.shape, self.fit_shape,
                                           (y_cen, x_cen), mode='trim')
            except NoOverlapError as exc:  # pragma: no cover
                # this should never happen because the initial positions
                # are checked in _prepare_fit_inputs
                msg = (f'Initial source at ({x_cen}, {y_cen}) does not '
                       'overlap with the input data.')
                raise ValueError(msg) from exc

            # get the local grid of pixel offsets for the fit_shape
            y_start = slc_lg[0].start
            x_start = slc_lg[1].start
            ny_cutout = slc_lg[0].stop - y_start
            nx_cutout = slc_lg[1].stop - x_start
            trimmed_y_offsets = y_offsets[:ny_cutout, :nx_cutout]
            trimmed_x_offsets = x_offsets[:ny_cutout, :nx_cutout]
            yy = trimmed_y_offsets + y_start
            xx = trimmed_x_offsets + x_start

            if mask is not None:
                inv_mask = ~mask[yy, xx]
                if np.count_nonzero(inv_mask) == 0:
                    msg = (f'Source at ({x_cen}, {y_cen}) is completely '
                           'masked. Remove the source from init_params or '
                           'correct the input mask.')
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
            x_cen = np.ceil(x_cen - 0.5).astype(int)
            y_cen = np.ceil(y_cen - 0.5).astype(int)

            idx = np.where((xx == x_cen) & (yy == y_cen))[0]
            if len(idx) == 0:
                idx = [np.nan]
            cen_index.append(idx[0])

        # flatten the lists, which may contain arrays of different lengths
        # due to masking
        xi = self._flatten(xi)
        yi = self._flatten(yi)
        cutout = self._flatten(cutout)

        self._group_results['npixfit'].append(npixfit)
        self._group_results['psfcenter_indices'].append(cen_index)

        return yi, xi, cutout

    @staticmethod
    def _split_compound_model(model, chunk_size):
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i:i + chunk_size]

    def _order_by_id(self, iterable):
        """
        Reorder the list from group-id to source-id order.
        """
        return [iterable[i] for i in self._group_results['ungroup_indices']]

    def _ungroup(self, iterable):
        """
        Expand a list of lists (groups) and reorder in source-id order.
        """
        iterable = self._flatten(iterable)
        return self._order_by_id(iterable)

    def _get_fit_error_indices(self):
        """
        Get the indices of fits that did not converge.

        This method checks the fit_info dictionary for each group and
        identifies the indices of fits that did not converge. The
        criteria for convergence depend on the fitter used:
        - For `scipy.optimize.leastsq`, it checks the `ierr` value.
        - For `scipy.optimize.least_squares`, it checks the `status`
          value.

        The ierr/status codes checked here are specific to the
        scipy.optimize fitters used by default in astropy. Custom
        fitters may require different logic.

        Returns
        -------
        result : `~numpy.ndarray`
            An array of indices where the fit did not converge.
        """
        # same "good" flags for both leastsq and least_squares
        converged_status = {1, 2, 3, 4}

        # ierr and status values that indicate convergence and will not both
        # be present in the fit_info dictionary.
        bad_indices = [
            idx for idx, info in enumerate(self.fit_info['fit_infos'])
            if (('ierr' in info and info['ierr'] not in converged_status)
                or ('status' in info and info['status']
                    not in converged_status))
        ]

        return np.array(bad_indices, dtype=int)

    def _parse_single_group(self, group_model, group_fit_info):
        """
        Parse the results for a single fitted group.

        This helper method unpacks a potentially compound model from
        a group fit into its constituent single-source models and
        parameter errors.

        Parameters
        ----------
        group_model : `astropy.modeling.Model`
            The fitted model for a single group. This can be a compound
            model.

        group_fit_info : dict
            The fit_info dictionary corresponding to the group fit.

        Returns
        -------
        source_models : list
            A list of the individual fitted models for each source in
            the group.

        source_param_errs : list
            A list of numpy arrays, where each array contains the
            parameter errors for a single source in the group.

        source_fit_infos : list
            A list of views of the `group_fit_info` dictionary, one for
            each source in the group.
        """
        psf_nsub = self.psf_model.n_submodels
        nfitparam = len(self._param_mapper.fitted_param_names)
        num_sources = group_model.n_submodels // psf_nsub

        # Extract parameter errors from the covariance matrix.
        # source_params_errs is a 2D array with a shape
        # of (num_sources, nfitparam).
        param_cov = group_fit_info.get('param_cov')
        if param_cov is None:
            # handle cases where the fitter doesn't return a covariance
            # matrix or where all parameters were fixed
            source_param_errs = np.full((num_sources, nfitparam), np.nan)
        else:
            param_err_1d = np.sqrt(np.diag(param_cov))
            source_param_errs = param_err_1d.reshape(num_sources, nfitparam)

        # split the models and errors for each source in the group
        if num_sources == 1:
            source_models = [group_model]
        else:
            # split the compound model into a list of individual models
            source_models = list(self._split_compound_model(group_model,
                                                            psf_nsub))

        # each source in the group shares the same fit_info dictionary
        source_fit_infos = [group_fit_info] * num_sources

        return source_models, source_param_errs, source_fit_infos

    def _parse_fit_results(self, group_models, group_fit_infos):
        """
        Parse and reorder all fit results from group-order to source-ID-
        order.

        This method orchestrates the parsing of all fitted groups,
        reorders the results to match the original source IDs, and
        populates the ``self.fit_info`` dictionary.
        """
        # Parse each group's results into lists of individual source
        # results. These lists will be in the order the groups were fit.
        all_models_grouped = []
        all_param_errs_grouped = []
        all_fit_infos_grouped = []
        for model, fit_info in zip(group_models, group_fit_infos, strict=True):
            s_models, s_errs, s_infos = self._parse_single_group(model,
                                                                 fit_info)
            all_models_grouped.extend(s_models)
            all_param_errs_grouped.extend(s_errs)
            all_fit_infos_grouped.extend(s_infos)

        # reorder the results from group-order to source-ID-order
        fit_models_by_id = self._order_by_id(all_models_grouped)
        fit_param_errs_by_id = np.array(
            self._order_by_id(all_param_errs_grouped))
        fit_infos_by_id = self._order_by_id(all_fit_infos_grouped)

        # finalize and store results in the fit_info attribute
        self.fit_info['fit_infos'] = fit_infos_by_id
        self.fit_info['fit_param_errs'] = fit_param_errs_by_id
        self.fit_info['fit_error_indices'] = self._get_fit_error_indices()

        return fit_models_by_id

    def _run_fitter(self, psf_model, sources_for_fit, data, mask, error):
        """
        Run the fitter for a single model and a set of sources.

        This is a helper function to consolidate the fitting logic that is
        common to both single and grouped source fitting.

        Parameters
        ----------
        psf_model : `astropy.modeling.Model`
            The model to be fit (can be simple or compound).
        sources_for_fit : `astropy.table.Table`
            A table of the source(s) corresponding to the ``psf_model``.
        data : 2D `~numpy.ndarray`
            The data array.
        mask : 2D bool `~numpy.ndarray` or `None`
            The mask array.
        error : 2D `~numpy.ndarray` or `None`
            The error array.

        Returns
        -------
        fit_model : `astropy.modeling.Model`
            The fitted model.
        fit_info : dict
            The dictionary of fit information from the fitter.
        """
        if self.fitter_maxiters is not None:
            kwargs = {'maxiter': self.fitter_maxiters}
        else:
            kwargs = {}

        yi, xi, cutout = self._define_fit_data(sources_for_fit, data, mask)

        weights = None
        if error is not None:
            weights = 1.0 / error[yi, xi]
            if np.any(~np.isfinite(weights)):
                msg = ('Fit weights contain a non-finite value. Check the '
                       'input error array for any zeros or non-finite '
                       'values.')
                raise ValueError(msg)

        fit_info_keys = ('fvec', 'fun', 'param_cov', 'ierr', 'message',
                         'status')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            fit_model = self.fitter(psf_model, xi, yi, cutout,
                                    weights=weights, **kwargs)
            with contextlib.suppress(AttributeError):
                fit_model.clear_cache()

            fit_info = {key: self.fitter.fit_info.get(key) for key in
                        fit_info_keys if self.fitter.fit_info.get(key)
                        is not None}

        return fit_model, fit_info

    def _perform_fits(self, source_groups, data, mask, error):
        """
        Fit all sources or groups of sources.

        This function is a generic fitting loop that iterates over a
        list of fittable units (either single sources or groups), runs
        the fitter for each, and returns the raw results.

        Parameters
        ----------
        source_groups : list of `~astropy.table.Table`
            A list where each element is a table representing a single
            source or a group of sources to be fit simultaneously.

        Returns
        -------
        fit_models : list
            A list of the raw fitted model objects from the fitter.

        fit_infos : list
            A list of the ``fit_info`` dictionaries from the fitter.
        """
        fit_models = []
        fit_infos = []

        if self.progress_bar:  # pragma: no cover
            source_groups = add_progress_bar(source_groups,
                                             desc='Fit source/group')

        for source_group in source_groups:
            nmodels = len(source_group)
            self._group_results['nmodels'].append([nmodels] * nmodels)
            psf_model = self._make_psf_model(source_group)
            fit_model, fit_info = self._run_fitter(psf_model, source_group,
                                                   data, mask, error)
            fit_models.append(fit_model)
            fit_infos.append(fit_info)

        return fit_models, fit_infos

    def _finalize_results(self, fit_models, fit_infos, is_grouped):
        """
        Parse the raw fitter results and assemble the final fit_params
        table.

        This method uses the `is_grouped` flag to determine whether to
        use the simple parsing logic for ungrouped sources or the more
        complex `_parse_fit_results` for grouped sources.

        Parameters
        ----------
        fit_models : list
            The raw fitted models from `_perform_fits`.

        fit_infos : list
            The raw `fit_info` dictionaries from `_perform_fits`.

        is_grouped : bool
            A flag indicating whether the fits were performed on grouped
            sources.

        Returns
        -------
        fit_params : `~astropy.table.QTable`
            The final, processed table of fitted parameters and errors.
        """
        self._group_results['fit_infos'] = fit_infos

        if is_grouped:
            # use the complex parser for compound models
            final_models = self._parse_fit_results(fit_models, fit_infos)
        else:
            # simple case: results are already 1-to-1 with sources
            final_models = fit_models
            param_covs = [info.get('param_cov', None) for info in fit_infos]
            nfitparam = len(self._param_mapper.fitted_param_names)

            fit_param_errs = []
            for cov in param_covs:
                if cov is None:
                    fit_param_errs.append(np.array([np.nan] * nfitparam))
                else:
                    fit_param_errs.append(np.sqrt(np.diag(cov)))

            self.fit_info['fit_infos'] = fit_infos
            self.fit_info['fit_error_indices'] = self._get_fit_error_indices()
            self.fit_info['fit_param_errs'] = np.array(fit_param_errs)

        fitted_models_table = self._all_model_params_to_table(final_models)
        self._fitted_models_table = fitted_models_table

        self.fit_params = self._prepare_fit_results(fitted_models_table)

        return self.fit_params

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        """
        Dispatcher to prepare, execute, and finalize PSF fitting.
        """
        # determine if sources are grouped
        _, counts = np.unique(init_params['group_id'], return_counts=True)
        is_grouped = np.any(counts > 1)

        # create list of source groups to fit
        if is_grouped:
            groups = init_params.group_by('group_id')
            self._group_results['ungroup_indices'] = np.argsort(
                groups['id'].value)
            source_groups = groups.groups
        else:
            self._group_results['ungroup_indices'] = np.arange(
                len(init_params))
            source_groups = [Table(row) for row in init_params]

        # run the fitting loop on the source groups
        raw_fit_models, raw_fit_infos = self._perform_fits(
            source_groups, data, mask, error,
        )

        # parse the raw results and assemble the final table
        return self._finalize_results(raw_fit_models, raw_fit_infos,
                                      is_grouped)

    def _calc_fit_metrics(self, results_tbl):
        # Keep cen_idx as a list because it can have NaNs with the ints.
        # If NaNs are present, turning it into an array will convert the
        # ints to floats, which cannot be used as slices.
        cen_idx = self._ungroup(self._group_results['psfcenter_indices'])

        split_index = [np.cumsum(npixfit)[:-1]
                       for npixfit in self._group_results['npixfit']]

        # find the key with the fit residual (fitter dependent)
        finfo_keys = self._group_results['fit_infos'][0].keys()
        keys = ('fvec', 'fun')
        key = None
        for key_ in keys:
            if key_ in finfo_keys:
                key = key_

        # For fitters that do not return residuals (e.g.,
        # SimplexLSQFitter), return NaNs. We could manually compute the
        # residuals, but it would require storing the cutouts for each
        # source, which could increase memory usage significantly.
        if key is None:
            qfit = np.full(len(results_tbl), np.nan)
            cfit = np.full(len(results_tbl), np.nan)
            return qfit, cfit

        fit_residuals = []
        for idx, fit_info in zip(split_index,
                                 self._group_results['fit_infos'],
                                 strict=True):
            fit_residuals.extend(np.split(fit_info[key], idx))
        fit_residuals = self._order_by_id(fit_residuals)

        with warnings.catch_warnings():
            # ignore divide-by-zero if flux = 0
            warnings.simplefilter('ignore', RuntimeWarning)

            flux_col = self._param_mapper.fit_colnames['flux']
            qfit = []
            cfit = []
            for index, (residual, cen_idx_) in enumerate(
                    zip(fit_residuals, cen_idx, strict=True)):

                flux_fit = results_tbl[flux_col][index]
                if isinstance(flux_fit, u.Quantity):
                    flux_fit = flux_fit.value

                if flux_fit == 0:
                    qfit.append(np.inf)
                    cfit.append(np.inf)
                else:
                    qfit.append(np.sum(np.abs(residual)) / flux_fit)

                    if np.isnan(cen_idx_):  # masked central pixel
                        cen_residual = np.nan
                        cfit.append(np.nan)
                    else:
                        # find residual at center pixel;
                        # astropy fitters compute residuals as
                        # (model - data), thus need to negate the residual
                        cen_residual = -residual[cen_idx_]
                        cfit.append(cen_residual / flux_fit)

        return qfit, cfit

    def _define_flags(self, results_tbl, shape):
        """
        Define flags for the fit results based on various criteria.

        The flags are defined as follows:

        The flags are defined as a bitwise integer, where each bit
        represents a specific condition:

        - 0 : no flags
        - 1 : one or more pixels in the ``fit_shape`` region were masked
              (npixfit < fit_shape)
        - 2 : the fit x and/or y position lies outside of the input data
        - 4 : the fit flux is less than or equal to zero
        - 8 : the fitter may not have converged. In this case, you can
              try increasing the maximum number of fit iterations using the
              ``fitter_maxiters`` keyword.
        - 16 : the fitter parameter covariance matrix was not returned
        - 32 : the fit x or y position is at the bounded value

        Parameters
        ----------
        results_tbl : `~astropy.table.Table`
            The table of fit results.

        shape : tuple of int
            The shape of the input data array (height, width).

        Returns
        -------
        flags : `~numpy.ndarray`
            An array of flags for each source in the results table.
            The flags are integers where each bit represents a specific
            condition as described above.
        """
        flags = np.zeros(len(results_tbl), dtype=int)
        x_col = self._param_mapper.fit_colnames['x']
        y_col = self._param_mapper.fit_colnames['y']
        flux_col = self._param_mapper.fit_colnames['flux']

        # flag=1: if npixfit is less than the maximum number of pixels
        flag1_mask = results_tbl['npixfit'] < np.prod(self.fit_shape)
        flags[flag1_mask] += 1

        # flag=2: the fit x and/or y position lies outside of the input data
        # Since integer coordinates are at pixel centers, the image
        # boundaries are -0.5 to ny-0.5 and -0.5 to nx-0.5.
        ny, nx = shape
        x_fit = results_tbl[x_col]
        y_fit = results_tbl[y_col]
        flag2_mask = ((x_fit < -0.5) | (y_fit < -0.5) | (x_fit > nx - 0.5)
                      | (y_fit > ny - 0.5))
        flags[flag2_mask] += 2

        # flag=4: the fit flux is less than or equal to zero
        flag4_mask = results_tbl[flux_col] <= 0
        flags[flag4_mask] += 4

        # flag=8: the fitter may not have converged
        flags[self.fit_info['fit_error_indices']] += 8

        # flag=16: the fitter parameter covariance matrix was not returned
        try:
            for index, fit_info in enumerate(self.fit_info['fit_infos']):
                if fit_info['param_cov'] is None:
                    flags[index] += 16
        except (KeyError, IndexError):
            pass

        # flag=32: x or y fitted value is at the bounds
        if self.xy_bounds is not None:
            x_param = self._param_mapper.alias_to_model_param['x']
            y_param = self._param_mapper.alias_to_model_param['y']
            for index, row in enumerate(self._fitted_models_table):
                x_bounds = row[f'{x_param}_bounds']
                y_bounds = row[f'{y_param}_bounds']
                x_bounds = np.array([i for i in x_bounds if i is not None])
                y_bounds = np.array([i for i in y_bounds if i is not None])
                dx = x_bounds - row[x_param]
                dy = y_bounds - row[y_param]
                if np.any(dx == 0) or np.any(dy == 0):
                    flags[index] += 32

        return flags

    def _assemble_results_table(self, init_params, fit_params, data_shape):
        """
        Assemble the final results table from the initial parameters and
        fitted parameters.
        """
        results_tbl = join(init_params, fit_params)

        npixfit = np.array(self._ungroup(self._group_results['npixfit']))
        results_tbl['npixfit'] = npixfit

        nmodels = np.array(self._ungroup(self._group_results['nmodels']))
        index = results_tbl.index_column('group_id') + 1
        results_tbl.add_column(nmodels, name='group_size', index=index)

        qfit, cfit = self._calc_fit_metrics(results_tbl)
        results_tbl['qfit'] = qfit
        results_tbl['cfit'] = cfit

        results_tbl['flags'] = self._define_flags(results_tbl, data_shape)

        meta = _get_meta()
        attrs = ('fit_shape', 'fitter_maxiters', 'aperture_radius',
                 'progress_bar')
        for attr in attrs:
            meta[attr] = getattr(self, attr)
        results_tbl.meta = meta

        if len(self.fit_info['fit_error_indices']) > 0:
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table.',
                          AstropyUserWarning)

        self.fit_info = dict(self.fit_info)
        return results_tbl

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
            A table containing the initial guesses of the model
            parameters (e.g., x, y, flux) for each source. If the x and
            y values are not input, then the ``finder`` keyword must be
            defined. If the flux values are not input, then the initial
            fluxes will be measured using the ``aperture_radius``
            keyword, which must be defined. Note that the initial
            flux values refer to the model flux parameters and are
            not corrected for local background values (computed using
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

            * If the PSF model has additional free parameters that are
              fit, they can be included in the table. The column names
              must match the parameter names in the PSF model. They can
              also be suffixed with either the "_init" or "_fit" suffix.
              The suffix search order is "_init", "" (no suffix), and
              "_fit". For example, if the PSF model has an additional
              parameter named "sigma", then the allowed column names are:
              "sigma_init", "sigma", and "sigma_fit". If the column name
              is not found in the table, then the default value from the
              PSF model will be used.

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
            * ``qfit`` : a quality-of-fit metric defined as the the sum
              of the absolute value of the fit residuals divided by the
              fit flux
            * ``cfit`` : a quality-of-fit metric defined as the
              fit residual in the initial central pixel value divided by
              the fit flux. NaN values indicate that the central pixel
              was masked.
            * ``flags`` : bitwise flag values

              - 0 : no flags
              - 1 : one or more pixels in the ``fit_shape`` region
                were masked
              - 2 : the fit x and/or y position lies outside of the
                input data
              - 4 : the fit flux is less than or equal to zero
              - 8 : the fitter may not have converged. In this case,
                you can try increasing the maximum number of fit
                iterations using the ``fitter_maxiters`` keyword.
              - 16 : the fitter parameter covariance matrix was not
                returned
              - 32 : the fit x or y position is at the bounded value
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

        # reset state from previous runs
        self._reset_results()

        # prepare all inputs for sources to be fit
        data, mask, error, init_params = self._prepare_fit_inputs(
            data, mask=mask, error=error, init_params=init_params,
        )

        # handle the case where no sources were found
        if init_params is None:
            return None

        self.init_params = init_params

        # fit sources defined in init_params
        fit_params = self._fit_sources(data, init_params, error=error,
                                       mask=mask)

        # assemble the final results table
        self.results = self._assemble_results_table(init_params, fit_params,
                                                    data.shape)

        return self.results

    @staticmethod
    def _flatten(iterable):
        """
        Flatten a list of lists.
        """
        return list(chain.from_iterable(iterable))

    @property
    def _model_image_parameters(self):
        """
        A helper property that provides the necessary parameters to
        ModelImageMixin.
        """
        # the local_bkg values do not change during the fit
        return (self.psf_model, self._fitted_models_table,
                self.init_params['local_bkg'], self.progress_bar)

    def make_model_image(self, shape, *, psf_shape=None,
                         include_localbkg=False):
        return ModelImageMixin.make_model_image(
            self, shape, psf_shape=psf_shape,
            include_localbkg=include_localbkg)

    def make_residual_image(self, data, *, psf_shape=None,
                            include_localbkg=False):
        return ModelImageMixin.make_residual_image(
            self, data, psf_shape=psf_shape, include_localbkg=include_localbkg)
