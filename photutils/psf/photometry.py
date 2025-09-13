# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to perform PSF-fitting photometry.
"""

import contextlib
import inspect
import warnings
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
from astropy.utils import lazyproperty
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning

from photutils.background import LocalBackground
from photutils.psf._components import (PSFDataProcessor, PSFFitter,
                                       PSFResultsAssembler)
from photutils.psf.groupers import SourceGrouper
from photutils.psf.utils import (ModelImageMixin, _create_call_docstring,
                                 _get_psf_model_main_params, _make_mask,
                                 _validate_psf_model)
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr

__all__ = ['PSFPhotometry']


@dataclass
class _PSFParameterMapper:
    """
    Helper class to map PSF model parameter names to table column names.
    """

    psf_model: object
    alias_to_model_param: dict = field(init=False, repr=False)

    # Valid column names that can be used for initial (x, y, flux)
    # positions. Order matters: the first matched name in each tuple will
    # be used.
    VALID_INIT_COLNAMES = {  # noqa: RUF012
        'x': (
            'x_init', 'xinit', 'x', 'x_0', 'x0', 'xcentroid',
            'x_centroid', 'x_peak', 'xcen', 'x_cen', 'xpos', 'x_pos',
            'x_fit', 'xfit',
        ),
        'y': (
            'y_init', 'yinit', 'y', 'y_0', 'y0', 'ycentroid',
            'y_centroid', 'y_peak', 'ycen', 'y_cen', 'ypos', 'y_pos',
            'y_fit', 'yfit',
        ),
        'flux': (
            'flux_init', 'fluxinit', 'flux', 'flux_0', 'flux0',
            'flux_fit', 'fluxfit', 'source_sum', 'segment_flux',
            'kron_flux',
        ),
    }
    MAIN_ALIASES = ('x', 'y', 'flux')

    def __post_init__(self):
        self.alias_to_model_param = self._get_model_params_map()

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
        params_map = dict(zip(self.MAIN_ALIASES, main_params, strict=True))

        # extra parameters that are not 'x', 'y', or 'flux', but
        # are free to be fit (fixed = False), are added to the map
        # with their own aliases
        fitted_params = [
            param for param in self.psf_model.param_names
            if not self.psf_model.fixed[param]
        ]
        extra_params = [param for param in fitted_params
                        if param not in main_params]

        params_map.update({key: key for key in extra_params})
        return params_map

    @property
    def fitted_param_names(self):
        """
        Get list of model parameter names that will be fitted.
        """
        return [param for param in self.psf_model.param_names
                if not self.psf_model.fixed[param]]

    def get_init_colname(self, alias):
        """
        Get initialization column name for parameter alias.
        """
        return f'{alias}_init'

    def get_fit_colname(self, alias):
        """
        Get fitted parameter column name for parameter alias.
        """
        return f'{alias}_fit'

    def get_err_colname(self, alias):
        """
        Get error column name for parameter alias.
        """
        return f'{alias}_err'

    @property
    def init_colnames(self):
        """
        Dictionary mapping aliases to initialization column names.
        """
        return {alias: self.get_init_colname(alias)
                for alias in self.alias_to_model_param}

    @property
    def fit_colnames(self):
        """
        Dictionary mapping aliases to fitted parameter column names.
        """
        return {alias: self.get_fit_colname(alias)
                for alias in self.alias_to_model_param}

    @property
    def err_colnames(self):
        """
        Dictionary mapping aliases to error column names.
        """
        return {alias: self.get_err_colname(alias)
                for alias in self.alias_to_model_param}

    @property
    def model_param_to_alias(self):
        """
        Dictionary mapping model parameter names to aliases.
        """
        return {v: k for k, v in self.alias_to_model_param.items()}

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
            valid_names = self.VALID_INIT_COLNAMES[param_alias]
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
        flux parameters. The model must be two-dimensional such that it
        accepts 2 inputs (e.g., x and y) and provides 1 output.

    fit_shape : int or length-2 array_like
        The rectangular shape around the initial source position that
        will be used to define the PSF-fitting data. If ``fit_shape``
        is a scalar then a square shape of size ``fit_shape`` will
        be used. If ``fit_shape`` has two elements, they must be in
        ``(ny, nx)`` order. Each element of ``fit_shape`` must be an odd
        number. In general, ``fit_shape`` should be set to a small size
        (e.g., ``(5, 5)``) that covers the region with the highest flux
        signal-to-noise.

    finder : callable or `~photutils.detection.StarFinderBase` or `None`, \
            optional
        A callable used to identify sources in an image. The
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
        A callable used to group sources. Typically, grouped sources
        are those that overlap with their neighbors. Sources that are
        grouped are fit simultaneously. The ``grouper`` must accept
        the x and y coordinates of the sources and return an integer
        array of the group id numbers (starting from 1) indicating
        the group in which a given source belongs. If `None`, then no
        grouping is performed, i.e. each source is fit independently.
        The ``group_id`` values in ``init_params`` override this
        keyword. A warning is raised if any group size is larger than
        ``group_warning_threshold`` sources.

    fitter : `~astropy.modeling.fitting.Fitter`, optional
        The fitter object used to perform the fit of the
        model to the data. If `None`, then the default
        `astropy.modeling.fitting.TRFLSQFitter` is used.

    fitter_maxiters : int, optional
        The maximum number of iterations in which the ``fitter`` is
        called for each source. The value can be increased if the fit is
        not converging for sources.

    xy_bounds : `None`, float, or 2-tuple of float, optional
        The maximum distance in pixels that a fitted source can be from
        the initial (x, y) position. If a single float, then the same
        maximum distance is used for both x and y. If a 2-tuple of
        floats, then the distances are in ``(x, y)`` order. If `None`,
        then no bounds are applied. Either value can also be `None` to
        indicate no bound along that axis.

    aperture_radius : float or `None`, optional
        The radius of the circular aperture used to estimate the initial
        flux of each source. If `None`, then the initial flux values
        must be provided in the ``init_params`` table. The aperture
        radius must be a strictly positive scalar. If initial flux
        values are present in the ``init_params`` table, they will
        override this keyword.

    localbkg_estimator : `~photutils.background.LocalBackground` or `None`, \
            optional
        The object used to estimate the local background around each
        source. If `None`, then no local background is subtracted. The
        ``local_bkg`` values in ``init_params`` override this keyword.
        This option should be used with care, especially in crowded
        fields where the ``fit_shape`` of sources overlap (see Notes
        below).

    group_warning_threshold : int, optional
        The maximum number of sources in a group before a warning is
        raised. If the number of sources in a group exceeds this value,
        a warning is raised to inform the user that fitting such large
        groups may take a long time and be error-prone. The default is
        25 sources.

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
    cutout region for each source. After running `PSFPhotometry`,
    you can use the `results_to_init_params` method to generate a
    table of initial parameters that can be used in a subsequent call
    to `PSFPhotometry`. This table will contain the fitted (x, y)
    positions, fluxes, and any other model parameters that were fit.

    If the fitted model parameters are NaN, then the source was
    not valid, likely due to not enough valid data pixels in the
    ``fit_shape`` region. The ``flags`` column in the output ``results``
    table indicates the reason why a source was not valid.

    If the fitted model parameter errors are NaN, then either the fit
    did not converge, the model parameter was fixed, or the input
    ``fitter`` did not return parameter errors. For the later case, one
    can try a different Astropy fitter that returns parameter errors.

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

    Care should be taken in defining the source groups. Simultaneously
    fitting very large source groups is computationally expensive and
    error-prone. Internally, source grouping requires the creation of
    a compound Astropy model. Due to the way compound Astropy models
    are currently constructed, large groups also require excessively
    large amounts of memory; this will hopefully be fixed in a future
    Astropy version. A warning will be raised if the number of sources
    in a group exceeds the ``group_warning_threshold`` value.
    """

    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 aperture_radius=None, localbkg_estimator=None,
                 group_warning_threshold=25, progress_bar=False):

        self.psf_model = _validate_psf_model(psf_model)
        self._param_mapper = _PSFParameterMapper(self.psf_model)

        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 0),
                                 check_odd=True)
        self.finder = self._validate_callable(finder, 'finder')
        self.grouper = self._validate_grouper(grouper)
        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.fitter_maxiters = self._validate_maxiters(fitter_maxiters)
        self.xy_bounds = self._validate_bounds(xy_bounds)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.localbkg_estimator = self._validate_localbkg(
            localbkg_estimator, 'localbkg_estimator')
        self.group_warning_threshold = group_warning_threshold
        self.progress_bar = progress_bar

        self._data_processor = PSFDataProcessor(
            self._param_mapper, self.fit_shape, finder=self.finder,
            aperture_radius=self.aperture_radius,
            localbkg_estimator=self.localbkg_estimator,
        )

        self._psf_fitter = PSFFitter(
            self.psf_model, self._param_mapper, fitter=self.fitter,
            fitter_maxiters=self.fitter_maxiters, xy_bounds=self.xy_bounds,
            group_warning_threshold=self.group_warning_threshold,
        )

        self._results_assembler = PSFResultsAssembler(
            self._param_mapper, self.fit_shape, xy_bounds=self.xy_bounds,
        )

        # used by the __repr__ method and the output table metadata
        self._attrs = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                       'fitter_maxiters', 'xy_bounds', 'aperture_radius',
                       'localbkg_estimator', 'group_warning_threshold',
                       'progress_bar')

        self._reset_results()

    def _reset_results(self):
        """
        Reset internal state attributes for each __call__.
        """
        self.data_unit = None
        self.finder_results = None
        self.init_params = None
        self.results = None
        self.fit_info = []

        # sync data_unit with components
        if hasattr(self, '_data_processor'):
            self._data_processor.data_unit = None

        # internal state container
        self._state = {
            'valid_mask_by_id': None,
            'fit_param_errs': None,
            'fit_error_indices': None,
            'fitted_models_table': None,
            'npixfit': None,
            'group_size': None,
            'invalid_reasons': None,
            'sum_abs_residuals': None,
            'cen_residuals': None,
            'reduced_chi2': None,
        }

        # remove cached properties
        self.__dict__.pop('_model_image_params', None)

    def _initialize_source_state_storage(self, n_sources):
        """
        Initialize the per-source arrays used to store the fit results
        in the state container.

        Parameters
        ----------
        n_sources : int
            The number of sources to initialize the arrays for.
        """
        nfitparam = len(self._param_mapper.fitted_param_names)
        self._state.update({
            'fit_param_errs': np.full((n_sources, nfitparam), np.nan),
            'npixfit': np.zeros(n_sources, dtype=int),
            'invalid_reasons': [''] * n_sources,
            'sum_abs_residuals': np.full(n_sources, np.nan, dtype=float),
            'cen_residuals': np.full(n_sources, np.nan, dtype=float),
            'reduced_chi2': np.full(n_sources, np.nan, dtype=float),
            'group_size': np.ones(n_sources, dtype=int),
            'valid_mask_by_id': np.full(n_sources, fill_value=False,
                                        dtype=bool),
        })

        # Initialize model parameter storage directly
        self._init_model_param_storage(n_sources)
        self.fit_info = [{}] * n_sources

    def _init_model_param_storage(self, n_sources):
        """
        Initialize storage for model parameters directly in state.

        This avoids storing the full model objects and instead stores
        only the parameter values, fixed flags, and bounds that are
        needed for the results table.
        """
        # get all parameter names from the PSF model
        model_params = list(self.psf_model.param_names)
        flux_param = self._param_mapper.alias_to_model_param['flux']

        # initialize parameter value storage
        param_data = {}
        for model_param in model_params:
            # initialize with appropriate defaults
            default_val = (0.0 if model_param == flux_param else np.nan)

            param_data[model_param] = np.full(n_sources, default_val)
            param_data[f'{model_param}_fixed'] = [None] * n_sources
            param_data[f'{model_param}_bounds'] = [None] * n_sources

        # add placehold IDs column -- this will be updated later to
        # match IDs in init_params
        param_data['id'] = np.arange(1, n_sources + 1)

        self._state['model_param_data'] = param_data

    def _cache_fitted_parameters(self, row_index, model):
        """
        Extract and store model parameters directly instead of storing
        the full model object.

        Parameters
        ----------
        row_index : int
            The index of the source in the results arrays.

        model : astropy.modeling.Model or None
            The fitted model for this source, or None for invalid sources.
        """
        param_data = self._state['model_param_data']
        flux_param = self._param_mapper.alias_to_model_param['flux']

        if model is None:
            # For invalid sources, use default template from psf_model
            template_model = self.psf_model
            for param_name in template_model.param_names:
                if param_name == flux_param:
                    param_data[param_name][row_index] = 0.0
                else:
                    param_data[param_name][row_index] = np.nan

                template_param = getattr(template_model, param_name)
                param_data[f'{param_name}_fixed'][row_index] = (
                    template_param.fixed)
                param_data[f'{param_name}_bounds'][row_index] = (
                    template_param.bounds)
        else:
            # For valid sources, extract actual fitted values
            for param_name in model.param_names:
                param = getattr(model, param_name)
                param_data[param_name][row_index] = param.value
                param_data[f'{param_name}_fixed'][row_index] = param.fixed
                param_data[f'{param_name}_bounds'][row_index] = param.bounds

    def _build_fitted_models_table(self):
        """
        Build the fitted models table from stored parameter data.

        Returns
        -------
        table : `~astropy.table.QTable`
            The table of all model parameters for each source.
        """
        param_data = self._state['model_param_data']
        flux_param = self._param_mapper.alias_to_model_param['flux']

        # Apply data unit to flux parameter if needed
        if self.data_unit is not None:
            param_data[flux_param] = param_data[flux_param] * self.data_unit

        # Create table from parameter data
        table = QTable(param_data)

        # Set id column to match init_params for clean merging
        if hasattr(self, 'init_params') and self.init_params is not None:
            ids = self.init_params['id']
            table['id'] = ids

        return table

    def __repr__(self):
        return make_repr(self, self._attrs)

    @staticmethod
    def _validate_callable(obj, name):
        """
        Validate that the input object is callable.
        """
        if obj is not None and not callable(obj):
            msg = f'{name!r} must be a callable object'
            raise TypeError(msg)
        return obj

    def _validate_grouper(self, grouper):
        """
        Validate the input ``grouper`` value.
        """
        if grouper is not None and not isinstance(grouper, SourceGrouper):
            msg = 'grouper must be a SourceGrouper instance'
            raise ValueError(msg)
        return grouper

    def _validate_bounds(self, xy_bounds):
        """
        Validate the input ``xy_bounds`` value.
        """
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
            if bound is not None:
                if bound <= 0:
                    msg = 'xy_bounds must be strictly positive'
                    raise ValueError(msg)
                if not np.isfinite(bound):
                    msg = 'xy_bounds must be finite'
                    raise ValueError(msg)
        return xy_bounds

    @staticmethod
    def _validate_radius(radius):
        """
        Validate the input ``aperture_radius`` value.
        """
        if radius is not None and (not np.isscalar(radius)
                                   or radius <= 0 or not np.isfinite(radius)):
            msg = 'aperture_radius must be a strictly-positive scalar'
            raise ValueError(msg)
        return radius

    def _validate_localbkg(self, value, name):
        """
        Validate the input ``localbkg_estimator`` value.
        """
        if value is not None and not isinstance(value, LocalBackground):
            msg = 'localbkg_estimator must be a LocalBackground instance'
            raise ValueError(msg)
        return self._validate_callable(value, name)

    def _validate_maxiters(self, maxiters):
        """
        Validate the input ``maxiters`` value.
        """
        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            warnings.warn('"maxiters" will be ignored because it is not '
                          'accepted by the input fitter __call__ method',
                          AstropyUserWarning)
            maxiters = None
        return maxiters

    def _validate_array(self, array, name, data_shape=None):
        """
        Validate input arrays (data, error, mask).

        This method delegates to the data processor component.
        """
        return self._data_processor.validate_array(array, name, data_shape)

    def _validate_init_params(self, init_params):
        """
        Validate the input init_params table.

        This method delegates to the data processor component.
        """
        return self._data_processor.validate_init_params(init_params)

    def _sync_data_unit(self):
        """
        Synchronize data_unit between main class and components.
        """
        if hasattr(self, '_data_processor'):
            self._data_processor.data_unit = self.data_unit

    def _get_aper_fluxes(self, data, mask, init_params):
        """
        Estimate aperture fluxes at the initial (x, y) positions.

        This method delegates to the data processor component.
        """
        return self._data_processor.get_aper_fluxes(data, mask, init_params)

    def _find_sources_if_needed(self, data, mask, init_params):
        """
        Find sources using the finder if initial positions are not
        provided.

        This method delegates to the data processor component and syncs
        results.
        """
        result = self._data_processor.find_sources_if_needed(
            data, mask, init_params)
        self.finder_results = self._data_processor.finder_results
        return result

    def _estimate_flux_and_bkg_if_needed(self, data, mask, init_params):
        """
        Estimate initial fluxes and backgrounds if not provided.

        This method delegates to the data processor component.
        """
        return self._data_processor.estimate_flux_and_bkg_if_needed(
            data, mask, init_params)

    def _group_sources(self, init_params):
        """
        Group sources using the grouper or the user-provided 'group_id'
        column.

        Parameters
        ----------
        init_params : `~astropy.table.Table`
            The table of initial parameters.

        Returns
        -------
        init_params : `~astropy.table.Table`
            The table of initial parameters with a 'group_id' column.
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

        # ensure group_id contains only positive (> 0) integers
        group_id = init_params['group_id']
        if np.any(~np.isfinite(group_id)):
            msg = 'group_id must be finite'
            raise ValueError(msg)
        if not np.issubdtype(group_id.dtype, np.integer):
            msg = 'group_id must be an integer array'
            raise TypeError(msg)
        if np.any(group_id <= 0):
            msg = 'group_id must contain only positive (> 0) integers'
            raise ValueError(msg)

        return init_params

    def _build_initial_parameters(self, data, mask, init_params):
        """
        Build the table of initial parameters for fitting.

        This method orchestrates finding sources, estimating initial
        fluxes and backgrounds, and grouping sources.

        Parameters
        ----------
        data : 2D `numpy.ndarray`
            The input image data.

        mask : 2D `numpy.ndarray` or `None`
            A boolean mask where `True` values are masked.

        init_params : `~astropy.table.Table` or `None`
            The input table of initial parameters.

        Returns
        -------
        init_params : `~astropy.table.Table` or `None`
            The table of initial parameters ready for fitting, or `None`
            if no sources were found.
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
        # fit using the model's default value, if not already present.
        for alias, col_name in self._param_mapper.init_colnames.items():
            if col_name not in init_params.colnames:
                alias_map = self._param_mapper.alias_to_model_param
                model_param_name = alias_map[alias]
                init_params[col_name] = getattr(self.psf_model,
                                                model_param_name)

        # Define the final column order of the init_params table.
        # Extra aliases are those that are not in the main_aliases.
        # The alias and model_param names are the same for
        # extra parameters, so we can use the alias_to_model_param map
        # to get the extra aliases.
        main_aliases = self._param_mapper.MAIN_ALIASES
        extra_aliases = [param
                         for param in self._param_mapper.alias_to_model_param
                         if param not in main_aliases]

        main_cols = [self._param_mapper.init_colnames[alias]
                     for alias in main_aliases]
        extra_cols = [self._param_mapper.init_colnames[alias]
                      for alias in extra_aliases]
        col_order = ['id', 'group_id', 'local_bkg', *main_cols, *extra_cols]

        return init_params[col_order]

    def _prepare_fit_inputs(self, data, *, mask=None, error=None,
                            init_params=None):
        """
        Prepare all inputs for the PSF fitting.

        This method handles data validation, unit processing, source
        finding, initial parameter estimation, and grouping. It returns
        the processed inputs ready for the `_fit_sources` method.

        Parameters
        ----------
        data : 2D array_like
            The input image data.

        mask : 2D array_like or `None`, optional
            A boolean mask where `True` values are masked (ignored).

        error : 2D array_like or `None`, optional
            The 1-sigma uncertainties of the input data.

        init_params : `~astropy.table.Table` or `None`, optional
            The input table of initial parameters.

        Returns
        -------
        data : 2D `numpy.ndarray`
            The validated input image data.

        mask : 2D `numpy.ndarray` or `None`
            The validated boolean mask where `True` values are masked.
            If no mask was input, then `None` is returned.

        error : 2D `numpy.ndarray` or `None`
            The validated 1-sigma uncertainties of the input data.
            If no error was input, then `None` is returned.

        init_params : `~astropy.table.Table` or `None`
            The table of initial parameters ready for fitting, or `None`
            if no sources were found.
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        self.data_unit = unit
        self._sync_data_unit()  # Sync with components
        data = self._validate_array(data, 'data')
        error = self._validate_array(error, 'error', data_shape=data.shape)
        mask = self._validate_array(mask, 'mask', data_shape=data.shape)
        mask = _make_mask(data, mask)

        init_params = self._validate_init_params(init_params)
        init_params = self._build_initial_parameters(data, mask, init_params)

        if init_params is None:
            # no sources found
            return None, None, None, None

        return data, mask, error, init_params

    def _make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources
        within a group.

        For groups, a dynamic flat PSF model is created that avoids the
        performance and memory issues associated with the deeply-nested
        parameter structure and tree traversal of Astropy
        CompoundModels.
        """
        return self._psf_fitter.make_psf_model(sources)

    def _get_fit_offsets(self):
        """
        Return cached (y_offsets, x_offsets) arrays for fit_shape.

        This method delegates to the data processor component.
        """
        return self._data_processor.get_fit_offsets()

    def _should_skip_source(self, row, data_shape):
        """
        Quick validation to skip obviously invalid sources early.

        This method delegates to the data processor component.
        """
        return self._data_processor.should_skip_source(row, data_shape)

    def _get_source_cutout_data(self, row, data, mask, y_offsets, x_offsets):
        """
        Extract per-source pixel indices and cutout data.

        This method delegates to the data processor component.
        """
        return self._data_processor.get_source_cutout_data(
            row, data, mask, y_offsets, x_offsets)

    def _run_fitter(self, psf_model, xi, yi, cutout, error):
        """
        Fit the PSF model to the input cutout data.

        This method delegates to the PSF fitter component.
        """
        return self._psf_fitter.run_fitter(psf_model, xi, yi, cutout, error)

    def _extract_source_covariances(self, group_cov, num_sources, nfitparam):
        """
        Extract individual source covariance matrices from group
        covariance.

        This method delegates to the PSF fitter component.
        """
        return self._psf_fitter.extract_source_covariances(
            group_cov, num_sources, nfitparam)

    def _split_flat_model(self, flat_model, n_sources):
        """
        Split a flat model for grouped sources into individual source
        models.

        This method returns individual PSF models with parameters
        extracted from the flat model's parameter values.
        """
        return self._psf_fitter.split_flat_model(flat_model, n_sources)

    def _ungroup_fit_results(self, row_indices, valid_mask, group_model,
                             group_fit_info):
        """
        Ungroup fitted results and store per-source data.

        This method extracts individual source parameters, errors, and
        covariance information directly from the group fit results and
        stores them in the state container. This avoids storing large
        group (flat) model objects and covariance matrices.

        The results for each valid source in the group are stored
        directly in the state container, including the fitted model
        parameters, parameter errors, and fit_info dictionary.
        ``row_indices`` is used to ensure that the order of the sources
        in the state container matches the source ID order in the input
        ``init_params`` table.

        Parameters
        ----------
        row_indices : list
            The row indices for sources in this group.

        valid_mask : ndarray
            Boolean mask indicating which sources in the group are valid.

        group_model : `astropy.modeling.Model`
            The fitted model for a single group. This can be a compound
            model.

        group_fit_info : dict
            The fit_info dictionary corresponding to the group fit.
        """
        nfitparam = len(self._param_mapper.fitted_param_names)
        num_valid = int(np.count_nonzero(valid_mask))

        # Extract parameter errors from the group covariance matrix
        param_cov = group_fit_info.get('param_cov')
        if param_cov is None:
            source_param_errs = np.full((num_valid, nfitparam), np.nan)
            source_covs = [None] * num_valid
        else:
            param_err_1d = np.sqrt(np.diag(param_cov))

            # For grouped (flat) models, parameters are arranged as,
            # e.g., [flux_0, x_0_0, y_0_0, fwhm_0, flux_1, x_0_1, ...]
            source_param_errs = param_err_1d.reshape(num_valid, nfitparam)

            # Extract individual covariance matrices for each source
            source_covs = self._extract_source_covariances(
                param_cov, num_valid, nfitparam)

        # Split models and extract parameters
        if num_valid == 1:
            source_models = [group_model]
        else:
            # For grouped (flat) models, create individual models from
            # params
            source_models = self._split_flat_model(group_model, num_valid)

        # Store results for each valid source
        valid_idx = 0
        for i, row_index in enumerate(row_indices):
            if not valid_mask[i]:
                continue

            model = source_models[valid_idx]
            param_errs = source_param_errs[valid_idx]
            source_cov = source_covs[valid_idx]

            # Extract and store model parameters
            self._cache_fitted_parameters(row_index, model)
            self._state['fit_param_errs'][row_index] = param_errs

            # Create individual fit_info with source-specific covariance
            source_fit_info = dict(group_fit_info)
            if source_cov is not None:
                source_fit_info['param_cov'] = source_cov

            self.fit_info[row_index] = source_fit_info
            self._state['valid_mask_by_id'][row_index] = True

            valid_idx += 1

    def _calculate_residual_metrics(self, row_indices, valid_mask,
                                    npixfit_full, cen_index_full, error=None,
                                    xi_all=None, yi_all=None):
        """
        Calculate residual-based fit metrics for valid sources.

        Parameters
        ----------
        row_indices : array-like
            Source row indices.

        valid_mask : array-like
            Boolean mask for valid sources.

        npixfit_full : array-like
            Number of pixels used in fit for each source.

        cen_index_full : array-like
            Center pixel indices for each source.

        error : 2D array or None, optional
            The 1-sigma uncertainties of the input data. Used for
            calculating reduced chi-squared.

        xi_all : list or None, optional
            List of x-coordinates for each valid source's cutout pixels.

        yi_all : list or None, optional
            List of y-coordinates for each valid source's cutout pixels.

        Returns
        -------
        sum_abs_residuals : ndarray
            Sum of absolute residuals for each source, or np.nan for
            invalid sources.

        cen_residuals : ndarray
            Center residuals for each source, or np.nan for invalid
            sources.

        reduced_chi2 : ndarray
            Reduced chi-squared values for each source, or np.nan for
            invalid sources.
        """
        # Extract residuals from fit_info
        residual_key = None
        with contextlib.suppress(AttributeError):
            fit_info = self.fitter.fit_info
            if isinstance(fit_info, dict):
                if 'fun' in fit_info:
                    residual_key = 'fun'
                if 'fvec' in fit_info:  # LevMarLSQFitter
                    residual_key = 'fvec'

        if residual_key is not None:
            residuals = self.fitter.fit_info[residual_key]
        else:
            residuals = None

        n_sources = len(row_indices)
        sum_abs_residuals = np.full(n_sources, np.nan, dtype=float)
        cen_residuals = np.full(n_sources, np.nan, dtype=float)
        reduced_chi2 = np.full(n_sources, np.nan, dtype=float)

        if residuals is not None:
            # convert to numpy arrays for vectorized operations
            valid_mask_arr = np.array(valid_mask, dtype=bool)
            npixfit_arr = np.array(npixfit_full)
            cen_index_arr = np.array(cen_index_full)

            # get valid source indices
            valid_indices = np.where(valid_mask_arr)[0]
            if len(valid_indices) > 0:
                npix_valid = npixfit_arr[valid_indices]

                # calculate cumulative pixel positions
                cumsum_npix = np.concatenate(([0], np.cumsum(npix_valid)))

                # get the number of fitted parameters
                nfitparam = len(self._param_mapper.fitted_param_names)

                # process all valid sources
                for idx, valid_idx in enumerate(valid_indices):
                    start_pos = cumsum_npix[idx]
                    end_pos = cumsum_npix[idx + 1]
                    source_residuals = residuals[start_pos:end_pos]

                    # For qfit and cfit calculations, we need raw residuals
                    # (data - model), not weighted residuals
                    # (data - model)/error. If errors were provided, convert
                    # weighted residuals back to raw residuals.
                    raw_residuals = source_residuals
                    if (error is not None and xi_all is not None
                            and yi_all is not None):
                        # Extract error values for this source's pixels
                        xi_source = xi_all[idx]
                        yi_source = yi_all[idx]
                        error_vals = error[yi_source, xi_source]

                        # Convert weighted residuals to raw residuals:
                        # multiply by error
                        if (np.all(error_vals > 0)
                                and np.all(np.isfinite(error_vals))):
                            raw_residuals = source_residuals * error_vals

                    # sum of absolute residuals
                    sum_abs_residuals[valid_idx] = float(
                        np.abs(raw_residuals).sum())

                    # center residual
                    cen_idx = cen_index_arr[valid_idx]
                    if np.isfinite(cen_idx):
                        cen_residuals[valid_idx] = float(
                            -raw_residuals[int(cen_idx)])
                    else:
                        cen_residuals[valid_idx] = np.nan

                    # Calculate chi-squared. The residuals have already
                    # been weighted by (1 / error). If errors are not
                    # input, then reduced_chi2 will be NaN.
                    dof = float(npix_valid[idx] - nfitparam)
                    if (error is not None and xi_all is not None
                            and yi_all is not None):
                        # Extract error values for this source's pixels
                        xi_source = xi_all[idx]
                        yi_source = yi_all[idx]
                        error_vals = error[yi_source, xi_source]

                        if (np.all(error_vals > 0)
                                and np.all(np.isfinite(error_vals))):
                            chi2 = np.sum(source_residuals**2)
                            reduced_chi2[valid_idx] = chi2 / dof

        row_indices_arr = np.array(row_indices)
        self._state['sum_abs_residuals'][row_indices_arr] = sum_abs_residuals
        self._state['cen_residuals'][row_indices_arr] = cen_residuals
        self._state['reduced_chi2'][row_indices_arr] = reduced_chi2

        return sum_abs_residuals, cen_residuals, reduced_chi2

    def _fit_source_groups(self, source_groups, data, mask, error):
        """
        Fit PSF models to groups of sources in the input data.

        This method processes each group of sources, fits PSF models,
        and stores the results. Individual source results are extracted
        and stored as soon as each group is fitted.

        Parameters
        ----------
        source_groups : iterable
            Groups of sources to fit, where each group contains sources
            that should be fit simultaneously.

        data : 2D ndarray
            The input image data.

        mask : 2D ndarray or None
            Boolean mask for the input data.

        error : 2D ndarray or None
            The 1-sigma uncertainties of the input data.
        """
        if self.progress_bar:  # pragma: no cover
            source_groups = add_progress_bar(source_groups,
                                             desc='Fit source/group')

        y_offsets, x_offsets = self._get_fit_offsets()
        nfitparam_per_source = len(self._param_mapper.fitted_param_names)

        # sources are fit by groups in group ID order
        for source_group in source_groups:
            group_size = len(source_group)
            xi_all = []
            yi_all = []
            cutout_all = []
            npixfit_full = []
            cen_index_full = []
            valid_mask_list = []
            invalid_reasons = []
            row_indices = []

            # Process all sources with pre-filtering optimization
            for row in source_group:
                # Always use pre-filtering for all group sizes
                should_skip, reason = self._should_skip_source(row, data.shape)
                if should_skip:
                    res = {
                        'valid': False,
                        'reason': reason,
                        'xx': None,
                        'yy': None,
                        'cutout': None,
                        'npix': 0,
                        'cen_index': np.nan,
                    }
                else:
                    res = self._get_source_cutout_data(row, data, mask,
                                                       y_offsets, x_offsets)

                # Common processing for all sources
                npixfit_full.append(res['npix'])
                cen_index_full.append(res['cen_index'])
                invalid_reasons.append(res['reason'])
                row_indices.append(row['_row_index'])

                if res['valid'] and res['npix'] >= nfitparam_per_source:
                    valid_mask_list.append(True)
                    xi_all.append(res['xx'])
                    yi_all.append(res['yy'])
                    cutout_all.append(res['cutout'])
                else:
                    if res['valid'] and res['npix'] < nfitparam_per_source:
                        invalid_reasons[-1] = 'too_few_pixels'
                    valid_mask_list.append(False)

            valid_mask = np.array(valid_mask_list, dtype=bool)
            num_valid = int(np.count_nonzero(valid_mask))

            # Store basic info for all sources in group.
            # row_indices is used to store results in the original
            # source ID order given by init_params.
            row_indices_arr = np.array(row_indices)
            self._state['group_size'][row_indices_arr] = group_size
            self._state['npixfit'][row_indices_arr] = np.array(npixfit_full,
                                                               dtype=int)

            for i, row_index in enumerate(row_indices):
                reason = invalid_reasons[i]
                self._state['invalid_reasons'][row_index] = (
                    '' if reason is None else reason
                )

            if num_valid == 0:
                # Handle all-invalid group
                for row_index in row_indices:
                    self._state['valid_mask_by_id'][row_index] = False
                    self._cache_fitted_parameters(row_index, None)
                continue

            # Fit the group
            xi_concat = np.concatenate(xi_all)
            yi_concat = np.concatenate(yi_all)
            cutout_concat = np.concatenate(cutout_all)
            valid_sources = source_group[valid_mask]
            psf_model = self._make_psf_model(valid_sources)
            fit_model, fit_info = self._run_fitter(psf_model, xi_concat,
                                                   yi_concat, cutout_concat,
                                                   error)

            # Ungroup and store per-source results. row_indices is used
            # to ensure that results are stored in the original source
            # ID order given by init_params.
            self._ungroup_fit_results(row_indices, valid_mask, fit_model,
                                      fit_info)

            # Calculate residual metrics for valid sources
            self._calculate_residual_metrics(
                row_indices, valid_mask, npixfit_full, cen_index_full,
                error=error, xi_all=xi_all, yi_all=yi_all)

    def _get_fit_error_indices(self):
        """
        Get the indices of fits that did not converge.

        This method delegates to the results assembler component.
        """
        return self._results_assembler.get_fit_error_indices(self.fit_info)

    def _create_fit_results(self, fit_model_all_params):
        """
        Create the table of fitted parameter values and errors.

        This method delegates to the results assembler component.
        """
        fit_param_errs = self._state['fit_param_errs']
        valid_mask = self._state.get('valid_mask_by_id')

        return self._results_assembler.create_fit_results(
            fit_model_all_params, fit_param_errs, valid_mask, self.data_unit)

    def _assemble_fit_results(self):
        """
        Assemble the final fitted results tables and parameters.

        This method creates the fitted models table and fit parameters
        table from the per-source data that was stored during the
        fitting process. It also computes fit error indices.

        Returns
        -------
        fit_params : `~astropy.table.Table`
            Table containing the fitted parameters and their errors.
        """
        fit_error_indices = self._get_fit_error_indices()
        fitted_models_table = self._build_fitted_models_table()
        fit_params = self._create_fit_results(fitted_models_table)

        # store results in state for other methods that need them
        self._state['fit_error_indices'] = fit_error_indices
        self._state['fitted_models_table'] = fitted_models_table
        self._state['fit_params'] = fit_params

        return fit_params

    def _fit_sources(self, data, init_params, *, error=None, mask=None):
        """
        Fit PSF models to sources in the input data.

        Parameters
        ----------
        data : 2D ndarray
            The input image data.

        init_params : `~astropy.table.Table`
            The table of initial parameters for each source.

        error : 2D ndarray or `None`, optional
            The 1-sigma uncertainties of the input data.

        mask : 2D ndarray or `None`, optional
            A boolean mask where `True` values are masked (ignored).

        Returns
        -------
        fit_params : `~astropy.table.Table`
            The table of fitted parameters and fit quality metrics for
            each source.
        """
        # add row index for stable mapping
        if '_row_index' not in init_params.colnames:
            init_params['_row_index'] = np.arange(len(init_params))
        self._initialize_source_state_storage(len(init_params))

        source_groups = init_params.group_by('group_id').groups
        self._fit_source_groups(source_groups, data, mask, error)

        # clean up temporary row index column
        if '_row_index' in init_params.colnames:
            init_params.remove_column('_row_index')

        return self._assemble_fit_results()

    def _calc_fit_metrics(self, results_tbl):
        """
        Calculate fit quality metrics qfit, cfit, and reduced_chi2.

        This method delegates to the results assembler component.
        """
        sum_abs_residuals = self._state['sum_abs_residuals']
        cen_residuals = self._state['cen_residuals']
        reduced_chi2 = self._state['reduced_chi2']

        return self._results_assembler.calc_fit_metrics(
            results_tbl, sum_abs_residuals, cen_residuals, reduced_chi2)

    def _define_flags(self, results_tbl, shape):
        """
        Define per-source bitwise flags summarizing fit conditions.

        This method delegates to the results assembler component.
        """
        fit_error_indices = self._state.get('fit_error_indices')
        fitted_models_table = self._state.get('fitted_models_table')
        valid_mask = self._state.get('valid_mask_by_id')
        invalid_reasons = self._state.get('invalid_reasons')

        return self._results_assembler.define_flags(
            results_tbl, shape, fit_error_indices, self.fit_info,
            fitted_models_table, valid_mask, invalid_reasons)

    def _assemble_results_table(self, init_params, fit_params, data_shape):
        """
        Assemble the final results table.

        This method delegates to the results assembler component.
        """
        # prepare metadata attributes
        class_attrs = {'psf_model', 'finder', 'grouper', 'fitter',
                       'localbkg_estimator'}
        metadata_attrs = {}
        for attr in self._attrs:
            value = getattr(self, attr)
            if attr in class_attrs and value is not None:
                metadata_attrs[attr] = repr(value)
            else:
                metadata_attrs[attr] = value

        return self._results_assembler.assemble_results_table(
            init_params, fit_params, data_shape, self._state,
            self._calc_fit_metrics, self._define_flags,
            self.__class__.__name__, metadata_attrs)

    @staticmethod
    def _coerce_nddata(data):
        """
        Return normalized (data, mask, error) if ``data`` is NDData.

        This helper extracts ``data.data``, propagates units, and
        derives an error array from an attached ``StdDevUncertainty``
        (or compatible uncertainty) if present.

        Parameters
        ----------
        data : `~astropy.nddata.NDData`
            The input data.

        Returns
        -------
        data_array : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
            The 2D data array.

        mask : 2D bool `~numpy.ndarray` or `None`
            The boolean mask array.

        error : 2D `~numpy.ndarray` or `~astropy.units.Quantity` or `None`
            The 1-sigma error array.
        """
        data_array = data.data
        if data.unit is not None:
            data_array = data_array << data.unit

        mask = data.mask

        unc = data.uncertainty
        error = None
        if unc is not None:
            err = unc.represent_as(StdDevUncertainty).quantity
            if getattr(err, 'unit', None) == u.dimensionless_unscaled:
                err = err.value
            elif data.unit is not None:
                err = err.to(data.unit)
            error = err

        return data_array, mask, error

    @_create_call_docstring(iterative=False)
    def __call__(self, data, *, mask=None, error=None, init_params=None):
        # reset state from previous runs
        self._reset_results()

        try:
            # handle NDData input
            if isinstance(data, NDData):
                data, mask, error = self._coerce_nddata(data)

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
            # Note: _assemble_results_table handles _state cleanup
            self.results = self._assemble_results_table(
                init_params, fit_params, data.shape)

        except Exception:
            # ensure state cleanup even if an exception occurs
            self._reset_state()
            raise

        self._reset_state()
        return self.results

    def _reset_state(self):
        """
        Reset _state dictionary in case of exceptions.

        This ensures memory is freed even if the normal cleanup path is
        not reached due to an exception during processing.
        """
        if hasattr(self, '_state') and self._state:
            self._state.clear()

    @property
    @deprecated('2.3.0', alternative='results')
    def fit_params(self):
        """
        The table of fit parameters and their errors.

        This table is a subset of the ``results`` table, containing
        only the fit parameters and their errors. It can be used as the
        ``init_params`` for subsequent `PSFPhotometry` fits.
        """
        if self.results is None:
            return None

        tbl = QTable()
        for col_name in self.results.colnames:
            if col_name == 'id' or '_fit' in col_name or '_err' in col_name:
                tbl[col_name] = self.results[col_name]

        return tbl

    @staticmethod
    def _results_to_init_params(results_tbl, reset_id=True):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to those expected for the
        initial parameters table. It can be used as the ``init_params``
        for subsequent `PSFPhotometry` fits.

        Rows that contain non-finite fitted values are removed.

        This is a static helper method to allow it to be called by
        IterativePSFPhotometry.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable`
            The table of fit results from a previous `PSFPhotometry` run.

        reset_id : bool, optional
            If `True`, the 'id' column will be reset to a sequential
            numbering starting from 1. If `False`, the 'id' column will
            be copied as is from `results_tbl`. This is useful only in
            the the case where there are non-finite fitted values in the
            table, which would otherwise result in a non-sequential 'id'
            column.

        Returns
        -------
        init_params_tbl : `~astropy.table.QTable` or `None`
            A table of initial parameters for the next `PSFPhotometry`
            run, or `None` if `results_tbl` is `None`.
        """
        if results_tbl is None:
            return None

        tbl = QTable()
        for col_name in results_tbl.colnames:
            if col_name == 'id' or '_fit' in col_name:
                init_name = col_name.replace('_fit', '_init')
                tbl[init_name] = results_tbl[col_name]

        # remove rows that contain non-finite values
        keep = np.all([np.isfinite(tbl[col]) for col in tbl.colnames], axis=0)
        tbl = tbl[keep]

        # reset the 'id' column to a sequential numbering starting from 1
        if reset_id:
            tbl['id'] = np.arange(1, len(tbl) + 1)

        return tbl

    @staticmethod
    def _results_to_model_params(results_tbl, param_mapper, reset_id=True):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to the PSF model parameter
        names. It can also be used to reconstruct the fitted PSF models
        for visualization or further analysis.

        Rows that contain non-finite fitted values are removed.

        This is a static helper method to allow it to be called by
        IterativePSFPhotometry.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable`
            The table of fit results from a previous `PSFPhotometry`
            run.

        reset_id : bool, optional
            If `True`, the 'id' column will be reset to a sequential
            numbering starting from 1. If `False`, the 'id' column will
            be copied as is from `results_tbl`. This is useful only in
            the the case where there are non-finite fitted values in the
            table, which would otherwise result in a non-sequential 'id'
            column.

        param_mapper : `_PSFParameterMapper`
            The helper class that manages the mapping between PSF model
            parameter names and table column names.

        Returns
        -------
        model_params_tbl : `~astropy.table.QTable` or `None`
            A table of fitted model parameters, or `None` if
            `results_tbl` is `None`.
        """
        if results_tbl is None:
            return None

        tbl = QTable()
        for col_name in results_tbl.colnames:
            if col_name == 'id' or '_fit' in col_name:
                alias = col_name.replace('_fit', '')
                model_param_name = param_mapper.alias_to_model_param.get(
                    alias, alias)
                tbl[model_param_name] = results_tbl[col_name]

        # remove rows that contain non-finite values
        keep = np.all([np.isfinite(tbl[col]) for col in tbl.colnames], axis=0)
        tbl = tbl[keep]

        # reset the 'id' column to a sequential numbering starting from 1
        if reset_id:
            tbl['id'] = np.arange(1, len(tbl) + 1)

        return tbl

    def results_to_init_params(self):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to those expected for the
        initial parameters table. It can be used as the ``init_params``
        for subsequent `PSFPhotometry` fits.

        Rows that contain non-finite fitted values are removed.
        """
        return self._results_to_init_params(self.results, reset_id=True)

    def results_to_model_params(self):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to the PSF model parameter
        names. It can also be used to reconstruct the fitted PSF models
        for visualization or further analysis.

        Rows that contain non-finite fitted values are removed.
        """
        return self._results_to_model_params(self.results,
                                             self._param_mapper,
                                             reset_id=True)

    @lazyproperty
    def _model_image_params(self):
        """
        A helper property that provides the necessary parameters to
        ModelImageMixin.
        """
        return {'psf_model': self.psf_model,
                'model_params': self.results_to_model_params(),
                'local_bkg': self.init_params['local_bkg'],
                'progress_bar': self.progress_bar,
                }

    def make_model_image(self, shape, *, psf_shape=None,
                         include_localbkg=False):
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        return ModelImageMixin.make_model_image(
            self, shape, psf_shape=psf_shape,
            include_localbkg=include_localbkg)

    def make_residual_image(self, data, *, psf_shape=None,
                            include_localbkg=False):
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        return ModelImageMixin.make_residual_image(
            self, data, psf_shape=psf_shape, include_localbkg=include_localbkg)
