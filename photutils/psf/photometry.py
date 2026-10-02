# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for performing PSF-fitting photometry.
"""

import inspect
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from queue import SimpleQueue
from typing import NamedTuple

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning

from photutils.background import LocalBackground
from photutils.psf._components import (PSFDataProcessor, PSFFitter,
                                       PSFResultsAssembler,
                                       _make_model_image_docstring,
                                       _make_residual_image_docstring,
                                       _ModelImageMaker)
from photutils.psf.flags import decode_psf_flags
from photutils.psf.utils import (_create_call_docstring,
                                 _get_psf_model_main_params, _make_mask,
                                 _validate_psf_model)
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import PhotutilsDeprecationWarning

__all__ = ['PSFPhotometry']


@dataclass
class _PSFParameterMapper:
    """
    Helper class to map PSF model parameter names to table column names.
    """

    psf_model: object
    alias_to_model_param: dict = field(init=False, repr=False)

    # Valid column names that can be used for initial (x, y, flux)
    # positions. Order matters, as the first matched name in each tuple will
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
        # The order of the main parameters is important. It defines
        # the order of table outputs.
        main_params = _get_psf_model_main_params(self.psf_model)
        params_map = dict(zip(self.MAIN_ALIASES, main_params, strict=True))

        # Extra parameters that are not 'x', 'y', or 'flux', but
        # are free to be fit (fixed = False), are added to the map
        # with their own aliases.
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
            # Valid names for extra parameters are more limited
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


class _GroupFitResult(NamedTuple):
    """
    The fit results of one group of sources (see `_FitEngine.fit_group`).

    The per-source arrays and lists are in the order of the group rows.
    """

    row_indices: np.ndarray
    group_size: int
    n_pixels_fit: np.ndarray
    invalid_reasons: list
    valid_mask: np.ndarray
    param_values: dict
    param_fixed: dict
    param_bounds: dict
    fit_param_errs: np.ndarray
    fit_info: list
    sum_abs_residuals: np.ndarray
    cen_residuals: np.ndarray
    reduced_chi2: np.ndarray


class _FitEngine:
    """
    Fit groups of sources independently of a `PSFPhotometry` instance.

    The engine holds the image arrays and the fitting components and
    returns the results of each group as a `_GroupFitResult` without
    touching any `PSFPhotometry` state, so groups can be fitted in
    any order. An engine must not be used by two threads at once
    because the fitter keeps the information of its last fit. With
    ``n_threads`` > 1, `PSFPhotometry` gives every thread its own engine
    from `thread_copy`.

    Parameters
    ----------
    psf_model : `~astropy.modeling.Model`
        The PSF model.

    param_mapper : `~photutils.psf._components._PSFParameterMapper`
        The parameter mapper of the PSF model.

    data_processor : `~photutils.psf._components.PSFDataProcessor`
        The cutout extractor.

    psf_fitter : `~photutils.psf._components.PSFFitter`
        The group model builder and fitter.

    data, mask, error : `~numpy.ndarray` or `None`
        The image arrays, with any units already stripped.
    """

    def __init__(self, *, psf_model, param_mapper, data_processor, psf_fitter,
                 data, mask, error):
        self.psf_model = psf_model
        self.param_mapper = param_mapper
        self.data_processor = data_processor
        self.psf_fitter = psf_fitter
        self.data = data
        self.mask = mask
        self.error = error
        self.y_offsets, self.x_offsets = data_processor.get_fit_offsets()
        self.n_fit_params = len(param_mapper.fitted_param_names)

    def thread_copy(self):
        """
        Return an engine that can fit groups in another thread.

        Returns
        -------
        engine : `_FitEngine`
            An engine with its own copies of the PSF model and the
            fitter, which hold state that is modified during a fit.
            The image arrays are only read and are shared.
        """
        psf_model = self.psf_model.copy()
        param_mapper = _PSFParameterMapper(psf_model)
        data_processor = PSFDataProcessor(param_mapper,
                                          self.data_processor.fit_shape)
        psf_fitter = PSFFitter(
            psf_model, param_mapper, fitter=deepcopy(self.psf_fitter.fitter),
            fitter_maxiters=self.psf_fitter.fitter_maxiters,
            xy_bounds=self.psf_fitter.xy_bounds)
        return _FitEngine(psf_model=psf_model, param_mapper=param_mapper,
                          data_processor=data_processor,
                          psf_fitter=psf_fitter, data=self.data,
                          mask=self.mask, error=self.error)

    def fit_groups(self, sources):
        """
        Fit the source groups of a table.

        Parameters
        ----------
        sources : `~astropy.table.Table`
            The initial parameters of the sources, which must hold only
            complete groups.

        Returns
        -------
        results : list of `_GroupFitResult`
            One result per group.
        """
        return [self.fit_group(group)
                for group in sources.group_by('group_id').groups]

    def _fitter_residuals(self):
        """
        Return the residual vector of the last fit, or `None` if the
        fitter does not provide one.
        """
        fit_info = getattr(self.psf_fitter.fitter, 'fit_info', None)
        if not isinstance(fit_info, dict):
            return None
        for key in ('fvec', 'fun'):  # fvec is the LevMarLSQFitter key
            if key in fit_info:
                return fit_info[key]
        return None

    def _residual_metrics(self, residuals, valid_mask, n_pixels_fit,
                          cen_index, xi_all, yi_all):
        """
        Calculate the residual-based fit metrics of the valid sources
        of a group.

        Parameters
        ----------
        residuals : `~numpy.ndarray` or `None`
            The concatenated (weighted) residuals of the group fit.

        valid_mask : `~numpy.ndarray`
            Which sources of the group were fitted.

        n_pixels_fit, cen_index : `~numpy.ndarray`
            The number of fitted pixels and the index of the center
            pixel of each source of the group.

        xi_all, yi_all : list of `~numpy.ndarray`
            The pixel coordinates of each valid source's cutout.

        Returns
        -------
        sum_abs_residuals, cen_residuals, reduced_chi2 : `~numpy.ndarray`
            The metrics, NaN for invalid sources or where undefined.
        """
        n_sources = len(valid_mask)
        sum_abs_residuals = np.full(n_sources, np.nan)
        cen_residuals = np.full(n_sources, np.nan)
        reduced_chi2 = np.full(n_sources, np.nan)
        if residuals is None:
            return sum_abs_residuals, cen_residuals, reduced_chi2

        valid_indices = np.flatnonzero(valid_mask)
        n_pixels_valid = n_pixels_fit[valid_indices]
        starts = np.concatenate(([0], np.cumsum(n_pixels_valid)))
        for idx, valid_idx in enumerate(valid_indices):
            source_residuals = residuals[starts[idx]:starts[idx + 1]]

            # The residuals are (data - model) / error when errors are
            # input. qfit and cfit need the raw residuals, so multiply
            # by the errors, which run_fitter has validated as positive
            # and finite.
            raw_residuals = source_residuals
            error_vals = None
            if self.error is not None:
                error_vals = self.error[yi_all[idx], xi_all[idx]]
                raw_residuals = source_residuals * error_vals

            sum_abs_residuals[valid_idx] = float(np.abs(raw_residuals).sum())
            cen_idx = cen_index[valid_idx]
            if np.isfinite(cen_idx):
                cen_residuals[valid_idx] = float(-raw_residuals[int(cen_idx)])

            # The reduced chi-squared needs errors and degrees of
            # freedom
            dof = float(n_pixels_valid[idx] - self.n_fit_params)
            if error_vals is not None and dof > 0:
                reduced_chi2[valid_idx] = np.sum(source_residuals**2) / dof

        return sum_abs_residuals, cen_residuals, reduced_chi2

    def fit_group(self, source_group):
        """
        Fit one group of sources.

        Parameters
        ----------
        source_group : `~astropy.table.Table`
            The initial parameters of the sources of the group,
            including the ``_row_index`` column.

        Returns
        -------
        result : `_GroupFitResult`
            The per-source results, in the order of the group rows.
            Invalid sources have NaN parameters and metrics, the
            parameter ``fixed`` and ``bounds`` settings of the PSF
            model, and an empty ``fit_info``.
        """
        data = self.data
        group_size = len(source_group)
        xi_all = []
        yi_all = []
        cutout_all = []
        n_pixels_fit = []
        cen_index = []
        valid_list = []
        invalid_reasons = []
        row_indices = []

        for row in source_group:
            should_skip, reason = self.data_processor.should_skip_source(
                row, data.shape)
            if should_skip:
                res = {'valid': False, 'reason': reason, 'xx': None,
                       'yy': None, 'cutout': None, 'n_pixels': 0,
                       'cen_index': np.nan}
            else:
                res = self.data_processor.get_source_cutout_data(
                    row, data, self.mask, self.y_offsets, self.x_offsets)

            n_pixels_fit.append(res['n_pixels'])
            cen_index.append(res['cen_index'])
            invalid_reasons.append(res['reason'] or '')
            row_indices.append(row['_row_index'])
            if res['valid'] and res['n_pixels'] >= self.n_fit_params:
                valid_list.append(True)
                xi_all.append(res['xx'])
                yi_all.append(res['yy'])
                cutout_all.append(res['cutout'])
            else:
                if res['valid']:
                    invalid_reasons[-1] = 'too_few_pixels'
                valid_list.append(False)

        row_indices = np.array(row_indices, dtype=int)
        valid_mask = np.array(valid_list, dtype=bool)
        n_pixels_fit = np.array(n_pixels_fit, dtype=int)
        cen_index = np.array(cen_index, dtype=float)
        n_valid = int(np.count_nonzero(valid_mask))

        # The defaults of invalid sources come from the PSF model
        param_values = {}
        param_fixed = {}
        param_bounds = {}
        for name in self.psf_model.param_names:
            param = getattr(self.psf_model, name)
            param_values[name] = np.full(group_size, np.nan)
            param_fixed[name] = [param.fixed] * group_size
            param_bounds[name] = [param.bounds] * group_size
        fit_param_errs = np.full((group_size, self.n_fit_params), np.nan)
        fit_info = [{} for _ in range(group_size)]
        residuals = None

        if n_valid > 0:
            valid_sources = source_group[valid_mask]
            group_model = self.psf_fitter.make_psf_model(valid_sources)
            fit_model, group_fit_info = self.psf_fitter.run_fitter(
                group_model, np.concatenate(xi_all), np.concatenate(yi_all),
                np.concatenate(cutout_all), self.error)
            residuals = self._fitter_residuals()

            # Split the group model and covariance into per-source parts
            param_cov = group_fit_info.get('param_cov')
            if param_cov is None:
                source_errs = np.full((n_valid, self.n_fit_params), np.nan)
                source_covs = [None] * n_valid
            else:
                # The flat model parameters are ordered by source,
                # with all the parameters of the first source followed
                # by those of the second source and so on
                source_errs = np.sqrt(np.diag(param_cov)).reshape(
                    n_valid, self.n_fit_params)
                source_covs = self.psf_fitter.extract_source_covariances(
                    param_cov, n_valid, self.n_fit_params)
            if n_valid == 1:
                source_models = [fit_model]
            else:
                source_models = self.psf_fitter.split_flat_model(fit_model,
                                                                 n_valid)

            for valid_idx, i in enumerate(np.flatnonzero(valid_mask)):
                model = source_models[valid_idx]
                for name in model.param_names:
                    param = getattr(model, name)
                    param_values[name][i] = param.value
                    param_fixed[name][i] = param.fixed
                    param_bounds[name][i] = param.bounds
                fit_param_errs[i] = source_errs[valid_idx]
                source_fit_info = dict(group_fit_info)
                if source_covs[valid_idx] is not None:
                    source_fit_info['param_cov'] = source_covs[valid_idx]
                fit_info[i] = source_fit_info

        sum_abs_residuals, cen_residuals, reduced_chi2 = (
            self._residual_metrics(residuals, valid_mask, n_pixels_fit,
                                   cen_index, xi_all, yi_all))

        return _GroupFitResult(
            row_indices=row_indices, group_size=group_size,
            n_pixels_fit=n_pixels_fit, invalid_reasons=invalid_reasons,
            valid_mask=valid_mask, param_values=param_values,
            param_fixed=param_fixed, param_bounds=param_bounds,
            fit_param_errs=fit_param_errs, fit_info=fit_info,
            sum_abs_residuals=sum_abs_residuals, cen_residuals=cen_residuals,
            reduced_chi2=reduced_chi2)


class PSFPhotometry:
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
        is a scalar then a square shape of size ``fit_shape`` will be
        used. If ``fit_shape`` has two elements, they must be in ``(ny,
        nx)`` order. Each element of ``fit_shape`` must be a positive
        odd number. In general, ``fit_shape`` should be
        set to a small size (e.g., ``(5, 5)``) that covers the region
        with the highest flux signal-to-noise.

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
        array of the group ID numbers (starting from 1) indicating
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
        called for each source. The value can be increased if the fit
        is not converging for sources. This parameter is passed to the
        ``fitter`` if it supports the ``maxiter`` parameter and ignored
        otherwise.

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

    local_bkg_estimator : `~photutils.background.LocalBackground` or `None`, \
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

    n_threads : int, optional
        The number of threads used to fit the sources. The default is
        1 (no multithreading). When ``n_threads`` > 1, the source
        groups are divided into chunks that are fitted concurrently.
        Each group is fitted independently, so the results are
        identical to the single-threaded computation. Every thread
        fits with its own copy of the PSF model and the ``fitter``,
        so the input ``fitter`` is not called and its ``fit_info`` is
        not updated. The fitting runs mostly in Python code that
        holds the global interpreter lock (GIL), so multithreading
        speeds up the fitting only on a free-threaded Python build.
        On a build with the GIL it is slower than a single thread.

    progress_bar : bool, optional
        Whether to display a progress bar when fitting the sources
        (or groups). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Attributes
    ----------
    results : `~astropy.table.QTable` or `None`
        The table of fit results from the most recent call, or `None`
        if the instance has not yet been run. The rows are sorted by
        the source ``id`` column.

    fit_info : list of dict
        A list of dictionaries, one per source in the same row order
        as ``results``, containing information about each source fit
        with keys such as ``param_cov``, ``ierr``, ``message``, and
        ``status`` (the exact keys depend on the ``fitter``).

    finder_results : `~astropy.table.QTable` or `None`
        The table of sources returned by the ``finder`` during the
        most recent call, or `None` if the finder was not used.

    init_params : `~astropy.table.QTable` or `None`
        The table of initial parameters used in the most recent call,
        or `None` if the instance has not yet been run.

    data_unit : `~astropy.units.Unit` or `None`
        The unit of the input data from the most recent call, or
        `None` if the data did not have units.

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
    ``fitter`` did not return parameter errors. For the latter case, one
    can try a different Astropy fitter that returns parameter errors.

    The local background value around each source is optionally
    estimated using the ``local_bkg_estimator`` or obtained from the
    ``local_bkg`` column in the input ``init_params`` table. This local
    background is then subtracted from the data over the ``fit_shape``
    region for each source before fitting the PSF model. For sources
    where their ``fit_shape`` regions overlap, the local background will
    effectively be subtracted twice in the overlapping ``fit_shape``
    regions, even if the source ``grouper`` is input. This is not an
    issue if the sources are well-separated. However, for crowded
    fields, please use the ``local_bkg_estimator`` (or ``local_bkg``
    column in ``init_params``) with care.

    Care should be taken in defining the source groups. Simultaneously
    fitting very large source groups is computationally expensive and
    error-prone, because the number of fitted parameters grows with the
    group size. A warning will be raised if the number of sources in a
    group exceeds the ``group_warning_threshold`` value.

    This class stores per-call state on the instance (e.g., ``results``
    and ``fit_info``), so a single instance must not be called
    concurrently from multiple threads. Create one instance per thread
    for concurrent use. Sharing a single Astropy fitter instance across
    concurrently-used objects is also unsafe because Astropy fitters
    store ``fit_info`` on themselves.
    """

    # Default value for parameter initialization (invalid sources)
    _DEFAULT_PARAM_VALUE = np.nan

    # Results columns ending in '_fit' that are not fitted model
    # parameters
    _NON_PARAM_FIT_COLS = ('n_pixels_fit',)

    @deprecated_renamed_argument('localbkg_estimator',
                                 'local_bkg_estimator', '3.0',
                                 until='4.0')
    def __init__(self, psf_model, fit_shape, *, finder=None, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 aperture_radius=None, local_bkg_estimator=None,
                 group_warning_threshold=25, n_threads=1,
                 progress_bar=False):

        self.psf_model = _validate_psf_model(psf_model)
        self._param_mapper = _PSFParameterMapper(self.psf_model)

        self.fit_shape = as_pair('fit_shape', fit_shape, lower_bound=(1, 1),
                                 check_odd=True)
        self.finder = self._validate_callable(finder, 'finder')
        self.grouper = self._validate_callable(grouper, 'grouper')
        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = self._validate_callable(fitter, 'fitter')
        self.fitter_maxiters = self._validate_maxiters(fitter_maxiters)
        self.xy_bounds = self._validate_bounds(xy_bounds)
        self.aperture_radius = self._validate_radius(aperture_radius)
        self.local_bkg_estimator = self._validate_localbkg(
            local_bkg_estimator, 'local_bkg_estimator')
        self.group_warning_threshold = self._validate_group_threshold(
            group_warning_threshold)
        if (isinstance(n_threads, bool)
                or not isinstance(n_threads, (int, np.integer))
                or n_threads < 1):
            msg = 'n_threads must be a positive integer'
            raise ValueError(msg)
        self.n_threads = int(n_threads)
        self.progress_bar = progress_bar

        self._data_processor = PSFDataProcessor(
            self._param_mapper, self.fit_shape, finder=self.finder,
            aperture_radius=self.aperture_radius,
            local_bkg_estimator=self.local_bkg_estimator,
        )

        self._psf_fitter = PSFFitter(
            self.psf_model, self._param_mapper, fitter=self.fitter,
            fitter_maxiters=self.fitter_maxiters, xy_bounds=self.xy_bounds,
        )

        self._results_assembler = PSFResultsAssembler(
            self._param_mapper, self.fit_shape, xy_bounds=self.xy_bounds,
        )

        # Used by the __repr__ method and the output table metadata
        self._attrs = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                       'fitter_maxiters', 'xy_bounds', 'aperture_radius',
                       'local_bkg_estimator', 'group_warning_threshold',
                       'n_threads', 'progress_bar')

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

        # Sync state with components
        self._data_processor.data_unit = None
        self._data_processor.finder_results = None

        # Internal state container
        self._state = {
            'valid_mask_by_id': None,
            'fit_param_errs': None,
            'fit_error_indices': None,
            'fitted_models_table': None,
            'n_pixels_fit': None,
            'group_size': None,
            'invalid_reasons': None,
            'sum_abs_residuals': None,
            'cen_residuals': None,
            'reduced_chi2': None,
        }

    def _initialize_source_state_storage(self, n_sources):
        """
        Initialize the per-source arrays used to store the fit results
        in the state container.

        Parameters
        ----------
        n_sources : int
            The number of sources to initialize the arrays for.
        """
        n_fit_params = len(self._param_mapper.fitted_param_names)
        self._state.update({
            'fit_param_errs': np.full((n_sources, n_fit_params), np.nan),
            'n_pixels_fit': np.zeros(n_sources, dtype=int),
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
        self.fit_info = [{} for _ in range(n_sources)]

    def _init_model_param_storage(self, n_sources):
        """
        Initialize storage for model parameters directly in state.

        This avoids storing the full model objects and instead stores
        only the parameter values, fixed flags, and bounds that are
        needed for the results table.
        """
        # Get all parameter names from the PSF model
        model_params = list(self.psf_model.param_names)

        # Initialize parameter value storage
        param_data = {}
        for model_param in model_params:
            # Initialize all parameters with np.nan (for invalid sources)
            param_data[model_param] = np.full(n_sources,
                                              self._DEFAULT_PARAM_VALUE)
            param_data[f'{model_param}_fixed'] = [None] * n_sources
            param_data[f'{model_param}_bounds'] = [None] * n_sources

        # Add placehold IDs column. This will be updated later to
        # match IDs in init_params.
        param_data['id'] = np.arange(1, n_sources + 1)

        self._state['model_param_data'] = param_data

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
        if self.init_params is not None:
            ids = self.init_params['id']
            table['id'] = ids

        return table

    def __repr__(self):
        return make_repr(self, self._attrs)

    @staticmethod
    def _validate_type(obj, name, expected_type):
        """
        Validate that object is of expected type.

        Parameters
        ----------
        obj : object or None
            Object to validate.

        name : str
            Name of the parameter for error messages.

        expected_type : type or tuple of types
            Expected type(s) for the object.

        Returns
        -------
        obj : object or None
            The validated object.

        Raises
        ------
        error_type
            If obj is not None and not an instance of expected_type.
        """
        if obj is not None and not isinstance(obj, expected_type):
            type_name = expected_type.__name__
            msg = f'{name} must be a {type_name} instance'
            raise TypeError(msg)
        return obj

    @staticmethod
    def _validate_callable(obj, name):
        """
        Validate that the input object is callable.

        Parameters
        ----------
        obj : object or None
            Object to validate.

        name : str
            Name of the parameter for error messages.

        Returns
        -------
        obj : object or None
            The validated callable object.

        Raises
        ------
        TypeError
            If obj is not None and not callable.
        """
        if obj is not None and not callable(obj):
            msg = f'{name!r} must be a callable object'
            raise TypeError(msg)
        return obj

    def _validate_bounds(self, xy_bounds):
        """
        Validate the input ``xy_bounds`` value.

        Parameters
        ----------
        xy_bounds : float, tuple of float, or None
            The maximum distance(s) in pixels that fitted sources can be
            from initial positions.

        Returns
        -------
        xy_bounds : ndarray or None
            The validated xy_bounds as a 2-element array, or None if
            input was None.

        Raises
        ------
        ValueError
            If xy_bounds has incorrect shape, dimension, or contains
            invalid values (non-positive or non-finite).
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
        # None elements are allowed to indicate no bound along an axis
        if not all(bound is None or isinstance(bound, (int, float,
                                                       np.number))
                   for bound in xy_bounds):
            msg = 'xy_bounds must be numeric'
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

        Parameters
        ----------
        radius : float or None
            The aperture radius value to validate.

        Returns
        -------
        radius : float or None
            The validated aperture radius.

        Raises
        ------
        ValueError
            If radius is not None and is not a strictly positive finite
            scalar.
        """
        if radius is not None:
            if (isinstance(radius, bool) or np.ndim(radius) != 0
                    or radius <= 0 or not np.isfinite(radius)):
                msg = 'aperture_radius must be a strictly-positive scalar'
                raise ValueError(msg)
            radius = float(radius)
        return radius

    @staticmethod
    def _validate_group_threshold(value):
        """
        Validate the input ``group_warning_threshold`` value.

        Parameters
        ----------
        value : int
            The group size threshold to validate.

        Returns
        -------
        value : int
            The validated threshold value.

        Raises
        ------
        ValueError
            If value is not a positive integer.
        """
        if (isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0):
            msg = 'group_warning_threshold must be a positive integer'
            raise ValueError(msg)
        return int(value)

    def _validate_localbkg(self, value, name):
        """
        Validate the input ``local_bkg_estimator`` value.

        Parameters
        ----------
        value : LocalBackground or None
            The local background estimator to validate.

        name : str
            Name of the parameter for error messages.

        Returns
        -------
        value : LocalBackground or None
            The validated local background estimator.

        Raises
        ------
        TypeError
            If value is not None and not a LocalBackground instance.
        """
        value = self._validate_type(value, 'local_bkg_estimator',
                                    LocalBackground)
        return self._validate_callable(value, name)

    def _validate_maxiters(self, maxiters):
        """
        Validate the input ``fitter_maxiters`` value.

        Parameters
        ----------
        maxiters : int or None
            Maximum number of fitter iterations to validate.

        Returns
        -------
        maxiters : int or None
            The validated maxiters value, or None if the fitter doesn't
            support this parameter.

        Raises
        ------
        ValueError
            If maxiters is not a strictly-positive integer.
        """
        if (isinstance(maxiters, bool)
                or not isinstance(maxiters, (int, np.integer))
                or maxiters <= 0):
            msg = 'fitter_maxiters must be a strictly-positive integer'
            raise ValueError(msg)
        maxiters = int(maxiters)

        spec = inspect.signature(self.fitter.__call__)
        if 'maxiter' not in spec.parameters:
            msg = ("'fitter_maxiters' will be ignored because the "
                   "fitter's __call__ method does not accept a "
                   "'maxiter' keyword.")
            warnings.warn(msg, AstropyUserWarning)
            maxiters = None
        return maxiters

    def _sync_data_unit(self):
        """
        Synchronize data_unit between main class and components.

        This method ensures that the data_unit attribute is consistent
        between the PSFPhotometry instance and its internal component
        objects (e.g., _data_processor).

        This method modifies the internal state in-place.
        """
        self._data_processor.data_unit = self.data_unit

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
        # A user-provided group_id column takes precedence for this
        # call
        derived_from_id = False
        if 'group_id' not in init_params.colnames:
            if self.grouper is not None:
                x_col = self._param_mapper.init_colnames['x']
                y_col = self._param_mapper.init_colnames['y']
                init_params['group_id'] = self.grouper(
                    init_params[x_col], init_params[y_col])
            else:
                # No grouper provided, so each source is its own group
                init_params['group_id'] = init_params['id'].copy()
                derived_from_id = True

        # Ensure group_id contains only positive (> 0) integers
        group_id = init_params['group_id']
        if np.any(~np.isfinite(group_id)):
            msg = 'group_id must be finite'
            raise ValueError(msg)
        if not np.issubdtype(group_id.dtype, np.integer):
            if derived_from_id:
                msg = ('group_id (or the id column it was derived from) '
                       'must be an integer array')
            else:
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

        # Strip any units from the x/y position columns
        for axis in ('x', 'y'):
            colname = self._param_mapper.init_colnames[axis]
            if isinstance(init_params[colname], u.Quantity):
                init_params[colname] = init_params[colname].value

        add_flux_bkg = self._data_processor.estimate_flux_and_bkg_if_needed
        init_params = add_flux_bkg(data, mask, init_params)
        init_params = self._group_sources(init_params)

        # Check for large group sizes after grouping is complete
        warn_size = self.group_warning_threshold
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
        self._sync_data_unit()  # sync with components

        data = self._data_processor.validate_array(data, 'data')
        error = self._data_processor.validate_array(error, 'error',
                                                    data_shape=data.shape)
        mask = self._data_processor.validate_array(mask, 'mask',
                                                   data_shape=data.shape)
        mask = _make_mask(data, mask)

        init_params = self._data_processor.validate_init_params(init_params)
        if init_params is not None and len(init_params) == 0:
            msg = 'init_params must contain at least one row'
            raise ValueError(msg)
        init_params = self._build_initial_parameters(data, mask, init_params)

        if init_params is None:
            # No sources found
            return None, None, None, None

        ids = np.asarray(init_params['id'])
        if len(np.unique(ids)) != len(ids):
            msg = 'init_params id column must contain unique values'
            raise ValueError(msg)

        return data, mask, error, init_params

    def _fit_source_groups(self, init_params, data, mask, error):
        """
        Fit the source groups and store the per-source results.

        The groups are fitted by a `_FitEngine`, or by one engine per
        thread when ``n_threads`` > 1, and each group's results are
        scattered into the state container by `_store_group_result`.

        Parameters
        ----------
        init_params : `~astropy.table.Table`
            The initial parameters of the sources, including the
            ``group_id`` and ``_row_index`` columns.

        data : 2D ndarray
            The input image data.

        mask : 2D ndarray or None
            Boolean mask for the input data.

        error : 2D ndarray or None
            The 1-sigma uncertainties of the input data.
        """
        engine = _FitEngine(psf_model=self.psf_model,
                            param_mapper=self._param_mapper,
                            data_processor=self._data_processor,
                            psf_fitter=self._psf_fitter, data=data,
                            mask=mask, error=error)
        grouped = init_params.group_by('group_id')
        n_threads = min(self.n_threads, len(grouped.groups))

        if n_threads == 1:
            groups = grouped.groups
            if self.progress_bar:
                groups = add_progress_bar(groups, desc='Fit source/group')
            for group in groups:
                self._store_group_result(engine.fit_group(group))
            return

        # Every chunk borrows one of n_threads engines, so no engine
        # is ever used by two threads at once.
        engines = SimpleQueue()
        for _ in range(n_threads):
            engines.put(engine.thread_copy())

        def fit_chunk(chunk):
            thread_engine = engines.get()
            try:
                return thread_engine.fit_groups(chunk)
            finally:
                engines.put(thread_engine)

        # Several chunks per thread balance the uneven group costs.
        # The warning filter set by the fitting code in each thread is
        # process-wide on some Python builds, where concurrent threads
        # can restore the filters in the wrong order. Setting the same
        # filter here keeps it in place while the threads run and
        # restores the caller's filters afterward.
        chunks = self._chunk_groups(grouped, 4 * n_threads)
        with (warnings.catch_warnings(),
              ThreadPoolExecutor(max_workers=n_threads) as executor):
            warnings.simplefilter('ignore', AstropyUserWarning)
            futures = [executor.submit(fit_chunk, chunk) for chunk in chunks]
            completed = as_completed(futures)
            if self.progress_bar:
                completed = add_progress_bar(completed,
                                             desc='Fit source chunk',
                                             total=len(futures))
            try:
                for future in completed:
                    for result in future.result():
                        self._store_group_result(result)
            except BaseException:
                # Do not fit the remaining chunks after a failure
                executor.shutdown(cancel_futures=True)
                raise

    @staticmethod
    def _chunk_groups(grouped, n_chunks):
        """
        Split a grouped table into chunks of whole groups with roughly
        equal numbers of sources.

        Parameters
        ----------
        grouped : `~astropy.table.Table`
            The sources grouped by ``group_id``.

        n_chunks : int
            The requested number of chunks. Fewer are returned when
            there are fewer groups.

        Returns
        -------
        chunks : list of `~astropy.table.Table`
            The chunks, each holding one or more complete groups.
        """
        indices = grouped.groups.indices
        n_sources = len(grouped)
        n_chunks = min(n_chunks, len(indices) - 1)
        bounds = [0]
        for k in range(1, n_chunks):
            # The first group boundary at or after the ideal cut
            cut = int(indices[np.searchsorted(indices, k * n_sources
                                              / n_chunks)])
            if bounds[-1] < cut < n_sources:
                bounds.append(cut)
        bounds.append(n_sources)
        return [grouped[start:end] for start, end in pairwise(bounds)]

    def _store_group_result(self, result):
        """
        Scatter the results of one group into the state container.

        Parameters
        ----------
        result : `_GroupFitResult`
            The group results from `_FitEngine.fit_group`.
        """
        rows = result.row_indices
        state = self._state
        state['group_size'][rows] = result.group_size
        state['n_pixels_fit'][rows] = result.n_pixels_fit
        state['valid_mask_by_id'][rows] = result.valid_mask
        state['fit_param_errs'][rows] = result.fit_param_errs
        state['sum_abs_residuals'][rows] = result.sum_abs_residuals
        state['cen_residuals'][rows] = result.cen_residuals
        state['reduced_chi2'][rows] = result.reduced_chi2
        param_data = state['model_param_data']
        for name in self.psf_model.param_names:
            param_data[name][rows] = result.param_values[name]
        for i, row in enumerate(rows):
            state['invalid_reasons'][row] = result.invalid_reasons[i]
            self.fit_info[row] = result.fit_info[i]
            for name in self.psf_model.param_names:
                param_data[f'{name}_fixed'][row] = result.param_fixed[name][i]
                param_data[f'{name}_bounds'][row] = (
                    result.param_bounds[name][i])

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

        # Store results in state for other methods that need them
        self._state['fit_error_indices'] = fit_error_indices
        self._state['fitted_models_table'] = fitted_models_table

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
        # Add row index for stable mapping
        if '_row_index' not in init_params.colnames:
            init_params['_row_index'] = np.arange(len(init_params))
        self._initialize_source_state_storage(len(init_params))

        try:
            self._fit_source_groups(init_params, data, mask, error)
        finally:
            # Clean up temporary row index column
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

    def _define_flags(self, results_tbl, shape, init_params):
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
            fitted_models_table, valid_mask, invalid_reasons, init_params)

    def _assemble_results_table(self, init_params, fit_params, data_shape):
        """
        Assemble the final results table.

        This method delegates to the results assembler component.
        """
        # Prepare metadata attributes
        class_attrs = {'psf_model', 'finder', 'grouper', 'fitter',
                       'local_bkg_estimator'}
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

    @staticmethod
    def _check_nddata_kwargs(*, mask, error):
        """
        Reject explicit mask/error keywords for NDData inputs.

        Parameters
        ----------
        mask : 2D `~numpy.ndarray` or `None`
            The mask keyword value passed to ``__call__``.

        error : 2D `~numpy.ndarray` or `None`
            The error keyword value passed to ``__call__``.

        Raises
        ------
        ValueError
            If mask or error is not None.
        """
        if mask is not None or error is not None:
            msg = ('The mask and error keywords must be None when '
                   'data is an NDData instance. Define the mask and '
                   'uncertainty in the NDData object instead.')
            raise ValueError(msg)

    @_create_call_docstring(iterative=False)
    def __call__(self, data, *, mask=None, error=None, init_params=None):
        # Reset state from previous runs
        self._reset_results()

        try:
            # Handle NDData input
            if isinstance(data, NDData):
                self._check_nddata_kwargs(mask=mask, error=error)
                data, mask, error = self._coerce_nddata(data)

            # Prepare all inputs for sources to be fit
            data, mask, error, init_params = self._prepare_fit_inputs(
                data, mask=mask, error=error, init_params=init_params,
            )

            # Handle the case where no sources were found
            if init_params is None:
                return None

            self.init_params = init_params

            # Fit sources defined in init_params
            fit_params = self._fit_sources(data, init_params, error=error,
                                           mask=mask)

            # Assemble the final results table
            # Note: _assemble_results_table handles _state cleanup
            self.results = self._assemble_results_table(
                init_params, fit_params, data.shape)

            # Reorder fit_info to match the results table rows, which
            # are sorted by the id column during table assembly
            row_map = {src_id: idx
                       for idx, src_id in enumerate(init_params['id'])}
            self.fit_info = [self.fit_info[row_map[src_id]]
                             for src_id in self.results['id']]

        except Exception:
            # Ensure state cleanup even if an exception occurs
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
    @deprecated('2.3.0', alternative='results',
                warning_type=PhotutilsDeprecationWarning)
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
            if (col_name == 'id'
                    or (col_name.endswith('_fit')
                        and col_name not in self._NON_PARAM_FIT_COLS)
                    or col_name.endswith('_err')):
                tbl[col_name] = self.results[col_name]

        return tbl

    @staticmethod
    def _iter_fit_param_cols(results_tbl):
        """
        Yield the 'id' column name plus the fitted model-parameter
        column names.
        """
        for col_name in results_tbl.colnames:
            if col_name == 'id' or (
                    col_name.endswith('_fit')
                    and col_name
                    not in PSFPhotometry._NON_PARAM_FIT_COLS):
                yield col_name

    @staticmethod
    def _finite_fit_mask(tbl):
        """
        Return a boolean mask of rows where every column is finite.
        """
        return np.all([np.isfinite(tbl[col])
                       for col in tbl.colnames], axis=0)

    @staticmethod
    def _results_to_init_params(results_tbl, *, remove_invalid=True,
                                reset_ids=True):
        """
        Convert PSF photometry results to initial parameters format.

        This method extracts fitted parameters (columns ending with
        '_fit') from the results table and renames them to initial
        parameter format (ending with '_init'). This output can be used
        as ``init_params`` for subsequent `PSFPhotometry` runs, allowing
        iterative refinement of source positions and fluxes.

        This is a static helper method to allow it to be called by
        `~photutils.psf.IterativePSFPhotometry`.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable` or None
            The table of fit results from a previous `PSFPhotometry` run.
            If None, returns None.

        remove_invalid : bool, optional
            If `True`, rows containing non-finite fitted values are
            removed. Default is `True`.

        reset_ids : bool, optional
            If `True`, the 'id' column is reset to sequential numbering
            starting from 1. If `False`, the 'id' values are preserved
            from ``results_tbl``. This option is ignored if
            ``remove_invalid`` is `False`. Default is `True`.

        Returns
        -------
        init_params_tbl : `~astropy.table.QTable` or None
            A table with columns renamed from '*_fit' to '*_init',
            suitable for use as ``init_params`` in a subsequent
            `PSFPhotometry` call. Returns None if ``results_tbl`` is
            None.

        Notes
        -----
        Only the 'id' column and columns with '_fit' suffix are included
        in the output. All other columns from the results table (e.g.,
        quality metrics, flags) are excluded.
        """
        # PSF photometry not yet run
        if results_tbl is None:
            return None

        tbl = QTable()
        for col_name in PSFPhotometry._iter_fit_param_cols(results_tbl):
            init_name = col_name.replace('_fit', '_init')
            tbl[init_name] = results_tbl[col_name]

        if remove_invalid:
            # Remove rows with any non-finite fitted values
            keep = PSFPhotometry._finite_fit_mask(tbl)
            tbl = tbl[keep]

            if reset_ids:
                tbl['id'] = np.arange(1, len(tbl) + 1)

        return tbl

    @staticmethod
    def _results_to_model_params(results_tbl, param_mapper, *,
                                 remove_invalid=True, reset_ids=True):
        """
        Convert PSF photometry results to PSF model parameters format.

        This method extracts fitted parameters (columns ending with
        '_fit') from the results table and renames them to match the
        PSF model's parameter names (e.g., 'x_fit' → 'x_0', 'flux_fit'
        → 'flux'). This output can be used to reconstruct fitted PSF
        models for visualization or further analysis.

        This is a static helper method to allow it to be called by
        `~photutils.psf.IterativePSFPhotometry`.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable` or None
            The table of fit results from a previous `PSFPhotometry`
            run. If None, returns None.

        param_mapper : `_PSFParameterMapper`
            The helper class that manages the mapping between aliases
            (e.g., 'x', 'flux') and PSF model parameter names (e.g.,
            'x_0', 'flux').

        remove_invalid : bool, optional
            If `True`, rows containing non-finite fitted values are
            removed. Default is `True`.

        reset_ids : bool, optional
            If `True`, the 'id' column is reset to sequential
            numbering starting from 1. If `False`, the 'id' values are
            preserved from ``results_tbl``. This option is ignored if
            ``remove_invalid`` is `False`. Default is `True`.

        Returns
        -------
        model_params_tbl : `~astropy.table.QTable` or None
            A table with columns renamed to match the PSF model's
            parameter names, suitable for model reconstruction. Returns
            None if ``results_tbl`` is None.

        Notes
        -----
        Only the 'id' column and columns with '_fit' suffix are included
        in the output. All other columns from the results table (e.g.,
        initial parameters, quality metrics, flags) are excluded.
        """
        # PSF photometry not yet run
        if results_tbl is None:
            return None

        tbl = QTable()
        for col_name in PSFPhotometry._iter_fit_param_cols(results_tbl):
            alias = col_name.replace('_fit', '')
            model_param_name = param_mapper.alias_to_model_param.get(
                alias, alias)
            tbl[model_param_name] = results_tbl[col_name]

        if remove_invalid:
            # Remove rows with any non-finite fitted values
            keep = PSFPhotometry._finite_fit_mask(tbl)
            tbl = tbl[keep]

            if reset_ids:
                tbl['id'] = np.arange(1, len(tbl) + 1)

        return tbl

    def results_to_init_params(self, *, remove_invalid=True, reset_ids=True):
        """
        Create a table of initial parameters from the fitted results.

        The table columns are named according to those expected for the
        initial parameters table. It can be used as the ``init_params``
        for subsequent `PSFPhotometry` fits.

        Parameters
        ----------
        remove_invalid : bool, optional
            If `True`, rows that contain non-finite fitted values are
            removed.

        reset_ids : bool, optional
            If `True`, the 'id' column will be reset to a sequential
            numbering starting from 1. If `False`, the 'id' column will
            remain unchanged from the results table. This option is
            ignored if ``remove_invalid`` is `False`.

        Returns
        -------
        init_params_tbl : `~astropy.table.QTable` or `None`
            The table of initial parameters, or `None` if the
            instance has not yet been run.
        """
        return self._results_to_init_params(self.results,
                                            remove_invalid=remove_invalid,
                                            reset_ids=reset_ids)

    def results_to_model_params(self, *, remove_invalid=True, reset_ids=True):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to the PSF model parameter
        names. It can also be used to reconstruct the fitted PSF models
        for visualization or further analysis.

        Parameters
        ----------
        remove_invalid : bool, optional
            If `True`, rows that contain non-finite fitted values are
            removed.

        reset_ids : bool, optional
            If `True`, the 'id' column will be reset to a sequential
            numbering starting from 1. If `False`, the 'id' column will
            remain unchanged from the results table. This option is
            ignored if ``remove_invalid`` is `False`.

        Returns
        -------
        model_params_tbl : `~astropy.table.QTable` or `None`
            The table of model parameters, or `None` if the instance
            has not yet been run.
        """
        return self._results_to_model_params(self.results,
                                             self._param_mapper,
                                             remove_invalid=remove_invalid,
                                             reset_ids=reset_ids)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def decode_flags(self, return_bit_values=False):
        """
        Decode the PSF photometry flags from the results table.

        This is a convenience method that calls
        `~photutils.psf.decode_psf_flags` with the 'flags' column
        from the results table.

        Parameters
        ----------
        return_bit_values : bool, optional
            If `True`, return the decoded bit flags (integers) instead
            of the flag descriptions (strings). Default is `False`.

        Returns
        -------
        decoded : dict
            A dictionary mapping each source id from the results table
            to the list of its active flag names (or bit values). If no
            flags are set for a source, its value is an empty list. The
            entries follow the results-table order.

            .. versionchanged:: 3.1
                The result is now a dictionary keyed by source id.
                Previously, a positional list of lists was returned.

        Raises
        ------
        ValueError
            If no results are available. Please run the PSFPhotometry
            instance first.

        See Also
        --------
        photutils.psf.decode_psf_flags

        Examples
        --------
        Decode flags from PSF photometry results:

        >>> import numpy as np
        >>> from astropy.table import Table
        >>> from photutils.psf import CircularGaussianPRF, PSFPhotometry
        >>> yy, xx = np.mgrid[:21, :21]
        >>> psf_model = CircularGaussianPRF(flux=1, x_0=10, y_0=10, fwhm=2)
        >>> # Create a source with negative flux to trigger a flag
        >>> m1 = CircularGaussianPRF(flux=100, x_0=10, y_0=10, fwhm=2)
        >>> m2 = CircularGaussianPRF(flux=-50, x_0=5, y_0=5, fwhm=2)
        >>> data = m1(xx, yy) + m2(xx, yy)
        >>> init_params = Table({'x': [10, 5], 'y': [10, 5],
        ...                      'flux': [100, 100]})
        >>> photometry = PSFPhotometry(psf_model, (3, 3))
        >>> results = photometry(data, init_params=init_params)
        >>> for source_id, flags in photometry.decode_flags().items():
        ...     print(f'Source {source_id}: {flags}')  # doctest: +SKIP
        Source 1: []
        Source 2: ['non_positive_flux']
        """
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        decoded = decode_psf_flags(self.results['flags'],
                                   return_bit_values=return_bit_values)
        return {int(id_): flags
                for id_, flags in zip(self.results['id'], decoded,
                                      strict=True)}

    def _get_model_image_params(self):
        # Convert fitted parameters to model parameter names without
        # filtering, so the row indices align with self.results
        model_params = self.results_to_model_params(remove_invalid=False)

        # Filter out invalid sources (those with NaN fitted values)
        keep = np.all([np.isfinite(model_params[col])
                       for col in model_params.colnames], axis=0)
        model_params = model_params[keep]

        # Extract local_bkg for the same valid sources
        local_bkg = self.results['local_bkg'][keep]

        return model_params, local_bkg

    @deprecated_renamed_argument('include_localbkg', 'include_local_bkg',
                                 '3.0', until='4.0')
    @_make_model_image_docstring
    def make_model_image(self, shape, *, psf_shape=None,
                         include_local_bkg=False):
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        model_params, local_bkg = self._get_model_image_params()
        maker = _ModelImageMaker(self.psf_model, model_params,
                                 local_bkg=local_bkg,
                                 progress_bar=self.progress_bar)
        return maker.make_model_image(shape, psf_shape=psf_shape,
                                      include_local_bkg=include_local_bkg)

    @deprecated_renamed_argument('include_localbkg', 'include_local_bkg',
                                 '3.0', until='4.0')
    @_make_residual_image_docstring
    def make_residual_image(self, data, *, psf_shape=None,
                            include_local_bkg=False):
        if self.results is None:
            msg = ('No results available. Please run the PSFPhotometry '
                   'instance first.')
            raise ValueError(msg)

        model_params, local_bkg = self._get_model_image_params()
        maker = _ModelImageMaker(self.psf_model, model_params,
                                 local_bkg=local_bkg,
                                 progress_bar=self.progress_bar)
        return maker.make_residual_image(data, psf_shape=psf_shape,
                                         include_local_bkg=include_local_bkg)
