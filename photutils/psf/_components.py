# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides private implementation classes for PSF photometry
components.

This is a private module. The classes within are implementation details
of the PSFPhotometry class and are not intended for direct public use.
"""

import contextlib
import warnings

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NoOverlapError
from astropy.table import QTable, Table, hstack, join
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.utils._misc import _get_meta
from photutils.utils.cutouts import _overlap_slices as overlap_slices

from .flags import PSF_FLAGS

__all__ = ['PSFDataProcessor', 'PSFFitter', 'PSFResultsAssembler']


class PSFDataProcessor:
    """
    Helper class to handle data validation, preprocessing, and cutout
    extraction for PSF photometry.

    This class encapsulates all data-related operations including
    validation, source finding, initial parameter estimation, and cutout
    extraction.

    Parameters
    ----------
    param_mapper : _PSFParameterMapper
        Parameter mapper for handling column name mappings and model
        parameters.

    fit_shape : tuple of int
        The shape of the PSF fitting region as a (ny, nx) tuple.

    finder : object, optional
        Source finder instance for detecting sources if ``init_params``
        is not provided. Must have a ``__call__`` method that accepts
        data and mask arrays and returns a table of detected sources.

    aperture_radius : float, optional
        Radius in pixels of circular apertures for initial flux
        estimation when flux values are not provided in ``init_params``.

    localbkg_estimator : object, optional
        Local background estimator for determining background levels
        around sources. Must have a ``__call__`` method.
    """

    def __init__(self, param_mapper, fit_shape, finder=None,
                 aperture_radius=None, localbkg_estimator=None):
        self.param_mapper = param_mapper
        self.fit_shape = fit_shape
        self.finder = finder
        self.aperture_radius = aperture_radius
        self.localbkg_estimator = localbkg_estimator
        self.data_unit = None
        self.finder_results = None

        # Cache for offset grids
        self._cached_offsets = None
        self._cache_key = None

    def validate_array(self, array, name, data_shape=None):
        """
        Validate input arrays (data, error, mask).

        Parameters
        ----------
        array : array-like or None
            Input array to validate. Can be None for optional arrays.

        name : str
            Name of the array for error messages (e.g., 'data', 'error',
            'mask').

        data_shape : tuple of int, optional
            Expected shape of the array. If provided, validates that
            the array shape matches this shape.

        Returns
        -------
        array : `~numpy.ndarray` or None
            Validated 2D array or None if input was None.

        Raises
        ------
        ValueError
            If the array is not 2D or if the shape doesn't match
            ``data_shape`` when provided.
        """
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

    def normalize_init_units(self, init_params, colname):
        """
        Normalize the units of a column in the input init_params table
        to match the input data units.

        Parameters
        ----------
        init_params : `~astropy.table.Table`
            Table containing initial parameters for PSF fitting.

        colname : str
            Name of the column to normalize units for.

        Returns
        -------
        init_params : `~astropy.table.Table`
            The input table with normalized units for the specified
            column.

        Raises
        ------
        ValueError
            If there are unit compatibility issues between the column
            and the data units.
        """
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

    def validate_init_params(self, init_params):
        """
        Validate the input init_params table and rename columns to
        expected format.

        Parameters
        ----------
        init_params : `~astropy.table.Table` or None
            Table containing initial parameters for PSF fitting. Must
            contain columns for x and y positions. May contain flux
            and local_bkg columns.

        Returns
        -------
        init_params : `~astropy.table.Table` or None
            Validated table with renamed columns matching expected
            format, or None if input was None.

        Raises
        ------
        TypeError
            If ``init_params`` is not an astropy Table.

        ValueError
            If required position columns are missing or if local_bkg
            contains non-finite values.
        """
        if init_params is None:
            return init_params

        if not isinstance(init_params, Table):
            msg = 'init_params must be an astropy Table'
            raise TypeError(msg)

        # Copy to preserve the input init_params
        init_params = self.param_mapper.rename_table_columns(
            init_params.copy())

        if (self.param_mapper.init_colnames['x'] not in init_params.colnames
                or self.param_mapper.init_colnames['y'] not in
                init_params.colnames):
            msg = ('init_params must contain valid column names for the '
                   'x and y source positions')
            raise ValueError(msg)

        flux_col = self.param_mapper.init_colnames['flux']
        if flux_col in init_params.colnames:
            init_params = self.normalize_init_units(init_params, flux_col)

        if 'local_bkg' in init_params.colnames:
            if not np.all(np.isfinite(init_params['local_bkg'])):
                msg = ('init_params local_bkg column contains non-finite '
                       'values')
                raise ValueError(msg)
            init_params = self.normalize_init_units(init_params, 'local_bkg')

        return init_params

    def get_aper_fluxes(self, data, mask, init_params):
        """
        Estimate aperture fluxes at the initial (x, y) positions.

        Parameters
        ----------
        data : `~numpy.ndarray`
            2D image data array.

        mask : `~numpy.ndarray` or None
            2D boolean mask array with the same shape as ``data``, where
            ``True`` indicates masked pixels.

        init_params : `~astropy.table.Table`
            Table containing initial source positions with validated
            column names.

        Returns
        -------
        flux : `~numpy.ndarray`
            Array of aperture flux estimates for each source.
        """
        x_pos = init_params[self.param_mapper.init_colnames['x']]
        y_pos = init_params[self.param_mapper.init_colnames['y']]
        apertures = CircularAperture(zip(x_pos, y_pos, strict=True),
                                     r=self.aperture_radius)
        flux, _ = apertures.do_photometry(data, mask=mask)
        return flux

    def find_sources_if_needed(self, data, mask, init_params):
        """
        Find sources using the finder if initial positions are not
        provided.

        Parameters
        ----------
        data : `~numpy.ndarray`
            2D image data array.

        mask : `~numpy.ndarray` or None
            2D boolean mask array with the same shape as ``data``.

        init_params : `~astropy.table.Table` or None
            Table containing initial source parameters. If provided,
            an 'id' column is added if missing. If None, sources are
            found using the finder.

        Returns
        -------
        sources : `~astropy.table.Table` or None
            Table containing source information with required columns
            for PSF fitting, or None if no sources were found.

        Raises
        ------
        ValueError
            If ``init_params`` is None and no finder was provided.
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

        return self._convert_finder_to_init(sources)

    def _convert_finder_to_init(self, sources):
        """
        Convert the output table of the finder to a table with initial
        (x, y) position column names.

        Parameters
        ----------
        sources : `~astropy.table.Table`
            Table returned by the source finder containing detected
            source positions.

        Returns
        -------
        init_params : `~astropy.table.QTable`
            Table with standardized column names suitable for PSF
            fitting initialization.

        Raises
        ------
        ValueError
            If the sources table does not contain valid x and y
            coordinate columns.
        """
        # Find the first valid column names for x and y
        x_name_found = self.param_mapper.find_column(sources, 'x')
        y_name_found = self.param_mapper.find_column(sources, 'y')
        if x_name_found is None or y_name_found is None:
            msg = ("The table returned by the 'finder' must contain columns "
                   'for x and y coordinates. Valid column names are: '
                   f"x: {self.param_mapper.VALID_INIT_COLNAMES['x']}, "
                   f"y: {self.param_mapper.VALID_INIT_COLNAMES['y']}")
            raise ValueError(msg)

        # Create a new table with only the needed columns
        init_params = QTable()
        init_params['id'] = np.arange(len(sources)) + 1
        x_col = self.param_mapper.init_colnames['x']
        y_col = self.param_mapper.init_colnames['y']
        init_params[x_col] = sources[x_name_found]
        init_params[y_col] = sources[y_name_found]

        return init_params

    def estimate_flux_and_bkg_if_needed(self, data, mask, init_params):
        """
        Estimate initial fluxes and backgrounds if not provided.

        Parameters
        ----------
        data : `~numpy.ndarray`
            2D image data array.

        mask : `~numpy.ndarray` or None
            2D boolean mask array with the same shape as ``data``.

        init_params : `~astropy.table.Table`
            Table containing initial source parameters. Will be modified
            in-place to add flux and/or local_bkg columns if missing.

        Returns
        -------
        init_params : `~astropy.table.Table`
            The input table with added flux and local_bkg columns if
            they were missing.

        Raises
        ------
        ValueError
            If aperture_radius is None when flux estimation is needed.
        """
        x_col = self.param_mapper.init_colnames['x']
        y_col = self.param_mapper.init_colnames['y']
        flux_col = self.param_mapper.init_colnames['flux']

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
            # Check for aperture_radius before attempting to use it
            if self.aperture_radius is None:
                msg = ('aperture_radius must be defined if a flux column is '
                       'not in init_params')
                raise ValueError(msg)

            flux = self.get_aper_fluxes(data, mask, init_params)
            if self.data_unit is not None:
                flux <<= self.data_unit
            flux -= init_params['local_bkg']
            init_params[flux_col] = flux

        return init_params

    def get_fit_offsets(self):
        """
        Return cached (y_offsets, x_offsets) arrays for fit_shape.

        Returns
        -------
        offsets : tuple of `~numpy.ndarray`
            Tuple containing (y_offsets, x_offsets) arrays with shape
            ``fit_shape``, where each array contains coordinate offsets
            from the origin.
        """
        ny, nx = self.fit_shape
        cache_key = (ny, nx)

        # Optimized cache management
        if (self._cached_offsets is None or self._cache_key != cache_key):
            # Create new cache with validated shape
            self._cached_offsets = np.mgrid[0:ny, 0:nx]
            self._cache_key = cache_key

        return self._cached_offsets

    def should_skip_source(self, row, data_shape):
        """
        Quick validation to skip obviously invalid sources early.

        Parameters
        ----------
        row : `~astropy.table.Row`
            Row from the sources table containing source parameters.

        data_shape : tuple of int
            Shape of the input data array as (ny, nx).

        Returns
        -------
        should_skip : bool
            True if the source should be skipped, False otherwise.

        reason : str or None
            Reason for skipping ('invalid_position', 'no_overlap')
            or None if not skipping.
        """
        x_cen = row[self.param_mapper.init_colnames['x']]
        y_cen = row[self.param_mapper.init_colnames['y']]

        # check for non-finite positions
        if not (np.isfinite(x_cen) and np.isfinite(y_cen)):
            return True, 'invalid_position'

        # source that are clearly beyond any possible overlap
        half_fit = max(self.fit_shape) // 2
        clear_margin = half_fit + 1  # a bit beyond the fit region
        if (x_cen < -clear_margin or y_cen < -clear_margin
                or x_cen >= data_shape[1] + clear_margin
                or y_cen >= data_shape[0] + clear_margin):
            return True, 'no_overlap'

        return False, None

    def get_source_cutout_data(self, row, data, mask, y_offsets, x_offsets):
        """
        Extract per-source pixel indices and cutout data.

        Parameters
        ----------
        row : `~astropy.table.Row`
            Row from the sources table containing source parameters
            including position and local background.

        data : `~numpy.ndarray`
            2D image data array.

        mask : `~numpy.ndarray` or None
            2D boolean mask array with the same shape as ``data``.

        y_offsets, x_offsets : `~numpy.ndarray`
            2D array of y- and x-coordinate offsets from
            ``get_fit_offsets()``.

        Returns
        -------
        cutout_data : dict
            Dictionary containing cutout information:
            - 'valid' : bool, whether the cutout is valid
            - 'reason' : str or None, reason if invalid
            - 'xx' : array, x pixel coordinates (flattened)
            - 'yy' : array, y pixel coordinates (flattened)
            - 'cutout' : array, data values (background-subtracted)
            - 'npix' : int, number of pixels in cutout
            - 'cen_index' : int or nan, index of center pixel
        """
        x_cen = row[self.param_mapper.init_colnames['x']]
        y_cen = row[self.param_mapper.init_colnames['y']]

        try:
            slc_lg, _ = overlap_slices(data.shape, self.fit_shape,
                                       (y_cen, x_cen), mode='trim')
        except NoOverlapError:
            return {'valid': False,
                    'reason': 'no_overlap',
                    'xx': None,
                    'yy': None,
                    'cutout': None,
                    'npix': 0,
                    'cen_index': np.nan,
                    }

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
                return {'valid': False,
                        'reason': 'fully_masked',
                        'xx': None,
                        'yy': None,
                        'cutout': None,
                        'npix': 0,
                        'cen_index': np.nan,
                        }

            yy_flat = yy[inv_mask]
            xx_flat = xx[inv_mask]
        else:
            yy_flat = yy.ravel()
            xx_flat = xx.ravel()

        cutout = data[yy_flat, xx_flat]

        # Local background subtraction (local_bkg = 0 if not provided)
        local_bkg = row['local_bkg']
        if np.any(local_bkg != 0):
            if isinstance(local_bkg, u.Quantity):
                local_bkg = local_bkg.value
            cutout -= local_bkg

        # Center pixel index (before trimming)
        x_cen_idx = np.ceil(x_cen - 0.5).astype(int)
        y_cen_idx = np.ceil(y_cen - 0.5).astype(int)
        cen_match = np.where((xx_flat == x_cen_idx)
                             & (yy_flat == y_cen_idx))[0]
        cen_index = cen_match[0] if len(cen_match) > 0 else np.nan

        return {'valid': True,
                'reason': None,
                'xx': xx_flat,
                'yy': yy_flat,
                'cutout': cutout,
                'npix': len(xx_flat),
                'cen_index': cen_index,
                }


class PSFFitter:
    """
    Helper class to handle PSF model fitting operations.

    This class encapsulates all fitting-related operations including
    model creation, fitting execution, and parameter extraction.

    Parameters
    ----------
    psf_model : `~astropy.modeling.Model`
        PSF model to be fit to sources. This model will be copied for
        each source and fitted parameters will be set.

    param_mapper : _PSFParameterMapper
        Parameter mapper for handling column name mappings and model
        parameters.

    fitter : `~astropy.modeling.fitting.Fitter`, optional
        Astropy fitter instance to use for PSF fitting. If None,
        defaults to `~astropy.modeling.fitting.TRFLSQFitter`.

    fitter_maxiters : int, optional
        Maximum number of fitting iterations. Default is 100.

    xy_bounds : tuple of float or None, optional
        Bounds for x and y position parameters as (x_bound, y_bound).
        If provided, fitting positions will be constrained to within
        these bounds of the initial values.

    group_warning_threshold : int, optional
        Threshold for issuing warnings about large group sizes.
        Default is 25.
    """

    def __init__(self, psf_model, param_mapper, fitter=None,
                 fitter_maxiters=100, xy_bounds=None,
                 group_warning_threshold=25):
        self.psf_model = psf_model
        self.param_mapper = param_mapper
        self.fitter = fitter if fitter is not None else TRFLSQFitter()
        self.fitter_maxiters = fitter_maxiters
        self.xy_bounds = xy_bounds
        self.group_warning_threshold = group_warning_threshold

    def make_psf_model(self, sources):
        """
        Make a PSF model to fit a single source or several sources
        within a group.

        Parameters
        ----------
        sources : `~astropy.table.Table`
            Table containing source parameters including positions,
            flux estimates, and source IDs. For multiple sources,
            creates a compound model.

        Returns
        -------
        psf_model : `~astropy.modeling.Model`
            PSF model (single or compound) with initial parameters
            set from the sources table. For multiple sources, returns
            a compound model with each source as a submodel.
        """
        alias_map = self.param_mapper.alias_to_model_param
        init_colnames = self.param_mapper.init_colnames

        for index, source in enumerate(sources):
            model = self.psf_model.copy()
            for alias, col_name in init_colnames.items():
                model_param = alias_map[alias]
                value = source[col_name]
                if isinstance(value, u.Quantity):
                    value = value.value  # PSF model cannot be fit with units
                setattr(model, model_param, value)
            model.name = source['id']

            if self.xy_bounds is not None:
                if self.xy_bounds[0] is not None:
                    x_param_name = alias_map['x']
                    x_param = getattr(model, x_param_name)
                    x_param.bounds = (x_param.value - self.xy_bounds[0],
                                      x_param.value + self.xy_bounds[0])
                if self.xy_bounds[1] is not None:
                    y_param_name = alias_map['y']
                    y_param = getattr(model, y_param_name)
                    y_param.bounds = (y_param.value - self.xy_bounds[1],
                                      y_param.value + self.xy_bounds[1])

            if index == 0:
                psf_model = model
            else:
                psf_model += model

        return psf_model

    def run_fitter(self, psf_model, xi, yi, cutout, error):
        """
        Fit the PSF model to the input cutout data.

        Parameters
        ----------
        psf_model : `~astropy.modeling.Model`
            PSF model to fit, typically created by ``make_psf_model``.

        xi, yi : `~numpy.ndarray`
            1D array of x and y pixel coordinates for the cutout.

        cutout : `~numpy.ndarray`
            1D array of data values corresponding to the pixel
            coordinates.

        error : `~numpy.ndarray` or None
            2D error array for computing fit weights. If provided,
            weights are computed as 1/error for the cutout pixels.

        Returns
        -------
        fit_model : `~astropy.modeling.Model`
            Fitted PSF model with optimized parameters.

        fit_info : dict
            Dictionary containing fit information including parameter
            covariance, convergence status, and error messages.

        Raises
        ------
        ValueError
            If error array contains non-positive or non-finite values.
        """
        if self.fitter_maxiters is not None:
            kwargs = {'maxiter': self.fitter_maxiters}
        else:
            kwargs = {}

        weights = None
        if error is not None:
            # Extract cutout weights from full error array. Weights are
            # (1 / error), yielding residuals (objective function) of
            # (data - model) / error. The fitter minimizes the squared
            # residuals. If errors are input, the residuals returned by
            # the fitter are scaled by the errors, i.e., the objective
            # function is equivalent to chi2. If errors are not input,
            # the residuals are just (data - model), and the objective
            # function is a sum-of-squares. Note that the residuals
            # returned by astropy fitters are reversed in sign from the
            # definition here (model - data).
            err = error[yi, xi]
            if np.any(err <= 0) or np.any(~np.isfinite(err)):
                msg = ('Error array contains non-positive or non-finite '
                       'values. Cannot compute fit weights.')
                raise ValueError(msg)
            weights = 1.0 / error[yi, xi]

        # keep fit-info entries (but exclude residual vectors)
        fit_info_keys = ('param_cov', 'ierr', 'message', 'status')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            fit_model = self.fitter(psf_model, xi, yi, cutout,
                                    weights=weights, **kwargs)

            # clear any model cache, if supported by the model
            with contextlib.suppress(AttributeError):
                fit_model.clear_cache()

            fit_info = {key: self.fitter.fit_info.get(key) for key in
                        fit_info_keys if self.fitter.fit_info.get(key)
                        is not None}

        return fit_model, fit_info

    @staticmethod
    def split_compound_model(model, chunk_size):
        """
        Split a compound model into its constituent sub-models.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            Compound model containing multiple submodels.

        chunk_size : int
            Number of submodels per chunk.

        Yields
        ------
        chunk : `~astropy.modeling.Model`
            Chunk of the compound model containing ``chunk_size``
            submodels (or fewer for the last chunk).
        """
        for i in range(0, model.n_submodels, chunk_size):
            yield model[i:i + chunk_size]

    def extract_source_covariances(self, group_cov, num_sources, nfitparam):
        """
        Extract individual source covariance matrices from group
        covariance.

        This method assumes that the group covariance matrix is
        block-diagonal, with each block corresponding to a source.
        The extracted source covariance matrices do not include
        the covariances between different sources.

        Parameters
        ----------
        group_cov : `~numpy.ndarray`
            2D covariance matrix for the entire group of sources.

        num_sources : int
            Number of sources in the group.

        nfitparam : int
            Number of fitted parameters per source.

        Returns
        -------
        source_covs : list of `~numpy.ndarray`
            List of 2D covariance matrices, one for each source in
            the group.
        """
        source_covs = []
        for i in range(num_sources):
            start = i * nfitparam
            end = (i + 1) * nfitparam
            source_cov = group_cov[start:end, start:end]
            source_covs.append(source_cov)

        return source_covs


class PSFResultsAssembler:
    """
    Helper class to handles results table assembly and quality metrics
    calculation.

    This class encapsulates all operations related to assembling final
    results tables, calculating quality metrics, and generating flags.

    Parameters
    ----------
    param_mapper : _PSFParameterMapper
        Parameter mapper for handling column name mappings and model
        parameters.

    fit_shape : tuple of int
        The shape of the PSF fitting region as a (ny, nx) tuple.

    xy_bounds : tuple of float or None, optional
        Bounds for x and y position parameters as (x_bound, y_bound).
        Used for flag calculations to detect sources near boundaries.
    """

    def __init__(self, param_mapper, fit_shape, xy_bounds=None):
        self.param_mapper = param_mapper
        self.fit_shape = fit_shape
        self.xy_bounds = xy_bounds

    def get_fit_error_indices(self, fit_info):
        """
        Get the indices of fits that did not converge.

        Parameters
        ----------
        fit_info : list of dict
            List of fit information dictionaries from the fitter,
            containing 'ierr' or 'status' keys for convergence info.

        Returns
        -------
        bad_indices : `~numpy.ndarray`
            Array of integer indices for fits that did not converge.
        """
        # Same "good" flags for both leastsq and least_squares
        converged_status = {1, 2, 3, 4}

        n_sources = len(fit_info)
        ierr_vals = np.full(n_sources, None, dtype=object)
        status_vals = np.full(n_sources, None, dtype=object)

        # Extract all ierr and status values
        for idx, info in enumerate(fit_info):
            ierr_vals[idx] = info.get('ierr')
            status_vals[idx] = info.get('status')

        # Create masks for non-None values
        ierr_valid = ierr_vals != None  # noqa: E711
        status_valid = status_vals != None  # noqa: E711

        # Check convergence status for valid values
        converged_status_list = list(converged_status)
        ierr_bad = np.zeros(n_sources, dtype=bool)
        status_bad = np.zeros(n_sources, dtype=bool)

        for i in range(n_sources):
            if ierr_valid[i]:
                ierr_bad[i] = ierr_vals[i] not in converged_status_list
            if status_valid[i]:
                status_bad[i] = status_vals[i] not in converged_status_list

        # Combine conditions
        bad_mask = (ierr_valid & ierr_bad) | (status_valid & status_bad)
        bad_indices = np.where(bad_mask)[0]

        return bad_indices.astype(int)

    def param_errors_to_table(self, fit_param_errs, data_unit):
        """
        Convert the fitter's parameter errors to an astropy Table.

        Parameters
        ----------
        fit_param_errs : `~numpy.ndarray`
            2D array of parameter errors with shape (n_sources, n_params).

        data_unit : `~astropy.units.Unit` or None
            Unit of the input data, used to apply appropriate units
            to flux error columns.

        Returns
        -------
        table : `~astropy.table.QTable`
            Table containing error columns for fitted parameters,
            with NaN values for non-fitted (fixed) parameters.
        """
        table = QTable()

        # create error columns for models parameters that were fit
        mapper = self.param_mapper
        fitted_params = mapper.fitted_param_names
        model_param_to_alias = mapper.model_param_to_alias
        err_colnames = mapper.err_colnames

        fitted_aliases = [model_param_to_alias[param]
                          for param in fitted_params]
        fitted_err_cols = [err_colnames[alias] for alias in fitted_aliases]

        table_data = {err_col: fit_param_errs[:, i]
                      for i, err_col in enumerate(fitted_err_cols)}
        table = QTable(table_data)

        # ensure columns for non-fitted (fixed) params exist
        all_err_cols = list(err_colnames.values())
        for err_col in all_err_cols:
            if err_col not in table.colnames:
                table[err_col] = np.nan

        # apply data_unit to flux_err column
        if data_unit is not None:
            flux_err_col = err_colnames['flux']
            table[flux_err_col] <<= data_unit

        return table[all_err_cols]

    def create_fit_results(self, fit_model_all_params, fit_param_errs,
                           valid_mask, data_unit):
        """
        Create the table of fitted parameter values and errors.

        Parameters
        ----------
        fit_model_all_params : `~astropy.table.Table`
            Table containing fitted model parameters for all sources.

        fit_param_errs : `~numpy.ndarray`
            2D array of parameter errors with shape (n_sources, n_params).

        valid_mask : `~numpy.ndarray` or None
            Boolean array indicating which sources had valid fits.

        data_unit : `~astropy.units.Unit` or None
            Unit of the input data for applying to flux-related columns.

        Returns
        -------
        fit_table : `~astropy.table.QTable`
            Table containing fitted parameter values and their errors,
            with NaN values for invalid sources.
        """
        mapper = self.param_mapper
        alias_to_model = mapper.alias_to_model_param
        model_param_to_alias = mapper.model_param_to_alias
        fit_colnames = mapper.fit_colnames
        err_colnames = mapper.err_colnames

        col_names = ['id', *list(alias_to_model.values())]
        fit_params = fit_model_all_params[col_names]

        # Rename model parameter columns to *_fit
        for col_name in list(fit_params.colnames):
            if col_name == 'id':
                continue
            alias = model_param_to_alias[col_name]
            fit_params.rename_column(col_name, fit_colnames[alias])

        param_errs = self.param_errors_to_table(fit_param_errs, data_unit)
        fit_table = hstack([fit_params, param_errs])

        # Sort columns to match the expected order
        col_order = ['id', *list(fit_colnames.values()),
                     *list(err_colnames.values())]
        fit_table = fit_table[col_order]

        # Overwrite fit and error columns with NaN for invalid sources
        if valid_mask is not None:
            invalid = ~np.array(valid_mask, dtype=bool)
            for col_name in fit_table.colnames:
                if col_name == 'id':
                    continue
                col = fit_table[col_name]
                unit = getattr(col, 'unit', None)
                if unit is not None:
                    col[invalid] = (np.nan * unit)
                else:
                    col[invalid] = np.nan

        return fit_table

    def calc_fit_metrics(self, results_tbl, sum_abs_residuals, cen_residuals,
                         reduced_chi2):
        """
        Calculate fit quality metrics qfit, cfit, and reduced_chi2.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable`
            Results table containing fitted flux values.

        sum_abs_residuals : array-like
            Array of sum of absolute residuals for each source.

        cen_residuals : array-like
            Array of central pixel residuals for each source.

        reduced_chi2 : array-like
            Array of reduced chi-squared values for each source.

        Returns
        -------
        qfit : `~numpy.ndarray`
            Array of qfit quality metrics (sum of absolute residuals
            divided by flux).

        cfit : `~numpy.ndarray`
            Array of cfit quality metrics (central pixel residual
            divided by flux).

        reduced_chi2 : `~numpy.ndarray`
            Array of reduced chi-squared values.
        """
        flux_col = self.param_mapper.fit_colnames['flux']
        flux_vals = results_tbl[flux_col]
        if isinstance(flux_vals, u.Quantity):
            flux_vals = flux_vals.value

        nsrc = len(sum_abs_residuals)
        qfit = np.full(nsrc, np.nan, dtype=float)
        cfit = np.full(nsrc, np.nan, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            # create masks for valid data
            valid_sa = np.isfinite(sum_abs_residuals)
            valid_cr = np.isfinite(cen_residuals)
            nonzero_flux = (flux_vals != 0)

            # qfit: sum of absolute residuals divided by flux
            qfit_mask = valid_sa & nonzero_flux
            qfit[qfit_mask] = (sum_abs_residuals[qfit_mask]
                               / flux_vals[qfit_mask])

            # cfit: central pixel residual divided by flux
            cfit_mask = valid_cr & nonzero_flux
            cfit[cfit_mask] = (cen_residuals[cfit_mask]
                               / flux_vals[cfit_mask])

        return qfit, cfit, reduced_chi2

    def define_flags(self, results_tbl, shape, fit_error_indices, fit_info,
                     fitted_models_table, valid_mask, invalid_reasons):
        """
        Define per-source bitwise flags summarizing fit conditions.

        Parameters
        ----------
        results_tbl : `~astropy.table.QTable`
            Results table containing fitted parameters and metrics.

        shape : tuple of int
            Shape of the input data array as (ny, nx).

        fit_error_indices : `~numpy.ndarray` or None
            Array of indices for sources with convergence issues.

        fit_info : list of dict
            List of fit information dictionaries.

        fitted_models_table : `~astropy.table.Table`
            Table containing fitted model parameters with bounds info.

        valid_mask : `~numpy.ndarray` or None
            Boolean array indicating valid sources.

        invalid_reasons : list or None
            List of reasons why sources were invalid.

        Returns
        -------
        flags : `~numpy.ndarray`
            Array of integer flags where each bit indicates a specific
            condition:
            - 1: npixfit smaller than full fit_shape region
            - 2: fitted position outside input image bounds
            - 4: non-positive flux
            - 8: possible non-convergence
            - 16: missing parameter covariance
            - 32: near a positional bound
            - 64: no overlap with data
            - 128: fully masked source
            - 256: too few pixels for fitting
        """
        flags = np.zeros(len(results_tbl), dtype=int)
        x_col = self.param_mapper.fit_colnames['x']
        y_col = self.param_mapper.fit_colnames['y']
        flux_col = self.param_mapper.fit_colnames['flux']

        # Flag=1: npixfit smaller than full fit_shape region
        flag1_mask = results_tbl['npixfit'] < np.prod(self.fit_shape)
        flags[flag1_mask] |= PSF_FLAGS.NPIXFIT_PARTIAL

        # Flag=2: fitted position outside input image bounds
        ny, nx = shape
        x_fit = results_tbl[x_col]
        y_fit = results_tbl[y_col]
        flag2_mask = ((x_fit < -0.5) | (y_fit < -0.5) | (x_fit > nx - 0.5)
                      | (y_fit > ny - 0.5))
        flags[flag2_mask] |= PSF_FLAGS.OUTSIDE_BOUNDS

        # Flag=4: non-positive flux
        flag4_mask = results_tbl[flux_col] <= 0
        flags[flag4_mask] |= PSF_FLAGS.NEGATIVE_FLUX

        # Flag=8: possible non-convergence
        if fit_error_indices is not None:
            flags[fit_error_indices] |= PSF_FLAGS.NO_CONVERGENCE

        # Flag=16: missing parameter covariance
        missing_cov_mask = np.array(['param_cov' not in info
                                     for info in fit_info])
        flags[missing_cov_mask] |= PSF_FLAGS.NO_COVARIANCE

        # Flag=32: near a positional bound
        bound_tol = 0.01
        if self.xy_bounds is not None:
            alias_to_model = self.param_mapper.alias_to_model_param
            x_param = alias_to_model['x']
            y_param = alias_to_model['y']

            # Extract all parameter values and bounds into arrays
            x_vals = np.array([row[x_param] for row in fitted_models_table])
            y_vals = np.array([row[y_param] for row in fitted_models_table])

            # Create masks for valid sources and finite positions
            finite_mask = (np.isfinite(x_vals) & np.isfinite(y_vals)
                           & valid_mask)

            # Check bounds for valid sources
            for index in np.where(finite_mask)[0]:
                row = fitted_models_table[index]
                for param in (x_param, y_param):
                    bnds = row[f'{param}_bounds']
                    bounds = np.array([i for i in bnds if i is not None])
                    if bounds.size == 0:
                        continue
                    if np.any(np.abs(bounds - row[param]) <= bound_tol):
                        flags[index] |= PSF_FLAGS.NEAR_BOUND
                        break

        # Flag=64, 128, 256: invalid source reasons
        if invalid_reasons is not None:
            reasons = np.array(invalid_reasons, dtype=object)
            flags[reasons == 'no_overlap'] |= PSF_FLAGS.NO_OVERLAP
            flags[reasons == 'fully_masked'] |= PSF_FLAGS.FULLY_MASKED
            flags[reasons == 'too_few_pixels'] |= PSF_FLAGS.TOO_FEW_PIXELS

        return flags

    def assemble_results_table(self, init_params, fit_params, data_shape,
                               state, calc_fit_metrics_func, define_flags_func,
                               class_name, metadata_attrs):
        """
        Assemble the final results table.

        The final results table is built by merging the input
        ``init_params`` table with the ``fit_params`` table. Additional
        columns are added for ``npixfit``, ``group_size``, ``qfit``,
        ``cfit``, and ``flags``.

        This method also cleans up the state dictionary as data is
        consumed to reduce memory usage.

        Parameters
        ----------
        init_params : `~astropy.table.QTable`
            Initial parameter guesses for sources.

        fit_params : `~astropy.table.QTable`
            Fitted parameters with uncertainties.

        data_shape : tuple of int
            Shape of the input data array as (ny, nx).

        state : dict
            State dictionary containing fitting data and metadata.

        calc_fit_metrics_func : callable
            Function to calculate fit quality metrics (qfit, cfit).

        define_flags_func : callable
            Function to define per-source bitwise flags.

        class_name : str
            Name of the calling class for warning messages.

        metadata_attrs : dict
            Dictionary of metadata attributes to add to the table.

        Returns
        -------
        results_tbl : `~astropy.table.QTable`
            Comprehensive results table containing:

            - Source ID and group ID
            - Initial parameter estimates
            - Fitted parameters with uncertainties
            - Quality metrics (qfit, cfit)
            - Bitwise flags indicating fit conditions
            - Iterator statistics (npixfit, group_size)
        """
        # Add metrics and flags column to fit_params. The results in the
        # state container match the order of the fit_params results,
        # which are in the same source ID order as the init_params.

        # Consume npixfit and group_size data, removing from state
        fit_params['npixfit'] = state.pop('npixfit')
        fit_params['group_size'] = state.pop('group_size')

        # Calculate fit metrics and remove the underlying data
        qfit, cfit, reduced_chi2 = calc_fit_metrics_func(fit_params)
        fit_params['qfit'] = qfit
        fit_params['cfit'] = cfit
        fit_params['reduced_chi2'] = reduced_chi2

        # Clean up residual data after metrics calculation
        state.pop('sum_abs_residuals', None)
        state.pop('cen_residuals', None)
        state.pop('reduced_chi2', None)

        # Calculate flags and check for convergence warnings before cleanup
        fit_params['flags'] = define_flags_func(fit_params, data_shape)

        # Join the fit_params table (with metrics and flags) to the
        # init_params table. By default, join will sort the rows by the
        # 'id' column.
        results_tbl = join(init_params, fit_params, keys='id',
                           join_type='left')

        # Reorder columns to place group_size after group_id
        # (both columns should always be present).
        group_size = results_tbl['group_size']
        results_tbl.remove_column('group_size')
        index = results_tbl.colnames.index('group_id') + 1
        results_tbl.add_column(group_size, index=index)

        # Check for fit convergence warnings before cleaning up state
        fit_error_indices = state.get('fit_error_indices')
        if (fit_error_indices is not None and len(fit_error_indices) > 0):
            warnings.warn('One or more fit(s) may not have converged. Please '
                          'check the "flags" column in the output table.',
                          AstropyUserWarning)

        # Clean up flag-related state data after use
        state.pop('fit_error_indices', None)
        state.pop('fitted_models_table', None)
        state.pop('valid_mask_by_id', None)

        meta = _get_meta()
        meta['psf_class'] = class_name

        # Add attribute metadata
        meta.update(metadata_attrs)

        # Convert to QTable and set metadata
        return QTable(results_tbl, meta=meta)
