# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to perform iterative PSF-fitting photometry.
"""

import warnings
from copy import deepcopy
from itertools import chain

import numpy as np
from astropy.nddata import NDData
from astropy.table import QTable, vstack
from astropy.utils import lazyproperty

from photutils.psf.photometry import PSFPhotometry
from photutils.psf.utils import ModelImageMixin, _create_call_docstring
from photutils.utils._repr import make_repr
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['IterativePSFPhotometry']


class IterativePSFPhotometry(ModelImageMixin):
    """
    Class to iteratively perform PSF photometry.

    This is a convenience class that iteratively calls the
    `PSFPhotometry` class to perform PSF photometry on an input image.
    It can be useful for crowded fields where faint sources are very
    close to bright sources and are not detected in the first pass of
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
        The rectangular shape around the initial center of a source that
        will be used to define the PSF-fitting data. If ``fit_shape``
        is a scalar then a square shape of size ``fit_shape`` will
        be used. If ``fit_shape`` has two elements, they must be in
        ``(ny, nx)`` order. Each element of ``fit_shape`` must be an odd
        number. In general, ``fit_shape`` should be set to a small size
        (e.g., ``(5, 5)``) that covers the region with the highest flux
        signal-to-noise.

    finder : callable or `~photutils.detection.StarFinderBase`
        A callable used to identify sources in an image. This is a
        required input for `IterativePSFPhotometry`. The ``finder`` must
        accept a 2D image as input and return a `~astropy.table.Table`
        containing the x and y centroid positions. These positions are
        used as the starting points for the PSF fitting. The allowed
        ``x`` column names are (same suffix for ``y``): ``'x_init'``,
        ``'xinit'``, ``'x'``, ``'x_0'``, ``'x0'``, ``'xcentroid'``,
        ``'x_centroid'``, ``'x_peak'``, ``'xcen'``, ``'x_cen'``,
        ``'xpos'``, ``'x_pos'``, ``'x_fit'``, and ``'xfit'``. If `None`,
        then the initial (x, y) model positions must be input using
        the ``init_params`` keyword when calling the class. The (x, y)
        values in ``init_params`` override this keyword *only for the
        first iteration*. If this class is run on an image that has
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
        The ``group_id`` values in ``init_params`` override this keyword
        *only for the first iteration*. A warning is raised if any group
        size is larger than ``group_warning_threshold`` sources.

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

    aperture_radius : float, optional
        The radius of the circular aperture used to estimate the
        initial flux of each source. This is a required input for
        `IterativePSFPhotometry`. If `None`, then the initial flux
        values must be provided in the ``init_params`` table. The
        aperture radius must be a strictly positive scalar. If initial
        flux values are present in the ``init_params`` table, they will
        override this keyword *only for the first iteration*.

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

    sub_shape : `None`, int, or length-2 array_like
        The rectangular shape around the fitted center of a source
        that will be used when subtracting the fitted PSF models.
        If ``sub_shape`` is a scalar then a square shape of size
        ``sub_shape`` will be used. If ``sub_shape`` has two
        elements, they must be in ``(ny, nx)`` order. Each element
        of ``sub_shape`` must be an odd number. If `None`, then
        ``sub_shape`` will be defined by the model bounding box.
        This keyword must be specified if the model does not have a
        ``bounding_box`` attribute.

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

    If the fitted positions are significantly different from the
    initial positions, one can rerun the `IterativePSFPhotometry` class
    using the fit results as the input ``init_params``, which will
    change the fitted cutout region for each source. After running
    `IterativePSFPhotometry`, you can use the `results_to_init_params`
    method to generate a table of initial parameters that can be used
    in a subsequent call to `IterativePSFPhotometry`. This table will
    contain the fitted (x, y) positions, fluxes, and any other model
    parameters that were fit.

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

    Care should be taken in defining the source groups. Simultaneously
    fitting very large source groups is computationally expensive and
    error-prone. Internally, source grouping requires the creation of
    a compound Astropy model. Due to the way compound Astropy models
    are currently constructed, large groups also require excessively
    large amounts of memory; this will hopefully be fixed in a future
    Astropy version. A warning will be raised if the number of sources
    in a group exceeds the ``group_warning_threshold`` value.
    """

    def __init__(self, psf_model, fit_shape, finder, *, grouper=None,
                 fitter=None, fitter_maxiters=100, xy_bounds=None,
                 maxiters=3, mode='new', aperture_radius=None,
                 localbkg_estimator=None, group_warning_threshold=25,
                 sub_shape=None, progress_bar=False):

        if finder is None:
            msg = 'finder cannot be None for IterativePSFPhotometry'
            raise ValueError(msg)

        if aperture_radius is None:
            msg = 'aperture_radius cannot be None for IterativePSFPhotometry'
            raise ValueError(msg)

        threshold = group_warning_threshold
        self._psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                                      grouper=grouper, fitter=fitter,
                                      fitter_maxiters=fitter_maxiters,
                                      xy_bounds=xy_bounds,
                                      aperture_radius=aperture_radius,
                                      localbkg_estimator=localbkg_estimator,
                                      group_warning_threshold=threshold,
                                      progress_bar=progress_bar)

        self.maxiters = self._validate_maxiters(maxiters)

        if mode not in ['new', 'all']:
            msg = 'mode must be "new" or "all"'
            raise ValueError(msg)
        if mode == 'all' and grouper is None:
            msg = 'grouper must be input for the "all" mode'
            raise ValueError(msg)
        self.mode = mode

        self.sub_shape = sub_shape

        self._reset_results()

    def _reset_results(self):
        """
        Reset these attributes for each __call__.
        """
        self.fit_results = []
        self.results = None
        self.__dict__.pop('_model_image_params', None)  # lazyproperty

    def __repr__(self):
        params = ('psf_model', 'fit_shape', 'finder', 'grouper', 'fitter',
                  'fitter_maxiters', 'xy_bounds', 'maxiters', 'mode',
                  'localbkg_estimator', 'aperture_radius', 'sub_shape',
                  'progress_bar')
        overrides = {
            'psf_model': self._psfphot.psf_model,
            'fit_shape': self._psfphot.fit_shape,
            'finder': self._psfphot.finder,
            'grouper': self._psfphot.grouper,
            'fitter': self._psfphot.fitter,
            'fitter_maxiters': self._psfphot.fitter_maxiters,
            'xy_bounds': self._psfphot.xy_bounds,
            'localbkg_estimator': self._psfphot.localbkg_estimator,
            'aperture_radius': self._psfphot.aperture_radius,
            'progress_bar': self._psfphot.progress_bar,
        }
        return make_repr(self, params, overrides=overrides)

    @staticmethod
    def _validate_maxiters(maxiters):
        if (not np.isscalar(maxiters) or maxiters <= 0
                or ~np.isfinite(maxiters)):
            msg = 'maxiters must be a strictly-positive scalar'
            raise ValueError(msg)
        if maxiters != int(maxiters):
            msg = 'maxiters must be an integer'
            raise ValueError(msg)
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

    @staticmethod
    def _move_column(table, colname, colname_after):
        """
        Move a column to a new position in a table.

        The table is modified in place.

        Parameters
        ----------
        table : `~astropy.table.Table`
            The input table.

        colname : str
            The column name to move.

        colname_after : str
            The column name after which to place the moved column.

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
        return table[colnames]

    def _measure_init_fluxes(self, data, mask, sources):
        """
        Measure initial fluxes for the new sources from the residual
        data.

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
        flux_col = self._psfphot._param_mapper.init_colnames['flux']
        sources[flux_col] = flux
        return sources

    def _prepare_next_iteration_sources(self, residual_data, mask, new_sources,
                                        orig_sources):
        """
        Create an initial parameters table for the next iteration.

        This method combines the results from the previous iteration
        with newly found sources, ensuring all sources have unique
        IDs and correctly named '_init' columns for the next run of
        PSFPhotometry.

        Parameters
        ----------
        residual_data : 2D `~numpy.ndarray`
            The residual image from the previous iteration, used to
            measure initial fluxes for new sources.

        mask : 2D `~numpy.ndarray` or `None`
            The mask for the data.

        new_sources : `~astropy.table.Table`
            A table with '_init' columns for the x and y positions of
            newly detected sources.

        orig_sources : `~astropy.table.Table`
            The results table (from the previous iteration's fit) for
            the original sources.

        Returns
        -------
        init_params : `~astropy.table.Table`
            A table ready to be used as `init_params` for the next
            photometry iteration.
        """
        param_mapper = self._psfphot._param_mapper

        # build a new table constructively, converting _fit columns to
        # _init columns
        prepared_orig = QTable()
        prepared_orig['id'] = orig_sources['id']

        for alias in param_mapper.alias_to_model_param:
            init_col = param_mapper.init_colnames.get(alias)
            if init_col and init_col in orig_sources.colnames:
                # use the previous fit result as the initial guess for
                # the next iteration
                prepared_orig[init_col] = orig_sources[init_col]

        # prepare the newly found sources
        max_id = np.max(orig_sources['id']) if len(orig_sources) > 0 else 0
        new_sources['id'] = np.arange(len(new_sources)) + max_id + 1

        # measure initial fluxes and add default values for other model
        # parameters
        new_sources = self._measure_init_fluxes(residual_data, mask,
                                                new_sources)

        model_param_mapper = param_mapper.alias_to_model_param
        for alias, model_param_name in model_param_mapper.items():
            init_col = param_mapper.init_colnames.get(alias)
            if init_col and init_col not in new_sources.colnames:
                default_value = getattr(self._psfphot.psf_model,
                                        model_param_name)
                new_sources[init_col] = default_value

        # combine tables
        new_sources.meta.pop('date', None)  # prevent merge conflicts

        return vstack([prepared_orig, new_sources])

    @_create_call_docstring(iterative=True)
    def __call__(self, data, *, mask=None, error=None, init_params=None):
        if isinstance(data, NDData):
            data_, mask, error = PSFPhotometry._coerce_nddata(data)
            return self.__call__(data_, mask=mask, error=error,
                                 init_params=init_params)

        # reset results from previous runs
        self._reset_results()

        with warnings.catch_warnings(record=True) as rwarn0:
            phot_tbl = self._psfphot(data, mask=mask, error=error,
                                     init_params=init_params)
            self.fit_results.append(deepcopy(self._psfphot))

        # this needs to be run outside of the context manager to be able
        # to reemit any warnings
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
                    residual_data, psf_shape=self.sub_shape)

                # do not warn if no sources are found beyond the first
                # iteration
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', NoDetectionsWarning)

                    new_sources = self._psfphot.finder(residual_data,
                                                       mask=mask)
                    if new_sources is None:  # no new sources detected
                        break

                finder_results = new_sources.copy()

                # Convert finder results to init params format
                data_processor = self._psfphot._data_processor
                new_sources = data_processor._convert_finder_to_init(
                    new_sources)

                if self.mode == 'all':
                    init_params = self._prepare_next_iteration_sources(
                        residual_data, mask, new_sources,
                        self._psfphot.results_to_init_params())

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
            phot_tbl = self._move_column(phot_tbl, 'iter_detected',
                                         'group_size')

        # add table metadata not already set by PSFPhotometry
        phot_tbl.meta['psf_class'] = self.__class__.__name__
        phot_tbl.meta['maxiters'] = self.maxiters
        phot_tbl.meta['mode'] = self.mode
        phot_tbl.meta['sub_shape'] = self.sub_shape

        # emit unique warnings
        recorded_warnings = rwarn0 + rwarn1
        self._emit_warnings(recorded_warnings)

        self.results = phot_tbl

        return phot_tbl

    def results_to_init_params(self):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to those expected for the
        initial parameters table. It can be used as the ``init_params``
        for subsequent `PSFPhotometry` fits.

        Rows that contain non-finite fitted values are removed.
        """
        return self._psfphot._results_to_init_params(self.results,
                                                     reset_id=True)

    def results_to_model_params(self):
        """
        Create a table of the fitted model parameters from the results.

        The table columns are named according to the PSF model parameter
        names. It can also be used to reconstruct the fitted PSF models
        for visualization or further analysis.

        Rows that contain non-finite fitted values are removed.
        """
        return self._psfphot._results_to_model_params(
            self.results, self._psfphot._param_mapper, reset_id=True)

    @lazyproperty
    def _model_image_params(self):
        """
        A helper property that provides the necessary parameters to
        ModelImageMixin.
        """
        psf_model = self._psfphot.psf_model
        progress_bar = self._psfphot.progress_bar

        if self.mode == 'new':
            # in 'new' mode: we stack the results from all iterations
            all_fit_params = []
            all_local_bkgs = []
            for result_obj in self.fit_results:
                fm_tbl = result_obj.results_to_model_params()
                if fm_tbl is not None:
                    all_fit_params.append(fm_tbl)
                    all_local_bkgs.append(result_obj.init_params['local_bkg'])

            fit_params = vstack(all_fit_params) if all_fit_params else None
            local_bkgs = list(chain.from_iterable(all_local_bkgs))

        elif self.mode == 'all':
            # in 'all' mode: only the final iteration contains all sources
            final_result = self.fit_results[-1]
            fit_params = final_result.results_to_model_params()
            local_bkgs = final_result.init_params['local_bkg']

        else:  # pragma: no cover
            # should never happen due to the mode validation in __init__
            msg = f'Invalid mode "{self.mode}"'
            raise ValueError(msg)

        return {'psf_model': psf_model,
                'model_params': fit_params,
                'local_bkg': local_bkgs,
                'progress_bar': progress_bar,
                }

    def make_model_image(self, shape, *, psf_shape=None,
                         include_localbkg=False):

        if not self.fit_results:
            msg = ('No results available. Please run the '
                   'IterativePSFPhotometry instance first.')
            raise ValueError(msg)

        return ModelImageMixin.make_model_image(
            self, shape, psf_shape=psf_shape,
            include_localbkg=include_localbkg)

    def make_residual_image(self, data, *, psf_shape=None,
                            include_localbkg=False):

        if not self.fit_results:
            msg = ('No results available. Please run the '
                   'IterativePSFPhotometry instance first.')
            raise ValueError(msg)

        return ModelImageMixin.make_residual_image(
            self, data, psf_shape=psf_shape, include_localbkg=include_localbkg)
