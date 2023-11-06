# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform PSF-fitting photometry.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import StdDevUncertainty
from astropy.nddata.utils import NoOverlapError, overlap_slices
from astropy.stats import SigmaClip, gaussian_sigma_to_fwhm
from astropy.table import Column, QTable, hstack, vstack
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import MMMBackground
from photutils.detection import DAOStarFinder
from photutils.psf.groupstars import DAOGroup
from photutils.psf.utils import (_extract_psf_fitting_names,
                                 get_grouped_psf_model, subtract_psf)
from photutils.utils._misc import _get_meta
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['BasicPSFPhotometry', 'IterativelySubtractedPSFPhotometry',
           'DAOPhotPSFPhotometry']


@deprecated('1.9.0', alternative='`photutils.psf.PSFPhotometry`')
class BasicPSFPhotometry:
    """
    This class implements a PSF photometry algorithm that can find
    sources in an image, group overlapping sources into a single model,
    fit the model to the sources, and subtracting the models from the
    image. This is roughly equivalent to the DAOPHOT routines FIND,
    GROUP, NSTAR, and SUBTRACT.  This implementation allows a flexible
    and customizable interface to perform photometry. For instance, one
    is able to use different implementations for grouping and finding
    sources by using ``group_maker`` and ``finder`` respectively. In
    addition, sky background estimation is performed by
    ``bkg_estimator``.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given
        star overlaps with any other and label them as belonging
        to the same group. ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``. This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should
        contain integers starting from ``1`` that indicate which group a
        given source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any \
            `~photutils.background.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.IntegratedGaussianPRF` or any
        other suitable 2D model. This object needs to identify three
        parameters (position of center in x and y coordinates and the
        flux) in order to set them to suitable starting values for each
        fit. The names of these parameters should be given as ``x_0``,
        ``y_0`` and ``flux``. `~photutils.psf.prepare_psf_model` can be
        used to prepare any 2D model to match this assumption.
    fitshape : int or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-fitting region. If ``fitshape`` is a
        scalar then a square shape of size ``fitshape`` will be used.
        If ``fitshape`` has two elements, they must be in ``(ny, nx)``
        order. Each element of ``fitshape`` must be an odd number.
    finder : callable or instance of any \
            `~photutils.detection.StarFinderBase` subclasses or None
        ``finder`` should be able to identify stars, i.e., compute a
        rough estimate of the centroids, in a given 2D image.
        ``finder`` receives as input a 2D image and returns an
        `~astropy.table.Table` object which contains columns with names:
        ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
        ``id`` is an integer-valued column starting from ``1``,
        ``xcentroid`` and ``ycentroid`` are center position estimates of
        the sources and ``flux`` contains flux estimates of the sources.
        See, e.g., `~photutils.detection.DAOStarFinder`.  If ``finder``
        is ``None``, initial guesses for positions of objects must be
        provided.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        Fitter object used to compute the optimized centroid positions
        and/or flux of the identified sources. See
        `~astropy.modeling.fitting` for more details on fitters.
    aperture_radius : `None` or float
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources.  ``aperture_radius`` must
        be set if initial flux guesses are not input to the photometry
        class via the ``init_guesses`` keyword.  For tabular PSF models
        (e.g., an `EPSFModel`), you must input the ``aperture_radius``
        keyword.  For analytical PSF models, alternatively you may
        define a FWHM attribute on your input psf_model.
    extra_output_cols : list of str, optional
        List of additional columns for parameters derived by any of the
        intermediate fitting steps (e.g., ``finder``), such as roundness
        or sharpness.
    subshape : `None`, int, or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-subtraction region. If `None`, then
        ``fitshape`` will be used. If ``subshape`` is a scalar then a
        square shape of size ``subshape`` will be used. If ``subshape``
        has two elements, they must be in ``(ny, nx)`` order. Each
        element of ``subshape`` must be an odd number.

    Notes
    -----
    Note that an ambiguity arises whenever ``finder`` and
    ``init_guesses`` (keyword argument for ``do_photometry``) are both
    not ``None``. In this case, ``finder`` is ignored and initial
    guesses are taken from ``init_guesses``. In addition, an warning is
    raised to remind the user about this behavior.

    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g., manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape, *,
                 finder=None, fitter=LevMarLSQFitter(), aperture_radius=None,
                 extra_output_cols=None, subshape=None):
        self.group_maker = group_maker
        self.bkg_estimator = bkg_estimator
        self.psf_model = psf_model
        self.fitter = fitter
        self.fitshape = fitshape
        self.finder = finder
        self.aperture_radius = aperture_radius
        self._pars_to_set = None
        self._pars_to_output = None
        self._residual_image = None
        self._extra_output_cols = extra_output_cols
        self.subshape = subshape

    @property
    def fitshape(self):
        return self._fitshape

    @fitshape.setter
    def fitshape(self, value):
        value = np.asarray(value)

        # assume a lone value should mean both axes
        if value.shape == ():
            value = np.array((value, value))

        if value.size == 2:
            if np.all(value) > 0:
                if np.all(value % 2) == 1:
                    self._fitshape = tuple(value)
                else:
                    raise ValueError('fitshape must be odd integer-valued, '
                                     f'received fitshape={value}')
            else:
                raise ValueError('fitshape must have positive elements, '
                                 f'received fitshape={value}')
        else:
            raise ValueError('fitshape must have two dimensions, '
                             f'received fitshape={value}')

    @property
    def subshape(self):
        return self._subshape

    @subshape.setter
    def subshape(self, value):
        if value is None:
            self._subshape = self._fitshape
            return

        value = np.asarray(value)

        # assume a lone value should mean both axes
        if value.shape == ():
            value = np.array((value, value))

        if value.size == 2:
            if np.all(value) > 0:
                if np.all(value % 2) == 1:
                    self._subshape = tuple(value)
                else:
                    raise ValueError('subshape must be odd integer-valued, '
                                     f'received subshape={value}')
            else:
                raise ValueError('subshape must have positive elements, '
                                 f'received subshape={value}')
        else:
            raise ValueError('subshape must have two dimensions, '
                             f'received subshape={value}')

    @property
    def aperture_radius(self):
        return self._aperture_radius

    @aperture_radius.setter
    def aperture_radius(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._aperture_radius = value
        elif value is None:
            self._aperture_radius = value
        else:
            raise ValueError('aperture_radius must be a positive number')

    def get_residual_image(self):
        """
        Return an image that is the result of the subtraction between
        the original image and the fitted sources.

        Returns
        -------
        residual_image : 2D `~numpy.ndarray`
            The 2D residual image.
        """
        return self._residual_image

    def set_aperture_radius(self):
        """
        Set the fallback aperture radius for initial flux calculations
        in cases where no flux is supplied for a given star.
        """
        if hasattr(self.psf_model, 'fwhm'):
            self.aperture_radius = self.psf_model.fwhm.value
        elif hasattr(self.psf_model, 'sigma'):
            self.aperture_radius = (self.psf_model.sigma.value
                                    * gaussian_sigma_to_fwhm)
        # If PSF model doesn't have FWHM or sigma value -- as it
        # is not a Gaussian; most likely because it's an ePSF --
        # then we fall back on fitting a circle of the average
        # size of the fitting box. As ``fitshape`` is the width
        # of the box, we need (width-1)/2 as the radius.
        else:
            self.aperture_radius = float(np.amin((np.asanyarray(
                                         self.fitshape) - 1) / 2))
            warnings.warn('aperture_radius is None and could not '
                          'be determined by psf_model. Setting '
                          'radius to the smallest fitshape size. '
                          'This aperture radius will be used if '
                          'initial fluxes require computing for any '
                          'input stars. If fitshape is significantly '
                          'larger than the psf_model core lengthscale, '
                          'consider supplying a specific aperture_radius.',
                          AstropyUserWarning)

    @staticmethod
    def _make_mask(image, mask):
        if mask is not None:
            if image.shape != mask.shape:
                raise ValueError('image and mask must have the same shape')

        # if NaNs are in the data, no actually fitting takes place
        # https://github.com/astropy/astropy/pull/12811
        finite_mask = ~np.isfinite(image)

        if mask is not None:
            mask |= finite_mask
            if np.any(finite_mask & ~mask):
                warnings.warn('Input data contains unmasked non-finite '
                              'values (NaN or inf), which were '
                              'automatically ignored.', AstropyUserWarning)
        else:
            mask = finite_mask
            if np.any(finite_mask):
                warnings.warn('Input data contains unmasked non-finite '
                              'values (NaN or inf), which were '
                              'automatically ignored.', AstropyUserWarning)
        return mask

    def __call__(self, image, *, mask=None, init_guesses=None,
                 progress_bar=False, uncertainty=None):
        """
        Perform PSF photometry. See `do_photometry` for more details
        including the `__call__` signature.
        """
        return self.do_photometry(image, mask=mask, init_guesses=init_guesses,
                                  progress_bar=progress_bar,
                                  uncertainty=uncertainty)

    def do_photometry(self, image, *, mask=None, init_guesses=None,
                      progress_bar=False, uncertainty=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this
        method uses ``init_guesses`` as initial guesses for the
        centroids. If the centroid positions are set as ``fixed`` in the
        PSF model ``psf_model``, then the optimizer will only consider
        the flux as a variable.

        Parameters
        ----------
        image : 2D `~numpy.ndarray`
            Image to perform photometry. Invalid data values (i.e., NaN
            or inf) are automatically ignored.
        init_guesses : `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the
            set of parameters. Columns 'x_0' and 'y_0' which represent
            the positions (in pixel coordinates) for each object must be
            present.  'flux_0' can also be provided to set initial
            fluxes.  If 'flux_0' is not provided, aperture photometry is
            used to estimate initial values for the fluxes. Additional
            columns of the form '<parametername>_0' will be used to set
            the initial guess for any parameters of the ``psf_model``
            model that are not fixed. If ``init_guesses`` supplied with
            ``extra_output_cols`` the initial values are used; if the columns
            specified in ``extra_output_cols`` are not given in
            ``init_guesses`` then NaNs will be returned.
        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``image``, where
            a `True` value indicates the corresponding element of
            ``image`` is masked.
        progress_bar : bool, optional
            Whether to display a progress bar when fitting the
            star groups. The progress bar requires that the `tqdm
            <https://tqdm.github.io/>`_ optional dependency be
            installed. Note that the progress bar does not currently
            work in the Jupyter console due to limitations in ``tqdm``.
        uncertainty : 2D `~numpy.ndarray`, optional
            Stddev uncertainty for each element in ``image``.

        Returns
        -------
        output_tab : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process. Uncertainties on the fitted parameters
            are reported as columns called ``<paramname>_unc`` provided
            that the fitter object contains a dictionary called
            ``fit_info`` with the key ``param_cov``, which contains the
            covariance matrix. If ``param_cov`` is not present,
            uncertanties are not reported.
        """
        mask = self._make_mask(image, mask)

        if self.bkg_estimator is not None:
            image = image - self.bkg_estimator(image)

        if self.aperture_radius is None:
            self.set_aperture_radius()

        skip_group_maker = False
        if init_guesses is not None:
            # make sure the code does not modify user's input
            init_guesses = init_guesses.copy()

            if self.finder is not None:
                warnings.warn('Both init_guesses and finder are different '
                              'than None, which is ambiguous. finder is '
                              'going to be ignored.', AstropyUserWarning)

            colnames = init_guesses.colnames
            if 'group_id' in colnames:
                warnings.warn('init_guesses contains a "group_id" column. '
                              'The group_maker step will be skipped.',
                              AstropyUserWarning)
                skip_group_maker = True

            if 'flux_0' not in colnames:
                positions = np.transpose((init_guesses['x_0'],
                                          init_guesses['y_0']))
                apertures = CircularAperture(positions,
                                             r=self.aperture_radius)

                init_guesses['flux_0'] = aperture_photometry(
                    image, apertures, mask=mask)['aperture_sum']

            # if extra_output_cols have been given, check whether init_guesses
            # was supplied with extra_output_cols pre-attached and populate
            # columns not given with NaNs
            if self._extra_output_cols is not None:
                for col_name in self._extra_output_cols:
                    if col_name not in init_guesses.colnames:
                        init_guesses[col_name] = np.full(len(init_guesses),
                                                         np.nan)
        else:
            if self.finder is None:
                raise ValueError('Finder cannot be None if init_guesses are '
                                 'not given.')
            sources = self.finder(image, mask=mask)
            if sources is None:
                return None
            else:
                positions = np.transpose((sources['xcentroid'],
                                          sources['ycentroid']))
                apertures = CircularAperture(positions,
                                             r=self.aperture_radius)

                sources['aperture_flux'] = aperture_photometry(
                    image, apertures, mask=mask)['aperture_sum']

                # init_guesses should be the initial 3 required
                # parameters (x, y, flux) and then concatenated with any
                # additional sources, if there are any
                init_guesses = QTable(names=['x_0', 'y_0', 'flux_0'],
                                      data=[sources['xcentroid'],
                                            sources['ycentroid'],
                                            sources['aperture_flux']])

                # Currently only needed for the finder, as group_maker and
                # nstar return the original table with new columns, unlike
                # finder
                self._get_additional_columns(sources, init_guesses)

        self._define_fit_param_names()
        for p0, param in self._pars_to_set.items():
            if p0 not in init_guesses.colnames:
                init_guesses[p0] = (len(init_guesses)
                                    * [getattr(self.psf_model, param).value])

        if skip_group_maker:
            star_groups = init_guesses
        else:
            star_groups = self.group_maker(init_guesses)

        output_tab, self._residual_image = self.nstar(
            image, star_groups, mask=mask, progress_bar=progress_bar,
            uncertainty=uncertainty
        )
        star_groups = star_groups.group_by('group_id')

        if hasattr(output_tab, 'update'):  # requires Astropy >= 5.0
            star_groups.update(output_tab)
        else:
            common_cols = set(star_groups.colnames).intersection(
                output_tab.colnames)
            for name, col in output_tab.items():
                if name in common_cols:
                    star_groups.replace_column(name, col, copy=True)
                else:
                    star_groups.add_column(col, name=name, copy=True)

        star_groups.meta = _get_meta()

        return star_groups

    def nstar(self, image, star_groups, *, mask=None, progress_bar=False,
              uncertainty=None):
        """
        Fit, as appropriate, a compound or single model to the given
        ``star_groups``. Groups are fitted sequentially from the
        smallest to the biggest. In each iteration, ``image`` is
        subtracted by the previous fitted group.

        Parameters
        ----------
        image : 2D `~numpy.ndarray`
            Background-subtracted image. Invalid data values (i.e., NaN
            or inf) are automatically ignored.

        star_groups : `~astropy.table.Table`
            This table must contain the following columns: ``id``,
            ``group_id``, ``x_0``, ``y_0``, ``flux_0``.  ``x_0`` and
            ``y_0`` are initial estimates of the centroids and
            ``flux_0`` is an initial estimate of the flux. Additionally,
            columns named as ``<param_name>_0`` are required if any
            other parameter in the psf model is free (i.e., the
            ``fixed`` attribute of that parameter is ``False``).

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``image``, where
            a `True` value indicates the corresponding element of
            ``image`` is masked.

        progress_bar : bool, optional
            Use a progress bar to show progress over the star groups.
            The progress bar requires that the `tqdm
            <https://tqdm.github.io/>`_ optional dependency be
            installed. Note that the progress bar does not currently
            work in the Jupyter console due to limitations in ``tqdm``.

        uncertainty : 2D `~numpy.ndarray`, optional
            Stddev uncertainty for each element in ``image``.

        Returns
        -------
        result_tab : `~astropy.table.QTable`
            Astropy table that contains photometry results.

        image : 2D `~numpy.ndarray`
            Residual image.
        """
        result_tab = QTable()
        for param_tab_name in self._pars_to_output:
            result_tab.add_column(Column(name=param_tab_name))

        unc_tab = QTable()
        for param, isfixed in self.psf_model.fixed.items():
            if not isfixed:
                unc_tab.add_column(Column(name=param + '_unc'))

        y, x = np.indices(image.shape)

        star_groups = star_groups.group_by('group_id')
        group_iter = star_groups.groups

        if progress_bar:
            group_iter = add_progress_bar(group_iter, desc='Fit source/group')  # pragma: no cover

        for group in group_iter:
            group_psf = get_grouped_psf_model(self.psf_model, group,
                                              self._pars_to_set)
            usepixel = np.zeros_like(image, dtype=bool)

            for row in group:
                usepixel[overlap_slices(large_array_shape=image.shape,
                                        small_array_shape=self.fitshape,
                                        position=(row['y_0'], row['x_0']),
                                        mode='trim')[0]] = True

            if mask is not None:
                usepixel &= ~mask

            if hasattr(image, 'uncertainty'):
                sigma = image.uncertainty.represent_as(StdDevUncertainty).array
                weights = 1 / sigma[usepixel]
            elif uncertainty is not None:
                weights = 1 / uncertainty[usepixel]
            else:
                weights = None

            fit_model = self.fitter(group_psf, x[usepixel], y[usepixel],
                                    image[usepixel], weights=weights)
            param_table = self._model_params2table(fit_model, group)
            result_tab = vstack([result_tab, param_table])
            unc_tab = vstack([unc_tab, self._get_uncertainties(len(group))])

            # do not subtract if the fitting did not go well
            try:
                image = subtract_psf(image, self.psf_model, param_table,
                                     subshape=self.subshape)
            except NoOverlapError:
                pass

        for col in unc_tab.colnames:
            if np.all(np.isnan(unc_tab[col])):
                unc_tab.remove_column(col)

        result_tab = hstack([result_tab, unc_tab])

        return result_tab, image

    def _get_additional_columns(self, in_table, out_table):
        """
        Function to parse additional columns from ``in_table`` and add them to
        ``out_table``.
        """
        if self._extra_output_cols is not None:
            for col_name in self._extra_output_cols:
                if col_name in in_table.colnames:
                    out_table[col_name] = in_table[col_name]

    def _define_fit_param_names(self):
        """
        Convenience function to define mappings between the names of the
        columns in the initial guess table (and the name of the fitted
        parameters) and the actual name of the parameters in the model.

        This method sets the following parameters on the ``self`` object:
        * ``pars_to_set`` : Dict which maps the names of the parameters
          initial guesses to the actual name of the parameter in the
          model.
        * ``pars_to_output`` : Dict which maps the names of the fitted
          parameters to the actual name of the parameter in the model.
        """
        xname, yname, fluxname = _extract_psf_fitting_names(self.psf_model)
        self._pars_to_set = {'x_0': xname, 'y_0': yname, 'flux_0': fluxname}
        self._pars_to_output = {'x_fit': xname, 'y_fit': yname,
                                'flux_fit': fluxname}

        for p, isfixed in self.psf_model.fixed.items():
            p0 = p + '_0'
            pfit = p + '_fit'
            if p not in (xname, yname, fluxname) and not isfixed:
                self._pars_to_set[p0] = p
                self._pars_to_output[pfit] = p

    def _get_uncertainties(self, star_group_size):
        """
        Retrieve uncertainties on fitted parameters from the fitter
        object.

        Parameters
        ----------
        star_group_size : int
            Number of stars in the given group.

        Returns
        -------
        unc_tab : `~astropy.table.QTable`
            A table which contains uncertainties on the fitted parameters.
            The uncertainties are reported as one standard deviation.
        """
        unc_tab = QTable()
        for param_name in self.psf_model.param_names:
            if not self.psf_model.fixed[param_name]:
                unc_tab.add_column(Column(name=param_name + '_unc',
                                          data=np.empty(star_group_size)))

        k = 0
        n_fit_params = len(unc_tab.colnames)
        param_cov = self.fitter.fit_info.get('param_cov', None)

        # variance is sometimes returned as a negative value
        # ignore sqrt(negative value) RuntimeWarnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for i in range(star_group_size):
                if param_cov is None:
                    unc_tab[i] = [np.nan] * n_fit_params
                else:
                    sig = np.sqrt(np.diag(param_cov))
                    unc_tab[i] = sig[k: k + n_fit_params]
                    k += n_fit_params

        return unc_tab

    def _model_params2table(self, fit_model, star_group):
        """
        Place fitted parameters into an astropy table.

        Parameters
        ----------
        fit_model : `astropy.modeling.Fittable2DModel` instance
            PSF or PRF model to fit the data. Could be one of the models
            in this package like `~photutils.psf.IntegratedGaussianPRF`
            or any other suitable 2D model.

        star_group : `~astropy.table.Table`
            the star group instance.

        Returns
        -------
        param_tab : `~astropy.table.QTable`
            A table that contains the fitted parameters.
        """
        param_tab = QTable()

        for param_tab_name in self._pars_to_output:
            param_tab.add_column(Column(name=param_tab_name,
                                        data=np.empty(len(star_group))))

        if len(star_group) > 1:
            for i in range(len(star_group)):
                for param_tab_name, param_name in self._pars_to_output.items():
                    # get sub_model corresponding to star with index i as name
                    # name was set in utils.get_grouped_psf_model()
                    # we can't use model['name'] here as that only
                    # searches leaves and we might want a intermediate
                    # node of the tree
                    sub_models = [model for model
                                  in fit_model.traverse_postorder()
                                  if model.name == i]
                    if len(sub_models) != 1:
                        raise ValueError('sub_models must have a length of 1')
                    sub_model = sub_models[0]

                    param_tab[param_tab_name][i] = getattr(sub_model,
                                                           param_name).value
        else:
            for param_tab_name, param_name in self._pars_to_output.items():
                param_tab[param_tab_name] = getattr(fit_model,
                                                    param_name).value

        return param_tab


@deprecated('1.9.0', alternative='`photutils.psf.IterativePSFPhotometry`')
class IterativelySubtractedPSFPhotometry(BasicPSFPhotometry):
    """
    This class implements an iterative algorithm to perform point spread
    function photometry in crowded fields. This consists of applying a
    loop of find sources, make groups, fit groups, subtract groups, and
    then repeat until no more stars are detected or a given number of
    iterations is reached.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given
        star overlaps with any other and label them as belonging
        to the same group. ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``. This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should
        contain integers starting from ``1`` that indicate which group a
        given source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any \
            `~photutils.background.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.IntegratedGaussianPRF` or any
        other suitable 2D model. This object needs to identify three
        parameters (position of center in x and y coordinates and the
        flux) in order to set them to suitable starting values for each
        fit. The names of these parameters should be given as ``x_0``,
        ``y_0`` and ``flux``. `~photutils.psf.prepare_psf_model` can be
        used to prepare any 2D model to match this assumption.
    fitshape : int or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-fitting region. If ``fitshape`` is a
        scalar then a square shape of size ``fitshape`` will be used.
        If ``fitshape`` has two elements, they must be in ``(ny, nx)``
        order. Each element of ``fitshape`` must be an odd number.
    finder : callable or instance of any \
            `~photutils.detection.StarFinderBase` subclasses
        ``finder`` should be able to identify stars, i.e., compute a
        rough estimate of the centroids, in a given 2D image.
        ``finder`` receives as input a 2D image and returns an
        `~astropy.table.Table` object which contains columns with names:
        ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
        ``id`` is an integer-valued column starting from ``1``,
        ``xcentroid`` and ``ycentroid`` are center position estimates of
        the sources and ``flux`` contains flux estimates of the sources.
        See, e.g., `~photutils.detection.DAOStarFinder` or
        `~photutils.detection.IRAFStarFinder`.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        Fitter object used to compute the optimized centroid positions
        and/or flux of the identified sources. See
        `~astropy.modeling.fitting` for more details on fitters.
    aperture_radius : float
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources. If ``None``, one FWHM will
        be used if it can be determined from the ```psf_model``.
    niters : int or None
        Number of iterations to perform of the loop FIND, GROUP,
        SUBTRACT, NSTAR. If None, iterations will proceed until no more
        stars remain.  Note that in this case it is *possible* that the
        loop will never end if the PSF has structure that causes
        subtraction to create new sources infinitely.
    extra_output_cols : list of str, optional
        List of additional columns for parameters derived by any of the
        intermediate fitting steps (e.g., ``finder``), such as roundness
        or sharpness.
    subshape : `None`, int, or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-subtraction region. If `None`, then
        ``fitshape`` will be used. If ``subshape`` is a scalar then a
        square shape of size ``subshape`` will be used. If ``subshape``
        has two elements, they must be in ``(ny, nx)`` order. Each
        element of ``subshape`` must be an odd number.

    Notes
    -----
    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g., manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape,
                 finder, *, fitter=LevMarLSQFitter(), niters=3,
                 aperture_radius=None, extra_output_cols=None, subshape=None):

        super().__init__(group_maker, bkg_estimator, psf_model, fitshape,
                         finder=finder, fitter=fitter,
                         aperture_radius=aperture_radius,
                         extra_output_cols=extra_output_cols,
                         subshape=subshape)
        self.niters = niters

    @property
    def niters(self):
        return self._niters

    @niters.setter
    def niters(self, value):
        if value is None:
            self._niters = None
        else:
            try:
                if value <= 0:
                    raise ValueError('niters must be positive.')
                self._niters = int(value)
            except ValueError as exc:
                raise ValueError('niters must be None or an integer or '
                                 'convertable into an integer.') from exc

    @property
    def finder(self):
        return self._finder

    @finder.setter
    def finder(self, value):
        if value is None:
            raise ValueError('finder cannot be None for '
                             'IterativelySubtractedPSFPhotometry - you may '
                             'want to use BasicPSFPhotometry. Please see the '
                             'Detection section on photutils documentation.')
        self._finder = value

    def do_photometry(self, image, *, mask=None, init_guesses=None,
                      progress_bar=False, uncertainty=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this
        method uses ``init_guesses`` as initial guesses for the
        centroids. If the centroid positions are set as ``fixed`` in the
        PSF model ``psf_model``, then the optimizer will only consider
        the flux as a variable.

        Parameters
        ----------
        image : 2D `~numpy.ndarray`
            Image to perform photometry. Invalid data values (i.e., NaN
            or inf) are automatically ignored.
        init_guesses : `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the
            set of parameters. Columns 'x_0' and 'y_0' which represent
            the positions (in pixel coordinates) for each object must be
            present.  'flux_0' can also be provided to set initial
            fluxes.  If 'flux_0' is not provided, aperture photometry is
            used to estimate initial values for the fluxes. Additional
            columns of the form '<parametername>_0' will be used to set
            the initial guess for any parameters of the ``psf_model``
            model that are not fixed. If ``init_guesses`` supplied with
            ``extra_output_cols`` the initial values are used; if the columns
            specified in ``extra_output_cols`` are not given in
            ``init_guesses`` then NaNs will be returned.
        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``image``, where
            a `True` value indicates the corresponding element of
            ``image`` is masked.
        uncertainty : 2D `~numpy.ndarray`, optional
            Stddev uncertainty for each element in ``image``.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            A table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process. Uncertainties on the fitted parameters
            are reported as columns called ``<paramname>_unc`` provided
            that the fitter object contains a dictionary called
            ``fit_info`` with the key ``param_cov``, which contains the
            covariance matrix.
        """
        mask = super()._make_mask(image, mask)

        if init_guesses is not None:
            table = super().do_photometry(image, mask=mask,
                                          init_guesses=init_guesses,
                                          progress_bar=progress_bar,
                                          uncertainty=uncertainty)
            table['iter_detected'] = np.ones(table['x_fit'].shape, dtype=int)

            # n_start = 2 because it starts in the second iteration
            # since the first iteration is above
            output_table = self._do_photometry(n_start=2, mask=mask,
                                               progress_bar=progress_bar,
                                               uncertainty=uncertainty)
            output_table = vstack([table, output_table])
        else:
            if self.bkg_estimator is not None:
                self._residual_image = image - self.bkg_estimator(image)
            else:
                self._residual_image = image

            if self.aperture_radius is None:
                self.set_aperture_radius()

            output_table = self._do_photometry(mask=mask,
                                               progress_bar=progress_bar,
                                               uncertainty=uncertainty)

        output_table.meta = _get_meta()

        return QTable(output_table)

    def _do_photometry(self, n_start=1, mask=None, progress_bar=False,
                       uncertainty=None):
        """
        Helper function which performs the iterations of the photometry
        process.

        Parameters
        ----------
        n_start : int
            Integer representing the start index of the iteration.  It
            is 1 if init_guesses are None, and 2 otherwise.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process.
        """
        output_table = QTable()
        self._define_fit_param_names()

        for (init_parname, fit_parname) in zip(self._pars_to_set.keys(),
                                               self._pars_to_output.keys()):
            output_table.add_column(Column(name=init_parname))
            output_table.add_column(Column(name=fit_parname))

        sources = self.finder(self._residual_image, mask=mask)

        n = n_start
        while ((sources is not None and len(sources) > 0)
               and (self.niters is None or n <= self.niters)):
            positions = np.transpose((sources['xcentroid'],
                                      sources['ycentroid']))
            apertures = CircularAperture(positions,
                                         r=self.aperture_radius)
            sources['aperture_flux'] = aperture_photometry(
                self._residual_image, apertures, mask=mask)['aperture_sum']

            init_guess_tab = QTable(names=['id', 'x_0', 'y_0', 'flux_0'],
                                    data=[sources['id'], sources['xcentroid'],
                                          sources['ycentroid'],
                                          sources['aperture_flux']])
            self._get_additional_columns(sources, init_guess_tab)

            for param_tab_name, param_name in self._pars_to_set.items():
                if param_tab_name not in (['x_0', 'y_0', 'flux_0']):
                    init_guess_tab.add_column(
                        Column(name=param_tab_name,
                               data=(getattr(self.psf_model, param_name)
                                     * np.ones(len(sources)))))

            star_groups = self.group_maker(init_guess_tab)
            table, self._residual_image = super().nstar(
                self._residual_image, star_groups, mask=mask,
                progress_bar=progress_bar, uncertainty=uncertainty)

            star_groups = star_groups.group_by('group_id')
            table = hstack([star_groups, table])

            table['iter_detected'] = n * np.ones(table['x_fit'].shape,
                                                 dtype=int)

            output_table = vstack([output_table, table])

            # do not warn if no sources are found beyond the first iteration
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NoDetectionsWarning)
                sources = self.finder(self._residual_image, mask=mask)

            n += 1

        return output_table


@deprecated('1.9.0', alternative='`photutils.psf.IterativePSFPhotometry`')
class DAOPhotPSFPhotometry(IterativelySubtractedPSFPhotometry):
    """
    This class implements  an iterative algorithm based on the DAOPHOT
    algorithm presented by Stetson (1987) to perform point spread
    function photometry in crowded fields. This consists of applying a
    loop of find sources, make groups, fit groups, subtract groups, and
    then repeat until no more stars are detected or a given number of
    iterations is reached.

    Basically, this classes uses
    `~photutils.psf.IterativelySubtractedPSFPhotometry`, but with
    grouping, finding, and background estimation routines defined a
    priori. More precisely, this class uses `~photutils.psf.DAOGroup`
    for grouping, `~photutils.detection.DAOStarFinder` for finding
    sources, and `~photutils.background.MMMBackground` for background
    estimation. Those classes are based on GROUP, FIND, and SKY routines
    used in DAOPHOT, respectively.

    The parameter ``crit_separation`` is associated with
    `~photutils.psf.DAOGroup`.  ``sigma_clip`` is associated with
    `~photutils.background.MMMBackground`.  ``threshold`` and ``fwhm``
    are associated with `~photutils.detection.DAOStarFinder`.
    Parameters from ``ratio`` to ``roundhi`` are also associated with
    `~photutils.detection.DAOStarFinder`.

    Parameters
    ----------
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated
        by less than this distance will be placed in the same group.
    threshold : float
        The absolute image value above which to select sources.
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.IntegratedGaussianPRF` or any
        other suitable 2D model. This object needs to identify three
        parameters (position of center in x and y coordinates and the
        flux) in order to set them to suitable starting values for each
        fit. The names of these parameters should be given as ``x_0``,
        ``y_0`` and ``flux``. `~photutils.psf.prepare_psf_model` can be
        used to prepare any 2D model to match this assumption.
    fitshape : int or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-fitting region. If ``fitshape`` is a
        scalar then a square shape of size ``fitshape`` will be used.
        If ``fitshape`` has two elements, they must be in ``(ny, nx)``
        order. Each element of ``fitshape`` must be an odd number.
    sigma : float, optional
        Number of standard deviations used to perform sigma clip with a
        `astropy.stats.SigmaClip` object.
    ratio : float, optional
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).
    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.
    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        (2.0*sqrt(2.0*log(2.0)))``].
    sharplo : float, optional
        The lower bound on sharpness for object detection.
    sharphi : float, optional
        The upper bound on sharpness for object detection.
    roundlo : float, optional
        The lower bound on roundess for object detection.
    roundhi : float, optional
        The upper bound on roundess for object detection.
    fitter : `~astropy.modeling.fitting.Fitter` instance
        Fitter object used to compute the optimized centroid positions
        and/or flux of the identified sources. See
        `~astropy.modeling.fitting` for more details on fitters.
    niters : int or None
        Number of iterations to perform of the loop FIND, GROUP,
        SUBTRACT, NSTAR. If None, iterations will proceed until no more
        stars remain.  Note that in this case it is *possible* that the
        loop will never end if the PSF has structure that causes
        subtraction to create new sources infinitely.
    aperture_radius : float
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources. If ``None``, one FWHM will
        be used if it can be determined from the ```psf_model``.
    extra_output_cols : list of str, optional
        List of additional columns for parameters derived by any of the
        intermediate fitting steps (e.g., ``finder``), such as roundness
        or sharpness.
    subshape : `None`, int, or length-2 array_like
        Rectangular shape around the center of a star that will be
        used to define the PSF-subtraction region. If `None`, then
        ``fitshape`` will be used. If ``subshape`` is a scalar then a
        square shape of size ``subshape`` will be used. If ``subshape``
        has two elements, they must be in ``(ny, nx)`` order. Each
        element of ``subshape`` must be an odd number.

    Notes
    -----
    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g., manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, crit_separation, threshold, fwhm, psf_model, fitshape,
                 *, sigma=3.0, ratio=1.0, theta=0.0, sigma_radius=1.5,
                 sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                 fitter=LevMarLSQFitter(),
                 niters=3, aperture_radius=None, extra_output_cols=None,
                 subshape=None):

        self.crit_separation = crit_separation
        self.threshold = threshold
        self.fwhm = fwhm
        self.sigma = sigma
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi

        group_maker = DAOGroup(crit_separation=self.crit_separation)
        bkg_estimator = MMMBackground(sigma_clip=SigmaClip(sigma=self.sigma))
        finder = DAOStarFinder(threshold=self.threshold, fwhm=self.fwhm,
                               ratio=self.ratio, theta=self.theta,
                               sigma_radius=self.sigma_radius,
                               sharplo=self.sharplo, sharphi=self.sharphi,
                               roundlo=self.roundlo, roundhi=self.roundhi)

        super().__init__(group_maker=group_maker, bkg_estimator=bkg_estimator,
                         psf_model=psf_model, fitshape=fitshape,
                         finder=finder, fitter=fitter, niters=niters,
                         aperture_radius=aperture_radius,
                         extra_output_cols=extra_output_cols,
                         subshape=subshape)
