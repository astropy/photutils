# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module which provides classes to perform PSF Photometry"""

from __future__ import division
import numpy as np
import warnings

from astropy.table import Table, Column, vstack, hstack
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils.exceptions import AstropyUserWarning
from .funcs import subtract_psf, _extract_psf_fitting_names
from . import DAOGroup
from .models import get_grouped_psf_model
from ..aperture import CircularAperture, aperture_photometry
from ..background import MMMBackground, SigmaClip
from ..detection import DAOStarFinder
from ..extern.decorators import deprecated_renamed_argument
from astropy.utils import minversion

ASTROPY_LT_1_1 = not minversion('astropy', '1.1')

if ASTROPY_LT_1_1:
    from ..extern.nddata_compat import _overlap_slices_astropy1p1 as overlap_slices
else:
    from astropy.nddata.utils import overlap_slices


__all__ = ['BasicPSFPhotometry', 'IterativelySubtractedPSFPhotometry',
           'DAOPhotPSFPhotometry']


class BasicPSFPhotometry(object):
    """
    This class implements a PSF photometry algorithm that can find sources in an
    image, group overlapping sources into a single model, fit the model to the
    sources, and subtracting the models from the image. This is roughly
    equivalent to the DAOPHOT routines FIND, GROUP, NSTAR, and SUBTRACT.
    This implementation allows a flexible and customizable interface to
    perform photometry. For instance, one is able to use different
    implementations for grouping and finding sources by using ``group_maker``
    and ``finder`` respectivelly. In addition, sky background estimation is
    performed by ``bkg_estimator``.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given star
        overlaps with any other and label them as beloging to the same
        group.  ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``.  This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should cotain
        integers starting from ``1`` that indicate which group a given
        source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any `~photutils.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This object needs to identify three parameters (position
        of center in x and y coordinates and the flux) in order to set
        them to suitable starting values for each fit. The names of
        these parameters should be given as ``x_0``, ``y_0`` and
        ``flux``.  `~photutils.psf.prepare_psf_model` can be used to
        prepare any 2D model to match this assumption.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be used
        to collect the data to do the fitting. Can be an integer to be
        the same along both axes. E.g., 5 is the same as (5, 5), which
        means to fit only at the following relative pixel positions:
        [-2, -1, 0, 1, 2].  Each element of ``fitshape`` must be an odd
        number.
    finder : callable or instance of any `~photutils.detection.StarFinderBase` subclasses or None
        ``finder`` should be able to identify stars, i.e. compute a
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
    aperture_radius : float or None
        The radius (in units of pixels) used to compute initial
        estimates for the fluxes of sources. If ``None``, one FWHM will
        be used if it can be determined from the ```psf_model``.

    Notes
    -----
    Note that an ambiguity arises whenever ``finder`` and ``init_guesses``
    (keyword argument for ``do_photometry`)` are both not ``None``. In this
    case, ``finder`` is ignored and initial guesses are taken from
    ``init_guesses``. In addition, an warning is raised to remaind the user
    about this behavior.

    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g. manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape,
                 finder=None, fitter=LevMarLSQFitter(), aperture_radius=None):
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
                                     'received fitshape = {}'.format(value))
            else:
                raise ValueError('fitshape must have positive elements, '
                                 'received fitshape = {}'.format(value))
        else:
            raise ValueError('fitshape must have two dimensions, '
                             'received fitshape = {}'.format(value))

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
            raise ValueError('aperture_radius must be a real-valued '
                             'number, received aperture_radius = {}'
                             .format(value))

    def get_residual_image(self):
        """
        Returns an image that is the result of the subtraction between
        the original image and the fitted sources.

        Returns
        -------
        residual_image : 2D array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
        """

        return self._residual_image

    @deprecated_renamed_argument('positions', 'init_guesses', '0.4')
    def __call__(self, image, init_guesses=None):
        """
        Performs PSF photometry. See `do_photometry` for more details
        including the `__call__` signature.
        """

        return self.do_photometry(image, init_guesses)

    @deprecated_renamed_argument('positions', 'init_guesses', '0.4')
    def do_photometry(self, image, init_guesses=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this method
        uses ``init_guesses`` as initial guesses for the centroids. If
        the centroid positions are set as ``fixed`` in the PSF model
        ``psf_model``, then the optimizer will only consider the flux as
        a variable.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Image to perform photometry.
        init_guesses: `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the set
            of parameters. Columns 'x_0' and 'y_0' which represent the
            positions (in pixel coordinates) for each object must be present.
            'flux_0' can also be provided to set initial fluxes.
            If 'flux_0' is not provided, aperture photometry is used to
            estimate initial values for the fluxes. Additional columns of the
            form '<parametername>_0' will be used to set the initial guess for
            any parameters of the ``psf_model`` model that are not fixed.

        Returns
        -------
        output_tab : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process.
            None is returned if no sources are found in ``image``.
        """

        if self.bkg_estimator is not None:
            image = image - self.bkg_estimator(image)

        if self.aperture_radius is None:
            if hasattr(self.psf_model, 'fwhm'):
                self.aperture_radius = self.psf_model.fwhm.value
            elif hasattr(self.psf_model, 'sigma'):
                self.aperture_radius = (self.psf_model.sigma.value *
                                        gaussian_sigma_to_fwhm)

        if init_guesses is not None:
            # make sure the code does not modify user's input
            init_guesses = init_guesses.copy()

        if init_guesses is not None:
            if self.aperture_radius is None:
                if 'flux_0' not in init_guesses.colnames:
                    raise ValueError('aperture_radius is None and could not be '
                                     'determined by psf_model. Please, either '
                                     'provided a value for aperture_radius or '
                                     'define fwhm/sigma at psf_model.')

            if self.finder is not None:
                warnings.warn('Both init_guesses and finder are different than '
                              'None, which is ambiguous. finder is going to '
                              'be ignored.', AstropyUserWarning)

            if 'flux_0' not in init_guesses.colnames:
                apertures = CircularAperture((init_guesses['x_0'],
                                              init_guesses['y_0']),
                                             r=self.aperture_radius)

                init_guesses['flux_0'] = aperture_photometry(image,
                        apertures)['aperture_sum']
        else:
            if self.finder is None:
                raise ValueError('Finder cannot be None if init_guesses are '
                                 'not given.')
            sources = self.finder(image)
            if len(sources) > 0:
                apertures = CircularAperture((sources['xcentroid'],
                                              sources['ycentroid']),
                                             r=self.aperture_radius)

                sources['aperture_flux'] = aperture_photometry(image,
                        apertures)['aperture_sum']

                init_guesses = Table(names=['x_0', 'y_0', 'flux_0'],
                                  data=[sources['xcentroid'],
                                  sources['ycentroid'],
                                  sources['aperture_flux']])

        self._define_fit_param_names()
        for p0, param in self._pars_to_set.items():
            if p0 not in init_guesses.colnames:
                init_guesses[p0] = len(init_guesses) * [getattr(self.psf_model, param).value]

        star_groups = self.group_maker(init_guesses)
        output_tab, self._residual_image = self.nstar(image, star_groups)

        star_groups = star_groups.group_by('group_id')
        output_tab = hstack([star_groups, output_tab])

        return output_tab

    def nstar(self, image, star_groups):
        """
        Fit, as appropriate, a compound or single model to the given
        ``star_groups``. Groups are fitted sequentially from the
        smallest to the biggest. In each iteration, ``image`` is
        subtracted by the previous fitted group.

        Parameters
        ----------
        image : numpy.ndarray
            Background-subtracted image.
        star_groups : `~astropy.table.Table`
            This table must contain the following columns: ``id``,
            ``group_id``, ``x_0``, ``y_0``, ``flux_0``.  ``x_0`` and
            ``y_0`` are initial estimates of the centroids and
            ``flux_0`` is an initial estimate of the flux. Additionally,
            columns named as ``<param_name>_0`` are required if any other
            parameter in the psf model is free (i.e., the ``fixed``
            attribute of that parameter is ``False``).

        Returns
        -------
        result_tab : `~astropy.table.Table`
            Astropy table that contains photometry results.
        image : numpy.ndarray
            Residual image.
        """

        result_tab = Table()

        for param_tab_name in self._pars_to_output.keys():
            result_tab.add_column(Column(name=param_tab_name))

        y, x = np.indices(image.shape)

        star_groups = star_groups.group_by('group_id')
        for n in range(len(star_groups.groups)):
            group_psf = get_grouped_psf_model(self.psf_model, star_groups.groups[n],
                                              self._pars_to_set)
            usepixel = np.zeros_like(image, dtype=np.bool)

            for row in star_groups.groups[n]:
                usepixel[overlap_slices(large_array_shape=image.shape,
                                        small_array_shape=self.fitshape,
                                        position=(row['y_0'], row['x_0']),
                                        mode='trim')[0]] = True

            fit_model = self.fitter(group_psf, x[usepixel], y[usepixel],
                                    image[usepixel])
            param_table = self._model_params2table(fit_model,
                                                   star_groups.groups[n])
            result_tab = vstack([result_tab, param_table])

            try:
                from astropy.nddata.utils import NoOverlapError
            except ImportError:
                raise ImportError("astropy 1.1 or greater is required in "
                                  "order to use this class.")
            # do not subtract if the fitting did not go well
            try:
                image = subtract_psf(image, self.psf_model, param_table,
                                     subshape=self.fitshape)
            except NoOverlapError:
                pass

        return result_tab, image

    def _define_fit_param_names(self):
        """
        Convenience function to define mappings between the names of the
        columns in the initial guess table (and the name of the fitted
        parameters) and the actual name of the parameters in the model.

        This method sets the following parameters on the ``self`` object:
        * ``pars_to_set`` : Dict which maps the names of the parameters
          initial guesses to the actual name of the parameter in the model.
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

    def _model_params2table(self, fit_model, star_group):
        """
        Place fitted parameters into an astropy table.

        Parameters
        ----------
        fit_model : `astropy.modeling.Fittable2DModel` instance
            PSF or PRF model to fit the data. Could be one of the models
            in this package like `~photutils.psf.sandbox.DiscretePRF`,
            `~photutils.psf.IntegratedGaussianPRF`, or any other
            suitable 2D model.
        star_group : `~astropy.table.Table`

        Returns
        -------
        param_tab : `~astropy.table.Table`
            Table that contains the fitted parameters.
        """

        param_tab = Table()

        for param_tab_name in self._pars_to_output.keys():
            param_tab.add_column(Column(name=param_tab_name,
                                        data=np.empty(len(star_group))))

        if hasattr(fit_model, 'submodel_names'):
            for i in range(len(star_group)):
                for param_tab_name, param_name in self._pars_to_output.items():
                    param_tab[param_tab_name][i] = getattr(fit_model,
                                                           param_name + '_' + str(i)).value
        else:
            for param_tab_name, param_name in self._pars_to_output.items():
                param_tab[param_tab_name] = getattr(fit_model, param_name).value

        return param_tab


class IterativelySubtractedPSFPhotometry(BasicPSFPhotometry):
    """
    This class implements an iterative algorithm to perform point spread
    function photometry in crowded fields. This consists of applying a
    loop of find sources, make groups, fit groups, subtract groups, and then
    repeat until no more stars are detected or a given number of iterations is
    reached.

    Parameters
    ----------
    group_maker : callable or `~photutils.psf.GroupStarsBase`
        ``group_maker`` should be able to decide whether a given star
        overlaps with any other and label them as beloging to the same
        group.  ``group_maker`` receives as input an
        `~astropy.table.Table` object with columns named as ``id``,
        ``x_0``, ``y_0``, in which ``x_0`` and ``y_0`` have the same
        meaning of ``xcentroid`` and ``ycentroid``.  This callable must
        return an `~astropy.table.Table` with columns ``id``, ``x_0``,
        ``y_0``, and ``group_id``. The column ``group_id`` should cotain
        integers starting from ``1`` that indicate which group a given
        source belongs to. See, e.g., `~photutils.psf.DAOGroup`.
    bkg_estimator : callable, instance of any `~photutils.BackgroundBase` subclass, or None
        ``bkg_estimator`` should be able to compute either a scalar
        background or a 2D background of a given 2D image. See, e.g.,
        `~photutils.background.MedianBackground`.  If None, no
        background subtraction is performed.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This object needs to identify three parameters (position
        of center in x and y coordinates and the flux) in order to set
        them to suitable starting values for each fit. The names of
        these parameters should be given as ``x_0``, ``y_0`` and
        ``flux``.  `~photutils.psf.prepare_psf_model` can be used to
        prepare any 2D model to match this assumption.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be used
        to collect the data to do the fitting. Can be an integer to be
        the same along both axes. E.g., 5 is the same as (5, 5), which
        means to fit only at the following relative pixel positions:
        [-2, -1, 0, 1, 2].  Each element of ``fitshape`` must be an odd
        number.
    finder : callable or instance of any `~photutils.detection.StarFinderBase` subclasses
        ``finder`` should be able to identify stars, i.e. compute a
        rough estimate of the centroids, in a given 2D image.
        ``finder`` receives as input a 2D image and returns an
        `~astropy.table.Table` object which contains columns with names:
        ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
        ``id`` is an integer-valued column starting from ``1``,
        ``xcentroid`` and ``ycentroid`` are center position estimates of
        the sources and ``flux`` contains flux estimates of the sources.
        See, e.g., `~photutils.detection.DAOStarFinder` or `~photutils.detection.IRAFStarFinder`.
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

    Notes
    -----
    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g. manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
    """

    def __init__(self, group_maker, bkg_estimator, psf_model, fitshape,
                 finder, fitter=LevMarLSQFitter(), niters=3,
                 aperture_radius=None):

        super(IterativelySubtractedPSFPhotometry, self).__init__(group_maker,
                bkg_estimator, psf_model, fitshape, finder, fitter, aperture_radius)
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
                else:
                    self._niters = int(value)
            except:
                raise ValueError('niters must be None or an integer or '
                                 'convertable into an integer.')

    @property
    def finder(self):
        return self._finder

    @finder.setter
    def finder(self, value):
        if value is None:
            raise ValueError("finder cannot be None for "
                             "IterativelySubtractedPSFPhotometry - you may "
                             "want to use BasicPSFPhotometry. Please see the "
                             "Detection section on photutils documentation.")
        else:
            self._finder = value

    @deprecated_renamed_argument('positions', 'init_guesses', '0.4')
    def do_photometry(self, image, init_guesses=None):
        """
        Perform PSF photometry in ``image``.

        This method assumes that ``psf_model`` has centroids and flux
        parameters which will be fitted to the data provided in
        ``image``. A compound model, in fact a sum of ``psf_model``,
        will be fitted to groups of stars automatically identified by
        ``group_maker``. Also, ``image`` is not assumed to be background
        subtracted.  If ``init_guesses`` are not ``None`` then this method
        uses ``init_guesses`` as initial guesses for the centroids. If
        the centroid positions are set as ``fixed`` in the PSF model
        ``psf_model``, then the optimizer will only consider the flux as
        a variable.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Image to perform photometry.
        init_guesses: `~astropy.table.Table`
            Table which contains the initial guesses (estimates) for the set
            of parameters. Columns 'x_0' and 'y_0' which represent the
            positions (in pixel coordinates) for each object must be present.
            'flux_0' can also be provided to set initial fluxes.
            If 'flux_0' is not provided, aperture photometry is used to
            estimate initial values for the fluxes. Additional columns of the
            form '<parametername>_0' will be used to set the initial guess for
            any parameters of the ``psf_model`` model that are not fixed.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process.
            None is returned if no sources are found in ``image``.
        """

        if init_guesses is not None:
            table = super(IterativelySubtractedPSFPhotometry,
                self).do_photometry(image, init_guesses)
            table['iter_detected'] = np.ones(table['x_fit'].shape,
                                             dtype=np.int32)

            # n_start = 2 because it starts in the second iteration
            # since the first iteration is above
            output_table = self._do_photometry(init_guesses.colnames,
                                               n_start=2)
            output_table = vstack([table, output_table])
        else:
            if self.bkg_estimator is not None:
                self._residual_image = image - self.bkg_estimator(image)

            if self.aperture_radius is None:
                if hasattr(self.psf_model, 'fwhm'):
                    self.aperture_radius = self.psf_model.fwhm.value
                elif hasattr(self.psf_model, 'sigma'):
                    self.aperture_radius = (self.psf_model.sigma.value *
                                            gaussian_sigma_to_fwhm)

            output_table = self._do_photometry(['x_0', 'y_0', 'flux_0'])
        return output_table

    def _do_photometry(self, param_tab, n_start=1):
        """
        Helper function which performs the iterations of the photometry process.

        Parameters
        ----------
        param_names :  list
            Names of the columns which represent the initial guesses.
            For example, ['x_0', 'y_0', 'flux_0'], for intial guesses on the
            center positions and the flux.
        n_start : int
            Integer representing the start index of the iteration.
            It is 1 if init_guesses are None, and 2 otherwise.

        Returns
        -------
        output_table : `~astropy.table.Table` or None
            Table with the photometry results, i.e., centroids and
            fluxes estimations and the initial estimates used to start
            the fitting process.
            None is returned if no sources are found in ``image``.
        """

        output_table = Table()
        self._define_fit_param_names()

        for (init_param_name, fit_param_name) in zip(self._pars_to_set.keys(),
                                                     self._pars_to_output.keys()):
            output_table.add_column(Column(name=init_param_name))
            output_table.add_column(Column(name=fit_param_name))

        sources = self.finder(self._residual_image)

        n = n_start
        while(len(sources) > 0 and
              (self.niters is None or n <= self.niters)):
            apertures = CircularAperture((sources['xcentroid'],
                                          sources['ycentroid']),
                                          r=self.aperture_radius)
            sources['aperture_flux'] = aperture_photometry(self._residual_image,
                    apertures)['aperture_sum']

            init_guess_tab = Table(names=['id', 'x_0', 'y_0', 'flux_0'],
                               data=[sources['id'], sources['xcentroid'],
                               sources['ycentroid'],
                               sources['aperture_flux']])

            for param_tab_name, param_name in self._pars_to_set.items():
                if param_tab_name not in (['x_0', 'y_0', 'flux_0']):
                    init_guess_tab.add_column(Column(name=param_tab_name,
                            data=getattr(self.psf_model,
                                         param_name)*np.ones(len(sources))))

            star_groups = self.group_maker(init_guess_tab)
            table, self._residual_image = super(IterativelySubtractedPSFPhotometry,
                    self).nstar(self._residual_image, star_groups)

            star_groups = star_groups.group_by('group_id')
            table = hstack([star_groups, table])

            table['iter_detected'] = n*np.ones(table['x_fit'].shape, dtype=np.int32)

            output_table = vstack([output_table, table])
            sources = self.finder(self._residual_image)
            n += 1

        return output_table


class DAOPhotPSFPhotometry(IterativelySubtractedPSFPhotometry):
    """
    This class implements  an iterative algorithm based on the
    DAOPHOT algorithm presented by Stetson (1987) to perform point
    spread function photometry in crowded fields. This consists of
    applying a loop of find sources, make groups, fit groups, subtract
    groups, and then repeat until no more stars are detected or a given
    number of iterations is reached.

    Basically, this classes uses `~photutils.psf.IterativelySubtractedPSFPhotometry`,
    but with grouping, finding, and background estimation routines defined a
    priori. More precisely, this class uses `~photutils.psf.DAOGroup` for
    grouping, `~photutils.detection.DAOStarFinder` for finding sources, and
    `~photutils.background.MMMBackground` for background estimation. Those
    classes are based on GROUP, FIND, and SKY routines used in DAOPHOT,
    respectively.

    The parameter ``crit_separation`` is associated with `~photutils.psf.DAOGroup`.
    ``sigma_clip`` is associated with `~photutils.background.MMMBackground`.
    ``threshold`` and ``fwhm`` are associated with `~photutils.detection.DAOStarFinder`.
    Parameters from ``ratio`` to ``roundhi`` are also associated with
    `~photutils.detection.DAOStarFinder`.

    Parameters
    ----------
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.
    threshold : float
        The absolute image value above which to select sources.
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF or PRF model to fit the data. Could be one of the models in
        this package like `~photutils.psf.sandbox.DiscretePRF`,
        `~photutils.psf.IntegratedGaussianPRF`, or any other suitable 2D
        model.  This object needs to identify three parameters (position
        of center in x and y coordinates and the flux) in order to set
        them to suitable starting values for each fit. The names of
        these parameters should be given as ``x_0``, ``y_0`` and
        ``flux``.  `~photutils.psf.prepare_psf_model` can be used to
        prepare any 2D model to match this assumption.
    fitshape : int or length-2 array-like
        Rectangular shape around the center of a star which will be used
        to collect the data to do the fitting. Can be an integer to be
        the same along both axes. E.g., 5 is the same as (5, 5), which
        means to fit only at the following relative pixel positions:
        [-2, -1, 0, 1, 2].  Each element of ``fitshape`` must be an odd
        number.
    sigma : float, optional
        Number of standard deviations used to perform sigma clip with a
        `~photutils.SigmaClip` object.
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

    Notes
    -----
    If there are problems with fitting large groups, change the
    parameters of the grouping algorithm to reduce the number of sources
    in each group or input a ``star_groups`` table that only includes
    the groups that are relevant (e.g. manually remove all entries that
    coincide with artifacts).

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
    """

    def __init__(self, crit_separation, threshold, fwhm, psf_model,
                 fitshape, sigma=3., ratio=1.0, theta=0.0, sigma_radius=1.5, sharplo=0.2,
                 sharphi=1.0, roundlo=-1.0, roundhi=1.0, fitter=LevMarLSQFitter(),
                 niters=3, aperture_radius=None):

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
        finder = DAOStarFinder(threshold=self.threshold, fwhm=self.fwhm, ratio=self.ratio,
                               theta=self.theta, sigma_radius=self.sigma_radius,
                               sharplo=self.sharplo, sharphi=self.sharphi,
                               roundlo=self.roundlo, roundhi=self.roundhi)

        super(DAOPhotPSFPhotometry, self).__init__(group_maker=group_maker,
                                                   bkg_estimator=bkg_estimator,
                                                   psf_model=psf_model, fitshape=fitshape,
                                                   finder=finder, fitter=fitter, niters=niters,
                                                   aperture_radius=aperture_radius)
