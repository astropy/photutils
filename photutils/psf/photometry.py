# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module which provides classes to perform PSF Photometry"""

from __future__ import division
import numpy as np
from astropy.table import Table, vstack, hstack
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata.utils import overlap_slices
from astropy.stats import gaussian_sigma_to_fwhm
from ..psf import subtract_psf
from ..aperture import CircularAperture, aperture_photometry


__all__ = ['DAOPhotPSFPhotometry']


class DAOPhotPSFPhotometry(object):
    """
    This class implements the DAOPHOT algorithm proposed by Stetson
    (1987) to perform point spread function photometry in crowded fields. This
    consists of applying the loop FIND, GROUP, NSTAR, SUBTRACT, FIND until no
    more stars are detected or a given number of iterations is reached.
    """

    def __init__(self, grouper, bkg_estimator, psf_model, fitshape, finder=None,
                 fitter=LevMarLSQFitter(), niters=3, aperture_radius=None):
        """
        Parameters
        ----------
        grouper : callable or instance of any `~photutils.psf.GroupStarsBase` subclasses
            ``grouper`` should be able to decide whether a given star overlaps
            with any other and label them as beloging to the same group.
            ``grouper`` receives as input an `~astropy.table.Table` object with
            columns named as ``id``, ``x_0``, ``y_0``, in which ``x_0`` and
            ``y_0`` have the same meaning of ``xcentroid`` and ``ycentroid``.
            This callable must return an `~astropy.table.Table` with columns
            ``id``, ``x_0``, ``y_0``, and ``group_id``. The column
            ``group_id`` should cotain integers starting from ``1`` that
            indicate which group a given source belongs to. See, e.g.,
            `~photutils.psf.DAOGroup`.
        bkg_estimator : callable, instance of any `~photutils.BackgroundBase` subclass, or None
            ``bkg_estimator`` should be able to compute either a scalar
            background or a 2D background of a given 2D image. See, e.g.,
            `~photutils.background.MedianBackground`.  Can be None to do no
            background subtraction.
        psf_model : `astropy.modeling.Fittable2DModel` instance
            PSF or PRF model to fit the data. Could be one of the models in
            this package like `~photutils.psf.sandbox.DiscretePRF`,
            `~photutils.psf.IntegratedGaussianPRF`, or any other suitable
            2D model.
            This object needs to identify three parameters (position of
            center in x and y coordinates and the flux) in order to set them
            to suitable starting values for each fit. The names of these
            parameters should be given as ``x_0``, ``y_0`` and ``flux``.
            `~photutils.psf.prepare_psf_model` can be used to prepare any 2D
            model to match this assumption.
        fitshape : int or length-2 array-like
            Rectangular shape around the center of a star which will be used
            to collect the data to do the fitting. Can be an integer to be the
            same along both axes. E.g., 5 is the same as (5, 5), which means to
            fit only at the following relative pixel positions: [-2, -1, 0, 1, 2].
            Each element of ``fitshape`` must be an odd number.
        finder : callable or instance of any `~photutils.detection.StarFinderBase` subclasses
            ``finder`` should be able to identify stars, i.e. compute a rough
            estimate of the centroids, in a given 2D image.
            ``finder`` receives as input a 2D image an return an
            `~astropy.table.Table` object which contains columns with names:
            ``id``, ``xcentroid``, ``ycentroid``, and ``flux``. In which
            ``id`` is an integer-valued column starting from ``1``,
            ``xcentroid`` and ``ycentroid`` are center position estimates of
            the sources and ``flux`` contains flux estimates of the sources.
            See, e.g., `~photutils.detection.DAOStarFinder`.
        fitter : `~astropy.modeling.fitting.Fitter` instance
            Fitter object used to compute the optimized centroid positions
            and/or flux of the identified sources. See
            `~astropy.modeling.fitting` for more details on fitters.
        niters : int or None
            Number of iterations to perform of the loop FIND, GROUP, SUBTRACT,
            NSTAR. If None, iterations will proceed until no more stars remain.
            Note that in this case it is *possible* that the loop will never
            end if the PSF has structure that causes subtraction to create new
            sources infinitely.
        aperture_radius : float
            The radius (in units of pixels) used to compute initial estimates
            for the fluxes of sources. If ``None``, one FWHM will be used if it
            can be determined from the ```psf_model``.

        Notes
        -----
        If there are problems with fitting large groups, change the parameters
        of the grouping algorithm to reduce the number of sources in each
        group or input a ``star_groups`` table that only includes the groups
        that are relevant (e.g. manually remove all entries that coincide with
        artifacts).

        References
        ----------
        [1] Stetson, Astronomical Society of the Pacific, Publications,
            (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
            Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
        """

        self.finder = finder
        self.grouper = grouper
        self.bkg_estimator = bkg_estimator
        self.psf_model = psf_model
        self.fitter = fitter
        self.niters = niters
        self.fitshape = fitshape
        self.aperture_radius = aperture_radius

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
                raise ValueError('niters must be None or an integer or convertable '
                                 'into an integer.')

    @property
    def fitshape(self):
        return self._fitshape

    @fitshape.setter
    def fitshape(self, value):
        value = np.asarray(value)

        # assume a lone value should mean both axes
        if value.shape == ():
            value = (value, value)

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

    def __call__(self, image, positions=None):
        """
        Performs PSF photometry using the DAOPHOT algorithm.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Image to perform photometry.
        positions : `~astropy.table.Table` (optional)
            Positions, in pixel coordinates, at which stars are located.
            The columns must be named as 'x_0' and 'y_0'. 'flux_0' can also
            be provided to set initial fluxes.

        Returns
        -------
        outtab : `~astropy.table.Table`
            Table with the photometry results, i.e., centroids and fluxes
            estimations.
        residual_image : array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Residual image calculated by subtracting the fitted sources
            and the original image.
        """

        return self.do_photometry(image, positions)

    def do_photometry(self, image, positions=None):
        """
        Perform PSF photometry in ``image``. This method assumes that
        ``psf_model`` has centroids and flux parameters which will be fitted to
        the data provided in ``image``. A compound model, in fact a sum of
        ``psf_model``, will be fitted to groups of stars automatically
        identified by ``grouper``. Also, ``image`` is not assumed to be
        background subtracted.
        If positions are not ``None`` then this method performs forced PSF
        photometry, i.e., the positions are assumed to be known with high
        accuracy and only fluxes are fitted. If the centroid positions are
        set as ``fixed`` in the PSF model ``psf_model``, then the optimizer will
        only consider the flux as a variable. Otherwise, ``positions`` will be
        used as initial guesses for the centroids.

        Parameters
        ----------
        image : 2D array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Image to perform photometry.
        positions: `~astropy.table.Table`
            Positions (in pixel coordinates) at which to *start* the fit for
            each object. Columns 'x_0' and 'y_0' must be present.
            'flux_0' can also be provided to set initial fluxes.

        Returns
        -------
        outtab : `~astropy.table.Table`
            Table with the photometry results, i.e., centroids and fluxes
            estimations and the initial estimates used to start the fitting
            process.
        residual_image : array-like, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`
            Residual image calculated by subtracting the fitted sources
            and the original image.
        """

        if self.bkg_estimator is None:
            residual_image = image.copy()
        else:
            residual_image = image - self.bkg_estimator(image)

        if self.aperture_radius is None:
            if hasattr(self.psf_model, 'fwhm'):
                self.aperture_radius = self.psf_model.fwhm.value
            elif hasattr(self.psf_model, 'sigma'):
                self.aperture_radius = self.psf_model.sigma.value * gaussian_sigma_to_fwhm

        if positions is None:
            outtab = Table([[], [], [], [], [], []],
                           names=('id', 'group_id', 'iter_detected', 'x_fit',
                                  'y_fit', 'flux_fit'),
                           dtype=('i4', 'i4', 'i4', 'f8', 'f8', 'f8'))

            intab = Table([[], [], []],
                          names=('x_0', 'y_0', 'flux_0'),
                          dtype=('f8', 'f8', 'f8'))

            sources = self.finder(residual_image)

            apertures = CircularAperture((sources['xcentroid'],
                                          sources['ycentroid']),
                                         r=self.aperture_radius)

            sources['aperture_flux'] = aperture_photometry(residual_image, apertures)['aperture_sum']
            n = 1
            while(len(sources) > 0 and (self.niters is not None and n <= self.niters)):
                init_guess_tab = Table(names=['x_0', 'y_0', 'flux_0'],
                                       data=[sources['xcentroid'],
                                             sources['ycentroid'],
                                             sources['aperture_flux']])
                intab = vstack([intab, init_guess_tab])

                star_groups = self.grouper(init_guess_tab)

                result_tab, residual_image = self.nstar(residual_image,
                                                        star_groups)
                result_tab['iter_detected'] = n*np.ones(result_tab['x_fit'].shape,
                                                        dtype=np.int)

                outtab = vstack([outtab, result_tab])

                sources = self.finder(residual_image)

                if len(sources) > 0:
                    apertures = CircularAperture((sources['xcentroid'],
                                                  sources['ycentroid']),
                                                 r=self.aperture_radius)
                    sources['aperture_flux'] = aperture_photometry(residual_image, apertures)['aperture_sum']
                n += 1

            outtab = hstack([intab, outtab])
        else:
            if 'flux_0' not in positions.colnames:
                apertures = CircularAperture((positions['x_0'],
                                              positions['y_0']),
                                             r=self.aperture_radius)

                positions['flux_0'] = aperture_photometry(residual_image, apertures)['aperture_sum']

            intab = Table(names=['x_0', 'y_0', 'flux_0'],
                          data=[positions['x_0'], positions['y_0'],
                          positions['flux_0']])

            star_groups = self.grouper(intab)
            outtab, residual_image = self.nstar(residual_image, star_groups)
            outtab = hstack([intab, outtab])

        return outtab, residual_image

    def get_uncertainties(self):
        """
        Return the uncertainties on the fitted parameters
        """

        raise NotImplementedError

    def nstar(self, image, star_groups):
        """
        Fit, as appropriate, a compound or single model to the given
        ``star_groups``. Groups are fitted sequentially from the smallest to
        the biggest. In each iteration, ``image`` is subtracted by the
        previous fitted group.

        Parameters
        ----------
        image : numpy.ndarray
            Background-subtracted image.
        star_groups : `~astropy.table.Table`
            This table must contain the following columns: ``id``,
            ``group_id``, ``x_0``, ``y_0``, ``flux_0``.
            ``x_0`` and ``y_0`` are initial estimates of the centroids
            and ``flux_0`` is an initial estimate of the flux.

        Returns
        -------
        result_tab : `~astropy.table.Table`
            Astropy table that contains photometry results.
        image : numpy.ndarray
            Residual image.
        """

        result_tab = Table([[], [], [], [], []],
                           names=('id', 'group_id', 'x_fit', 'y_fit',
                                  'flux_fit'),
                           dtype=('i4', 'i4', 'f8', 'f8', 'f8'))
        star_groups = star_groups.group_by('group_id')

        y, x = np.indices(image.shape)

        for n in range(len(star_groups.groups)):
            group_psf = self.GroupPSF(self.psf_model,
                                      star_groups.groups[n]).get_model()
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
                raise ImportError("astropy 1.2.1 is required in order to use"
                                  "this class.")
            # do not subtract if the fitting did not go well
            try:
                image = subtract_psf(image, self.psf_model, param_table,
                                     subshape=self.fitshape)
            except NoOverlapError:
                pass

        return result_tab, image

    def _model_params2table(self, fit_model, star_group):
        """
        Place fitted parameters into an astropy table.

        Parameters
        ----------
        fit_model : `astropy.modeling.Fittable2DModel` instance
            PSF or PRF model to fit the data. Could be one of the models in
            this package like `~photutils.psf.sandbox.DiscretePRF`,
            `~photutils.psf.IntegratedGaussianPRF`, or any other suitable
            2D model.
        star_group : ~astropy.table.Table

        Returns
        -------
        param_tab : ~astropy.table.Table
            Table that contains the fitted parameters.
        """

        param_tab = Table([[], [], [], [], []],
                          names=('id', 'group_id', 'x_fit', 'y_fit',
                                 'flux_fit'),
                          dtype=('i4', 'i4', 'f8', 'f8', 'f8'))

        if hasattr(fit_model, 'submodel_names'):
            for i in range(len(fit_model.submodel_names)):
                param_tab.add_row([[star_group['id'][i]],
                                   [star_group['group_id'][i]],
                                   [getattr(fit_model, 'x_0_'+str(i)).value],
                                   [getattr(fit_model, 'y_0_'+str(i)).value],
                                   [getattr(fit_model, 'flux_'+str(i)).value]])
        else:
            param_tab.add_row([[star_group['id'][0]],
                               [star_group['group_id'][0]],
                               [getattr(fit_model, 'x_0').value],
                               [getattr(fit_model, 'y_0').value],
                               [getattr(fit_model, 'flux').value]])

        return param_tab

    class GroupPSF(object):
        """
        Construct a joint PSF model which consists of a sum of `self.psf_model`
        whose parameters are given in `star_group`.

        Attributes
        ----------
        star_group : `~astropy.table.Table`
            Table from which the compound PSF will be constructed.
            It must have columns named as `x_0`, `y_0`, and `flux_0`.
        psf_model : `astropy.modeling.Fittable2DModel` instance
        """

        def __init__(self, psf_model, star_group):
            self.star_group = star_group
            self.psf_model = psf_model

        def get_model(self):
            """
            Returns
            -------
            group_psf : CompoundModel
                `CompoundModel` instance which is a sum of the given PSF
                models.
            """

            group_psf = None
            for i in range(len(self.star_group)):
                psf_to_add = self.psf_model.copy()
                psf_to_add.flux = self.star_group['flux_0'][i+1]
                psf_to_add.x_0 = self.star_group['x_0'][i+1]
                psf_to_add.y_0 = self.star_group['y_0'][i+1]

                if group_psf is None:
                    # this is the first one only
                    group_psf = psf_to_add
                else:
                    group_psf += psf_to_add

            return group_psf
