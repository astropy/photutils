# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module which provides classes to perform PSF Photometry"""

from __future__ import division
import abc
import numpy as np
import astropy.extern.six
from astropy.table import Table
from astropy.table import vstack
from photutils.psf import subtract_psf


__all__ = ['PSFPhotometryBase', 'StetsonPSFPhotometry']


class PSFPhotometryBase(abc.ABCMeta):
    @abc.abstractmethod
    def do_photometry(self):
        pass

@six.add_metaclass(PSFPhotometry)
class StetsonPSFPhotometry(object):
    """
    This class implements the NSTAR algorithm proposed by Stetson
    (1987) to perform point spread function photometry in crowded fields.

    This class basically implements the loop FIND, GROUP, NSTAR,
    SUBTRACT, FIND until no more stars are detected.
    """

    def __init__(self, find, group, bkg, psf, fitter, niters, fitshape):
        """
        Attributes
        ----------
        find : an instance of any StarFinderBase subclasses
        group : an instance of any GroupStarsBase subclasses
        bkg : an instance of any BackgroundBase2D (?) subclasses
        psf : Fittable2DModel instance
        fitter : Fitter instance
        niters : int
            number of iterations to perform the loop FIND, GROUP, SUBTRACT,
            NSTAR.
        fitshape : array-like
            rectangular shape around the center of a star which will be used
            to collect the data to do the fitting, e.g. (5, 5), [5, 5],
            np.array([5, 5]). Also, each element must be an odd number.
        """

        self.find = find
        self.group = group
        self.bkg = bkg
        self.psf = psf
        self.fitter = fitter
        self.niters = niters
        self.fitshape = fitshape

        @property
        def niters(self):
            return self._niters
        @niters.setter
        def niters(self, niters):
            if isinstance(niters, int) and niters > 0:
                self._niters = niters
            else:
                raise ValueError('niters must be an interger-valued number, '
                                 'received niters = {}'.format(niters))
        @property
        def fitshape(self):
            return self._fitshape
        @fitshape.setter
        def fitshape(self, fitshape):
            fitshape = np.asarray(fitshape)
            if len(fitshape) == 2:
                if np.all(fitshape) > 0:
                    if np.all(fitshape) % 2 == 1):
                        self._fitshape = fitshape
                    else:
                        raise ValueError('fitshape must be odd integer-valued, '
                                         'received fitshape = {}'\
                                         .format(fitshape))
                else:
                    raise ValueError('fitshape must have positive elements, '
                                     'received fitshape = {}'\
                                     .format(fitshape))
            else:
                raise ValueError('fitshape must have two dimensions, '
                                 'received fitshape = {}'.format(fitshape))

    def __call__(self, image):
        """
        Parameters
        ----------
        image : array-like, ImageHDU, HDUList
            image to perform photometry
        
        Returns
        -------
        outtab : astropy.table.Table
            Table with the photometry results, i.e., centroids and flux
            estimations.
        residual_image : array-like, ImageHDU, HDUList
            Residual image calculated by subtracting the fitted sources
            and the original image.
        """

        return self.do_photometry(image)

    def do_photometry(self, image):
        # prepare output table
        outtab = Table([[], [], [], [], [], []],
                       names=('id', 'group_id', 'x_fit', 'y_fit', 'flux_fit',
                              'iter_detected'),
                       dtype=('i4', 'i4', 'f8', 'f8', 'f8', 'i4'))

        # make a copy of the input image
        residual_image = image.copy()

        # perform background subtraction
        residual_image = residual_image - self.bkg(image)

        # find potential sources on the given image
        sources = self.find(residual_image)

        n = 1
        # iterate until no more sources are found or the number of iterations
        # has been reached
        while(n <= self.niters and len(sources) > 0):
            # prepare input table
            intab = Table(names=['id', 'x_0', 'y_0', 'flux_0'],
                          data=[sources['id'], sources['xcentroid'],
                          sources['ycentroid'], sources['flux']])

            # find groups of overlapping sources
            star_groups = self.group(intab)

            # fit the sources within in each group in a simultaneous manner
            # and get the residual image
            tab, residual_image = self._nstar(residual_image, star_groups)

            # mark in which iteration those sources were fitted
            tab['iter_detected'] = n*np.ones(tab['x_fit'].shape, dtype=np.int)

            # populate output table
            outtab = vstack([outtab, tab])

            # find remaining sources in the residual image
            sources = self.find(residual_image)
            n += 1

        return outtab, residual_image

    def get_uncertainties(self):
        """
        Return the uncertainties on the fitted parameters
        """
        pass

    def nstar(self, image, star_groups):
        """
        Fit, as appropriate, a compound or single model to the given
        `star_groups`. Groups are fitted sequentially from the smallest to
        the biggest. In each iteration, `image` is subtracted by the previous
        fitted group. 
        
        Parameters
        ----------
        image : numpy.ndarray
            Background-subtracted image.
        star_groups : `~astropy.table.Table`

        Return
        ------
        result_tab : `~astropy.table.Table`
            Astropy table that contains the results of the photometry.
        image : numpy.ndarray
            Residual image.
        """

        result_tab = Table([[], [], [], [], []],
                           names=('id', 'group_id', 'x_fit', 'y_fit',
                                  'flux_fit'),
                           dtype=('i4', 'i4', 'f8', 'f8', 'f8'))

        star_groups = star_groups.group_by('group_id')
        
        groups_order = []
        for g in star_groups.groups:
            groups_order.append(len(g))

        N = len(groups_order)
        while N > 0:
            curr_order = np.min(groups_order)
            n = 0
            while(n < N):
                if curr_order == len(star_groups.groups[n]):
                    group_psf = self._get_sum_psf_model(star_groups.groups[n])
                    x, y, data = self._get_shape_and_data(shape=self.fitshape,\
                                             star_group=star_groups.groups[n],
                                                          image=image)
                    fit_model = self.fitter(group_psf, x, y, data)
                    param_table = self._model_params2table(fit_model,\
                                                        star_groups.groups[n])
                    result_tab = vstack([result_tab, param_table])
                    image = self.subtract_psf(image, x, y, fit_model)
                    N = N - 1
                n += 1
        return result_tab, image

    def _model_params2table(self, fit_model, star_group):
        """
        Place fitted parameters into an astropy table.
        
        Parameters
        ----------
        fit_model : Fittable2DModel
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
        if np.size(fit_model) == 1:
            param_tab.add_row([[star_group['id'][0]],
                               [star_group['group_id'][0]],
                               [getattr(fit_model,'x_0').value],
                               [getattr(fit_model, 'y_0').value],
                               [getattr(fit_model, 'flux').value]])
        else:
            for i in range(np.size(fit_model)):
                param_tab.add_row([[star_group['id'][i]],
                                   [star_group['group_id'][i]],
                                   [getattr(fit_model,'x_0_'+str(i)).value],
                                   [getattr(fit_model, 'y_0_'+str(i)).value],
                                   [getattr(fit_model, 'flux_'+str(i)).value]])
        return param_tab

    def _get_shape_and_data(self, shape, star_group, image):
        """
        Parameters
        ----------
        shape : tuple
            Shape of a rectangular region around the center of an isolated source.
        star_group : `astropy.table.Table`
            Group of stars
        image : numpy.ndarray

        Returns
        -------
        x, y : numpy.mgrid
            All coordinate pairs (x,y) in a rectangular region which encloses all
            sources of the given group
        image : numpy.ndarray
            Pixel value
        """

        xmin = int(np.around(np.min(star_group['x_0'])) - shape[0])
        xmax = int(np.around(np.max(star_group['x_0'])) + shape[0])
        ymin = int(np.around(np.min(star_group['y_0'])) - shape[1])
        ymax = int(np.around(np.max(star_group['y_0'])) + shape[1])
        y, x = np.mgrid[ymin:ymax+1, xmin:xmax+1]

        return x, y, image[ymin:ymax+1, xmin:xmax+1]

    def _get_sum_psf_model(self, star_group):
        """
        Construct a joint psf model which consists in a sum of `self.psf`
        whose parameters are given in `star_group`.

        Parameters
        ----------
        star_group : `~astropy.table.Table`
            Table from which the compound PSF will be constructed.
            It must have columns named as `x_0`, `y_0`, and `flux_0`.
        
        Returns
        -------
        sum_psf : CompoundModel
            `CompoundModel` instance which is a sum of the given PSF
            models.
        """

        psf_class = type(self.psf)
        sum_psf = psf_class(sigma=self.psf.sigma.value,
                            flux=star_group['flux_0'][0],
                            x_0=star_group['x_0'][0], y_0=star_group['y_0'][0])
        for i in range(len(star_group) - 1):
            sum_psf += psf_class(sigma=self.psf.sigma.value,
                                 flux=star_group['flux_0'][i+1],
                                 x_0=star_group['x_0'][i+1],
                                 y_0=star_group['y_0'][i+1])
        return sum_psf 
