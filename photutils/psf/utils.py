# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module which provides utility functions and classes for photometry"""

from __future__ import division

import math
import numpy as np

from photutils.psf.models import IntegratedGaussianPRF
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils import minversion
from ..aperture import CircularAperture, aperture_photometry
from ..extern.nddata_compat import extract_array


ASTROPY_LT_1_1 = not minversion('astropy', '1.1')

if ASTROPY_LT_1_1:
    from ..extern.nddata_compat import _overlap_slices_astropy1p1 as overlap_slices
else:
    from astropy.nddata.utils import overlap_slices


__all__ = ['PSFMeasure']


class PSFMeasure(object):
    """
    This class implements an algorithm to estimate the PSF FWHM by fitting
    2D Gaussians to the detected stars.
    """

    def __init__(self, fitshape, finder=None, bkg_estimator=None,
                 fitter=LevMarLSQFitter()):
        """
        Parameters
        ----------

        fitshape : int or length-2 array-like
            Rectangular shape around the center of a star which will be used
            to collect the data to do the fitting. Can be an integer to be
            the same along both axes. E.g., 5 is the same as (5, 5), which
            means to fit only at the following relative pixel positions:
            [-2, -1, 0, 1, 2].  Each element of ``fitshape`` must be an odd
            number.
        bkg_estimator : callable, instance of any `~photutils.BackgroundBase` subclass, or None
            ``bkg_estimator`` should be able to compute either a scalar
            background or a 2D background of a given 2D image. See, e.g.,
            `~photutils.background.MedianBackground`.  If None, no
            background subtraction is performed.
        """

        self.fitshape = fitshape
        self.finder = finder
        self.bkg_estimator = bkg_estimator
        self.fitter = fitter

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

    def __call__(self, image, positions=None):
        return self.compute_fwhm(image, positions)

    def compute_fwhm(self, image, positions=None):
        """
        Estimate the PSF FWHM by fitting a 2D Gaussians for every star given
        in ``positions`` or the ones detected in ``image``.

        Parameters
        ----------
        image : 2D array-like
            Image to estimate the PSF width.
        positions: `~astropy.table.Table`
            Positions (in pixel coordinates) at which to *start* the fit
            for each object.

        Return
        ------
        fwhm : 1D array-like
            Estimated fwhm of all stars in detected in ``image`` or given by
            ``positions``.
        """

        if self.bkg_estimator is not None:
            image = image - bkg_estimator(image)

        if self.finder is None and positions is None:
            raise ValueError("finder and positions cannot be None "
                             "simultaneously.")

        if positions is None:
            positions = self.finder(image)

        indices = np.indices(image.shape)
        fwhm = np.zeros(len(positions))
        for n, star in enumerate(positions):
            row = (star['ycentroid'], star['xcentroid'])

            y = extract_array(indices[0], self.fitshape, row)
            x = extract_array(indices[1], self.fitshape, row)

            sub_array_image = extract_array(image, self.fitshape, row)

            sigma_0 = self._sigma_init_guess(sub_array_image)

            self.fitshape = 2*int(3*sigma_0) + 1

            aperture = CircularAperture((star['xcentroid'], star['ycentroid']),
                                        r=3*sigma_0)

            flux_0 =  aperture_photometry(image, aperture)['aperture_sum']
            gaussianPRF = IntegratedGaussianPRF(flux=flux_0,
                                                x_0=star['xcentroid'],
                                                y_0=star['ycentroid'],
                                                sigma=sigma_0)
            gaussianPRF.sigma.fixed = False
            gaussian_fit = self.fitter(gaussianPRF, x, y, sub_array_image)
            fwhm[n] = gaussian_fit.sigma.value*gaussian_sigma_to_fwhm

        return fwhm

    def _sigma_init_guess(self, data):
        """
        Compute the initial guess for PSF width using the sample moments of
        the data.

        Parameters
        ----------
        data : 2D array-like
            Image data.

        Return
        ------
        sigma : float
            Initial guess for the width of the PSF.
        """

        total = data.sum()
        Y, X = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total

        marg_x = data[:, int(x)]
        marg_y = data[int(y), :]

        sigma_y = math.sqrt(np.abs((np.arange(marg_y.size) - y)**2*marg_y).sum()/marg_y.sum())
        sigma_x = math.sqrt(np.abs((np.arange(marg_x.size) - x)**2*marg_x).sum()/marg_x.sum())
        sigma = math.sqrt((sigma_x**2 + sigma_y**2)/2.0)

        return sigma
