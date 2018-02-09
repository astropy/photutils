# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build an ePSF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import SigmaClip


__all__ = ['EPSFFitter', 'EPSFBuilder']


class EPSFFitter(object):
    def __init__(self, psf, psf_fit_box=5, fitter=LevMarLSQFitter,
                 residuals=False, **kwargs):
        self.psf = psf
        self.psf_fit_box = psf_fit_box
        self.fitter = fitter
        self.residuals = residuals
        self.fitter_kwargs = kwargs

    def __call__(self, data, psf, star_table):
        return self.fit_psf(data, psf, star_table)

    def fit_psf(self, data, psf, star_table):
        pass


class EPSFBuilder(object):
    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 ignore_badfit_stars=True, stat='median',
                 sigma_clip=SigmaClip(sigma=3., iters=10),
                 smoothing_kernel='quar', fitter=EPSFFitter(residuals=True),
                 max_iters=50, accuracy=1e-4, epsf=None):

        self.peak_fit_box = peak_fit_box
        self.peak_search_box = peak_search_box
        self.recenter_accuracy = recenter_accuracy
        self.recenter_max_iters = recenter_max_iters
        self.ignore_badfit_stars = ignore_badfit_stars
        self.stat = stat
        self.sigma_clip = sigma_clip
        self.smoothing_kernel = smoothing_kernel
        self.fitter = fitter
        self.max_iters = max_iters
        self.accuracy = accuracy
        self.epsf = epsf

    def __call__(self, data, star_table):
        return self.build_psf(data, star_table)

    def build_psf(self, data, star_table):
        pass
