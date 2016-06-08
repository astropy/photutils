# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate a scalar background
and background rms from an array (which may be masked) of any dimension.
These classes were designed as part of an object-oriented interface for
the tools in the PSF subpackage.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import abc
import numpy as np
from astropy.stats import (sigma_clip, mad_std, biweight_location,
                           biweight_midvariance)


__all__ = ['MeanBackground', 'MedianBackground', 'MMMBackground',
           'SExtractorBackground', 'BiweightLocationBackground',
           'MADStdBackgroundRMS', 'BiweightMidvarianceBackgroundRMS']


class BackgroundBase(object):
    def __init__(self, sigclip=True, sigma=3, sigma_lower=None,
                 sigma_upper=None, iters=5):

        self.sigclip = sigclip
        self.sigma = sigma
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.iters = iters

    def sigma_clip(self, data):
        return sigma_clip(data, sigma=self.sigma,
                          sigma_lower=self.sigma_lower,
                          sigma_upper=self.sigma_upper,
                          iters=self.iters)

    @abc.abstractmethod
    def calc_background(self, data):
        """
        Calculate the background.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background.

        Returns
        -------
        result : float
            The calculated background.
        """


class MeanBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    mean.
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return np.ma.mean(data)


class MedianBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    median.
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return np.ma.median(data)


class MMMBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the DAOPHOT MMM
    algorithm.

    The background is calculated using a mode estimator of the form
    ``(3 * median) - (2 * mean)``.
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return (3. * np.ma.median(data)) - (2. * np.ma.mean(data))


class SExtractorBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the
    `SExtractor`_ algorithm.

    The background is calculated using a mode estimator of the form
    ``(2.5 * median) - (1.5 * mean)``.

    If ``(mean - median) / std > 0.3`` then the median is used instead.
    Despite what the `SExtractor`_ User's Manual says, this is the
    method it *always* uses.

    .. _SExtractor: http://www.astromatic.net/software/sextractor
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        _median = np.ma.median(data)
        _mean = np.ma.mean(data)
        _std = np.ma.std(data)
        condition = (np.abs(_mean - _median) / _std) < 0.3
        mode = (2.5 * _median) - (1.5 * _mean)
        bkg = np.ma.where(condition, mode, _median)
        bkg = np.ma.where(_std == 0, _mean, bkg)    # handle std = 0


class BiweightLocationBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the biweight
    location.
    """

    def __init__(self, c=6, M=None, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return biweight_location(c=self.c, M=self.M)


class StdBackgroundRMS(BackgroundBase):
    """
    Class to calculate the background rms in an array as the
    (sigma-clipped) standard deviation.
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return np.ma.std(data)


class MADStdBackgroundRMS(BackgroundBase):
    """
    Class to calculate the background rms in an array as using the
    `median absolute deviation (MAD)
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \\sigma \\approx \\frac{\\textrm{MAD}}{\Phi^{-1}(3/4)}
            \\approx 1.4826 \ \\textrm{MAD}

    where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.
    """

    def __init__(self, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return mad_std(data)


class BiweightMidvarianceBackgroundRMS(BackgroundBase):
    """
    Class to calculate the background rms in an array as the
    (sigma-clipped) biweight midvariance.
    """

    def __init__(self, c=9.0, M=None, **kwargs):

        super(MeanBackground, self).__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return biweight_midvariance(data, c=self.c, M=self.M)
