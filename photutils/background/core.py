# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate a scalar background
and background rms from an array (which may be masked) of any dimension.
These classes were designed as part of an object-oriented interface for
the tools in the PSF subpackage.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
import abc
import numpy as np
import warnings
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.stats import (sigma_clip, mad_std, biweight_location,
                           biweight_midvariance)

import astropy
if LooseVersion(astropy.__version__) < LooseVersion('1.1'):
    ASTROPY_LT_1P1 = True
else:
    ASTROPY_LT_1P1 = False


__all__ = ['BkgBase', 'BackgroundBase', 'BackgroundRMSBase',
           'MeanBackground', 'MedianBackground', 'MMMBackground',
           'SExtractorBackground', 'BiweightLocationBackground',
           'StdBackgroundRMS', 'MADStdBackgroundRMS',
           'BiweightMidvarianceBackgroundRMS']


class _ABCMetaAndInheritDocstrings(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(_ABCMetaAndInheritDocstrings)
class BkgBase(object):
    """
    Base class for Background and Background RMS classes.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, sigclip=True, sigma=3, sigma_lower=None,
                 sigma_upper=None, iters=5):

        self.sigclip = sigclip
        self.sigma = sigma
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.iters = iters

    def sigma_clip(self, data):

        if ASTROPY_LT_1P1:
            warnings.warn('sigma_lower and sigma_upper will be ignored '
                          'because they are not supported astropy < 1.1',
                          AstropyUserWarning)
            return sigma_clip(data, sig=self.sigma,
                              iters=self.iters)
        else:
            return sigma_clip(data, sigma=self.sigma,
                              sigma_lower=self.sigma_lower,
                              sigma_upper=self.sigma_upper,
                              iters=self.iters)


class BackgroundBase(BkgBase):
    """
    Base class for classes that estimate scalar background values.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    @abc.abstractmethod
    def calc_background(self, data):
        """
        Calculate the background value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background value.

        Returns
        -------
        result : float
            The calculated background value.
        """


class BackgroundRMSBase(BkgBase):
    """
    Base class for classes that estimate scalar background rms values.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    @abc.abstractmethod
    def calc_background_rms(self, data):
        """
        Calculate the background rms value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background rms value.

        Returns
        -------
        result : float
            The calculated background rms value.
        """


class MeanBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    mean.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
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

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, **kwargs):

        super(MedianBackground, self).__init__(**kwargs)

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

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, **kwargs):

        super(MMMBackground, self).__init__(**kwargs)

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

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, **kwargs):

        super(SExtractorBackground, self).__init__(**kwargs)

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)

        # Use .item() to make the median a scalar for numpy 1.10.
        # Even when fixed in numpy, this needs to remain for
        # compatibility with numpy 1.10 (until no longer supported).
        # https://github.com/numpy/numpy/pull/7635
        _median = np.ma.median(data).item()
        _mean = np.ma.mean(data)
        _std = np.ma.std(data)

        if _std == 0:
            return _mean

        if (np.abs(_mean - _median) / _std) < 0.3:
            return (2.5 * _median) - (1.5 * _mean)
        else:
            return _median


class BiweightLocationBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the biweight
    location.

    Parameters
    ----------
    c : float, optional
        Tuning constant for the biweight estimator.  Default value is
        6.0.
    M : float, optional
        Initial guess for the biweight location.  Default value is
        `None`.
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, c=6, M=None, **kwargs):

        super(BiweightLocationBackground, self).__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return biweight_location(data, c=self.c, M=self.M)


class StdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background rms in an array as the
    (sigma-clipped) standard deviation.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, **kwargs):

        super(StdBackgroundRMS, self).__init__(**kwargs)

    def calc_background_rms(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return np.ma.std(data)


class MADStdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background rms in an array as using the
    `median absolute deviation (MAD)
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \\sigma \\approx \\frac{{\\textrm{{MAD}}}}{{\Phi^{{-1}}(3/4)}}
            \\approx 1.4826 \ \\textrm{{MAD}}

    where :math:`\Phi^{{-1}}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.

    Parameters
    ----------
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, **kwargs):

        super(MADStdBackgroundRMS, self).__init__(**kwargs)

    def calc_background_rms(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return mad_std(data)


class BiweightMidvarianceBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background rms in an array as the
    (sigma-clipped) biweight midvariance.

    Parameters
    ----------
    c : float, optional
        Tuning constant for the biweight estimator.  Default value is
        9.0.
    M : float, optional
        Initial guess for the biweight location.  Default value is
        `None`.
    sigma : float, optional
        The number of standard deviations to use for both the lower and
        upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 3.
    sigma_lower : float or `None`, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    sigma_upper : float or `None`, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. Defaults to `None`.
    iters : int or `None`, optional
        The number of iterations to perform sigma clipping, or `None` to
        clip until convergence is achieved (i.e., continue until the
        last iteration clips nothing). Defaults to 5.
    """

    def __init__(self, c=9.0, M=None, **kwargs):

        super(BiweightMidvarianceBackgroundRMS, self).__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background_rms(self, data):

        if self.sigclip:
            data = self.sigma_clip(data)
        return biweight_midvariance(data, c=self.c, M=self.M)
