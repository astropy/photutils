# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines background classes to estimate a scalar background
and background RMS from an array (which may be masked) of any dimension.
These classes were designed as part of an object-oriented interface for
the tools in the PSF subpackage.
"""

import abc

import numpy as np
from astropy.stats import biweight_location, biweight_scale, mad_std

from ..utils.misc import _ABCMetaAndInheritDocstrings

from astropy.version import version as astropy_version
if astropy_version < '3.1':
    from astropy.stats import SigmaClip
    SIGMA_CLIP = SigmaClip(sigma=3., iters=10)
else:
    from ..extern import SigmaClip
    SIGMA_CLIP = SigmaClip(sigma=3., maxiters=10)


__all__ = ['BackgroundBase', 'BackgroundRMSBase', 'MeanBackground',
           'MedianBackground', 'ModeEstimatorBackground',
           'MMMBackground', 'SExtractorBackground',
           'BiweightLocationBackground', 'StdBackgroundRMS',
           'MADStdBackgroundRMS', 'BiweightScaleBackgroundRMS']


def _masked_median(data, axis=None):
    """
    Calculate the median of a (masked) array.

    This function is necessary for a consistent interface across all
    numpy versions.  A bug was introduced in numpy v1.10 where
    `numpy.ma.median` (with ``axis=None``) returns a single-valued
    `~numpy.ma.MaskedArray` if the input data is a `~numpy.ndarray` or
    if the data is a `~numpy.ma.MaskedArray`, but the mask is `False`
    everywhere.

    Parameters
    ----------
    data : array-like
        The input data.
    axis : int or `None`, optional
        The array axis along which the median is calculated.  If
        `None`, then the entire array is used.

    Returns
    -------
    result : float or `~numpy.ma.MaskedArray`
        The resulting median.  If ``axis`` is `None`, then a float is
        returned, otherwise a `~numpy.ma.MaskedArray` is returned.
    """

    _median = np.ma.median(data, axis=axis)
    if axis is None and np.ma.isMaskedArray(_median):
        _median = _median.item()

    return _median


class BackgroundBase(metaclass=_ABCMetaAndInheritDocstrings):
    """
    Base class for classes that estimate scalar background values.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        self.sigma_clip = sigma_clip

    def __call__(self, data, axis=None):
        return self.calc_background(data, axis=axis)

    @abc.abstractmethod
    def calc_background(self, data, axis=None):
        """
        Calculate the background value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background value.
        axis : int or `None`, optional
            The array axis along which the background is calculated.  If
            `None`, then the entire array is used.

        Returns
        -------
        result : float or `~numpy.ma.MaskedArray`
            The calculated background value.  If ``axis`` is `None` then
            a scalar will be returned, otherwise a
            `~numpy.ma.MaskedArray` will be returned.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')


class BackgroundRMSBase(metaclass=_ABCMetaAndInheritDocstrings):
    """
    Base class for classes that estimate scalar background RMS values.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        self.sigma_clip = sigma_clip

    def __call__(self, data, axis=None):
        return self.calc_background_rms(data, axis=axis)

    @abc.abstractmethod
    def calc_background_rms(self, data, axis=None):
        """
        Calculate the background RMS value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background RMS value.
        axis : int or `None`, optional
            The array axis along which the background RMS is calculated.
            If `None`, then the entire array is used.

        Returns
        -------
        result : float or `~numpy.ma.MaskedArray`
            The calculated background RMS value.  If ``axis`` is `None`
            then a scalar will be returned, otherwise a
            `~numpy.ma.MaskedArray` will be returned.
        """

        raise NotImplementedError('Needs to be implemented in a subclass.')


class MeanBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    mean.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import MeanBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = MeanBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return np.ma.mean(data, axis=axis)


class MedianBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    median.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import MedianBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = MedianBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return _masked_median(data, axis=axis)


class ModeEstimatorBackground(BackgroundBase):
    """
    Class to calculate the background in an array using a mode estimator
    of the form ``(median_factor * median) - (mean_factor * mean)``.

    Parameters
    ----------
    median_factor : float, optional
        The multiplicative factor for the data median.  Defaults to 3.
    mean_factor : float, optional
        The multiplicative factor for the data mean.  Defaults to 2.
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import ModeEstimatorBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = ModeEstimatorBackground(median_factor=3., mean_factor=2.,
    ...                               sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, median_factor=3., mean_factor=2., **kwargs):
        super().__init__(**kwargs)
        self.median_factor = median_factor
        self.mean_factor = mean_factor

    def calc_background(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)
        return ((self.median_factor * _masked_median(data, axis=axis)) -
                (self.mean_factor * np.ma.mean(data, axis=axis)))


class MMMBackground(ModeEstimatorBackground):
    """
    Class to calculate the background in an array using the DAOPHOT MMM
    algorithm.

    The background is calculated using a mode estimator of the form
    ``(3 * median) - (2 * mean)``.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import MMMBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = MMMBackground(sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `~photutils.background.core.ModeEstimatorBackground.calc_background`
    method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, **kwargs):
        kwargs['median_factor'] = 3.
        kwargs['mean_factor'] = 2.
        super().__init__(**kwargs)


class SExtractorBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the
    SExtractor algorithm.

    The background is calculated using a mode estimator of the form
    ``(2.5 * median) - (1.5 * mean)``.

    If ``(mean - median) / std > 0.3`` then the median is used instead.
    Despite what the `SExtractor`_ User's Manual says, this is the
    method it *always* uses.

    .. _SExtractor: http://www.astromatic.net/software/sextractor

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import SExtractorBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = SExtractorBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        _median = np.atleast_1d(_masked_median(data, axis=axis))
        _mean = np.atleast_1d(np.ma.mean(data, axis=axis))
        _std = np.atleast_1d(np.ma.std(data, axis=axis))
        bkg = np.atleast_1d((2.5 * _median) - (1.5 * _mean))

        bkg = np.ma.where(_std == 0, _mean, bkg)

        idx = np.ma.where(_std != 0)
        condition = (np.abs(_mean[idx] - _median[idx]) / _std[idx]) < 0.3
        bkg[idx] = np.ma.where(condition, bkg[idx], _median[idx])

        # np.ma.where always returns a masked array
        if axis is None and np.ma.isMaskedArray(bkg):
            bkg = bkg.item()

        return bkg


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
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import BiweightLocationBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkg = BiweightLocationBackground(sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)    # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, c=6, M=None, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return biweight_location(data, c=self.c, M=self.M, axis=axis)


class StdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as the
    (sigma-clipped) standard deviation.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import StdBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkgrms = StdBackgroundRMS(sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    28.86607004772212

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    28.86607004772212
    """

    def calc_background_rms(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return np.ma.std(data, axis=axis)


class MADStdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as using the
    `median absolute deviation (MAD)
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \\sigma \\approx \\frac{{\\textrm{{MAD}}}}{{\\Phi^{{-1}}(3/4)}}
            \\approx 1.4826 \\ \\textrm{{MAD}}

    where :math:`\\Phi^{{-1}}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import MADStdBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkgrms = MADStdBackgroundRMS(sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    37.06505546264005

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    37.06505546264005
    """

    def calc_background_rms(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return mad_std(data, axis=axis)


class BiweightScaleBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as the
    (sigma-clipped) biweight scale.

    Parameters
    ----------
    c : float, optional
        Tuning constant for the biweight estimator.  Default value is
        9.0.
    M : float, optional
        Initial guess for the biweight location.  Default value is
        `None`.
    sigma_clip : `astropy.stats.SigmaClip` object, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters.  If `None` then no sigma clipping will be
        performed.  The default is to perform sigma clipping with
        ``sigma=3.`` and ``maxiters=5``.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils import BiweightScaleBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.)
    >>> bkgrms = BiweightScaleBackgroundRMS(sigma_clip=sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    30.09433848589339

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)    # doctest: +FLOAT_CMP
    30.09433848589339
    """

    def __init__(self, c=9.0, M=None, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.M = M

    def calc_background_rms(self, data, axis=None):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis)

        return biweight_scale(data, c=self.c, M=self.M, axis=axis)
