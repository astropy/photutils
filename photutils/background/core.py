# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define classes to estimate the background and background RMS in an array
of any dimension.
"""

import abc
import warnings

import numpy as np
from astropy.stats import SigmaClip, mad_std

from photutils.extern.biweight import biweight_location, biweight_scale
from photutils.utils._parameters import (SigmaClipSentinelDefault,
                                         create_default_sigmaclip)
from photutils.utils._repr import make_repr
from photutils.utils._stats import nanmean, nanmedian, nanstd

__all__ = [
    'BackgroundBase',
    'BackgroundRMSBase',
    'BiweightLocationBackground',
    'BiweightScaleBackgroundRMS',
    'MADStdBackgroundRMS',
    'MMMBackground',
    'MeanBackground',
    'MedianBackground',
    'ModeEstimatorBackground',
    'SExtractorBackground',
    'StdBackgroundRMS',
]


SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)


class BackgroundBase(metaclass=abc.ABCMeta):
    """
    Base class for classes that estimate scalar background values.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)

        if not isinstance(sigma_clip, SigmaClip) and sigma_clip is not None:
            msg = 'sigma_clip must be an astropy SigmaClip instance or None'
            raise TypeError(msg)

        self.sigma_clip = sigma_clip

    def __repr__(self):
        return make_repr(self, ('sigma_clip',))

    def __call__(self, data, axis=None, masked=False):
        return self.calc_background(data, axis=axis, masked=masked)

    @abc.abstractmethod
    def calc_background(self, data, axis=None, masked=False):
        """
        Calculate the background value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background value.

        axis : int or `None`, optional
            The array axis along which the background is calculated. If
            `None`, then the entire array is used.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned. If
            `False`, then a `~numpy.ndarray` is returned, where masked
            values have a value of NaN. The default is `False`.

        Returns
        -------
        result : float, `~numpy.ndarray`, or `~numpy.ma.MaskedArray`
            The calculated background value. If ``masked`` is
            `False`, then a `~numpy.ndarray` is returned, otherwise a
            `~numpy.ma.MaskedArray` is returned. A scalar result is
            always returned as a float.
        """
        raise NotImplementedError  # pragma: no cover


class BackgroundRMSBase(metaclass=abc.ABCMeta):
    """
    Base class for classes that estimate scalar background RMS values.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)

        if not isinstance(sigma_clip, SigmaClip) and sigma_clip is not None:
            msg = 'sigma_clip must be an astropy SigmaClip instance or None'
            raise TypeError(msg)

        self.sigma_clip = sigma_clip

    def __repr__(self):
        return make_repr(self, ('sigma_clip',))

    def __call__(self, data, axis=None, masked=False):
        return self.calc_background_rms(data, axis=axis, masked=masked)

    @abc.abstractmethod
    def calc_background_rms(self, data, axis=None, masked=False):
        """
        Calculate the background RMS value.

        Parameters
        ----------
        data : array_like or `~numpy.ma.MaskedArray`
            The array for which to calculate the background RMS value.

        axis : int or `None`, optional
            The array axis along which the background RMS is calculated.
            If `None`, then the entire array is used.

        masked : bool, optional
            If `True`, then a `~numpy.ma.MaskedArray` is returned. If
            `False`, then a `~numpy.ndarray` is returned, where masked
            values have a value of NaN. The default is `False`.

        Returns
        -------
        result : float, `~numpy.ndarray`, or `~numpy.ma.MaskedArray`
            The calculated background RMS value. If ``masked`` is
            `False`, then a `~numpy.ndarray` is returned, otherwise a
            `~numpy.ma.MaskedArray` is returned. A scalar result is
            always returned as a float.
        """
        raise NotImplementedError  # pragma: no cover


class MeanBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    mean.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import MeanBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = MeanBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanmean(data, axis=axis)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class MedianBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    median.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import MedianBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = MedianBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanmedian(data, axis=axis)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class ModeEstimatorBackground(BackgroundBase):
    """
    Class to calculate the background in an array using a mode estimator
    of the form ``(median_factor * median) - (mean_factor * mean)``.

    Parameters
    ----------
    median_factor : float, optional
        The multiplicative factor for the median value. Defaults to 3.

    mean_factor : float, optional
        The multiplicative factor for the mean value. Defaults to 2.

    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import ModeEstimatorBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = ModeEstimatorBackground(median_factor=3.0, mean_factor=2.0,
    ...                               sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, median_factor=3.0, mean_factor=2.0,
                 sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.median_factor = median_factor
        self.mean_factor = mean_factor

    def __repr__(self):
        params = ('median_factor', 'mean_factor', 'sigma_clip')
        return make_repr(self, params)

    def calc_background(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = ((self.median_factor * nanmedian(data, axis=axis))
                      - (self.mean_factor * nanmean(data, axis=axis)))

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class MMMBackground(ModeEstimatorBackground):
    """
    Class to calculate the background in an array using the DAOPHOT MMM
    algorithm.

    The background is calculated using a mode estimator of the form
    ``(3 * median) - (2 * mean)``.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import MMMBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = MMMBackground(sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `~photutils.background.core.ModeEstimatorBackground.calc_background`
    method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        super().__init__(median_factor=3.0, mean_factor=2.0,
                         sigma_clip=sigma_clip)


class SExtractorBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the Source
    Extractor algorithm.

    The background is calculated using a mode estimator of the form
    ``(2.5 * median) - (1.5 * mean)``. If ``(mean - median) / std >
    0.3`` then the median is used instead.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import SExtractorBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = SExtractorBackground(sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def calc_background(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            _median = np.atleast_1d(nanmedian(data, axis=axis))
            _mean = np.atleast_1d(nanmean(data, axis=axis))
            _std = np.atleast_1d(nanstd(data, axis=axis))
            bkg = (2.5 * _median) - (1.5 * _mean)

            # set the background to the mean where the std is zero
            mean_mask = _std == 0
            bkg[mean_mask] = _mean[mean_mask]

            # set the background to the median when the absolute
            # difference between the mean and median divided by the
            # standard deviation is greater than or equal to 0.3

            med_mask = (np.abs(_mean - _median) / _std) >= 0.3
            mask = np.logical_and(med_mask, np.logical_not(mean_mask))
            bkg[mask] = _median[mask]

            # if bkg is a scalar, return it as a float
            if bkg.shape == (1,) and axis is None:
                bkg = bkg[0]

        if masked and isinstance(bkg, np.ndarray):
            bkg = np.ma.masked_where(np.isnan(bkg), bkg)

        return bkg


class BiweightLocationBackground(BackgroundBase):
    """
    Class to calculate the background in an array using the biweight
    location.

    Parameters
    ----------
    c : float, optional
        Tuning constant for the biweight estimator. Default value is
        6.0.

    M : float, optional
        Initial guess for the biweight location. Default value is
        `None`.

    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import BiweightLocationBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = BiweightLocationBackground(sigma_clip=sigma_clip)

    The background value can be calculated by using the
    `calc_background` method, e.g.:

    >>> bkg_value = bkg.calc_background(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5

    Alternatively, the background value can be calculated by calling the
    class instance as a function, e.g.:

    >>> bkg_value = bkg(data)
    >>> print(bkg_value)  # doctest: +FLOAT_CMP
    49.5
    """

    def __init__(self, c=6, M=None, sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.c = c
        self.M = M

    def __repr__(self):
        params = ('c', 'M', 'sigma_clip')
        return make_repr(self, params)

    def calc_background(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = biweight_location(data, c=self.c, M=self.M, axis=axis,
                                       ignore_nan=True)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class StdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as the (sigma-
    clipped) standard deviation.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import StdBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkgrms = StdBackgroundRMS(sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    28.86607004772212

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    28.86607004772212
    """

    def calc_background_rms(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanstd(data, axis=axis)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class MADStdBackgroundRMS(BackgroundRMSBase):
    r"""
    Class to calculate the background RMS in an array as using the
    `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The standard deviation estimator is given by:

    .. math::

        \sigma \approx \frac{{\textrm{{MAD}}}}{{\Phi^{{-1}}(3/4)}}
            \approx 1.4826 \ \textrm{{MAD}}

    where :math:`\Phi^{{-1}}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.

    Parameters
    ----------
    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import MADStdBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkgrms = MADStdBackgroundRMS(sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    37.06505546264005

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    37.06505546264005
    """

    def calc_background_rms(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = mad_std(data, axis=axis, ignore_nan=True)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result


class BiweightScaleBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as the (sigma-
    clipped) biweight scale.

    Parameters
    ----------
    c : float, optional
        Tuning constant for the biweight estimator. Default value is
        9.0.

    M : float, optional
        Initial guess for the biweight location. Default value is
        `None`.

    sigma_clip : `astropy.stats.SigmaClip` or `None`, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters. If `None` then no sigma clipping will be
        performed.

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import BiweightScaleBackgroundRMS
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkgrms = BiweightScaleBackgroundRMS(sigma_clip=sigma_clip)

    The background RMS value can be calculated by using the
    `calc_background_rms` method, e.g.:

    >>> bkgrms_value = bkgrms.calc_background_rms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    30.09433848589339

    Alternatively, the background RMS value can be calculated by calling
    the class instance as a function, e.g.:

    >>> bkgrms_value = bkgrms(data)
    >>> print(bkgrms_value)  # doctest: +FLOAT_CMP
    30.09433848589339
    """

    def __init__(self, c=9.0, M=None, sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.c = c
        self.M = M

    def __repr__(self):
        params = ('c', 'M', 'sigma_clip')
        return make_repr(self, params)

    def calc_background_rms(self, data, axis=None, masked=False):
        if self.sigma_clip is not None:
            data = self.sigma_clip(data, axis=axis, masked=False)
        elif isinstance(data, np.ma.MaskedArray):
            # convert to ndarray with masked values replaced by NaN
            data = data.filled(np.nan)

        # ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = biweight_scale(data, c=self.c, M=self.M, axis=axis,
                                    ignore_nan=True)

        if masked and isinstance(result, np.ndarray):
            result = np.ma.masked_where(np.isnan(result), result)

        return result
