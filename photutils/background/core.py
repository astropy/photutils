# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for estimating the background and background RMS in an array of
any dimension.
"""

import abc
import warnings

import numpy as np
from astropy.stats import SigmaClip, biweight_location, biweight_scale, mad_std

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

_SIGMA_CLIP_PARAM_DOC = (
    'sigma_clip : `astropy.stats.SigmaClip` or `None`, optional\n'
    '    A `~astropy.stats.SigmaClip` object that defines the sigma\n'
    '    clipping parameters. If `None` then no sigma clipping will be\n'
    '    performed.'
)


def _insert_sigma_clip_doc(cls):
    """
    Class decorator that replaces the ``<sigma_clip_param>`` placeholder
    in a class docstring with the shared ``sigma_clip`` parameter
    description.
    """
    cls.__doc__ = cls.__doc__.replace(
        '<sigma_clip_param>', _SIGMA_CLIP_PARAM_DOC,
    )
    return cls


def _validate_sigma_clip(sigma_clip):
    """
    Validate and generate the ``sigma_clip`` parameter.

    If ``sigma_clip`` is the sentinel default, a fresh
    `~astropy.stats.SigmaClip` instance is created from its stored
    parameters. `None` is accepted (meaning no sigma clipping). Any
    other value must be a `~astropy.stats.SigmaClip` instance.

    Parameters
    ----------
    sigma_clip : `~photutils.utils._parameters.SigmaClipSentinelDefault`,\
            `~astropy.stats.SigmaClip`, or `None`
        The value supplied to a base-class ``__init__``.

    Returns
    -------
    sigma_clip : `~astropy.stats.SigmaClip` or `None`
        A concrete `~astropy.stats.SigmaClip` instance, or `None`.
    """
    if sigma_clip is SIGMA_CLIP:
        return create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                        maxiters=SIGMA_CLIP.maxiters)
    if not isinstance(sigma_clip, SigmaClip) and sigma_clip is not None:
        msg = 'sigma_clip must be an astropy SigmaClip instance or None'
        raise TypeError(msg)

    return sigma_clip


def _prepare_data(sigma_clip, data, axis):
    """
    Prepare input data for a background estimation step.

    Applies sigma clipping when a `~astropy.stats.SigmaClip` instance is
    provided, or fills masked-array fill values with NaN when the input
    is a `~numpy.ma.MaskedArray` and sigma clipping is disabled.

    Parameters
    ----------
    sigma_clip : `~astropy.stats.SigmaClip` or `None`
        The sigma-clipping object to apply. If `None`, no clipping is
        performed.

    data : array_like or `~numpy.ma.MaskedArray`
        The input data array.

    axis : int, tuple of int, or `None`
        The axis along which sigma clipping is applied.

    Returns
    -------
    data : `~numpy.ndarray`
        The prepared data array, with masked or clipped values replaced
        by NaN.
    """
    if sigma_clip is not None:
        return sigma_clip(data, axis=axis, masked=False)

    if isinstance(data, np.ma.MaskedArray):
        # convert to ndarray with masked values replaced by NaN
        return data.filled(np.nan)

    return data


def _apply_masked(result, masked):
    """
    Optionally wrap NaN values in a masked array.

    Parameters
    ----------
    result : `~numpy.ndarray` or scalar
        The computed background or background RMS value(s).

    masked : bool
        If `True` and ``result`` is an `~numpy.ndarray`, return a
        `~numpy.ma.MaskedArray` with NaN values masked. Otherwise return
        ``result`` unchanged.

    Returns
    -------
    result : `~numpy.ndarray`, `~numpy.ma.MaskedArray`, or scalar
        The result, optionally wrapped as a masked array.
    """
    if masked and isinstance(result, np.ndarray):
        return np.ma.masked_where(np.isnan(result), result)
    return result


class _BackgroundCommonBase:
    """
    Internal mixin providing shared infrastructure for `BackgroundBase`
    and `BackgroundRMSBase`.

    This class is not part of the public API and should not be
    instantiated directly or subclassed outside of this module.
    """

    def __init__(self, sigma_clip=SIGMA_CLIP):
        self.sigma_clip = _validate_sigma_clip(sigma_clip)

    def __repr__(self):
        return make_repr(self, ('sigma_clip',))


class BackgroundBase(_BackgroundCommonBase, abc.ABC):
    """
    Base class for classes that estimate scalar background values.
    """

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


class BackgroundRMSBase(_BackgroundCommonBase, abc.ABC):
    """
    Base class for classes that estimate scalar background RMS values.
    """

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


@_insert_sigma_clip_doc
class MeanBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    mean.

    Parameters
    ----------
    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanmean(data, axis=axis)
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
class MedianBackground(BackgroundBase):
    """
    Class to calculate the background in an array as the (sigma-clipped)
    median.

    Parameters
    ----------
    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanmedian(data, axis=axis)
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
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

    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = ((self.median_factor * nanmedian(data, axis=axis))
                      - (self.mean_factor * nanmean(data, axis=axis)))
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
class MMMBackground(ModeEstimatorBackground):
    """
    Class to calculate the background in an array using the DAOPHOT MMM
    algorithm.

    The background is calculated using a mode estimator of the form
    ``(3 * median) - (2 * mean)``.

    Parameters
    ----------
    <sigma_clip_param>

    Examples
    --------
    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import MMMBackground
    >>> data = np.arange(100)
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg = MMMBackground(sigma_clip=sigma_clip)

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

    def __init__(self, sigma_clip=SIGMA_CLIP):
        super().__init__(median_factor=3.0, mean_factor=2.0,
                         sigma_clip=sigma_clip)


@_insert_sigma_clip_doc
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
    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            _median = np.atleast_1d(nanmedian(data, axis=axis))
            _mean = np.atleast_1d(nanmean(data, axis=axis))
            _std = np.atleast_1d(nanstd(data, axis=axis))
            result = (2.5 * _median) - (1.5 * _mean)

            # Set the background to the mean where the std is zero
            mean_mask = _std == 0
            result[mean_mask] = _mean[mean_mask]

            # Set the background to the median when the absolute
            # difference between the mean and median divided by the
            # standard deviation is greater than or equal to 0.3
            med_mask = (np.abs(_mean - _median) / _std) >= 0.3
            mask = np.logical_and(med_mask, np.logical_not(mean_mask))
            result[mask] = _median[mask]

            # If result is a scalar, return it as a float
            if result.shape == (1,) and axis is None:
                result = result[0]

        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
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

    <sigma_clip_param>

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

    def __init__(self, c=6.0, M=None, sigma_clip=SIGMA_CLIP):
        super().__init__(sigma_clip=sigma_clip)
        self.c = c
        self.M = M

    def __repr__(self):
        params = ('c', 'M', 'sigma_clip')
        return make_repr(self, params)

    def calc_background(self, data, axis=None, masked=False):
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = biweight_location(data, c=self.c, M=self.M, axis=axis,
                                       ignore_nan=True)
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
class StdBackgroundRMS(BackgroundRMSBase):
    """
    Class to calculate the background RMS in an array as the (sigma-
    clipped) standard deviation.

    Parameters
    ----------
    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = nanstd(data, axis=axis)
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
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
    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = mad_std(data, axis=axis, ignore_nan=True)
        return _apply_masked(result, masked)


@_insert_sigma_clip_doc
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

    <sigma_clip_param>

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
        data = _prepare_data(self.sigma_clip, data, axis)
        # Ignore RuntimeWarning where axis is all NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = biweight_scale(data, c=self.c, M=self.M, axis=axis,
                                    ignore_nan=True)
        return _apply_masked(result, masked)
