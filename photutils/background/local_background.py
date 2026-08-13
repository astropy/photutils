# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for estimating local background using a circular annulus aperture.
"""

import astropy.units as u
import numpy as np

from photutils.aperture import ApertureStats, CircularAnnulus
from photutils.background.core import (BiweightLocationBackground,
                                       BiweightScaleBackgroundRMS,
                                       MADStdBackgroundRMS, MeanBackground,
                                       MedianBackground, MMMBackground,
                                       ModeEstimatorBackground,
                                       SExtractorBackground, StdBackgroundRMS)
from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._repr import make_repr

__all__ = ['LocalBackground']


class LocalBackground:
    """
    Class to compute a local background using a circular annulus
    aperture.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the circular annulus in pixels.

    outer_radius : float
        The outer radius of the circular annulus in pixels.

    bkg_estimator : callable, optional
        A callable object (a function or e.g., an instance of any
        `~photutils.background.BackgroundBase` subclass) used to
        estimate the background in each aperture. The callable object
        must take in a 1D `~numpy.ndarray` or `~numpy.ma.MaskedArray`.
        The default is an instance of
        `~photutils.background.MedianBackground` with sigma clipping
        (i.e., sigma-clipped median).

    n_threads : int, optional
        The number of threads to use to measure the local backgrounds.
        The default is 1 (no multithreading). The value is passed
        through to the internal `~photutils.aperture.ApertureStats`
        computation, which divides the annulus positions into chunks
        that are processed concurrently, producing results identical
        to the single-threaded computation. Multithreading is used
        only for the standard photutils estimator classes (see
        ``bkg_estimator``). A custom estimator callable uses the serial
        per-aperture loop.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.background import LocalBackground
    >>> data = np.ones((101, 101))
    >>> local_bkg = LocalBackground(5, 10)
    >>> bkg = local_bkg(data, 50, 50)
    >>> print(bkg)
    1.0

    >>> # Multiple positions
    >>> x = [30, 50, 70]
    >>> y = [30, 50, 70]
    >>> bkg = local_bkg(data, x, y)
    >>> print(bkg)
    [1. 1. 1.]
    """

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __init__(self, inner_radius, outer_radius, bkg_estimator=None, *,
                 n_threads=1):
        if inner_radius <= 0:
            msg = 'inner_radius must be positive.'
            raise ValueError(msg)
        if outer_radius <= 0:
            msg = 'outer_radius must be positive.'
            raise ValueError(msg)
        if outer_radius <= inner_radius:
            msg = 'outer_radius must be greater than inner_radius.'
            raise ValueError(msg)

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        if bkg_estimator is None:
            bkg_estimator = MedianBackground()
        self.bkg_estimator = bkg_estimator

        if not isinstance(n_threads, (int, np.integer)) or n_threads < 1:
            msg = 'n_threads must be a positive integer'
            raise ValueError(msg)
        self.n_threads = int(n_threads)

    def __repr__(self):
        params = ('inner_radius', 'outer_radius', 'bkg_estimator',
                  'n_threads')
        return make_repr(self, params)

    def _fast_estimator_spec(self):
        """
        The fast batched estimator specification, or `None`.

        Returns ``(kind, median_factor, mean_factor)`` when
        ``bkg_estimator`` is a standard photutils estimator
        class whose statistic can be computed with the batched
        `~photutils.aperture.ApertureStats` machinery, and `None`
        otherwise (in which case the per-aperture loop is used). The
        biweight-based estimators are supported only with their default
        tuning constants and location anchors, because the batched
        biweight kernel uses fixed conventions.
        """
        estimator = self.bkg_estimator
        median_factor = mean_factor = 0.0
        if type(estimator) is MeanBackground:
            kind = 'mean'
        elif type(estimator) is MedianBackground:
            kind = 'median'
        elif type(estimator) in (ModeEstimatorBackground, MMMBackground):
            kind = 'mode'
            median_factor = float(estimator.median_factor)
            mean_factor = float(estimator.mean_factor)
        elif type(estimator) is SExtractorBackground:
            kind = 'sextractor'
        elif type(estimator) is BiweightLocationBackground:
            if estimator.c != 6.0 or estimator.M is not None:
                return None
            kind = 'biweight_location'
        elif type(estimator) is StdBackgroundRMS:
            kind = 'std'
        elif type(estimator) is MADStdBackgroundRMS:
            kind = 'mad_std'
        elif type(estimator) is BiweightScaleBackgroundRMS:
            if estimator.c != 9.0 or estimator.M is not None:
                return None
            kind = 'biweight_scale'
        else:
            return None

        return (kind, median_factor, mean_factor)

    def _fast_background(self, data, apertures, mask, spec):
        """
        Measure the local backgrounds for all positions using the
        batched `~photutils.aperture.ApertureStats` machinery.

        This computes the same sigma-clipped statistic as calling the
        estimator on each annulus, but gathers, clips, and reduces the
        pixels of all apertures in compiled batch kernels instead of
        looping over the apertures in Python.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The data array (without units).

        apertures : `~photutils.aperture.CircularAnnulus`
            The annulus apertures.

        mask : 2D bool `~numpy.ndarray` or `None`
            The data mask.

        spec : tuple
            The estimator specification from `_fast_estimator_spec`.

        Returns
        -------
        result : 1D float `~numpy.ndarray`
            The local background values.
        """
        kind, median_factor, mean_factor = spec
        apstats = ApertureStats(data, apertures, mask=mask,
                                sigma_clip=self.bkg_estimator.sigma_clip,
                                sum_method='center',
                                n_threads=self.n_threads)

        if kind == 'mean':
            result = apstats.mean
        elif kind == 'median':
            result = apstats.median
        elif kind == 'mode':
            result = ((median_factor * apstats.median)
                      - (mean_factor * apstats.mean))
        elif kind == 'sextractor':
            mean = apstats.mean
            median = apstats.median
            std = apstats.std
            result = (2.5 * median) - (1.5 * mean)

            # Set the background to the mean where the std is zero
            mean_mask = std == 0
            result[mean_mask] = mean[mean_mask]

            # Set the background to the median when the absolute
            # difference between the mean and median divided by the
            # standard deviation is greater than or equal to 0.3
            with np.errstate(divide='ignore', invalid='ignore'):
                med_mask = (np.abs(mean - median) / std) >= 0.3
            med_mask = np.logical_and(med_mask, np.logical_not(mean_mask))
            result[med_mask] = median[med_mask]
        elif kind == 'biweight_location':
            result = apstats.biweight_location
        elif kind == 'std':
            result = apstats.std
        elif kind == 'mad_std':
            result = apstats.mad_std
        else:  # 'biweight_scale'
            with np.errstate(invalid='ignore'):
                result = np.sqrt(apstats.biweight_midvariance)

        return np.asarray(result, dtype=float)

    def to_aperture(self, x, y):
        """
        Return a `~photutils.aperture.CircularAnnulus` instance
        representing the local background annulus at the given
        positions.

        Parameters
        ----------
        x, y : float or 1D float `~numpy.ndarray`
            The aperture center (x, y) position(s) at which to create
            the annulus aperture.

        Returns
        -------
        apertures : `~photutils.aperture.CircularAnnulus` instance
            The circular annulus aperture(s) at the given position(s).

        Examples
        --------
        >>> from photutils.background import LocalBackground
        >>> local_bkg = LocalBackground(5, 10)
        >>> aperture = local_bkg.to_aperture(50, 50)
        >>> aperture
        <CircularAnnulus([[50., 50.]], r_in=5.0, r_out=10.0)>

        >>> # Multiple positions
        >>> aperture = local_bkg.to_aperture([30, 70], [40, 80])
        >>> aperture
        <CircularAnnulus([[30., 40.],
                          [70., 80.]], r_in=5.0, r_out=10.0)>
        >>> print(len(aperture.positions))
        2
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.shape != y.shape:
            msg = 'x and y must have the same shape'
            raise ValueError(msg)

        positions = np.column_stack((x, y))
        return CircularAnnulus(positions, self.inner_radius,
                               self.outer_radius)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, x, y, mask=None):
        """
        Measure the local background in a circular annulus.

        Parameters
        ----------
        data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
            The 2D array on which to measure the local background.

        x, y : float or 1D float `~numpy.ndarray`
            The aperture center (x, y) position(s) at which to measure
            the local background.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

        Returns
        -------
        value : float or 1D float `~numpy.ndarray`
            The local background values. If ``data`` is
            a `~astropy.units.Quantity`, the result is a
            `~astropy.units.Quantity` with the same unit. If all pixels
            in an annulus are masked or outside the data bounds, the
            corresponding value will be NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from photutils.background import LocalBackground
        >>> data = np.ones((101, 101))
        >>> local_bkg = LocalBackground(5, 10)
        >>> bkg = local_bkg(data, 50, 50)
        >>> print(bkg)
        1.0

        >>> # Multiple positions
        >>> bkg = local_bkg(data, [30, 50], [40, 60])
        >>> print(bkg)
        [1. 1.]

        >>> # Position outside data returns NaN
        >>> bkg = local_bkg(data, -50, -50)
        >>> print(np.isnan(bkg))
        True
        """
        unit = None
        if isinstance(data, u.Quantity):
            unit = data.unit
            data = data.value

        apertures = self.to_aperture(x, y)

        spec = self._fast_estimator_spec()
        if spec is not None:
            bkg = self._fast_background(data, apertures, mask, spec)
        else:
            apermasks = apertures.to_mask(method='center')
            n_apertures = len(apermasks)
            bkg = np.empty(n_apertures)
            for i, apermask in enumerate(apermasks):
                values = apermask.get_values(data, mask=mask)
                bkg[i] = self.bkg_estimator(values)

        if bkg.size == 1:
            bkg = bkg[0]
        if unit is not None:
            bkg = bkg << unit

        return bkg
