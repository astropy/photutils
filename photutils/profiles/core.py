# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides a base class for profiles.
"""

import abc
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._quantity_helpers import process_quantities

__all__ = ['ProfileBase']


class ProfileBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for profile classes.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        The 2D data array. The data should be background-subtracted.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D float `numpy.ndarray`
        An array of radii defining the edges of the radial bins.
        ``radii`` must be strictly increasing with a minimum value
        greater than or equal to zero, and contain at least 2 values.
        The radial spacing does not need to be constant. The output
        `radius` attribute will be defined at the bin centers.

    error : 2D `numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error
        of the sources (see `~photutils.utils.calc_total_error`) .
        ``error`` must have the same shape as the input ``data``.

    mask : 2D bool `numpy.ndarray`, optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on the
        pixel grid:

            * ``'exact'`` (default):
              The the exact fractional overlap of the aperture and each
              pixel is calculated. The aperture weights will contain
              values between 0 and 1.

            * ``'center'``:
              A pixel is considered to be entirely in or out of the
              aperture depending on whether its center is in or out of
              the aperture. The aperture weights will contain values
              only of 0 (out) and 1 (in).

            * ``'subpixel'``:
              A pixel is divided into subpixels (see the ``subpixels``
              keyword), each of which are considered to be entirely in
              or out of the aperture depending on whether its center is
              in or out of the aperture. If ``subpixels=1``, this method
              is equivalent to ``'center'``. The aperture weights will
              contain values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor
        in each dimension. That is, each pixel is divided into
        ``subpixels**2`` subpixels. This keyword is ignored unless
        ``method='subpixel'``.
    """

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))

        if error is not None and error.shape != data.shape:
            raise ValueError('error must have the same shape as data')

        self.data = data
        self.unit = unit
        self.xycen = xycen
        self.radii = self._validate_radii(radii)
        self.error = error
        self.mask = self._compute_mask(data, error, mask)
        self.method = method
        self.subpixels = subpixels

    def _validate_radii(self, edge_radii):
        edge_radii = np.array(edge_radii)
        if edge_radii.ndim != 1 or edge_radii.size < 2:
            raise ValueError('edge_radii must be a 1D array and have at '
                             'least two values')
        if edge_radii.min() < 0:
            raise ValueError('minimum edge_radii must be >= 0')

        if not np.all(edge_radii[1:] > edge_radii[:-1]):
            raise ValueError('edge_radii must be strictly increasing')

        return edge_radii

    def _compute_mask(self, data, error, mask):
        """
        Compute the mask array, automatically masking non-finite data or
        error values.
        """
        badmask = ~np.isfinite(data)
        if error is not None:
            badmask |= ~np.isfinite(error)
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask must have the same shape as data')
            badmask &= ~mask  # non-finite values not in input mask
            mask |= badmask  # all masked pixels
        else:
            mask = badmask

        if np.any(badmask):
            warnings.warn('Input data contains non-finite values (e.g., NaN '
                          'or inf) that were automatically masked.',
                          AstropyUserWarning)

        return mask

    @lazyproperty
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.
        """
        return self.radii

    @property
    @abc.abstractmethod
    def profile(self):
        """
        The radial profile as a 1D `~numpy.ndarray`.
        """
        raise NotImplementedError('Needs to be implemented in a subclass.')

    @property
    @abc.abstractmethod
    def profile_error(self):
        """
        The radial profile errors as a 1D `~numpy.ndarray`.
        """
        raise NotImplementedError('Needs to be implemented in a subclass.')

    @lazyproperty
    def _circular_apertures(self):
        """
        A list of `~photutils.aperture.CircularAperture` objects.

        The first element may be `None`.
        """
        from photutils.aperture import CircularAperture

        apertures = []
        for radius in self.radii:
            if radius <= 0.0:
                apertures.append(None)
            else:
                apertures.append(CircularAperture(self.xycen, radius))
        return apertures

    @lazyproperty
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        radius.
        """
        fluxes = []
        fluxerrs = []
        areas = []
        for aperture in self._circular_apertures:
            if aperture is None:
                flux, fluxerr = [0.0], [0.0]
                area = 0.0
            else:
                flux, fluxerr = aperture.do_photometry(
                    self.data, error=self.error, mask=self.mask,
                    method=self.method, subpixels=self.subpixels)
                area = aperture.area_overlap(self.data, mask=self.mask,
                                             method=self.method,
                                             subpixels=self.subpixels)
            fluxes.append(flux[0])
            if self.error is not None:
                fluxerrs.append(fluxerr[0])
            areas.append(area)

        fluxes = np.array(fluxes)
        fluxerrs = np.array(fluxerrs)
        areas = np.array(areas)
        if self.unit is not None:
            fluxes <<= self.unit
            fluxerrs <<= self.unit

        return fluxes, fluxerrs, areas

    def normalize(self, method='max'):
        """
        Normalize the profile.

        Parameters
        ----------
        method : {'max', 'sum'}, optional
            The method used to normalize the profile:

                * 'max' (default):
                  The profile is normalized such that its peak value is
                  1.
                * 'sum':
                  The profile is normalized such that its sum (integral)
                  is 1.
        """
        if method == 'max':
            normalization = np.nanmax(self.profile)
        elif method == 'sum':
            normalization = np.nansum(self.profile)
        else:
            raise ValueError('invalid method, must be "peak" or "integral"')

        if normalization == 0:
            warnings.warn('The profile cannot be normalized because the '
                          'max or sum is zero.', AstropyUserWarning)
        else:
            self.__dict__['profile'] = self.profile / normalization
            self.__dict__['profile_error'] = self.profile_error / normalization

    def plot(self, ax=None, **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        **kwargs : `dict`
            Any keyword arguments accepted by `matplotlib.pyplot.plot`.

        Returns
        -------
        lines : list of `~matplotlib.lines.Line2D`
            A list of lines representing the plotted data.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        lines = ax.plot(self.radius, self.profile, **kwargs)
        ax.set_xlabel('Radius (pixels)')
        ylabel = 'Profile'
        if self.unit is not None:
            ylabel = f'{ylabel} ({self.unit})'
        ax.set_ylabel(ylabel)

        return lines

    def plot_error(self, ax=None, **kwargs):
        """
        Plot the profile errors.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.pyplot.fill_between`.

        Returns
        -------
        lines : `matplotlib.collections.PolyCollection`
            A `~matplotlib.collections.PolyCollection` containing the
            plotted polygons.
        """
        if self.profile_error.shape == (0,):
            warnings.warn('Errors were not input', AstropyUserWarning)
            return None

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        # set default fill_between facecolor
        # facecolor must be first key, otherwise it will override color kwarg
        # (i.e., cannot use setdefault here)
        if 'facecolor' not in kwargs:
            kws = {'facecolor': (0.5, 0.5, 0.5, 0.3)}
            kws.update(kwargs)
        else:
            kws = kwargs

        profile = self.profile
        profile_error = self.profile_error
        if self.unit is not None:
            profile = profile.value
            profile_error = profile_error.value
        ymin = profile - profile_error
        ymax = profile + profile_error
        polycoll = ax.fill_between(self.radius, ymin, ymax, **kws)

        return polycoll
