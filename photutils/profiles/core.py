# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Base class for profiles.
"""

import abc
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmax, nansum

__all__ = ['ProfileBase']


class ProfileBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for profile classes.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. The data should be background-subtracted.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D float `~numpy.ndarray`
        An array of radii defining the profile apertures. ``radii`` must
        be strictly increasing with a minimum value greater than or
        equal to zero, and contain at least 2 values. The radial spacing
        does not need to be constant. See the subclass documentation for
        details on how ``radii`` is interpreted.

    error : 2D `~numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    method : {'exact', 'center', 'subpixel'}, optional
        The method used to determine the overlap of the aperture on the
        pixel grid:

        * ``'exact'`` (default):
          The exact fractional overlap of the aperture and each pixel is
          calculated. The aperture weights will contain values between 0
          and 1.

        * ``'center'``:
          A pixel is considered to be entirely in or out of the aperture
          depending on whether its center is in or out of the aperture.
          The aperture weights will contain values only of 0 (out) and 1
          (in).

        * ``'subpixel'``:
          A pixel is divided into subpixels (see the ``subpixels``
          keyword), each of which are considered to be entirely in or
          out of the aperture depending on whether its center is in
          or out of the aperture. If ``subpixels=1``, this method is
          equivalent to ``'center'``. The aperture weights will contain
          values between 0 and 1.

    subpixels : int, optional
        For the ``'subpixel'`` method, resample pixels by this factor
        in each dimension. That is, each pixel is divided into
        ``subpixels**2`` subpixels. This keyword is ignored unless
        ``method='subpixel'``.
    """

    # Define axis labels used by `~photutils.profiles.ProfileBase.plot`.
    # Subclasses may override these.
    _xlabel = 'Radius (pixels)'
    _ylabel = 'Profile'

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))

        if error is not None and error.shape != data.shape:
            msg = 'error must have the same shape as data'
            raise ValueError(msg)

        self.data = data
        self.unit = unit
        self.xycen = xycen
        self.radii = self._validate_radii(radii)
        self.error = error
        self.mask = self._compute_mask(data, error, mask)
        self.method = method
        self.subpixels = subpixels
        self.normalization_value = 1.0

    def _validate_radii(self, radii):
        """Validate and return the radii array."""
        radii = np.array(radii)
        if radii.ndim != 1 or radii.size < 2:
            msg = 'radii must be a 1D array and have at least two values'
            raise ValueError(msg)
        if radii.min() < 0:
            msg = 'minimum radii must be >= 0'
            raise ValueError(msg)

        if not np.all(radii[1:] > radii[:-1]):
            msg = 'radii must be strictly increasing'
            raise ValueError(msg)

        return radii

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
                msg = 'mask must have the same shape as data'
                raise ValueError(msg)
            # Keep only non-finite values not already masked by the user
            badmask &= ~mask
            combined_mask = mask | badmask  # all masked pixels
        else:
            combined_mask = badmask

        if np.any(badmask):
            warnings.warn('Input data contains non-finite values (e.g., NaN '
                          'or inf) that were automatically masked.',
                          AstropyUserWarning)

        return combined_mask

    @property
    @abc.abstractmethod
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.
        """

    @property
    @abc.abstractmethod
    def profile(self):
        """
        The radial profile as a 1D `~numpy.ndarray`.
        """

    @property
    @abc.abstractmethod
    def profile_error(self):
        """
        The profile errors as a 1D `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """

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

    def _compute_photometry(self, apertures):
        """
        Compute aperture fluxes, flux errors, and areas for the given
        apertures.

        Parameters
        ----------
        apertures : list
            A list of aperture objects. Elements may be `None`, in which
            case the corresponding flux, error, and area are set to
            zero.

        Returns
        -------
        fluxes : `~numpy.ndarray`
            The aperture fluxes.

        fluxerrs : `~numpy.ndarray`
            The aperture flux errors.

        areas : `~numpy.ndarray`
            The aperture areas.
        """
        fluxes = []
        fluxerrs = []
        areas = []
        for aperture in apertures:
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

    @lazyproperty
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        radius.
        """
        return self._compute_photometry(self._circular_apertures)

    def normalize(self, method='max'):
        """
        Normalize the profile.

        Parameters
        ----------
        method : {'max', 'sum'}, optional
            The method used to normalize the profile:

            * ``'max'`` (default):
              The profile is normalized such that its maximum value is
              1.

            * ``'sum'``:
              The profile is normalized such that its sum (integral) is
              1.
        """
        if method == 'max':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                normalization = nanmax(self.profile)
        elif method == 'sum':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                normalization = nansum(self.profile)
        else:
            msg = 'invalid method, must be "max" or "sum"'
            raise ValueError(msg)

        if normalization == 0 or not np.isfinite(normalization):
            msg = ('The profile cannot be normalized because the max or '
                   'sum is zero or non-finite.')
            warnings.warn(msg, AstropyUserWarning)
        else:
            # normalization_values accumulate if normalize is run
            # multiple times (e.g., different methods)
            self.normalization_value *= normalization

            # Need to use __dict__ as these are lazy properties
            self.__dict__['profile'] = self.profile / normalization
            self.__dict__['profile_error'] = self.profile_error / normalization
            self._normalize_hook(normalization)

    def _normalize_hook(self, normalization):  # noqa: B027
        """
        Hook called by `normalize` after normalizing ``profile`` and
        ``profile_error``.

        This hook is only called when normalization succeeds (i.e., when
        the normalization value is non-zero and finite).

        Subclasses can override this to normalize additional lazy
        properties (e.g., ``data_profile``).

        Parameters
        ----------
        normalization : float
            The normalization value applied to the profile.
        """

    def unnormalize(self):
        """
        Unnormalize the profile back to the original state before any
        calls to `normalize`.
        """
        self.__dict__['profile'] = self.profile * self.normalization_value
        self.__dict__['profile_error'] = (self.profile_error
                                          * self.normalization_value)
        self._unnormalize_hook()
        self.normalization_value = 1.0

    def _unnormalize_hook(self):  # noqa: B027
        """
        Hook called by `unnormalize` after unnormalizing ``profile`` and
        ``profile_error``, but before resetting ``normalization_value``.

        Subclasses can override this to unnormalize additional lazy
        properties (e.g., ``data_profile``).
        """

    @staticmethod
    def _trim_to_monotonic(xarr, profile, name):
        """
        Trim arrays to the first monotonically increasing region.

        This is used by interpolation methods that require a
        monotonically increasing profile.

        Parameters
        ----------
        xarr : 1D `~numpy.ndarray`
            The x-axis values (e.g., radius or half-size).

        profile : 1D `~numpy.ndarray`
            The profile values.

        name : str
            A descriptive name for the profile used in the error
            message.

        Returns
        -------
        xarr, profile : tuple of `~numpy.ndarray`
            The trimmed arrays.
        """
        finite_mask = np.isfinite(profile)
        if not np.all(finite_mask):
            # Keep only the leading finite segment
            first_nonfinite = np.argmin(finite_mask)
            xarr = xarr[:first_nonfinite]
            profile = profile[:first_nonfinite]

        # np.diff produces an array of length n-1: diff[i] represents
        # the step from profile[i] to profile[i+1]. A value <= 0 means
        # the profile stopped increasing at that step.
        diff = np.diff(profile) <= 0
        if np.any(diff):
            # idx is an index into the *diff* array, not the profile
            # array. diff[idx] <= 0 means the drop occurs between
            # profile[idx] and profile[idx+1], so profile[idx] is
            # the last good value. We therefore need profile[:idx+1]
            # (inclusive) to retain it.
            idx = np.argmax(diff)  # first non-monotonic step in diff-space
            xarr = xarr[:idx + 1]
            profile = profile[:idx + 1]

        if len(xarr) < 2:
            msg = (f'The {name} profile is not monotonically '
                   'increasing even at the smallest values -- cannot '
                   'interpolate. Try using different input values '
                   '(especially the starting values) and/or using the '
                   '"exact" aperture overlap method.')
            raise ValueError(msg)

        return xarr, profile

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_radii = len(self.radii)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, n_radii={n_radii}, '
                f'normalized={normalized})')

    def plot(self, ax=None, **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        **kwargs : dict, optional
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
        ax.set_xlabel(self._xlabel)
        ylabel = self._ylabel
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
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.pyplot.fill_between`.

        Returns
        -------
        poly : `matplotlib.collections.PolyCollection` or `None`
            A `~matplotlib.collections.PolyCollection` containing the
            plotted polygons, or `None` if no errors were input.
        """
        if len(self.profile_error) == 0:
            warnings.warn('Errors were not input', AstropyUserWarning)
            return None

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        # Set default fill_between facecolor.
        # facecolor must be first key, otherwise it will override color
        # kwarg (i.e., cannot use setdefault here)
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
        return ax.fill_between(self.radius, ymin, ymax, **kws)
