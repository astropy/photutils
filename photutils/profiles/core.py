# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Base class for profiles.
"""

import abc
import warnings
from functools import cached_property

import astropy.units as u
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture._common import validate_mask_method
from photutils.aperture.core import _update_method_subpixels_docstring
from photutils.aperture.photometry import AperturePhotometry
from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._stats import nanmax, nansum

__all__ = ['ProfileBase']


@_update_method_subpixels_docstring
class ProfileBase(metaclass=abc.ABCMeta):
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Abstract base class for profile classes.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D data array. The data should be background-subtracted.
        Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D array_like of float
        An array of radii defining the profile apertures. ``radii`` must
        be strictly increasing with a minimum value greater than or
        equal to zero, and contain at least 2 values. The radial spacing
        does not need to be constant. ``radii`` must be in pixels and
        cannot be a `~astropy.units.Quantity`. See the subclass
        documentation for details on how ``radii`` is interpreted.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity`, then ``error`` must have the same
        units.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    <method_subpixels_descriptions>

    Raises
    ------
    TypeError
        If ``radii`` is a `~astropy.units.Quantity`.

    ValueError
        If ``data`` is not a 2D array, if ``radii`` is not a 1D array
        with at least two values, has a negative value, or is not
        strictly increasing, if ``error`` or ``mask`` does not have the
        same shape as ``data``, if ``data`` and ``error`` do not have
        the same units, or if ``method`` or ``subpixels`` is invalid.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If the ``data`` or ``error`` array contains nonfinite values
        that are not already masked by ``mask``.
    """

    # Define axis labels used by `~photutils.profiles.ProfileBase.plot`.
    # Subclasses may override these.
    _xlabel = 'Radius (pixels)'
    _ylabel = 'Profile'

    # The user-facing name of the ``radii`` parameter, used in
    # validation error messages. Subclasses that rename the parameter
    # (e.g., ``half_sizes``) may override this.
    _radii_name = 'radii'

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))

        if np.ndim(data) != 2:
            msg = 'data must be a 2D array'
            raise ValueError(msg)

        validate_mask_method(method, subpixels, method_name='method')

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

        # The total normalization applied to the profile. This is the
        # only mutable state of the class. All normalization-dependent
        # attributes (e.g., ``profile``) are derived from immutable
        # cached values divided by this value, and `normalize` and
        # `unnormalize` update it with a single atomic attribute
        # store, making the class safe for concurrent reads during
        # normalization changes.
        self.normalization_value = 1.0

    def _validate_radii(self, radii):
        """
        Validate and return the radii array.
        """
        name = self._radii_name
        if isinstance(radii, u.Quantity):
            msg = (f'{name} must be a plain array of pixel values, '
                   'not a Quantity')
            raise TypeError(msg)

        radii = np.array(radii)
        if radii.ndim != 1 or radii.size < 2:
            msg = f'{name} must be a 1D array and have at least two values'
            raise ValueError(msg)
        if radii.min() < 0:
            msg = f'minimum {name} must be >= 0'
            raise ValueError(msg)

        if not np.all(radii[1:] > radii[:-1]):
            msg = f'{name} must be strictly increasing'
            raise ValueError(msg)

        return radii

    def _compute_mask(self, data, error, mask):
        """
        Compute the mask array, automatically masking nonfinite data or
        error values.
        """
        badmask = ~np.isfinite(data)
        if error is not None:
            badmask |= ~np.isfinite(error)
        if mask is not None:
            if mask.shape != data.shape:
                msg = 'mask must have the same shape as data'
                raise ValueError(msg)
            # Keep only nonfinite values not already masked by the user
            badmask &= ~mask
            combined_mask = mask | badmask  # all masked pixels
        else:
            combined_mask = badmask

        if np.any(badmask):
            msg = ('Input data contains nonfinite values (e.g., NaN '
                   'or inf) that were automatically masked.')
            warnings.warn(msg, AstropyUserWarning)

        return combined_mask

    @property
    @abc.abstractmethod
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.
        """

    @property
    @abc.abstractmethod
    def _raw_profile(self):
        """
        The raw (unnormalized) profile as a 1D `~numpy.ndarray`, or a
        `~astropy.units.Quantity` if the input ``data`` has units.
        """

    @property
    @abc.abstractmethod
    def _raw_profile_error(self):
        """
        The raw (unnormalized) profile errors as a 1D `~numpy.ndarray`,
        or a `~astropy.units.Quantity` if the input ``data`` has units.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """

    @property
    def profile(self):
        """
        The profile as a 1D `~numpy.ndarray`, or a
        `~astropy.units.Quantity` if the input ``data`` has units.

        The returned values reflect the current profile normalization
        (see `normalize`). A normalized profile of data with units is a
        dimensionless `~astropy.units.Quantity`.
        """
        return self._raw_profile / self.normalization_value

    @property
    def profile_error(self):
        """
        The profile errors as a 1D `~numpy.ndarray`, or a
        `~astropy.units.Quantity` if the input ``data`` has units.

        The returned values reflect the current profile normalization
        (see `normalize`). If no ``error`` array was provided, an empty
        array with shape ``(0,)`` is returned.
        """
        return self._raw_profile_error / self.normalization_value

    @cached_property
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
        apertures using a single batched photometry call.

        Parameters
        ----------
        apertures : list
            A list of aperture objects sharing the same position.
            Leading elements may be `None` (e.g., for a zero radius), in
            which case the corresponding flux, error, and area are set
            to zero.

        Returns
        -------
        flux : `~numpy.ndarray` or `~astropy.units.Quantity`
            The aperture fluxes. A `~astropy.units.Quantity` if the
            input ``data`` has units.

        flux_err : `~numpy.ndarray` or `~astropy.units.Quantity`
            The aperture flux errors, or an empty array if no ``error``
            array was provided.

        areas : `~numpy.ndarray`
            The aperture areas in pixels squared, without units.
        """
        n_none = sum(aperture is None for aperture in apertures)
        result = AperturePhotometry(
            self.data, apertures[n_none:], error=self.error,
            mask=self.mask, method=self.method, subpixels=self.subpixels)

        fluxes = result.flux
        areas = result.area.to_value(u.pix ** 2)
        flux_errs = (result.flux_err if self.error is not None
                     else np.array([]))

        if n_none > 0:
            zeros = np.zeros(n_none)
            fluxes = np.concatenate((zeros, fluxes))
            areas = np.concatenate((zeros, areas))
            if self.error is not None:
                flux_errs = np.concatenate((zeros, flux_errs))

        if self.unit is not None:
            fluxes = fluxes << self.unit
            flux_errs = flux_errs << self.unit

        return fluxes, flux_errs, areas

    @cached_property
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        radius.
        """
        return self._compute_photometry(self._circular_apertures)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def normalize(self, method='max'):
        """
        Normalize the profile.

        The profile errors (`profile_error`) are divided by the same
        normalization value as the profile.

        The normalization is computed from the raw (unnormalized)
        profile values, so repeated calls do not accumulate. The
        most recent call determines the normalization. Because both
        normalization methods scale linearly with the profile values,
        this is equivalent to normalizing an already-normalized profile.
        NaN values in the profile are ignored when computing the maximum
        or sum.

        Parameters
        ----------
        method : {'max', 'sum'}, optional
            The method used to normalize the profile:

            * ``'max'`` (default):
              The profile is normalized such that its maximum value is
              1.

            * ``'sum'``:
              The profile is normalized such that the sum of its values
              is 1.

        Raises
        ------
        ValueError
            If ``method`` is not ``'max'`` or ``'sum'``.

        Warns
        -----
        `~astropy.utils.exceptions.AstropyUserWarning`
            If the maximum or sum of the profile is not positive or is
            nonfinite. In that case, the current normalization is left
            unchanged.
        """
        if method == 'max':
            func = nanmax
        elif method == 'sum':
            func = nansum
        else:
            msg = "invalid method, must be 'max' or 'sum'"
            raise ValueError(msg)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            normalization = func(self._raw_profile)

        if not (np.isfinite(normalization) and normalization > 0):
            msg = ('The profile cannot be normalized because the max or '
                   'sum is not positive or is nonfinite.')
            warnings.warn(msg, AstropyUserWarning)
        else:
            # A single atomic attribute store. Concurrent readers see
            # either the old or the new normalization, never a mixed
            # state.
            self.normalization_value = normalization

    def unnormalize(self):
        """
        Unnormalize the profile back to the original state before any
        calls to `normalize`.
        """
        self.normalization_value = 1.0

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_radii = len(self.radii)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, n_radii={n_radii}, '
                f'normalized={normalized})')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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

        profile = self.profile
        unit = None
        if isinstance(profile, u.Quantity):
            unit = profile.unit
            profile = profile.value

        lines = ax.plot(self.radius, profile, **kwargs)
        ax.set_xlabel(self._xlabel)
        ylabel = self._ylabel
        # A normalized profile is dimensionless, so the unit is
        # included only when the profile has a physical unit
        if unit is not None and unit != u.dimensionless_unscaled:
            ylabel = f'{ylabel} ({unit})'
        ax.set_ylabel(ylabel)

        return lines

    @deprecated_positional_kwargs(since='3.0', until='4.0')
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
            `matplotlib.pyplot.fill_between`. If ``facecolor`` is not
            given, a semi-transparent gray is used.

        Returns
        -------
        poly : `matplotlib.collections.PolyCollection` or `None`
            A `~matplotlib.collections.PolyCollection` containing the
            plotted polygons, or `None` if no errors were input.

        Warns
        -----
        `~astropy.utils.exceptions.AstropyUserWarning`
            If no errors were input. In that case, nothing is plotted
            and `None` is returned.
        """
        if len(self.profile_error) == 0:
            msg = 'Errors were not input'
            warnings.warn(msg, AstropyUserWarning)
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
        if isinstance(profile, u.Quantity):
            profile = profile.value
        if isinstance(profile_error, u.Quantity):
            profile_error = profile_error.value
        ymin = profile - profile_error
        ymax = profile + profile_error
        return ax.fill_between(self.radius, ymin, ymax, **kws)
