# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for generating curves of growth.
"""

import abc
from functools import cached_property

import astropy.units as u
import numpy as np
from scipy.interpolate import PchipInterpolator

from photutils.aperture.core import _update_method_subpixels_docstring
from photutils.profiles.core import ProfileBase

__all__ = ['CurveOfGrowth', 'EllipticalCurveOfGrowth',
           'EnsquaredCurveOfGrowth']


class _CurveOfGrowthBase(ProfileBase):
    """
    Base class for curve-of-growth profile classes.

    Curve-of-growth profiles measure the cumulative flux directly
    within the apertures defined by the ``apertures`` property.
    """

    # The name of the size argument of the interpolation methods, used
    # in validation error messages. Subclasses may override this.
    _size_name = 'radius'

    @property
    @abc.abstractmethod
    def apertures(self):
        """
        A list of the aperture objects used to measure the profile.
        """

    @cached_property
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        radius.
        """
        return self._compute_photometry(self.apertures)

    @cached_property
    def _raw_profile(self):
        """
        The raw (unnormalized) curve-of-growth profile as a 1D
        `~numpy.ndarray`, or a `~astropy.units.Quantity` if the input
        ``data`` has units.
        """
        return self._photometry[0]

    @cached_property
    def _raw_profile_error(self):
        """
        The raw (unnormalized) curve-of-growth profile errors as a 1D
        `~numpy.ndarray`, or a `~astropy.units.Quantity` if the input
        ``data`` has units.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """
        return self._photometry[1]

    @cached_property
    def area(self):
        """
        The unmasked area within each aperture as a 1D
        `~numpy.ndarray`.
        """
        return self._photometry[2]

    def _finite_profile(self):
        """
        The finite profile values used by the interpolation methods.

        Returns
        -------
        radius : 1D `~numpy.ndarray`
            The profile radii with a finite profile value.

        profile : 1D `~numpy.ndarray`
            The finite profile values, without units.

        unit : `~astropy.units.Unit` or `None`
            The profile unit.
        """
        profile = self.profile
        unit = getattr(profile, 'unit', None)
        if unit is not None:
            profile = profile.value

        finite = np.isfinite(profile)
        if np.count_nonzero(finite) < 2:
            msg = ('The curve-of-growth profile must have at least two '
                   'finite values to be interpolated.')
            raise ValueError(msg)

        return self.radius[finite], profile[finite], unit

    def _calc_ee(self, radius):
        """
        Calculate the enclosed energy at the given radii by
        interpolating the profile.
        """
        if isinstance(radius, u.Quantity):
            msg = (f'{self._size_name} must be in pixels and cannot be a '
                   'Quantity')
            raise TypeError(msg)

        xvals, profile, unit = self._finite_profile()
        ee = PchipInterpolator(xvals, profile, extrapolate=False)(radius)
        if unit is not None:
            ee = ee << unit
        return ee

    def _calc_size(self, ee):
        """
        Calculate the smallest radius at which the interpolated profile
        reaches the given enclosed energies.
        """
        xvals, profile, unit = self._finite_profile()
        if isinstance(ee, u.Quantity):
            if unit is None:
                unit = u.dimensionless_unscaled
            ee = ee.to_value(unit)
        ee = np.asarray(ee, dtype=float)

        # The interpolant is monotonic between adjacent profile values,
        # so it first reaches a value in the first interval whose end
        # values bracket it
        lower = np.minimum(profile[:-1], profile[1:])
        upper = np.maximum(profile[:-1], profile[1:])
        interp = PchipInterpolator(xvals, profile, extrapolate=False)
        result = np.full(ee.shape, np.nan)
        for idx, value in np.ndenumerate(ee):
            bracket = (lower <= value) & (value <= upper)
            if not np.any(bracket):
                continue  # outside the profile range, or NaN

            first = np.argmax(bracket)
            if profile[first] == value:
                result[idx] = xvals[first]
            elif profile[first + 1] == value:
                result[idx] = xvals[first + 1]
            else:
                roots = interp.solve(value, extrapolate=False)
                inside = ((roots >= xvals[first])
                          & (roots <= xvals[first + 1]))
                result[idx] = roots[inside].min()

        return result


@_update_method_subpixels_docstring
class CurveOfGrowth(_CurveOfGrowthBase):
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Class to create a curve of growth using concentric circular
    apertures.

    The curve of growth profile represents the circular aperture flux as
    a function of circular radius.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D data array. The data should be background-subtracted.
        Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D array_like of float
        An array of the circular radii. ``radii`` must be strictly
        increasing with a minimum value greater than zero, and contain
        at least 2 values. The radial spacing does not need to be
        constant. ``radii`` must be in pixels and cannot be a
        `~astropy.units.Quantity`.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity`, then ``error`` must have the same
        units. Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

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
        with at least two values, has a value less than or equal to
        zero, or is not strictly increasing, if ``error`` or ``mask``
        does not have the same shape as ``data``, if ``data`` and
        ``error`` do not have the same units, or if ``method`` or
        ``subpixels`` is invalid.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If the ``data`` or ``error`` array contains nonfinite values
        that are not already masked by ``mask``.

    See Also
    --------
    EllipticalCurveOfGrowth, EnsquaredCurveOfGrowth, RadialProfile

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_2dg
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import CurveOfGrowth

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the curve of growth.

    >>> xycen = centroid_2dg(data)
    >>> radii = np.arange(1, 26)
    >>> cog = CurveOfGrowth(data, xycen, radii, error=error)

    >>> print(cog.radius)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

    >>> print(cog.profile)
    [ 135.14750208  514.49674293 1076.4617132  1771.53866121 2510.94382666
     3238.51695898 3907.08459943 4456.90125492 4891.00892262 5236.59326527
     5473.66400376 5643.72239573 5738.24972738 5803.31693644 5842.00525018
     5850.45854739 5855.76123671 5844.9631235  5847.72359025 5843.23189459
     5852.05251106 5875.32009699 5869.86235184 5880.64741302 5872.16333953]

    >>> print(cog.profile_error)
    [ 3.11062819  6.83127792 10.55630757 14.27852358 18.00330858 21.72296997
     25.44442261 29.16767174 32.88949473 36.61224389 40.33235754 44.0555786
     47.77789677 51.49947445 55.22141896 58.94275538 62.66630873 66.38766913
     70.10917057 73.83146673 77.55360137 81.27674225 84.99747377 88.71963388
     92.44198203]

    Plot the curve of growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

        # Create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error)

        # Plot the curve of growth
        fig, ax = plt.subplots(figsize=(8, 6))
        cog.plot(ax=ax)
        cog.plot_error(ax=ax)

    Normalize the profile and plot the normalized curve of growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

        # Create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error)

        # Plot the curve of growth
        cog.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        cog.plot(ax=ax)
        cog.plot_error(ax=ax)

    Plot three of the apertures on the data.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

        # Create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        cog.apertures[5].plot(ax=ax, color='C0', lw=2)
        cog.apertures[10].plot(ax=ax, color='C1', lw=2)
        cog.apertures[15].plot(ax=ax, color='C3', lw=2)
    """

    # Define y-axis label used by `~photutils.profiles.ProfileBase.plot`
    _ylabel = 'Curve of Growth'

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        if np.min(radii) <= 0:
            msg = 'radii must be > 0'
            raise ValueError(msg)

        super().__init__(data, xycen, radii, error=error, mask=mask,
                         method=method, subpixels=subpixels)

    @cached_property
    def radius(self):
        """
        The profile radius in pixels as a 1D `~numpy.ndarray`.

        This is the same as the input ``radii``.

        Note that these are the radii of the circular apertures used
        to measure the profile. Thus, they are the radial values that
        enclose the given flux. They can be used directly to measure the
        encircled energy/flux at a given radius.
        """
        return self.radii

    @cached_property
    def apertures(self):
        """
        A list of `~photutils.aperture.CircularAperture` objects used to
        measure the profile.
        """
        return self._circular_apertures

    def calc_ee_at_radius(self, radius):
        """
        Calculate the encircled energy at a given radius using a cubic
        interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile such that
        its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``radii``.

        The interpolated curve passes through the measured profile
        values. For a noisy profile, the curve can therefore decrease
        with radius where the enclosed flux has leveled off.

        Parameters
        ----------
        radius : float or array_like
            The circular radius/radii in pixels. It cannot be a
            `~astropy.units.Quantity`.

        Returns
        -------
        ee : `~numpy.ndarray` or `~astropy.units.Quantity`
            The encircled energy at each radius. It is a
            `~astropy.units.Quantity` with the units of `profile` if the
            input ``data`` has units. Returns NaN for radii outside the
            range of the profile data.

        Raises
        ------
        TypeError
            If ``radius`` is a `~astropy.units.Quantity`.

        See Also
        --------
        calc_radius_at_ee
        """
        return self._calc_ee(radius)

    def calc_radius_at_ee(self, ee):
        """
        Calculate the radius at a given encircled energy using a cubic
        interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile such that
        its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``radii``.

        This method returns the smallest radius at which the curve
        interpolated by `calc_ee_at_radius` reaches the encircled
        energy. Converting the result back with `calc_ee_at_radius`
        returns the input encircled energy. For a noisy profile, the
        curve can reach the same encircled energy again at a larger
        radius.

        Where a noisy profile has leveled off, the radius of a given
        encircled energy is poorly constrained, and the maximum used by
        `normalize` is biased high by the noise. If you need the radius
        at a large encircled energy (e.g., 0.95 or more), use a profile
        with a high signal-to-noise ratio, or divide the input ``data``
        by a known total flux instead of calling `normalize`.

        Parameters
        ----------
        ee : float, array_like, or `~astropy.units.Quantity`
            The encircled energy. A value without units is assumed to be
            in the units of `profile`. A `~astropy.units.Quantity` is
            converted to the units of `profile`, and must be
            dimensionless if `profile` has no units.

        Returns
        -------
        radius : `~numpy.ndarray`
            The radius in pixels at each encircled energy. Returns NaN
            for encircled energies outside the range of the profile
            data.

        Raises
        ------
        `~astropy.units.UnitConversionError`
            If ``ee`` is a `~astropy.units.Quantity` with units that
            cannot be converted to the units of `profile`.

        See Also
        --------
        calc_ee_at_radius
        """
        return self._calc_size(ee)


@_update_method_subpixels_docstring
class EnsquaredCurveOfGrowth(_CurveOfGrowthBase):
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Class to create a curve of growth using concentric square
    apertures.

    The ensquared curve of growth profile represents the square aperture
    flux as a function of the square half-size (half side length).

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D data array. The data should be background-subtracted.
        Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    half_sizes : 1D array_like of float
        An array of the square half side lengths. ``half_sizes`` must
        be strictly increasing with a minimum value greater than zero,
        and contain at least 2 values. The spacing does not need to be
        constant. The full side length of each square aperture is
        ``2 * half_sizes``. ``half_sizes`` must be in pixels and cannot
        be a `~astropy.units.Quantity`.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity`, then ``error`` must have the same
        units. Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.

    <method_subpixels_descriptions>

    Raises
    ------
    TypeError
        If ``half_sizes`` is a `~astropy.units.Quantity`.

    ValueError
        If ``data`` is not a 2D array, if ``half_sizes`` is not a 1D
        array with at least two values, has a value less than or equal
        to zero, or is not strictly increasing, if ``error`` or ``mask``
        does not have the same shape as ``data``, if ``data`` and
        ``error`` do not have the same units, or if ``method`` or
        ``subpixels`` is invalid.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If the ``data`` or ``error`` array contains nonfinite values
        that are not already masked by ``mask``.

    See Also
    --------
    CurveOfGrowth, EllipticalCurveOfGrowth

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_2dg
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import EnsquaredCurveOfGrowth

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the ensquared curve of growth.

    >>> xycen = centroid_2dg(data)
    >>> half_sizes = np.arange(1, 26)
    >>> ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)

    >>> print(ecog.half_size)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

    >>> print(ecog.profile)
    [ 171.51682931  640.93965938 1328.92447125 2143.49339746 2954.67817356
     3718.43574763 4357.67409109 4845.38866928 5199.93124392 5481.4874771
     5641.60852855 5752.43870109 5791.01446371 5819.46039533 5832.44691816
     5825.39056549 5834.14890108 5833.42003487 5852.25143102 5856.59671815
     5869.46227464 5872.94734439 5868.92606685 5849.96903786 5839.55875017]

    >>> print(ecog.profile_error)
    [  3.57264883   7.77582626  11.97677454  16.17723038  20.3774983
      24.57767465  28.77779953  32.97789259  37.17796463  41.37802205
      45.57806888  49.77810781  53.97814068  58.17816881  62.37819315
      66.57821441  70.77823316  74.9782498   79.17826468  83.37827806
      87.57829015  91.77830114  95.97831117 100.17832036 104.3783288 ]

    Normalize the profile and plot the normalized ensquared curve of
    growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import EnsquaredCurveOfGrowth

        # Create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the ensquared curve of growth
        half_sizes = np.arange(1, 26)
        ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)

        # Plot the ensquared curve of growth
        ecog.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        ecog.plot(ax=ax)
        ecog.plot_error(ax=ax)

    Plot three of the apertures on the data.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import EnsquaredCurveOfGrowth

        # Create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the ensquared curve of growth
        half_sizes = np.arange(1, 26)
        ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        ecog.apertures[5].plot(ax=ax, color='C0', lw=2)
        ecog.apertures[10].plot(ax=ax, color='C1', lw=2)
        ecog.apertures[15].plot(ax=ax, color='C3', lw=2)
    """

    # Axis labels used by `~photutils.profiles.ProfileBase.plot`.
    _xlabel = 'Half-Size (pixels)'
    _ylabel = 'Ensquared Curve of Growth'

    # The user-facing name of the sizes parameter, used in validation
    # error messages
    _radii_name = 'half_sizes'
    _size_name = 'half_size'

    def __init__(self, data, xycen, half_sizes, *, error=None, mask=None,
                 method='exact', subpixels=5):

        if np.min(half_sizes) <= 0:
            msg = 'half_sizes must be > 0'
            raise ValueError(msg)

        super().__init__(data, xycen, half_sizes, error=error, mask=mask,
                         method=method, subpixels=subpixels)
        # Alias the validated sizes under the user-facing name. The
        # inherited ``radii`` attribute holds the same array.
        self.half_sizes = self.radii

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_half_sizes = len(self.half_sizes)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, '
                f'n_half_sizes={n_half_sizes}, '
                f'normalized={normalized})')

    @cached_property
    def half_size(self):
        """
        The profile half-sizes (half side lengths) in pixels as a 1D
        `~numpy.ndarray`.

        This is the same as the input ``half_sizes``.

        Note that these are the half side lengths of the square
        apertures used to measure the profile. The full side length of
        each square aperture is ``2 * half_size``. They can be used
        directly to measure the ensquared energy/flux at a given
        half-size.
        """
        return self.half_sizes

    @cached_property
    def radius(self):
        """
        The profile half-sizes (half side lengths) in pixels as a 1D
        `~numpy.ndarray`.

        This is an alias for `half_size`.
        """
        return self.half_sizes

    @cached_property
    def apertures(self):
        """
        A list of `~photutils.aperture.RectangularAperture` objects used
        to measure the profile.
        """
        from photutils.aperture import RectangularAperture

        return [RectangularAperture(self.xycen, 2 * hs, 2 * hs)
                for hs in self.half_sizes]

    def calc_ee_at_half_size(self, half_size):
        """
        Calculate the ensquared energy at a given half-size using a
        cubic interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large half-size. You can also use the `normalize`
        method before calling this method to normalize the profile such
        that its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``half_sizes``.

        The interpolated curve passes through the measured profile
        values. For a noisy profile, the curve can therefore decrease
        with half-size where the enclosed flux has leveled off.

        Parameters
        ----------
        half_size : float or array_like
            The square half side length(s) in pixels. It cannot be a
            `~astropy.units.Quantity`.

        Returns
        -------
        ee : `~numpy.ndarray` or `~astropy.units.Quantity`
            The ensquared energy at each half-size. It is a
            `~astropy.units.Quantity` with the units of `profile` if the
            input ``data`` has units. Returns NaN for half-sizes outside
            the range of the profile data.

        Raises
        ------
        TypeError
            If ``half_size`` is a `~astropy.units.Quantity`.

        See Also
        --------
        calc_half_size_at_ee
        """
        return self._calc_ee(half_size)

    def calc_half_size_at_ee(self, ee):
        """
        Calculate the half-size at a given ensquared energy using a
        cubic interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large half-size. You can also use the `normalize`
        method before calling this method to normalize the profile such
        that its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``half_sizes``.

        This method returns the smallest half-size at which the curve
        interpolated by `calc_ee_at_half_size` reaches the ensquared
        energy. Converting the result back with `calc_ee_at_half_size`
        returns the input ensquared energy. For a noisy profile, the
        curve can reach the same ensquared energy again at a larger
        half-size.

        Where a noisy profile has leveled off, the half-size of a given
        ensquared energy is poorly constrained, and the maximum used by
        `normalize` is biased high by the noise. If you need the
        half-size at a large ensquared energy (e.g., 0.95 or more), use
        a profile with a high signal-to-noise ratio, or divide the input
        ``data`` by a known total flux instead of calling `normalize`.

        Parameters
        ----------
        ee : float, array_like, or `~astropy.units.Quantity`
            The ensquared energy. A value without units is assumed to be
            in the units of `profile`. A `~astropy.units.Quantity` is
            converted to the units of `profile`, and must be
            dimensionless if `profile` has no units.

        Returns
        -------
        half_size : `~numpy.ndarray`
            The half-size in pixels at each ensquared energy. Returns
            NaN for ensquared energies outside the range of the profile
            data.

        Raises
        ------
        `~astropy.units.UnitConversionError`
            If ``ee`` is a `~astropy.units.Quantity` with units that
            cannot be converted to the units of `profile`.

        See Also
        --------
        calc_ee_at_half_size
        """
        return self._calc_size(ee)


@_update_method_subpixels_docstring
class EllipticalCurveOfGrowth(_CurveOfGrowthBase):
    # numpydoc ignore: PR01,PR02,PR04,PR07
    """
    Class to create a curve of growth using concentric elliptical
    apertures with a fixed axis ratio and orientation.

    The elliptical curve of growth profile represents the elliptical
    aperture flux as a function of the semimajor axis length.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D data array. The data should be background-subtracted.
        Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D array_like of float
        An array of the ellipse semimajor-axis lengths. ``radii`` must
        be strictly increasing with a minimum value greater than zero,
        and contain at least 2 values. The spacing does not need to be
        constant. ``radii`` must be in pixels and cannot be a
        `~astropy.units.Quantity`.

    axis_ratio : float
        The ratio of the semiminor axis to the semimajor axis
        (``b / a``). Must be in the range ``0 < axis_ratio <= 1``.

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        value in radians (as a float) from the positive x-axis. The
        rotation angle increases counterclockwise. The default is 0.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity`, then ``error`` must have the same
        units. Nonfinite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

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
        with at least two values, has a value less than or equal to
        zero, or is not strictly increasing, if ``axis_ratio`` is not in
        the range ``0 < axis_ratio <= 1``, if ``error`` or ``mask`` does
        not have the same shape as ``data``, if ``data`` and ``error``
        do not have the same units, or if ``method`` or ``subpixels`` is
        invalid.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If the ``data`` or ``error`` array contains nonfinite values
        that are not already masked by ``mask``.

    See Also
    --------
    CurveOfGrowth, EnsquaredCurveOfGrowth

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_2dg
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import EllipticalCurveOfGrowth

    Create an artificial elliptical source. Note that this image does
    not have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the elliptical curve of growth with an axis ratio of 0.5 and
    a rotation angle of 42 degrees.

    >>> xycen = centroid_2dg(data)
    >>> radii = np.arange(1, 40)
    >>> ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
    ...                                theta=np.deg2rad(42), error=error)

    >>> print(ecog.radius)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]

    >>> print(ecog.profile)
    [   67.39762867   267.711181     588.47524874  1021.31994307
      1546.53867489  2152.12698084  2824.97954482  3541.64650208
      4284.363828    5040.93586551  5777.06397177  6488.33779084
      7179.10371288  7826.17773764  8418.08027957  8948.7310004
      9418.56480323  9810.0373925  10163.11467352 10477.42357537
     10731.89184641 10920.11723061 11092.34059512 11235.12552706
     11347.43721424 11454.03577845 11520.64656354 11555.89668261
     11571.27935302 11583.89774142 11605.79810845 11639.93073462
     11648.27293403 11660.34772581 11662.89065496 11643.07787619
     11630.36674411 11636.61537567 11636.60448497]

    >>> print(ecog.profile_error)
    [  1.82558544   4.4811361    7.21875141   9.88960965  12.44181885
      15.0767893   17.74361162  20.3959057   23.02040853  25.60886742
      28.25910457  30.93376937  33.54721095  36.14973269  38.78350009
      41.44751786  44.08722476  46.67911616  49.30783947  51.9760285
      54.61090354  57.21854088  59.84454217  62.48828381  65.14297876
      67.7591664   70.36843954  73.01750591  75.66115523  78.29244888
      80.90838345  83.53056346  86.19047387  88.82463241  91.43141788
      94.06294498  96.71101089  99.35484622 101.96990083]

    Normalize the profile and plot the normalized elliptical curve of
    growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import EllipticalCurveOfGrowth

        # Create an artificial elliptical source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the elliptical curve of growth
        radii = np.arange(1, 40)
        ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       theta=np.deg2rad(42), error=error)

        # Plot the elliptical curve of growth
        ecog.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        ecog.plot(ax=ax)
        ecog.plot_error(ax=ax)

    Plot three of the apertures on the data.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import EllipticalCurveOfGrowth

        # Create an artificial elliptical source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.1
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # Find the source centroid
        xycen = centroid_2dg(data)

        # Create the elliptical curve of growth
        radii = np.arange(1, 40)
        ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                       theta=np.deg2rad(42), error=error)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        ecog.apertures[5].plot(ax=ax, color='C0', lw=2)
        ecog.apertures[10].plot(ax=ax, color='C1', lw=2)
        ecog.apertures[15].plot(ax=ax, color='C3', lw=2)
    """

    # Define axis labels used by `~photutils.profiles.ProfileBase.plot`
    _xlabel = 'Semimajor Axis (pixels)'
    _ylabel = 'Elliptical Curve of Growth'

    def __init__(self, data, xycen, radii, axis_ratio, *, theta=0.0,
                 error=None, mask=None, method='exact', subpixels=5):

        if np.min(radii) <= 0:
            msg = 'radii must be > 0'
            raise ValueError(msg)

        if not 0 < axis_ratio <= 1:
            msg = 'axis_ratio must be in the range 0 < axis_ratio <= 1'
            raise ValueError(msg)

        self.axis_ratio = axis_ratio
        self.theta = theta

        super().__init__(data, xycen, radii, error=error, mask=mask,
                         method=method, subpixels=subpixels)

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_radii = len(self.radii)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, n_radii={n_radii}, '
                f'axis_ratio={self.axis_ratio}, theta={self.theta}, '
                f'normalized={normalized})')

    @cached_property
    def radius(self):
        """
        The profile semimajor-axis lengths in pixels as a 1D
        `~numpy.ndarray`.

        This is the same as the input ``radii``.

        Note that these are the semimajor-axis lengths of the elliptical
        apertures used to measure the profile. Thus, they are the
        semimajor-axis values that enclose the given flux.
        """
        return self.radii

    @cached_property
    def apertures(self):
        """
        A list of `~photutils.aperture.EllipticalAperture` objects used
        to measure the profile.
        """
        from photutils.aperture import EllipticalAperture

        return [EllipticalAperture(self.xycen, a, a * self.axis_ratio,
                                   theta=self.theta)
                for a in self.radii]

    def calc_ee_at_radius(self, radius):
        """
        Calculate the encircled energy at a given semimajor-axis length
        using a cubic interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile such that
        its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``radii``.

        The interpolated curve passes through the measured profile
        values. For a noisy profile, the curve can therefore decrease
        with semimajor-axis length where the enclosed flux has leveled
        off.

        Parameters
        ----------
        radius : float or array_like
            The semimajor-axis length(s) in pixels. It cannot be a
            `~astropy.units.Quantity`.

        Returns
        -------
        ee : `~numpy.ndarray` or `~astropy.units.Quantity`
            The encircled energy at each semimajor-axis length. It is a
            `~astropy.units.Quantity` with the units of `profile` if the
            input ``data`` has units. Returns NaN for semimajor-axis
            lengths outside the range of the profile data.

        Raises
        ------
        TypeError
            If ``radius`` is a `~astropy.units.Quantity`.

        See Also
        --------
        calc_radius_at_ee
        """
        return self._calc_ee(radius)

    def calc_radius_at_ee(self, ee):
        """
        Calculate the semimajor-axis length at a given encircled energy
        using a cubic interpolator based on the profile data.

        Note that this method assumes that the input data have been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile such that
        its maximum value is 1. For noisy data, the maximum may not
        occur at the largest input ``radii``.

        This method returns the smallest semimajor-axis length at which
        the curve interpolated by `calc_ee_at_radius` reaches the
        encircled energy. Converting the result back with
        `calc_ee_at_radius` returns the input encircled energy. For a
        noisy profile, the curve can reach the same encircled energy
        again at a larger semimajor-axis length.

        Where a noisy profile has leveled off, the semimajor-axis length
        of a given encircled energy is poorly constrained, and the
        maximum used by `normalize` is biased high by the noise. If you
        need the semimajor-axis length at a large encircled energy
        (e.g., 0.95 or more), use a profile with a high signal-to-noise
        ratio, or divide the input ``data`` by a known total flux
        instead of calling `normalize`.

        Parameters
        ----------
        ee : float, array_like, or `~astropy.units.Quantity`
            The encircled energy. A value without units is assumed to be
            in the units of `profile`. A `~astropy.units.Quantity` is
            converted to the units of `profile`, and must be
            dimensionless if `profile` has no units.

        Returns
        -------
        radius : `~numpy.ndarray`
            The semimajor-axis length in pixels at each encircled
            energy. Returns NaN for encircled energies outside the range
            of the profile data.

        Raises
        ------
        `~astropy.units.UnitConversionError`
            If ``ee`` is a `~astropy.units.Quantity` with units that
            cannot be converted to the units of `profile`.

        See Also
        --------
        calc_ee_at_radius
        """
        return self._calc_size(ee)
