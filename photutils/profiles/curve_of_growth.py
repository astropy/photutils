# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for generating curves of growth.
"""

import numpy as np
from astropy.utils import lazyproperty
from scipy.interpolate import PchipInterpolator

from photutils.profiles.core import ProfileBase

__all__ = ['CurveOfGrowth', 'EllipticalCurveOfGrowth',
           'EnsquaredCurveOfGrowth']


class CurveOfGrowth(ProfileBase):
    """
    Class to create a curve of growth using concentric circular
    apertures.

    The curve of growth profile represents the circular aperture flux as
    a function of circular radius.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. The data should be background-subtracted.
        Non-finite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D float `~numpy.ndarray`
        An array of the circular radii. ``radii`` must be strictly
        increasing with a minimum value greater than zero, and contain
        at least 2 values. The radial spacing does not need to be
        constant.

    error : 2D `~numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. Non-finite
        values (e.g., NaN or inf) in the ``data`` or ``error`` array are
        automatically masked.

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

    >>> print(cog.radius)  # doctest: +FLOAT_CMP
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

    >>> print(cog.profile)  # doctest: +FLOAT_CMP
    [ 135.14750208  514.49674293 1076.4617132  1771.53866121 2510.94382666
     3238.51695898 3907.08459943 4456.90125492 4891.00892262 5236.59326527
     5473.66400376 5643.72239573 5738.24972738 5803.31693644 5842.00525018
     5850.45854739 5855.76123671 5844.9631235  5847.72359025 5843.23189459
     5852.05251106 5875.32009699 5869.86235184 5880.64741302 5872.16333953]

    >>> print(cog.profile_error)  # doctest: +FLOAT_CMP
    [ 3.72215309  7.44430617 11.16645926 14.88861235 18.61076543 22.33291852
     26.05507161 29.7772247  33.49937778 37.22153087 40.94368396 44.66583704
     48.38799013 52.11014322 55.8322963  59.55444939 63.27660248 66.99875556
     70.72090865 74.44306174 78.16521482 81.88736791 85.609521   89.33167409
     93.05382717]

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

    Plot a couple of the apertures on the data.

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

    @lazyproperty
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

    @lazyproperty
    def apertures(self):
        """
        A list of `~photutils.aperture.CircularAperture` objects used to
        measure the profile.
        """
        return self._circular_apertures

    @lazyproperty
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        radius.
        """
        return self._compute_photometry(self.apertures)

    @lazyproperty
    def profile(self):
        """
        The curve-of-growth profile as a 1D `~numpy.ndarray`.
        """
        return self._photometry[0]

    @lazyproperty
    def profile_error(self):
        """
        The curve-of-growth profile errors as a 1D `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """
        return self._photometry[1]

    @lazyproperty
    def area(self):
        """
        The unmasked area in each circular aperture as a function of
        radius as a 1D `~numpy.ndarray`.
        """
        return self._photometry[2]

    def calc_ee_at_radius(self, radius):
        """
        Calculate the encircled energy at a given radius using a cubic
        interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile to be 1 at
        the largest input ``radii``.

        Parameters
        ----------
        radius : float or 1D `~numpy.ndarray`
            The circular radius/radii.

        Returns
        -------
        ee : `~numpy.ndarray`
            The encircled energy at each radius/radii. Returns NaN for
            radii outside the range of the profile data.
        """
        return PchipInterpolator(self.radius, self.profile,
                                 extrapolate=False)(radius)

    def calc_radius_at_ee(self, ee):
        """
        Calculate the radius at a given encircled energy using a cubic
        interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile to be 1 at
        the largest input ``radii``.

        This interpolator returns values only for regions where the
        curve-of-growth profile is monotonically increasing.

        Parameters
        ----------
        ee : float or 1D `~numpy.ndarray`
            The encircled energy.

        Returns
        -------
        radius : `~numpy.ndarray`
            The radius at each encircled energy. Returns NaN for
            encircled energies outside the range of the profile data.
        """
        radius, profile = self._trim_to_monotonic(
            self.radius, self.profile, 'curve-of-growth')

        return PchipInterpolator(profile, radius, extrapolate=False)(ee)


class EnsquaredCurveOfGrowth(ProfileBase):
    """
    Class to create a curve of growth using concentric square
    apertures.

    The ensquared curve of growth profile represents the square aperture
    flux as a function of the square half-size (half side length).

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. The data should be background-subtracted.
        Non-finite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    half_sizes : 1D float `~numpy.ndarray`
        An array of the square half side lengths. ``half_sizes`` must
        be strictly increasing with a minimum value greater than zero,
        and contain at least 2 values. The spacing does not need to be
        constant. The full side length of each square aperture is
        ``2 * half_sizes``.

    error : 2D `~numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. Non-finite
        values (e.g., NaN or inf) in the ``data`` or ``error`` array are
        automatically masked.

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

    >>> print(ecog.half_size)  # doctest: +FLOAT_CMP
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

    >>> print(ecog.profile)  # doctest: +FLOAT_CMP
    [ 171.35182895  640.63717997 1328.55725483 2142.84258293 2954.12152275
     3717.5208724  4356.82277842 4844.97997426 5199.74452363 5480.78438494
     5641.63617089 5751.92491894 5790.90751883 5819.30778391 5832.38652883
     5825.14679788 5833.55196333 5833.54737611 5851.79194687 5856.58494602
     5869.76637039 5872.91078217 5868.62195688 5850.11085443 5838.889818  ]

    >>> print(ecog.profile_error)  # doctest: +FLOAT_CMP
    [  4.2   8.4  12.6  16.8  21.   25.2  29.4  33.6  37.8  42.   46.2  50.4
      54.6  58.8  63.   67.2  71.4  75.6  79.8  84.   88.2  92.4  96.6 100.8
     105. ]

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

    Plot a couple of the apertures on the data.

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

    def __init__(self, data, xycen, half_sizes, *, error=None, mask=None,
                 method='exact', subpixels=5):

        if np.min(half_sizes) <= 0:
            msg = 'half_sizes must be > 0'
            raise ValueError(msg)

        super().__init__(data, xycen, half_sizes, error=error, mask=mask,
                         method=method, subpixels=subpixels)
        # self.radii is set by the parent class
        self.half_sizes = self.radii

    def __repr__(self):
        cls_name = self.__class__.__name__
        n_half_sizes = len(self.half_sizes)
        normalized = self.normalization_value != 1.0
        return (f'{cls_name}(xycen={self.xycen}, '
                f'n_half_sizes={n_half_sizes}, '
                f'normalized={normalized})')

    @lazyproperty
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

    @lazyproperty
    def radius(self):
        """
        The profile half-sizes (half side lengths) in pixels as a 1D
        `~numpy.ndarray`.

        This is an alias for `half_size`.
        """
        return self.half_sizes

    @lazyproperty
    def apertures(self):
        """
        A list of `~photutils.aperture.RectangularAperture` objects used
        to measure the profile.
        """
        from photutils.aperture import RectangularAperture

        return [RectangularAperture(self.xycen, 2 * hs, 2 * hs)
                for hs in self.half_sizes]

    @lazyproperty
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        size.
        """
        return self._compute_photometry(self.apertures)

    @lazyproperty
    def profile(self):
        """
        The ensquared curve-of-growth profile as a 1D `~numpy.ndarray`.
        """
        return self._photometry[0]

    @lazyproperty
    def profile_error(self):
        """
        The ensquared curve-of-growth profile errors as a 1D
        `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """
        return self._photometry[1]

    @lazyproperty
    def area(self):
        """
        The unmasked area in each square aperture as a function of size
        as a 1D `~numpy.ndarray`.
        """
        return self._photometry[2]

    def calc_ee_at_half_size(self, half_size):
        """
        Calculate the ensquared energy at a given half-size using a
        cubic interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large half-size. You can also use the `normalize`
        method before calling this method to normalize the profile to
        be 1 at the largest input ``half_sizes``.

        Parameters
        ----------
        half_size : float or 1D `~numpy.ndarray`
            The square half side length(s).

        Returns
        -------
        ee : `~numpy.ndarray`
            The ensquared energy at each half-size. Returns NaN for
            half-sizes outside the range of the profile data.
        """
        return PchipInterpolator(self.half_size, self.profile,
                                 extrapolate=False)(half_size)

    def calc_half_size_at_ee(self, ee):
        """
        Calculate the half-size at a given ensquared energy using a
        cubic interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large half-size. You can also use the `normalize`
        method before calling this method to normalize the profile to
        be 1 at the largest input ``half_sizes``.

        This interpolator returns values only for regions where the
        ensquared curve-of-growth profile is monotonically increasing.

        Parameters
        ----------
        ee : float or 1D `~numpy.ndarray`
            The ensquared energy.

        Returns
        -------
        half_size : `~numpy.ndarray`
            The half-size at each ensquared energy. Returns NaN for
            ensquared energies outside the range of the profile data.
        """
        half_size, profile = self._trim_to_monotonic(
            self.half_size, self.profile,
            'ensquared curve-of-growth')

        return PchipInterpolator(profile, half_size,
                                 extrapolate=False)(ee)


class EllipticalCurveOfGrowth(ProfileBase):
    """
    Class to create a curve of growth using concentric elliptical
    apertures with a fixed axis ratio and orientation.

    The elliptical curve of growth profile represents the elliptical
    aperture flux as a function of the semimajor axis length.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. The data should be background-subtracted.
        Non-finite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D float `~numpy.ndarray`
        An array of the ellipse semimajor-axis lengths. ``radii`` must
        be strictly increasing with a minimum value greater than zero,
        and contain at least 2 values. The spacing does not need to be
        constant.

    axis_ratio : float
        The ratio of the semiminor axis to the semimajor axis
        (``b / a``). Must be in the range ``0 < axis_ratio <= 1``.

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        value in radians (as a float) from the positive ``x`` axis. The
        rotation angle increases counterclockwise.

    error : 2D `~numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error of
        the sources (see `~photutils.utils.calc_total_error`). ``error``
        must have the same shape as the input ``data``. Non-finite
        values (e.g., NaN or inf) in the ``data`` or ``error`` array are
        automatically masked.

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
    ...                               theta=np.deg2rad(42), error=error)

    >>> print(ecog.radius)  # doctest: +FLOAT_CMP
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]

    >>> print(ecog.profile)  # doctest: +FLOAT_CMP
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

    >>> print(ecog.profile_error)  # doctest: +FLOAT_CMP
    [  2.63195969   5.26391938   7.89587907  10.52783875  13.15979844
      15.79175813  18.42371782  21.05567751  23.6876372   26.31959688
      28.95155657  31.58351626  34.21547595  36.84743564  39.47939533
      42.11135501  44.7433147   47.37527439  50.00723408  52.63919377
      55.27115346  57.90311314  60.53507283  63.16703252  65.79899221
      68.4309519   71.06291159  73.69487127  76.32683096  78.95879065
      81.59075034  84.22271003  86.85466972  89.4866294   92.11858909
      94.75054878  97.38250847 100.01446816 102.64642785]

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

    Plot a couple of the apertures on the data.

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

    @lazyproperty
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

    @lazyproperty
    def apertures(self):
        """
        A list of `~photutils.aperture.EllipticalAperture` objects used
        to measure the profile.
        """
        from photutils.aperture import EllipticalAperture

        return [EllipticalAperture(self.xycen, a, a * self.axis_ratio,
                                   theta=self.theta)
                for a in self.radii]

    @lazyproperty
    def _photometry(self):
        """
        The aperture fluxes, flux errors, and areas as a function of
        semimajor axis.
        """
        return self._compute_photometry(self.apertures)

    @lazyproperty
    def profile(self):
        """
        The elliptical curve-of-growth profile as a 1D `~numpy.ndarray`.
        """
        return self._photometry[0]

    @lazyproperty
    def profile_error(self):
        """
        The elliptical curve-of-growth profile errors as a 1D
        `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """
        return self._photometry[1]

    @lazyproperty
    def area(self):
        """
        The unmasked area in each elliptical aperture as a function of
        semimajor axis as a 1D `~numpy.ndarray`.
        """
        return self._photometry[2]

    def calc_ee_at_radius(self, radius):
        """
        Calculate the encircled energy at a given semimajor-axis length
        using a cubic interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile to be 1 at
        the largest input ``radii``.

        Parameters
        ----------
        radius : float or 1D `~numpy.ndarray`
            The semimajor-axis length(s).

        Returns
        -------
        ee : `~numpy.ndarray`
            The encircled energy at each radius. Returns NaN for radii
            outside the range of the profile data.
        """
        return PchipInterpolator(self.radius, self.profile,
                                 extrapolate=False)(radius)

    def calc_radius_at_ee(self, ee):
        """
        Calculate the semimajor-axis length at a given encircled energy
        using a cubic interpolator based on the profile data.

        Note that this method assumes that input data has been
        normalized such that the total enclosed flux is 1 for an
        infinitely large radius. You can also use the `normalize` method
        before calling this method to normalize the profile to be 1 at
        the largest input ``radii``.

        This interpolator returns values only for regions where the
        elliptical curve-of-growth profile is monotonically increasing.

        Parameters
        ----------
        ee : float or 1D `~numpy.ndarray`
            The encircled energy.

        Returns
        -------
        radius : `~numpy.ndarray`
            The semimajor-axis length at each encircled energy. Returns
            NaN for encircled energies outside the range of the profile
            data.
        """
        radius, profile = self._trim_to_monotonic(
            self.radius, self.profile,
            'elliptical curve-of-growth')

        return PchipInterpolator(profile, radius, extrapolate=False)(ee)
