# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for generating radial profiles.
"""

import math
import warnings

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils import lazyproperty

from photutils.profiles.core import ProfileBase

__all__ = ['RadialProfile']

__doctest_requires__ = {('RadialProfile'): ['scipy']}


class RadialProfile(ProfileBase):
    """
    Class to create a radial profile using concentric apertures.

    The radial profile represents the azimuthally-averaged flux in
    circular annuli apertures as a function of radius.

    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    array are automatically masked.

    Parameters
    ----------
    data : 2D `numpy.ndarray`
        The 2D data array. The data should be background-subtracted.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    min_radius : float
        The minimum radius for the profile.

    max_radius : float
        The maximum radius for the profile.

    radius_step : float
        The radial step size in pixels.

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

    Notes
    -----
    If the ``min_radius`` is less than or equal to half the
    ``radius_step``, then a circular aperture with radius equal to
    ``min_radius + 0.5 * radius_step`` will be used for the innermost
    aperture.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from astropy.visualization import simple_norm
    >>> from photutils.centroids import centroid_quadratic
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import RadialProfile

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
    >>> data += error

    Create the radial profile.

    >>> xycen = centroid_quadratic(data, xpeak=48, ypeak=52)
    >>> min_radius = 0.0
    >>> max_radius = 25.0
    >>> radius_step = 1.0
    >>> rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
    ...                    error=error, mask=None)

    >>> print(rp.radius)  # doctest: +FLOAT_CMP
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
     18. 19. 20. 21. 22. 23. 24. 25.]

    >>> print(rp.profile)  # doctest: +FLOAT_CMP
    [ 4.27430150e+01  4.02150658e+01  3.81601146e+01  3.38116846e+01
      2.89343205e+01  2.34250297e+01  1.84368533e+01  1.44310461e+01
      9.55543388e+00  6.55415896e+00  4.49693014e+00  2.56010523e+00
      1.50362911e+00  7.35389056e-01  6.04663625e-01  8.08820954e-01
      2.31751912e-01 -1.39063329e-01  1.25181410e-01  4.84601845e-01
      1.94567871e-01  4.49109676e-01 -2.00995374e-01 -7.74387397e-02
      5.70302749e-02 -3.27578439e-02]

    >>> print(rp.profile_error)  # doctest: +FLOAT_CMP
    [2.95008692 1.17855895 0.6610777  0.51902503 0.47524302 0.43072819
     0.39770113 0.37667594 0.33909996 0.35356048 0.30377721 0.29455808
     0.25670656 0.26599511 0.27354232 0.2430305  0.22910334 0.22204777
     0.22327174 0.23816561 0.2343794  0.2232632  0.19893783 0.17888776
     0.18228289 0.19680823]

    Plot the radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
        data += error

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        min_radius = 0.0
        max_radius = 25.0
        radius_step = 1.0
        rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                           error=error, mask=None)

        # plot the radial profile
        rp.plot()
        rp.plot_error()

    Normalize the profile and plot the normalized radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
        data += error

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        min_radius = 0.0
        max_radius = 25.0
        radius_step = 1.0
        rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                           error=error, mask=None)

        # plot the radial profile
        rp.normalize()
        rp.plot()
        rp.plot_error()

    Plot two of the annulus apertures on the data.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
        data += error

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        min_radius = 0.0
        max_radius = 25.0
        radius_step = 1.0
        rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                           error=error, mask=None)

        norm = simple_norm(data, 'sqrt')
        plt.figure(figsize=(5, 5))
        plt.imshow(data, norm=norm)
        rp.apertures[5].plot(color='C0', lw=2)
        rp.apertures[10].plot(color='C1', lw=2)

    Fit a 1D Gaussian to the radial profile and return the Gaussian
    model.

    >>> rp.gaussian_fit  # doctest: +FLOAT_CMP
    <Gaussian1D(amplitude=41.80620963, mean=0., stddev=4.69126969)>

    >>> print(rp.gaussian_fwhm)  # doctest: +FLOAT_CMP
    11.04709589620093

    Plot the fitted 1D Gaussian on the radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from astropy.visualization import simple_norm

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        error = make_noise_image(data.shape, mean=0., stddev=2.4, seed=123)
        data += error

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        min_radius = 0.0
        max_radius = 25.0
        radius_step = 1.0
        rp = RadialProfile(data, xycen, min_radius, max_radius, radius_step,
                           error=error, mask=None)

        # plot the radial profile
        rp.normalize()
        rp.plot(label='Radial Profile')
        rp.plot_error()
        plt.plot(rp.radius, rp.gaussian_profile, label='Gaussian Fit')
        plt.legend()
    """

    @lazyproperty
    def _circular_radii(self):
        """
        The circular aperture radii for the radial bin edges (inner and
        outer annulus radii).
        """
        shift = self.radius_step / 2
        min_radius = self.min_radius - shift
        max_radius = self.max_radius + shift
        nsteps = int(math.floor((max_radius - min_radius)
                                / self.radius_step))
        max_radius = min_radius + (nsteps * self.radius_step)
        return np.linspace(min_radius, max_radius, nsteps + 1)

    @lazyproperty
    def apertures(self):
        """
        A list of the circular annulus apertures used to measure the
        radial profile.

        If the ``min_radius`` is less than or equal to half the
        ``radius_step``, then a circular aperture with radius equal
        to ``min_radius + 0.5 * radius_step`` will be used for the
        innermost aperture.
        """
        from photutils.aperture import CircularAnnulus, CircularAperture

        apertures = []
        for i in range(len(self._circular_radii) - 1):
            try:
                aperture = CircularAnnulus(self.xycen, self._circular_radii[i],
                                           self._circular_radii[i + 1])
            except ValueError:  # zero radius
                aperture = CircularAperture(self.xycen,
                                            self._circular_radii[i + 1])
            apertures.append(aperture)

        return apertures

    @lazyproperty
    def _flux(self):
        """
        The flux in a circular annulus.
        """
        return np.diff(self._photometry[0])

    @lazyproperty
    def _fluxerr(self):
        """
        The flux error in a circular annulus.
        """
        return np.sqrt(np.diff(self._photometry[1] ** 2))

    @lazyproperty
    def area(self):
        """
        The unmasked area in each circular annulus (or aperture) as a
        function of radius as a 1D `~numpy.ndarray`.
        """
        return np.diff(self._photometry[2])

    @lazyproperty
    def profile(self):
        """
        The radial profile as a 1D `~numpy.ndarray`.
        """
        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return self._flux / self.area

    @lazyproperty
    def profile_error(self):
        """
        The radial profile errors as a 1D `~numpy.ndarray`.
        """
        if self.error is None:
            return self._fluxerr

        # ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return self._fluxerr / self.area

    @lazyproperty
    def _profile_nanmask(self):
        return np.isfinite(self.profile)

    @lazyproperty
    def gaussian_fit(self):
        """
        The fitted 1D Gaussian to the radial profile as
        a `~astropy.modeling.functional_models.Gaussian1D` model.
        """
        profile = self.profile[self._profile_nanmask]
        radius = self.radius[self._profile_nanmask]

        amplitude = np.max(profile)
        std = np.sqrt(abs(np.sum(profile * radius**2) / np.sum(profile)))
        g_init = Gaussian1D(amplitude=amplitude, mean=0.0, stddev=std)
        g_init.mean.fixed = True
        fitter = LevMarLSQFitter()
        g_fit = fitter(g_init, radius, profile)

        return g_fit

    @lazyproperty
    def gaussian_profile(self):
        """
        The fitted 1D Gaussian profile to the radial profile as a 1D
        `~numpy.ndarray`.
        """
        return self.gaussian_fit(self.radius)

    @lazyproperty
    def gaussian_fwhm(self):
        """
        The full-width at half-maximum (FWHM) in pixels of the 1D
        Gaussian fitted to the radial profile.
        """
        return self.gaussian_fit.stddev.value * gaussian_sigma_to_fwhm
