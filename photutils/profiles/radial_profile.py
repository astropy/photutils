# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for generating radial profiles.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils import lazyproperty

from photutils.profiles.core import ProfileBase

__all__ = ['RadialProfile']


class RadialProfile(ProfileBase):
    """
    Class to create a radial profile using concentric circular annulus
    apertures.

    The radial profile represents the azimuthally-averaged flux in
    circular annuli apertures as a function of radius.

    For this class, the input radii represent the edges of the radial
    bins. This differs from the `RadialProfile` class, where the inputs
    represent the centers of the radial bins.

    The output `radius` attribute represents the bin centers.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D data array. The data should be background-subtracted.
        Non-finite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

    xycen : tuple of 2 floats
        The ``(x, y)`` pixel coordinate of the source center.

    radii : 1D float `~numpy.ndarray`
        An array of radii defining the *edges* of the radial bins.
        ``radii`` must be strictly increasing with a minimum value
        greater than or equal to zero, and contain at least 2 values.
        The radial spacing does not need to be constant. The output
        `radius` attribute will be defined at the bin centers.

    error : 2D `~numpy.ndarray`, optional
        The 1-sigma errors of the input ``data``. ``error`` is assumed
        to include all sources of error, including the Poisson error
        of the sources (see `~photutils.utils.calc_total_error`) .
        ``error`` must have the same shape as the input ``data``.
        Non-finite values (e.g., NaN or inf) in the ``data`` or
        ``error`` array are automatically masked.

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
    RadialProfile

    Notes
    -----
    If the minimum of ``radii`` is zero, then a circular aperture
    with radius equal to ``radii[1]`` will be used for the innermost
    aperture.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_quadratic
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import RadialProfile

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.4
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the radial profile.

    >>> xycen = centroid_quadratic(data, xpeak=48, ypeak=52)
    >>> edge_radii = np.arange(26)
    >>> rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    >>> print(rp.radius)  # doctest: +FLOAT_CMP
    [ 0.5  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5 12.5 13.5
     14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5]

    >>> print(rp.profile)  # doctest: +FLOAT_CMP
    [ 4.15632243e+01  3.93402079e+01  3.59845746e+01  3.15540506e+01
      2.62300757e+01  2.07297033e+01  1.65106801e+01  1.19376723e+01
      7.75743772e+00  5.56759777e+00  3.44112671e+00  1.91350281e+00
      1.17092981e+00  4.22261078e-01  9.70256904e-01  4.16355795e-01
      1.52328707e-02 -6.69985111e-02  4.15522650e-01  2.48494731e-01
      4.03348112e-01  1.43482678e-01 -2.62777461e-01  7.30653622e-02
      7.84616804e-04]

    >>> print(rp.profile_error)  # doctest: +FLOAT_CMP
    [1.354055   0.78176402 0.60555181 0.51178468 0.45135167 0.40826294
     0.37554729 0.3496155  0.32840658 0.31064152 0.29547903 0.28233999
     0.270811   0.26058801 0.2514417  0.24319546 0.23571072 0.22887707
     0.22260527 0.21682233 0.21146786 0.20649145 0.2018506  0.19750922
     0.19343643]

    Plot the radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.4
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        # plot the radial profile
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax)
        rp.plot_error(ax=ax)

    Plot the radial profile, including the raw data profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.4
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        # plot the radial profile
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax, color='C0')
        rp.plot_error(ax=ax)
        ax.scatter(rp.data_radius, rp.data_profile, s=1, color='C1')

    Normalize the profile and plot the normalized radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.4
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        # plot the radial profile
        rp.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax)
        rp.plot_error(ax=ax)

    Plot three of the annulus apertures on the data.

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
        bkg_sig = 2.4
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        rp.apertures[5].plot(ax=ax, color='C0', lw=2)
        rp.apertures[10].plot(ax=ax, color='C1', lw=2)
        rp.apertures[15].plot(ax=ax, color='C3', lw=2)

    Fit a 1D Gaussian to the radial profile and return the Gaussian
    model.

    >>> rp.gaussian_fit  # doctest: +FLOAT_CMP
    <Gaussian1D(amplitude=41.54880743, mean=0., stddev=4.71059406)>

    >>> print(rp.gaussian_fwhm)  # doctest: +FLOAT_CMP
    11.09260130738712

    Plot the fitted 1D Gaussian on the radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

        # create an artificial single source
        gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
        yy, xx = np.mgrid[0:100, 0:100]
        data = gmodel(xx, yy)
        bkg_sig = 2.4
        noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
        data += noise
        error = np.zeros_like(data) + bkg_sig

        # find the source centroid
        xycen = centroid_quadratic(data, xpeak=48, ypeak=52)

        # create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

        # plot the radial profile
        rp.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax, label='Radial Profile')
        rp.plot_error(ax=ax)
        ax.plot(rp.radius, rp.gaussian_profile, label='Gaussian Fit')
        ax.legend()
    """

    @lazyproperty
    def radius(self):
        """
        The profile radius (bin centers) in pixels as a 1D
        `~numpy.ndarray`.

        The returned radius values are defined as the arithmetic means
        of the input radial-bins edges (``radii``).

        For logarithmically-spaced input ``radii``, one could instead
        use a radius array defined using the geometric mean of the bin
        edges, i.e. ``np.sqrt(radii[:-1] * radii[1:])``.
        """
        # define the radial bin centers from the radial bin edges
        return (self.radii[:-1] + self.radii[1:]) / 2

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
        for i in range(len(self.radii) - 1):
            try:
                aperture = CircularAnnulus(self.xycen, self.radii[i],
                                           self.radii[i + 1])
            except ValueError:  # zero radius
                aperture = CircularAperture(self.xycen,
                                            self.radii[i + 1])
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
        The fitted 1D Gaussian to the radial profile as a
        `~astropy.modeling.functional_models.Gaussian1D` model.

        The Gaussian fit will not change if the profile normalization is
        changed after performing the fit.
        """
        profile = self.profile[self._profile_nanmask]
        radius = self.radius[self._profile_nanmask]

        amplitude = np.max(profile)
        std = np.sqrt(abs(np.sum(profile * radius**2) / np.sum(profile)))
        g_init = Gaussian1D(amplitude=amplitude, mean=0.0, stddev=std)
        g_init.mean.fixed = True
        fitter = TRFLSQFitter()
        return fitter(g_init, radius, profile)

    @lazyproperty
    def gaussian_profile(self):
        """
        The fitted 1D Gaussian profile to the radial profile as a 1D
        `~numpy.ndarray`.

        The Gaussian profile will not change if the profile
        normalization is changed after performing the fit.
        """
        return self.gaussian_fit(self.radius)

    @lazyproperty
    def gaussian_fwhm(self):
        """
        The full-width at half-maximum (FWHM) in pixels of the 1D
        Gaussian fitted to the radial profile.
        """
        return self.gaussian_fit.stddev.value * gaussian_sigma_to_fwhm

    @lazyproperty
    def _data_profile(self):
        """
        The raw data profile returned as a 1D arrays (`~numpy.ndarray`)
        of radii and data values.

        This method returns the radii and values of the data points
        within the maximum radius defined by the input radii.
        """
        shape = self.data.shape
        max_radius = np.max(self.radii)
        x_min = int(max(np.floor(self.xycen[0] - max_radius), 0))
        x_max = int(min(np.ceil(self.xycen[0] + max_radius), shape[1]))
        y_min = int(max(np.floor(self.xycen[1] - max_radius), 0))
        y_max = int(min(np.ceil(self.xycen[1] + max_radius), shape[0]))
        yidx, xidx = np.indices((y_max - y_min, x_max - x_min))
        xidx += x_min
        yidx += y_min
        radii = np.hypot(xidx - self.xycen[0], yidx - self.xycen[1])
        mask = radii <= max_radius
        radii = radii[mask]
        data_values = self.data[yidx[mask], xidx[mask]]

        return radii, data_values

    @lazyproperty
    def data_radius(self):
        """
        The radii of the raw data profile as a 1D `~numpy.ndarray`.
        """
        return self._data_profile[0]

    @lazyproperty
    def data_profile(self):
        """
        The raw data profile as a 1D `~numpy.ndarray`.
        """
        return self._data_profile[1]
