# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for generating radial profiles.
"""

import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Gaussian1D, Moffat1D
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.profiles.core import ProfileBase

__all__ = ['RadialProfile']


class RadialProfile(ProfileBase):
    """
    Class to create a radial profile using concentric circular annulus
    apertures.

    The radial profile represents the azimuthally-averaged flux in
    circular annuli apertures as a function of radius.

    For this class, the input radii represent the edges of the radial
    bins. This differs from the `CurveOfGrowth` class, where the input
    radii are the radii of the circular apertures used to compute the
    cumulative flux.

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
    CurveOfGrowth

    Notes
    -----
    If the minimum of ``radii`` is zero, then a circular aperture
    with radius equal to ``radii[1]`` will be used for the innermost
    aperture.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_2dg
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import RadialProfile

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the radial profile.

    >>> xycen = centroid_2dg(data)
    >>> edge_radii = np.arange(25)
    >>> rp = RadialProfile(data, xycen, edge_radii, error=error)

    >>> print(rp.radius)  # doctest: +FLOAT_CMP
    [ 0.5  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5 12.5 13.5
     14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5]

    >>> print(rp.profile)  # doctest: +FLOAT_CMP
    [ 4.30187860e+01  4.02502046e+01  3.57758011e+01  3.16071235e+01
      2.61511082e+01  2.10539746e+01  1.63701300e+01  1.16674718e+01
      8.12828014e+00  5.78962699e+00  3.59342666e+00  2.35353336e+00
      1.20355937e+00  7.67093923e-01  4.24650784e-01  8.67989701e-02
      5.11484374e-02 -9.82041768e-02  2.37482124e-02 -3.66602855e-02
      6.84802299e-02  1.72239596e-01 -3.86056497e-02  7.30423743e-02]

    >>> print(rp.profile_error)  # doctest: +FLOAT_CMP
    [1.18479813 0.68404352 0.52985783 0.4478116  0.39493271 0.35723008
     0.32860388 0.30591356 0.28735575 0.27181133 0.25854415 0.24704749
     0.23695963 0.22801451 0.22001149 0.21279603 0.20624688 0.20026744
     0.19477961 0.18971954 0.18503438 0.18068002 0.17661928 0.17282057]

    Plot the radial profile, including the raw data profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

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

        # Create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error)

        # Plot the radial profile
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax, color='C0')
        rp.plot_error(ax=ax)
        ax.scatter(rp.data_radius, rp.data_profile, s=1, color='C1')

    Normalize the profile and plot the normalized radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

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

        # Create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error)

        # Plot the radial profile
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

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

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

        # Create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        rp.apertures[5].plot(ax=ax, color='C0', lw=2)
        rp.apertures[10].plot(ax=ax, color='C1', lw=2)
        rp.apertures[15].plot(ax=ax, color='C3', lw=2)

    Fit 1D Gaussian and Moffat models to the normalized radial profile.

    >>> rp.normalize()
    >>> rp.gaussian_fit  # doctest: +FLOAT_CMP
    <Gaussian1D(amplitude=0.98231088, mean=0., stddev=4.67512776)>
    >>> rp.moffat_fit  # doctest: +ELLIPSIS
    <Moffat1D(amplitude=..., x_0=0., gamma=..., alpha=...)>
    >>> print(rp.gaussian_fwhm)  # doctest: +FLOAT_CMP
    11.009084813327846
    >>> print(rp.moffat_fwhm)  # doctest: +FLOAT_CMP
    10.868426507928344

    Plot the fitted 1D Gaussian and Moffat models on the radial profile.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_2dg
        from photutils.datasets import make_noise_image
        from photutils.profiles import RadialProfile

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

        # Create the radial profile
        edge_radii = np.arange(26)
        rp = RadialProfile(data, xycen, edge_radii, error=error)

        # Plot the radial profile
        rp.normalize()
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax, label='Radial Profile')
        rp.plot_error(ax=ax)
        ax.plot(rp.radius, rp.gaussian_profile, label='Gaussian Fit')
        ax.plot(rp.radius, rp.moffat_profile, label='Moffat Fit')
        ax.legend()
    """

    # Define y-axis label used by `~photutils.profiles.ProfileBase.plot`
    _ylabel = 'Radial Profile'

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
        # Define the radial bin centers from the radial bin edges
        return (self.radii[:-1] + self.radii[1:]) / 2

    @lazyproperty
    def apertures(self):
        """
        A list of the circular annulus apertures used to measure the
        radial profile, as `~photutils.aperture.CircularAnnulus`
        objects.

        If the minimum of ``radii`` is zero, then the innermost element
        will be a `~photutils.aperture.CircularAperture` with radius
        equal to ``radii[1]``.
        """
        # Prevent circular imports
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
        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return self._flux / self.area

    @lazyproperty
    def profile_error(self):
        """
        The radial profile errors as a 1D `~numpy.ndarray`.

        If no ``error`` array was provided, an empty array with shape
        ``(0,)`` is returned.
        """
        if self.error is None:
            return self._fluxerr

        # Ignore divide-by-zero RuntimeWarning
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

        if len(profile) == 0:
            msg = ('The radial profile is entirely non-finite or masked; '
                   'cannot fit a Gaussian.')
            warnings.warn(msg, AstropyUserWarning)
            return None

        amplitude = np.max(profile)
        sum_profile = np.sum(profile)
        if sum_profile == 0:
            warnings.warn('The profile sum is zero; the Gaussian fit '
                          'initial guess may be inaccurate.',
                          AstropyUserWarning)
            std = 1.0  # fallback to avoid zero initial guess
        else:
            std = np.sqrt(abs(np.sum(profile * radius**2) / sum_profile))
            std = max(std, 1.0)  # guard against near-zero initial guess
        g_init = Gaussian1D(amplitude=amplitude, mean=0.0, stddev=std)
        g_init.mean.fixed = True
        fitter = TRFLSQFitter()
        gaussian_fit = fitter(g_init, radius, profile)

        if radius.min() > 0.3 * gaussian_fit.stddev.value:
            msg = ('Gaussian fit may be unreliable because the input '
                   'radii do not extend close to the source center.')
            warnings.warn(msg, AstropyUserWarning)

        return gaussian_fit

    @lazyproperty
    def gaussian_profile(self):
        """
        The fitted 1D Gaussian profile to the radial profile as a 1D
        `~numpy.ndarray`.

        The Gaussian profile will not change if the profile
        normalization is changed after performing the fit.

        Returns `None` if the fit failed (e.g., the profile is entirely
        non-finite or masked).
        """
        if self.gaussian_fit is None:
            return None
        return self.gaussian_fit(self.radius)

    @lazyproperty
    def gaussian_fwhm(self):
        """
        The full-width at half-maximum (FWHM) in pixels of the 1D
        Gaussian fitted to the radial profile.

        Returns `None` if the fit failed (e.g., the profile is entirely
        non-finite or masked).
        """
        if self.gaussian_fit is None:
            return None
        return self.gaussian_fit.stddev.value * gaussian_sigma_to_fwhm

    @lazyproperty
    def moffat_fit(self):
        """
        The fitted 1D Moffat to the radial profile as a
        `~astropy.modeling.functional_models.Moffat1D` model.

        The Moffat fit will not change if the profile normalization is
        changed after performing the fit.
        """
        profile = self.profile[self._profile_nanmask]
        radius = self.radius[self._profile_nanmask]

        if len(profile) == 0:
            msg = ('The radial profile is entirely non-finite or masked; '
                   'cannot fit a Moffat.')
            warnings.warn(msg, AstropyUserWarning)
            return None

        amplitude = np.max(profile)
        sum_profile = np.sum(profile)
        if sum_profile == 0:
            warnings.warn('The profile sum is zero; the Moffat fit '
                          'initial guess may be inaccurate.',
                          AstropyUserWarning)
            gamma = 1.0  # fallback to avoid zero initial guess
        else:
            # Estimate gamma from the half-max radius
            half_max = amplitude / 2.0
            above = profile >= half_max
            gamma = (max(np.max(radius[above]), 1.0)
                     if np.any(above) else 1.0)

        m_init = Moffat1D(amplitude=amplitude, x_0=0.0, gamma=gamma,
                          alpha=2.5)
        m_init.x_0.fixed = True
        m_init.gamma.bounds = (0, None)
        m_init.alpha.bounds = (1, 25)
        fitter = TRFLSQFitter()
        return fitter(m_init, radius, profile)

    @lazyproperty
    def moffat_profile(self):
        """
        The fitted 1D Moffat profile to the radial profile as a 1D
        `~numpy.ndarray`.

        The Moffat profile will not change if the profile normalization
        is changed after performing the fit.

        Returns `None` if the fit failed (e.g., the profile is entirely
        non-finite or masked).
        """
        if self.moffat_fit is None:
            return None
        return self.moffat_fit(self.radius)

    @lazyproperty
    def moffat_fwhm(self):
        """
        The full-width at half-maximum (FWHM) in pixels of the 1D Moffat
        fitted to the radial profile.

        Returns `None` if the fit failed (e.g., the profile is entirely
        non-finite or masked).
        """
        if self.moffat_fit is None:
            return None
        return self.moffat_fit.fwhm

    @lazyproperty
    def _data_profile(self):
        """
        The raw data profile returned as 1D arrays (`~numpy.ndarray`) of
        radii and data values.

        Returns the radii and values of the unmasked data points within
        the maximum radius defined by the input radii. Pixels flagged
        in ``self.mask`` (including auto-masked non-finite values) are
        excluded.
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

        # Calculate the radii of the pixels from the center and select
        # those within the maximum radius defined by the input radii
        radii = np.hypot(xidx - self.xycen[0], yidx - self.xycen[1])
        within = radii <= max_radius
        radii = radii[within]
        yidx_sub = yidx[within]
        xidx_sub = xidx[within]

        # Exclude masked pixels (user mask and auto-masked non-finite
        # values)
        valid = ~self.mask[yidx_sub, xidx_sub]
        radii = radii[valid]
        data_values = self.data[yidx_sub[valid], xidx_sub[valid]]

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

    def _normalize_hook(self, normalization):
        """
        Also normalize ``data_profile`` if it has been computed.
        """
        if 'data_profile' in self.__dict__:
            self.__dict__['data_profile'] = self.data_profile / normalization

    def _unnormalize_hook(self):
        """
        Also unnormalize ``data_profile`` if it has been computed.
        """
        if 'data_profile' in self.__dict__:
            self.__dict__['data_profile'] = (self.data_profile
                                             * self.normalization_value)
