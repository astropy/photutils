# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for generating curves of growth.
"""
import numpy as np
from astropy.utils import lazyproperty
from scipy.interpolate import PchipInterpolator

from photutils.profiles.core import ProfileBase

__all__ = ['CurveOfGrowth']


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

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from astropy.visualization import simple_norm
    >>> from photutils.centroids import centroid_quadratic
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import CurveOfGrowth

    Create an artificial single source. Note that this image does not
    have any background.

    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.4
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=123)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    Create the curve of growth.

    >>> xycen = centroid_quadratic(data, xpeak=48, ypeak=52)
    >>> radii = np.arange(1, 26)
    >>> cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

    >>> print(cog.radius)  # doctest: +FLOAT_CMP
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

    >>> print(cog.profile)  # doctest: +FLOAT_CMP
    [ 130.57472018  501.34744442 1066.59182074 1760.50163608 2502.13955554
     3218.50667597 3892.81448231 4455.36403436 4869.66609313 5201.99745378
     5429.02043984 5567.28370644 5659.24831854 5695.06577065 5783.46217755
     5824.01080702 5825.59003768 5818.22316662 5866.52307412 5896.96917375
     5948.92254787 5968.30540534 5931.15611704 5941.94457249 5942.06535486]

    >>> print(cog.profile_error)  # doctest: +FLOAT_CMP
    [  4.25388924   8.50777848  12.76166773  17.01555697  21.26944621
      25.52333545  29.7772247   34.03111394  38.28500318  42.53889242
      46.79278166  51.04667091  55.30056015  59.55444939  63.80833863
      68.06222787  72.31611712  76.57000636  80.8238956   85.07778484
      89.33167409  93.58556333  97.83945257 102.09334181 106.34723105]

    Plot the curve of growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

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

        # create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

        # plot the curve of growth
        fig, ax = plt.subplots(figsize=(8, 6))
        cog.plot(ax=ax)
        cog.plot_error(ax=ax)

    Normalize the profile and plot the normalized curve of growth.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

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

        # create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

        # plot the curve of growth
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

        from photutils.centroids import centroid_quadratic
        from photutils.datasets import make_noise_image
        from photutils.profiles import CurveOfGrowth

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

        # create the curve of growth
        radii = np.arange(1, 26)
        cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

        norm = simple_norm(data, 'sqrt')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, norm=norm, origin='lower')
        cog.apertures[5].plot(ax=ax, color='C0', lw=2)
        cog.apertures[10].plot(ax=ax, color='C1', lw=2)
        cog.apertures[15].plot(ax=ax, color='C3', lw=2)
    """

    def __init__(self, data, xycen, radii, *, error=None, mask=None,
                 method='exact', subpixels=5):

        radii = np.array(radii)
        if radii.min() <= 0:
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

        If ``radius_min`` is zero, then the first item will be `None`.
        """
        return self._circular_apertures

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

        Note that this function assumes that input data has been
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

        Note that this function assumes that input data has been
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
        # restrict the profile to the monotonically increasing region;
        # this is necessary for the interpolator
        radius = self.radius
        profile = self.profile
        diff = np.diff(profile) <= 0
        if np.any(diff):
            idx = np.argmax(diff)  # first non-monotonic point
            radius = radius[0:idx]
            profile = profile[0:idx]

        if len(radius) < 2:
            msg = ('The curve-of-growth profile is not monotonically '
                   'increasing even at the smallest radii -- cannot '
                   'interpolate. Try using different input radii '
                   '(especially the starting radii) and/or using the '
                   '"exact" aperture overlap method.')
            raise ValueError(msg)

        return PchipInterpolator(profile, radius, extrapolate=False)(ee)
